"""Collect finished runs, group the ones that may be compared, and draw them.

`report.py` only ever reads what `storage.py` wrote.
It does not parse directory paths - `--output` lets a run live anywhere - so everything it needs comes from the manifest inside `results.json`.

Partitioning
------------
Several runs go into one figure, but not every run found may share an axis.
If the document set differs, or a signal's definition changed, putting the numbers side by side produces a comparison that is wrong while looking fine.
So runs are grouped before anything is drawn, and the grouping is printed *first*: a figure that appears before the grouping is a figure you read without knowing which runs are missing from it.

Two modes, one machinery
------------------------
What a report holds constant and what it varies are two different questions, and this module answers both without a second pipeline.

* the default varies the **model** and holds the data fixed.
  Comparability is *proved*: runs share a ``doc_id_set_hash``, and the examples additionally require an identical ``prompt_hash`` per document.
* ``report --multilingual`` varies the **language** and holds the model fixed - the same benchmark evaluated in several languages.
  Here the proof is not available: different languages mean different documents and different prompts by construction, so ``doc_id_set_hash`` and ``prompt_hash`` can never agree and requiring them would reject every real case.

The multilingual mode therefore rests on a *claim* - that these runs are translations of one benchmark, so ``doc_id`` 5 is the same question in each - and a claim that nothing checks is how a report ends up placing MMLU next to KMMLU as though document 5 were shared.
So the claim is stated - ``--multilingual en=global_mmlu_en,ko=global_mmlu_ko`` names the datasets outright, because benchmarks spell their languages in too many ways for any rule to guess - and then *tested* against the data: `gold_alignment` asks whether the runs agree on which choice is the gold one, document by document.
A translated benchmark agrees on essentially all of them; two unrelated benchmarks agree at chance.
Below the threshold the group keeps its curves - averages over different documents are still each meaningful - and loses the per-document half, with the measured rate printed as the reason.
"""

from __future__ import annotations

import dataclasses
import os
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .storage import (
    SAMPLING_SEED,
    SCHEMA_VERSION,
    TABLES,
    load_signals,
    read_results,
    read_table,
)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


@dataclass
class Run:
    """One run directory, as far as the report is concerned.

    Attributes:
        path: the directory holding `results.json`.
        manifest: the manifest block of `results.json`.
        scores: lm-eval's scores, printed alongside the signals.
    """

    path: str
    manifest: dict[str, Any]
    scores: dict[str, Any]

    @property
    def name(self) -> str:
        """Short label for listings: the directory name."""
        return os.path.basename(os.path.normpath(self.path))

    @property
    def model_id(self) -> str:
        return str(self.manifest.get("model_id", "?"))

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("tasks", ()))

    @property
    def reducer_versions(self) -> dict[str, int]:
        """Reducer name -> version, the thing that decides comparability.

        Example:
            >>> run.reducer_versions                     # doctest: +SKIP
            {'logit_lens': 1, 'layer_similarity': 2}
        """
        return {r["name"]: int(r["version"]) for r in self.manifest.get("reducers", [])}

    @property
    def n_documents(self) -> int:
        return int(self.manifest.get("n_documents", 0))

    @property
    def num_fewshot(self) -> int:
        return int(self.manifest.get("num_fewshot", 0) or 0)

    @property
    def vocab_size(self) -> int:
        return int(self.manifest.get("vocab_size", 0))


@dataclass(frozen=True)
class Dataset:
    """One language of a multilingual comparison, as the user declared it.

    Attributes:
        label: what the line is called - "ko", or the task name when no label was given.
        task: the lm-eval task name that was run for it.
    """

    label: str
    task: str


def parse_datasets(spec: str) -> list[Dataset]:
    """Parse `en=global_mmlu_en,ko=global_mmlu_ko` into the datasets being compared.

    There is no inference here, and that is the design.
    Benchmarks spell their languages in every possible way - `global_mmlu_ko` puts the tag in the middle, `m_mmlu_ko` at the end, `kmmlu` at the front, `belebele_kor_Hang` with a script subtag, and `mmlu_pro` is not a language at all - so any rule that guesses is right for one convention and quietly wrong for the rest.
    Which runs are being compared is therefore stated, once, in the same place the comparison is asked for.

    A bare task name is allowed and labels itself, for when the task names are already short enough to read in a legend.

    Raises:
        ValueError: on fewer than two datasets, or a repeated label or task.
            One language is not a comparison, and a repeat would draw two lines with one meaning.

    Example:
        >>> parse_datasets("en=global_mmlu_en,ko=global_mmlu_ko")
        [Dataset(label='en', task='global_mmlu_en'), Dataset(label='ko', task='global_mmlu_ko')]
        >>> parse_datasets("mmlu,kmmlu")[1]
        Dataset(label='kmmlu', task='kmmlu')
    """
    datasets: list[Dataset] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, _, task = entry.partition("=")
        label, task = label.strip(), task.strip()
        if not task:                      # a bare task name labels itself
            label, task = label, label
        if not task:
            raise ValueError(f"empty dataset in --multilingual: {spec!r}")
        datasets.append(Dataset(label=label, task=task))
    if len(datasets) < 2:
        raise ValueError(
            "--multilingual needs at least two datasets to compare, as "
            f"LANG=TASK,LANG=TASK (got {spec!r})")
    for field_name in ("label", "task"):
        seen = [getattr(d, field_name) for d in datasets]
        repeated = [name for name in seen if seen.count(name) > 1]
        if repeated:
            raise ValueError(f"--multilingual repeats the {field_name} {repeated[0]!r}; "
                             "each language needs its own line")
    return datasets


def discover_runs(paths: Iterable[str | os.PathLike]) -> list[Run]:
    """Walk the given directories and load every run found.

    Works with a single run directory just as well as with a tree of them.

    Example:
        >>> runs = discover_runs(["results/"])              # doctest: +SKIP
        >>> len(runs)
        10
    """
    found: list[Run] = []
    seen: set[str] = set()
    for root in paths:
        for directory, _subdirs, files in os.walk(str(root)):
            if "results.json" not in files:
                continue
            real = os.path.realpath(directory)
            if real in seen:
                continue
            seen.add(real)
            try:
                payload = read_results(directory)
            except (OSError, ValueError):
                continue  # not one of ours
            found.append(
                Run(
                    path=directory,
                    manifest=payload.get("manifest", {}),
                    scores=payload.get("results", {}),
                )
            )
    return sorted(found, key=lambda run: run.path)


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------


@dataclass
class Group:
    """A set of runs that may be plotted on the same axis.

    Attributes:
        label: "A", "B", ... in size order.
        axis: what varies between the members - "model" or "language".
        tasks / doc_id_set_hash / reducer_versions: what made them one group.
        datasets: under `--multilingual`, the languages that were declared, in the order given.
        runs: the members.
        warnings: why this group must not simply be read next to another one.
    """

    label: str
    tasks: tuple[str, ...]
    doc_id_set_hash: str
    n_documents: int
    reducer_versions: dict[str, int]
    runs: list[Run] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    axis: str = "model"
    datasets: tuple[Dataset, ...] = ()
    pairing: "Pairing | None" = None

    @property
    def limited(self) -> bool:
        """Whether this group ran with `--limit` applied."""
        return any(run.manifest.get("limit") for run in self.runs)

    def vocab_sizes(self) -> set[int]:
        return {run.vocab_size for run in self.runs if run.vocab_size}

    def dataset_of(self, run: Run) -> Dataset | None:
        """The declared dataset this run evaluated, matched on its task name."""
        for dataset in self.datasets:
            if run.tasks == (dataset.task,):
                return dataset
        return None

    def series_of(self, run: Run) -> str:
        """The label this run is drawn and named under.

        Normally that is the model; under `--multilingual` the model is what is held constant, so it is the declared label instead.
        Two runs of one language would otherwise collapse into a single line, so a repeated label keeps its run directory.
        """
        if self.axis != "language":
            return run.model_id
        def label_of(one: Run) -> str:
            found = self.dataset_of(one)
            return found.label if found else one.name
        label = label_of(run)
        repeated = sum(1 for other in self.runs if label_of(other) == label)
        return label if repeated < 2 else f"{label} ({run.name})"

    def doc_key(self, run: Run, task_name: str, doc_id: int) -> str:
        """The key a document is compared under across this group's runs.

        Normally every run ran the same task, so the task name is part of the key and carries its own meaning.

        Under `--multilingual` the task name is what differs, and the declaration says exactly how: whatever the user named as this language's dataset is removed from the front, and what remains is the part that has to match.
        `global_mmlu_en_stem` and `global_mmlu_ko_stem` both reduce to `_stem`, so no rule about where a language tag sits is needed, and none is used.

        The suffix is kept rather than dropped, because an lm-eval group expands into subtasks that each number their documents from zero: collapsing them would make `..._business` #5 and `..._stem` #5 one key, and two different questions would be compared as though they were the same one.

        A subtask whose name does not extend the declared one keeps its full name, so the keys simply fail to meet across languages - which `gold_alignment` reports as sharing no documents, rather than pairing the wrong ones.

        All of that is the fallback. When the runs carry a field that *identifies* a document - Global-MMLU logs `sample_id`, the same `high_school_world_history/test/33` in every language - `attach_pairing` puts it here instead, and position stops being trusted at all.
        """
        if self.axis == "language":
            source = self.pairing.source if self.pairing else "position"
            if source in ("field", "mapping"):
                found = self.pairing.keys.get(run.path, {}).get((task_name, int(doc_id)))
                # A document the pairing key does not cover cannot be matched to another
                # language, and a positional fallback for it alone would be the guess the
                # key exists to avoid. Its own run and doc_id keep it out of every set.
                return found if found is not None else f"{run.path}\x00{task_name}#{doc_id}"
            if source == "none":
                # Strict pairing without an identity: no document meets another language.
                return f"{run.path}\x00{task_name}#{doc_id}"
            dataset = self.dataset_of(run)
            declared = dataset.task if dataset else ""
            suffix = (task_name[len(declared):]
                      if declared and task_name.startswith(declared) else task_name)
            return f"{suffix}#{doc_id}"
        return f"{task_name}#{doc_id}"

    @property
    def held_constant(self) -> str:
        """What this group fixes, for the header line of a figure."""
        if self.axis == "language":
            return self.runs[0].model_id if self.runs else "?"
        return "+".join(self.tasks)

    @property
    def missing_languages(self) -> list[str]:
        """Declared languages that no run in this group covers."""
        present = {self.dataset_of(run) for run in self.runs}
        return [d.label for d in self.datasets if d not in present]


@dataclass
class Partition:
    """The result of grouping: what is comparable, and what was left out and why."""

    groups: list[Group]
    excluded: list[tuple[Run, str]]
    n_found: int
    axis: str = "model"
    datasets: tuple[Dataset, ...] = ()


def screen_runs(runs: Sequence[Run]) -> tuple[list[Run], list[tuple[Run, str]], dict[str, int]]:
    """Drop the runs that cannot enter *any* group, whatever the axis, and say why.

    Exclusions, i.e. runs that join no group at all:

    * no completion marker - the run died part-way, so its document set is truncated in a way nothing downstream can see;
    * a different `schema_version` - the columns do not line up, so the files cannot even be read together;
    * reducer versions that disagree with the majority - the column names are the same but the numbers mean something else.

    Returns:
        (candidates, excluded, majority reducer versions).
    """
    excluded: list[tuple[Run, str]] = []
    candidates: list[Run] = []
    for run in runs:
        if not run.manifest.get("completed"):
            excluded.append((run, "no completion marker (the run did not finish)"))
        elif str(run.manifest.get("schema_version")) != SCHEMA_VERSION:
            excluded.append(
                (run, f"schema_version {run.manifest.get('schema_version')} (current {SCHEMA_VERSION})")
            )
        else:
            candidates.append(run)

    # The reducer version set that most runs agree on.
    # A run that disagrees is computing a differently defined number under the same column name.
    version_counts = Counter(
        tuple(sorted(run.reducer_versions.items())) for run in candidates
    )
    majority = dict(version_counts.most_common(1)[0][0]) if version_counts else {}

    kept: list[Run] = []
    for run in candidates:
        versions = run.reducer_versions
        if versions != majority:
            # Name the reducer that differs and what everyone else used, so the reason is actionable without opening the manifests.
            differing = []
            for name, version in sorted(versions.items()):
                if majority.get(name) != version:
                    expected = majority.get(name)
                    others = f"other runs v{expected}" if expected is not None else "not in other runs"
                    differing.append(f"{name} v{version} ({others})")
            for name in sorted(set(majority) - set(versions)):
                differing.append(f"{name} missing (other runs v{majority[name]})")
            excluded.append((run, "; ".join(differing)))
            continue
        kept.append(run)
    return kept, excluded, majority


def partition_runs(
    runs: Sequence[Run], datasets: Sequence[Dataset] | None = None
) -> Partition:
    """Split runs into comparable groups, with a reason for every exclusion.

    Args:
        datasets: the languages `--multilingual` declared.
            Given, the report fixes the model and varies the language over exactly these datasets; omitted, it fixes the data and varies the model, which is the default report.

    Grouping key, default: task set plus `doc_id_set_hash`.
    Two runs of the same task that solved different documents (because one used `--limit`) end up in different groups; they are still each usable, they just must not be read as one series.

    Grouping key, `--multilingual`: model plus `num_fewshot`, over the declared datasets.
    The document sets differ on purpose here, so they cannot be part of the key - which is exactly why this axis carries `gold_alignment` instead.
    `num_fewshot` is in the key rather than merely warned about: a group that varies both the language and the number of examples answers neither question.

    Example:
        >>> part = partition_runs(discover_runs(["results/"]))   # doctest: +SKIP
        >>> [g.label for g in part.groups], len(part.excluded)
        (['A', 'B'], 3)
    """
    axis = "language" if datasets else "model"
    candidates, excluded, _majority = screen_runs(runs)
    if datasets and any(run.manifest.get("benchmarks") for run in candidates):
        raise ValueError("custom benchmark multilingual protocol comparison is not supported yet")
    groups, more_excluded = (
        _group_by_language(candidates, datasets) if datasets
        else _group_by_model(candidates))
    excluded += more_excluded

    groups = sorted(groups, key=lambda g: (-len(g.runs), g.tasks))
    for index, group in enumerate(groups):
        group.label = chr(ord("A") + index)
    for group in groups:
        vocabs = group.vocab_sizes()
        if len(vocabs) > 1:
            group.warnings.append(
                f"mixed vocabulary sizes {sorted(vocabs)}: signals marked "
                "same-vocab-only must not share an axis here"
            )
    return Partition(groups=groups, excluded=excluded, n_found=len(runs), axis=axis,
                     datasets=tuple(datasets or ()))


def _group_by_model(candidates: Sequence[Run]) -> tuple[list[Group], list[tuple[Run, str]]]:
    """The default axis: one group per (task set, document set)."""
    grouped: dict[tuple[Any, ...], Group] = {}
    for run in candidates:
        key = (run.tasks, run.manifest.get("doc_id_set_hash"),
               json.dumps(run.manifest.get("benchmarks", {}), sort_keys=True))
        group = grouped.get(key)
        if group is None:
            group = Group(
                label="",
                tasks=run.tasks,
                doc_id_set_hash=str(run.manifest.get("doc_id_set_hash", "")),
                n_documents=run.n_documents,
                reducer_versions=run.reducer_versions,
            )
            grouped[key] = group
        group.runs.append(run)

    groups = sorted(grouped.values(), key=lambda g: (-len(g.runs), g.tasks))
    # Grouping alone is not enough: it has to be visible *why* the split happened, or the second group looks like an arbitrary omission.
    first = groups[0] if groups else None
    for group in groups[1:]:
        # Name the actual reason, so the split does not look arbitrary.
        if group.tasks != first.tasks:
            group.warnings.append(f"different task from group {first.label or 'A'}")
        else:
            group.warnings.append(f"different document set from group {first.label or 'A'}")
    return groups, []


def _group_by_language(
    candidates: Sequence[Run], datasets: Sequence[Dataset]
) -> tuple[list[Group], list[tuple[Run, str]]]:
    """The multilingual mode: one group per (model, shot count), over the declared datasets.

    Which runs take part is not discovered, it is declared - so a run is excluded here for exactly one reason, that its task is not on the list, and the message says which task that was. A typo in a dataset name shows up as the run it failed to match rather than as a comparison that is quietly one language short.

    The model stays in the key because a comparison that varies the language *and* the model answers neither question, and `num_fewshot` stays in it for the same reason.

    A group covering fewer than two of the declared languages is excluded: one line is not a comparison.
    """
    excluded: list[tuple[Run, str]] = []
    by_task = {dataset.task: dataset for dataset in datasets}
    grouped: dict[tuple[Any, ...], Group] = {}
    for run in candidates:
        if len(run.tasks) != 1 or run.tasks[0] not in by_task:
            excluded.append(
                (run, f"ran {'+'.join(run.tasks) or '(none)'!r}, "
                      "which is not one of the declared datasets"))
            continue
        key = (run.model_id, run.num_fewshot)
        group = grouped.get(key)
        if group is None:
            group = Group(
                label="",
                tasks=(),
                doc_id_set_hash="",
                n_documents=0,
                reducer_versions=run.reducer_versions,
                axis="language",
                datasets=tuple(datasets),
            )
            grouped[key] = group
        group.runs.append(run)

    order = {dataset.task: index for index, dataset in enumerate(datasets)}
    groups: list[Group] = []
    for group in grouped.values():
        group.runs.sort(key=lambda r: (order.get(r.tasks[0], len(order)), r.path))
        group.tasks = tuple(run.tasks[0] for run in group.runs)
        covered = {group.dataset_of(run) for run in group.runs}
        if len(covered) < 2:
            for run in group.runs:
                excluded.append((run, f"only one of the declared languages was run for "
                                      f"{run.model_id} at {run.num_fewshot}-shot - "
                                      "nothing to compare it against"))
            continue
        if group.missing_languages:
            group.warnings.append(
                "no run for " + ", ".join(group.missing_languages)
                + ": the comparison is missing a declared language, not showing it as absent")
        sizes = {run.n_documents for run in group.runs}
        group.n_documents = sizes.pop() if len(sizes) == 1 else 0
        if group.n_documents == 0:
            group.warnings.append(
                "the languages cover different numbers of documents "
                + ", ".join(f"{group.series_of(r)} {r.n_documents}" for r in group.runs)
                + " - the score difference is partly a difference in what was asked")
        if len({run.manifest.get("revision") for run in group.runs}) > 1:
            group.warnings.append(
                "the languages ran different model revisions, so the model is not held constant")
        groups.append(group)
    return groups, excluded


def _format_score(run: Run) -> str:
    """The run's headline metric, for the listing that is printed before any figure.

    Under `--multilingual` this line *is* the score comparison - the question "does this model lose accuracy in Korean" is answered by four characters, and it would be perverse to make someone open a parquet for them.
    """
    score = _score_of(run)
    protocols = run.manifest.get("benchmarks", {}).get("protocols", {})
    if score is not None and len(run.tasks) == 1 and run.tasks[0] in protocols:
        spec = protocols[run.tasks[0]]
        direction = "higher is better" if spec["higher_is_better"] else "lower is better"
        return f"{spec['primary_metric']},{spec['primary_filter']} {score:.4f} ({direction})"
    return "score n/a" if score is None else f"score {score:.4f}"


def format_partition(partition: Partition) -> str:
    """Render the grouping as text, printed before anything is drawn.

    Under `--multilingual` the line also says where the languages came from, because an inferred language and a declared one are not equally trustworthy and the difference has to be visible before the figure, not after it.

    Example:
        >>> print(format_partition(part))                  # doctest: +SKIP
        Found 10 runs
        comparing across: model  (data held constant)
        <BLANKLINE>
        [Group A] xnli_ko / 2490 docs / logit_lens v1, layer_similarity v2 -> 5 runs
          Qwen/Qwen3-8B                        score 0.7120
          meta-llama/Llama-3.1-8B              score 0.6840
        <BLANKLINE>
        [Group B] xnli_ko / 500 docs (limit) / logit_lens v1, layer_similarity v2 -> 2 runs
          warning: different document set from group A
        <BLANKLINE>
        [Excluded] 3 runs
          qwen3-8b-mmlu-jan   schema_version 0.1 (current 0.3)
    """
    axis = partition.axis
    lines = [f"Found {partition.n_found} runs",
             "mode: multilingual - one model, several declared languages"
             if axis == "language" else
             "mode: several models over one dataset",
             ""]
    for group in partition.groups:
        versions = ", ".join(f"{name} v{v}" for name, v in sorted(group.reducer_versions.items()))
        limit_note = " (limit)" if group.limited else ""
        plural = "run" if len(group.runs) == 1 else "runs"
        if group.axis == "language":
            docs = (f"{group.n_documents} docs each" if group.n_documents
                    else "differing doc counts")
            lines.append(
                f"[Group {group.label}] {group.runs[0].model_id} / "
                f"{group.runs[0].num_fewshot}-shot / {docs}{limit_note} / {versions} "
                f"-> {len(group.runs)} {plural}"
            )
            for run in group.runs:
                lines.append(
                    f"  {group.series_of(run):<12} {run.tasks[0]:<24} "
                    f"{run.n_documents:>6} docs   {_format_score(run)}")
            lines.append("  declared: " + ", ".join(
                f"{d.label}={d.task}" for d in group.datasets))
        else:
            lines.append(
                f"[Group {group.label}] {'+'.join(group.tasks)} / "
                f"{group.n_documents} docs{limit_note} / {versions} -> {len(group.runs)} {plural}"
            )
            for run in group.runs:
                lines.append(f"  {run.model_id:<36} {_format_score(run)}")
        for warning in group.warnings:
            lines.append(f"  warning: {warning}")
        lines.append("")
    if not partition.groups:
        # Eighteen lines carrying one reason is a reader's job that belongs to the tool: what they need is the one sentence saying nothing was compared, and what was there instead.
        lines.append("Nothing was compared: no group could be formed.")
        if partition.datasets:
            lines.append("  declared: " + ", ".join(
                f"{d.label}={d.task}" for d in partition.datasets))
            found = Counter(
                "+".join(run.tasks) or "(none)" for run, _reason in partition.excluded)
            lines.append("  tasks found instead: " + ", ".join(
                f"{task} ({count} {'run' if count == 1 else 'runs'})"
                for task, count in sorted(found.items())))
        lines.append("")
    if partition.excluded:
        plural = "run" if len(partition.excluded) == 1 else "runs"
        lines.append(f"[Excluded] {len(partition.excluded)} {plural}")
        width = max(len(run.name) for run, _ in partition.excluded)
        for run, reason in partition.excluded:
            lines.append(f"  {run.name:<{width}}  {reason}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Pairing: which document in one language is which in another
# --------------------------------------------------------------------------


@dataclass
class Pairing:
    """How documents of different languages are matched to each other.

    Attributes:
        field: the `doc` field that identifies a document, or None when position is used.
        keys: run path -> (task_name, doc_id) -> the shared key, for paired documents.
        matched: documents the key covers in every language.
        note: the sentence printed and recorded, saying what was paired on and why.
        source: "field" (an identity field in the data), "mapping" (`--pair-mapping`),
            "position" (document order) or "none" (`--pair-strict` found no identity).
        strict: whether position pairing was refused.
        mapping: the mapping's path, content hash and scope, when one was used.
        canonical: run path -> (task_name, doc_id) -> canonical document id, including
            documents that were then excluded, so the table can say what they mapped to.
        excluded: run path -> (task_name, doc_id) -> why the pairing itself left it out.
            Not being in every language is derived by the table, not stored here.
        choice_ids / answer_ids: run path -> (task_name, doc_id) -> canonical choice ids by
            choice index, or the canonical answer id of a generative document.
    """

    field: str | None
    keys: dict[str, dict[tuple[str, int], str]]
    matched: int
    note: str
    source: str = ""
    strict: bool = False
    mapping: dict[str, Any] | None = None
    canonical: dict[str, dict[tuple[str, int], str]] = dataclasses.field(default_factory=dict)
    excluded: dict[str, dict[tuple[str, int], str]] = dataclasses.field(default_factory=dict)
    choice_ids: dict[str, dict[tuple[str, int], tuple[str, ...]]] = dataclasses.field(
        default_factory=dict)
    answer_ids: dict[str, dict[tuple[str, int], str]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            self.source = "field" if self.field else "position"


def _documents(run: Run) -> set[tuple[str, int]]:
    """Every (task_name, doc_id) the run graded, from `docs`, else from `samples.jsonl`."""
    docs = read_table(run.path, "docs")
    if len(docs):
        return {(str(task), int(doc)) for task, doc in zip(docs.task_name, docs.doc_id)}
    from .storage import read_samples

    return {(str(s["task_name"]), int(s["doc_id"])) for s in read_samples(run.path)}


# : Columns a `--pair-mapping` row must carry. `language` is a declared label or task.
MAPPING_FIELDS = ("language", "task_name", "doc_id", "canonical_doc_id")


@dataclass(frozen=True)
class PairingMapping:
    """An external document mapping, as read: its rows and the hash of its bytes."""

    path: str
    sha256: str
    rows: tuple[dict[str, Any], ...]


def load_pairing_mapping(path: str | os.PathLike) -> PairingMapping:
    """Read a `--pair-mapping` file: CSV with a header, or JSON Lines.

    Each row names one document of one language and the canonical document it is:

        language,task_name,doc_id,canonical_doc_id,canonical_choice_ids,canonical_answer_id
        en,mmlu_en,0,q-17,a|b|c|d,
        ko,mmlu_ko,3,q-17,b|a|c|d,

    `canonical_choice_ids` lists a canonical id per choice index (`|`-separated in CSV, a
    list in JSON Lines), for a translation that shuffled its choices; `canonical_answer_id`
    names the answer of a generative document whose translated target strings differ.
    Both are optional. Rows are validated for shape here; duplicates, many-to-one and
    coverage are properties of a group and are recorded when the mapping is applied.

    Raises:
        ValueError: on an unknown extension, a malformed line, a missing required column
            or a doc_id that is not a non-negative integer.
    """
    import csv
    import hashlib
    import io

    with open(path, "rb") as handle:
        content = handle.read()
    text = content.decode("utf-8")
    lower = str(path).lower()
    if lower.endswith(".csv"):
        records = list(csv.DictReader(io.StringIO(text)))
    elif lower.endswith(".jsonl"):
        records = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: not JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: each line must be a JSON object")
            records.append(record)
    else:
        raise ValueError(f"--pair-mapping must be a .csv or .jsonl file, got {path}")

    rows = []
    for number, record in enumerate(records, 1):
        missing = [name for name in MAPPING_FIELDS
                   if record.get(name) is None or str(record.get(name)).strip() == ""]
        if missing:
            raise ValueError(f"{path} row {number}: missing {', '.join(missing)}")
        raw = record["doc_id"]
        if isinstance(raw, bool) or not re.fullmatch(r"\d+", str(raw).strip()):
            raise ValueError(f"{path} row {number}: doc_id {raw!r} is not a non-negative integer")
        choices = record.get("canonical_choice_ids")
        if isinstance(choices, str):
            choices = choices.split("|") if choices.strip() else None
        elif choices is not None and not isinstance(choices, list):
            raise ValueError(f"{path} row {number}: canonical_choice_ids must be a list")
        answer = record.get("canonical_answer_id")
        rows.append({
            "row": number,
            "language": str(record["language"]).strip(),
            "task_name": str(record["task_name"]).strip(),
            "doc_id": int(str(raw).strip()),
            "canonical_doc_id": str(record["canonical_doc_id"]).strip(),
            "canonical_choice_ids": (tuple(str(c).strip() for c in choices)
                                     if choices is not None else None),
            "canonical_answer_id": (None if answer is None or str(answer).strip() == ""
                                    else str(answer).strip()),
        })
    return PairingMapping(str(path), hashlib.sha256(content).hexdigest(), tuple(rows))


def _index_mapping_rows(
    datasets: Sequence[Dataset], mapping: PairingMapping,
) -> tuple[dict[Dataset, dict[tuple[str, int], list[dict[str, Any]]]], int]:
    """dataset별 문서 키로 mapping을 묶고, 그룹 밖의 행 수를 센다.

    label과 task 이름을 모두 허용한다. 같은 이름이 여러 dataset에 걸리면 기존과
    같이 먼저 등록한 dataset을 사용한다. 중복 행은 덮어쓰지 않고 검사에 넘긴다.
    """
    by_name: dict[str, Dataset] = {}
    for dataset in datasets:
        by_name.setdefault(dataset.label, dataset)
        by_name.setdefault(dataset.task, dataset)
    table: dict[Dataset, dict[tuple[str, int], list[dict[str, Any]]]] = {}
    outside = 0
    for row in mapping.rows:
        dataset = by_name.get(row["language"])
        if dataset is None:
            outside += 1
            continue
        table.setdefault(dataset, {}).setdefault(
            (row["task_name"], row["doc_id"]), []).append(row)

    return table, outside


def _match_run_documents(
    documents: set[tuple[str, int]],
    rows_by_document: dict[tuple[str, int], list[dict[str, Any]]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], str], set[str]]:
    """한 실행 안에서 mapping 누락·중복을 판정하고 모호한 canonical ID를 반환한다.

    한 문서에 mapping 행이 여러 개면 duplicate_mapping_rows로 제외한다.
    남은 문서들끼리 canonical ID를 공유하면 many_to_one이다. 후자의 행은
    반환값에 남겨 제외된 문서도 원래 canonical ID를 추적할 수 있게 한다.
    """
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    excluded: dict[tuple[str, int], str] = {}
    for document in sorted(documents):
        found = rows_by_document.get(document)
        if not found:
            excluded[document] = "unmapped"
        elif len(found) > 1:
            excluded[document] = "duplicate_mapping_rows"
        else:
            selected[document] = found[0]

    owners = Counter(row["canonical_doc_id"] for row in selected.values())
    ambiguous: set[str] = set()
    for document, row in selected.items():
        if owners[row["canonical_doc_id"]] > 1:
            excluded[document] = "many_to_one"
            ambiguous.add(row["canonical_doc_id"])
    return selected, excluded, ambiguous


def _mapping_pairing(group: Group, mapping: PairingMapping) -> Pairing:
    """Pair this group's documents through an explicit mapping, recording what it left out.

    A row applies to the runs of the declared dataset its `language` names. Within one
    run, a document with no row is `unmapped`, one with several rows is
    `duplicate_mapping_rows`, and documents sharing one canonical id are `many_to_one` -
    which also removes that id from every other language (`counterpart_many_to_one`),
    because there is no ground for choosing which of the two is the counterpart.
    """
    table, outside = _index_mapping_rows(group.datasets, mapping)

    used: set[tuple[Dataset, tuple[str, int]]] = set()
    chosen: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    canonical: dict[str, dict[tuple[str, int], str]] = {}
    excluded: dict[str, dict[tuple[str, int], str]] = {}
    ambiguous: set[str] = set()
    for run in group.runs:
        dataset = group.dataset_of(run)
        rows_of = table.get(dataset, {})
        documents = _documents(run)
        run_rows, run_excluded, run_ambiguous = _match_run_documents(documents, rows_of)
        # 중복 때문에 pairing에서 빠진 행도 이 실행의 문서에 적용된 mapping이다.
        # 따라서 provenance의 미사용 행으로 세지 않는다.
        used.update((dataset, document) for document in documents if rows_of.get(document))
        ambiguous.update(run_ambiguous)
        chosen[run.path] = run_rows
        canonical[run.path] = {doc: row["canonical_doc_id"] for doc, row in run_rows.items()}
        excluded[run.path] = run_excluded

    # 한 실행에서 모호한 canonical ID는 다른 언어에서도 대응 대상을 확정할 수
    # 없다. 모든 실행을 검사한 뒤 전파하되 이미 기록한 제외 사유는 덮어쓰지 않는다.
    candidates = []
    for run in group.runs:
        for document, row in chosen[run.path].items():
            if row["canonical_doc_id"] in ambiguous and document not in excluded[run.path]:
                excluded[run.path][document] = "counterpart_many_to_one"
        candidates.append({row["canonical_doc_id"] for document, row in chosen[run.path].items()
                           if document not in excluded[run.path]})
    # 단순히 다른 언어에 없는 문서는 excluded에 넣지 않는다. 그 사유는
    # pairing_table에서 계산하고, 여기서는 모든 실행의 교집합만 선택한다.
    shared = set.intersection(*candidates) if candidates else set()

    keys: dict[str, dict[tuple[str, int], str]] = {}
    choice_ids: dict[str, dict[tuple[str, int], tuple[str, ...]]] = {}
    answer_ids: dict[str, dict[tuple[str, int], str]] = {}
    for run in group.runs:
        paired = {doc: row for doc, row in chosen[run.path].items()
                  if doc not in excluded[run.path] and row["canonical_doc_id"] in shared}
        keys[run.path] = {doc: row["canonical_doc_id"] for doc, row in paired.items()}
        choice_ids[run.path] = {doc: row["canonical_choice_ids"] for doc, row in paired.items()
                                if row["canonical_choice_ids"] is not None}
        answer_ids[run.path] = {doc: row["canonical_answer_id"] for doc, row in paired.items()
                                if row["canonical_answer_id"] is not None}

    without_document = Counter(
        dataset.label for dataset, rows_of in table.items()
        for document in rows_of if (dataset, document) not in used)
    provenance = {
        "path": mapping.path,
        "sha256": mapping.sha256,
        "rows": len(mapping.rows),
        "rows_outside_group": outside,
        "applied_to": [d.label for d in group.datasets if d in table],
        "keys_without_document": {d.label: without_document.get(d.label, 0)
                                  for d in group.datasets if d in table},
        "duplicate_keys": sum(1 for rows_of in table.values()
                              for found in rows_of.values() if len(found) > 1),
        "scope": "rows whose language names a declared dataset label or task, matched on "
                 "(task_name, doc_id) in that dataset's runs of this group",
    }
    return Pairing(
        None, keys, len(shared),
        f"paired through the mapping {os.path.basename(mapping.path)} (sha256 "
        f"{mapping.sha256[:12]}), which pairs {len(shared)} documents in every language - "
        "taken as given, not as proof that the documents say the same thing",
        source="mapping", mapping=provenance, canonical=canonical, excluded=excluded,
        choice_ids=choice_ids, answer_ids=answer_ids)


def _doc_fields(run: Run) -> dict[tuple[str, int], dict[str, str]]:
    """Each document's raw dataset row, as lm-eval logged it, reduced to scalar strings."""
    from .storage import read_samples

    found: dict[tuple[str, int], dict[str, str]] = {}
    for sample in read_samples(run.path):
        document = sample.get("doc")
        if not isinstance(document, dict):
            continue
        found[(sample["task_name"], int(sample["doc_id"]))] = {
            name: str(value) for name, value in document.items()
            if isinstance(value, (str, int, float, bool)) and value is not None
        }
    return found


def find_pairing_field(per_run: Sequence[dict[tuple[str, int], dict[str, str]]]) -> list[str]:
    """Fields that could identify the same document in every language.

    The test is structural, not a list of field names to hope for:

    * **unique within each run** - a field repeated across documents names a category, not a document;
    * **the same set of values in every run** - a field whose values differ between languages is the content, not the identity. `question` fails here and `sample_id` does not, which is the whole distinction.

    Both conditions are checked against the data, so a field that does not identify documents cannot be accepted by mistake; the cost of the rule being too strict is a fallback to position, which is what the report did before.

    Returns:
        Every qualifying field name, sorted. More than one is not yet an answer - two of them can induce different pairings - so the caller compares them.
    """
    if not per_run or any(not rows for rows in per_run):
        return []
    shared_names = set.intersection(*(
        set.intersection(*(set(fields) for fields in rows.values())) for rows in per_run))
    qualifying = []
    for name in sorted(shared_names):
        values = [[fields[name] for fields in rows.values()] for rows in per_run]
        if any(len(set(column)) != len(column) for column in values):
            continue                      # repeated inside one run: a category, not an identity
        if len({frozenset(column) for column in values}) != 1:
            continue                      # a different set per language: content, not identity
        qualifying.append(name)
    return qualifying


def attach_pairing(group: Group, field: str | None = None, strict: bool = False,
                   mapping: "PairingMapping | str | os.PathLike | None" = None) -> Pairing:
    """Decide how this group's documents are matched, and record the decision on the group.

    Position is a guess that usually happens to be right: it assumes the languages were published in one order and never reordered, filtered or deduplicated. When the runs carry a field that identifies a document, that assumption can be dropped rather than defended - and it is then the languages' *intersection* on that field that is compared, so differing document counts stop being a problem at all.

    Args:
        field: a field to use, from `--pair-on`. `"position"` forces the old behaviour.
            Without it the field is looked for and accepted only if it verifies.
        strict: `--pair-strict`. With no identity field and no mapping, compare no
            document one to one instead of falling back to position.
        mapping: `--pair-mapping`, a file or an already loaded `PairingMapping`. It
            replaces field discovery, so it cannot be combined with `field`.

    Ambiguity is refused rather than resolved: if several fields qualify and they do not induce the same pairing, there is no ground for preferring one, and `--pair-on` is how the user says which.
    """
    if group.axis != "language":
        return Pairing(None, {}, 0, "")
    if mapping is not None and field:
        raise ValueError("--pair-mapping and --pair-on both say how documents are paired; "
                         "pass one of them")
    if strict and field == "position":
        raise ValueError("--pair-strict refuses position pairing, which --pair-on position "
                         "asks for")
    if mapping is not None:
        if not isinstance(mapping, PairingMapping):
            mapping = load_pairing_mapping(mapping)
        group.pairing = _mapping_pairing(group, mapping)
        group.pairing.strict = strict
        return group.pairing
    if field == "position":
        group.pairing = Pairing(None, {}, 0, "paired on position, as --pair-on position asked")
        return group.pairing
    per_run = [_doc_fields(run) for run in group.runs]

    candidates = find_pairing_field(per_run)
    if field:
        if field not in candidates:
            raise LookupError(
                f"--pair-on {field!r} does not identify documents across these languages. "
                f"Fields that do: {candidates or 'none'}. Pass --pair-on position to "
                "compare by document order instead")
        candidates = [field]
    elif len(candidates) > 1:
        pairings = {
            name: [sorted(fields[name] for fields in rows.values()) for rows in per_run]
            for name in candidates}
        if len({tuple(map(tuple, value)) for value in pairings.values()}) != 1:
            raise LookupError(
                f"several fields identify documents here and they disagree: {candidates}. "
                "Pass --pair-on FIELD to say which, or --pair-on position")

    if not candidates:
        if strict:
            group.pairing = Pairing(
                None, {}, 0,
                "strict pairing: no field identifies a document in every language and no "
                "mapping was given, so no document is compared one to one - position is "
                "not used", source="none", strict=True)
            return group.pairing
        group.pairing = Pairing(
            None, {}, 0,
            "no field identifies a document in every language, so documents are paired by "
            "position - which trusts that the languages were published in one order")
        return group.pairing

    chosen = candidates[0]
    keys = {
        run.path: {key: fields[chosen] for key, fields in rows.items()}
        for run, rows in zip(group.runs, per_run)}
    matched = len(set.intersection(*(set(values.values()) for values in keys.values())))
    excluded = {run.path: {document: "no_identity_field" for document in _documents(run)
                           if document not in rows}
                for run, rows in zip(group.runs, per_run)}
    group.pairing = Pairing(
        chosen, keys, matched,
        f"paired on `{chosen}`, which identifies {matched} documents in every language - "
        "so the document order is not relied on",
        source="field", strict=strict,
        canonical={path: dict(values) for path, values in keys.items()}, excluded=excluded)
    return group.pairing


# --------------------------------------------------------------------------
# Does a shared doc_id mean a shared question?
# --------------------------------------------------------------------------

# : How much of the gold-choice agreement has to hold before documents of
# : different languages are compared one against one.
# A translated benchmark
# : preserves the position of the correct answer and scores near 1.0; two
# : unrelated benchmarks agree at chance, which is 1/n_choices. Nothing sits
# : near 0.9 by accident.
ALIGNMENT_THRESHOLD = 0.9


@dataclass
class Alignment:
    """Evidence that two runs' `doc_id`s refer to the same questions.

    Attributes:
        shared: documents present in every run of the group.
        agree: how many of those put the gold answer in the same place.
        rate: `agree / shared`.
        aligned: whether the per-document half of the report is allowed to run.
        source: "evidence" when measured, "asserted" when `--assume-aligned` overrode it.
        note: the sentence printed and written into the provenance.
        disagreeing: the documents whose gold answer sits in a different place per language.
        unverifiable: shared documents whose languages carry different kinds of gold
            evidence, or canonical choice ids that do not fit their choices. Neither
            agreement nor disagreement, so they are outside `rate`.
        status: shared doc key -> "agree", "disagree" or "unverifiable".
        evidence: run path -> doc key -> the gold evidence compared, as `kind:value`.
    """

    shared: int
    agree: int
    rate: float
    aligned: bool
    source: str
    note: str
    disagreeing: tuple[str, ...] = ()
    unverifiable: tuple[str, ...] = ()
    status: dict[str, str] = dataclasses.field(default_factory=dict)
    evidence: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)


def _gold_position(run: Run, group: Group) -> dict[str, str]:
    """doc key -> something that identifies the gold answer without naming it in one language.

    For a multiple-choice task that is the *index* of the gold choice: a translation replaces every string in the document, but the answer stays in position C.
    For a single-row (generative) task there are no positions, so the target string itself is used - which works for the numeric answers of a translated maths benchmark and for little else, and that limit is why the rate is reported rather than only its verdict.

    A `--pair-mapping` can say more, and where it does its evidence replaces those two: `choice_id:` is the canonical id of the gold choice, so a translation that shuffled its choices is compared by which choice is gold rather than where it sits; `answer_id:` is a generative document's canonical answer, for targets whose strings were translated. Canonical choice ids that do not cover exactly this document's choice indices are `invalid:` rather than guessed at.
    """
    docs = read_table(run.path, "docs")
    if not len(docs):
        return {}
    pairing = group.pairing
    choice_ids = pairing.choice_ids.get(run.path, {}) if pairing else {}
    answer_ids = pairing.answer_ids.get(run.path, {}) if pairing else {}
    rows_of: dict[str, tuple[tuple[str, int], list[Any]]] = {}
    for row in docs.itertuples():
        document = (str(row.task_name), int(row.doc_id))
        rows_of.setdefault(group.doc_key(run, *document), (document, []))[1].append(row)
    positions: dict[str, str] = {}
    for key, (document, rows) in rows_of.items():
        gold = [int(row.choice_idx) for row in rows
                if getattr(row, "is_target_choice", None) is True]
        if gold:
            ids = choice_ids.get(document)
            if ids is None:
                positions[key] = f"choice:{gold[-1]}"
            elif (sorted({int(row.choice_idx) for row in rows}) != list(range(len(ids)))
                  or len(set(ids)) != len(ids)):
                positions[key] = "invalid:canonical_choice_ids_do_not_match_choices"
            else:
                positions[key] = f"choice_id:{ids[gold[-1]]}"
        elif len(rows) == 1:
            answer = answer_ids.get(document)
            positions[key] = (f"answer_id:{answer}" if answer is not None
                              else f"target:{str(rows[0].target).strip()}")
    return positions


def gold_alignment(group: Group, assume: bool = False) -> Alignment:
    """Ask the data whether this group's `doc_id`s line up, instead of assuming it.

    The claim under a multilingual comparison is that document 5 is the same question in every language.
    Nothing in a manifest can establish that - both runs will happily report 500 documents whether they are translations of each other or two unrelated benchmarks that happen to be the same size.

    What *can* be measured is where the gold answer sits.
    A translated multiple-choice benchmark keeps the correct answer in the same position, so the runs agree on essentially every document; two different benchmarks agree at chance, around 1/n_choices.
    The gap between those is wide enough that a single rate decides it, and the rate is printed either way so the decision can be checked.

    Args:
        assume: take alignment as given, still measuring and reporting the rate.
            For the case this check cannot see: a translation that also shuffled the choices, where the documents do correspond but the gold position does not.

    Example:
        >>> gold_alignment(group).rate                      # doctest: +SKIP
        0.998
    """
    per_run = [_gold_position(run, group) for run in group.runs]
    if not per_run or any(not positions for positions in per_run):
        # `is_target_choice` is false on every row of at least one run, which happens when the task's target could not be matched to any of its choices. Naming the run and the cause is the difference between a check that declined and a check that looks broken.
        silent = [group.series_of(run) for run, positions in zip(group.runs, per_run)
                  if not positions]
        return Alignment(
            0, 0, 0.0, bool(assume), "asserted" if assume else "evidence",
            f"no gold answer is marked in {', '.join(silent) or 'any run'} - the task's "
            "target matched none of its choices, so where the answer sits cannot be "
            "compared. Re-run to rebuild `docs` if this run predates that being resolved "
            "for label targets")
    evidence = {run.path: positions for run, positions in zip(group.runs, per_run)}
    shared = set(per_run[0])
    for positions in per_run[1:]:
        shared &= set(positions)
    if not shared:
        strict = group.pairing is not None and group.pairing.source == "none"
        return Alignment(0, 0, 0.0, bool(assume), "asserted" if assume else "evidence",
                         "strict pairing paired no document across the languages" if strict
                         else "the languages share no doc_id at all", evidence=evidence)
    status: dict[str, str] = {}
    for key in shared:
        values = [positions[key] for positions in per_run]
        kinds = {value.partition(":")[0] for value in values}
        status[key] = ("unverifiable" if "invalid" in kinds or len(kinds) > 1
                       else "disagree" if len(set(values)) > 1 else "agree")
    disagreeing = tuple(sorted(key for key, state in status.items() if state == "disagree"))
    unverifiable = tuple(sorted(key for key, state in status.items() if state == "unverifiable"))
    agree = len(shared) - len(disagreeing) - len(unverifiable)
    compared = agree + len(disagreeing)
    rate = agree / compared if compared else 0.0
    aligned = bool(compared) and rate >= ALIGNMENT_THRESHOLD
    kinds = {positions[key].partition(":")[0] for positions in per_run for key in shared}
    note = (f"{agree}/{compared} shared documents ({rate:.1%}) put the gold answer in the "
            "same place across the languages"
            + (" (canonical choice/answer ids from the mapping where given)"
               if kinds & {"choice_id", "answer_id"} else "")
            + (f"; {len(unverifiable)} more cannot be checked, because their languages carry "
               "different kinds of gold evidence or canonical choice ids that do not fit "
               "their choices" if unverifiable else ""))
    if aligned:
        refused = len(disagreeing) + len(unverifiable)
        dropped = (f"; the {refused} that do not or cannot be checked are left out of the "
                   "per-document half, because a document whose answer key differs by "
                   "language is not the same question however the group as a whole scored"
                   if refused else "")
        return Alignment(len(shared), agree, rate, True, "evidence",
                         note + " - read as translations of one document set" + dropped,
                         disagreeing, unverifiable, status, evidence)
    if assume:
        # Asserted alignment is the case where the gold positions are *expected* to differ - a translation that shuffled its choices - so dropping the documents that differ would drop exactly what was asserted.
        return Alignment(len(shared), agree, rate, True, "asserted",
                         note + " - below the threshold, but alignment was asserted "
                                "with --assume-aligned, so every shared document is kept",
                         (), (), status, evidence)
    return Alignment(
        len(shared), agree, rate, False, "evidence",
        note + f" - below {ALIGNMENT_THRESHOLD:.0%}, so these doc_ids are not treated as the "
               "same questions; curves are kept, per-document examples are not. If the "
               "translation shuffled the choices, give canonical choice ids with "
               "--pair-mapping or pass --assume-aligned",
        disagreeing, unverifiable, status, evidence)


# --------------------------------------------------------------------------
# Relative depth
# --------------------------------------------------------------------------


def relative_depth(index: float, n_blocks: int, layer_domain: str) -> float:
    """Where a layer sits on the 0..1 depth axis.

    Absolute layer numbers cannot be compared between models of different depths, so the report always normalises.
    The two axes normalise differently:

    * residual (`layer`, 0..L): `i / L`.
        Layer 0 is 0.0, layer L is 1.0.
    * block (`block`, 0..L-1): `(j + 0.5) / L`.
      Block j's attention happens between `layer j` and `layer j+1`, so it belongs at the midpoint.
      Placing it at `j / L` would shift the attention curve half a layer against the residual curve whenever the two are drawn together.

    Example:
        >>> relative_depth(0, 32, "residual"), relative_depth(32, 32, "residual")
        (0.0, 1.0)
        >>> relative_depth(0, 32, "block")
        0.015625
    """
    if n_blocks <= 0:
        return 0.0
    if layer_domain == "block":
        return (index + 0.5) / n_blocks
    return index / n_blocks


def prediction_depth(signals, n_blocks: int):
    """The depth at which the lens settles on the model's final answer.

    For one scored position: the smallest layer `k` such that the logit-lens top-1 at every layer from `k` to `L` equals the top-1 at `L`.
    Returned as a relative depth, so models of different depths are comparable.

    A value near 1.0 means the answer only appears in the last layers; a smaller value means it is settled earlier and the remaining layers only sharpen it.

    Returns:
        A DataFrame with one row per (doc_id, choice_idx, step).

    Example:
        >>> prediction_depth(signals, n_blocks=24).depth.median()   # doctest: +SKIP
        0.7916666666666666
    """
    import pandas as pd

    rows = []
    keys = ["task_name", "doc_id", "choice_idx", "step"]
    for key, group in signals.groupby(keys, sort=False):
        group = group.sort_values("layer")
        tokens = group.lens_token_id.tolist()
        layers = group.layer.tolist()
        final = tokens[-1]
        # Walk back from the last layer while the prediction is unchanged; the first layer that disagrees ends the settled run.
        settled = layers[-1]
        for layer, token in zip(reversed(layers), reversed(tokens)):
            if token != final:
                break
            settled = layer
        record = dict(zip(keys, key))
        record["settled_layer"] = settled
        record["depth"] = relative_depth(settled, n_blocks, "residual")
        record["is_correct"] = bool(group.is_correct.iloc[0]) if "is_correct" in group else None
        record["is_target_choice"] = (
            bool(group.is_target_choice.iloc[0]) if "is_target_choice" in group else None
        )
        rows.append(record)
    return pd.DataFrame(rows)


# : Which columns of which table become a depth curve, and on which axis they : are placed.
# "pair" means the value describes the gap between layer i and : i+1, so it is drawn at the midpoint like a block-domain signal.
COMPARISON_SIGNALS: list[dict[str, Any]] = [
    {"table": "signals", "column": "lens_prob", "index": "layer", "domain": "residual"},
    {"table": "signals", "column": "target_percentile", "index": "layer", "domain": "residual"},
    {"table": "signals", "column": "target_rank", "index": "layer", "domain": "residual"},
    {"table": "similarity", "column": "cos", "index": "layer_i", "domain": "pair",
     "filter": "adjacent"},
    {"table": "attn_norm", "column": "value_norm", "index": "block", "domain": "block"},
]


def _column_is_vocab_comparable(table: str, column: str) -> bool:
    """Read `comparable_across_vocab` straight off the schema declaration."""
    for spec in TABLES[table].columns:
        if spec.name == column:
            return spec.comparable_across_vocab
    return True


def summarize_run(run: Run, group: "Group | None" = None):
    """Reduce one run to a long-format depth curve per signal.

    Averages over documents, choices and steps, which is a sanity-check view - the per-document detail stays in the parquet files for whoever draws the real figure.

    Returns:
        A pandas DataFrame with one row per (signal, layer index).

    Example:
        >>> summarize_run(run).head(2)                       # doctest: +SKIP
           signal     layer_domain  layer_index  relative_depth  value
        0  lens_prob  residual      0            0.0             0.0021
        1  lens_prob  residual      1            0.03125         0.0034
    """
    import pandas as pd

    # `series` is the line a curve is drawn as, and it is the one column that changes meaning with the axis: the model when the data is fixed, the language when the model is.
    # Both `model_id` and `language` stay in the table regardless, so a saved `comparison.parquet` can always be re-read without knowing which command wrote it.
    group_label = group.label if group else ""
    dataset = group.dataset_of(run) if group else None
    series = group.series_of(run) if group else run.model_id
    n_blocks = int(run.manifest.get("n_blocks", 0))
    frames = []
    for signal in COMPARISON_SIGNALS:
        table = read_table(run.path, signal["table"])
        if not len(table) or signal["column"] not in table.columns:
            continue
        if signal.get("filter") == "adjacent":
            # Only the layer i -> i+1 pairs: the full triangle is a matrix, not a curve, and belongs in the parquet rather than in this summary.
            table = table[table["layer_j"] == table["layer_i"] + 1]
        values = table.dropna(subset=[signal["column"]])
        if not len(values):
            continue
        grouped = values.groupby(signal["index"])[signal["column"]].agg(["mean", "size"])
        domain = "block" if signal["domain"] == "pair" else signal["domain"]
        frames.append(
            pd.DataFrame(
                {
                    "group": group_label,
                    "axis": group.axis if group else "model",
                    "series": series,
                    "run_path": run.path,
                    "run_name": run.name,
                    "model_id": run.model_id,
                    "language": dataset.label if dataset else None,
                    "task": "+".join(run.tasks),
                    "vocab_size": run.vocab_size,
                    "n_blocks": n_blocks,
                    "signal": signal["column"],
                    "layer_domain": signal["domain"],
                    "layer_index": grouped.index.astype(int),
                    "relative_depth": [
                        relative_depth(int(i), n_blocks, domain) for i in grouped.index
                    ],
                    "value": grouped["mean"].to_numpy(),
                    "n_rows": grouped["size"].to_numpy(),
                    "comparable_across_vocab": _column_is_vocab_comparable(
                        signal["table"], signal["column"]
                    ),
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_comparison(partition: Partition):
    """Long-format model x layer x signal table across every grouped run.

    Example:
        >>> build_comparison(part).head(1)                   # doctest: +SKIP
          group model_id      signal     layer_index  relative_depth  value
        0 A     Qwen/Qwen3-8B lens_prob  0            0.0             0.0021
    """
    import pandas as pd

    frames = [
        summarize_run(run, group)
        for group in partition.groups
        for run in group.runs
    ]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def wrap_for_page(text: str, width: int = 108) -> str:
    """Fold long lines so the page keeps them.

    `figure.text` does not wrap: anything past the page edge is drawn outside the canvas and simply lost. At 8pt monospace on A4 that edge is 118 characters, and three lines of a seven-model report ran past it - including the one naming which models a group contains, so the page could not say what it had compared, and a vocabulary warning that ended mid-word.

    Continuations keep the original line's indentation plus two spaces, so a folded model list still reads as one item.
    """
    import textwrap

    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if len(line) <= width or not stripped:
            out.append(line)
            continue
        out.extend(textwrap.wrap(
            stripped, width=width, initial_indent=indent,
            subsequent_indent=indent + "  ", break_long_words=False,
            break_on_hyphens=False))
    return "\n".join(out)


def render_pdf(partition: Partition, comparison, output_path: str) -> str:
    """Draw one page of grouping text plus one page of curves per group.

    This is a sanity check, not a paper figure.
    There is no plotting API: the real figures are drawn by the user from `load_signals()`.

    Signals that are not comparable across vocabularies are skipped in a group whose runs disagree on vocabulary size, rather than being drawn with a caveat nobody reads.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        # Page 1: what was found, what was grouped, what was dropped.
        figure = plt.figure(figsize=(8.27, 11.69))
        header = (
            f"runs found: {partition.n_found}   "
            f"grouped: {sum(len(g.runs) for g in partition.groups)}   "
            f"excluded: {len(partition.excluded)}\n\n"
        )
        figure.text(0.05, 0.95, wrap_for_page(header + format_partition(partition)),
                    va="top", family="monospace", fontsize=8)
        pdf.savefig(figure)
        plt.close(figure)

        for group in partition.groups:
            rows = comparison[comparison["group"] == group.label] if len(comparison) else comparison
            if not len(rows):
                continue
            mixed_vocab = len(group.vocab_sizes()) > 1
            signals = [
                name
                for name in rows["signal"].unique()
                if not (mixed_vocab and not rows[rows["signal"] == name]
                        ["comparable_across_vocab"].iloc[0])
            ]
            if not signals:
                continue
            figure, axes = plt.subplots(
                len(signals), 1, figsize=(8.27, 2.4 * len(signals) + 1), squeeze=False
            )
            for axis, signal in zip(axes[:, 0], signals):
                subset = rows[rows["signal"] == signal]
                for series, curve in subset.groupby("series"):
                    curve = curve.sort_values("relative_depth")
                    axis.plot(curve["relative_depth"], curve["value"], marker=".", label=series)
                axis.set_ylabel(signal)
                axis.set_xlabel("relative depth  l / L")
                axis.legend(fontsize=6)
            if group.axis == "language":
                title = (f"Group {group.label}: {group.runs[0].model_id} across "
                         + ", ".join(group.series_of(run) for run in group.runs))
            else:
                title = (f"Group {group.label}: {'+'.join(group.tasks)} "
                         f"({group.n_documents} docs)")
            axes[0, 0].set_title(title)
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)
    return output_path


def run_report(
    paths: Sequence[str],
    output_dir: str,
    reference: str | None = None,
    examples_per_category: int = 3,
    datasets: Sequence[Dataset] | None = None,
    assume_aligned: bool = False,
    pair_on: str | None = None,
    pair_strict: bool = False,
    pair_mapping: str | None = None,
) -> Partition:
    """The `report` command: discover, partition, print, then write the outputs.

    Writes four files into `output_dir`:

    * `comparison.parquet` and `report.pdf` - the aggregate view, model x layer x signal, which says *which* model is better;
    * `examples.parquet` and `examples.md` - the qualitative view, which says *on what*, by bucketing documents by who got them right.

    The two share a partition rather than each computing their own.
    Which runs may be compared is one question, and letting the curves and the examples answer it differently would be a way to draw a figure from one set of runs and quote examples from another.

    Args:
        reference: substring of the series label the example buckets are defined against - a model id normally, a language under `--multilingual`.
            Default: the highest-scoring run in each group, printed.
        examples_per_category: documents sampled per bucket; 0 skips the qualitative half.
        datasets: what `--multilingual` declared, or None for the default report.
            One report answers one of those questions: the reference, the buckets and the figure legends all mean something different in each, so producing both at once would produce a file whose columns mean two things.
        assume_aligned: under `--multilingual`, compare documents by `doc_id` even when the gold-answer check falls below the threshold.
        pair_on: a `doc` field that identifies a document across languages, or "position".
            Without it a field is looked for and used only if it verifies.
        pair_strict: under `--multilingual`, compare no document one to one when neither
            an identity field nor a mapping pairs it, instead of pairing by position.
        pair_mapping: a CSV/JSONL file mapping (language, task_name, doc_id) to canonical
            document, choice and answer ids. Its hash and scope go into `pairing.json`.

    Under `--multilingual` two more files are written: `pairing.parquet`, one row per graded document with its pairing and answer status, and `pairing.json`, the per-group source, mapping scope, coverage and alignment.

    Example:
        >>> run_report(["results/"], ".")                    # doctest: +SKIP
        Found 10 runs
        ...
    """
    runs = discover_runs(paths)
    for run in runs:
        if run.manifest.get("internal_signals_available") is False:
            print(f"{run.name}: scoring-only run; internal-signal curves and prediction depth are unavailable")
    partition = partition_runs(runs, datasets=datasets)
    # Printed before anything is drawn, on purpose.
    print(format_partition(partition))

    os.makedirs(output_dir, exist_ok=True)
    comparison = build_comparison(partition)
    if len(comparison):
        comparison.to_parquet(os.path.join(output_dir, "comparison.parquet"), index=False)
    render_pdf(partition, comparison, os.path.join(output_dir, "report.pdf"))

    if pair_mapping is not None and partition.axis != "language":
        raise ValueError("--pair-mapping applies to --multilingual reports only")
    mapping = load_pairing_mapping(pair_mapping) if pair_mapping is not None else None
    if partition.axis == "language":
        pairing = write_pairing(partition, output_dir, pair_on, pair_strict, mapping,
                                assume_aligned)
        if len(pairing):
            print(f"wrote {len(pairing)} pairing rows to {output_dir}/pairing.parquet "
                  "and pairing.json")

    if examples_per_category > 0 and partition.groups:
        examples = write_examples(partition, output_dir, reference, examples_per_category,
                                  assume_aligned=assume_aligned, pair_on=pair_on,
                                  pair_strict=pair_strict, pair_mapping=mapping)
        if len(examples):
            print(f"wrote {len(examples)} example rows to {output_dir}/examples.parquet "
                  "and examples.md")
        else:
            # Saying a file was written when it was not is how someone ends up looking for it.
            print("no examples were written: no document was eligible in any group "
                  "(the reason per group is printed above)")
    return partition


# --------------------------------------------------------------------------
# Qualitative examples
# --------------------------------------------------------------------------

# : How documents are bucketed when several models solved the same set. : Aggregate curves say a model is better; they never say on *what*.
# These : buckets do, and the two disagreement ones are where a paper's examples come : from - a document everyone gets right shows nothing.
CATEGORIES = (
    "reference_only_correct",   # the reference is right where every other model is wrong
    "reference_only_wrong",     # the reference is wrong where every other model is right
    "all_correct",              # control
    "all_wrong",                # control
)


def reference_run(group: Group, name: str | None = None) -> Run:
    """Pick the run the buckets are defined against.

    Args:
        name: substring of the series label - a model id normally, a language under `--multilingual`.
            Without it the highest-scoring run is used, and the choice is printed.

    "the last run in the sweep" is deliberately not the default: it is whatever order the shell loop happened to use, so reordering the loop would silently change which model the examples are about.

    Raises:
        LookupError: when `name` matches no run, or more than one.
    """
    if name:
        matches = [run for run in group.runs if name in group.series_of(run)]
        if len(matches) != 1:
            raise LookupError(
                f"--reference {name!r} matched {len(matches)} runs in group "
                f"{group.label}: {[group.series_of(r) for r in group.runs]}"
            )
        return matches[0]
    def rank(run):
        score = _score_of(run)
        if score is None:
            return float('-inf')
        protocols = run.manifest.get("benchmarks", {}).get("protocols", {})
        spec = protocols.get(run.tasks[0], {}) if len(run.tasks) == 1 else {}
        return score if spec.get("higher_is_better", True) else -score
    return max(group.runs, key=rank)


def _score_of(run: Run) -> float | None:
    """The run's headline metric, preferring a length-normalised accuracy.

    Only the rows for the tasks the run actually asked for are read.
    An lm-eval *group* writes one row per subtask **and** its own aggregate row into the same results dict, so taking whichever row came first makes the headline an arbitrary subtask's score: `global_mmlu_en` yields `global_mmlu_en_business`, 20 documents standing in for 120. The aggregate row is lm-eval's own, computed with the weighting the group declared (`weight_by_size` for Global-MMLU), so it is the number to read - and it agrees with the `docs` table, which is how this was found.

    A run that asked for several tasks has no single headline, so those are averaged, unweighted.

    Keys without a comma - `sample_len` and other bookkeeping lm-eval writes alongside the metrics - are skipped: a metric key carries its filter after a comma, and `sample_len` would otherwise be eligible to become the score of a task with no recognised metric.
    """
    protocols = run.manifest.get("benchmarks", {}).get("protocols", {})
    if protocols:
        # Different tasks/metrics have no common scale; do not invent an average.
        if len(run.tasks) != 1 or run.tasks[0] not in protocols:
            return None
        task = run.tasks[0]
        spec = protocols[task]
        value = run.scores.get(task, {}).get(f"{spec['primary_metric']},{spec['primary_filter']}")
        return float(value) if isinstance(value, (float, int)) else None
    rows = [run.scores[task] for task in run.tasks if task in run.scores]
    if not rows:                       # not a group run, or the task was renamed
        rows = list(run.scores.values())
    collected: dict[str, list[float]] = {}
    for task_scores in rows:
        for key, value in task_scores.items():
            if "," in key and "stderr" not in key and isinstance(value, (int, float)):
                collected.setdefault(key.split(",")[0], []).append(float(value))
    for name in ("acc_norm", "acc", "exact_match"):
        if name in collected:
            return sum(collected[name]) / len(collected[name])
    first = next(iter(collected.values()), None)
    return sum(first) / len(first) if first else None


def _verdicts(run: Run, group: Group) -> dict[str, bool]:
    """doc key -> did this run answer it correctly.

    Read from the `docs` table, which carries lm-eval's own grading; correctness is per document, so any choice row of it will do.
    The key comes from the group rather than from the row, because under `--multilingual` the task name is what differs between the runs being compared.
    """
    docs = read_table(run.path, "docs")
    if not len(docs):
        return {}
    return {
        group.doc_key(run, task, int(doc_id)): bool(correct)
        for task, doc_id, correct in zip(docs.task_name, docs.doc_id, docs.is_correct)
        if correct is not None and not _isna(correct)
    }


def _isna(value: object) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


def prompt_hashes(run: Run, group: Group | None = None) -> dict[str, str]:
    """doc key -> the hash lm-eval recorded for that document's prompt."""
    from .storage import read_samples

    found = {}
    for sample in read_samples(run.path):
        digest = sample.get("prompt_hash")
        if digest:
            key = (group.doc_key(run, sample["task_name"], int(sample["doc_id"])) if group
                   else (sample["task_name"], int(sample["doc_id"])))
            found[key] = digest
    return found


def comparable_documents(
    group: Group, alignment: "Alignment | None" = None
) -> tuple[list[str], list[str]]:
    """The documents whose answers may be put in one row, and the ones that may not.

    The two axes are held to different standards, because only one of them can meet the stricter one.

    **Default - the same prompt.** Sharing a `doc_id` set is not enough to put two models' answers in one row.
    Few-shot examples are drawn per run, so the same `doc_id` can carry a different prompt, and a side-by-side table of answers to different questions looks entirely normal. lm-eval records a `prompt_hash` per document, so this is cheap to check rather than assume.

    **`--multilingual` - the same question.** Here the prompts differ by construction: that is the whole point of the comparison, and requiring an identical `prompt_hash` would reject every document and quietly produce an empty examples file.
    So the gate becomes the one thing that is still checkable - `gold_alignment` - and it is a gate on the *group*: either the languages are translations of one document set and every shared `doc_id` is eligible, or they are not and none of them is.

    Returns:
        (comparable, mismatched) keys.
    """
    if group.axis == "language":
        alignment = alignment if alignment is not None else gold_alignment(group)
        per_run = [set(_verdicts(run, group)) for run in group.runs]
        if not per_run:
            return [], []
        shared = sorted(set.intersection(*per_run))
        if not alignment.aligned:
            return [], shared
        # The group-level threshold decides whether to compare at all; the evidence is
        # per document, so it is also applied per document. A document the languages
        # disagree about has a different answer key in each, and quoting it as "right in
        # English, wrong in Korean" would describe a difference in the dataset as a
        # difference in the model.
        refused = set(alignment.disagreeing) | set(alignment.unverifiable)
        return ([key for key in shared if key not in refused],
                [key for key in shared if key in refused])

    per_run = [prompt_hashes(run, group) for run in group.runs]
    if not per_run or not per_run[0]:
        return [], []
    shared = set(per_run[0])
    for hashes in per_run[1:]:
        shared &= set(hashes)
    comparable, mismatched = [], []
    for key in sorted(shared):
        digests = {hashes[key] for hashes in per_run}
        (comparable if len(digests) == 1 else mismatched).append(key)
    return comparable, mismatched


def categorize(group: Group, reference: Run, keys: Sequence[str]):
    """Bucket documents by who got them right.

    Returns:
        (buckets, mixed) - `mixed` holds documents where the other models disagree among themselves, which is neither a clean contrast nor a control; it is counted and reported, not sampled.
    """
    verdicts = {run.path: _verdicts(run, group) for run in group.runs}
    others = [run for run in group.runs if run.path != reference.path]

    buckets: dict[str, list[str]] = {name: [] for name in CATEGORIES}
    mixed: list[str] = []
    for key in keys:
        if any(key not in verdicts[run.path] for run in group.runs):
            continue
        mine = verdicts[reference.path][key]
        rest = [verdicts[run.path][key] for run in others]
        if not rest:                                    # a group of one
            buckets["all_correct" if mine else "all_wrong"].append(key)
        elif mine and all(rest):
            buckets["all_correct"].append(key)
        elif not mine and not any(rest):
            buckets["all_wrong"].append(key)
        elif mine and not any(rest):
            buckets["reference_only_correct"].append(key)
        elif not mine and all(rest):
            buckets["reference_only_wrong"].append(key)
        else:
            mixed.append(key)
    return buckets, mixed


def _settle_depth(run: Run, group: Group, keys: Sequence[str]) -> dict[str, float]:
    """Relative depth at which each document's answer stops changing.

    This is what makes these examples worth more than a generic evaluation dump: alongside "model A was right and model B was wrong" it says where in the stack each of them made up its mind.
    """
    signals = load_signals(run.path, "signals")
    if not len(signals):
        return {}
    wanted = set(keys)
    signals = signals[
        [group.doc_key(run, task, int(doc_id)) in wanted
         for task, doc_id in zip(signals.task_name, signals.doc_id)]
    ]
    if "is_target_choice" in signals and signals.is_target_choice.any():
        signals = signals[signals.is_target_choice == True]  # noqa: E712
    if not len(signals):
        return {}
    depths = prediction_depth(signals, int(run.manifest.get("n_blocks", 0)))
    return {
        group.doc_key(run, task, int(doc_id)): float(depth)
        for task, doc_id, depth in zip(depths.task_name, depths.doc_id, depths.depth)
    }


def _answers(run: Run, group: Group, keys: Sequence[str]) -> dict[str, dict]:
    """What this run predicted for each document, as lm-eval graded it."""
    docs = read_table(run.path, "docs")
    wanted = set(keys)
    found: dict[str, dict] = {}
    for row in docs.itertuples():
        key = group.doc_key(run, row.task_name, int(row.doc_id))
        if key not in wanted or key in found:
            continue
        found[key] = {
            "predicted": row.predicted,
            "target": row.target,
            "is_correct": None if _isna(row.is_correct) else bool(row.is_correct),
            # Where *this* run keeps it. The shared key names the document across languages and locates it in none of them, so the prompt could not be looked up again from it - and under a pairing key each language has its own doc_id as well as its own task.
            "task_name": row.task_name,
            "doc_id": int(row.doc_id),
        }
    return found


def select_examples(
    group: Group,
    reference: Run,
    per_category: int = 3,
    seed: int = SAMPLING_SEED,
    assume_aligned: bool = False,
):
    """Sample documents from each bucket, and record how the sample was drawn.

    The selection rule is part of the output on purpose.
    A tool that picks examples is ordinary; what separates a systematic sample from a hand-picked one, to a reader, is being able to see the seed, the buckets, how many documents were eligible and how many were taken.

    Returns:
        (rows, provenance) - `rows` is one record per (document, model), `provenance` describes the draw.
    """
    import random

    alignment = gold_alignment(group, assume_aligned) if group.axis == "language" else None
    comparable, mismatched = comparable_documents(group, alignment)
    buckets, mixed = categorize(group, reference, comparable)

    rng = random.Random(seed)
    chosen: dict[str, list[str]] = {}
    for name in CATEGORIES:
        pool = buckets[name]
        chosen[name] = sorted(rng.sample(pool, min(per_category, len(pool))))

    picked = [key for keys in chosen.values() for key in keys]
    depths = {run.path: _settle_depth(run, group, picked) for run in group.runs}
    answers = {run.path: _answers(run, group, picked) for run in group.runs}

    rows = []
    for category, keys in chosen.items():
        for key in keys:
            for run in group.runs:
                answer = answers[run.path].get(key, {})
                rows.append({
                    "group": group.label,
                    "axis": group.axis,
                    "category": category,
                    # `document` is the shared identity; `task_name` and `doc_id` are where this run keeps it, which under a pairing key differ per language.
                    "document": key,
                    "task_name": answer.get("task_name"),
                    "doc_id": answer.get("doc_id"),
                    "series": group.series_of(run),
                    # The subtask this run actually ran, which is what locates the prompt in its own samples.jsonl.
                    "run_task": answer.get("task_name") or (
                        run.tasks[0] if len(run.tasks) == 1 else "+".join(run.tasks)),

                    # `model` and `language` are both kept whichever mode wrote the file: which of the two varies is a property of the report, not of the row, and a saved parquet outlives the command that produced it.
                    "model": run.model_id,
                    "language": (group.dataset_of(run).label
                                 if group.dataset_of(run) else None),
                    "is_reference": run.path == reference.path,
                    "is_correct": answer.get("is_correct"),
                    "predicted": answer.get("predicted"),
                    "target": answer.get("target"),
                    "settle_depth": depths[run.path].get(key),
                    "n_blocks": int(run.manifest.get("n_blocks", 0)),
                    "run_path": run.path,
                })

    provenance = {
        "group": group.label,
        "axis": group.axis,
        "datasets": [f"{d.label}={d.task}" for d in group.datasets],
        "tasks": list(group.tasks),
        "reference": group.series_of(reference),
        "models": [group.series_of(run) for run in group.runs],
        "seed": seed,
        "per_category": per_category,
        "eligible": len(comparable),
        # The two modes exclude documents for different reasons, and one count holding both would be a count nobody can read: normally a document drops out because the prompts differed, under `--multilingual` because the whole group failed the alignment check.
        "excluded_prompt_mismatch": len(mismatched) if group.axis != "language" else 0,
        "excluded_not_aligned": len(mismatched) if group.axis == "language" else 0,
        "disagreeing": list(mismatched) if group.axis == "language" else [],
        "pool_sizes": {name: len(buckets[name]) for name in CATEGORIES},
        "taken": {name: len(chosen[name]) for name in CATEGORIES},
        "mixed_not_sampled": len(mixed),
        "paired_on": group.pairing.note if group.pairing else "",
        "pairing_source": group.pairing.source if group.pairing else "",
        "unverifiable": list(alignment.unverifiable) if alignment else [],
        "eligibility": (
            "identical prompt_hash per document" if group.axis != "language"
            else f"gold-answer alignment across languages ({alignment.source}): {alignment.note}"
        ),
    }
    return rows, provenance


def _prompt_and_choices(run_dir: str, task_name: str | None, doc_id: int, tail: int = 400):
    """The question as the model saw it, plus its choices.

    Taken from `samples.jsonl`'s `arguments`, which is the actual request, so this works for any task without knowing that task's dataset fields.
    Few-shot prompts run to hundreds of tokens, so only the tail is shown - that is the question itself.
    """
    from .storage import read_samples

    for sample in read_samples(run_dir):
        # `task_name` is None when the caller has no name to give: the doc_id alone then locates the sample, which is only safe for a run with a single task.
        if (task_name is not None and sample["task_name"] != task_name) \
                or int(sample["doc_id"]) != doc_id:
            continue
        arguments = sample.get("arguments") or []
        if not arguments:
            return "", []
        context = str(arguments[0][0])
        choices = [str(a[1]) for a in arguments if len(a) > 1 and isinstance(a[1], str)]
        prefix = "..." if len(context) > tail else ""
        return prefix + context[-tail:], choices
    return "", []


def render_examples(rows, provenance_list, output_path: str) -> str:
    """Write the readable half: one markdown section per bucket.

    Markdown rather than a figure, because these go into a draft as text.
    """
    import pandas as pd

    frame = pd.DataFrame(rows)
    lines = ["# Qualitative examples", ""]
    for provenance in provenance_list:
        per_language = provenance.get("axis") == "language"
        heading = (f"## Group {provenance['group']} - {', '.join(provenance['datasets'])}"
                   if per_language
                   else f"## Group {provenance['group']} - {'+'.join(provenance['tasks'])}")
        dropped = provenance.get("disagreeing") or []
        eligibility = (
            f"- {provenance['eligible']} documents are shared by every language and treated as "
            f"the same question. Paired: {provenance['paired_on']}. "
            f"Checked: {provenance['eligibility']}."
            + ("\n- Left out, because the languages disagree on which answer is gold, or "
               "that cannot be checked: "
               + ", ".join(f"`{str(key).lstrip('_')}`" for key in dropped[:20])
               + (" ..." if len(dropped) > 20 else "")
               + ". These are worth reading as dataset findings rather than model findings."
               if dropped else "")
            if per_language else
            f"- {provenance['eligible']} documents were solved by every model with an "
            f"identical prompt (`prompt_hash`); {provenance['excluded_prompt_mismatch']} "
            "were excluded because the prompt differed.")
        lines += [
            heading, "",
            f"Reference {'language' if per_language else 'model'}: "
            f"**{provenance['reference']}**  ",
            f"Compared against: {', '.join(m for m in provenance['models'] if m != provenance['reference'])}  ",
            "",
            "How these were drawn:", "",
            eligibility,
            f"- Bucket sizes: " + ", ".join(
                f"`{name}` {provenance['pool_sizes'][name]}" for name in CATEGORIES) +
            f"; `mixed` {provenance['mixed_not_sampled']} (not sampled).",
            f"- Up to {provenance['per_category']} taken per bucket at random, seed "
            f"{provenance['seed']}: " + ", ".join(
                f"`{name}` {provenance['taken'][name]}" for name in CATEGORIES) + ".",
            "",
        ]
        per_language = provenance.get("axis") == "language"
        column = "language" if per_language else "model"
        subset = frame[frame.group == provenance["group"]]
        for category in CATEGORIES:
            picked = subset[subset.category == category]
            if not len(picked):
                continue
            lines += [f"### {category.replace('_', ' ')}", ""]
            for document, rows_for_doc in picked.groupby("document", sort=False):
                rows_for_doc = rows_for_doc.sort_values("is_reference", ascending=False)
                reference_row = rows_for_doc[rows_for_doc.is_reference].iloc[0]
                # Paired on a field, the key *is* the document's name and says everything.
                # Paired on position it is `_stem#5`, or bare `#5` for a run with a single
                # task, so the reference's own task fills in what the key cannot say.
                subject = str(document).lstrip("_")
                if subject.startswith("#"):
                    subject = f"{reference_row.run_task} {subject}"
                lines += [f"**{subject}** - gold: `{reference_row.target}`", ""]
                if per_language:
                    # One prompt per language, not one for the reference. The document is only "the same question" by the alignment argument, and the way to see whether that argument holds - or where a translation went wrong - is to read the languages next to each other.
                    for row in rows_for_doc.itertuples():
                        prompt, choices = _prompt_and_choices(
                            row.run_path, row.run_task, int(row.doc_id), tail=300)
                        lines += [f"*{row.series}* (`{row.run_task}`)", "",
                                  "```", prompt, "```", ""]
                        if choices:
                            lines.append("Choices: " + ", ".join(
                                f"`{c.strip()}`" for c in choices))
                            lines.append("")
                else:
                    prompt, choices = _prompt_and_choices(
                        reference_row.run_path, reference_row.task_name,
                        int(reference_row.doc_id))
                    lines += ["```", prompt, "```", ""]
                    if choices:
                        lines.append("Choices: " + ", ".join(f"`{c.strip()}`" for c in choices))
                        lines.append("")
                lines += [f"| {column} | correct | predicted | settles at depth |",
                          "| --- | --- | --- | --- |"]
                for row in rows_for_doc.itertuples():
                    mark = "**<-**" if row.is_reference else ""
                    depth = "-" if row.settle_depth is None or _isna(row.settle_depth) \
                        else f"{row.settle_depth:.2f}"
                    predicted = str(row.predicted or "")[:60].replace("\n", " ")
                    lines.append(
                        f"| {row.series} {mark} | {'yes' if row.is_correct else 'no'} "
                        f"| `{predicted}` | {depth} |")
                lines.append("")
    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


def write_examples(
    partition: Partition,
    output_dir: str,
    reference: str | None = None,
    per_category: int = 3,
    assume_aligned: bool = False,
    pair_on: str | None = None,
    pair_strict: bool = False,
    pair_mapping: "PairingMapping | str | None" = None,
):
    """Write the qualitative half of a report: which documents to actually read.

    Takes an already-computed `Partition` rather than re-discovering runs, because deciding which runs may be compared is the same question for the curves and for the examples - and the answer must not be allowed to differ between them.
    A group already paired by `write_pairing` keeps that pairing.

    Returns:
        The examples DataFrame, empty when nothing could be sampled.
    """
    import pandas as pd

    all_rows, provenance_list = [], []
    for group in partition.groups:
        if group.axis == "language" and group.pairing is None:
            # Before anything is compared, and printed: how two documents were decided to be the same document is the first thing the rest of this section depends on.
            pairing = attach_pairing(group, pair_on, strict=pair_strict, mapping=pair_mapping)
            print(f"[Group {group.label}] {pairing.note}")
        chosen_reference = reference_run(group, reference)
        if reference is None and len(group.runs) > 1:
            noun = "language" if group.axis == "language" else "model"
            print(f"[Group {group.label}] examples reference {noun}: "
                  f"{group.series_of(chosen_reference)} (highest score; "
                  "pass --reference to choose another)")
        rows, provenance = select_examples(
            group, chosen_reference, per_category, assume_aligned=assume_aligned)
        if provenance["excluded_prompt_mismatch"]:
            print(f"[Group {group.label}] warning: "
                  f"{provenance['excluded_prompt_mismatch']} documents excluded from "
                  "examples - the models saw different prompts for the same doc_id")
        if group.axis == "language":
            # Printed whether it passed or failed. Every per-document row under `--multilingual` rests on this one measurement, and evidence that is only visible when it refuses is evidence the reader has to go looking for - by which point the figure has already been read.
            headline = ("no per-document examples" if not provenance["eligible"]
                        else f"{provenance['eligible']} documents compared one to one")
            print(f"[Group {group.label}] {headline}: {provenance['eligibility']}")
        all_rows += rows
        provenance_list.append(provenance)

    frame = pd.DataFrame(all_rows)
    if len(frame):
        frame.to_parquet(os.path.join(output_dir, "examples.parquet"), index=False)
        render_examples(all_rows, provenance_list,
                        os.path.join(output_dir, "examples.md"))
    return frame


# --------------------------------------------------------------------------
# The pairing table: what was compared with what, and what was not
# --------------------------------------------------------------------------

PAIRING_STATEMENT = (
    "A pairing records which documents were compared and on what ground. An identity "
    "field or an explicit mapping is taken as given; it is not verified to be the same "
    "content. Gold alignment measures only whether the answer keys agree. Neither says "
    "that the tasks' metrics are comparable.")


def pairing_table(group: Group, alignment: Alignment | None = None) -> list[dict[str, Any]]:
    """One row per graded document of every run in a multilingual group.

    `pairing_status` and `answer_status` are separate on purpose: a document can be paired
    and still have an answer key that disagrees, or cannot be checked, across languages.
    A document that was not paired says why in `exclusion_reason` - `unmapped`,
    `duplicate_mapping_rows`, `many_to_one`, `counterpart_many_to_one`,
    `no_identity_field`, `strict_no_identity` or `not_in_every_language`.
    """
    pairing = group.pairing
    source = pairing.source if pairing else "position"
    documents = {run.path: sorted(_documents(run)) for run in group.runs}
    keys = {run.path: {group.doc_key(run, *doc) for doc in documents[run.path]}
            for run in group.runs}
    shared = set.intersection(*keys.values()) if keys else set()
    rows = []
    for run in group.runs:
        dataset = group.dataset_of(run)
        excluded = pairing.excluded.get(run.path, {}) if pairing else {}
        canonical = pairing.canonical.get(run.path, {}) if pairing else {}
        evidence = alignment.evidence.get(run.path, {}) if alignment else {}
        for document in documents[run.path]:
            key = group.doc_key(run, *document)
            paired = key in shared
            reason = None if paired else (
                excluded.get(document)
                or ("strict_no_identity" if source == "none" else "not_in_every_language"))
            gold = evidence.get(key)
            kind, _, value = gold.partition(":") if gold else (None, None, None)
            answer = None
            if paired:
                answer = (alignment.status.get(key, "no_gold_evidence") if alignment
                          else None)
            rows.append({
                "group": group.label,
                "model": run.model_id,
                "num_fewshot": run.num_fewshot,
                "series": group.series_of(run),
                "language": dataset.label if dataset else None,
                "run_path": run.path,
                "task_name": document[0],
                "doc_id": document[1],
                "pairing_source": source,
                "pairing_field": pairing.field if pairing else None,
                "mapping_sha256": (pairing.mapping or {}).get("sha256") if pairing else None,
                "canonical_doc_id": (key if source == "position" else canonical.get(document)),
                "pairing_status": "paired" if paired else "excluded",
                "exclusion_reason": reason,
                "gold_evidence": kind,
                "gold_value": value,
                "answer_status": answer,
            })
    return rows


def pairing_provenance(group: Group, alignment: Alignment | None,
                       rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The group-level record beside the table: source, mapping scope, coverage, alignment."""
    pairing = group.pairing
    coverage: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = coverage.setdefault(row["series"], {
            "run_path": row["run_path"], "task": None, "documents": 0, "paired": 0,
            "excluded": {}})
        dataset = next((run for run in group.runs if run.path == row["run_path"]), None)
        entry["task"] = "+".join(dataset.tasks) if dataset else None
        entry["documents"] += 1
        if row["pairing_status"] == "paired":
            entry["paired"] += 1
        else:
            reason = row["exclusion_reason"]
            entry["excluded"][reason] = entry["excluded"].get(reason, 0) + 1
    counts = Counter(alignment.status.values()) if alignment else Counter()
    return {
        "group": group.label,
        "model": group.held_constant,
        "num_fewshot": group.runs[0].num_fewshot if group.runs else None,
        "datasets": [f"{d.label}={d.task}" for d in group.datasets],
        "source": pairing.source if pairing else "position",
        "field": pairing.field if pairing else None,
        "strict": pairing.strict if pairing else False,
        "note": pairing.note if pairing else "",
        "mapping": pairing.mapping if pairing else None,
        "paired_documents": pairing.matched if pairing and pairing.source != "position"
                            else len({row["canonical_doc_id"] for row in rows
                                      if row["pairing_status"] == "paired"}),
        "coverage": coverage,
        "alignment": None if alignment is None else {
            "shared": alignment.shared,
            "agree": counts.get("agree", 0),
            "disagree": counts.get("disagree", 0),
            "unverifiable": counts.get("unverifiable", 0),
            "rate": alignment.rate,
            "aligned": alignment.aligned,
            "source": alignment.source,
            "threshold": ALIGNMENT_THRESHOLD,
            "evidence_kinds": dict(sorted(Counter(
                value.partition(":")[0] for positions in alignment.evidence.values()
                for value in positions.values()).items())),
            "note": alignment.note,
        },
        "statement": PAIRING_STATEMENT,
    }


def write_pairing(
    partition: Partition,
    output_dir: str,
    pair_on: str | None = None,
    pair_strict: bool = False,
    pair_mapping: "PairingMapping | str | None" = None,
    assume_aligned: bool = False,
):
    """Pair every multilingual group, then write `pairing.parquet` and `pairing.json`.

    Written whether or not examples are drawn: which documents were treated as the same
    document is an output in its own right, and the curves never depended on it.

    Returns:
        The pairing DataFrame, empty for a report with no multilingual group.
    """
    import pandas as pd

    rows, groups = [], []
    for group in partition.groups:
        if group.axis != "language":
            continue
        pairing = attach_pairing(group, pair_on, strict=pair_strict, mapping=pair_mapping)
        print(f"[Group {group.label}] {pairing.note}")
        alignment = gold_alignment(group, assume_aligned)
        group_rows = pairing_table(group, alignment)
        rows += group_rows
        groups.append(pairing_provenance(group, alignment, group_rows))
    frame = pd.DataFrame(rows)
    if groups:
        if len(frame):
            frame.to_parquet(os.path.join(output_dir, "pairing.parquet"), index=False)
        with open(os.path.join(output_dir, "pairing.json"), "w", encoding="utf-8") as handle:
            json.dump({"pairing_schema": 1, "groups": groups}, handle, indent=2,
                      ensure_ascii=False)
            handle.write("\n")
    return frame


__all__ = [
    "Run",
    "Group",
    "Partition",
    "Alignment",
    "discover_runs",
    "Dataset",
    "parse_datasets",
    "screen_runs",
    "partition_runs",
    "gold_alignment",
    "Pairing",
    "attach_pairing",
    "find_pairing_field",
    "PairingMapping",
    "load_pairing_mapping",
    "pairing_table",
    "pairing_provenance",
    "write_pairing",
    "format_partition",
    "relative_depth",
    "summarize_run",
    "build_comparison",
    "render_pdf",
    "run_report",
    "load_signals",
    "CATEGORIES",
    "reference_run",
    "comparable_documents",
    "categorize",
    "select_examples",
    "render_examples",
    "write_examples",
]
