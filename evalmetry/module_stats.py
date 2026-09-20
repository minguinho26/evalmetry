"""Sample-level module statistics and reproducible dataset aggregation.

Execution traces answer where a call stopped. This file instead stores one row per
physical batch row, module invocation and tensor, then pools those observations by
document before comparing documents. SQLite keeps collection and aggregation bounded
in host memory; Parquet exports are for downstream analysis. No activations are kept.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 5
CHUNK_ELEMENTS = 65536
GROUP = ("task_name", "module", "io", "tensor_path", "dtype", "stage", "layout")
# One axis index of one tensor is its own population, so it joins the group key
# rather than being pooled into it. `axis_role` is determined by the rest of the key
# and travels with it so the meaning of an index is readable without a join.
AXIS_GROUP = GROUP + ("axis", "axis_role", "axis_index")

# Boundaries whose tensors are [batch * width, features] in the model's own token
# order, keyed by the *parent* module's class and the attribute holding the child.
# The child's own class is not enough to identify one: a transformers 4.57 router is
# a bare `nn.Linear`, and 5.16 fuses every expert into one module.
#
# Read from the installed implementations, not inferred. `Qwen3MoeSparseMoeBlock`
# and `MixtralSparseMoeBlock` both flatten with `hidden_states.view(-1, hidden_dim)`
# before calling `gate` and `experts`, and reshape the experts' result back to
# [batch, sequence, hidden]. Row `r` of either boundary is therefore document row
# `r // width` at position `r % width`, and the expert-major gather happens strictly
# inside the experts module. In 4.57 the experts are an `nn.ModuleList`, so each
# expert's parent is that list and none of them is registered here: a single expert
# sees only the tokens routed to it, in routing order.
FLATTENED_BOUNDARIES = {
    "Qwen3MoeSparseMoeBlock.gate", "Qwen3MoeSparseMoeBlock.experts",
    "MixtralSparseMoeBlock.gate", "MixtralSparseMoeBlock.experts",
}

# Registered routers whose own output carries the dispatch its parent block used, and
# what each of their outputs means. Read from the installed implementations, not
# inferred: `Qwen3MoeTopKRouter.forward` and `MixtralTopKRouter.forward` both return
# `(router_logits, router_scores, router_indices)`, and both blocks hand exactly those
# scores and indices to `experts`. So the selection here is observed, not reconstructed.
#
# The logits also appear alone: in transformers 4.57 the same `gate` attribute is a bare
# `nn.Linear` returning one tensor, and the softmax and top-k are taken in the parent
# block, where no registered boundary can see them. Such a run records the logits per
# expert and says routing was unavailable, rather than recomputing a selection whose
# dtype, normalisation and k are the implementation's and not observable at the boundary.
_TOP_K_ROUTER = {
    "logits": ("output.0", "output"),
    "weights": "output.1",
    "experts": "output.2",
    "selection": "observed",
}
ROUTER_BOUNDARIES = {
    "Qwen3MoeSparseMoeBlock.gate": _TOP_K_ROUTER,
    "MixtralSparseMoeBlock.gate": _TOP_K_ROUTER,
}


def _transformers_version() -> str | None:
    """Identify the implementation the registry was matched against, if it is there."""
    try:
        import transformers
    except ImportError:
        return None
    return getattr(transformers, "__version__", None)


def merge_moments(a: dict, b: dict) -> dict:
    """Combine finite-value moments using the parallel variance formula.

    Counts include non-finite values, while mean/M2/min/max describe finite values
    only. Population variance is M2 / finite_count, never an average of variances.
    """
    out = {k: a.get(k, 0) + b.get(k, 0)
           for k in ("count", "finite_count", "nan", "posinf", "neginf")}
    n, m = a.get("finite_count", 0), b.get("finite_count", 0)
    if not n or not m:
        source = a if n else b
        out.update({k: source.get(k) for k in ("mean", "m2", "min", "max")})
    else:
        delta = b["mean"] - a["mean"]
        out.update(mean=a["mean"] + delta * (m / (n + m)),
                   m2=a["m2"] + b["m2"] + delta * delta * (n * m / (n + m)),
                   min=min(a["min"], b["min"]), max=max(a["max"], b["max"]))
    return out


def tensor_moments(tensor: Any, chunk_elements: int | None = None) -> dict:
    """Reduce a floating tensor in bounded float64 chunks without keeping its graph.

    Float64 prevents the tracer from overflowing when squaring large float32/bf16
    activations. The temporary reduction buffers are bounded; flattening a strided
    input can still require a contiguous copy of that sample, so the chunk size alone
    does not bound the memory a reduction takes.
    """
    import torch

    total: dict = {}
    for chunk in tensor.detach().reshape(-1).split(chunk_elements or CHUNK_ELEMENTS):
        values = chunk.to(dtype=torch.float64)
        counts = torch.stack([torch.isnan(values).sum(), torch.isposinf(values).sum(),
                              torch.isneginf(values).sum()]).cpu().tolist()
        finite = values[torch.isfinite(values)]
        part = dict(count=chunk.numel(), finite_count=finite.numel(),
                    nan=counts[0], posinf=counts[1], neginf=counts[2])
        if finite.numel():
            mean = finite.mean()
            packed = torch.stack([mean, ((finite - mean) ** 2).sum(),
                                  finite.min(), finite.max()]).cpu().tolist()
            part.update(zip(("mean", "m2", "min", "max"), packed))
        total = merge_moments(total, part)
    return merge_moments({}, total)


def _output_tensor(value: Any, spec: str) -> Any:
    """The tensor a registry path names, or None when this output has no such member.

    `output` is the returned value itself, `output.N` its N-th member. Anything else at
    that position - a different container, a version that returns fewer values - is None
    rather than a guess at which member was meant.
    """
    import torch

    if spec == "output":
        return value if isinstance(value, torch.Tensor) else None
    index = int(spec.rpartition(".")[2])
    if isinstance(value, (tuple, list)) and index < len(value):
        member = value[index]
        return member if isinstance(member, torch.Tensor) else None
    return None


def _tensors(value: Any, path: str, depth: int = 0):
    """Walk tensor containers without truncating layers or retaining cache objects.

    KV caches are deliberately opaque: re-counting cached tokens on every decode
    step is a different population from newly computed module activations.
    """
    import torch

    if isinstance(value, torch.Tensor):
        yield path, value
    elif depth < 8 and isinstance(value, dict):
        for key, child in value.items():
            yield from _tensors(child, f"{path}.{key}", depth + 1)
    elif depth < 8 and isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _tensors(child, f"{path}.{index}", depth + 1)
    elif value is not None and not isinstance(value, (bool, int, float, str)):
        yield path, None


class ModuleStatistics:
    """Persist sample observations in a new session, never mixing resumed attempts.

    Supported activation layout is [batch, current_sequence, features]. Attention
    matrices use [batch, heads, query, key]. Other layouts are counted as excluded,
    so a shared tensor is not silently assigned to each document. The backend must
    supply true input lengths; scoring positions alone are not a padding mask.
    """

    def __init__(self, trace_path: str, config: dict, chunk_elements: int | None = None):
        base = Path(trace_path)
        # A reduction setting, not part of `config`: two sessions that differ only here
        # must hold the same numbers, and keeping it out of `config` lets them compare so.
        self.chunk_elements = int(chunk_elements or CHUNK_ELEMENTS)
        if self.chunk_elements < 1:
            raise ValueError("chunk_elements must be at least 1")
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "-" + uuid.uuid4().hex[:8]
        self.directory = base.parent / "module_stats" / base.stem / session
        self.directory.mkdir(parents=True)
        self.path = self.directory / "statistics.sqlite"
        self._configure_observations(config)
        self._create_database()
        self._write_session_metadata(config, base.stem, session)

        # 아래 상태는 forward마다 갱신되거나 모듈 등록 때 채워진다.
        # 저장 스키마·설정과 분리해 현재 관측 상태를 한곳에서 확인한다.
        self.contexts: list[Any] = []
        self.next_step = 0
        self.forward = 0
        self.stage = "unknown"
        self.step = 0
        self.lengths: list[int] | None = None
        self.width = 0
        self.batch = 0
        self.flattened: dict[str, str | None] = {}
        self.extremes: dict[str, int] = {}
        self.axes: dict[str, str | None] = {}
        self.routers: dict[str, dict | None] = {}
        self.routing: dict[str, dict | None] = {}
        self.closed = False

    def _configure_observations(self, config: dict) -> None:
        """추가 관측의 옵션과 모듈 selector를 해석한다. 실제 경로는 등록 때 확정한다."""
        # Extreme positions are opt-in twice over: a k, and the modules that record it.
        self.extremes_k = int(config.get("extremes") or 0)
        selector = config.get("extremes_modules")
        self.extremes_pattern = re.compile(selector) if self.extremes_k and selector else None
        # Per-axis statistics the same way: which axis to keep, and where to keep it.
        # One row per index is a different order of output from one row per tensor.
        self.axes_spec = config.get("axes") or None
        axis_selector = config.get("axes_modules")
        self.axes_pattern = (re.compile(axis_selector)
                             if self.axes_spec and axis_selector else None)
        # Routing needs a selector alone: the k, the expert ids and the weights are the
        # router's own output, so there is nothing left for a second flag to decide.
        routing_selector = config.get("routing")
        self.routing_pattern = re.compile(routing_selector) if routing_selector else None

    def _create_database(self) -> None:
        """새 세션의 DB와 원시 관측 테이블을 만든다. 집계 테이블은 종료 때 생성한다."""
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE modules (
                module TEXT PRIMARY KEY, module_type TEXT, parent_type TEXT,
                flattened_rule TEXT, extremes_k INTEGER, axis_stats TEXT,
                router_rule TEXT, routing TEXT, router_experts INTEGER,
                router_top_k INTEGER, registration_status TEXT,
                registration_error TEXT, call_count INTEGER NOT NULL DEFAULT 0);
            CREATE VIEW coverage AS SELECT m.*,
                (SELECT count(*) FROM observations o WHERE o.module=m.module) AS observation_count,
                coalesce((SELECT sum(e.count) FROM exclusions e WHERE e.module=m.module), 0) AS exclusion_count
                FROM modules m;
            CREATE TABLE forwards (forward INTEGER PRIMARY KEY, status TEXT NOT NULL);
            CREATE TABLE observations (
                forward INTEGER, call INTEGER, batch_row INTEGER,
                task_name TEXT, doc_id INTEGER, choices TEXT,
                module TEXT, io TEXT, tensor_path TEXT, dtype TEXT, stage TEXT,
                step INTEGER, layout TEXT, shape TEXT, source_shape TEXT,
                count INTEGER, finite_count INTEGER,
                nan INTEGER, posinf INTEGER, neginf INTEGER,
                mean REAL, m2 REAL, min REAL, max REAL);
            CREATE INDEX observations_module ON observations(module);
            CREATE TABLE extremes (
                forward INTEGER, call INTEGER, batch_row INTEGER,
                task_name TEXT, doc_id INTEGER, choices TEXT,
                module TEXT, io TEXT, tensor_path TEXT, dtype TEXT, stage TEXT,
                step INTEGER, layout TEXT, position INTEGER, feature INTEGER,
                value REAL, abs_rank INTEGER);
            CREATE INDEX extremes_module ON extremes(module);
            CREATE TABLE axis_stats (
                forward INTEGER, call INTEGER, batch_row INTEGER,
                task_name TEXT, doc_id INTEGER, choices TEXT,
                module TEXT, io TEXT, tensor_path TEXT, dtype TEXT, stage TEXT,
                step INTEGER, layout TEXT, axis TEXT, axis_role TEXT, axis_index INTEGER,
                count INTEGER, finite_count INTEGER,
                nan INTEGER, posinf INTEGER, neginf INTEGER,
                mean REAL, m2 REAL, min REAL, max REAL);
            CREATE INDEX axis_stats_module ON axis_stats(module);
            CREATE TABLE routing (
                forward INTEGER, call INTEGER, batch_row INTEGER,
                task_name TEXT, doc_id INTEGER, choices TEXT,
                module TEXT, stage TEXT, step INTEGER, layout TEXT, dtype TEXT,
                position INTEGER, rank INTEGER, expert INTEGER,
                weight REAL, weight_state TEXT, source TEXT, weight_scope TEXT);
            CREATE INDEX routing_module ON routing(module);
            CREATE TABLE exclusions (
                module TEXT, io TEXT, tensor_path TEXT, reason TEXT, count INTEGER,
                PRIMARY KEY (module, io, tensor_path, reason));
        """)

    def _write_session_metadata(self, config: dict, pass_name: str, session: str) -> None:
        """집계 기준과 좌표 의미를 DB에 기록해 모델 없이도 결과를 해석하게 한다.

        아래 문자열은 출력 데이터의 일부다. 설명을 정리할 때도 저장되는 값과
        키를 유지해야 기존 세션과 같은 기준으로 비교할 수 있다.
        """
        self._meta("schema_version", SCHEMA_VERSION)
        self._meta("config", config)
        self._meta("pass", pass_name)
        self._meta("session", session)
        self._meta("status", "running")
        self._meta("aggregation_status", "pending")
        self._meta("attention_mask_policy", "exclude exact attention_mask path components, including nested containers; no positional or value inference")
        self._meta("document_statistics", {
            "weighting": "equal weight per document with finite values",
            "sample_mean_std": "population std of document means: sqrt(M2/n)",
            "sample_mean_m2": "sum of squared deviations of finite document means",
            "sample_mean_sample_std": "sqrt(M2/(n-1)); null for n<2",
            "sample_mean_sem": "sample std/sqrt(n); assumes independent sampled documents; null for n<2",
            "n": "finite_documents; excludes documents containing only nonfinite values"})
        self._meta("coverage_units", {
            "call_count": "selected module pre-hook entries, all phases, including failed calls",
            "observation_count": "stored document/tensor rows, input and output, including incomplete forwards",
            "exclusion_count": "exclusion events; usually tensor visits, per-document for key-length or moment failures",
            "durability": "targets and registration transitions committed immediately; calls at entry; observations after each observer"})
        self._meta("flattened_layout", {
            "rule": "parent module class + attribute, matched before the first hook",
            "registered": sorted(FLATTENED_BOUNDARIES),
            "row_index": "batch_row * width + position; right padding dropped per document",
            "requires": "exactly two dimensions and first axis == batch * width",
            "refuses": "unregistered boundaries, size mismatch from dropped or padded "
                       "tokens, and any expert-major child such as 4.57 experts.N",
            "pooling": "router logit statistics reduce over the expert axis; they are "
                       "not per-expert observations",
            "transformers": _transformers_version()})
        self._meta("extremes", {
            "k": self.extremes_k,
            "modules": config.get("extremes_modules"),
            "unit": "one row per rank, per document, module call and tensor",
            "ranking": "largest absolute value first; non-finite elements are not "
                       "ranked and stay counted in the observation row",
            "ties": "equal absolute values are ordered by ascending flat index, and the "
                    "same index order decides which of them the k-th rank keeps",
            "position": "column of the current model input, right padding already "
                        "excluded, so it is the same coordinate before and after "
                        "removal; a decode step's own position is its generation step",
            "layouts": "bsh and flattened_bs only; attention matrices are measured but "
                       "never unfolded into query/key positions",
            "join": "forward, call, batch_row, module, io, tensor_path give the "
                    "observation row with finite_count and the non-finite counts"})
        self._meta("axis_statistics", {
            "axes": self.axes_spec,
            "modules": config.get("axes_modules"),
            "unit": "one row per axis index, per document, module call and tensor",
            "feature": "reduces the document's valid positions and keeps the tensor's "
                       "own channel index",
            "position": "reduces the feature axis and keeps the column of the current "
                        "model input, right padding already excluded",
            "count": "elements reduced into this index, which is the other axis's length; "
                     "mean/M2/min/max describe the finite ones and the rest are counted",
            "grid": "only the chosen axis is stored; the token x feature grid is never "
                    "expanded, which is the point of choosing an axis",
            "alignment": "the same position index in two documents is the same column of "
                         "the model input, not a claim that those tokens mean the same "
                         "thing; a decode step is column 0, so `step` separates the steps",
            "documents": "axis_dataset.documents counts the documents that reached this "
                         "index at all, so a position only long documents have is visibly "
                         "pooled over fewer of them",
            "axis_role": "input_position for every position row; for a feature row, "
                         "expert at a registered router's logits, router_rank at its "
                         "selected weights or indices, and channel otherwise",
            "layouts": "bsh and flattened_bs only; attention matrices are measured but "
                       "never unfolded into query/key positions",
            "join": "forward, call, batch_row, module, io, tensor_path give the pooled "
                    "observation row these reduce, with its shape and source_shape"})
        self._meta("routing", {
            "modules": config.get("routing"),
            "registered": sorted(ROUTER_BOUNDARIES),
            "source": "observed - the weights and expert ids the registered router "
                      "returned, which are the tensors its parent block passes to the "
                      "experts module unchanged in the installed implementation",
            "derived": "not stored: a top-k recomputed from logits would need the "
                       "implementation's softmax dtype, its normalisation and a k, none "
                       "of which is observable at the boundary. A router that returns "
                       "logits alone - transformers 4.57 takes the top-k in the parent "
                       "block - records routing_unavailable instead",
            "rank": "the column of the router's own top-k output; torch.topk returns "
                    "descending probability in both installed implementations",
            "weight_scope": "selected_top_k - the weight of one selected expert among the "
                            "k selected for that token. Whether they are normalised over "
                            "that k is the implementation's choice (Mixtral always; Qwen3 "
                            "when config.norm_topk_prob), so sum a token's ranks to see",
            "weight_state": "finite, nan, posinf or neginf; a non-finite weight is stored "
                            "as its state with a null value, never as a number",
            "coverage": "modules.routing says the module was selected, not that rows "
                        "exist; a selected router with none has its reason in exclusions, "
                        "and router_experts/router_top_k are filled from the first row",
            "expert_count": "coverage.router_experts, the width of the router's logits, so "
                            "an expert with no rows is distinguishable from an index that "
                            "does not exist. Selection frequency is a count over these "
                            "rows and is not stored separately",
            "position": "as in extremes: the column of the current model input, right "
                        "padding already excluded"})
        self._meta("population", "all valid current input positions; float tensors only")
        self._meta("reduction", {
            "chunk_elements": self.chunk_elements,
            "default": CHUNK_ELEMENTS,
            "moments": "flattened sample split into chunks of this many elements",
            "axis_statistics": "blocks of max(1, chunk_elements // reduced-axis length) "
                               "kept indices",
            "invariance": "a setting of the reduction, not of the data: stored values "
                          "must not depend on it",
            "memory": "bounds the float64 buffers only; flattening a strided sample can "
                      "still copy the whole sample"})
        self.db.commit()

    def _meta(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                        (key, json.dumps(value)))

    def resolve_modules(self, targets: Iterable[tuple[str, str, str | None]]) -> None:
        """Persist the complete selection, and its layouts, before the first hook.

        A flattened boundary is decided here, from the model structure, and never
        from a tensor's shape at observation time: an expert-major child can have the
        same first axis as a token-major one on the batch that happens to route that
        way.
        """
        rows = []
        for path, module_type, parent_type in targets:
            rule = f"{parent_type}.{path.rpartition('.')[2]}" if parent_type else None
            rule = rule if rule in FLATTENED_BOUNDARIES else None
            self.flattened[path] = rule
            k = self.extremes_k if (self.extremes_pattern is not None
                                    and self.extremes_pattern.search(path)) else 0
            self.extremes[path] = k
            axes = self.axes_spec if (self.axes_pattern is not None
                                      and self.axes_pattern.search(path)) else None
            self.axes[path] = axes
            # A router is one of the flattened boundaries, so its rows already map to
            # documents. Knowing it is a router is separate from being asked for its
            # selection: the registry also names what a feature index means there.
            router = ROUTER_BOUNDARIES.get(rule) if rule else None
            self.routers[path] = router
            selected = (router is not None and self.routing_pattern is not None
                        and bool(self.routing_pattern.search(path)))
            self.routing[path] = router if selected else None
            rows.append((path, module_type, parent_type, rule, k or None, axes,
                         rule if router else None, "selected" if selected else None))
        self.db.executemany("INSERT INTO modules(module, module_type, parent_type, "
                            "flattened_rule, extremes_k, axis_stats, router_rule, routing, "
                            "registration_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                            rows)
        self._meta("resolved_module_count", len(rows))
        self._meta("flattened_module_count", sum(row[3] is not None for row in rows))
        self._meta("extremes_module_count", sum(bool(row[4]) for row in rows))
        self._meta("axis_module_count", sum(bool(row[5]) for row in rows))
        routers = sum(bool(row[7]) for row in rows)
        self._meta("routing_module_count", routers)
        self.db.commit()
        if self.routing_pattern is not None and not routers:
            # Refuse the run rather than leave an empty table that reads like a model
            # which never routed. This is the earliest point the module paths exist.
            raise ValueError(
                "--module-stats-routing matched no registered router boundary among the "
                f"{len(rows)} selected modules. Routing selections are observable only at "
                f"{sorted(ROUTER_BOUNDARIES)}, matched by the parent class and the "
                "attribute holding the child; --debug-modules must select one of them")

    def registration(self, module: str, status: str, error: str | None = None) -> None:
        """Keep partial registration distinguishable from a never-attempted target."""
        self.db.execute("UPDATE modules SET registration_status=?, registration_error=? WHERE module=?",
                        (status, error, module))
        self.db.commit()

    def called(self, module: str) -> None:
        """Count one invocation before inspection, independently of its tensor count."""
        self.db.execute("UPDATE modules SET call_count=call_count+1 WHERE module=?", (module,))
        self.db.commit()

    def set_samples(self, contexts: Iterable[Any]) -> None:
        self.contexts = list(contexts)
        self.next_step = 0

    def begin(self, forward: int, args: tuple, kwargs: dict) -> None:
        """Read the real root-call shape, before any selected child module runs."""
        import torch

        self.forward = forward
        self.lengths = None
        self.width = self.batch = 0
        self.stage = "unknown"
        self.db.execute("INSERT INTO forwards VALUES (?, 'incomplete')", (forward,))
        ids = kwargs.get("input_ids")
        if ids is None:
            ids = kwargs.get("inputs_embeds")
        if ids is None and args:
            ids = args[0]
        if not isinstance(ids, torch.Tensor) or ids.ndim < 2 or not self.contexts:
            return
        self.batch, self.width = ids.shape[:2]
        kind = getattr(self.contexts[0], "task_kind", "generate")
        incremental = kind == "generate" and not hasattr(self.contexts[0], "positions")
        self.stage = ("prefill" if self.next_step == 0 else "decode") if incremental else (
            "teacher_forced" if kind == "generate" else "loglikelihood")
        self.step = self.next_step if incremental else 0
        self.next_step += int(incremental)
        if incremental:
            # The backend generates one unpadded document at a time. Recomputed
            # prefixes (use_cache=False) are excluded rather than counted again.
            if self.batch == 1 and (self.step == 0 or self.width == 1):
                self.lengths = [self.width]
        else:
            lengths = {}
            for ctx in self.contexts:
                length = getattr(ctx, "input_length", None)
                if length is not None:
                    lengths[ctx.batch_row] = int(length)
            if set(lengths) == set(range(self.batch)):
                proposed = [lengths[i] for i in range(self.batch)]
                if all(0 < n <= self.width for n in proposed):
                    self.lengths = proposed

    def finish_forward(self) -> None:
        self.db.execute("UPDATE forwards SET status='complete' WHERE forward=?", (self.forward,))
        self.db.commit()

    def exclude(self, module: str, io: str, path: str, reason: str) -> None:
        self.db.execute("""INSERT INTO exclusions VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(module, io, tensor_path, reason) DO UPDATE SET count=count+1""",
                        (module, io, path, reason))

    def _documents(self) -> list[tuple[int, str, int, str]]:
        """The documents of this forward, one entry per physical row they share.

        Choice aliases of one document collapse into a single entry so that a shared
        row is one observation, and the aliases are preserved beside it instead.
        """
        grouped: dict[tuple, set] = {}
        for ctx in self.contexts:
            key = (ctx.batch_row, ctx.task_name, ctx.doc_id)
            grouped.setdefault(key, set()).add(ctx.choice_idx)
        documents = []
        for (row, task, doc), choices in grouped.items():
            if not 0 <= row < self.batch:
                raise ValueError("sample batch row is outside the model input")
            documents.append((row, task, doc, json.dumps(sorted(choices))))
        return documents

    def _feature_role(self, module: str, path: str) -> str:
        """What a feature index means: an expert, a selection rank, or a channel."""
        rule = self.routers.get(module)
        if rule:
            if path in rule["logits"]:
                return "expert"
            if path in (rule["weights"], rule["experts"]):
                return "router_rank"
        return "channel"

    def observe(self, module: str, io: str, value: Any, call: int, phase: str,
                module_type: str = "") -> None:
        """Split confirmed batch/sequence layouts, exclude padding, then reduce.

        Choice aliases sharing the same physical row produce one observation per
        document. The choices column preserves the aliases without multiplying the
        population. Distinct choice inputs remain separate observations of a document.
        """
        for path, tensor in _tensors(value, io):
            layout, reason = self._classify_tensor(module, path, tensor, phase, module_type)
            if reason:
                self.exclude(module, io, path, reason)
                continue

            for row, task, doc, choices in self._documents():
                sample = self._document_tensor(tensor, layout, row)
                if sample is None:
                    self.exclude(module, io, path, "unsupported_key_length")
                    continue
                # 원시 관측·극값·축별 통계가 공유하는 식별 필드다. 전체 SQL 행을
                # 만든 뒤 앞부분을 잘라 쓰지 않고, 같은 키를 명시적으로 전달한다.
                observation_key = [self.forward, call, row, task, doc, choices,
                                   module, io, path, str(tensor.dtype), self.stage,
                                   self.step, layout]
                self._record_observation(sample, tensor.shape, observation_key, module, io, path, layout)

        # The selection is a property of the whole output, not of one of its tensors:
        # the weights and the expert ids only mean anything as a pair.
        rule = self.routing.get(module) if io == "output" else None
        if rule is not None:
            self._record_routing(module, value, call, rule, phase)
        self.db.commit()

    def _classify_tensor(self, module: str, path: str, tensor: Any, phase: str,
                         module_type: str) -> tuple[str | None, str | None]:
        """지원 layout 또는 제외 사유를 반환한다. 둘 중 하나만 값이 있다.

        판정 순서는 제외 통계의 의미를 결정한다. 여러 조건에 걸리는 tensor도
        기존과 같은 사유로 집계되도록 phase·mask·공유값 검사를 먼저 수행한다.
        """
        if phase != "model_forward":
            return None, "not_model_forward"
        if "attention_mask" in path.split("."):
            return None, "attention_mask"
        if "RotaryEmbedding" in module_type:
            return None, "shared_or_cached"
        if self.lengths is None:
            return None, "missing_sample_or_valid_length"
        if tensor is None:
            return None, "opaque_or_deep_container"
        if not tensor.is_floating_point():
            return None, "nonfloating"
        if any(name in path.split(".") for name in (
                "position_ids", "cache_position", "position_embeddings", "past_key_values",
                "cos", "sin")):
            return None, "shared_or_cached"
        if tensor.ndim == 3 and tuple(tensor.shape[:2]) == (self.batch, self.width):
            return "bsh", None
        if tensor.ndim == 2 and self.flattened.get(module):
            # 등록된 경계만 batch * width 순서를 보장한다. 크기가 달라졌다면
            # 토큰이 제거·추가·재배열됐을 수 있어 문서에 귀속시키지 않는다.
            if tensor.shape[0] == self.batch * self.width:
                return "flattened_bs", None
            return None, "flattened_size_mismatch"
        if (tensor.ndim == 4 and tensor.shape[0] == self.batch
                and tensor.shape[2] in (1, self.width)):
            # 4차원이라는 사실만으로 B,H,Q,K 의미를 부여하지 않는다.
            if "attentions" in path or "Attention" in module_type or "self_attn" in module:
                return "bhqk", None
        return None, "unsupported_layout"

    def _document_tensor(self, tensor: Any, layout: str, row: int) -> Any:
        """검증된 layout에서 한 문서의 padding을 제외한다. 부족한 key 축은 None.

        flattened 입력은 reshape하지 않고 해당 문서의 행만 자른다. 이렇게 해야
        non-contiguous tensor 전체를 복사하지 않고 원래 토큰 순서를 유지한다.
        """
        length = self.lengths[row]
        if layout == "bsh":
            return tensor[row, :length]
        if layout == "flattened_bs":
            start = row * self.width
            return tensor[start:start + length]
        # bhqk: decode의 key에는 KV cache가 포함된다. batch=1이라 key padding이 없다.
        key_length = tensor.shape[-1] if self.stage == "decode" else length
        if tensor.shape[-1] < key_length:
            return None
        return tensor[row, :, :min(length, tensor.shape[2]), :key_length]

    def _record_observation(self, sample: Any, source_shape: Any, observation_key: list,
                            module: str, io: str, path: str, layout: str) -> None:
        """한 문서의 moments와 선택된 추가 통계를 같은 식별 키로 기록한다.

        commit은 observe가 담당한다. 중간에 실패했을 때 남는 관측 범위와
        forward 완료 여부에 따른 집계 규칙을 기존과 동일하게 유지한다.
        """
        stats = tensor_moments(sample, self.chunk_elements)
        if stats["finite_count"] and not all(math.isfinite(stats[k]) for k in ("mean", "m2")):
            # float64도 넘칠 수 있다. NaN이 SQL NULL로 저장돼 정상 평균처럼
            # 취급되지 않도록 전체 관측을 제외한다.
            self.exclude(module, io, path, "moment_overflow")
            return
        values = observation_key + [json.dumps(list(sample.shape)), json.dumps(list(source_shape))]
        values += [stats[k] for k in ("count", "finite_count", "nan", "posinf", "neginf",
                                      "mean", "m2", "min", "max")]
        placeholders = ",".join("?" for _ in values)
        self.db.execute(f"INSERT INTO observations VALUES ({placeholders})", values)
        k = self.extremes.get(module, 0)
        if k and layout in ("bsh", "flattened_bs"):
            self._record_extremes(sample, min(k, stats["finite_count"]), observation_key)
        if self.axes.get(module) and layout in ("bsh", "flattened_bs"):
            if self._record_axis_stats(sample, self.axes[module], observation_key,
                                       self._feature_role(module, path)):
                self.exclude(module, io, path, "axis_moment_overflow")

    def _record_extremes(self, sample: Any, k: int, keys: list) -> None:
        """Rank one document's slice by absolute value, and keep where the k largest are.

        Only confirmed [position, feature] slices reach this, so a row's position is a
        token of this model input and its feature is a channel of that tensor. Non-finite
        elements are never ranked - an Inf would otherwise be every rank - and stay
        counted in the observation row this shares a key with. Equal magnitudes are
        ordered by ascending flat index, which also decides which of them the last rank
        keeps, so the same tensor always writes the same rows.
        """
        import torch

        if k <= 0:
            return
        magnitude = sample.abs()                        # a fresh contiguous tensor
        magnitude.masked_fill_(~torch.isfinite(sample), -1.0)
        flat = magnitude.reshape(-1)
        _, selected = torch.topk(flat, k)
        threshold = flat[selected[-1]]
        if int((flat == threshold).sum()) > int((flat[selected] == threshold).sum()):
            # More elements share the smallest kept magnitude than there is room for.
            above = selected[flat[selected] > threshold]
            tied = (flat == threshold).nonzero().flatten()
            selected = torch.cat([above, tied[:k - above.numel()]])
        selected = selected.sort().values
        selected = selected[torch.argsort(flat[selected], descending=True, stable=True)]
        features = sample.shape[1]
        positions = torch.div(selected, features, rounding_mode="floor")
        columns = selected % features
        # Advanced indexing, so a non-contiguous slice is gathered, not copied whole.
        signed = sample[positions, columns].double().tolist()
        rows = [keys + [int(position), int(column), value, rank]
                for rank, (position, column, value) in enumerate(
                    zip(positions.tolist(), columns.tolist(), signed))]
        self.db.executemany("INSERT INTO extremes VALUES (" + ",".join("?" * 17) + ")", rows)

    def _record_axis_stats(self, sample: Any, axes: str, keys: list,
                           feature_role: str) -> bool:
        """Reduce one document's slice along one axis, keeping the other axis's index.

        A feature row reduces this document's valid positions and keeps the tensor's own
        channel; a position row reduces the channels and keeps the column of the current
        model input. Only confirmed [position, feature] slices reach this, so both indices
        mean what they are called. Non-finite elements are counted per index and excluded
        from that index's mean/M2/min/max, exactly as the pooled observation does, so a
        channel of NaN is visible rather than averaged into one.

        Chunking is along the axis that is *kept*, never the one being reduced: each
        block's float64 copy and its comparison masks are bounded, and no partial moment
        has to be merged across chunks. Returns whether any index exceeded float64.
        """
        import torch

        positions, features = sample.shape
        overflow = False
        for axis in ("feature", "position"):
            if axes not in (axis, "both"):
                continue
            reduced = 0 if axis == "feature" else 1
            kept, other = ((features, positions) if axis == "feature"
                           else (positions, features))
            # A position index is a column of the model input whatever the tensor is;
            # only a feature index takes its meaning from the module.
            role = feature_role if axis == "feature" else "input_position"
            if not kept or not other:
                continue
            span = max(1, self.chunk_elements // other)
            rows = []
            for start in range(0, kept, span):
                block = (sample[:, start:start + span] if axis == "feature"
                         else sample[start:start + span])
                values = block.detach().to(dtype=torch.float64)
                mask = torch.isfinite(values)
                zero = torch.zeros((), dtype=torch.float64, device=values.device)
                finite = mask.sum(dim=reduced)
                mean = torch.where(mask, values, zero).sum(dim=reduced) / finite.clamp(min=1)
                # Masked after subtracting, so an Inf does not poison its neighbours in
                # the sum; an index whose every element is non-finite gets None below.
                deviation = torch.where(mask, values - mean.unsqueeze(reduced), zero)
                counters = torch.stack([finite,
                                        torch.isnan(values).sum(dim=reduced),
                                        torch.isposinf(values).sum(dim=reduced),
                                        torch.isneginf(values).sum(dim=reduced)]).cpu().tolist()
                moments = torch.stack([
                    mean, (deviation * deviation).sum(dim=reduced),
                    torch.where(mask, values, zero + math.inf).amin(dim=reduced),
                    torch.where(mask, values, zero - math.inf).amax(dim=reduced),
                ]).cpu().tolist()
                for offset in range(block.shape[1 - reduced]):
                    finite_count = counters[0][offset]
                    if finite_count:
                        found = [moments[j][offset] for j in range(4)]
                        if not all(math.isfinite(value) for value in found[:2]):
                            # As in the pooled row: never store a NaN mean as SQL NULL
                            # and let it read later as a missing measurement.
                            overflow = True
                            continue
                    else:
                        found = [None, None, None, None]
                    rows.append(keys + [axis, role, start + offset, other]
                                + [counters[j][offset] for j in range(4)] + found)
            if rows:
                self.db.executemany(
                    "INSERT INTO axis_stats VALUES (" + ",".join("?" * 25) + ")", rows)
        return overflow

    def _record_routing(self, module: str, value: Any, call: int, rule: dict,
                        phase: str) -> None:
        """Store the expert selection the router returned, per token and per rank.

        This is the dispatch itself, not a reconstruction: in the installed
        implementations the parent block passes exactly these two tensors to its experts
        module. Rows map to documents by the flattened rule, so a padded position can
        never appear. A router that returns logits alone leaves `routing_unavailable`
        rather than a top-k recomputed here, because the softmax dtype, the
        normalisation and k all belong to the parent block that took it.
        """
        import torch

        weights = _output_tensor(value, rule["weights"])
        experts = _output_tensor(value, rule["experts"])
        if phase != "model_forward":
            reason = "not_model_forward"
        elif self.lengths is None:
            reason = "missing_sample_or_valid_length"
        elif weights is None or experts is None:
            reason = "routing_unavailable"
        elif not weights.is_floating_point() or experts.is_floating_point():
            reason = "routing_unexpected_dtype"
        elif (weights.ndim != 2 or tuple(weights.shape) != tuple(experts.shape)
              or weights.shape[0] != self.batch * self.width):
            # The same refusal as the moments: a first axis that is not batch * width
            # means the rows are no longer the caller's tokens, in the caller's order.
            reason = "routing_size_mismatch"
        else:
            reason = None
        if reason:
            self.exclude(module, "output", rule["weights"], reason)
            return

        logits = next((tensor for tensor in (_output_tensor(value, spec)
                                             for spec in rule["logits"])
                       if tensor is not None), None)
        # The logits' width is the only place the expert count is observable, and it is
        # what tells a never-selected expert apart from an index that does not exist.
        self.db.execute("UPDATE modules SET router_experts=coalesce(?, router_experts), "
                        "router_top_k=? WHERE module=?",
                        (None if logits is None else int(logits.shape[-1]),
                         int(weights.shape[1]), module))
        dtype = str(weights.dtype)
        rows = []
        for row, task, doc, choices in self._documents():
            start = row * self.width
            length = self.lengths[row]
            chosen = weights[start:start + length].detach().to(
                dtype=torch.float64).cpu().tolist()
            ids = experts[start:start + length].detach().cpu().tolist()
            for position, (token_weights, token_experts) in enumerate(zip(chosen, ids)):
                for rank, (weight, expert) in enumerate(zip(token_weights, token_experts)):
                    state = ("finite" if math.isfinite(weight) else
                             "nan" if math.isnan(weight) else
                             "posinf" if weight > 0 else "neginf")
                    rows.append([self.forward, call, row, task, doc, choices, module,
                                 self.stage, self.step, "flattened_bs", dtype,
                                 position, rank, int(expert),
                                 weight if state == "finite" else None, state,
                                 rule["selection"], "selected_top_k"])
        if rows:
            self.db.executemany(
                "INSERT INTO routing VALUES (" + ",".join("?" * 18) + ")", rows)

    def close(self, success: bool) -> None:
        """Keep partial observations, but aggregate only completed forwards.

        A failure to aggregate is written into this session's metadata before being
        raised, so the session says why it has no tables instead of looking unfinished.
        The caller decides what to do with it: `debug.py` reports it and carries on,
        because an evaluation that has already been scored must not be lost to a failure
        in a side artifact.
        """
        if self.closed:
            return
        self.closed = True
        try:
            self._meta("status", "aggregating" if success else "failed")
            self._meta("aggregation_status", "running")
            self.db.commit()
            try:
                aggregate(self.db)
                export_tables(self.db, self.directory)
            except BaseException as error:
                self._meta("status", "failed")
                self._meta("aggregation_status", "failed")
                self._meta("aggregation_error", f"{type(error).__name__}: {error}"[:2000])
                self.db.commit()
                raise
            self._meta("status", "complete" if success else "failed")
            self._meta("aggregation_status", "complete")
            self.db.commit()
        finally:
            self.db.close()


def _pool_calls(db: sqlite3.Connection, source: str, target: str, group: tuple) -> None:
    """Pool every call of one document into one row per group key.

    Only completed forwards contribute: a forward that died mid-way stays readable in
    the raw table and never becomes part of a summary.
    """
    rows = db.execute(f"SELECT o.* FROM {source} o JOIN forwards f USING (forward) "
                      "WHERE f.status='complete' ORDER BY " + ", ".join(group) + ", doc_id")
    key = lambda row: tuple(row[k] for k in group) + (row["doc_id"],)
    for identity, records in itertools.groupby(rows, key):
        pooled: dict = {}
        count = 0
        for record in records:
            pooled = merge_moments(pooled, dict(record))
            count += 1
        std = math.sqrt(pooled["m2"] / pooled["finite_count"]) if pooled["finite_count"] else None
        values = list(identity) + [count] + [pooled[k] for k in (
            "count", "finite_count", "nan", "posinf", "neginf", "mean", "m2", "min", "max")] + [std]
        db.execute(f"INSERT INTO {target} VALUES (" + ",".join("?" * len(values)) + ")", values)


def _pool_documents(db: sqlite3.Connection, source: str, target: str, group: tuple) -> None:
    """Combine documents two ways: equal element weight, and equal document weight.

    `documents` counts the documents present in this group at all. For a position axis
    that is the per-position valid document count, because a position only the longer
    documents reach is pooled over exactly those.
    """
    rows = db.execute(f"SELECT * FROM {source} ORDER BY " + ", ".join(group))
    for identity, records in itertools.groupby(rows, lambda row: tuple(row[k] for k in group)):
        pooled, means = {}, {}
        documents = observations = 0
        for record in records:
            record = dict(record)
            documents += 1
            observations += record["observations"]
            pooled = merge_moments(pooled, record)
            if record["finite_count"]:
                means = merge_moments(means, dict(count=1, finite_count=1,
                    mean=record["mean"], m2=0.0, min=record["mean"], max=record["mean"]))
        means = merge_moments({}, means)
        std = math.sqrt(pooled["m2"] / pooled["finite_count"]) if pooled["finite_count"] else None
        mean_std = math.sqrt(means["m2"] / means["finite_count"]) if means["finite_count"] else None
        values = list(identity) + [documents, means["finite_count"], observations]
        values += [pooled[k] for k in ("count", "finite_count", "nan", "posinf", "neginf",
                                      "mean", "m2", "min", "max")]
        values += [std, means["mean"], mean_std, means["min"], means["max"]]
        n = means["finite_count"]
        sample_std = math.sqrt(means["m2"] / (n - 1)) if n >= 2 else None
        sem = sample_std / math.sqrt(n) if n >= 2 else None
        values += [means["m2"], sample_std, sem]
        db.execute(f"INSERT INTO {target} VALUES (" + ",".join("?" * len(values)) + ")", values)


def aggregate(db: sqlite3.Connection) -> None:
    """Pool calls per document, then compute pooled and equal-document summaries.

    SQLite sorts on disk. Python holds only one group's moments at a time, so the
    number of documents does not determine aggregation RAM. Failed/incomplete
    forwards remain inspectable in observations and never enter these summaries.

    The per-axis tables are the same two stages over the same moments, with the axis and
    its index added to the group key, so a channel or a position is summarised exactly
    the way the whole tensor is - and never by averaging the documents' variances.
    """
    db.row_factory = sqlite3.Row
    db.executescript("""
        DROP TABLE IF EXISTS samples;
        DROP TABLE IF EXISTS dataset;
        DROP TABLE IF EXISTS axis_samples;
        DROP TABLE IF EXISTS axis_dataset;
        CREATE TABLE samples (
            task_name TEXT, module TEXT, io TEXT, tensor_path TEXT, dtype TEXT, stage TEXT,
            layout TEXT, doc_id INTEGER, observations INTEGER, count INTEGER, finite_count INTEGER,
            nan INTEGER, posinf INTEGER, neginf INTEGER,
            mean REAL, m2 REAL, min REAL, max REAL, std REAL);
        CREATE TABLE dataset (
            task_name TEXT, module TEXT, io TEXT, tensor_path TEXT, dtype TEXT, stage TEXT,
            layout TEXT, documents INTEGER, finite_documents INTEGER, observations INTEGER,
            count INTEGER, finite_count INTEGER, nan INTEGER, posinf INTEGER, neginf INTEGER,
            pooled_mean REAL, pooled_m2 REAL, min REAL, max REAL, pooled_std REAL,
            sample_mean REAL, sample_mean_std REAL, sample_mean_min REAL, sample_mean_max REAL,
            sample_mean_m2 REAL, sample_mean_sample_std REAL, sample_mean_sem REAL);
        CREATE TABLE axis_samples (
            task_name TEXT, module TEXT, io TEXT, tensor_path TEXT, dtype TEXT, stage TEXT,
            layout TEXT, axis TEXT, axis_role TEXT, axis_index INTEGER,
            doc_id INTEGER, observations INTEGER, count INTEGER, finite_count INTEGER,
            nan INTEGER, posinf INTEGER, neginf INTEGER,
            mean REAL, m2 REAL, min REAL, max REAL, std REAL);
        CREATE TABLE axis_dataset (
            task_name TEXT, module TEXT, io TEXT, tensor_path TEXT, dtype TEXT, stage TEXT,
            layout TEXT, axis TEXT, axis_role TEXT, axis_index INTEGER,
            documents INTEGER, finite_documents INTEGER, observations INTEGER,
            count INTEGER, finite_count INTEGER, nan INTEGER, posinf INTEGER, neginf INTEGER,
            pooled_mean REAL, pooled_m2 REAL, min REAL, max REAL, pooled_std REAL,
            sample_mean REAL, sample_mean_std REAL, sample_mean_min REAL, sample_mean_max REAL,
            sample_mean_m2 REAL, sample_mean_sample_std REAL, sample_mean_sem REAL);
    """)
    _pool_calls(db, "observations", "samples", GROUP)
    _pool_documents(db, "samples", "dataset", GROUP)
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='axis_stats'").fetchone():
        _pool_calls(db, "axis_stats", "axis_samples", AXIS_GROUP)
        _pool_documents(db, "axis_samples", "axis_dataset", AXIS_GROUP)
    db.commit()


def export_tables(db: sqlite3.Connection, directory: Path) -> None:
    """Export analysis tables in batches, including an explicit schema for NULLs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for table in ("samples", "dataset", "axis_samples", "axis_dataset", "coverage"):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            continue
        columns = list(db.execute(f"PRAGMA table_info({table})"))
        schema = pa.schema([(c[1], {"TEXT": pa.string(), "INTEGER": pa.int64(),
                                   "REAL": pa.float64(), "": pa.int64()}[c[2]]) for c in columns])
        cursor = db.execute(f"SELECT * FROM {table}")
        temporary = directory / f".{table}.parquet.tmp"
        with pq.ParquetWriter(temporary, schema) as writer:
            while rows := cursor.fetchmany(2048):
                writer.write_table(pa.Table.from_pylist([dict(r) for r in rows], schema=schema))
        os.replace(temporary, directory / f"{table}.parquet")


def resolve_database(path: str, pass_name: str = "trace") -> Path:
    """A direct database/session path, or the latest session of the selected pass."""
    given = Path(path)
    if given.is_file():
        return given
    if (given / "statistics.sqlite").is_file():
        return given / "statistics.sqlite"
    root = given / "debug" / "module_stats"
    found = sorted((root / pass_name).glob("*/statistics.sqlite"))
    if not found:
        # "nothing was recorded" and "recorded, but not this pass" need different advice.
        # A run with --module-stats but no collection pass is not misconfigured, so telling
        # its owner to re-run with --module-stats sends them to change a flag already set.
        recorded = sorted(directory.name for directory in root.glob("*")
                          if any(directory.glob("*/statistics.sqlite")))
        if recorded:
            raise FileNotFoundError(
                f"no {pass_name} module statistics under {path}; this run recorded: "
                f"{', '.join(recorded)}. The collection pass is written only when the run "
                f"also collects (--save-attention, --save-hidden or a collection hook).")
        raise FileNotFoundError(f"no module statistics under {path}; run with --module-stats")
    return found[-1]


def read_statistics(path: str, table: str = "dataset", pass_name: str = "trace"):
    """Load one analysis table as a DataFrame; never combine runs implicitly."""
    import pandas as pd

    if table not in {"dataset", "samples", "observations", "exclusions", "forwards",
                     "coverage", "metadata", "extremes", "axis_stats", "axis_samples",
                     "axis_dataset", "routing"}:
        raise ValueError(f"unknown module statistics table: {table}")
    database = resolve_database(path, pass_name)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name=?", (table,)).fetchone():
            if table in ("coverage", "extremes", "axis_stats", "axis_samples",
                         "axis_dataset", "routing"):
                raise RuntimeError(f"module {table} is unavailable in this historical schema")
            raise RuntimeError("module statistics have not been aggregated yet; the session is "
                               "still running or was interrupted. Completed observations remain "
                               "available in statistics.sqlite")
        return pd.read_sql_query(f"SELECT * FROM {table}", db)


def report_statistics(path: str, pass_name: str = "trace", doc: str | None = None,
                      module: str | None = None) -> str:
    """Render an explicitly scoped sample or dataset summary and its coverage."""
    import re

    database = resolve_database(path, pass_name)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        status = json.loads(db.execute("SELECT value FROM metadata WHERE key='status'").fetchone()[0])
        aggregation_status = json.loads(db.execute(
            "SELECT value FROM metadata WHERE key='aggregation_status'").fetchone()[0])
        failure = db.execute("SELECT value FROM metadata WHERE key='aggregation_error'").fetchone()
        if failure:
            aggregation_status += f" ({json.loads(failure[0])})"
        incomplete = db.execute("SELECT count(*) FROM forwards WHERE status!='complete'").fetchone()[0]
        has_coverage = db.execute("SELECT 1 FROM sqlite_master WHERE name='coverage'").fetchone()
        counted = {}
        for table in ("extremes", "axis_stats", "routing"):
            counted[table] = (db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                              if db.execute("SELECT 1 FROM sqlite_master WHERE name=?",
                                            (table,)).fetchone() else None)
        extremes = counted["extremes"]
    module_coverage = read_statistics(str(database), "coverage") if has_coverage else None
    if module_coverage is not None and module:
        module_coverage = module_coverage[module_coverage.module.astype(str).str.contains(module, regex=True)]
    coverage_text = (module_coverage.to_string(index=False) if module_coverage is not None
                     else "unavailable in this historical schema")
    try:
        frame = read_statistics(str(database), "samples" if doc is not None else "dataset")
    except RuntimeError:
        # Abrupt termination can leave durable coverage before aggregation exists.
        # Report that state without fabricating an empty completed dataset.
        if aggregation_status == "complete":
            raise
        return (f"statistics: {database}\nstatus: {status}; aggregation: {aggregation_status}; "
                f"incomplete forwards excluded: {incomplete}\n"
                "Aggregate tables unavailable; last committed module coverage "
                "(session-wide, all phases, includes incomplete forwards):\n" + coverage_text)
    if module:
        pattern = re.compile(module)
        frame = frame[frame.module.map(lambda name: bool(pattern.search(name))).astype(bool)]
    if doc is not None:
        task, _, number = doc.rpartition("#")
        frame = frame[frame.doc_id == int(number)]
        if task:
            frame = frame[frame.task_name == task]
    excluded = read_statistics(str(database), "exclusions")
    columns = ["task_name", "module", "io", "tensor_path", "stage"]
    columns += ["layout"] if "layout" in frame.columns else []
    columns += (["doc_id", "count", "finite_count", "mean", "std", "min", "max", "nan", "posinf", "neginf"]
                if doc is not None else ["documents", "finite_count", "sample_mean", "sample_mean_std",
                                        "pooled_mean", "pooled_std", "nan", "posinf", "neginf"])
    if doc is None:
        columns += [name for name in ("finite_documents", "sample_mean_m2",
                    "sample_mean_sample_std", "sample_mean_sem") if name in frame.columns]
    coverage = (excluded.groupby("reason")["count"].sum().to_string()
                if not excluded.empty else "none")
    return (f"statistics: {database}\nstatus: {status}; aggregation: {aggregation_status}; "
            f"incomplete forwards excluded: {incomplete}\n"
            "sample_mean: equal document weights; pooled_mean: equal finite-element weights.\n"
            "Population: observed valid positions, separated by task/module/tensor/stage.\n"
            "sample_mean_std uses n; sample_mean_sample_std uses n-1; SEM assumes independent documents.\n"
            + frame[columns].to_string(index=False) + "\nExclusions (events; units in metadata):\n" + coverage
            + "\nModule coverage (session-wide, all phases, includes incomplete forwards):\n" + coverage_text
            + (f"\nExtreme positions: {extremes} rows; read with "
               "read_statistics(path, 'extremes')." if extremes else "")
            + (f"\nPer-axis statistics: {counted['axis_stats']} rows; read with "
               "read_statistics(path, 'axis_dataset') or 'axis_samples'."
               if counted["axis_stats"] else "")
            + (f"\nRouting selections: {counted['routing']} rows; read with "
               "read_statistics(path, 'routing'); expert counts in coverage."
               if counted["routing"] else ""))
