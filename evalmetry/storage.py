"""Schema declaration and on-disk layout for a run directory.

This module is the single source of truth for what a run writes to disk.
The schema is declared *in code* rather than in a hand-maintained markdown file, so that documentation cannot drift away from the parquet files that are actually produced.

Everything a reader needs is therefore derived from `TABLES` below:

* `ShardWriter` builds the pyarrow schema from it,
* the column descriptions are embedded into every parquet file as key/value metadata (so a stray file is self-describing),
* `describe_schema()` prints it for humans,
* `report.py` reads `layer_domain` / `comparable_across_vocab` from it to decide what may be plotted on the same axis.

Directory layout produced by a run:

    results.json                scores + manifest + list of signal files
    samples.jsonl               sample results and scored inputs
    docs/part-0000.parquet       document/choice grading
    steps/part-0000.parquet      recorded positions and token IDs
    signals/part-0000.parquet    logit-lens signals
    similarity/part-0000.parquet layer-pair similarities
    attn_norm/part-0000.parquet  opt-in (--save-attention)
    attention/*.safetensors     opt-in (--save-attention)
    raw/*.safetensors           opt-in (--save-hidden)
    custom/<name>/              custom hook schemas, rows and optional tensors
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


# -------------------------------------------------------------------------- Fixed settings
#
# These are deliberately *not* CLI arguments: there is no reason to choose a different value, and exposing them would multiply the number of run configurations we have to reason about.  They are still written into the manifest, so that a run made today stays interpretable after a constant here is changed tomorrow.
# --------------------------------------------------------------------------

# : Version of the on-disk column layout.
# Bump when columns are added/removed : or retyped.
# `report.py` refuses to mix runs with different values.
SCHEMA_VERSION = "0.4"

#: The only supported lm-eval backend name (registered in `backend.py`).
BACKEND_NAME = "hf-traced"

# : Start a new parquet shard every this many documents.
# Shards are never : appended to, so a crash leaves every already-written shard intact.
SHARD_SIZE = 500

# : Seed for picking which documents get collectioned (--save-attention / : --save-hidden).
# Fixed, so the same run always selects the same documents.
SAMPLING_SEED = 1234

#: dtype used when dumping raw hidden states to safetensors.
RAW_DTYPE = "fp16"

# : report x-axis convention.
# Layers are plotted at relative depth l/L, never : at their absolute index, so models with different depths can be overlaid.
USE_RELATIVE_DEPTH = True

# : Attention sinks (the mass that piles up on the first token) are kept as-is; : the report does not filter them out.
KEEP_ATTENTION_SINK = True

FIXED_SETTINGS: dict[str, Any] = {
    "backend_name": BACKEND_NAME,
    "shard_size": SHARD_SIZE,
    "sampling_seed": SAMPLING_SEED,
    "raw_dtype": RAW_DTYPE,
    "relative_depth_axis": USE_RELATIVE_DEPTH,
    "keep_attention_sink": KEEP_ATTENTION_SINK,
}


# --------------------------------------------------------------------------
# Schema declaration
# --------------------------------------------------------------------------

# : Which layer index axis a table lives on. : : - "residual": index column is `layer`, range 0..L.
# `layer j` is the *input* :   of block j; `layer L` is the output of the last block. : - "block":    index column is `block`, range 0..L-1.
# `block j` is the :   attention *inside* block j. : - None:       the table has no layer axis at all (docs, steps).
LayerDomain = Literal["residual", "block"]


@dataclass(frozen=True)
class ColumnSpec:
    """One column of one table.

    Attributes:
        name: Column name as it appears in the parquet file.
        dtype: pyarrow type.
            Declared here so the writer never has to guess a type from the first batch of rows it happens to see.
        description: Human readable meaning; embedded into the parquet key/value metadata.
        unit: Physical unit or scale ("probability", "count", ...).
            "" when the column is an identifier or a flag.
        comparable_across_vocab: Whether values may be compared between models with different vocabularies.
            `cos` is a cosine and therefore comparable; `target_rank` is a raw rank out of |V| and therefore is not, while `target_percentile` divides it out and is.
    """

    name: str
    dtype: pa.DataType
    description: str
    unit: str = ""
    comparable_across_vocab: bool = True


@dataclass(frozen=True)
class TableSpec:
    """One parquet table, i.e. one subdirectory of the run directory.

    Signals have different axis shapes, so they get different tables.
    Forcing them into one wide table would produce rows that are mostly nulls.

    Attributes:
        name: Subdirectory name (`signals` -> `signals/part-0000.parquet`).
        key: Columns that together identify a row.
        columns: All columns, key columns first.
        layer_domain: See `LayerDomain`.
        description: What one row of this table means.
    """

    name: str
    key: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    layer_domain: LayerDomain | None
    description: str

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def arrow_schema(self) -> pa.Schema:
        """Build the pyarrow schema, with column docs attached as metadata.

        Example:
            >>> schema = TABLES["similarity"].arrow_schema()
            >>> schema.field("cos").type
            DataType(float)
            >>> json.loads(schema.metadata[b"eval_framework"])["layer_domain"]
            'residual'
        """
        fields = [pa.field(c.name, c.dtype) for c in self.columns]
        meta = {
            "schema_version": SCHEMA_VERSION,
            "table": self.name,
            "description": self.description,
            "key": list(self.key),
            "layer_domain": self.layer_domain,
            "columns": {
                c.name: {
                    "type": str(c.dtype),
                    "unit": c.unit,
                    "description": c.description,
                    "comparable_across_vocab": c.comparable_across_vocab,
                }
                for c in self.columns
            },
        }
        return pa.schema(
            fields, metadata={b"eval_framework": json.dumps(meta).encode("utf-8")}
        )


# The four axes shared by every signal table.
# `task_name` + `doc_id` is what lm-eval uses to identify a document; `choice_idx` separates the per-choice forwards of a multiple-choice document; `step` is the position inside the scored span (loglikelihood) or the decoding step (generate).
_AXIS_TASK = ColumnSpec("task_name", pa.string(), "lm-eval task name.")
_AXIS_DOC = ColumnSpec(
    "doc_id", pa.int64(), "Document id assigned by lm-eval. Unique only together with task_name."
)
_AXIS_CHOICE = ColumnSpec(
    "choice_idx",
    pa.int32(),
    "Which choice of a multiple-choice document this forward pass was for. 0 for generate tasks.",
)
_AXIS_STEP = ColumnSpec(
    "step",
    pa.int32(),
    "Position index. generate: decoding step 0..N. loglikelihood: position inside the scored span.",
)


TABLES: dict[str, TableSpec] = {
    "docs": TableSpec(
        name="docs",
        key=("task_name", "doc_id", "choice_idx"),
        layer_domain=None,
        description=(
            "One row per forward pass we ran, carrying lm-eval's grading result. "
            "Grading is never recomputed here; it is joined in from --log_samples. "
            "Exactly one row per (task_name, doc_id, choice_idx): a task with "
            "several filters is reduced to its primary one, named in `filter`."
        ),
        columns=(
            _AXIS_TASK,
            _AXIS_DOC,
            _AXIS_CHOICE,
            ColumnSpec(
                "is_correct",
                pa.bool_(),
                "Did the model answer this document correctly? A per-document value, "
                "so it repeats across the choices of one document.",
            ),
            ColumnSpec(
                "is_target_choice",
                pa.bool_(),
                "Is this row the forward pass of the *gold* choice? Distinguishes signals "
                "taken from the correct choice from signals taken from a distractor.",
            ),
            ColumnSpec("predicted", pa.string(), "The answer lm-eval says the model chose."),
            ColumnSpec("target", pa.string(), "The gold answer."),
            ColumnSpec(
                "choice_logprob",
                pa.float32(),
                "Summed log probability of this choice's continuation.",
                unit="log probability",
                comparable_across_vocab=False,
            ),
            ColumnSpec(
                "filter",
                pa.string(),
                "Which lm-eval filter produced is_correct/predicted. A task can "
                "define several (gsm8k has strict-match and flexible-extract) and "
                "lm-eval logs one sample record per filter; keeping them all would "
                "make this table non-unique on its key and fan out every join. The "
                "other filters' results stay in samples.jsonl.",
            ),
        ),
    ),
    "steps": TableSpec(
        name="steps",
        key=("task_name", "doc_id", "choice_idx", "step"),
        layer_domain=None,
        description=(
            "What the target token of each step was. Cannot be replaced by the layer-L row "
            "of `signals`: for loglikelihood the scored token is the gold continuation token, "
            "not the model's prediction, and for sampled generation the emitted token may "
            "differ from the last layer's top-1."
        ),
        columns=(
            _AXIS_TASK,
            _AXIS_DOC,
            _AXIS_CHOICE,
            _AXIS_STEP,
            ColumnSpec(
                "token_id",
                pa.int64(),
                "generate: the token actually generated at this step. "
                "loglikelihood: the continuation token being scored. "
                "A generate task records every decoding step that ran, which "
                "includes the tokens forming a stop sequence: generation only halts "
                "after emitting it, and lm-eval strips it from the text it reports. "
                "So decoding this column yields lm-eval's `resps` as a prefix, "
                "usually with a stop string after it - not an exact match.",
                comparable_across_vocab=False,
            ),
            ColumnSpec("token", pa.string(), "Tokenizer output for token_id, kept verbatim.",
                       comparable_across_vocab=False),
            ColumnSpec(
                "position",
                pa.int32(),
                "Absolute position in the input sequence. Prompt lengths differ per document, "
                "so `step` alone does not locate the token.",
                unit="tokens",
            ),
        ),
    ),
    "signals": TableSpec(
        name="signals",
        key=("task_name", "doc_id", "choice_idx", "step", "layer"),
        layer_domain="residual",
        description=(
            "Logit-lens top-1 and gold-token rank, one row per residual layer per step."
        ),
        columns=(
            _AXIS_TASK,
            _AXIS_DOC,
            _AXIS_CHOICE,
            _AXIS_STEP,
            ColumnSpec(
                "layer",
                pa.int32(),
                "Residual index 0..L. `layer j` is the input of block j; `layer L` is the "
                "output of the last block.",
            ),
            ColumnSpec(
                "lens_token_id",
                pa.int64(),
                "Top-1 token id after decoding this layer through the model's own final "
                "norm + lm_head. Note: on tied-embedding models `layer 0` usually decodes "
                "back to the input token itself. That is a property of the convention, "
                "not a bug, and must not be read as a prediction.",
                comparable_across_vocab=False,
            ),
            ColumnSpec(
                "lens_prob",
                pa.float32(),
                "Softmax probability of the top-1 token.",
                unit="probability",
                comparable_across_vocab=False,
            ),
            ColumnSpec(
                "lens_token",
                pa.string(),
                "Tokenizer output for lens_token_id, stored verbatim. Leading-space markers "
                "and partial byte pieces are NOT cleaned up; byte-level BPE can split a "
                "character in half, so a single token may decode to a broken glyph.",
                comparable_across_vocab=False,
            ),
            ColumnSpec(
                "lens_is_special",
                pa.bool_(),
                "Whether lens_token_id is a special token, as judged by the tokenizer.",
                comparable_across_vocab=False,
            ),
            ColumnSpec(
                "target_rank",
                pa.int64(),
                "Rank of the gold token in this layer's distribution (0 = top-1). "
                "loglikelihood only; null for generate tasks. Being a count rather "
                "than a quantity, it is not reproducible to the last place: where "
                "several tokens are nearly tied, a last-bit difference in the logits "
                "moves it a place or two. Recomputing it independently on Qwen2.5-0.5B "
                "moved 1.6% of rows by at most 3 out of a 151936-token vocabulary, "
                "with no directional bias. Treat single-place differences as noise.",
                unit="rank",
                comparable_across_vocab=False,
            ),
            ColumnSpec(
                "target_percentile",
                pa.float32(),
                "target_rank / vocab_size. Normalised, so it is comparable across models "
                "with different vocabulary sizes.",
                unit="fraction",
                comparable_across_vocab=True,
            ),
        ),
    ),
    "similarity": TableSpec(
        name="similarity",
        key=("task_name", "doc_id", "choice_idx", "step", "layer_i", "layer_j"),
        layer_domain="residual",
        description=(
            "Pairwise cosine similarity between residual layers at one position. "
            "The matrix is symmetric, so only the upper triangle including the diagonal "
            "is stored (layer_i <= layer_j)."
        ),
        columns=(
            _AXIS_TASK,
            _AXIS_DOC,
            _AXIS_CHOICE,
            _AXIS_STEP,
            ColumnSpec("layer_i", pa.int32(), "Residual index 0..L."),
            ColumnSpec("layer_j", pa.int32(), "Residual index 0..L, always >= layer_i."),
            ColumnSpec(
                "cos",
                pa.float32(),
                "Cosine similarity without centering. Interior values cluster near 1 "
                "because the residual stream is only ever added to, giving every token a "
                "large shared component; read relative changes between layer pairs, not "
                "absolute size. The two endpoint pairs are structurally different and "
                "should not be read as part of that trend: (0, 1) crosses from the "
                "embedding output into the first block, and (L-1, L) crosses the final "
                "norm, since layer L is the normed hidden state. Both are far lower than "
                "their neighbours - on Qwen2.5-1.5B, 0.07 and 0.27 against an interior "
                "0.73 to 0.98.",
                unit="cosine",
                comparable_across_vocab=True,
            ),
        ),
    ),
    "attn_norm": TableSpec(
        name="attn_norm",
        key=("task_name", "doc_id", "choice_idx", "step", "block", "head"),
        layer_domain="block",
        description=(
            "Per-head value-vector norm. Written only with --save-attention. "
            "One scalar per head per step, so unlike the attention weight tensors this "
            "table is dense along the step axis."
        ),
        columns=(
            _AXIS_TASK,
            _AXIS_DOC,
            _AXIS_CHOICE,
            _AXIS_STEP,
            ColumnSpec(
                "block",
                pa.int32(),
                "Block index 0..L-1. `block j` is the attention inside block j: it reads "
                "`layer j` and contributes to `layer j+1`.",
            ),
            ColumnSpec(
                "head",
                pa.int32(),
                "Query head index. Under GQA several query heads share one key/value head, "
                "so the same value norm repeats across the heads of a group.",
            ),
            ColumnSpec(
                "value_norm",
                pa.float32(),
                "L2 norm of this head's value vector at the current position. "
                "Captured at the projection, so it is computed from the *normalised* "
                "block input - a pre-norm block applies input_layernorm before "
                "attention - not from the raw residual stream. Recomputing it from "
                "hidden_states without that norm is off by orders of magnitude, "
                "because the residual norm grows with depth and the normalised input "
                "does not.",
                unit="L2 norm",
                comparable_across_vocab=True,
            ),
        ),
    ),
}


def describe_schema() -> str:
    """Render the declared schema as a table for humans.

    This is the documentation.
    `load_signals()` prints it, and the README's output section is expected to quote this rather than restate it.

    Example:
        >>> print(describe_schema())          # doctest: +ELLIPSIS
        schema_version 0.3
        <BLANKLINE>
        == docs == key=(task_name, doc_id, choice_idx) layer_domain=-
        ...
    """
    lines = [f"schema_version {SCHEMA_VERSION}", ""]
    for spec in TABLES.values():
        lines.append(
            f"== {spec.name} == key=({', '.join(spec.key)}) "
            f"layer_domain={spec.layer_domain or '-'}"
        )
        lines.append(f"   {spec.description}")
        for col in spec.columns:
            vocab = "any-vocab" if col.comparable_across_vocab else "same-vocab-only"
            unit = f" [{col.unit}]" if col.unit else ""
            lines.append(f"   - {col.name}: {col.dtype}{unit} ({vocab})")
            lines.append(f"       {col.description}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Writing: parquet shards
# --------------------------------------------------------------------------


class ShardWriter:
    """Buffers rows for one table and flushes them into numbered parquet shards.

    Shards are write-once: we never append to an existing file.
    If a run dies halfway through, every shard already on disk is a complete, readable parquet file.

    Numbering continues after whatever is already in the directory, so a restart adds shards rather than overwriting them.
    Restarting with numbering reset to zero is worse than losing the old data outright: the shorter of the two runs only overwrites its own prefix, and what is left on disk is a mixture of both runs that still reads as a valid table.

    Example:
        >>> w = ShardWriter("/tmp/run", TABLES["docs"])          # doctest: +SKIP
        >>> w.add({"task_name": "xnli_ko", "doc_id": 0, "choice_idx": 0,
        ...        "is_correct": True, "is_target_choice": True,
        ...        "predicted": "0", "target": "0", "choice_logprob": -1.5})
        >>> w.flush()      # writes /tmp/run/docs/part-0000.parquet
        >>> w.close()
    """

    def __init__(self, run_dir: str | os.PathLike, spec: TableSpec) -> None:
        self.spec = spec
        self.dir = os.path.join(str(run_dir), spec.name)
        self._schema = spec.arrow_schema()
        self._buffer: list[dict[str, Any]] = []
        self._shard_index = _next_shard_index(self.dir)

    def add(self, row: dict[str, Any]) -> None:
        """Buffer one row.
Missing columns become null."""
        self._buffer.append(row)

    def extend(self, rows: Iterable[dict[str, Any]]) -> None:
        """Buffer many rows at once."""
        self._buffer.extend(rows)

    def flush(self) -> str | None:
        """Write the buffer as the next shard.
Returns the path, or None if empty."""
        if not self._buffer:
            return None
        os.makedirs(self.dir, exist_ok=True)
        # Build columns explicitly from the declared schema so that a row that forgot a column fails loudly here rather than producing a file whose column set silently differs from every other run.
        columns = {
            name: [row.get(name) for row in self._buffer] for name in self.spec.column_names
        }
        table = pa.Table.from_pydict(columns, schema=self._schema)
        path = os.path.join(self.dir, f"part-{self._shard_index:04d}.parquet")
        # A process killed inside `write_table` left a footerless shard under its real name on RunPod, and resume then refused the directory.
        # Write under a name no reader matches (readers take `*.parquet`), make it durable, then publish it with one rename.
        import uuid
        temporary = os.path.join(self.dir, f".part-{self._shard_index:04d}.parquet.{uuid.uuid4().hex}.tmp")
        pq.write_table(table, temporary, compression="zstd")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(self.dir)
        self._buffer.clear()
        self._shard_index += 1
        return path

    def close(self) -> None:
        """Flush whatever is left in the buffer."""
        self.flush()


def _next_shard_index(directory: str) -> int:
    """The first shard number not already taken in `directory`.

    Example:
        >>> _next_shard_index("/tmp/empty")
        0
    """
    if not os.path.isdir(directory):
        return 0
    used = [
        int(name[len("part-") : -len(".parquet")])
        for name in os.listdir(directory)
        if name.startswith("part-") and name.endswith(".parquet")
        and name[len("part-") : -len(".parquet")].isdigit()
    ]
    return max(used) + 1 if used else 0


class TensorStore:
    """Writes opt-in raw tensors (attention weights, hidden states) as safetensors.

    One file per document and choice, containing all requested layers.
    Saving the same key replaces its file. This store does not provide the
    transactional resume contract used by custom collection artifacts.

    Example:
        >>> store = TensorStore("/tmp/run", "attention")             # doctest: +SKIP
        >>> store.save("xnli_ko", 12, 0, {"block_00": weights},
        ...            meta={"step": "0"})
        '/tmp/run/attention/xnli_ko__000012__0.safetensors'
    """

    def __init__(self, run_dir: str | os.PathLike, name: str) -> None:
        self.dir = os.path.join(str(run_dir), name)
        self.name = name
        self._count = 0

    def path_for(self, task_name: str, doc_id: int, choice_idx: int) -> str:
        """Deterministic file name, so existence alone answers "did we do this?"."""
        safe_task = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in task_name)
        return os.path.join(self.dir, f"{safe_task}__{doc_id:06d}__{choice_idx}.safetensors")

    def save(
        self,
        task_name: str,
        doc_id: int,
        choice_idx: int,
        tensors: dict[str, Any],
        meta: dict[str, str] | None = None,
    ) -> str:
        from safetensors.torch import save_file  # lazy: keeps storage.py torch-free

        os.makedirs(self.dir, exist_ok=True)
        path = self.path_for(task_name, doc_id, choice_idx)
        metadata = {"task_name": task_name, "doc_id": str(doc_id), "choice_idx": str(choice_idx)}
        metadata.update(meta or {})
        save_file({k: v.contiguous() for k, v in tensors.items()}, path, metadata=metadata)
        self._count += 1
        return path

    @property
    def files_written(self) -> int:
        return self._count


class RunWriter:
    """Owns every writer of one run directory and rolls all shards together.

    Rolling every table at the same document boundary means shard N of `signals` and shard N of `steps` cover the same documents, which makes a partially written run easy to reason about.

    Example:
        >>> with RunWriter("/tmp/run") as w:                        # doctest: +SKIP
        ...     w.write_document({"steps": [...], "signals": [...]})
    """

    def __init__(self, run_dir: str | os.PathLike, shard_size: int = SHARD_SIZE) -> None:
        self.run_dir = str(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        acquire_writer_lock(self.run_dir)
        self.shard_size = shard_size
        # Read before opening any writer: which documents a restart must not record a second time.
        self.already_recorded = existing_doc_keys(self.run_dir, "steps")
        self.tables = {name: ShardWriter(self.run_dir, spec) for name, spec in TABLES.items()}
        self.attention = TensorStore(self.run_dir, "attention")
        self.raw_hidden = TensorStore(self.run_dir, "raw")
        self._docs_since_flush = 0
        self.documents_written = 0

    def register_table(self, spec: TableSpec) -> None:
        """실행별 custom schema를 등록하고 기존 schema와의 혼합을 거부한다."""
        import base64
        import re
        if not re.fullmatch(r"custom/[A-Za-z][A-Za-z0-9_]*", spec.name):
            raise ValueError("custom table must use custom/<name>")
        schema = spec.arrow_schema()
        path = os.path.join(self.run_dir, spec.name, "schema.json")
        encoded = base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as stream:
                previous = json.load(stream)
            if previous["arrow_schema"] != encoded:
                raise ValueError(f"custom schema changed: {spec.name}")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({"arrow_schema": encoded, "metadata": json.loads(schema.metadata[b"eval_framework"])}, stream, indent=2)
        if spec.name not in self.tables:
            self.tables[spec.name] = ShardWriter(self.run_dir, spec)

    def write_document(
        self, rows_by_table: dict[str, list[dict[str, Any]]], documents: int = 1
    ) -> None:
        """Add one batch's worth of rows and roll the shards when due.

        Args:
            rows_by_table: table name -> rows.
                Tables absent from the dict are simply not written for this document (e.g. `attn_norm` when --save-attention is off).
            documents: how many documents these rows cover.
                Greater than one when a batched forward pass finished several documents at once; it only affects when the next shard is rolled.
        """
        for table_name, rows in rows_by_table.items():
            if table_name not in self.tables:
                raise KeyError(f"unknown table {table_name!r}; declared tables: {list(self.tables)}")
            self.tables[table_name].extend(rows)
        self.documents_written += documents
        self._docs_since_flush += documents
        if self._docs_since_flush >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        """Close the current shard of every table and start the next one."""
        for writer in self.tables.values():
            writer.flush()
        self._docs_since_flush = 0

    def signal_files(self) -> list[str]:
        """Relative paths of everything this run produced, for `results.json`."""
        found: list[str] = []
        custom_root = os.path.join(self.run_dir, "custom")
        custom_tables = [f"custom/{name}" for name in os.listdir(custom_root)
                         if os.path.isdir(os.path.join(custom_root, name))] if os.path.isdir(custom_root) else []
        for sub in dict.fromkeys(list(self.tables) + ["attention", "raw"] + custom_tables):
            directory = os.path.join(self.run_dir, sub)
            if not os.path.isdir(directory):
                continue
            if sub.startswith("custom/"):
                from pathlib import Path
                found.extend(str(p.relative_to(self.run_dir)) for p in Path(directory).iterdir() if p.is_file())
                found.extend(str(p.relative_to(self.run_dir)) for p in (Path(directory) / "tensors").glob("*") if p.is_file() and p.suffix != ".tmp")
                for marker in (Path(self.run_dir) / "custom_collection/commits").glob("*.json"):
                    data = _read_custom_commit(self.run_dir, marker)
                    found.extend(p for p in data["files"] if p.startswith(sub + "/"))
            else:
                for name in sorted(os.listdir(directory)):
                    found.append(f"{sub}/{name}")
        return found

    def close(self) -> None:
        for writer in self.tables.values():
            writer.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
        release_writer_lock(self.run_dir)


# One lock per run directory held by this process; see `acquire_writer_lock`.
_WRITER_LOCKS: dict[str, Any] = {}


def acquire_writer_lock(run_dir: str | os.PathLike) -> None:
    """Hold an exclusive lock on a run directory, refusing a second writing process.

    Shard numbers are chosen from the files already on disk, so two processes writing one
    directory would reuse them. Reopening a directory this process already holds is allowed;
    `run` and `collect-research-data` release the lock when they return or fail.
    """
    import fcntl

    path = os.path.realpath(str(run_dir))
    if path in _WRITER_LOCKS:
        return
    handle = open(os.path.join(path, ".writer.lock"), "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise ValueError(
            f"another process is writing to {run_dir}; a run directory takes one writer at a time"
        ) from None
    _WRITER_LOCKS[path] = handle


def release_writer_lock(run_dir: str | os.PathLike) -> None:
    """Release this process's lock on a run directory, if it holds one."""
    handle = _WRITER_LOCKS.pop(os.path.realpath(str(run_dir)), None)
    if handle is not None:
        handle.close()


def existing_doc_keys(run_dir: str | os.PathLike, table: str = "steps") -> set[tuple[str, int, int]]:
    """Read back which (task_name, doc_id, choice_idx) triples are already stored.

    Used to resume a run that died: whatever a completed shard contains does not need to be recomputed.

    Example:
        >>> existing_doc_keys("results/xnli_ko/qwen3-8b/2026-01-01-ab12cd")  # doctest: +SKIP
        {('xnli_ko', 0, 0), ('xnli_ko', 0, 1), ('xnli_ko', 1, 0)}
    """
    directory = os.path.join(str(run_dir), table)
    if not os.path.isdir(directory):
        return set()
    keys: set[tuple[str, int, int]] = set()
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".parquet"):
            continue
        path = os.path.join(directory, name)
        try:
            shard = pq.read_table(path, columns=["task_name", "doc_id", "choice_idx"])
        except (pa.ArrowException, OSError) as error:
            raise ValueError(f"cannot read recorded shard {path}: {error}") from error
        for task_name, doc_id, choice_idx in zip(
            shard.column("task_name").to_pylist(),
            shard.column("doc_id").to_pylist(),
            shard.column("choice_idx").to_pylist(),
        ):
            keys.add((task_name, int(doc_id), int(choice_idx)))
    return keys


def check_recorded_tables_agree(run_dir: str | os.PathLike, tables: Iterable[str]) -> None:
    """Refuse to resume when per-document tables on disk cover different documents.

    Resume skips every (task_name, doc_id, choice_idx) already in `steps`. A missing shard,
    or a flush that stopped between tables, would otherwise resume as silently missing
    signal rows, or as a second copy of them.
    """
    steps = existing_doc_keys(run_dir, "steps")
    for table in tables:
        keys = existing_doc_keys(run_dir, table)
        if keys != steps:
            raise ValueError(
                f"{run_dir}: {table} covers {len(keys)} (task, doc, choice) keys but steps covers "
                f"{len(steps)} ({len(steps - keys)} missing from {table}, {len(keys - steps)} absent "
                "from steps). A shard is missing or a write stopped between tables, so resuming "
                "would drop or duplicate rows. Restore the shards or use a new output directory."
            )


# --------------------------------------------------------------------------
# Manifest and results.json
# --------------------------------------------------------------------------

RESULTS_FILENAME = "results.json"
SAMPLES_FILENAME = "samples.jsonl"

# : Keys the manifest must carry.
# Two runs whose manifests : differ in any of these are not comparable, so a missing key would let : `report.py` group runs it should have kept apart.
REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "tool_version",
    "lm_eval_version",
    "model_id",
    "tokenizer_id",
    "tasks",
    "num_fewshot",
    "limit",
    "doc_id_set_hash",
    "reducers",
    "n_blocks",
    "layer_index_convention",
    "attn_index_convention",
    "fixed_settings",
    "seed",
    "started_at",
    "completed",
)


def write_results(
    run_dir: str | os.PathLike,
    manifest: dict[str, Any],
    scores: dict[str, Any],
    signal_files: list[str] | None = None,
) -> str:
    """Write `results.json`: lm-eval's scores plus our manifest plus a file list.

    The manifest is what `report.py` reads; it never parses the directory path, because `--output` lets the user put a run anywhere.

    Example:
        >>> write_results("/tmp/run", manifest, {"xnli_ko": {"acc": 0.71}})  # doctest: +SKIP
        '/tmp/run/results.json'
    """
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest,
        "results": scores,
        "signal_files": signal_files or [],
    }
    os.makedirs(str(run_dir), exist_ok=True)
    path = os.path.join(str(run_dir), RESULTS_FILENAME)
    _atomic_json(path, payload, ensure_ascii=False, default=str, allow_nan=True)
    return path


def read_results(run_dir: str | os.PathLike) -> dict[str, Any]:
    """Load `results.json`.
Raises FileNotFoundError if this is not a run directory."""
    with open(os.path.join(str(run_dir), RESULTS_FILENAME), encoding="utf-8") as fh:
        return json.load(fh)


def read_manifest(run_dir: str | os.PathLike) -> dict[str, Any]:
    """Load only the manifest part of `results.json`.

    Example:
        >>> read_manifest("results/xnli_ko/qwen3-8b/2026-01-01-ab12cd")["n_blocks"]  # doctest: +SKIP
        36
    """
    return read_results(run_dir)["manifest"]


def mark_complete(run_dir: str | os.PathLike, extra: dict[str, Any] | None = None) -> None:
    """Stamp a run as finished.

    `report.py` excludes runs without this marker: a run that died mid-way has a truncated document set, and averaging it against a complete run is a silent mistake.
    """
    payload = read_results(run_dir)
    payload["manifest"]["completed"] = True
    payload["manifest"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    payload["manifest"].update(extra or {})
    _atomic_json(os.path.join(str(run_dir), RESULTS_FILENAME), payload,
                 ensure_ascii=False, default=str, allow_nan=True)


# --------------------------------------------------------------------------
# samples.jsonl  (lm-eval --log_samples, kept verbatim inside the run dir)
# --------------------------------------------------------------------------


def write_samples(run_dir: str | os.PathLike, samples_by_task: dict[str, list[dict]]) -> str:
    """Persist lm-eval's per-sample log inside the run directory.

    We write it ourselves from `simple_evaluate(..., log_samples=True)` rather than letting lm-eval's EvaluationTracker do it, for two reasons: the tracker writes outside our run directory, and it rewrites `arguments` into a flattened dict, which `collect-research-data` needs in its original (context, continuation) form.

    Example of one written line:
        {"task_name": "xnli_ko", "doc_id": 0, "target": 1, "arguments": [["...ctx...", " Yes"], ["...ctx...", " No"]], "filtered_resps": [[-3.1, false], [-2.4, false]], "acc": 1.0, "prompt_hash": "d41d8c..."}
    """
    path = os.path.join(str(run_dir), SAMPLES_FILENAME)
    os.makedirs(str(run_dir), exist_ok=True)
    # Judge grading checkpoints rewrite this log. Keep the previous complete
    # generation log if serialization or writing the replacement fails.
    temporary = path + '.tmp'
    with open(temporary, "w", encoding="utf-8") as fh:
        for task_name, samples in samples_by_task.items():
            for sample in samples:
                record = dict(sample)
                record["task_name"] = task_name
                # `arguments` holds tuples; json turns them into lists anyway, but we normalise here so collection always sees the same shape.
                record["arguments"] = [list(arg) for arg in record.get("arguments", [])]
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)
    return path


def read_samples(run_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """Read `samples.jsonl` back as a list of records.

    Example:
        >>> samples = read_samples("results/xnli_ko/qwen3-8b/2026-01-01-ab12cd")  # doctest: +SKIP
        >>> samples[0]["task_name"], samples[0]["doc_id"]
        ('xnli_ko', 0)
    """
    path = os.path.join(str(run_dir), SAMPLES_FILENAME)
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# Joining lm-eval's grading result onto the `docs` table
# --------------------------------------------------------------------------

# : Metrics we treat as a per-document right/wrong verdict, in preference order. : We never re-derive correctness ourselves; we only read what lm-eval scored.
_CORRECTNESS_METRICS = ("acc", "exact_match", "acc_norm", "em", "f1")


def _extract_is_correct(sample: dict[str, Any]) -> bool | None:
    """Pull a boolean verdict out of an lm-eval sample record.

    lm-eval stores the metric values it computed directly on the record (`sample["acc"] = 1.0`).
    We take the first metric we recognise and treat 1.0 as correct.
    Returns None when the task reports no such metric (e.g. a purely generative task scored by ROUGE), in which case `is_correct` stays null rather than being guessed.

    Example:
        >>> _extract_is_correct({"metrics": ["acc"], "acc": 1.0})
        True
        >>> _extract_is_correct({"metrics": ["rouge1"], "rouge1": 0.42}) is None
        True
    """
    if "_eval_framework" in sample:
        return sample["_eval_framework"]["is_correct"]
    for metric in _CORRECTNESS_METRICS:
        if metric in sample:
            try:
                return float(sample[metric]) == 1.0
            except (TypeError, ValueError):
                return None
    return None


def _gold_choice_index(
    sample: dict[str, Any], n_choices: int, choices: Sequence[str] | None = None
) -> int | None:
    """Work out which choice was the gold one.

    Two forms, because lm-eval accepts two and resolves them the same way itself (`api/task.py`: `gold = choices.index(gold) if isinstance(gold, str)`).

    * an **index**, which is what `doc_to_target` yields for a task whose target field is a number. It is stringified on the way to disk, hence the int() attempt.
    * a **label**, for a task whose `doc_to_target` names the answer and whose `doc_to_choice` lists the labels - `doc_to_target: answer` over `["A", "B", "C", "D"]`, which is how Global-MMLU and most letter-choice benchmarks are written. The target is then `"A"`, and reading it as an index gives nothing.

    Matching a label needs the choices, which are the continuations of the logged request, compared with surrounding whitespace removed: lm-eval joins a choice to its context with a delimiter (a space by default), so the continuation on disk is `" A"` where the target is `"A"`.

    Anything still unmatched - a free-form target, an out-of-range index - yields None, and `is_target_choice` is then false for every choice rather than being invented.

    Example:
        >>> _gold_choice_index({"target": "1"}, n_choices=3)
        1
        >>> _gold_choice_index({"target": "C"}, 4, choices=[" A", " B", " C", " D"])
        2
        >>> _gold_choice_index({"target": "Paris"}, n_choices=3) is None
        True
    """
    target = sample.get("target")
    try:
        index = int(target)
    except (TypeError, ValueError):
        pass
    else:
        return index if 0 <= index < n_choices else None
    if choices and isinstance(target, str):
        wanted = target.strip()
        for position, choice in enumerate(choices):
            if str(choice).strip() == wanted:
                return position if position < n_choices else None
    return None


def available_filters(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Which lm-eval filters appear per task, in order of first appearance.

    Example:
        >>> available_filters([{"task_name": "gsm8k", "filter": "strict-match"},
        ...                    {"task_name": "gsm8k", "filter": "flexible-extract"},
        ...                    {"task_name": "gsm8k", "filter": "strict-match"}])
        {'gsm8k': ['strict-match', 'flexible-extract']}
    """
    found: dict[str, list[str]] = {}
    for sample in samples:
        name = sample.get("filter")
        if name is None:
            continue
        seen = found.setdefault(sample["task_name"], [])
        if name not in seen:
            seen.append(name)
    return found


def primary_filters(samples: list[dict[str, Any]]) -> dict[str, str]:
    """The one filter per task whose verdict goes into `docs`.

    lm-eval emits one sample record per filter, so a task with two filters would otherwise give `docs` two rows per document - and since `load_signals` joins on (task_name, doc_id, choice_idx), that would silently double every signal row.
    The first filter lm-eval reports is used, which is its own declaration order and therefore stable across runs of the same task.

    Example:
        >>> primary_filters([{"task_name": "gsm8k", "filter": "strict-match"},
        ...                  {"task_name": "gsm8k", "filter": "flexible-extract"}])
        {'gsm8k': 'strict-match'}
    """
    primary = {task: names[0] for task, names in available_filters(samples).items()}
    for sample in samples:
        if "_eval_framework" in sample:
            primary[sample["task_name"]] = sample["_eval_framework"]["primary_filter"]
    return primary


def build_docs_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn lm-eval sample records into `docs` rows, one per (doc, choice).

    Grading stays lm-eval's job here: and all this does is reshape its output from per-document to per-(document, choice) so it lines up with the signal tables.

    Records for a non-primary filter are skipped, so the result stays unique on (task_name, doc_id, choice_idx).

    Example:
        >>> rows = build_docs_rows([{
        ...     "task_name": "xnli_ko", "doc_id": 7, "target": "1",
        ...     "arguments": [["ctx", " Yes"], ["ctx", " No"]],
        ...     "filtered_resps": [[-3.1, False], [-2.4, False]],
        ...     "metrics": ["acc"], "acc": 1.0}])
        >>> [(r["choice_idx"], r["is_target_choice"], r["predicted"]) for r in rows]
        [(0, False, ' No'), (1, True, ' No')]
    """
    primary = primary_filters(samples)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        task_name = sample["task_name"]
        sample_filter = sample.get("filter")
        if sample_filter is not None and primary.get(task_name) != sample_filter:
            continue
        doc_id = int(sample["doc_id"])
        is_correct = _extract_is_correct(sample)
        responses = sample.get("filtered_resps") or []
        arguments = sample.get("arguments") or []

        # A loglikelihood response is [logprob, is_greedy]; a generate response is a plain string.
        # The shape tells us which kind of task this is.
        is_loglikelihood = bool(responses) and isinstance(responses[0], (list, tuple))

        if is_loglikelihood:
            logprobs = [float(resp[0]) for resp in responses]
            continuations = [
                str(argument[1]) for argument in arguments if len(argument) > 1]
            gold = _gold_choice_index(sample, len(logprobs), continuations)
            best = max(range(len(logprobs)), key=logprobs.__getitem__)
            # Report the winning continuation string when we have it, so the column is readable without cross-referencing the dataset.
            predicted = (
                str(arguments[best][1]) if best < len(arguments) and len(arguments[best]) > 1
                else str(best)
            )
            target_text = (
                str(arguments[gold][1]) if gold is not None and gold < len(arguments)
                and len(arguments[gold]) > 1 else str(sample.get("target"))
            )
            for choice_idx, logprob in enumerate(logprobs):
                rows.append(
                    {
                        "task_name": task_name,
                        "doc_id": doc_id,
                        "choice_idx": choice_idx,
                        "is_correct": is_correct,
                        "is_target_choice": gold is not None and choice_idx == gold,
                        "predicted": predicted,
                        "target": target_text,
                        "choice_logprob": logprob,
                        "filter": sample_filter,
                    }
                )
        else:
            # A generate task has one result row per document, with choice_idx=0.
            # Generation may use many forward passes; this table stores the final
            # response and has no per-choice log probability.
            rows.append(
                {
                    "task_name": task_name,
                    "doc_id": doc_id,
                    "choice_idx": 0,
                    "is_correct": is_correct,
                    "is_target_choice": True,
                    "predicted": str(responses[0]) if responses else "",
                    "target": str(sample.get("target")),
                    "choice_logprob": None,
                    "filter": sample_filter,
                }
            )
    return rows


def write_docs_table(run_dir: str | os.PathLike, samples: list[dict[str, Any]]) -> str | None:
    """Write the whole `docs` table in one shot from `samples.jsonl`.

    Done at the end of a run, not afterwards by a separate script: if the join lived outside the run, the run directory alone would not be reproducible.

    Tasks with more than one filter are reduced to the primary one and say so on stderr, because which filter `is_correct` came from changes what a conditional analysis means.

    Unlike the signal tables this one is rebuilt in full from `samples.jsonl` every time, so any earlier shards are removed first.
    Adding to them would duplicate the key that `load_signals` joins on.
    """
    import sys

    directory = os.path.join(str(run_dir), "docs")
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            if name.endswith(".parquet"):
                os.remove(os.path.join(directory, name))

    for task, names in available_filters(samples).items():
        if len(names) > 1:
            print(
                f"note: task {task} defines filters {names}; `docs` carries "
                f"{names[0]!r}. The others remain in {SAMPLES_FILENAME}.",
                file=sys.stderr,
            )
    writer = ShardWriter(run_dir, TABLES["docs"])
    writer.extend(build_docs_rows(samples))
    return writer.flush()


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_table(run_dir: str | os.PathLike, table: str):
    """Read every shard of one table as a pandas DataFrame (empty if absent).

    Example:
        >>> read_table(run_dir, "signals").columns.tolist()        # doctest: +SKIP
        ['task_name', 'doc_id', 'choice_idx', 'step', 'layer', 'lens_token_id', ...]
    """
    import pandas as pd

    if table.startswith("custom/"):
        import base64
        import re
        if not re.fullmatch(r"custom/[A-Za-z][A-Za-z0-9_]*", table):
            raise ValueError("invalid custom table name")
        directory = os.path.join(str(run_dir), table)
        with open(os.path.join(directory, "schema.json"), encoding="utf-8") as stream:
            stored = json.load(stream)
        schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(stored["arrow_schema"])))
        shards = sorted(os.path.join(directory, n) for n in os.listdir(directory) if n.endswith(".parquet"))
        from pathlib import Path
        for marker in (Path(run_dir) / "custom_collection" / "commits").glob("*.json"):
            data = _read_custom_commit(run_dir, marker)
            if table in data["tables"]:
                shards.append(str(Path(run_dir) / data["tables"][table]))
        return pq.read_table(sorted(shards), schema=schema).to_pandas() if shards else pd.DataFrame(columns=schema.names)
    spec = TABLES[table]
    directory = os.path.join(str(run_dir), table)
    if not os.path.isdir(directory):
        return pd.DataFrame(columns=spec.column_names)
    shards = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".parquet")
    )
    if not shards:
        return pd.DataFrame(columns=spec.column_names)
    return pq.read_table(shards).to_pandas()


def load_signals(path: str | os.PathLike, table: str = "signals", join_docs: bool = True):
    """Load one signal table of one run, with the grading columns joined in.

    Grading is stored once in `docs` and joined at read time rather than being copied into every signal row: `signals` has (layers x steps x choices) rows per document, so duplicating the answer strings there would bloat the files for no gain.

    Args:
        path: A run directory (the one containing `results.json`).
        table: Which signal table to load - "signals", "similarity", "steps" or "attn_norm".
        join_docs: Attach `is_correct` / `is_target_choice` / `predicted` / `target` / `choice_logprob` from the `docs` table.

    Returns:
        A pandas DataFrame.

    Example:
        >>> df = load_signals("results/xnli_ko/qwen3-8b/2026-01-01-ab12cd")  # doctest: +SKIP
        >>> df.query("is_correct and layer == 20").lens_prob.mean()
        0.42
    """
    frame = read_table(path, table)
    if join_docs and table != "docs" and len(frame):
        docs = read_table(path, "docs")
        if len(docs):
            # A duplicated key here would multiply every signal row instead of annotating it, and the result looks like ordinary data.
            # Refuse.
            key = ["task_name", "doc_id", "choice_idx"]
            duplicated = docs.duplicated(subset=key).sum()
            if duplicated:
                raise ValueError(
                    f"`docs` has {duplicated} rows duplicating its key {key}; joining "
                    "would silently multiply every signal row. This usually means "
                    "several lm-eval filters were written to `docs` instead of one."
                )
            frame = frame.merge(docs, on=key, how="left")
    return frame


def signal_columns_report(table: str = "signals") -> str:
    """The per-column documentation for one table, as printed next to the data.

    Example:
        >>> print(signal_columns_report("similarity"))       # doctest: +ELLIPSIS
        similarity: Pairwise cosine similarity...
          layer_i        int32 ...
    """
    spec = TABLES[table]
    lines = [f"{spec.name}: {spec.description}"]
    for col in spec.columns:
        vocab = "any-vocab" if col.comparable_across_vocab else "same-vocab-only"
        lines.append(f"  {col.name:<18} {str(col.dtype):<8} {vocab:<15} {col.description}")
    return "\n".join(lines)


def read_sample_metrics(run_dir: str | os.PathLike, *, metric: str, filter_name: str):
    """Read exactly one metric/filter per document, without duplicating signal joins.

    Returns a DataFrame keyed by task_name/doc_id with sample_id and value. Values
    remain objects: corpus metrics may store tuples rather than scalar scores.
    Join with signals using ``validate="many_to_one"`` on task_name/doc_id.
    """
    import pandas as pd

    rows, seen = [], set()
    for sample in read_samples(run_dir):
        if sample.get("filter", "none") != filter_name or metric not in sample.get("metrics", []):
            continue
        key = (sample["task_name"], int(sample["doc_id"]))
        if key in seen:
            raise ValueError(f"duplicate metric/filter sample: {key}")
        seen.add(key)
        rows.append({"task_name": key[0], "doc_id": key[1],
                     "sample_id": sample.get("_eval_framework", {}).get("sample_id"),
                     "value": sample[metric]})
    return pd.DataFrame(rows, columns=["task_name", "doc_id", "sample_id", "value"])


def _custom_checkpoint(phase):
    """Fault-injection seam for subprocess durability tests; production is a no-op."""


# Custom tensors deliberately use unique artifacts, not TensorStore.path_for(): that
# legacy path identifies only a document/choice and overwrites repeated calls.
def _sync_directory(path):
    """Persist an atomic rename's directory entry on supported local filesystems."""
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path, value, **json_options):
    """Publish metadata only after its complete bytes are durable on the local filesystem."""
    from pathlib import Path
    import uuid
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        options = {"sort_keys": True, "indent": 2, "allow_nan": False, **json_options}
        json.dump(value, stream, **options)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def save_custom_tensors(run_dir, name, tensors, rows, output_features, attempt=None, max_bytes=67108864, input_features="input_features"):
    """Write an immutable safetensors file, then publish its independent JSON index.

    Limits include the serialized header. A crash between the two publications leaves
    an orphan artifact, detected by audit_custom_tensors; it is never silently indexed.
    """
    from pathlib import Path
    import uuid
    import hashlib
    from safetensors.torch import save_file
    import re
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or (attempt is not None and not re.fullmatch(r"[a-f0-9]{32}", attempt)):
        raise ValueError("invalid custom tensor namespace")
    root = Path(run_dir)
    namespace = root / "custom" / name
    if attempt is not None:
        namespace = namespace / "attempts" / attempt
    directory = namespace / "tensors"
    directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    path = directory / (identifier + ".safetensors")
    temporary = path.with_suffix(".tmp")
    _custom_checkpoint("tensor_before")
    save_file({k: t.detach().cpu().contiguous() for k, t in tensors.items()}, str(temporary))
    size = temporary.stat().st_size
    if size > max_bytes:
        temporary.unlink()
        raise ValueError("custom max_tensor_bytes exceeded (serialized file)")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)
    _custom_checkpoint("tensor_after")
    index = {"schema_version": 1, "tensor_id": identifier, "path": str(path.relative_to(root)),
             "bytes": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": rows,
             "tensors": {k: {"dtype": str(t.dtype), "shape": list(t.shape),
                              "axes": ["selected_input_position", output_features if k == "output" else input_features]}
                         for k, t in tensors.items()}}
    index_path = path.with_suffix(".json")
    _custom_checkpoint("index_before")
    _atomic_json(index_path, index)
    _custom_checkpoint("index_after")
    return str(index_path.relative_to(root)), size


def _custom_index_paths(run_dir, name):
    from pathlib import Path
    root = Path(run_dir)
    paths = list((root / "custom" / name / "tensors").glob("*.json"))
    for marker in (root / "custom_collection" / "commits").glob("*.json"):
        data = _read_custom_commit(root, marker)
        paths.extend(root / p for p in data["indexes"] if p.startswith(f"custom/{name}/"))
    return sorted(paths)


def read_custom_tensors(run_dir, name):
    """Return (index, tensor dict) pairs; no user factory or arbitrary pickle is imported.

    By default collection attempts are visible only through their committed marker.
    Content hashes and declared dtype/shape are checked before returning values.
    """
    from pathlib import Path
    import hashlib
    from safetensors.torch import load_file
    root = Path(run_dir)
    result = []
    for index_path in _custom_index_paths(root, name):
        data = json.loads(index_path.read_text())
        path = root / data["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("custom tensor path escapes run directory")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != data["sha256"]:
            raise ValueError("custom tensor artifact missing or corrupt")
        tensors = load_file(str(path))
        if set(tensors) != set(data["tensors"]):
            raise ValueError("custom tensor index keys mismatch")
        for key, tensor in tensors.items():
            declared = data["tensors"][key]
            if str(tensor.dtype) != declared["dtype"] or list(tensor.shape) != declared["shape"]:
                raise ValueError("custom tensor dtype/shape mismatch")
        result.append((data, tensors))
    return result


def audit_custom_tensors(run_dir, name):
    """Report orphan artifacts and missing artifacts, including uncommitted attempts."""
    from pathlib import Path
    root = Path(run_dir)
    base = root / "custom" / name
    artifacts = set(base.rglob("*.safetensors"))
    indexes = list(base.rglob("tensors/*.json"))
    indexed = {root / json.loads(p.read_text())["path"] for p in indexes}
    return {"orphan_artifacts": sorted(str(p.relative_to(root)) for p in artifacts - indexed),
            "missing_artifacts": sorted(str(p.relative_to(root)) for p in indexed - artifacts),
            "temporary_files": sorted(str(p.relative_to(root)) for p in base.rglob("*.tmp"))}


def _read_custom_commit(root, marker):
    """완료 marker의 구조·관측 범위·파일을 검증한 뒤 반환한다.

    marker가 있어도 참조 파일이 손상되면 완료된 수집으로 읽지 않는다.
    검사 순서를 유지해 여러 문제가 있을 때도 같은 오류를 먼저 보고한다.
    """
    from pathlib import Path
    root = Path(root)
    data = json.loads(Path(marker).read_text())
    required = {'version', 'key', 'attempt', 'tables', 'indexes', 'files', 'hooks'}
    if set(data) != required or data['version'] != 1:
        raise ValueError('incomplete custom collection marker')
    contract_path = root / 'custom_collection' / 'contract.json'
    if not contract_path.is_file():
        raise ValueError('incomplete custom collection manifest: missing contract')
    contract = json.loads(contract_path.read_text())
    if set(contract) != {'version', 'specs', 'resolved', 'config_identity', 'samples_sha256'} or contract['version'] != 1:
        raise ValueError('incomplete custom collection manifest')
    _validate_custom_coverage(data, contract)
    _validate_custom_files(root, data)
    return data


def _validate_custom_coverage(data: dict[str, Any], contract: dict[str, Any]) -> None:
    """계약의 모든 hook·모듈이 marker에 있는지 확인한다. 호출 0회도 유효한 기록이다.

    호출 횟수는 음이 아닌 int만 허용한다. bool을 횟수로 받아들이지 않도록
    isinstance 대신 정확한 타입을 검사한다.
    """
    expected = contract['resolved']
    if set(data['hooks']) != set(expected) or set(data['tables']) != {f'custom/{n}' for n in expected}:
        raise ValueError('incomplete custom collection hook coverage')
    for name, paths in expected.items():
        calls = data['hooks'][name]
        if set(calls) != set(paths) or any(type(count) is not int or count < 0 for count in calls.values()):
            raise ValueError('incomplete custom collection call coverage')


def _validate_custom_files(root, data: dict[str, Any]) -> None:
    """scalar·index·tensor 참조가 정확히 같은 파일 집합을 가리키는지 검사한다.

    먼저 marker가 나열한 모든 파일의 경로와 hash를 확인한다. 그다음 JSON index를
    읽어 tensor 참조까지 대조한다. 손상된 index를 먼저 해석하지 않는 순서다.
    """
    import hashlib

    references = list(data['tables'].values()) + data['indexes']
    if any(path not in data['files'] for path in references):
        raise ValueError('incomplete custom collection references')
    for relative, digest in data['files'].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('custom collection path escapes run directory')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'committed custom collection file missing or corrupt: {relative}')
    indexed_artifacts = []
    for relative in data['indexes']:
        index = json.loads((root / relative).read_text())
        indexed_artifacts.append(index['path'])
        if data['files'].get(index['path']) != index['sha256']:
            raise ValueError('incomplete custom collection tensor references')
    if set(data['files']) != set(references + indexed_artifacts):
        raise ValueError('custom collection file set mismatch')


class CustomCollectionStore:
    """One committed unit is (task, document, choice, collection pass).

    Each attempt has unique scalar/tensor/index paths. The only publication point is
    the final marker, after all outputs and hashes exist. Interrupted attempts remain
    available for diagnosis and never become default reader output. A process lock
    prevents two collectors from racing to publish the same unit.
    """
    def __init__(self, run_dir, contract):
        from pathlib import Path
        import fcntl
        self.root = Path(run_dir)
        self.directory = self.root / 'custom_collection'
        self.directory.mkdir(exist_ok=True)
        self.lock = (self.directory / 'writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._ensure_contract(contract)
            # 파일 존재만으로 완료를 추정하지 않는다. 각 marker의 참조를 검증한
            # 단위만 completed에 넣어 다음 수집에서 건너뛸 수 있게 한다.
            self.completed = {}
            for marker in (self.directory / 'commits').glob('*.json'):
                data = _read_custom_commit(self.root, marker)
                key = tuple(data['key'])
                if key in self.completed or marker.stem != self.key_id(key):
                    raise ValueError('duplicate or misidentified custom collection commit')
                if set(data['hooks']) != set(contract['resolved']):
                    raise ValueError('incomplete custom collection hook coverage')
                self.completed[key] = data
        except BaseException:
            self.close()
            raise

    def _ensure_contract(self, contract: dict[str, Any]) -> None:
        """writer lock을 잡은 상태에서 기존 계약과 비교하거나 최초 계약을 저장한다.

        JSON 왕복으로 tuple/list 차이를 없애 메모리의 계약과 디스크의 계약을
        같은 표현으로 비교한다. 기존 수집 흔적이 있으면 누락된 계약을 새로
        만들지 않는다. 그러면 과거 데이터의 의미를 새 계약으로 덮어쓰게 된다.
        """
        path = self.directory / 'contract.json'
        normalized = json.loads(json.dumps(contract, sort_keys=True))
        if path.exists():
            if json.loads(path.read_text()) != normalized:
                raise ValueError('custom collection provenance/selector/range/schema changed')
        else:
            if (any((self.directory / 'commits').glob('*.json'))
                    or any((self.root / 'custom').glob('*/attempts/*'))):
                raise ValueError('incomplete custom collection manifest: missing contract')
            _atomic_json(path, normalized)

    @staticmethod
    def key_id(key):
        """문서·choice·pass 키를 재시작 후에도 같은 marker 이름으로 변환한다."""
        import hashlib
        return hashlib.sha256(json.dumps(list(key), ensure_ascii=True).encode()).hexdigest()

    def commit(self, key, attempt, rows_by_table, specs, indexes, coverage):
        """scalar 저장 → 전체 파일 hash → 완료 marker 순서로 한 수집 단위를 공개한다.

        marker 이전에 실패한 파일은 미완료 attempt로 남아 기본 reader에서 제외된다.
        marker 이후에는 참조 파일 전체가 복구 가능해야 한다. fault checkpoint와
        메모리의 completed 갱신도 이 공개 순서를 따른다.
        """
        import hashlib

        tables = self._write_scalar_tables(attempt, rows_by_table, specs)
        # marker는 index 자체와 그 index가 가리키는 tensor를 모두 hash로 묶는다.
        files = list(tables.values()) + list(indexes)
        for index in indexes:
            files.append(json.loads((self.root / index).read_text())['path'])
        hashes = {p: hashlib.sha256((self.root / p).read_bytes()).hexdigest() for p in files}
        marker = {'version': 1, 'key': list(key), 'attempt': attempt, 'tables': tables,
                  'indexes': list(indexes), 'files': hashes, 'hooks': coverage}
        path = self.directory / 'commits' / (self.key_id(key) + '.json')
        if path.exists():
            raise ValueError('custom collection unit already committed')
        _custom_checkpoint("marker_before")
        _atomic_json(path, marker)
        _custom_checkpoint("marker_after")
        self.completed[tuple(key)] = marker

    def _write_scalar_tables(
        self, attempt: str, rows_by_table: dict[str, list[dict[str, Any]]], specs: Sequence[Any],
    ) -> dict[str, str]:
        """attempt별 scalar 파일을 쓰고 run 디렉터리 기준 상대 경로를 반환한다.

        행이 없어도 스키마를 가진 파일을 쓴다. 유효한 빈 관측과 파일 누락을
        구분하기 위해서다. 파일 fsync → rename → 디렉터리 fsync 순서를 유지하며,
        여기서는 완료 marker를 쓰지 않아 아직 수집 결과로 공개되지 않는다.
        """
        tables = {}
        for spec in specs:
            name = f'custom/{spec.name}'
            directory = self.root / name / 'attempts' / attempt
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'scalars.parquet'
            temporary = directory / 'scalars.tmp'
            rows = rows_by_table.get(name, [])
            schema = spec.table_spec().arrow_schema()
            table = pa.Table.from_pylist(rows, schema=schema)
            _custom_checkpoint("scalar_before")
            pq.write_table(table, temporary, compression='zstd')
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
            _custom_checkpoint("scalar_after")
            tables[name] = str(path.relative_to(self.root))
        return tables

    def close(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None
