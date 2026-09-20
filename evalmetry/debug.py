"""Per-module execution tracing, for finding out where a forward pass died.

This is a debugging aid, off by default, and deliberately separate from `recorder.py`.
The recorder exists to capture *signals* and is driven by what the reducers ask for; this exists to capture *what happened*, must keep working when the recorder is not recording (a batch-size probe, a document already on disk), and must survive the process that produced it.

CLI options are available through `evalmetry debug --help`.
Three things are worth repeating here, because they are what the code looks strange without.

**`always_call=True` is not used, on purpose.**
PyTorch's always-called forward hooks fire for every ancestor when a forward raises, innermost first, with `output=None`.
A tracer that used them would see its module stack unwind perfectly on the way out and would lose the only record of where execution stopped.
Without the flag the pops simply do not happen, so whatever is left on the stack *is* the failure path, root first and failure site last.

**Shapes are free; values are not; the allocator's counters are cheap only if you ask correctly.**
Reading `tensor.shape` queues no kernel and syncs nothing, and neither does reading the caching allocator's host-side counters - so the execution trace never adds memory pressure and is safe to leave on in a run that is already near the ceiling.
`torch.isnan(x).any()` allocates a bool tensor the size of `x`, and reading the count back blocks on the device.
That is why numeric diagnosis is a second flag rather than a verbosity level: switched on in an OOM-prone run it can be the allocation that kills it.

Costing nothing in *memory* is not the same as costing nothing in *time*, and the first version of this file conflated them. `torch.cuda.memory_allocated()` is 69us because it builds the entire statistics dict to return one number; three per event was most of a 6.5x slowdown. See `_build_memory_reader`.

**The logit lens re-enters the model.**
`adapters._build_decode_stack` calls the model's real `final_norm` and `lm_head`, inside the decoder's own forward hook.
Every event therefore carries a phase, and `forward_id` does not advance for work done in `signal_collection`.

Example:
    >>> cfg = DebugConfig(enabled=True, numeric=True)          # doctest: +SKIP
    >>> tracer = ModuleTracer(model, cfg, "run/debug/trace.jsonl")
    >>> with tracer.session():
    ...     with phase("model_forward"):
    ...         model(input_ids)
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence


#: Version of the trace record layout, written into the header.
TRACE_SCHEMA_VERSION = "1.1"

# : How many events `--debug-tail` keeps when no size is given.
# A 32-layer model has roughly 500 modules, so one loglikelihood forward is about 1000
# events and a generate document is that many per decoded token; five thousand is four or
# five forwards, which is the window a post-mortem reads.
TRACE_BUFFER_EVENTS = 5000

#: Deepest nesting level described for a container passed to a module.
DESCRIBE_DEPTH = 2

#: Most items described inside one container.
DESCRIBE_WIDTH = 8

#: The name given to the model object itself, which `named_modules()` calls "".
ROOT_NAME = "<root>"

#: Phase set when nothing has claimed the work: lm-eval's own code, scoring, the batch-size probe.
DEFAULT_PHASE = "other"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class DebugConfig:
    """What tracing was asked for.

    Kept out of `RunConfig.identity`: tracing changes how long a run takes and, with `sync` or `stop_on_nonfinite`, whether it finishes, but it does not change what the signals mean.
    Two runs differing only in these flags belong in the same directory.

    Attributes:
        enabled: register hooks at all.
        modules: regex matched against the dotted module path, e.g. `r"layers\\.\\d+$"`.
            Unset traces every module.
        numeric: also summarise tensor values - NaN/Inf counts and the finite range.
            Costs several same-size intermediates and a device sync per tensor.
        stop_on_nonfinite: raise at the first module whose output is not all finite.
            Implies `numeric`. Changes what the run does, so it is recorded in the manifest.
        tail: keep only the last N events in memory and write them at the end, instead of writing every event as it happens.
            Off by default, because "trace the run" should mean the whole run: a buffer that silently discards everything but the last few forwards is the right tool for a post-mortem and the wrong one for reading what a given document did.
            Worth setting when hunting a crash in a long run, where the full trace would be gigabytes and only the end of it matters. Note that the buffered trace is written when the session ends, so it does not survive the process being killed without an exception - which is what host RAM exhaustion looks like.
        sync: `torch.cuda.synchronize()` at every module boundary.
            Pins an asynchronous device fault to the module that caused it, at a large cost in wall time and at the risk of changing whether an intermittent failure reproduces. Buys nothing for OOM, whose allocation is synchronous already.
    """

    enabled: bool = False
    modules: str | None = None
    numeric: bool = False
    stop_on_nonfinite: bool = False
    tail: int | None = None
    sync: bool = False
    # Research statistics use document/position mappings, independently of the
    # optional whole-tensor diagnostics in the execution trace.
    sample_stats: bool = False
    # Largest absolute values per document and call, with their positions. Both the
    # modules and k are explicit: this writes k rows per document, tensor and call,
    # which is a different order of output from one moment row.
    extremes: int = 0
    extremes_modules: str | None = None
    # Statistics of one axis of the same slice: `feature`, `position` or `both`. One row
    # per index rather than per tensor, so the modules are named separately again.
    axes: str | None = None
    axes_modules: str | None = None
    # Which routers record the expert selection they returned. One selector is enough:
    # the k, the expert ids and the weights are the router's own output.
    routing: str | None = None
    # Elements per float64 reduction chunk. None keeps module_stats.CHUNK_ELEMENTS; the
    # stored values must not depend on it, only the size of the temporary buffers does.
    chunk_elements: int | None = None

    def __post_init__(self) -> None:
        if self.stop_on_nonfinite:
            self.numeric = True
        if self.chunk_elements is not None:
            if self.chunk_elements < 1:
                raise ValueError("--module-stats-chunk-elements must be at least 1")
            if not self.sample_stats:
                raise ValueError("--module-stats-chunk-elements needs --module-stats: it "
                                 "sets how the moments are reduced")
        if bool(self.extremes) != (self.extremes_modules is not None):
            raise ValueError("--module-stats-extremes and --module-stats-extremes-modules "
                             "are set together: k alone would collect positions for every "
                             "traced module, and a selector alone collects nothing")
        if self.extremes < 0:
            raise ValueError("--module-stats-extremes must not be negative")
        if self.extremes and not self.sample_stats:
            raise ValueError("--module-stats-extremes needs --module-stats: extreme "
                             "positions are recorded beside the moments of the same tensor")
        if self.extremes_modules is not None:
            try:
                re.compile(self.extremes_modules)
            except re.error as error:
                raise ValueError("--module-stats-extremes-modules is not a valid regular "
                                 f"expression: {error}") from error
        if bool(self.axes) != (self.axes_modules is not None):
            raise ValueError("--module-stats-axes and --module-stats-axes-modules are set "
                             "together: an axis alone would write a row per channel of "
                             "every traced module, and a selector alone collects nothing")
        if self.axes not in (None, "feature", "position", "both"):
            raise ValueError("--module-stats-axes must be feature, position or both")
        if self.axes and not self.sample_stats:
            raise ValueError("--module-stats-axes needs --module-stats: per-axis statistics "
                             "reduce the same document slice the moments describe")
        if self.axes_modules is not None:
            try:
                re.compile(self.axes_modules)
            except re.error as error:
                raise ValueError("--module-stats-axes-modules is not a valid regular "
                                 f"expression: {error}") from error
        if self.routing and not self.sample_stats:
            raise ValueError("--module-stats-routing needs --module-stats: routing "
                             "selections are mapped to documents by the same layout rules")
        if self.routing is not None:
            try:
                re.compile(self.routing)
            except re.error as error:
                raise ValueError("--module-stats-routing is not a valid regular "
                                 f"expression: {error}") from error
        if self.modules is not None:
            # Fail here rather than on the first forward, half an hour into a run - and as
            # a ValueError, which `main.CONFIGURATION_ERRORS` turns into a message instead
            # of a traceback. `re.error` is not a ValueError and would escape it.
            try:
                re.compile(self.modules)
            except re.error as error:
                raise ValueError(
                    f"--debug-modules is not a valid regular expression: {error}"
                ) from error

    @property
    def active(self) -> bool:
        """True when anything should be traced at all."""
        return self.enabled or self.stop_on_nonfinite or self.numeric or self.sample_stats

    def manifest_entry(self) -> dict[str, Any]:
        """What the manifest records, so a traced run is never mistaken for a plain one."""
        return {
            "enabled": self.active,
            "modules": self.modules,
            "numeric": self.numeric,
            "stop_on_nonfinite": self.stop_on_nonfinite,
            "tail": self.tail,
            "sync": self.sync,
            "sample_stats": self.sample_stats,
            "extremes": self.extremes,
            "extremes_modules": self.extremes_modules,
            "axes": self.axes,
            "axes_modules": self.axes_modules,
            "routing": self.routing,
            "chunk_elements": self.chunk_elements,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
        }


class NonFiniteOutput(RuntimeError):
    """Raised by `--debug-stop-on-nonfinite` at the module that first produced a NaN or an Inf.

    Deliberately an exception rather than a log line: a corruption that is only logged goes on propagating, and twenty layers later the module named by a trace is not the module that caused it.

    `module` carries the culprit separately from the message, because the module stack cannot: the check runs in the exit hook, after that module has already returned, so the deepest frame still open is its *parent*. The stack answers "where did execution stop"; this answers "whose output was bad", and for this one failure kind they differ by one level.
    """

    def __init__(self, message: str, module: str) -> None:
        super().__init__(message)
        self.module = module


# --------------------------------------------------------------------------
# The active tracer
# --------------------------------------------------------------------------

#: The tracer of the current session, or None. One model per process, so one tracer.
_ACTIVE: "ModuleTracer | None" = None


def active() -> "ModuleTracer | None":
    """The tracer of the current session, or None when tracing is off."""
    return _ACTIVE


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Label the module events raised inside this block.

    A no-op when tracing is off, which is why call sites can use it unconditionally.

    Example:
        >>> with phase("model_forward"):        # doctest: +SKIP
        ...     logits = model(input_ids).logits
    """
    tracer = _ACTIVE
    if tracer is None:
        yield
        return
    tracer._phases.append(name)
    try:
        yield
    finally:
        tracer._phases.pop()


def note_samples(contexts: Iterable[Any]) -> None:
    """Say which documents the next forward passes are for.

    A no-op when tracing is off.
    Takes anything with `task_name`, `doc_id`, `choice_idx` and `batch_row` attributes - `ForwardContext` in practice - and is deliberately duck-typed so this module does not import `reducers`.

    Pass the *unfiltered* contexts. A document already recorded by an interrupted run is scored again without being recorded, and it is still worth tracing.
    """
    tracer = _ACTIVE
    if tracer is None:
        return
    tracer.set_samples(contexts)


@contextmanager
def without_samples() -> Iterator[None]:
    """Do not attribute synthetic batch-size probes to the previous real batch."""
    tracer = _ACTIVE
    if tracer is None:
        yield
        return
    samples, batch_rows = tracer._samples, tracer._batch_rows
    stats = tracer.statistics
    contexts = stats.contexts if stats is not None else None
    tracer._samples = tracer._batch_rows = None
    if stats is not None:
        stats.contexts = []
    try:
        yield
    finally:
        tracer._samples, tracer._batch_rows = samples, batch_rows
        if stats is not None:
            stats.contexts = contexts


# --------------------------------------------------------------------------
# Describing what a module was handed
# --------------------------------------------------------------------------


def _build_memory_reader():
    """Resolve the cheapest way this torch will hand over the allocator's counters.

    Measured on an RTX PRO 4500 with torch 2.8, 20k iterations each:

        torch.cuda.memory_allocated()          69.5 us
        torch.cuda.memory_reserved()           71.5 us
        torch.cuda.memory_stats()              71.3 us
        torch.cuda.memory_stats_as_nested_dict  8.6 us
        reading tensor.shape / dtype / device    0.6 us

    The three public one-value accessors each build the whole ~60-entry statistics dict and return a single number out of it, so asking for three numbers pays for it three times. `memory_stats_as_nested_dict` is that same dict before the flattening pass, and one call answers all three questions.

    It is a public name with a documented return shape, but it is also more of an implementation detail than `memory_allocated`, so the shape is probed once here and the flattened accessors stand in if a future torch changes it. The probe costs one call per session.

    Returns:
        A `() -> dict | None` reader, or None when there is no CUDA device.
    """
    import torch

    if not torch.cuda.is_available():
        return None

    nested = getattr(torch.cuda, "memory_stats_as_nested_dict", None)
    if nested is not None:
        try:
            probe = nested()
            if probe:
                probe["allocated_bytes"]["all"]["current"]
                probe["reserved_bytes"]["all"]["current"]
                probe["allocated_bytes"]["all"]["peak"]
            else:
                # CUDA is not initialised yet; the shape cannot be probed but the call is
                # the right one. It returns {} until the first allocation, handled below.
                pass
        except (KeyError, TypeError):
            nested = None

    if nested is None:
        def read_flattened() -> dict[str, int]:
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
            }

        return read_flattened

    def read_nested() -> dict[str, int] | None:
        stats = nested()
        if not stats:
            return None
        allocated = stats["allocated_bytes"]["all"]
        return {
            "allocated": allocated["current"],
            "reserved": stats["reserved_bytes"]["all"]["current"],
            "max_allocated": allocated["peak"],
        }

    return read_nested


def _tensor_stats(tensor: Any) -> dict[str, Any]:
    """NaN/Inf counts and the distribution of one tensor's finite values.

    Reports min, max, mean, standard deviation and median rather than a single magnitude. A largest-absolute-value is derivable from min and max and says nothing they do not; what it cannot show is the *shape* of the distribution, which is where an activation going wrong actually announces itself - a mean drifting off zero, a standard deviation collapsing, a median far from the mean because a handful of outliers are carrying the tensor.

    Everything is over the finite values only. Non-finite entries are replaced with NaN and the `nan*` reductions skip them, so a single Inf cannot swallow the mean and the counts stay the honest record of how many there were.

    The arithmetic is done in float64 so squaring a large finite float32/bf16
    activation does not itself overflow during variance calculation.

    All eight numbers are stacked and read back together. The intermediates are
    the real price: boolean masks, a float64 copy and variance/median workspace.
    That is why `--debug-numeric` is a separate flag from `--debug`.

    A tensor with no finite values at all reports no distribution rather than infinities that would read like data.
    """
    import torch

    tensor = tensor.detach()
    if tensor.numel() == 0:
        return {"empty": True}
    if not tensor.is_floating_point():
        # An integer or boolean tensor cannot hold NaN or Inf, and reporting three zeros
        # for every `input_ids` teaches the reader to skim the column that matters on the
        # float tensors next to it. Its mean and median say nothing either: the average of
        # a set of token ids is not a token.
        low, high = torch.stack([tensor.min().float(), tensor.max().float()]).tolist()
        return {"min": low, "max": high}

    # `isposinf`/`isneginf` rather than `isinf & (tensor > 0)`: one kernel each instead of
    # a comparison and an and, which measured 540us against 270us on a 368k-element tensor.
    is_nan = torch.isnan(tensor)
    positive = torch.isposinf(tensor)
    negative = torch.isneginf(tensor)
    finite = ~(is_nan | positive | negative)
    masked = torch.where(finite, tensor.double(), float("nan"))
    mean = torch.nanmean(masked)
    centred = masked - mean
    numbers = torch.stack(
        [
            is_nan.sum().float(),
            positive.sum().float(),
            negative.sum().float(),
            torch.nan_to_num(masked, nan=float("inf")).amin(),
            torch.nan_to_num(masked, nan=float("-inf")).amax(),
            mean,
            torch.nanmean(centred * centred).sqrt(),
            torch.nanmedian(masked),
        ]
    )
    nan, posinf, neginf, low, high, average, deviation, middle = numbers.tolist()
    stats = {"nan": int(nan), "posinf": int(posinf), "neginf": int(neginf)}
    if low != float("inf"):
        stats.update(
            {
                "min": low,
                "max": high,
                "mean": average,
                "std": deviation,
                "median": middle,
            }
        )
    return stats


def _is_nonfinite(stats: dict[str, Any]) -> bool:
    return bool(stats.get("nan") or stats.get("posinf") or stats.get("neginf"))


def _describe(value: Any, numeric: bool, depth: int = 0) -> dict[str, Any]:
    """One JSON-able description of whatever a module was passed or returned.

    Bounded in both directions - `DESCRIBE_DEPTH` levels and `DESCRIBE_WIDTH` items - because a `past_key_values` is a list of layer-many tuples and describing it in full would bury the tensor that matters.
    """
    import torch

    if isinstance(value, torch.Tensor):
        described = {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
        if numeric:
            described["stats"] = _tensor_stats(value)
        return described
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": type(value).__name__, "value": value if not isinstance(value, str) else value[:60]}
    if isinstance(value, (list, tuple)) and depth < DESCRIBE_DEPTH:
        items = [_describe(item, numeric, depth + 1) for item in list(value)[:DESCRIBE_WIDTH]]
        return {"type": type(value).__name__, "len": len(value), "items": items}
    if isinstance(value, dict) and depth < DESCRIBE_DEPTH:
        keys = list(value)[:DESCRIBE_WIDTH]
        return {
            "type": "dict",
            "len": len(value),
            "items": {str(k): _describe(value[k], numeric, depth + 1) for k in keys},
        }
    cache = _describe_cache(value)
    if cache is not None:
        return cache
    return {"type": type(value).__name__}


def _describe_cache(value: Any) -> dict[str, Any] | None:
    """Size up a transformers KV cache, or return None if this is not one.

    Worth a special case of its own because the cache is *the* thing that grows during
    generation, and it is the usual answer to "what filled the card". It is not a list,
    tuple or dict, so the generic path saw only `{"type": "DynamicCache"}` - a trace that
    named the opaque object responsible for the allocation and said nothing about its size.

    `bytes` is the number to read: keys and values summed across every layer. It comes from
    `numel() * element_size()`, so it is metadata arithmetic - no device work, nothing
    allocated, consistent with the rest of the execution trace.

    Two layouts are handled because both are in scope: transformers 5.x keeps
    `cache.layers[i].keys`, and 4.x keeps `cache.key_cache[i]`.
    """
    import torch

    if not hasattr(value, "get_seq_length") and not hasattr(value, "key_cache"):
        return None

    pairs: list[tuple[Any, Any]] = []
    layers = getattr(value, "layers", None)
    if layers is not None:
        pairs = [(getattr(x, "keys", None), getattr(x, "values", None)) for x in layers]
    elif hasattr(value, "key_cache"):
        keys, values = value.key_cache, getattr(value, "value_cache", [])
        pairs = list(zip(keys, values)) if len(keys) == len(values) else [(k, None) for k in keys]

    described: dict[str, Any] = {"type": type(value).__name__, "layers": len(pairs)}
    total = 0
    first: Any = None
    for key, val in pairs:
        for tensor in (key, val):
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
                first = first if first is not None else tensor
    if first is not None:
        described.update(
            {
                "shape": list(first.shape),
                "dtype": str(first.dtype).replace("torch.", ""),
                "device": str(first.device),
                "bytes": total,
            }
        )
    try:
        described["seq"] = int(value.get_seq_length())
    except Exception:  # noqa: BLE001 - an empty or exotic cache simply has no length yet
        pass
    return described


def _tensor_descriptions(described: Any) -> Iterator[dict[str, Any]]:
    """Walk a `_describe` result and yield the entries that are tensors."""
    if isinstance(described, dict):
        if "shape" in described:
            yield described
            return
        items = described.get("items")
        if isinstance(items, list):
            for item in items:
                yield from _tensor_descriptions(item)
        elif isinstance(items, dict):
            for item in items.values():
                yield from _tensor_descriptions(item)


def _jsonable(value: Any) -> Any:
    """Replace the float values JSON has no literal for, so the log stays strict JSON.

    `json.dumps` would happily emit `NaN` and `Infinity`, which no conforming reader accepts - and this log is full of both by design.
    """
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# --------------------------------------------------------------------------
# The tracer
# --------------------------------------------------------------------------


@dataclass
class _Frame:
    """One module entry that has not yet returned."""

    path: str
    type_name: str
    call_id: int
    phase: str
    inputs: list[dict[str, Any]] = field(default_factory=list)


class ModuleTracer:
    """Registers enter/exit hooks on every module and writes what it sees.

    Args:
        model: the loaded model. Every `nn.Module` under it is hooked unless `config.modules` says otherwise.
        config: what to trace.
        path: where the JSONL goes. Its directory is created on session entry.

    Example:
        >>> tracer = ModuleTracer(model, DebugConfig(enabled=True), "run/debug/trace.jsonl")
        ...                                                        # doctest: +SKIP
        >>> with tracer.session():
        ...     model(input_ids)
        >>> summarize_trace("run/debug/trace.jsonl")               # doctest: +SKIP
    """

    def __init__(self, model: Any, config: DebugConfig, path: str | os.PathLike) -> None:
        self.model = model
        self.config = config
        self.path = str(path)

        self._handles: list[Any] = []
        # A deque rather than a list: the ring is trimmed once per event, and `list.pop(0)`
        # is linear in the window size. At the 10^5 events a generate document produces
        # that is the difference between a tracer that costs nothing and one that shows up
        # in the measurement it was brought in to explain.
        self._events: deque[dict[str, Any]] = deque(maxlen=config.tail or None)
        self._stream_handle: Any = None
        self._seq = 0
        self._call_id = 0
        self._forward_id = 0
        self._stack: list[_Frame] = []
        self._phases: list[str] = []
        self._samples: list[list[Any]] | None = None
        self._batch_rows: int | None = None
        self._pattern = re.compile(config.modules) if config.modules else None
        self._traced: list[tuple[str, Any]] = []
        self._cuda = False
        self._reader: Any = None
        self._written = False
        self.statistics: Any = None
        self._inspection: dict[str, Any] | None = None

    # -- lifetime ------------------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator["ModuleTracer"]:
        """Hook the model for a block of work, and always write the trace afterwards.

        An exception escaping the block is recorded - type, message, traceback, and the module stack it left behind - and then re-raised untouched.
        The trace is written either way: a run that succeeded still leaves the tail of what it did, which is what makes "it worked this time" checkable.
        """
        global _ACTIVE
        if _ACTIVE is not None:
            raise RuntimeError("a module tracer is already active in this process")

        import torch

        self._cuda = torch.cuda.is_available()
        self._reader = _build_memory_reader()
        _ACTIVE = self
        success = False
        try:
            # Inside the try, so that a failure to open the log - an unwritable path, a
            # full disk - does not leave the process with an active tracer nobody can
            # replace. Everything here happens before a single hook fires.
            self._open()
            self._write_header()
            if self.config.sample_stats:
                from .module_stats import ModuleStatistics

                config = self.config.manifest_entry()
                # Recorded in the session's `reduction` metadata instead of its config, so
                # sessions that differ only in chunk size keep identical config rows.
                chunk_elements = config.pop("chunk_elements")
                self.statistics = ModuleStatistics(self.path, config, chunk_elements)
            self.register_hooks()
            yield self
            success = True
        except BaseException as exc:      # noqa: BLE001 - recorded, then re-raised
            try:
                self._record_error(exc)
            except OSError as write_error:
                print(f"warning: could not record trace error: {write_error}", file=sys.stderr)
            raise
        finally:
            self.remove_hooks()
            try:
                try:
                    self._reconcile("session ended")
                finally:
                    self._close()
            except OSError as write_error:
                # Never mask the failure being traced with a failure to write about it.
                print(f"warning: could not write {self.path}: {write_error}", file=sys.stderr)
            finally:
                _ACTIVE = None
                if self.statistics is not None:
                    # An analysis failure must not replace the model's exception - nor
                    # become one of its own. `main.py` writes the scores *after* this
                    # context manager exits, so raising here would discard an evaluation
                    # that had already finished, over tables that are a side artifact and
                    # can be rebuilt from the observations still in the file.
                    try:
                        self.statistics.close(success=success)
                    except Exception as analysis_error:
                        print(f"warning: module statistics were not aggregated: "
                              f"{analysis_error}. The evaluation is unaffected; the "
                              f"observations remain in {self.statistics.path}",
                              file=sys.stderr)

    def register_hooks(self) -> None:
        """Hook every module the filter accepts.

        `named_modules()` de-duplicates: a module reachable by two attribute paths is hooked once, under the first name it is given.
        """
        if self._handles:
            return
        modules = dict(self.model.named_modules())
        targets = [(name or ROOT_NAME, module) for name, module in modules.items()
                   if self._pattern is None or self._pattern.search(name or ROOT_NAME)]
        if self.statistics is not None:
            def parent_type(path: str) -> str | None:
                """The class that called a module, which is what names its layout.

                A child that flattens batch and sequence looks like any other module
                on its own. A parent reachable only under a different name - the
                de-duplication above - leaves this unknown rather than guessed.
                """
                if path == ROOT_NAME:
                    return None
                parent = modules.get(path.rpartition(".")[0])
                return None if parent is None else type(parent).__name__

            self.statistics.resolve_modules(
                (path, type(module).__name__, parent_type(path)) for path, module in targets)
        # Root boundaries are bookkeeping, independent of the selected modules.
        self._handles.append(self.model.register_forward_pre_hook(self._root_enter, with_kwargs=True))
        for path, module in targets:
            self._traced.append((path, type(module).__name__))
            if self.statistics is not None:
                self.statistics.registration(path, "registering")
            try:
                self._handles.append(module.register_forward_pre_hook(
                    self._make_enter(path, type(module).__name__), with_kwargs=True))
                self._handles.append(module.register_forward_hook(self._make_exit(path)))
            except BaseException as error:
                if self.statistics is not None:
                    self.statistics.registration(path, "failed", f"{type(error).__name__}: {error}"[:2000])
                raise
            if self.statistics is not None:
                self.statistics.registration(path, "registered")
        self._handles.append(self.model.register_forward_hook(self._root_exit))

    def _root_enter(self, module, args, kwargs):
        self._reconcile("a new forward began")
        self._inspection = None
        self._begin_forward()
        if self.statistics is not None:
            self.statistics.begin(self._forward_id, args, kwargs)

    def _root_exit(self, module, args, output):
        if self.statistics is not None:
            self.statistics.finish_forward()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- hooks ---------------------------------------------------------------------

    def _make_enter(self, path: str, type_name: str):
        def enter(module, args, kwargs):  # noqa: ANN001 - torch signature
            if self.statistics is not None:
                self.statistics.called(path)
            if self.config.sync and self._cuda:
                self._synchronize()
            # Publish metadata before numerical inspection: the inspection itself
            # can OOM, and must not erase the module that was about to run.
            described = [_describe(a, False) for a in args]
            described_kwargs = {
                k: _describe(v, False) for k, v in kwargs.items()
            }
            frame = _Frame(path, type_name, self._next_call(), self._phase(), described)
            self._stack.append(frame)
            self._emit(
                {
                    "event": "enter",
                    "module": path,
                    "type": type_name,
                    "depth": len(self._stack) - 1,
                    "phase": frame.phase,
                    "forward": self._forward_id,
                    "call": frame.call_id,
                    "args": described,
                    "kwargs": described_kwargs,
                    "mem": self._memory(),
                }
            )
            if self.config.numeric:
                described[:] = [_describe(a, True) for a in args]
                described_kwargs.update({k: _describe(v, True) for k, v in kwargs.items()})
                self._emit({"event": "input_numeric", "module": path,
                            "forward": self._forward_id, "call": frame.call_id,
                            "args": described, "kwargs": described_kwargs})
            if self.statistics is not None:
                self._collect_statistics(path, "input", {"args": args, "kwargs": kwargs},
                                         frame.call_id, type_name)
            return None

        return enter

    def _make_exit(self, path: str):
        def exit_(module, args, output):  # noqa: ANN001 - torch signature
            if self.config.sync and self._cuda:
                self._synchronize()
            described = _describe(output, self.config.numeric)
            if self.statistics is not None:
                frame = self._stack[-1] if self._stack else None
                self._collect_statistics(path, "output", output,
                                         frame.call_id if frame else -1, type(module).__name__)
            frame = self._pop_to(path)
            self._emit(
                {
                    "event": "exit",
                    "module": path,
                    "type": frame.type_name if frame else None,
                    "depth": len(self._stack),
                    "phase": self._phase(),
                    "forward": self._forward_id,
                    "call": frame.call_id if frame else None,
                    "output": described,
                    # No `mem` here, unlike `enter`. In a sequential trace a module's exit
                    # state is the next event's entry state, so the second reading buys
                    # almost nothing and doubles the one cost that actually showed up in
                    # the measurement. The readings that matter - the failing module's
                    # entry, and the error itself - are both still taken.
                    **({} if frame else {"orphan": True}),
                }
            )
            if self.config.stop_on_nonfinite:
                self._check_finite(path, described)
            return None

        return exit_

    def _collect_statistics(self, path, io, value, call, type_name):
        """Keep an inspection failure distinguishable from a model failure."""
        self._inspection = {"module": path, "io": io}
        self.statistics.observe(path, io, value, call, self._phase(), type_name)
        self._inspection = None

    def _check_finite(self, path: str, described: dict[str, Any]) -> None:
        """Raise at the module that produced the first non-finite output.

        Known false positive: an architecture whose module *returns* an additive attention mask returns `-inf` on purpose. `--debug-modules` is how that gets scoped out.
        """
        for tensor in _tensor_descriptions(described):
            stats = tensor.get("stats")
            if stats and _is_nonfinite(stats):
                raise NonFiniteOutput(
                    f"{path} produced a non-finite output: "
                    f"{stats['nan']} NaN, {stats['posinf']} +Inf, {stats['neginf']} -Inf "
                    f"in a {tensor['shape']} {tensor['dtype']} tensor",
                    module=path,
                )

    def _synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()

    # -- bookkeeping ---------------------------------------------------------------

    def _phase(self) -> str:
        return self._phases[-1] if self._phases else DEFAULT_PHASE

    def _next_call(self) -> int:
        self._call_id += 1
        return self._call_id

    def _begin_forward(self) -> None:
        """Start a new forward and say what it is for.

        Not called for work in `signal_collection`: the logit lens re-enters the model's own final norm and head, and counting that as a new forward would make the trace disagree with the number of passes the model actually ran.
        """
        self._forward_id += 1
        self._emit(
            {
                "event": "forward",
                "forward": self._forward_id,
                "phase": self._phase(),
                "samples": self._samples,
                "batch_rows": self._batch_rows,
            }
        )

    def set_samples(self, contexts: Iterable[Any]) -> None:
        """Record which documents the following forwards cover, and reconcile the stack.

        Sticky: a generate document runs one forward per decoded token and they all belong to it, so this is replaced rather than consumed.
        """
        contexts = list(contexts)
        if self.statistics is not None:
            self.statistics.set_samples(contexts)
        rows: list[list[Any]] = []
        widest = -1
        for ctx in contexts:
            rows.append(
                [
                    getattr(ctx, "task_name", None),
                    getattr(ctx, "doc_id", None),
                    getattr(ctx, "choice_idx", None),
                    getattr(ctx, "batch_row", 0),
                ]
            )
            widest = max(widest, int(getattr(ctx, "batch_row", 0) or 0))
        self._samples = rows or None
        self._batch_rows = widest + 1 if rows else None
        self._reconcile("a new batch was declared")

    def _pop_to(self, path: str) -> _Frame | None:
        """Pop the frame for `path`, recording anything above it as never having returned.

        Frames above it exist when an exception unwound inner modules and something between here and there caught it.
        """
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index].path == path:
                if index < len(self._stack) - 1:
                    self._unwind(self._stack[index + 1 :], "an inner module did not return")
                    del self._stack[index + 1 :]
                return self._stack.pop()
        return None

    def _reconcile(self, reason: str) -> None:
        """Record and clear whatever is still on the stack."""
        if not self._stack:
            return
        self._unwind(list(self._stack), reason)
        self._stack.clear()

    def _unwind(self, frames: Sequence[_Frame], reason: str) -> None:
        """Write one `unwound` event per frame that was entered and never exited.

        The deepest is the failure site: execution stopped inside it. Its ancestors are the path that led there, which is why they are recorded too but only one carries `deepest`.
        """
        for position, frame in enumerate(frames):
            self._emit(
                {
                    "event": "unwound",
                    "module": frame.path,
                    "type": frame.type_name,
                    "depth": len(self._stack) - len(frames) + position,
                    "phase": frame.phase,
                    "forward": self._forward_id,
                    "call": frame.call_id,
                    "args": frame.inputs,
                    "deepest": position == len(frames) - 1,
                    "reason": reason,
                }
            )

    def _record_error(self, exc: BaseException) -> None:
        """Record an exception on its way out of the session.

        The phase comes from the deepest frame still on the stack, not from `_phase()`: by the time this runs the exception has already unwound through `phase()`'s own `finally`, so the live phase stack is empty and would report every failure as `other`. Each frame kept the phase it was entered under, which is the one that describes the work that died.
        """
        self._emit(
            {
                "event": "error",
                "exc_type": type(exc).__name__,
                "message": str(exc)[:2000],
                "phase": ("module_statistics" if self._inspection else
                          self._stack[-1].phase if self._stack else self._phase()),
                "module": (self._inspection["module"] if self._inspection else
                           self._stack[-1].path if self._stack else None),
                "inspection": self._inspection,
                # Only `--debug-stop-on-nonfinite` sets this, and it is the module whose
                # output was bad rather than the one execution stopped inside.
                "detected_at": getattr(exc, "module", None),
                "forward": self._forward_id,
                "samples": self._samples,
                "batch_rows": self._batch_rows,
                "mem": self._memory(),
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)[-12:],
            }
        )

    def _memory(self) -> dict[str, int] | None:
        """The allocator's own counters, read through whichever path this torch makes cheap.

        `reserved` is what the allocator holds from the driver and `allocated` is what tensors are using; an OOM with a wide gap between them is fragmentation rather than genuine exhaustion, which is the one thing an OOM traceback never says.

        Neither counter queues a kernel or synchronizes - they are host-side numbers the caching allocator already maintains. That made them look free, and they are not: measured on an RTX PRO 4500, `torch.cuda.memory_allocated()` costs 69us, because it builds the *entire* ~60-entry statistics dict and then returns one value from it. Three of those per event, two events per module call, 547 modules per forward came to 420us of pure bookkeeping per module - about 90 seconds over a 200-document run, which was most of the tracer's overhead.

        `memory_stats_as_nested_dict()` is the same data before the flattening, at 8.6us, and one call yields all three numbers. `_reader` is resolved once per session and falls back to the flattened path if a future torch drops it.
        """
        return self._reader() if self._reader is not None else None

    # -- output --------------------------------------------------------------------

    def _emit(self, event: dict[str, Any]) -> None:
        self._seq += 1
        event["i"] = self._seq
        if self._stream_handle is not None:
            self._stream_handle.write(json.dumps(_jsonable(event), allow_nan=False) + "\n")
            self._stream_handle.flush()
            return
        self._events.append(event)

    def _open(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        if not self.config.tail:
            self._stream_handle = open(self.path, "w")

    def _write_header(self) -> None:
        self._emit(
            {
                "event": "header",
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": self.config.manifest_entry(),
                "model_type": getattr(getattr(self.model, "config", None), "model_type", None),
                "torch": self._torch_version(),
                "cuda": self._cuda,
            }
        )

    @staticmethod
    def _torch_version() -> str:
        import torch

        return torch.__version__

    def _close(self) -> None:
        """Write the buffer, or close the stream.

        Called from the `finally` of `session()`, so it runs after an exception has been recorded and before it is re-raised.
        Nothing here touches the device: after an OOM the allocator is the one thing not to ask for more, and after a device-side fault it would not answer.
        """
        if self._written:
            return
        self._written = True
        if self._stream_handle is not None:
            self._stream_handle.close()
            self._stream_handle = None
            return
        with open(self.path, "w") as handle:
            for event in self._events:
                handle.write(json.dumps(_jsonable(event), allow_nan=False) + "\n")
            dropped = self._seq - len(self._events)
            if dropped:
                handle.write(
                    json.dumps({"event": "dropped", "count": dropped, "i": self._seq + 1}) + "\n"
                )

    # -- what was traced -----------------------------------------------------------

    @property
    def traced_modules(self) -> list[tuple[str, str]]:
        """`(path, class name)` for every hooked module, in `named_modules()` order."""
        return list(self._traced)

    @property
    def events(self) -> list[dict[str, Any]]:
        """The buffered events. Empty unless `--debug-tail` is holding them."""
        return list(self._events)

    @property
    def bytes_written(self) -> int:
        """Size of the trace on disk, so a long run's file is not a surprise."""
        return os.path.getsize(self.path) if os.path.exists(self.path) else 0


# --------------------------------------------------------------------------
# Reading a trace back
# --------------------------------------------------------------------------


def trace_path(run_dir: str | os.PathLike, name: str = "trace") -> str:
    """Where a run's trace lives.

    Deliberately outside the parquet tables: `ShardWriter` buffers rows and writes a shard only every `SHARD_SIZE` documents, so a crash discards exactly the rows a post-mortem needs.

    A run has one trace per pass, because the passes are different programs. The first scores every document; the second re-runs a sample with eager attention and dumps `(heads, seq, seq)` maps, which is the heaviest thing this tool does and the likeliest place for it to run out of memory. One file holding both would have the interesting half of a two-hour run scrolled out of the ring buffer by the cheap half.
    """
    return os.path.join(str(run_dir), "debug", f"{name}.jsonl")


def list_traces(run_dir: str | os.PathLike) -> list[str]:
    """Every trace file a run directory holds, scoring pass first."""
    directory = os.path.join(str(run_dir), "debug")
    if not os.path.isdir(directory):
        return []
    found = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".jsonl")
    )
    return sorted(found, key=lambda path: not path.endswith("trace.jsonl"))


def read_trace(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a trace file, or the trace inside a run directory.

    Example:
        >>> events = read_trace("results/.../run")            # doctest: +SKIP
        >>> [e["module"] for e in events if e.get("deepest")]
        ['model.layers.3.self_attn']
    """
    given = str(path)
    if os.path.isdir(given):
        found = list_traces(given)
        if not found:
            raise FileNotFoundError(
                f"no trace under {os.path.join(given, 'debug')}. A run only writes one "
                "when it was given --debug; a run that died before its directory existed "
                "has none at all."
            )
        path = found[0]
    else:
        path = given
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trace at {path}")
    with open(path) as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    # Metadata is durable before numerical inspection starts. For readers, enrich
    # the matching entry when inspection succeeded; a failed inspection has none.
    entries = {e.get("call"): e for e in events if e.get("event") == "enter"}
    for event in events:
        if event.get("event") == "input_numeric" and event["call"] in entries:
            entries[event["call"]].update(args=event["args"], kwargs=event["kwargs"])
    return events


def _format_tensor(described: dict[str, Any]) -> str:
    """The failure summary's rendering: the same statistics, spelled out a little wider.

    Shares `_stats_text` with the per-module listing rather than formatting its own. The
    two used to diverge, and the drift was not cosmetic: this one read `stats["nan"]`
    unconditionally, which an integer tensor does not carry, so a run that died with token
    ids on the stack would have crashed while printing the report about it.
    """
    if "shape" in described:
        text = f"{described['shape']} {described['dtype']} {described['device']}"
        stats = _stats_text(described.get("stats") or {})
        return f"{text}  {stats}" if stats else text
    if "len" in described:
        return f"{described['type']}[{described['len']}]"
    return described.get("type", "?")


def _format_args(event: dict[str, Any], indent: str) -> list[str]:
    lines = []
    for described in event.get("args") or []:
        lines.append(f"{indent}in   {_format_tensor(described)}")
    for name, described in (event.get("kwargs") or {}).items():
        lines.append(f"{indent}in   {name}={_format_tensor(described)}")
    return lines


def _format_samples(event: dict[str, Any]) -> str:
    rows = event.get("samples")
    if not rows:
        return "none recorded (a batch-size probe, or a document already on disk)"
    by_doc: dict[tuple[Any, Any], list[Any]] = {}
    for task, doc_id, choice, _row in rows:
        by_doc.setdefault((task, doc_id), []).append(choice)
    parts = [f"{task}#{doc}x{len(choices)}" for (task, doc), choices in by_doc.items()]
    shown = ", ".join(parts[:8])
    return shown + (f", +{len(parts) - 8} more" if len(parts) > 8 else "")


def _stats_text(stats: dict[str, Any]) -> str:
    """The numeric summary of one tensor as a line, or empty when there is none.

    One function for both renderings. Every field is optional: an integer tensor carries no
    non-finite counts because it cannot hold any, an empty tensor carries nothing, and a
    tensor with no finite values carries the counts but no distribution.
    """
    if not stats or stats.get("empty"):
        return "(empty)" if stats.get("empty") else ""
    stats = {k: float(v) if isinstance(v, str) and v in {"nan", "inf", "-inf"} else v
             for k, v in stats.items()}
    parts = []
    if stats.get("nan") or stats.get("posinf") or stats.get("neginf"):
        parts.append(
            f"!nan={stats.get('nan', 0)}/+inf={stats.get('posinf', 0)}"
            f"/-inf={stats.get('neginf', 0)}"
        )
    if "min" in stats:
        parts.append(f"min={stats['min']:.3g} max={stats['max']:.3g}")
    if "mean" in stats:
        parts.append(
            f"mean={stats['mean']:.3g} sd={stats['std']:.3g} med={stats['median']:.3g}"
        )
    return " ".join(parts)


def _io_summary(described: dict[str, Any]) -> str:
    """One tensor, container or cache rendered short enough to sit in a column."""
    if described is None:
        return "-"
    if "shape" in described:
        text = "x".join(str(n) for n in described["shape"])
        if described.get("type"):                      # a cache carries both
            text = f"{described['type']}[{described.get('layers', '?')}] {text}"
        stats = _stats_text(described.get("stats") or {})
        if stats:
            text += f" {stats}"
        if "bytes" in described:
            text += f" {described['bytes'] / 1e6:.1f}MB"
        return text
    if "len" in described:
        return f"{described['type']}[{described['len']}]"
    if "value" in described:
        # A scalar's value is the point of it: `use_cache=True` and `use_cache=False` are
        # different runs, and rendering both as "bool" loses the only bit it carries.
        return "None" if described["value"] is None else repr(described["value"])
    return str(described.get("type", "?"))


def _io_columns(event: dict[str, Any]) -> str:
    """The inputs of one module entry, positional first then keyword, named."""
    parts = [_io_summary(d) for d in (event.get("args") or [])]
    # `None` keyword arguments are the architecture's defaults and there are many of them;
    # printing `attention_mask=None, position_ids=None, inputs_embeds=None, ...` on every
    # line would hide the two that carry a shape.
    parts += [
        f"{name}={_io_summary(d)}"
        for name, d in (event.get("kwargs") or {}).items()
        if d.get("type") != "NoneType"
    ]
    return ", ".join(parts) or "-"


def describe_forward(
    events: Sequence[dict[str, Any]], forward_id: int, module: str | None = None
) -> str:
    """Every module one forward pass ran, in call order, with what went in and out.

    This is the view the feature was asked for and the one a summary cannot give: not
    "where did it stop" but "what did each module see". It is deliberately one line per
    module - a 32-layer model runs several hundred of them per forward - with `module` as
    the way to narrow to the handful worth reading.

    Entries are paired with their exits by call id, so a module that never returned shows
    its inputs and `(never returned)` where its output would be.
    """
    pattern = re.compile(module) if module else None
    header = next(
        (e for e in events if e.get("event") == "forward" and e["forward"] == forward_id),
        None,
    )
    outputs = {
        e["call"]: e.get("output")
        for e in events
        if e.get("event") == "exit" and e.get("call") is not None
    }
    entries = [
        e for e in events
        if e.get("event") in {"enter", "unwound"} and e.get("forward") == forward_id
    ]
    if not entries:
        return f"forward {forward_id}: nothing recorded (the ring buffer may have dropped it)"

    lines = [
        f"forward {forward_id}   phase {entries[0].get('phase')}   "
        f"documents: {_format_samples(header) if header else 'unknown'}"
        + (f"   {header['batch_rows']} batch row(s)" if header and header.get("batch_rows") else "")
    ]
    if pattern is not None:
        matched = [e["module"] for e in entries if pattern.search(e["module"])]
        depths = sorted({m.split(".")[2] for m in matched if m.startswith("model.layers.")},
                        key=lambda n: int(n) if n.isdigit() else -1)
        note = f"modules matching {module!r} only: {len(matched)} of {len(entries)}"
        if len(depths) > 1:
            # An unescaped dot is why this line exists. `layers.1.` matches layers 1 and
            # 10 through 19, because the trailing `.` happily matches the `0` of `10`;
            # `layers\.1\.` matches one. The difference is invisible at layer 0, which
            # is the layer everyone tries first, and otherwise shows up only as a longer
            # listing than expected.
            note += f", across layers {', '.join(depths)}"
        lines.append(note)
    if any("stats" in d for e in entries for d in (e.get("args") or [])):
        # Say what the notation means in the output that uses it. A reader who has to ask
        # is reading a number they cannot act on.
        lines.append(
            "  shapes are AxBxC. min/max/mean/sd/med are over the finite values only, so "
            "an Inf\n  cannot swallow the mean; where a tensor holds NaN or Inf the counts "
            "are shown too.\n  A median far from the mean means a few outliers are "
            "carrying the tensor."
        )
    lines.append("")

    shown = 0
    for event in entries:
        if pattern is not None and not pattern.search(event["module"]):
            continue
        shown += 1
        indent = "  " * event["depth"]
        name = f"{indent}{event['module']}"
        returned = (
            "(never returned)" if event["event"] == "unwound"
            else _io_summary(outputs.get(event.get("call")))
        )
        lines.append(f"  {name:<46} {event['type']:<22} {_io_columns(event)}")
        lines.append(f"  {'':<46} {'':<22} -> {returned}")
    if not shown:
        lines.append(f"  no module in this forward matches {module!r}")
    return "\n".join(lines)


def forward_index(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """What each recorded forward covers: its phase, its documents, its module count."""
    counts: dict[int, int] = {}
    phases: dict[int, str] = {}
    for event in events:
        if event.get("event") == "enter":
            counts[event["forward"]] = counts.get(event["forward"], 0) + 1
            phases.setdefault(event["forward"], event.get("phase", DEFAULT_PHASE))
    headers = {
        e["forward"]: e for e in events if e.get("event") == "forward"
    }
    return [
        {
            "forward": number,
            "modules": counts[number],
            "phase": phases.get(number),
            "samples": headers.get(number, {}).get("samples"),
            "batch_rows": headers.get(number, {}).get("batch_rows"),
        }
        for number in sorted(counts)
    ]


def summarize_trace(path: str | os.PathLike, tail: int = 10) -> str:
    """Turn a trace into the paragraph a person actually wanted.

    A raw JSONL of thousands of events is storage, not a debugging tool. This answers the three questions the trace exists for: how far it got, where it stopped, and on whose behalf.

    Example:
        >>> print(summarize_trace("results/.../run"))          # doctest: +SKIP
    """
    events = read_trace(path)
    if not events:
        return f"{path}: empty trace"

    header = events[0] if events[0].get("event") == "header" else {}
    dropped = next((e["count"] for e in events if e.get("event") == "dropped"), 0)
    error = next((e for e in events if e.get("event") == "error"), None)
    unwound = [e for e in events if e.get("event") == "unwound"]
    forwards = {e["forward"] for e in events if e.get("event") == "enter"}

    config = header.get("config", {})
    lines = [
        f"trace: {path}   ({len(events)} events"
        + (f", {dropped} dropped" if dropped else "")
        + ")",
        f"model_type: {header.get('model_type')}   torch {header.get('torch')}   "
        f"cuda={header.get('cuda')}   numeric={config.get('numeric')}   sync={config.get('sync')}",
    ]
    if dropped:
        # Worth saying in forwards rather than events. A 547-module model spends about
        # 1100 events on one forward, so the default ring is four or five of them - which
        # is ample for locating a failure and nothing like enough to watch a trend. On a
        # generate task, where one forward is one token, that is the last few tokens.
        lines.append(
            f"this is the tail: the last {len(forwards)} forward"
            f"{'s' if len(forwards) != 1 else ''} of {dropped + len(events)} events. "
            "--debug-stream keeps all of it."
        )
    else:
        lines.append(f"forwards traced: {len(forwards)}")
    lines.append("")

    if error is None and not unwound:
        lines.extend(_format_healthy_trace(path, events))
    else:
        lines.extend(_format_failed_trace(events, error, unwound, tail))
    return "\n".join(lines)


def _format_healthy_trace(path: str | os.PathLike, events: list[dict[str, Any]]) -> list[str]:
    """실패가 없는 trace의 forward 목록과 상세 조회 명령을 출력 줄로 만든다."""
    lines: list[str] = []
    # A healthy trace is the normal case, not a dead end. The log holds every module's
    # inputs and outputs; stopping at "nothing went wrong" would leave the feature
    # unable to answer the question it was built for - what did each module see -
    # except when something had already crashed.
    lines.append("no failure recorded: every module that was entered also returned.")
    lines.append("")
    index = forward_index(events)
    lines.append(f"{len(index)} forward(s) in this trace:")
    for entry in index[:20]:
        lines.append(
            f"  forward {entry['forward']:<5} {entry['modules']:>5} modules   "
            f"{entry['phase']:<18} {_format_samples(entry)}"
        )
    if len(index) > 20:
        lines.append(f"  ... and {len(index) - 20} more")
    first = index[0]["forward"] if index else 1
    lines.append("")
    lines.append("read one of them:")
    lines.append(f"  evalmetry debug {path} --forward {first}")
    lines.append(f"  evalmetry debug {path} --forward {first} "
                 "--module 'layers\\.0\\.'")
    return lines


def _format_failed_trace(
    events: list[dict[str, Any]], error: dict[str, Any] | None,
    unwound: list[dict[str, Any]], tail: int,
) -> list[str]:
    """오류·미반환 모듈·직전 이벤트를 출력한다. error 또는 unwound가 있어야 한다.

    error가 있으면 그 시점을, 없으면 첫 unwound 이벤트를 경계로 사용한다.
    이벤트를 다시 정렬하지 않아 trace가 기록한 호출 순서를 유지한다.
    """
    lines: list[str] = []
    if error is not None:
        lines.append(
            f"FAILED in phase {error['phase']}, forward {error['forward']}: "
            f"{error['exc_type']}: {error['message'].splitlines()[0][:160]}"
        )
        if error.get("detected_at"):
            lines.append(
                f"  the bad output came from {error['detected_at']}, which had already "
                "returned; the stack below is its caller"
            )
        lines.append(f"  documents in flight: {_format_samples(error)}")
        memory = error.get("mem")
        if memory:
            lines.append(
                f"  cuda: {memory['allocated'] / 1e9:.2f} GB allocated, "
                f"{memory['reserved'] / 1e9:.2f} GB reserved"
            )
        lines.append("")

    if unwound:
        lines.append("entered and never returned, outermost first:")
        for event in unwound:
            marker = "   <- stopped here" if event.get("deepest") else ""
            lines.append(
                f"  {'  ' * event['depth']}{event['module']:<40} {event['type']}{marker}"
            )
            if event.get("deepest"):
                lines.extend(_format_args(event, "  " + "  " * event["depth"] + "  "))
        lines.append("")

    boundary = error["i"] if error is not None else unwound[0]["i"]
    before = [e for e in events if e["i"] < boundary and e.get("event") in {"enter", "exit"}]
    if before:
        lines.append(f"last {min(tail, len(before))} module events before that:")
        for event in before[-tail:]:
            lines.append(
                f"  {event['i']:>6}  {event['event']:<5} {event['module']:<40} "
                f"{event.get('phase', '')}"
            )
    return lines
