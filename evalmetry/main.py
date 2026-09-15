"""Entry point: argument parsing, run configuration, and the run/collection/report commands.

All arguments live here, in one place, so that adding one means editing one file.
Model loading arguments are passed straight through to lm-eval in its own format; we do not redefine them.

Commands:

    run                    evaluate a model and record selected signals
    collect-research-data  collect additional data from a saved evaluation
    report                 compare runs and select documents for inspection
    debug                  inspect execution traces
    module-stats           inspect per-document and dataset module statistics

There is no sweep runner and no scheduler; repeated runs are a shell loop.
A `run` into a directory that already holds shards does resume, in the narrow sense that documents already recorded are not recorded again - shards are write-once, so an interrupted run leaves valid data that a restart must add to rather than duplicate or overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import dataclasses
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Sequence

from . import debug, storage
from .debug import TRACE_BUFFER_EVENTS
from .storage import FIXED_SETTINGS, SAMPLING_SEED, SCHEMA_VERSION

#: Version of this tool, recorded in every manifest.
TOOL_VERSION = "0.1.0"

#: Default number of documents collected per correctness group.
DEFAULT_COLLECT_LIMIT = 500


# --------------------------------------------------------------------------
# Argument helpers
# --------------------------------------------------------------------------


def parse_hidden_layers(spec: str, n_residual: int) -> list[int]:
    """Parse `--hidden-layers`: `all`, a list (`0,1,2`), or ranges (`0-8`).

    Duplicates are removed and the result is sorted.
    An index outside `0..n_residual-1` fails here rather than being clipped: a silently clipped index means the run stores a different set of layers than was asked for, and nothing downstream would show that.

    Args:
        n_residual: L+1, the number of residual entries.

    Returns:
        Ascending, deduplicated absolute indices.

    Example:
        >>> parse_hidden_layers("all", 5)
        [0, 1, 2, 3, 4]
        >>> parse_hidden_layers("0-2,4,4", 5)
        [0, 1, 2, 4]
        >>> parse_hidden_layers("-1", 5)
        [4]
        >>> parse_hidden_layers("9", 5)
        Traceback (most recent call last):
        ValueError: hidden layer index 9 is outside 0..4
    """
    if spec.strip().lower() == "all":
        return list(range(n_residual))

    chosen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", part)
        if match:
            start, stop = int(match.group(1)), int(match.group(2))
            chosen.update(range(start, stop + 1))
        else:
            chosen.add(int(part))

    resolved: set[int] = set()
    for index in chosen:
        # A negative index is recorded in the manifest as the absolute one it resolves to, so the run stays readable without knowing L.
        absolute = index + n_residual if index < 0 else index
        if not 0 <= absolute < n_residual:
            raise ValueError(f"hidden layer index {index} is outside 0..{n_residual - 1}")
        resolved.add(absolute)
    return sorted(resolved)


def parse_batch_size(value: str) -> int | str:
    """`--batch-size`: a positive integer, or "auto".

    "auto" is lm-eval's own probe for the largest batch that fits, and `backend.py` already mirrors its handling because that code is part of the copied `_loglikelihood_tokens`.
    Accepting the string here is what makes that branch reachable; without it the mirrored code could never run.

    Example:
        >>> parse_batch_size("8"), parse_batch_size("auto")
        (8, 'auto')
        >>> parse_batch_size("0")
        Traceback (most recent call last):
        ValueError: --batch-size must be a positive integer or "auto", got '0'
    """
    if value.strip().lower() == "auto":
        return "auto"
    try:
        size = int(value)
    except ValueError:
        size = 0
    if size < 1:
        raise ValueError(f'--batch-size must be a positive integer or "auto", got {value!r}')
    return size


def model_slug(model_id: str) -> str:
    """Filesystem-safe short name for a model id.

    Example:
        >>> model_slug("Qwen/Qwen3-8B")
        'Qwen__Qwen3-8B'
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id.replace("/", "__"))


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Everything that defines one run, plus the hash that names its directory.

    Attributes:
        model_args: passed to lm-eval verbatim (`pretrained=...,dtype=...`).
        tasks: task names.
        include_path: directories containing local benchmark YAML, JSONL and scoring code.
            Paths inside YAML resolve relative to that YAML. See examples/custom_benchmarks.
        save_attention / save_hidden: opt-in signals, both collected in a second pass because correctness is not known during the first one.
        debug: module tracing, off by default.
            Absent from `identity` and `config_hash`: it changes how long a run takes, and with `sync` or `stop_on_nonfinite` whether it finishes, but not what the signals mean.
    """

    model_args: str
    tasks: list[str]
    num_fewshot: int = 0
    limit: int | None = None
    batch_size: int | str = 1
    adapter: str | None = None
    save_attention: bool = False
    save_hidden: bool = False
    hidden_layers: str = "all"
    collect_limit: int = DEFAULT_COLLECT_LIMIT
    output: str | None = None
    include_path: list[str] = dataclasses.field(default_factory=list)
    model_factory: str | None = None
    model_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    model_bundle: Any = dataclasses.field(default=None, repr=False)
    resolved_bundle: Any = dataclasses.field(default=None, init=False, repr=False)
    model_provenance: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)
    signals: tuple[str, ...] = ("logit_lens", "similarity")
    system_instruction: str | None = None
    apply_chat_template: bool | str = False
    hook_factory: str | None = None
    hook_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    hooks: list[Any] = dataclasses.field(default_factory=list)
    resolved_hooks: list[Any] = dataclasses.field(default_factory=list, init=False, repr=False)
    hook_provenance: list[dict[str, Any]] = dataclasses.field(default_factory=list, init=False)
    # 이름 -> selector가 실제로 고른 경로. 모델이 있어야 알 수 있으므로 실행 식별에는
    # 넣지 않고 manifest에만 남긴다. 식별에 들어가는 것은 selector 쪽이다.
    hook_resolved: dict[str, list[str]] | None = dataclasses.field(default=None, init=False)
    benchmark_provenance: dict[str, Any] = dataclasses.field(default_factory=dict, init=False)
    debug: "debug.DebugConfig" = dataclasses.field(default_factory=lambda: debug.DebugConfig())

    def model_kwargs(self) -> dict[str, str]:
        """The lm-eval model argument string parsed into a dict.

        Example:
            >>> RunConfig("pretrained=Qwen/Qwen3-8B,dtype=bfloat16", []).model_kwargs()
            {'pretrained': 'Qwen/Qwen3-8B', 'dtype': 'bfloat16'}
        """
        parsed: dict[str, str] = {}
        for part in self.model_args.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key.strip()] = value.strip()
        return parsed

    @property
    def model_id(self) -> str:
        return self.model_provenance.get("model_id") or self.model_kwargs().get("pretrained", "unknown-model")

    @property
    def revision(self) -> str:
        return self.model_kwargs().get("revision", "main")

    def identity(
        self, reducer_descriptors: Sequence[dict[str, Any]], lm_eval_version: str
    ) -> dict[str, Any]:
        """What has to match for two runs to belong in the same directory.

        The seed is deliberately absent: two runs differing only in seed are the same configuration.
        Everything else here changes what the signals mean, so mixing them in one directory would produce a run that misdescribes itself.

        Example:
            >>> config.identity(descriptors, "0.4.9.1")["model_id"]   # doctest: +SKIP
            'Qwen/Qwen3-8B'
        """
        identity = {
            "model_id": self.model_id,
            "revision": self.revision,
            "tasks": sorted(self.tasks),
            "num_fewshot": self.num_fewshot,
            "limit": self.limit,
            "reducers": reducer_descriptors,
            "lm_eval_version": lm_eval_version,
        }
        # 사용하지 않은 확장 기능은 키 자체를 생략한다. 빈 값이라도 추가하면
        # 기존 실행과 identity/hash가 달라져 resume과 결과 비교에 영향을 준다.
        if self.model_provenance:
            identity["custom_model"] = self.model_provenance
            identity["custom_model_batch_size"] = self.batch_size
        if self.system_instruction is not None or self.apply_chat_template:
            identity["prompt_protocol"] = {
                "system_instruction": self.system_instruction,
                "apply_chat_template": self.apply_chat_template,
            }
        if self.benchmark_provenance:
            identity["benchmarks"] = self.benchmark_provenance
        if self.hook_provenance:
            identity["hooks"] = {
                "factory": self.hook_factory,
                "config": self.hook_config,
                "specs": self.hook_provenance,
            }
        return identity

    def config_hash(self, reducer_descriptors: Sequence[dict[str, Any]], lm_eval_version: str) -> str:
        """Short hash naming this configuration.

        The seed is deliberately not an input: two runs that differ only in seed are the same configuration.
        Reducer versions are, because they change what the numbers mean.

        Example:
            >>> config.config_hash(descriptors, "0.4.9.1")        # doctest: +SKIP
            'ab12cd34'
        """
        payload = self.identity(reducer_descriptors, lm_eval_version)
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:8]

    def default_output(self, config_hash: str) -> str:
        """`results/{task}/{model_slug}/{date}-{config_hash}`.

        Only a convenience for browsing; `report` reads the manifest and never parses this path.

        Example:
            >>> RunConfig("pretrained=Qwen/Qwen3-8B", ["xnli_ko"]).default_output("ab12cd34")
            'results/xnli_ko/Qwen__Qwen3-8B/2026-01-31-ab12cd34'   # date varies
        """
        task = "+".join(sorted(self.tasks)) or "unknown-task"
        today = date.today().isoformat()
        return os.path.join("results", task, model_slug(self.model_id), f"{today}-{config_hash}")


# --------------------------------------------------------------------------
# Environment checks
# --------------------------------------------------------------------------


def validate_single_gpu_execution(config: RunConfig) -> None:
    """Reject lm-eval parallelism before tasks, models, or output paths are touched.

    Accelerate 1.14 enters its GPU distributed path when ``LOCAL_RANK`` is set
    to a non-negative integer.  ``torchrun`` supplies that variable too.  A
    stray ``WORLD_SIZE`` alone is deliberately not enough: schedulers and
    parent shells can leave it set for an otherwise ordinary process.
    """
    import torch

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    try:
        launched_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    except ValueError:
        launched_rank = -1
    if distributed or launched_rank >= 0:
        detail = "an initialized torch process group" if distributed else f"LOCAL_RANK={launched_rank}"
        raise RuntimeError(
            "only a single process on one CUDA GPU is supported; distributed/data-parallel "
            f"execution was detected from {detail}. Do not use `accelerate launch`, "
            "`torchrun`, or another distributed launcher; run `evalmetry run` "
            "directly with one visible GPU."
        )

    options = config.model_kwargs()
    rejected: list[str] = []
    if "parallelize" in options and options["parallelize"].strip().lower() not in {
        "false", "0", "none", "null", ""
    }:
        rejected.append("parallelize")
    for name in ("tp_plan", "device_map", "device_mesh", "tp_size", "tensor_parallel_size"):
        if name in options and options[name].strip().lower() not in {"none", "null", ""}:
            rejected.append(name)
    if rejected:
        rendered = ", ".join(f"`{name}`" for name in rejected)
        raise ValueError(
            "only a single process on one CUDA GPU is supported; model sharding and tensor "
            f"parallelism are unsupported, but {rendered} was set. Remove these model args "
            "and load the complete model on one GPU."
        )


def _short(value: Any, width: int = 90) -> str:
    """Render one identity field compactly enough to read in an error.

    The reducer list is the reason this exists: printed in full it is several hundred characters of configuration, which buries the field that actually differs.
    """
    if isinstance(value, list) and value and isinstance(value[0], dict) and "name" in value[0]:
        return "[" + ", ".join(f"{item['name']} v{item.get('version')}" for item in value) + "]"
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def check_resume_is_the_same_run(run_dir: str, identity: dict[str, Any]) -> None:
    """Refuse to resume into a directory that belongs to a different run.

    Resuming skips documents already recorded, which is only correct if the directory holds *this* run.
    Nothing about a run directory forces that: pointing a second model at it produced a directory whose manifest named one model while its `signals` held another's layers, and both halves looked entirely valid on their own.

    Raises:
        ValueError: naming the fields that differ, so the fix is obvious - a typo in --model-args, or an --output that should have been new.
    """
    try:
        manifest = storage.read_manifest(run_dir)
    except (OSError, ValueError, KeyError) as error:
        if (not isinstance(error, FileNotFoundError)
                and os.path.isfile(os.path.join(run_dir, storage.RESULTS_FILENAME))):
            # A damaged manifest is not an empty directory: resuming would overwrite the only record of the run.
            raise ValueError(f"{run_dir}: results.json exists but cannot be read ({error}); "
                             "restore it or use a new --output") from error
        if (identity.get("benchmarks") or identity.get("hooks") or identity.get("custom_model")) and os.path.isdir(run_dir) and os.listdir(run_dir):
            raise ValueError("custom extension resume requires an intact provenance manifest")
        return  # nothing recorded here yet, or not a run directory at all
    previous = manifest.get("config_identity")
    if not previous:
        if (identity.get("benchmarks") or identity.get("hooks") or identity.get("custom_model")):
            raise ValueError("custom extension resume requires provenance")
        return  # written before this check existed; nothing to compare against

    if identity.get("custom_model") and not identity["custom_model"].get("resume_safe"):
        raise ValueError("custom model is not reproducibly identified; automatic resume is disabled")
    if any(not h["resume_safe"] for h in identity.get("hooks", {}).get("specs", [])):
        raise ValueError("custom hook source is unavailable; automatic resume is disabled")
    differing = {
        key: (previous.get(key), identity.get(key))
        for key in set(previous) | set(identity)
        if previous.get(key) != identity.get(key)
    }
    if not differing:
        return
    lines = "\n".join(
        f"    {key}: existing run has {_short(old)}, this one has {_short(new)}"
        for key, (old, new) in sorted(differing.items())
    )
    raise ValueError(
        f"{run_dir} already holds a different run, so resuming would mix them:\n"
        f"{lines}\n"
        "  Use a new --output, or correct the arguments to match the existing run."
    )


def check_environment(model: Any) -> dict[str, Any]:
    """Refuse unsupported device setups and describe the accepted environment.

    * multi-GPU sharding is rejected: layers would sit on different devices, and the similarity reducer would pay a cross-device transfer per layer pair;
    * CPU and MPS are not supported;
    * fp16 and bf16 are the verified dtypes.
      Quantized models are allowed but not guaranteed - the point at which weights are dequantized differs per kernel, so what the hooks see differs too.
      `check_decode_identity()` is the practical test for whether a given quantized model is usable.

    Returns:
        A dict describing the device setup, for the manifest.
    """
    import torch

    devices = {str(p.device) for p in model.parameters()}
    cuda_devices = {d for d in devices if d.startswith("cuda")}
    if not cuda_devices:
        raise RuntimeError(
            f"CUDA is required; model parameters are on {sorted(devices)}. "
            "CPU and MPS are not supported."
        )
    if len(cuda_devices) > 1:
        raise RuntimeError(
            f"model is sharded across {sorted(cuda_devices)}. Multi-GPU sharding is not "
            "supported: the similarity reducer would move tensors between devices for "
            "every layer pair. Load the model on a single GPU."
        )

    dtypes = {str(p.dtype) for p in model.parameters()}
    verified = {"torch.float16", "torch.bfloat16"}
    warnings: list[str] = []
    if not dtypes & verified:
        warnings.append(
            f"model dtype {sorted(dtypes)} is outside the verified set (fp16, bf16); "
            "signals are produced but not guaranteed"
        )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return {
        "devices": sorted(devices),
        "dtypes": sorted(dtypes),
        "cuda_device_count": torch.cuda.device_count(),
        "warnings": warnings,
    }


def force_eager_attention(model: Any) -> str:
    """Switch the model to eager attention so probabilities are materialised.

    FlashAttention and SDPA compute the output without ever building the probability matrix, so there is nothing for a hook to catch.
    Transformers dispatches on `config._attn_implementation` at forward time, so this can be flipped on a loaded model instead of reloading it.

    Expect this to be slower and to use more memory: an explicit (seq x seq) matrix per head is exactly what the fast kernels avoid.
    It only runs on the collected documents, not on the whole task.
    """
    print(
        "warning: switching to eager attention to capture attention weights. "
        "This is slower than SDPA/FlashAttention and raises peak memory.",
        file=sys.stderr,
    )
    configs = [model.config] + [
        value
        for value in vars(model.config).values()
        if hasattr(value, "_attn_implementation")
    ]
    for config in configs:
        config._attn_implementation = "eager"
    return "eager"


# --------------------------------------------------------------------------
# Assembling a run
# --------------------------------------------------------------------------


def build_reducers(
    adapter: Any,
    tokenizer: Any,
    *,
    attention: bool,
    hidden_layers: list[int] | None,
    include_always_on: bool = True,
    signals: Sequence[str] = ("logit_lens", "similarity"),
):
    """Pick the reducer set for one pass.

    The evaluation signals default to logit lens and layer-pair similarity.
    `signals` can select either or disable both; tensor dumps are opt-in.

    `include_always_on` is False on the collection pass.
    The first pass already collected the always-on signals for *every* document, so re-running them on the collected subset would write a second copy of those rows.

    Example:
        >>> [r.name for r in build_reducers(adapter, tok, attention=True, hidden_layers=None)]
        ['logit_lens', 'layer_similarity', 'value_norm', 'attention_weights']
        >>> [r.name for r in build_reducers(adapter, tok, attention=True,
        ...                                 hidden_layers=[0], include_always_on=False)]
        ['value_norm', 'attention_weights', 'raw_hidden']
    """
    from . import reducers as reducer_module

    chosen = []
    if include_always_on:
        if "logit_lens" in signals:
            chosen.append(reducer_module.LogitLensReducer(adapter.decode_stack, tokenizer, adapter.vocab_size))
        if "similarity" in signals:
            chosen.append(reducer_module.SimilarityReducer(adapter.n_residual))
    if attention:
        chosen.append(
            reducer_module.ValueNormReducer(adapter.n_heads, adapter.n_kv_heads, adapter.head_dim,
                                            block_shapes=adapter.value_shapes())
        )
        chosen.append(reducer_module.AttentionWeightReducer())
    if hidden_layers:
        chosen.append(reducer_module.RawHiddenReducer(hidden_layers))
    return chosen


def load_model(config: RunConfig, extra: dict[str, Any] | None = None):
    """Load the model through our lm-eval backend.

    Model loading arguments are lm-eval's, unchanged, so `pretrained`, `revision`, `dtype`, `trust_remote_code` and `peft` all behave as documented there.
    """
    # cmd_run calls this before task setup as well. Keep the loading boundary
    # guarded for callers that use this helper directly.
    validate_single_gpu_execution(config)

    from .backend import TracedHFLM

    kwargs = {"batch_size": config.batch_size}
    kwargs.update(extra or {})
    if config.model_factory or config.model_bundle is not None:
        from .models import load_bundle
        from lm_eval.utils import simple_parse_args_string
        options = simple_parse_args_string(config.model_args)
        allowed = {"max_length", "max_gen_toks", "add_bos_token", "prefix_token_id",
                   "logits_cache", "truncation", "softmax_dtype"}
        invalid = set(options) - allowed
        if invalid:
            raise ValueError(f"custom factories own loading/device/dtype; unsupported model_args: {sorted(invalid)}")
        options.update(kwargs)
        bundle = load_bundle(config)
        lm = TracedHFLM(pretrained=bundle.model, tokenizer=bundle.tokenizer, **options)
        if lm.model is not bundle.model or lm.tokenizer is not bundle.tokenizer:
            raise RuntimeError("HFLM replaced the supplied model or tokenizer")
        lm.prompt_transform = bundle.prompt_transform
        return lm
    config.resolved_bundle = None
    config.model_provenance = {}
    if config.model_config:
        raise ValueError("model_config requires a model_factory or model_bundle")
    if not config.model_kwargs().get("pretrained"):
        raise ValueError("provide pretrained in --model-args, --model-factory, or model_bundle")
    if not config.model_kwargs().get("device"):
        import torch
        if not torch.cuda.is_available():
            # lm-eval would place the model on its default device, cuda, and fail inside torch.
            raise ValueError("CUDA is required, but torch reports no CUDA device; "
                             "this tool runs on a single CUDA GPU")
    return TracedHFLM.create_from_arg_string(config.model_args, kwargs)


def doc_id_set_hash(samples: Sequence[dict[str, Any]]) -> str:
    """Hash of the exact document set that was evaluated.

    Two runs of the same task can still cover different documents (one of them used `--limit`), and averaging those together is a silent mistake.
    `report` groups on this value.

    Example:
        >>> doc_id_set_hash([{"task_name": "t", "doc_id": 1}, {"task_name": "t", "doc_id": 0}])
        '7c17a9358555ccf7'
    """
    keys = sorted(f"{s['task_name']}:{int(s['doc_id'])}" for s in samples)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]


def lm_eval_install_info() -> dict[str, Any]:
    """Version, install form and git state of the installed lm-eval.

    An editable checkout can report the same version string while running different code, so without the commit and the dirty flag `report` could group two runs that were not produced by the same harness.
    """
    import subprocess

    import lm_eval

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(lm_eval.__file__)))
    info: dict[str, Any] = {
        "version": lm_eval.__version__,
        "path": os.path.abspath(lm_eval.__file__),
        "install_form": "editable" if os.path.isdir(os.path.join(package_dir, ".git")) else "wheel",
        "commit": None,
        "dirty": None,
    }
    if info["install_form"] == "editable":
        try:
            info["commit"] = subprocess.check_output(
                ["git", "-C", package_dir, "rev-parse", "HEAD"], text=True
            ).strip()
            info["dirty"] = bool(
                subprocess.check_output(
                    ["git", "-C", package_dir, "status", "--porcelain"], text=True
                ).strip()
            )
        except Exception:
            pass
    return info


def build_manifest(
    config: RunConfig,
    adapter: Any,
    lm: Any,
    reducer_descriptors: list[dict[str, Any]],
    environment: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest.

    Everything `report` needs to decide comparability goes in here, along with enough provenance to interpret the run after the code has moved on: the fixed settings as they were at the time, the decode path that was used, and which lm-eval actually ran.
    """
    from .backend import upstream_source_hashes

    install = lm_eval_install_info()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "lm_eval_version": install["version"],
        "lm_eval_install": install,
        "lm_eval_source_hashes": upstream_source_hashes(),
        "backend": storage.BACKEND_NAME,
        "model_id": config.model_id,
        "revision": _resolved_revision(lm),
        "model_args": config.model_args,
        "tokenizer_id": getattr(lm.tokenizer, "name_or_path", config.model_id),
        "tokenizer_revision": config.revision,
        "tasks": list(config.tasks),
        "benchmarks": config.benchmark_provenance,
        "custom_model": config.model_provenance,
        "signals": list(config.signals),
        "prompt_protocol": {"system_instruction": config.system_instruction,
                            "apply_chat_template": config.apply_chat_template},
        "internal_signals_available": adapter is not None,
        "custom_hooks": {"factory": config.hook_factory, "config": config.hook_config,
                         "specs": config.hook_provenance, "resolved": config.hook_resolved},
        "collection_sampling": {"method": "per_task_correctness", "seed": SAMPLING_SEED, "limit_per_group": config.collect_limit},
        "num_fewshot": config.num_fewshot,
        "limit": config.limit,
        "n_documents": len(samples),
        # Which filter's verdict `docs` carries, and what else the task offered.
        "filters_available": storage.available_filters(samples),
        "filters_used": storage.primary_filters(samples),
        "doc_id_set_hash": doc_id_set_hash(samples),
        # What a later `run` into this directory has to match before it may resume, so two configurations cannot end up in one run.
        "config_identity": extra.pop("config_identity", None) if extra else None,
        "reducers": reducer_descriptors,
        "batch_size": config.batch_size,
        "resolved_batch_size": getattr(lm, "batch_size", None),
        "generate_batch_size": 1,
        # Recorded because it is our one deliberate deviation from lm-eval's defaults, and in fp16 it can move a borderline document.
        # Under matched settings the traced backend is bit-identical to stock lm-eval; see scripts/verify_scores.py.
        "logits_cache": getattr(lm, "logits_cache", None),
        "attn_implementation": getattr(lm.model.config, "_attn_implementation", "unknown"),
        "options": {
            "save_attention": config.save_attention,
            "save_hidden": config.save_hidden,
            "hidden_layers": config.hidden_layers,
            "collect_limit": config.collect_limit,
        },
        "fixed_settings": dict(FIXED_SETTINGS),
        "seed": SAMPLING_SEED,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed": False,
    }
    if adapter is not None:
        manifest.update(adapter.manifest_entry())
    else:
        manifest.update({"model_type": getattr(lm.model.config, "model_type", "custom"),
                         "n_blocks": 0, "n_hidden_states": 0,
                         "layer_index_convention": "unavailable",
                         "attn_index_convention": "unavailable"})
    manifest.update(environment)
    manifest.update(extra or {})
    return manifest


def _resolved_revision(lm: Any) -> str:
    """The model's actual commit, because a hub revision like `main` moves."""
    commit = getattr(lm.model.config, "_commit_hash", None)
    return str(commit) if commit else "unknown"


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


@contextmanager
def _tracing(config: RunConfig, model: Any, run_dir: str, name: str = "trace"):
    """Hook the model for the duration, or do nothing at all.

    A plain context manager rather than a flag threaded through the call chain, so that the traced and untraced paths are the same code. When tracing is off this yields immediately and nothing in `debug.py` is touched.

    `name` separates the two passes. They are different programs with different memory profiles - the second re-runs a sample with eager attention to dump `(heads, seq, seq)` maps - and sharing one ring buffer would let the long cheap pass push the expensive one out of it.
    """
    if not config.debug.active:
        yield None
        return
    path = debug.trace_path(run_dir, name)
    tracer = debug.ModuleTracer(model, config.debug, path)
    with tracer.session():
        print(f"module tracing on: {len(tracer.traced_modules)} modules -> {path}")
        if tracer.statistics is not None:
            print(f"sample module statistics -> {tracer.statistics.directory}")
        yield tracer
    size = tracer.bytes_written
    print(f"trace written: {path}   {size / 1e6:.1f} MB   "
          f"(evalmetry debug {run_dir})")


def cmd_debug(args: argparse.Namespace) -> None:
    """Print what a run's traces say - the failure, or the modules of one forward."""
    path = args.path
    traces = debug.list_traces(path) if os.path.isdir(path) else [path]
    if not traces:
        # Let `read_trace` raise the message that explains why there is nothing here.
        debug.read_trace(path)

    if args.forward is None and args.doc is None:
        for trace in traces:
            print(debug.summarize_trace(trace, tail=args.events))
            print()
        return

    for trace in traces:
        events = debug.read_trace(trace)
        wanted = _forwards_wanted(events, args)
        if not wanted:
            continue
        print(f"trace: {trace}")
        for number in wanted:
            print()
            print(debug.describe_forward(events, number, module=args.module))


def _forwards_wanted(events: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    """Which forward numbers the user asked for, by number or by document."""
    if args.forward is not None:
        return [args.forward]
    task, _, raw = str(args.doc).rpartition("#")
    try:
        doc_id = int(raw)
    except ValueError:
        raise ValueError(f"--doc wants a document id, e.g. arc_easy#42 or 42, not {args.doc!r}")
    found = []
    for entry in debug.forward_index(events):
        for row in entry.get("samples") or []:
            if row[1] == doc_id and (not task or row[0] == task):
                found.append(entry["forward"])
                break
    if not found:
        print(f"no forward in this trace covers document {args.doc}")
    return found


def _releasing_writer_locks(command):
    """Release the run-directory locks a command took when it returns or fails.

    A failed run otherwise keeps its lock and file descriptor until the process exits,
    one per directory, in a notebook or sweep that keeps going after the failure.
    """
    @functools.wraps(command)
    def wrapper(*args, **kwargs):
        held = set(storage._WRITER_LOCKS)
        try:
            return command(*args, **kwargs)
        finally:
            for path in set(storage._WRITER_LOCKS) - held:
                storage.release_writer_lock(path)
    return wrapper


def _prepare_evaluation_tasks(config: RunConfig) -> tuple[Any, list[Any]]:
    """모델을 올리기 전에 custom hook과 benchmark를 검증하고 provenance를 채운다.

    로컬 task는 여기서 생성해 YAML과 사용자 함수 오류를 먼저 발견한다.
    built-in group 이름은 그대로 넘겨 lm-eval의 그룹 집계를 유지한다.
    """
    from .benchmarks import prepare_benchmarks
    from .hooks import load_hooks

    config.resolved_hooks = load_hooks(config.hook_factory, config.hook_config, config.hooks)
    if any(h.collection_resume for h in config.resolved_hooks) and (config.save_attention or config.save_hidden):
        raise ValueError("custom collection_resume cannot share legacy attention/hidden dumps; use custom raw_tensors")
    config.hook_provenance = [h.descriptor() for h in config.resolved_hooks]
    task_manager, evaluation_tasks, config.benchmark_provenance = prepare_benchmarks(
        config.include_path, config.tasks
    )
    if task_manager is not None:
        # Instantiate local tasks once for schema/function validation before GPU allocation.
        # Built-in group names stay intact so lm-eval retains their aggregate results.
        evaluation_tasks = [
            next(iter(task_manager.load([task])["tasks"].values()))
            if isinstance(task, dict) else task for task in evaluation_tasks
        ]
    return task_manager, evaluation_tasks


@_releasing_writer_locks
def cmd_run(config: RunConfig) -> str:
    """Evaluate with lm-eval, collect signals, and leave a run directory behind.

    With `--save-attention` or `--save-hidden` this runs two passes.
    The first scores and records the always-on signals; the second collects a balanced sample of right and wrong answers with the expensive hooks attached.
    It has to be two passes because correctness is not known until lm-eval has finished scoring, and deciding correctness ourselves would mean reimplementing it.

    Returns:
        The run directory.
    """
    validate_single_gpu_execution(config)

    import lm_eval

    from .adapters import check_decode_identity
    from .backend import check_upstream_source
    from .recorder import Recorder
    from .hooks import ensure_collection_is_empty

    # 1. 사용자 확장을 검증한 뒤 모델과 관측 구성을 준비한다.
    task_manager, evaluation_tasks = _prepare_evaluation_tasks(config)
    check_upstream_source(strict=False)

    from .models import resolve_model_adapter
    unknown = set(config.signals) - {"logit_lens", "similarity"}
    if unknown:
        raise ValueError(f"unknown signals: {sorted(unknown)}")
    lm = load_model(config)
    adapter = resolve_model_adapter(config, lm)
    environment = check_environment(lm.model)

    import torch

    decode_identity = (check_decode_identity(
        adapter, lm.model, torch.tensor([[lm.eot_token_id]], device=lm.device)
    ) if "logit_lens" in config.signals else None)

    hidden_layers = (
        parse_hidden_layers(config.hidden_layers, adapter.n_residual)
        if config.save_hidden
        else None
    )
    reducers = build_reducers(adapter, lm.tokenizer, attention=False, hidden_layers=None, signals=config.signals)
    descriptors = [r.descriptor() for r in reducers]

    # 2. 기존 기록과 호환되는지 확인하고 저장·관측 객체를 연결한다.
    identity = config.identity(descriptors, lm_eval.__version__)
    run_dir = config.output or config.default_output(
        config.config_hash(descriptors, lm_eval.__version__)
    )
    check_resume_is_the_same_run(run_dir, identity)
    ensure_collection_is_empty(run_dir, config.resolved_hooks)
    if config.hook_provenance:
        # 이 실행이 어떤 모듈을 관측했는지는 아래에서 다시 쓰지만, 먼저 이전 기록을
        # 그대로 옮겨 둔다. 그러지 않으면 재개 직전의 manifest 덮어쓰기가 비교 기준을
        # 지워버리고, 경로 집합이 달라진 것을 아무도 알아채지 못한다.
        try:
            config.hook_resolved = (storage.read_manifest(run_dir).get("custom_hooks")
                                    or {}).get("resolved")
        except (OSError, ValueError, KeyError):
            pass
    os.makedirs(run_dir, exist_ok=True)
    print(f"run directory: {run_dir}")

    if config.benchmark_provenance or config.hook_provenance or config.model_provenance:
        storage.write_results(run_dir, build_manifest(
            config, adapter, lm, descriptors, environment, [],
            extra={"config_identity": identity}), {})
    writer = storage.RunWriter(run_dir)
    if adapter is not None:
        storage.check_recorded_tables_agree(run_dir, [r.table for r in reducers if r.table])
    if writer.already_recorded:
        print(
            f"resuming: {len(writer.already_recorded)} requests are already recorded "
            "and will be scored again but not re-recorded"
        )
    recorder = None
    if adapter is not None:
        recorder = Recorder(adapter, reducers, writer, lm.tokenizer, hooks=config.resolved_hooks)
        lm.attach_recorder(recorder)

    # 3. 평가와 judge 채점을 완료하고 문서별 결과를 저장한다.
    results, samples = _evaluate_and_save_samples(
        config, lm, writer, run_dir, task_manager, evaluation_tasks
    )

    # Read before the collection pass, which may switch the model to eager attention: the manifest should say what the scored pass actually ran with.
    attn_implementation = getattr(lm.model.config, "_attn_implementation", "unknown")

    # 4. 필요한 경우 후속 수집을 수행한다. 완료 표시는 이 단계까지 성공한 뒤 쓴다.
    # Collection can die after scoring. Persist the scored result before entering any
    # resumable unit so standalone replay cannot mark an empty score payload complete.
    if any(h.collection_resume for h in config.resolved_hooks):
        if recorder is not None:
            config.hook_resolved = {**(config.hook_resolved or {}), **recorder.custom.resolved_paths()}
        storage.write_results(run_dir, build_manifest(config, adapter, lm, descriptors,
            environment, samples, extra={"config_identity": identity, "evaluation_completed": True,
                                        "attn_implementation": attn_implementation,
                                        "decode_identity": decode_identity}),
            results.get("results", {}), writer.signal_files())
    collection_counts: dict[str, int] = {}
    if config.save_attention or config.save_hidden or any(h.pass_name == "collection" for h in config.resolved_hooks):
        # Its own trace: the collection pass forces eager attention and materialises a
        # (heads, seq, seq) map per block, which is the heaviest thing this tool does.
        with _tracing(config, lm.model, run_dir, name="collection"):
            collection_counts = _collect_research_data(
                config, lm, adapter, writer, run_dir, hidden_layers
            )
        print(f"collection: {collection_counts}")

    if recorder is not None and config.resolved_hooks:
        # 확정은 첫 forward 전에 끝났다. 여기서는 그것을 기록할 뿐이다. 두 pass의 hook은
        # 이름이 다르므로 합친다.
        config.hook_resolved = {**(config.hook_resolved or {}),
                                **recorder.custom.resolved_paths()} or None
    writer.close()
    if recorder is not None and recorder.skipped:
        print(f"skipped recording for {recorder.skipped} already-present requests")
    manifest = build_manifest(
        config,
        adapter,
        lm,
        descriptors,
        environment,
        samples,
        extra={
            "config_identity": identity,
            "attn_implementation": attn_implementation,
            "collection_attn_implementation": getattr(
                lm.model.config, "_attn_implementation", "unknown"
            ),
            "decode_identity": decode_identity,
            "collection_counts": collection_counts,
            "evaluation_completed": True,
            "generation_kwargs_source": "lm-eval task config",
            "debug": config.debug.manifest_entry(),
        },
    )
    storage.write_results(run_dir, manifest, results.get("results", {}), writer.signal_files())
    storage.mark_complete(run_dir)
    if recorder is None:
        print(f"done: {len(samples)} samples scored; internal collection disabled")
    else:
        print(f"done: {writer.documents_written} documents recorded")
    return run_dir


def _evaluate_and_save_samples(
    config: RunConfig, lm: Any, writer: storage.RunWriter, run_dir: str,
    task_manager: Any, evaluation_tasks: Sequence[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """평가하고 judge 결과를 반영한 samples/docs를 저장한다.

    forward trace는 lm-eval 평가까지만 감싼다. signal을 flush한 다음 judge를
    실행하고, 성공한 경우에만 최종 samples/docs를 쓴다. judge 실패 시의 부분
    저장은 judges 모듈이 담당하며 여기서는 실패를 그대로 전달한다.
    """
    import lm_eval

    from .benchmarks import annotate_samples

    # 미지정 옵션은 키 자체를 생략해 lm-eval의 기본 prompt 처리를 유지한다.
    evaluation_options: dict[str, Any] = {}
    if task_manager is not None:
        evaluation_options["task_manager"] = task_manager
    if config.system_instruction is not None or config.apply_chat_template:
        evaluation_options["system_instruction"] = config.system_instruction
        evaluation_options["apply_chat_template"] = config.apply_chat_template

    # The tracer wraps the *whole* evaluation rather than each request loop, which is what
    # puts the `--batch-size auto` probe inside it. That probe registers no documents and so
    # records no signals, and it is a likely place to run out of memory - precisely the
    # combination that leaves nothing behind without this.
    with _tracing(config, lm.model, run_dir):
        # `log_samples=True` is not optional: the grading join needs it, and forcing the output into the run directory keeps the run self-contained.
        # `use_cache=None` keeps lm-eval's request cache off - a cached request is not re-run, which would leave a hole in the signals.
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=evaluation_tasks,
            **evaluation_options,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            log_samples=True,
            use_cache=None,
            write_out=False,
        )
    writer.flush()

    samples_by_task = results.get("samples", {})
    from .models import save_effective_prompts
    save_effective_prompts(samples_by_task, lm)
    from .judges import score_judge_tasks
    score_judge_tasks(results, evaluation_tasks, run_dir, config.benchmark_provenance)
    annotate_samples(samples_by_task, config.benchmark_provenance)
    storage.write_samples(run_dir, samples_by_task)
    samples = storage.read_samples(run_dir)
    storage.write_docs_table(run_dir, samples)

    return results, samples


def _collect_research_data(
    config: RunConfig,
    lm: Any,
    adapter: Any,
    writer: storage.RunWriter,
    run_dir: str,
    hidden_layers: list[int] | None,
    *,
    write_steps: bool = False,
) -> dict[str, int]:
    """Collect selected documents through the shared second-pass pipeline.

    Both `run` and standalone collection use the same reducers, hook runtime and
    output checks. `write_steps` is needed only when the saved evaluation did not
    record internal signals; otherwise writing steps again would duplicate them.
    Resolved collection hook paths are returned through `config.hook_resolved`
    so each command can merge them into its own manifest checkpoint.
    """
    from .backend import ResearchDataCollector
    from .recorder import Recorder

    if config.save_attention:
        force_eager_attention(lm.model)
    reducers = build_reducers(
        adapter,
        lm.tokenizer,
        attention=config.save_attention,
        hidden_layers=hidden_layers,
        include_always_on=False,
    )
    # skip_recorded=False is essential, not incidental: collection exists to re-run documents that *are* already recorded, in order to collect signals the first pass did not.
    # The restart skip would drop every one of them and the command would report success while writing nothing.
    recorder = Recorder(adapter, reducers, writer, lm.tokenizer,
                        write_steps=write_steps, skip_recorded=False,
                        hooks=config.resolved_hooks, pass_name="collection")
    runner = ResearchDataCollector(lm, recorder, run_dir)
    counts = runner.run(limit=config.collect_limit)
    writer.flush()
    # A collection-pass hook is not in the evaluation runtime, so this is the only
    # place its resolved paths exist.
    config.hook_resolved = {**(config.hook_resolved or {}),
                            **recorder.custom.resolved_paths()} or None
    _assert_collection_produced_output(writer, counts, config)
    return counts


def _assert_collection_produced_output(
    writer: storage.RunWriter, counts: dict[str, int], config: RunConfig
) -> None:
    """Fail loudly when a collection pass wrote nothing.

    A collection pass that silently produces no files is indistinguishable from a successful one at the command line, and only shows up later as an empty directory.
    It has happened, so it is checked.
    """
    if not counts.get("documents"):
        raise RuntimeError(
            "collection selected no documents; samples.jsonl may be empty or the task "
            "reports no correctness metric"
        )
    if config.save_attention and not writer.attention.files_written:
        raise RuntimeError(
            "collection ran but wrote no attention tensors. The recorder skipped every "
            "document, or the attention modules returned no weights."
        )
    if config.save_hidden and not writer.raw_hidden.files_written:
        raise RuntimeError("collection ran but wrote no hidden-state tensors.")


def _collection_config_from_manifest(
    args: argparse.Namespace, manifest: dict[str, Any],
) -> RunConfig:
    """완료된 평가의 모델·hook을 복원하고 수집 옵션을 합친다.

    CLI에서는 수집할 데이터와 trace 옵션만 받는다. 여기서는 평가 완료 여부와
    hook 복원을 검증한다. 모델 구현의 일치는 호출자가 모델을 로딩한 뒤 확인한다.
    """
    run_dir = args.run_dir
    if not (manifest.get("completed") or manifest.get("evaluation_completed")):
        # A checkpoint written before scoring or judge grading finished has no scores;
        # marking it complete here would publish an empty result as a finished run.
        raise ValueError(
            "collection needs a finished evaluation: the manifest records neither completed "
            "nor evaluation_completed. Rerun `run` into the same directory first."
        )
    saved_model = manifest.get("custom_model", {})
    if saved_model and (not saved_model.get("factory") or not saved_model.get("resume_safe")):
        raise ValueError("standalone collection requires a reproducibly identified model factory; in-memory models can collect during run")
    saved_hooks = manifest.get("custom_hooks", {})
    from .hooks import load_hooks, ensure_collection_is_empty
    hook_specs = saved_hooks.get("specs", [])
    if hook_specs and not saved_hooks.get("factory"):
        if any(h["pass_name"] == "collection" for h in hook_specs):
            raise ValueError("standalone collection requires a restorable hook factory")
        hooks = []
    else:
        hooks = load_hooks(saved_hooks.get("factory"), saved_hooks.get("config", {}))
        if any(not h["resume_safe"] for h in hook_specs):
            raise ValueError("custom hook source is unavailable; standalone collection is disabled")
        if [h.descriptor() for h in hooks] != hook_specs:
            raise ValueError("custom hook implementation changed since evaluation")
    collection_hooks = [h for h in hooks if h.pass_name == "collection"]
    ensure_collection_is_empty(run_dir, collection_hooks)
    if any(h.collection_resume for h in collection_hooks) and not manifest.get("evaluation_completed"):
        raise ValueError("incomplete custom collection manifest: durable evaluation checkpoint missing")
    if any(h.collection_resume for h in collection_hooks) and (args.save_attention or args.save_hidden):
        raise ValueError("custom collection_resume cannot share legacy attention/hidden dumps; use custom raw_tensors")
    if not (args.save_attention or args.save_hidden or collection_hooks):
        raise ValueError(
            "collection has nothing to collect: pass --save-attention and/or --save-hidden. "
            "The always-on signals were already recorded for every document by `run`."
        )
    config = RunConfig(
        hooks=hooks,
        model_args=manifest["model_args"],
        model_factory=saved_model.get("factory"),
        model_config=saved_model.get("config", {}),
        signals=(),
        tasks=list(manifest["tasks"]),
        batch_size=1,                      # collection is always unbatched
        adapter=saved_model.get("adapter") if saved_model else manifest.get("model_type"),
        save_attention=args.save_attention,
        save_hidden=args.save_hidden,
        hidden_layers=args.hidden_layers,
        collect_limit=args.collect_limit,
        debug=debug_config_from_args(args),
    )
    config.resolved_hooks = hooks
    return config


@_releasing_writer_locks
def cmd_collect_research_data(args: argparse.Namespace) -> str:
    """Re-feed part of a finished run to collect signals it does not have yet.

    Model and tokenizer settings come from the run's manifest, never from the command line: the two passes have to produce identical input tokens, so letting them be re-specified would be a way to get that wrong.
    """
    run_dir = args.run_dir
    manifest = storage.read_manifest(run_dir)
    config = _collection_config_from_manifest(args, manifest)
    saved_model = manifest.get("custom_model", {})
    lm = load_model(config)
    if saved_model and config.model_provenance != saved_model:
        raise ValueError("custom model implementation/config/checkpoint changed since evaluation")
    from .models import resolve_model_adapter
    adapter = resolve_model_adapter(config, lm, collection=True)
    check_environment(lm.model)
    hidden_layers = (
        parse_hidden_layers(config.hidden_layers, adapter.n_residual)
        if config.save_hidden
        else None
    )
    writer = storage.RunWriter(run_dir)
    with _tracing(config, lm.model, run_dir, name="collection"):
        counts = _collect_research_data(
            config, lm, adapter, writer, run_dir, hidden_layers,
            write_steps=not manifest.get("internal_signals_available", True),
        )
    writer.close()
    # The manifest has to describe what the directory now holds, not only what the original `run` asked for.
    # A standalone collection adds dumps the first pass never made, and forces eager attention to make them; leaving the manifest saying `save_attention: false` and `sdpa` would misdescribe the files sitting next to it.
    options = dict(manifest.get("options", {}))
    options.update({
        "save_attention": options.get("save_attention") or config.save_attention,
        "save_hidden": options.get("save_hidden") or config.save_hidden,
        "hidden_layers": config.hidden_layers,
        "collect_limit": config.collect_limit,
    })
    payload = storage.read_results(run_dir)
    storage.write_results(run_dir, payload["manifest"], payload["results"], writer.signal_files())
    collection_metadata = {
        "collection_counts": counts,
        "collection_attn_implementation": getattr(
            lm.model.config, "_attn_implementation", "unknown"),
        "options": options,
        "collected_separately": True,
        "internal_signals_available": True,
        **adapter.manifest_entry(),
        "collection_sampling": {"method": "per_task_correctness", "seed": SAMPLING_SEED,
                                "limit_per_group": config.collect_limit},
    }
    if config.resolved_hooks:
        saved_custom = manifest.get("custom_hooks") or {}
        collection_metadata["custom_hooks"] = {
            **saved_custom,
            "resolved": {
                **(saved_custom.get("resolved") or {}),
                **(config.hook_resolved or {}),
            },
        }
    storage.mark_complete(run_dir, extra=collection_metadata)
    print(f"collection: {counts}")
    return run_dir


def cmd_report(args: argparse.Namespace) -> None:
    """Gather runs, print the grouping, then write the aggregate and qualitative views.

    Both come out of one partition: the curves say which model is better, the examples say on what.
    `--multilingual` asks the other question - one model, one benchmark, several languages - which is a different partition and a different reference, so it is a mode rather than an extra figure.
    """
    from .report import run_report

    from .report import parse_datasets

    run_report(args.paths, args.output, args.reference, args.examples_per_category,
               datasets=parse_datasets(args.multilingual) if args.multilingual else None,
               assume_aligned=args.assume_aligned, pair_on=args.pair_on,
               pair_strict=args.pair_strict, pair_mapping=args.pair_mapping)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_debug_arguments(parser: argparse.ArgumentParser) -> None:
    """The module-tracing flags, shared by `run` and `collect-research-data`.

    Off by default and absent from `RunConfig.identity`: tracing changes how long a run takes, not what its signals mean, so a traced and an untraced run belong in the same directory.

    The levels are two flags rather than one verbosity number on purpose. Reading shapes and the allocator's counters queues no kernel and syncs nothing, so `--debug` is safe to leave on in a run that is already near the memory ceiling. `--debug-numeric` allocates several same-size intermediates per tensor and can itself be the allocation that fails.
    """
    group = parser.add_argument_group(
        "module tracing",
        "Record what each module was handed and how far the forward pass got. "
        "Read it back with `evalmetry debug <run_dir>`.",
    )
    group.add_argument("--debug", action="store_true",
                       help="trace every module: enter/exit, input and output "
                            "shape/dtype/device, and the documents in flight")
    group.add_argument("--debug-modules", default=None, metavar="REGEX",
                       help=r"trace only modules whose dotted path matches, "
                            r"e.g. 'layers\.\d+$'")
    group.add_argument("--debug-numeric", action="store_true",
                       help="also summarise tensor values: NaN/Inf counts and the "
                            "finite range. Costs a device sync and several same-size "
                            "intermediates per tensor - it can itself cause an OOM")
    group.add_argument("--module-stats", action="store_true",
                       help="collect per-document module input/output moments, exclude "
                            "padding, and export sample and dataset tables. Enables tracing; "
                            "uses --debug-modules to select modules. Costs tensor reductions")
    group.add_argument("--module-stats-extremes", type=int, default=0, metavar="K",
                       help="also record the K largest absolute values per document, "
                            "module call and tensor, with their token position and "
                            "feature index. Needs --module-stats and "
                            "--module-stats-extremes-modules; writes K rows where the "
                            "moments write one, so the selector is deliberately separate")
    group.add_argument("--module-stats-extremes-modules", default=None, metavar="REGEX",
                       help=r"which traced modules record extreme positions, e.g. "
                            r"'layers\.\d+\.mlp$'. Confirmed token/feature axes only: "
                            r"attention matrices are measured but never unfolded")
    group.add_argument("--module-stats-axes", default=None,
                       choices=("feature", "position", "both"),
                       help="also record statistics along one axis of the same document "
                            "slice: `feature` keeps the channel index and reduces the "
                            "valid positions, `position` keeps the input column and "
                            "reduces the channels. Needs --module-stats and "
                            "--module-stats-axes-modules; writes one row per index")
    group.add_argument("--module-stats-axes-modules", default=None, metavar="REGEX",
                       help=r"which traced modules record per-axis statistics, e.g. "
                            r"'layers\.\d+\.mlp$'. Confirmed token/feature axes only: "
                            r"attention matrices are measured but never unfolded")
    group.add_argument("--module-stats-routing", default=None, metavar="REGEX",
                       help="which traced routers record the expert selection they "
                            "returned: one row per token, rank, expert and weight. Needs "
                            "--module-stats, and the selector must match a registered "
                            "router boundary, or the run stops before it starts")
    group.add_argument("--module-stats-chunk-elements", type=int, default=None, metavar="N",
                       help="elements per float64 reduction chunk (default 65536). Changes "
                            "the size of temporary buffers, never a stored value; a strided "
                            "sample may still be copied whole, so this is not a memory cap")
    group.add_argument("--debug-stop-on-nonfinite", action="store_true",
                       help="raise at the first module whose output is not all "
                            "finite, instead of letting it propagate. Implies "
                            "--debug-numeric and changes what the run does")
    group.add_argument("--debug-tail", type=int, nargs="?", const=TRACE_BUFFER_EVENTS,
                       default=None, metavar="N",
                       help=f"keep only the last N events (default {TRACE_BUFFER_EVENTS}) "
                            "instead of the whole run. For hunting a crash in a long run, "
                            "where the full trace would be gigabytes and only the end "
                            "matters. It is written when the run ends, so it does not "
                            "survive the process being killed without an exception")
    group.add_argument("--debug-sync", action="store_true",
                       help="torch.cuda.synchronize() at every module boundary, to "
                            "pin an asynchronous device fault to its module. Slow, "
                            "and it changes timing. Not needed for OOM, whose "
                            "allocation is synchronous already")


def debug_config_from_args(args: argparse.Namespace) -> "debug.DebugConfig":
    """Build the tracing configuration from parsed arguments."""
    from . import debug as debug_module

    return debug_module.DebugConfig(
        enabled=args.debug,
        modules=args.debug_modules,
        numeric=args.debug_numeric,
        stop_on_nonfinite=args.debug_stop_on_nonfinite,
        tail=args.debug_tail,
        sync=args.debug_sync,
        sample_stats=args.module_stats,
        extremes=args.module_stats_extremes,
        extremes_modules=args.module_stats_extremes_modules,
        axes=args.module_stats_axes,
        axes_modules=args.module_stats_axes_modules,
        routing=args.module_stats_routing,
        chunk_elements=args.module_stats_chunk_elements,
    )


def build_parser() -> argparse.ArgumentParser:
    """Every argument the tool takes.

    Example:
        >>> build_parser().parse_args(
        ...     ["run", "--model-args", "pretrained=Qwen/Qwen3-8B", "--tasks", "xnli_ko"]
        ... ).tasks
        'xnli_ko'
    """
    parser = argparse.ArgumentParser(
        prog="evalmetry",
        description="Evaluate a model with lm-eval while collecting per-layer internal signals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="evaluate a model and collect signals; resumes an interrupted run",
        description="Evaluate a model with lm-eval while collecting per-layer "
                    "signals. Re-running into a directory that already holds "
                    "shards resumes it: whatever is recorded there is scored "
                    "again but not recorded twice, so no rows are duplicated.")
    run.add_argument("--model-args", default="",
                     help="passed to lm-eval verbatim, e.g. pretrained=Qwen/Qwen3-8B,dtype=bfloat16")
    run.add_argument("--model-factory", help="package.module:function returning ModelBundle")
    run.add_argument("--model-config", type=json.loads, default={}, help="JSON object passed to model factory")
    run.add_argument("--signals", default="logit_lens,similarity",
                     help="comma separated logit_lens,similarity; none for scoring only")
    run.add_argument("--system-instruction", default=None)
    run.add_argument("--apply-chat-template", action="store_true")
    run.add_argument("--tasks", required=True, help="comma separated lm-eval task names")
    run.add_argument("--include-path", action="append", default=[],
                     help="local benchmark directory; repeat for multiple bundles")
    run.add_argument("--num-fewshot", type=int, default=0)
    run.add_argument("--limit", type=int, default=None, help="cap on the number of documents")
    run.add_argument("--batch-size", type=parse_batch_size, default=1,
                     help='positive integer, or "auto" to let lm-eval find the '
                          "largest batch that fits. generate tasks always run at 1")
    run.add_argument("--hook-factory", help="package.module:function returning HookSpec objects")
    run.add_argument("--hook-config", type=json.loads, default={}, help="JSON object passed to hook factory")
    run.add_argument("--output", default=None, help="run directory (default: results/...)")
    run.add_argument("--adapter", default=None,
                     help="force a module-path adapter; the escape hatch for trust_remote_code models")
    run.add_argument("--save-attention", action="store_true",
                     help="second pass: store attention weights and value norms")
    run.add_argument("--save-hidden", action="store_true",
                     help="second pass: store raw hidden states")
    run.add_argument("--hidden-layers", default="all", help="all | 0,1,2 | 0-8")
    run.add_argument("--collect-limit", type=int, default=DEFAULT_COLLECT_LIMIT,
                     help="documents collected per task and correctness group")

    collection = sub.add_parser(
        "collect-research-data",
        help="add attention / hidden-state dumps to a finished run, without re-scoring",
        description="Add the opt-in signals to a run that already finished. This is "
                    "NOT how an interrupted run is resumed - `run` does that by "
                    "itself. It exists because which documents to dump can only be "
                    "chosen once lm-eval has scored them: the sample is balanced "
                    "across correct and incorrect answers, and correctness is not "
                    "known during the first pass. Re-running `run` would reach the "
                    "same result by re-scoring every document, which this avoids.")
    collection.add_argument("run_dir", help="the run directory of the first pass")
    collection.add_argument("--save-attention", action="store_true")
    collection.add_argument("--save-hidden", action="store_true")
    collection.add_argument("--hidden-layers", default="all")
    collection.add_argument("--collect-limit", type=int, default=DEFAULT_COLLECT_LIMIT)

    add_debug_arguments(run)
    add_debug_arguments(collection)

    trace = sub.add_parser(
        "debug",
        help="read back the module trace a --debug run left behind",
        description="Summarise a debug trace: how far the forward pass got, which "
                    "module was executing when it stopped, what it was handed, and "
                    "which documents were in flight. Takes a run directory or a "
                    "trace file.")
    trace.add_argument("path", help="a run directory, or a debug/trace.jsonl")
    trace.add_argument("--events", type=int, default=10,
                       help="module events to print from just before the failure")
    trace.add_argument("--forward", type=int, default=None, metavar="N",
                       help="print every module of forward N, in call order, with what "
                            "it was handed and what it returned. With no failure to "
                            "report this is the view worth having")
    trace.add_argument("--doc", default=None, metavar="TASK#ID",
                       help="the forwards covering one document, e.g. arc_easy#42 "
                            "or just 42")
    trace.add_argument("--module", default=None, metavar="REGEX",
                       help=r"narrow --forward / --doc to matching module paths. A "
                            r"regex, so escape the dots: 'layers\.1\.' is layer 1, while "
                            r"'layers.1.' is layers 1 and 10-19 because the trailing dot "
                            r"matches the 0. The listing reports which layers it matched")

    stats = sub.add_parser("module-stats", help="read sample and dataset module statistics")
    stats.add_argument("path", help="run directory, statistics session directory, or SQLite file")
    stats.add_argument("--pass", dest="pass_name", choices=("trace", "collection"), default="trace",
                       help="which evaluation pass to inspect (latest session only)")
    stats.add_argument("--doc", metavar="TASK#ID", help="show one document's pooled statistics")
    stats.add_argument("--module", metavar="REGEX", help="filter module paths")

    report = sub.add_parser("report", help="group runs, draw them, and pick examples")
    report.add_argument("paths", nargs="+", help="directories to search recursively")
    report.add_argument("--output", default=".",
                        help="where comparison.parquet, report.pdf, examples.parquet "
                             "and examples.md go")
    report.add_argument(
        "--reference", default=None,
        help="substring of the series label the example buckets are defined against - a "
             "model id normally, a language under --multilingual. "
             "Default: the highest-scoring run in each group, printed when chosen")
    report.add_argument("--examples-per-category", type=int, default=3,
                        help="documents sampled per bucket; 0 skips the examples")
    report.add_argument("--multilingual", default=None, metavar="LANG=TASK,LANG=TASK",
                        help="compare one model across languages instead of several models "
                             "over one dataset. Name the datasets outright, e.g. "
                             "en=global_mmlu_en,ko=global_mmlu_ko - benchmarks spell their "
                             "languages in too many ways to be guessed from a task name. A "
                             "bare task name labels itself. Runs are grouped by (model, shot "
                             "count) and drawn one line per language")
    report.add_argument("--pair-on", default=None, metavar="FIELD",
                        help="--multilingual: the logged `doc` field that identifies the same "
                             "document in every language, e.g. sample_id for Global-MMLU. "
                             "Without it one is looked for and used only if it is unique "
                             "within each run and carries the same values in all of them; "
                             "`--pair-on position` compares by document order instead")
    report.add_argument("--pair-strict", action="store_true",
                        help="--multilingual: when no identity field or mapping pairs the "
                             "documents, compare none of them one to one instead of pairing "
                             "by position. Curves are unaffected")
    report.add_argument("--pair-mapping", default=None, metavar="FILE",
                        help="--multilingual: a .csv or .jsonl mapping with columns language, "
                             "task_name, doc_id, canonical_doc_id and optionally "
                             "canonical_choice_ids (a|b|c) and canonical_answer_id. Its sha256 "
                             "and coverage are written to pairing.json; the mapping is taken "
                             "as given, not as proof that documents mean the same thing")
    report.add_argument("--assume-aligned", action="store_true",
                        help="--multilingual: compare documents by doc_id even when the "
                             "gold-answer check falls below the threshold. For a translation "
                             "that shuffled the choices, where the documents do correspond "
                             "but their gold positions do not. The measured rate is reported "
                             "either way")

    return parser


# CLI에서 traceback 대신 오류 메시지를 출력할 예외들이다. 설정 오류뿐 아니라
# 수집·검증 중 발생한 RuntimeError도 포함한다. 그 밖의 예외는 원래 traceback을
# 유지한다. LookupError에는 KeyError와 reference 검색 실패가 모두 포함된다.
CONFIGURATION_ERRORS = (LookupError, ValueError, RuntimeError, FileNotFoundError)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch the command, and return a process exit code.

    Raises:
        SystemExit: for CONFIGURATION_ERRORS, carrying the message rather than a traceback.
            Anything else propagates - an unexpected failure should show where it happened.
    """
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except CONFIGURATION_ERRORS as error:
        raise SystemExit(f"error: {error}") from error


def _dispatch(args: argparse.Namespace) -> int:
    """Run the chosen subcommand."""
    if args.command == "run":
        cmd_run(
            RunConfig(
                model_args=args.model_args,
                model_factory=args.model_factory,
                model_config=args.model_config,
                signals=tuple(x.strip() for x in args.signals.split(",") if x.strip() and x.strip() != "none"),
                system_instruction=args.system_instruction,
                apply_chat_template=args.apply_chat_template,
                tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
                include_path=args.include_path,
                hook_factory=args.hook_factory,
                hook_config=args.hook_config,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
                batch_size=args.batch_size,
                adapter=args.adapter,
                save_attention=args.save_attention,
                save_hidden=args.save_hidden,
                hidden_layers=args.hidden_layers,
                collect_limit=args.collect_limit,
                output=args.output,
                debug=debug_config_from_args(args),
            )
        )
    elif args.command == "collect-research-data":
        cmd_collect_research_data(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "debug":
        cmd_debug(args)
    elif args.command == "module-stats":
        from .module_stats import report_statistics

        print(report_statistics(args.path, args.pass_name, args.doc, args.module))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
