"""Custom HF models, provenance and optional input formatting.

Factories receive a JSON object and return ModelBundle. They own model loading,
placement and modifications; the evaluator never reloads their checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable


def import_factory(path: str) -> Callable:
    """Resolve an explicit package.module:function and report invalid entrypoints."""
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError("model factory must be package.module:function")
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise TypeError(f"{path} is not callable")
    return value


def fingerprint_files(paths) -> dict[str, str]:
    """Hash files or directory contents, streaming checkpoints without loading them.

    This reads every byte, including large checkpoints. Run it once in a factory;
    do not call it for each forward. Missing inputs are errors.
    """
    result = {}
    for item in paths:
        path = Path(item).resolve()
        if not path.exists():
            raise ValueError(f"provenance file does not exist: {path}")
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(file)] = digest.hexdigest()
    return result


@dataclass
class ModelBundle:
    """One already-loaded text causal LM with HF forward/generate interfaces.

    model and tokenizer are passed by identity to HFLM. The caller owns device,
    dtype and deterministic construction. Evaluation may call model.eval().
    version identifies all implementation dependencies not in source_files.
    checkpoint_id must identify immutable weights (Hub commit or file hashes),
    not a mutable branch/path. Missing version/checkpoint disables resume.
    adapter_factory(model) may return ModelAdapter or ModulePaths, and is only
    called when internal signals are requested.
    prompt_transform(text) changes context only, never answer continuations.
    It must be deterministic and stateless. Samples retain original task prompts;
    evaluation applies this function before tokenization and stores its result
    alongside the original context; replay uses the saved result.
    capabilities lists signals whose contracts the custom implementation retains.

    Example: ModelBundle(model, tokenizer, "my-mlp", version="v1",
                         checkpoint_id=fingerprint_files([checkpoint_dir]))
    """
    model: Any
    tokenizer: Any
    model_id: str
    version: str = ""
    checkpoint_id: Any = None
    adapter_factory: Callable | None = None
    prompt_transform: Callable[[str], str] | None = None
    source_files: list[str] = field(default_factory=list)
    capabilities: tuple[str, ...] = ("logit_lens", "similarity", "hidden", "attention", "hooks")


def load_bundle(config):
    """Validate a factory/object and capture the reproducible construction contract."""
    if config.model_bundle is not None and config.model_factory:
        raise ValueError("pass model_bundle or model_factory, not both")
    if not isinstance(config.model_config, dict):
        raise TypeError("model_config must be a JSON object")
    json.dumps(config.model_config, allow_nan=False)
    factory = import_factory(config.model_factory) if config.model_factory else None
    # JSON round-trip isolates nested factory mutations from the saved settings.
    bundle = factory(json.loads(json.dumps(config.model_config))) if factory else config.model_bundle
    if not isinstance(bundle, ModelBundle):
        raise TypeError("model factory must return ModelBundle")
    if not bundle.model_id or bundle.model is None or bundle.tokenizer is None:
        raise ValueError("ModelBundle requires model, tokenizer and model_id")
    sources = list(bundle.source_files)
    source_available = True
    for function in (factory, bundle.adapter_factory, bundle.prompt_transform):
        if function is None:
            continue
        if not callable(function):
            raise TypeError("adapter_factory and prompt_transform must be callable")
        try:
            source = inspect.getsourcefile(function)
        except TypeError:
            source = None
        if source and Path(source).is_file():
            sources.append(source)
            # Include helper modules in the entrypoint's local package, not just
            # the factory function. External dependencies remain version's contract.
            module = importlib.import_module(function.__module__.split('.')[0])
            for root in getattr(module, "__path__", []):
                sources.extend(str(p) for p in Path(root).rglob("*.py"))
        else:
            source_available = False
    tokenizer = bundle.tokenizer
    tok_backend = getattr(tokenizer, "backend_tokenizer", None)
    slow_files = {}
    if tok_backend is None:
        # Vocabulary alone misses SentencePiece scores and tokenizer rules.
        # Persist the actual tokenizer assets briefly, then hash their contents.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix="eval-model-tokenizer-") as directory:
            tokenizer.save_pretrained(directory)
            slow_files = {str(Path(path).relative_to(directory)): digest
                          for path, digest in fingerprint_files([directory]).items()}
    from importlib.metadata import version
    provenance = {
        "packages": {name: version(name) for name in ("torch", "transformers", "lm-eval")},
        "backend_options": config.model_kwargs(),
        "adapter": config.adapter,
        "dtypes": sorted({str(p.dtype) for p in bundle.model.parameters()}),
        "factory": config.model_factory, "config": config.model_config,
        "model_id": bundle.model_id, "version": bundle.version,
        "checkpoint_id": bundle.checkpoint_id,
        "sources": fingerprint_files(sources),
        "capabilities": sorted(bundle.capabilities),
        "model_config": bundle.model.config.to_dict(),
        "generation_config": bundle.model.generation_config.to_dict(),
        "tokenizer": {"vocab_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
                      "special_tokens": tokenizer.special_tokens_map,
                      "model_max_length": tokenizer.model_max_length,
                      "init_sha256": hashlib.sha256(json.dumps(tokenizer.init_kwargs, sort_keys=True, default=str).encode()).hexdigest(),
                      "slow_files": slow_files,
                      "add_bos_token": getattr(tokenizer, "add_bos_token", None),
                      "add_eos_token": getattr(tokenizer, "add_eos_token", None),
                      "clean_up_tokenization_spaces": getattr(tokenizer, "clean_up_tokenization_spaces", None),
                      "chat_template": getattr(tokenizer, "chat_template", None),
                      "padding_side": tokenizer.padding_side,
                      "truncation_side": tokenizer.truncation_side,
                      "backend_sha256": hashlib.sha256(tok_backend.to_str().encode()).hexdigest() if tok_backend else None},
        "resume_safe": bool(factory and bundle.version and bundle.checkpoint_id and source_available),
    }
    # Normalize tuples etc. before comparing a new object with a JSON manifest.
    config.model_provenance = json.loads(json.dumps(provenance, allow_nan=False))
    config.resolved_bundle = bundle
    return bundle


def resolve_model_adapter(config, lm, *, collection=False):
    """Resolve only requested capabilities; scoring alone needs no adapter."""
    from .adapters import ModelAdapter, ModulePaths, resolve_adapter
    requested = set(() if collection else config.signals)
    if config.save_hidden:
        requested.add("hidden")
    if config.save_attention:
        requested.add("attention")
    if config.resolved_hooks:
        requested.add("hooks")
    if not requested:
        return None
    bundle = config.resolved_bundle
    if bundle:
        missing = requested - set(bundle.capabilities)
        if missing:
            raise ValueError(f"custom model does not support requested signals: {sorted(missing)}")
    if bundle and bundle.adapter_factory:
        if config.adapter:
            raise ValueError("--adapter conflicts with ModelBundle.adapter_factory")
        adapter = bundle.adapter_factory(lm.model)
        if isinstance(adapter, ModulePaths):
            adapter = resolve_adapter(lm.model, paths=adapter, require_decode="logit_lens" in requested)
        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter_factory must return ModelAdapter or ModulePaths")
    else:
        adapter = resolve_adapter(lm.model, config.adapter, require_decode="logit_lens" in requested)
    if "attention" in requested and (not any(adapter.attn_modules) or not any(adapter.v_projs)):
        raise ValueError("attention collection requires attention and value projection modules")
    return adapter


def save_effective_prompts(samples_by_task, lm):
    """Save the formatted contexts so replay and report use the scored protocol.

    The formatter contract is deterministic/stateless. Keep originals for audit,
    and mark effective contexts so replay never applies the formatter twice.
    """
    if getattr(lm, "prompt_transform", None) is None:
        return
    import copy
    from lm_eval.utils import hash_string
    for samples in samples_by_task.values():
        for sample in samples:
            arguments = sample.get("arguments", [])
            sample["original_arguments"] = copy.deepcopy(arguments)
            sample["original_prompt_hash"] = sample.get("prompt_hash")
            prepared = [[lm.prepare_context(args[0]), *args[1:]] for args in arguments]
            sample["arguments"] = prepared
            sample["model_prompt_prepared"] = True
            if prepared:
                sample["prompt_hash"] = hash_string(prepared[0][0])
