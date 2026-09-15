"""Where the interesting modules live inside a given model.

There is exactly one hook implementation and one set of reducers.
What differs between architectures is almost entirely *where things are*: the block list, the final norm, the head and whether it has a bias, logit softcapping, and how GQA groups key/value heads.
Those differences are data, held here, so that `recorder.py` and `reducers.py` never branch on a model type.

Adding a model means adding a few lines to `MODEL_PATHS`, not writing new hooks.
If per-model hooks existed, the reducer calls would be duplicated once per model and a change to one of them would quietly not apply to the others - while every run still finished successfully.

Dispatch is on `config.model_type`.
An unregistered model type fails at startup rather than being guessed at; `--adapter` is the escape hatch for `trust_remote_code` models whose module paths are non-standard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn


# --------------------------------------------------------------------------
# Per-architecture module paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModulePaths:
    """Dotted attribute paths, relative to the unwrapped model.

    Attributes:
        blocks: the `nn.ModuleList` of transformer blocks.
        final_norm: the norm applied just before the head.
        lm_head: the unembedding. May be several candidate paths, tried in order,
            for an architecture whose head is not named the same thing in both
            pinned transformers versions. GPT-NeoX is the only such case today.
        attention: attention submodule, relative to one block.
        value_proj: value projection, relative to one block.
            On architectures that fuse q, k and v into one linear layer this is that layer, and `value_layout` says how to cut the value part out of its output.
        value_layout: how to read value vectors out of `value_proj`'s output.

            - "separate": the output is already only values, (.., n_kv*head_dim).
            - "fused_qkv_contiguous": [all q | all k | all v], so the values are the last n_kv*head_dim columns.
              Phi-3 does this.
            - "fused_qkv_per_head": the output reshapes to (.., heads, 3*head_dim) and q/k/v are interleaved *within each head*, so the values are the last head_dim of every head's block.
              GPT-NeoX does this, and slicing the tail of the flat row instead would silently return a mixture of other heads' queries and keys.
        last_hidden_is_normed: whether `output_hidden_states=True` returns a *final-normed* last element.
            HF pre-norm decoders do, and getting this wrong makes the logit lens apply the final norm twice on the last layer only - which looks plausible in a plot and is wrong.

    Example:
        >>> MODEL_PATHS["llama"].blocks
        'model.layers'
    """

    blocks: str
    final_norm: str
    lm_head: str | tuple[str, ...]
    attention: str = "self_attn"
    value_proj: str = "self_attn.v_proj"
    value_layout: str = "separate"
    last_hidden_is_normed: bool = True


#: The standard HF pre-norm decoder layout, shared by most Llama-derived models.
_LLAMA_STYLE = ModulePaths(blocks="model.layers", final_norm="model.norm", lm_head="lm_head")

MODEL_PATHS: dict[str, ModulePaths] = {
    "llama": _LLAMA_STYLE,
    "mistral": _LLAMA_STYLE,
    "mixtral": _LLAMA_STYLE,
    "qwen2": _LLAMA_STYLE,
    "qwen3": _LLAMA_STYLE,
    "qwen3_moe": _LLAMA_STYLE,
    "gemma": _LLAMA_STYLE,
    "gemma2": _LLAMA_STYLE,
    "gemma3_text": _LLAMA_STYLE,
    "olmo2": _LLAMA_STYLE,
    "cohere": _LLAMA_STYLE,
    "smollm3": _LLAMA_STYLE,
    # Qwen3.5 is a hybrid: only some blocks carry `self_attn`, the rest use a linear-attention module with a different internal layout.
    # The residual path is unaffected - `model.layers` still emits L+1 hidden states - so the logit lens and the similarity matrix work as usual; the attention and value-norm signals simply have no rows for the linear-attention blocks, which is what a sparse `block` axis is for.
    # Only `qwen3_5_text` is registered, the same split Gemma 3 uses. HFLM loads a Qwen3.5 repo through AutoModelForCausalLM as `Qwen3_5ForCausalLM`, which reports `qwen3_5_text`.
    # The wrapper `qwen3_5` is reported only by `Qwen3_5ForConditionalGeneration`, whose blocks sit at `model.language_model.layers` and whose head counts live in `text_config`, so these paths would never resolve on it.
    "qwen3_5_text": _LLAMA_STYLE,
    # Gemma 4 is the exception to that split. transformers maps a `gemma4` repo config to `Gemma4ForConditionalGeneration` for AutoModelForCausalLM as well, so HFLM loads the multimodal wrapper and the loaded model reports `gemma4`.
    # Its text blocks sit under the wrapper's `language_model`, the head is the wrapper's own, and head counts and softcapping live on the text config, which `resolve_adapter` reads.
    # A text-only checkpoint loads as `Gemma4ForCausalLM`, reports `gemma4_text`, and is Llama-shaped.
    # Its blocks differ in attention shape - full-attention heads are twice as wide as sliding ones - and the last `num_kv_shared_layers` blocks reuse an earlier block's keys and values and own no `v_proj`: they carry attention weights but no value-norm rows.
    "gemma4_text": _LLAMA_STYLE,
    "gemma4": ModulePaths(
        blocks="model.language_model.layers",
        final_norm="model.language_model.norm",
        lm_head="lm_head",
    ),
    "phi": ModulePaths(
        blocks="model.layers",
        # Phi calls its final norm `final_layernorm`, not `norm`.
        final_norm="model.final_layernorm",
        lm_head="lm_head",
    ),
    "phi3": ModulePaths(
        blocks="model.layers",
        final_norm="model.norm",
        lm_head="lm_head",
        # Phi-3 fuses q/k/v into one projection, laid out as [q | k | v].
        value_proj="self_attn.qkv_proj",
        value_layout="fused_qkv_contiguous",
    ),
    "gpt_neox": ModulePaths(
        blocks="gpt_neox.layers",
        final_norm="gpt_neox.final_layer_norm",
        # Renamed in transformers 5.x; 4.x, which this project also pins, still has
        # `embed_out`. Everything else about GPT-NeoX survived the rename.
        lm_head=("lm_head", "embed_out"),
        attention="attention",
        # GPT-NeoX interleaves q/k/v inside each head, not across the row.
        value_proj="attention.query_key_value",
        value_layout="fused_qkv_per_head",
    ),
}


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------


@dataclass
class ModelAdapter:
    """The resolved answer to "where is what" for one loaded model.

    Attributes:
        model_type: `config.model_type`, the key this adapter was resolved by.
        blocks: transformer blocks, index 0..L-1.
        decoder: the module that owns the block list and returns `hidden_states` - `model.model` on a Llama-style checkpoint.
            Hooking it is how the residual stream is captured on both the loglikelihood and the generate path, so both get the *same* L+1 tensors.
        attn_modules: attention module per block; None for a block that has no attention (hybrid architectures), which makes the `block` axis non-contiguous.
        v_projs: value projection per block; None where absent.
        decode_stack: `(n_layers, n_pos, d) -> (n_layers, n_pos, vocab)`, the model's own path from hidden state to logits.
        extract_value: `(tensor, block_idx) -> values`, pulls the value vectors out of whatever `value_proj` emits, per `ModulePaths.value_layout` and that block's attention shape.
        n_blocks: L.
        n_residual: L+1, matching what `output_hidden_states=True` returns.
        n_heads / n_kv_heads / head_dim: attention shape.
            `n_kv_heads` and `head_dim` are None when blocks differ; `block_shapes` then holds each block's.
        vocab_size: size of the head's output.
        tie_word_embeddings: recorded because on tied models `layer 0` decodes back to the input token, which must not be read as a prediction.
        paths: the `ModulePaths` used, recorded in the manifest.
        lm_head_path: the head path that actually resolved, which is what the manifest
            should say when the entry offered several candidates.
        block_shapes: `(n_heads, n_kv_heads, head_dim)` per block.
    """

    model_type: str
    decoder: nn.Module
    blocks: list[nn.Module]
    attn_modules: list[nn.Module | None]
    v_projs: list[nn.Module | None]
    decode_stack: Callable[[torch.Tensor], torch.Tensor]
    extract_value: Callable[..., torch.Tensor]
    n_blocks: int
    n_residual: int
    n_heads: int
    n_kv_heads: int | None
    head_dim: int | None
    vocab_size: int
    tie_word_embeddings: bool
    paths: ModulePaths
    lm_head_path: str = ""
    # Root is used only to delimit real custom observations, never to inspect containers.
    root_model: nn.Module | None = None
    block_shapes: list[tuple[int, int, int]] = field(default_factory=list)

    def attention_shape(self, block_idx: int) -> tuple[int, int, int]:
        """`(n_heads, n_kv_heads, head_dim)` of one block.

        Example:
            >>> adapter.attention_shape(5)                       # doctest: +SKIP
            (8, 2, 512)
        """
        if self.block_shapes:
            return self.block_shapes[block_idx]
        return (self.n_heads, self.n_kv_heads, self.head_dim)

    def value_shapes(self) -> dict[int, tuple[int, int, int]]:
        """Per-block shapes of the blocks that have a value projection, only when blocks differ.

        Empty on a uniform model, so its reducer configuration and manifest keep the scalar form
        every existing run was recorded with.
        """
        if self.n_kv_heads is not None and self.head_dim is not None:
            return {}
        return {i: shape for i, shape in enumerate(self.block_shapes) if self.v_projs[i] is not None}

    def manifest_entry(self) -> dict[str, Any]:
        """What goes into the manifest about the decode path.

        Example:
            >>> adapter.manifest_entry()["final_norm_path"]      # doctest: +SKIP
            'model.norm'
        """
        entry = {
            "model_type": self.model_type,
            "n_blocks": self.n_blocks,
            "n_hidden_states": self.n_residual,
            "layer_index_convention": "residual_input",
            "attn_index_convention": "block_output",
            "decoder_path": self.paths.blocks.rsplit(".", 1)[0],
            "value_proj_path": self.paths.value_proj,
            "value_layout": self.paths.value_layout,
            "final_norm_path": self.paths.final_norm,
            "lm_head_path": self.lm_head_path or self.paths.lm_head,
            "last_hidden_is_normed": self.paths.last_hidden_is_normed,
            "tie_word_embeddings": self.tie_word_embeddings,
            "vocab_size": self.vocab_size,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "blocks_with_attention": [i for i, m in enumerate(self.attn_modules) if m is not None],
        }
        if self.n_kv_heads is None or self.head_dim is None:
            entry["attention_shapes_by_block"] = [list(shape) for shape in self.block_shapes]
            entry["blocks_with_value_proj"] = [i for i, v in enumerate(self.v_projs) if v is not None]
        return entry


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def unwrap_model(model: nn.Module) -> nn.Module:
    """Peel wrappers until the real transformer is exposed.

    A `PeftModel` puts one or two extra objects between us and the block list, so the module paths have to be resolved against the unwrapped object.
    The hooks themselves are unaffected: a LoRA-replaced `v_proj` still reports the merged base+delta output, because we capture the module's output.

    Only known wrapper types are unwrapped.
    Anything looser - "unwrap while the object has `.base_model`", say - also unwraps ordinary models, because every `PreTrainedModel` has that attribute; that would peel a `GPTNeoXForCausalLM` down to its decoder and then fail to resolve any path against it.

    Example:
        >>> unwrap_model(peft_model) is peft_model.get_base_model()   # doctest: +SKIP
        True
        >>> unwrap_model(llama_for_causal_lm) is llama_for_causal_lm  # doctest: +SKIP
        True
    """
    parallel_wrappers = {"DataParallel", "DistributedDataParallel"}
    peft_wrappers = {"LoraModel", "AdaLoraModel", "IA3Model", "LoHaModel", "LoKrModel"}

    seen: set[int] = set()
    current = model
    while id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if hasattr(current, "get_base_model"):                  # peft.PeftModel
            current = current.get_base_model()
        elif name in parallel_wrappers and hasattr(current, "module"):
            current = current.module
        elif name in peft_wrappers and hasattr(current, "model"):
            current = current.model
        else:
            break
    return current


def _resolve_path(root: nn.Module, path: str | tuple[str, ...]) -> Any:
    """Follow a dotted attribute path, or raise a message naming what is missing.

    Example:
        >>> _resolve_path(model, "model.norm")             # doctest: +SKIP
        LlamaRMSNorm((4096,), eps=1e-05)
    """
    return _resolve_named(root, path)[1]


def _resolve_named(root: nn.Module, path: str | tuple[str, ...]) -> tuple[str, Any]:
    """Resolve the first candidate path that exists, and say which one answered.

    Several candidates exist because a path is not always the same in both pinned
    transformers versions: GPT-NeoX's head is `embed_out` in 4.x and `lm_head` in 5.x.
    Resolving by name rather than by version number keeps one registry entry valid in
    both environments, and the manifest records the name that actually resolved rather
    than the list of guesses. A single string stays a single string, so a model whose
    path is simply wrong still fails at startup with one name in the message.
    """
    candidates = (path,) if isinstance(path, str) else tuple(path)
    missing: list[str] = []
    for candidate in candidates:
        current: Any = root
        for part in candidate.split("."):
            if not hasattr(current, part):
                missing.append(
                    f"{candidate!r} ({type(current).__name__} has no attribute {part!r})")
                break
            current = getattr(current, part)
        else:
            return candidate, current
    raise AttributeError(
        f"cannot resolve {' or '.join(missing)} on {type(root).__name__}. "
        "Register the correct paths in adapters.MODEL_PATHS, or pass --adapter."
    )


def _build_decode_stack(
    final_norm: nn.Module, lm_head: nn.Module, paths: ModulePaths, config: Any
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the hidden-state -> logits function, reusing the model's own modules.

    We deliberately do not reimplement this path.
    Architectures differ in what sits between the final norm and the logits (head bias, logit softcapping, output scaling), and a reimplementation would disagree with the model's real logits at the last layer - which is precisely the check that tells us the lens is wired up correctly.
    Both of those extras are read off the config rather than the model type: `final_logit_softcapping` (Gemma 2, 3 and 4) and `logit_scale` (Cohere).
    `config` is the text config: a multimodal wrapper (Gemma 4) keeps softcapping there, not on its own config.

    The one place we deviate is the last element: HF returns it already final-normed, so feeding it through the norm again would norm it twice.

    The returned function always expects the **whole** L+1 stack, in residual order, because that is what tells it which element is the pre-normed last one.

    Example:
        >>> decode = _build_decode_stack(norm, head, paths, config)   # doctest: +SKIP
        >>> decode(torch.randn(33, 1, 4096)).shape
        torch.Size([33, 1, 151936])
    """
    softcap = getattr(config, "final_logit_softcapping", None)
    # Cohere scales its logits - upstream calls it "the main diff from Llama" - and it is
    # the only registered architecture that does. Leaving it out makes the lens disagree
    # with the model by a constant factor at every layer, which is what
    # `check_decode_identity` refuses at startup rather than letting it into a plot.
    scale = getattr(config, "logit_scale", None)

    def decode_stack(stack: torch.Tensor) -> torch.Tensor:
        if paths.last_hidden_is_normed:
            # Norm everything but the last element, which already has it.
            head_input = torch.cat([final_norm(stack[:-1]), stack[-1:]], dim=0)
        else:
            head_input = final_norm(stack)
        logits = lm_head(head_input)
        if scale:
            logits = logits * scale
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    return decode_stack


def _build_value_extractor(
    paths: ModulePaths, n_heads: int, n_kv_heads: int, head_dim: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the function that turns `value_proj`'s output into value vectors.

    Kept here rather than in the hook, because which columns hold the values is a per-architecture fact - exactly the kind of thing `adapters.py` exists to hold as data.

    Returns:
        `(.., width) -> (.., n_kv_heads * head_dim)`.

    Example:
        >>> paths = ModulePaths("l", "n", "h", value_layout="fused_qkv_per_head")
        >>> extract = _build_value_extractor(paths, n_heads=2, n_kv_heads=2, head_dim=2)
        >>> # two heads, each laid out as [q0 q1 | k0 k1 | v0 v1]
        >>> extract(torch.arange(12).reshape(1, 12)).tolist()
        [[4, 5, 10, 11]]
    """
    value_width = n_kv_heads * head_dim

    def separate(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] != value_width:
            raise ValueError(
                f"expected the value projection to emit {value_width} columns "
                f"(n_kv_heads * head_dim) but it emitted {tensor.shape[-1]}; "
                "check value_proj / value_layout for this architecture"
            )
        return tensor

    def fused_contiguous(tensor: torch.Tensor) -> torch.Tensor:
        # [ all queries | all keys | all values ]
        return tensor[..., -value_width:]

    def fused_per_head(tensor: torch.Tensor) -> torch.Tensor:
        # (.., heads * 3 * head_dim) -> (.., heads, 3 * head_dim), values last.
        reshaped = tensor.view(*tensor.shape[:-1], n_heads, 3 * head_dim)
        return reshaped[..., 2 * head_dim :].reshape(*tensor.shape[:-1], n_heads * head_dim)

    extractors = {
        "separate": separate,
        "fused_qkv_contiguous": fused_contiguous,
        "fused_qkv_per_head": fused_per_head,
    }
    if paths.value_layout not in extractors:
        raise KeyError(
            f"unknown value_layout {paths.value_layout!r}; "
            f"known: {sorted(extractors)}"
        )
    return extractors[paths.value_layout]


def _attention_shapes(config: Any, n_blocks: int) -> list[tuple[int, int, int]]:
    """`(n_heads, n_kv_heads, head_dim)` for every block, read from the (text) config.

    transformers 5.x gives every config a `per_layer_config` view, uniform model or not; a layer entry leaves `head_dim` None when the model derives it (OLMo-2), which falls back to `hidden_size // n_heads` exactly as the global rule does.
    On a config whose layers really differ (Gemma 4) the global `head_dim` cannot be read at all: it raises an error that is not an AttributeError, so `getattr` with a default does not catch it. Older configs without the view take the global rule.
    """
    try:
        per_layer = config.per_layer_config
    except AttributeError:
        per_layer = None
    if per_layer is not None:
        global_heads = int(getattr(config, "num_attention_heads", 1))
        shapes = []
        for index in range(n_blocks):
            layer = per_layer[index]
            heads = int(getattr(layer, "num_attention_heads", None) or global_heads)
            kv_heads = int(getattr(layer, "num_key_value_heads", None) or heads)
            head_dim = int(getattr(layer, "head_dim", None) or config.hidden_size // heads)
            shapes.append((heads, kv_heads, head_dim))
        return shapes
    n_heads = int(getattr(config, "num_attention_heads", 1))
    n_kv_heads = int(getattr(config, "num_key_value_heads", None) or n_heads)
    head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // n_heads)
    return [(n_heads, n_kv_heads, head_dim)] * n_blocks


def resolve_adapter(model: nn.Module, adapter_name: str | None = None, *, paths: ModulePaths | None = None, require_decode: bool = True) -> ModelAdapter:
    """Build the `ModelAdapter` for a loaded model.

    Args:
        model: the model as lm-eval loaded it, wrappers and all.
        paths: external module paths, without mutating the global registry.
        require_decode: False skips final norm/head resolution for non-lens signals.
        adapter_name: force a specific `MODEL_PATHS` entry.
            The escape hatch for `trust_remote_code` models whose paths do not match their `model_type`.

    Raises:
        KeyError: the model type is not registered.
            We fail at startup rather than guessing, because a wrong guess produces signals attached to the wrong modules and nothing downstream would notice.

    Example:
        >>> adapter = resolve_adapter(model)                # doctest: +SKIP
        >>> adapter.n_blocks, adapter.n_residual
        (32, 33)
    """
    base = unwrap_model(model)
    config = base.config
    model_type = adapter_name or getattr(config, "model_type", None)
    if paths is None and model_type not in MODEL_PATHS:
        raise KeyError(
            f"no adapter registered for model_type {model_type!r}. "
            f"Known: {sorted(MODEL_PATHS)}. "
            "Add an entry to adapters.MODEL_PATHS or pass --adapter <known type>."
        )
    paths = paths if paths is not None else MODEL_PATHS[model_type]
    # A multimodal wrapper keeps head counts, vocabulary and softcapping on its text config;
    # on a text model `get_text_config()` is the config itself.
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config

    # "model.layers" -> the block list lives on "model", which is the module that returns hidden_states.
    # Deriving it avoids a second path to maintain.
    decoder_path = paths.blocks.rsplit(".", 1)[0]
    decoder = _resolve_path(base, decoder_path) if decoder_path else base
    blocks = list(_resolve_path(base, paths.blocks))
    final_norm = _resolve_path(base, paths.final_norm) if require_decode else None
    lm_head_path, lm_head = (_resolve_named(base, paths.lm_head) if require_decode else ("", None))

    # A block without attention (hybrid architectures) is allowed; it simply contributes no row on the `block` axis.
    attn_modules: list[nn.Module | None] = []
    v_projs: list[nn.Module | None] = []
    for block in blocks:
        attn_modules.append(_optional_path(block, paths.attention))
        v_projs.append(_optional_path(block, paths.value_proj))

    block_shapes = _attention_shapes(text_config, len(blocks))
    uniform = len(set(block_shapes)) <= 1
    n_heads = int(getattr(text_config, "num_attention_heads", 1))
    if uniform and block_shapes:
        n_heads, n_kv_heads, head_dim = block_shapes[0]
    elif uniform:
        n_kv_heads = int(getattr(text_config, "num_key_value_heads", None) or n_heads)
        head_dim = int(getattr(text_config, "head_dim", None) or text_config.hidden_size // n_heads)
    else:
        n_kv_heads = head_dim = None
    extractors = {shape: _build_value_extractor(paths, *shape) for shape in set(block_shapes)}
    uniform_extractor = (_build_value_extractor(paths, n_heads, n_kv_heads, head_dim)
                         if uniform else None)

    def extract_value(tensor: torch.Tensor, block_idx: int | None = None) -> torch.Tensor:
        if uniform_extractor is not None:
            return uniform_extractor(tensor)
        if block_idx is None:
            raise ValueError("this model's blocks differ in attention shape; pass the block index")
        return extractors[block_shapes[block_idx]](tensor)

    return ModelAdapter(
        model_type=model_type,
        decoder=decoder,
        blocks=blocks,
        attn_modules=attn_modules,
        v_projs=v_projs,
        decode_stack=_build_decode_stack(final_norm, lm_head, paths, text_config) if require_decode else _decode_unavailable,
        extract_value=extract_value,
        n_blocks=len(blocks),
        n_residual=len(blocks) + 1,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vocab_size=int(getattr(lm_head, "out_features", text_config.vocab_size)),
        tie_word_embeddings=bool(getattr(text_config, "tie_word_embeddings", False)),
        paths=paths,
        lm_head_path=lm_head_path,
        root_model=base,
        block_shapes=block_shapes,
    )


def _decode_unavailable(stack):
    """Fail if a caller uses a decode path that was deliberately not resolved."""
    raise ValueError("logit lens was not selected; resolve the adapter with require_decode=True")


def _optional_path(root: nn.Module, path: str) -> nn.Module | None:
    """Like `_resolve_path`, but returns None instead of raising."""
    try:
        return _resolve_path(root, path)
    except AttributeError:
        return None


def check_decode_identity(
    adapter: ModelAdapter,
    model: nn.Module,
    input_ids: torch.Tensor,
    tolerance: float = 1e-2,
) -> dict[str, float]:
    """Verify the lens decode path reproduces the model's real logits.

    Runs one forward pass, decodes the full residual stack exactly the way the logit lens does, and compares the last layer's result against the logits the model itself produced.
    If they disagree, the final norm is in the wrong place or the head was resolved incorrectly - and every logit lens number in the run would be quietly wrong.

    This doubles as the usability test for quantized models: if the identity holds, the hooks are seeing dequantized activations and the run is usable.

    The tolerance is **relative to the logit scale**, not absolute.
    Logit magnitudes differ enormously between models - about 21 for SmolLM2-135M and about 837 for pythia-160m - so a fixed absolute bound is really a different test on every model, and one loose enough for pythia in bf16 would be far too loose for SmolLM2.
    The two paths use the same weights and inputs and differ only in how the matmul is tiled, so what is being allowed for here is accumulation order, which scales with the values.

    Args:
        input_ids: a small batch, e.g. `torch.tensor([[1, 2, 3]])` on the model's device.
        tolerance: allowed difference as a fraction of the largest logit.

    Returns:
        `{"max_abs_diff", "max_rel_diff", "logit_scale"}`, all recorded in the manifest so a marginal run can be judged after the fact.

    Raises:
        RuntimeError: if the relative difference exceeds `tolerance`.
            A misplaced final norm changes the vector completely, so it fails by orders of magnitude rather than marginally.

    Example:
        >>> check_decode_identity(adapter, model, ids)       # doctest: +SKIP
        {'max_abs_diff': 0.0014, 'max_rel_diff': 1.7e-06, 'logit_scale': 837.05}
    """
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        # (n_layers, batch, d): the same stack the logit lens builds.
        stack = torch.stack([hidden[:, -1, :] for hidden in out.hidden_states], dim=0)
        ours = adapter.decode_stack(stack)[-1]                # (batch, vocab)
        theirs = out.logits[:, -1, :]
        absolute = float((ours.float() - theirs.float()).abs().max())
        # max(1.0, ...) keeps a degenerate all-zero-logit model from dividing by something tiny and turning a harmless difference into a failure.
        scale = max(1.0, float(theirs.float().abs().max()))
    relative = absolute / scale
    if relative > tolerance:
        raise RuntimeError(
            f"logit lens decode path does not match the model's own logits: "
            f"max abs diff {absolute:.4g} against a logit scale of {scale:.4g} "
            f"is {relative:.3g} relative, over the {tolerance} tolerance. "
            f"Check final_norm/lm_head paths and last_hidden_is_normed for "
            f"model_type {adapter.model_type!r}."
        )
    return {"max_abs_diff": absolute, "max_rel_diff": relative, "logit_scale": scale}
