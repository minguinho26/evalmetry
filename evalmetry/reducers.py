"""Reducers: turn the tensors a hook catches into a handful of numbers.

A reducer is the only place a raw activation is allowed to live.
Hooks hand it a tensor, it keeps just what it needs, and it drops the reference immediately - vocab-sized logits and full hidden states cannot be written out at evaluation scale, so the shrinking has to happen inside the forward pass.

One signal = one reducer.
Adding a new signal means adding a class here; no hook code and no lm-eval glue has to change.

Lifecycle, driven by `recorder.py`:

    update(index, tensor, ctx)   once per captured tensor (per layer, per hook) end_forward(ctx)             every layer of one forward pass has arrived; do the maths now and let the tensors go finalize()                   the document is done; hand back rows and reset

The `update` / `finalize` pair is the reducer interface; `end_forward` exists because a generate task runs one forward pass per decoding step, and each step has to be reduced on the spot rather than piled up until the document ends - the whole point is never to hold vocab-sized tensors.

Every reducer declares three things about its output, which `report.py` reads to decide what may share an axis:

    comparable_across_vocab   may models with different vocabularies be overlaid? layer_domain              "residual" (0..L, column `layer`) or "block" (0..L-1, column `block`) version                   bump whenever the *definition* of the number changes, even if the column names do not
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch


# : Positions are decoded in chunks so the transient (chunk x layers x vocab) : logit tensor stays bounded.  33 layers x 150k vocab x fp32 is ~20 MB per : position, which is fine one at a time and not fine for a long scored span.
LENS_POSITION_CHUNK = 4


@dataclass
class ForwardContext:
    """Everything the reducers need to know about the forward pass in flight.

    One instance describes one forward pass of one (document, choice).
    For a loglikelihood request that is the whole document; for a generate request there is one context per decoding step.

    Attributes:
        task_name: lm-eval task name.
        doc_id: lm-eval document id.
        choice_idx: which choice of a multiple-choice document, else 0.
        steps: `step` axis value for each position carried by this forward. loglikelihood: 0..contlen-1, the positions lm-eval actually scores. generate: a single decoding step, e.g. [7].
        positions: absolute position in the input sequence for each step.
        seq_indices: index along the tensor's sequence axis for each step.
            Usually the same as `positions`, but a generate decode step feeds a single-token tensor, where the only valid index is 0.
        batch_row: which row of the batched tensors belongs to this document.
        target_token_ids: gold continuation token at each step, or None for a generate task where there is no gold token to rank.
        step_token_ids: token to report in the `steps` table.
            Identical to `target_token_ids` for loglikelihood; for generate it is the token that was actually emitted, which sampling can make differ from the last layer's top-1.
        n_residual: number of residual entries, L+1.
        n_blocks: number of transformer blocks, L.
        task_kind: "loglikelihood" or "generate".
            Reducers that store data sparsely along the step axis (attention weights) use it to decide which steps to keep.
        shared: scratch space reducers may use to pass intermediate values to each other within one forward pass.
            Cleared by the recorder afterwards, so nothing survives into the next pass.

    Example:
        >>> ctx = ForwardContext("xnli_ko", 7, 1, steps=[0], positions=[41],
        ...                      target_token_ids=[9891], n_residual=33, n_blocks=32)
        >>> ctx.key
        ('xnli_ko', 7, 1)
        >>> ctx.n_positions
        1
        >>> ctx.seq_indices
        [41]
    """

    task_name: str
    doc_id: int
    choice_idx: int
    steps: list[int]
    positions: list[int]
    target_token_ids: list[int] | None
    n_residual: int
    n_blocks: int
    task_kind: str = "loglikelihood"
    batch_row: int = 0
    seq_indices: list[int] | None = None
    step_token_ids: list[int] | None = None
    shared: dict[str, Any] = field(default_factory=dict)
    # Actual unpadded model input length, not the length of the scored span.
    # Module statistics need this to exclude padding across every input position.
    input_length: int | None = None
    # Tokens removed from the left before the actual model input; None means unknown.
    input_offset: int | None = None

    def __post_init__(self) -> None:
        # By default a step reads the tensor at its own absolute position; only incremental decoding needs a different mapping.
        if self.seq_indices is None:
            self.seq_indices = list(self.positions)
        if self.step_token_ids is None:
            self.step_token_ids = self.target_token_ids

    @property
    def key(self) -> tuple[str, int, int]:
        """The (task_name, doc_id, choice_idx) triple every row is keyed by."""
        return (self.task_name, self.doc_id, self.choice_idx)

    @property
    def n_positions(self) -> int:
        return len(self.steps)

    def axis_columns(self, step_index: int) -> dict[str, Any]:
        """The four shared axis columns for one position of this forward pass.

        Example:
            >>> ctx = ForwardContext("t", 3, 0, [0, 1], [10, 11], None, 33, 32)
            >>> ctx.axis_columns(1)
            {'task_name': 't', 'doc_id': 3, 'choice_idx': 0, 'step': 1}
        """
        return {
            "task_name": self.task_name,
            "doc_id": self.doc_id,
            "choice_idx": self.choice_idx,
            "step": self.steps[step_index],
        }


class Reducer:
    """Base class.
Subclasses override `update`, `end_forward` and `finalize`.

    The defaults are no-ops so a reducer only has to implement the callbacks it actually cares about: a residual-stream reducer ignores the attention hooks, and an attention reducer ignores the residual ones.
    """

    #: Short identifier written into the manifest.
    name: str = "reducer"
    #: Bump when the meaning of the numbers changes.
    version: int = 1
    # : Which table in `storage.TABLES` the rows go to, or None for a reducer : that only writes tensor files.
    table: str | None = None
    #: "residual" (0..L) or "block" (0..L-1), or None if it has no layer axis.
    layer_domain: str | None = None
    #: Whether its values may be plotted next to a model with another vocab.
    comparable_across_vocab: bool = True
    #: Which hook stream feeds it: "residual", "attention" or "value".
    source: str = "residual"

    def config(self) -> dict[str, Any]:
        """Settings recorded in the manifest, so an old run stays interpretable."""
        return {}

    def descriptor(self) -> dict[str, Any]:
        """The manifest entry for this reducer.

        Example:
            >>> SimilarityReducer(n_residual=3).descriptor()["name"]
            'layer_similarity'
        """
        return {
            "name": self.name,
            "version": self.version,
            "table": self.table,
            "layer_domain": self.layer_domain,
            "comparable_across_vocab": self.comparable_across_vocab,
            "config": self.config(),
        }

    def update(self, index: int, tensor: torch.Tensor, ctx: ForwardContext) -> None:
        """One captured tensor.
`index` is a layer index or a block index."""

    def end_forward(self, ctx: ForwardContext) -> None:
        """Every tensor of this forward pass has arrived: reduce and release."""

    def finalize(self) -> list[dict[str, Any]]:
        """The document is finished: return its rows and clear internal state."""
        return []

    def pop_tensors(self) -> dict[str, torch.Tensor]:
        """Tensors to be written to safetensors for this document, if any.

        Only the opt-in dumpers return anything here; the recorder hands the result to `storage.TensorStore`.
        """
        return {}


# --------------------------------------------------------------------------
# Always-on signals
# --------------------------------------------------------------------------


class LogitLensReducer(Reducer):
    """Logit lens: read every layer's residual stream through the model's own head.

    For each (step, layer) it records the top-1 token and, for loglikelihood tasks, where the gold token sits in that layer's ranking.

    Why one reducer and not two Plan section 4 lists "logit lens" and "gold token rank" as separate signals.
        They are computed together here because both are functions of the *same* layer-wise logit tensor, and that tensor is the one thing we are not allowed to keep alive (~20 MB per position at 33 layers x 150k vocab).
        Handing it to a second reducer would mean either holding it longer or paying for a second unembedding GEMM.
        The two signals therefore share this reducer's `version`.

    Why k is fixed at 1 With k=1 there is exactly one row per (doc_id, choice_idx, step, layer), so the table needs no extra k axis and lines up with every other signal.
        "Where is the gold token?" is answered by `target_rank`, not by a top-k list.
        Raising k adds an axis, so it would need a `version` bump.

    Args:
        decode_stack: `(n_layers, n_pos, d) -> (n_layers, n_pos, vocab)`.
            Supplied by the adapter, because the path from a hidden state to logits (final norm, head bias, logit softcapping, output scaling) differs per architecture and must not be reimplemented here.
        tokenizer: used only to turn ids into strings and to flag specials.
        vocab_size: denominator of `target_percentile`.

    Example of one produced row:
        {"task_name": "xnli_ko", "doc_id": 7, "choice_idx": 1, "step": 0, "layer": 20, "lens_token_id": 9891, "lens_prob": 0.31, "lens_token": "Ġyes", "lens_is_special": False, "target_rank": 2, "target_percentile": 1.3e-05}
    """

    name = "logit_lens"
    version = 1
    table = "signals"
    layer_domain = "residual"
    comparable_across_vocab = False  # token ids and probabilities are vocab-bound
    source = "residual"

    def __init__(
        self,
        decode_stack: Callable[[torch.Tensor], torch.Tensor],
        tokenizer: Any,
        vocab_size: int,
        position_chunk: int = LENS_POSITION_CHUNK,
    ) -> None:
        self._decode_stack = decode_stack
        self._tokenizer = tokenizer
        self._vocab_size = int(vocab_size)
        self._position_chunk = position_chunk
        self._special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self._token_text: dict[int, str] = {}  # id -> string, cached for the whole run
        self._layers: dict[int, torch.Tensor] = {}
        self._rows: list[dict[str, Any]] = []

    def config(self) -> dict[str, Any]:
        return {"top_k": 1, "vocab_size": self._vocab_size, "position_chunk": self._position_chunk}

    def update(self, index: int, tensor: torch.Tensor, ctx: ForwardContext) -> None:
        # `tensor` is already sliced down to the scored positions, (n_pos, d), so what we hold on to between here and end_forward is small.
        self._layers[index] = tensor

    def end_forward(self, ctx: ForwardContext) -> None:
        if not self._layers:
            return
        stack = torch.stack([self._layers[i] for i in range(ctx.n_residual)], dim=0)
        self._layers.clear()

        for start in range(0, ctx.n_positions, self._position_chunk):
            stop = min(start + self._position_chunk, ctx.n_positions)
            chunk = stack[:, start:stop, :]  # (n_layers, chunk, d)
            # One GEMM for the whole layer stack.
            # Looping layer by layer costs noticeably more on generate tasks, where this runs every step.
            logits = self._decode_stack(chunk).float()  # (n_layers, chunk, vocab)
            logprobs = torch.log_softmax(logits, dim=-1)
            top_prob, top_id = logprobs.max(dim=-1)
            top_prob = top_prob.exp()

            gold_rank = None
            if ctx.target_token_ids is not None:
                gold = torch.tensor(
                    ctx.target_token_ids[start:stop], device=logits.device, dtype=torch.long
                )
                gold_logit = logits.gather(
                    -1, gold.view(1, -1, 1).expand(logits.shape[0], -1, 1)
                )
                # 0-based rank: how many tokens beat the gold token.
                # Ties count as not beating it, so the rank is the optimistic one.
                gold_rank = (logits > gold_logit).sum(dim=-1)

            del logits, logprobs
            self._emit(ctx, start, stop, top_id.cpu(), top_prob.cpu(),
                       None if gold_rank is None else gold_rank.cpu())

    def _emit(
        self,
        ctx: ForwardContext,
        start: int,
        stop: int,
        top_id: torch.Tensor,
        top_prob: torch.Tensor,
        gold_rank: torch.Tensor | None,
    ) -> None:
        """Turn the reduced tensors of one position chunk into rows."""
        ids = top_id.tolist()
        probs = top_prob.tolist()
        ranks = gold_rank.tolist() if gold_rank is not None else None
        for local_pos, pos in enumerate(range(start, stop)):
            axis = ctx.axis_columns(pos)
            for layer in range(ctx.n_residual):
                token_id = int(ids[layer][local_pos])
                row = dict(axis)
                row.update(
                    layer=layer,
                    lens_token_id=token_id,
                    lens_prob=float(probs[layer][local_pos]),
                    lens_token=self._text_for(token_id),
                    lens_is_special=token_id in self._special_ids,
                    target_rank=None,
                    target_percentile=None,
                )
                if ranks is not None:
                    rank = int(ranks[layer][local_pos])
                    row["target_rank"] = rank
                    row["target_percentile"] = rank / self._vocab_size
                self._rows.append(row)

    def _text_for(self, token_id: int) -> str:
        """Tokenizer string for a token id, stored verbatim.

        Uses `convert_ids_to_tokens`, i.e. the raw vocabulary piece, so leading space markers ("Ġ") and partial byte pieces survive untouched.
        Cleaning them up is left to whoever draws the figure.
        """
        cached = self._token_text.get(token_id)
        if cached is None:
            try:
                cached = self._tokenizer.convert_ids_to_tokens(token_id)
            except Exception:  # tokenizers without the fast-tokenizer API
                cached = self._tokenizer.decode([token_id])
            cached = "" if cached is None else str(cached)
            self._token_text[token_id] = cached
        return cached

    def finalize(self) -> list[dict[str, Any]]:
        rows, self._rows = self._rows, []
        self._layers.clear()
        return rows


class SimilarityReducer(Reducer):
    """Cosine similarity between every pair of residual layers, per position.

    Stores the upper triangle including the diagonal, since the matrix is symmetric: 561 values for a 32-block model, about 2.2 KB per step in fp32.

    No centering is applied.
    The residual stream is only ever added to, so all tokens share a large common component and the cosines sit close to 1; read the *relative* structure between layer pairs, not the absolute value.
    Subtracting a corpus mean needs a second pass and is left as a TODO, which is exactly the kind of change that would require a `version` bump.

    Example of one produced row:
        {"task_name": "xnli_ko", "doc_id": 7, "choice_idx": 1, "step": 0, "layer_i": 3, "layer_j": 17, "cos": 0.94}
    """

    name = "layer_similarity"
    version = 1
    table = "similarity"
    layer_domain = "residual"
    comparable_across_vocab = True  # a cosine does not depend on vocabulary size
    source = "residual"

    def __init__(self, n_residual: int) -> None:
        # The (i, j) index pairs are fixed for a model, so build them once.
        rows, cols = torch.triu_indices(n_residual, n_residual, offset=0).tolist()
        self._pair_i = rows
        self._pair_j = cols
        self._layers: dict[int, torch.Tensor] = {}
        self._rows: list[dict[str, Any]] = []

    def config(self) -> dict[str, Any]:
        return {"centering": False, "store": "upper_triangle_with_diagonal"}

    def update(self, index: int, tensor: torch.Tensor, ctx: ForwardContext) -> None:
        self._layers[index] = tensor

    def end_forward(self, ctx: ForwardContext) -> None:
        if not self._layers:
            return
        stack = torch.stack([self._layers[i] for i in range(ctx.n_residual)], dim=0).float()
        self._layers.clear()
        # (n_layers, n_pos, d) -> unit vectors -> per position Gram matrix.
        normed = torch.nn.functional.normalize(stack, dim=-1)
        for pos in range(ctx.n_positions):
            vectors = normed[:, pos, :]                     # (n_layers, d)
            matrix = (vectors @ vectors.T).cpu()            # (n_layers, n_layers)
            values = matrix[self._pair_i, self._pair_j].tolist()
            axis = ctx.axis_columns(pos)
            for layer_i, layer_j, cos in zip(self._pair_i, self._pair_j, values):
                row = dict(axis)
                row.update(layer_i=layer_i, layer_j=layer_j, cos=float(cos))
                self._rows.append(row)

    def finalize(self) -> list[dict[str, Any]]:
        rows, self._rows = self._rows, []
        self._layers.clear()
        return rows


# --------------------------------------------------------------------------
# Opt-in signals (--save-attention, --save-hidden)
# --------------------------------------------------------------------------


class ValueNormReducer(Reducer):
    """L2 norm of each attention head's value vector at the current position.

    Cheap enough to keep on every step: one scalar per head, so 32 blocks x 32 heads in fp32 is 4 KB per step.
    That is why `attn_norm` is dense along the step axis while the attention weight tensors next to it are sparse.

    It has to be captured during the forward pass - the value vectors are gone afterwards, and nothing in the saved attention weights lets you recover them.

    GQA note: several query heads share one key/value head.
    We expand the per-kv-head norm back out to query heads so this table joins cleanly with the attention weights on (block, head); the value therefore repeats within a group.

    Example of one produced row:
        {"task_name": "xnli_ko", "doc_id": 7, "choice_idx": 1, "step": 0, "block": 12, "head": 5, "value_norm": 1.84}
    """

    name = "value_norm"
    version = 1
    table = "attn_norm"
    layer_domain = "block"
    comparable_across_vocab = True
    source = "value"

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int | None,
        head_dim: int | None,
        block_shapes: dict[int, tuple[int, int, int]] | None = None,
    ) -> None:
        # A model whose blocks differ in attention shape passes `block_shapes`, block -> (n_heads, n_kv_heads, head_dim).
        # Gemma 4 does: its sliding blocks use 256-wide heads and its full-attention blocks 512-wide ones, so one reshape for every block would split a 1024-column value row into the wrong heads.
        self._n_heads = n_heads
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._block_shapes = dict(block_shapes or {})
        for heads, kv_heads, _ in set(self._block_shapes.values()) or {(n_heads, n_kv_heads, head_dim)}:
            if not kv_heads or heads % kv_heads != 0:
                raise ValueError(
                    f"n_heads ({heads}) is not a multiple of n_kv_heads ({kv_heads}); "
                    "cannot map query heads onto key/value heads"
                )
        self._rows: list[dict[str, Any]] = []

    def config(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "n_heads": self._n_heads,
            "n_kv_heads": self._n_kv_heads,
            "head_dim": self._head_dim,
            "gqa_expansion": "repeat_interleave to query heads",
        }
        if self._block_shapes:
            entry["shapes_by_block"] = {
                str(block): list(shape) for block, shape in sorted(self._block_shapes.items())}
        return entry

    def update(self, index: int, tensor: torch.Tensor, ctx: ForwardContext) -> None:
        """`tensor` is the v_proj output at the scored positions, (n_pos, n_kv*head_dim)."""
        if self._block_shapes:
            if index not in self._block_shapes:
                raise KeyError(f"block {index} has a value projection but no recorded attention shape")
            n_heads, n_kv_heads, head_dim = self._block_shapes[index]
        else:
            n_heads, n_kv_heads, head_dim = self._n_heads, self._n_kv_heads, self._head_dim
        values = tensor.float().view(tensor.shape[0], n_kv_heads, head_dim)
        norms = values.norm(dim=-1)                                        # (n_pos, n_kv_heads)
        norms = norms.repeat_interleave(n_heads // n_kv_heads, dim=-1)     # (n_pos, n_heads)
        norms = norms.cpu().tolist()
        for pos in range(ctx.n_positions):
            axis = ctx.axis_columns(pos)
            for head in range(n_heads):
                row = dict(axis)
                row.update(block=index, head=head, value_norm=float(norms[pos][head]))
                self._rows.append(row)

    def finalize(self) -> list[dict[str, Any]]:
        rows, self._rows = self._rows, []
        return rows


class AttentionWeightReducer(Reducer):
    """Dumps the last position's attention weights to safetensors.

    Storage, not reduction: the weights go out as `(blocks, heads, seq)` per saved step.
    A full (seq x seq) map would be ~537 MB per document at seq=512; keeping only the last query position brings that to ~1 MB.

    The step axis keeps its usual meaning, but only a subset of steps is written:

    * loglikelihood - every step of the scored span (usually one, at most a few)
    * generate      - only `step 0`, the last prefill position, because a 100-step generation would otherwise cost ~100 MB per document, orders of magnitude more than every other signal

    Attention sinks are left in.
    The report labels this axis "attention weight", never "contribution" or "reference".
    """

    name = "attention_weights"
    version = 1
    table = None          # writes tensor files, not parquet rows
    layer_domain = "block"
    comparable_across_vocab = True
    source = "attention"

    def __init__(self) -> None:
        self._per_step: dict[int, dict[int, torch.Tensor]] = {}

    def config(self) -> dict[str, Any]:
        return {
            "position": "last query position",
            "steps_saved": {"loglikelihood": "all scored steps", "generate": "step 0 only"},
            "attn_implementation": "eager",
        }

    @staticmethod
    def keeps_step(ctx: ForwardContext, step: int) -> bool:
        """Whether this step's weights are written out.

        Example:
            >>> ctx = ForwardContext("t", 0, 0, [3], [12], None, 33, 32, task_kind="generate")
            >>> AttentionWeightReducer.keeps_step(ctx, 3)
            False
            >>> AttentionWeightReducer.keeps_step(ctx, 0)
            True
        """
        if ctx.task_kind == "generate":
            return step == 0
        return True

    def update(self, index: int, tensor: torch.Tensor, ctx: ForwardContext) -> None:
        """`tensor` is (n_pos, n_heads, seq): the attention row of each scored position."""
        for pos in range(ctx.n_positions):
            step = ctx.steps[pos]
            if not self.keeps_step(ctx, step):
                continue
            self._per_step.setdefault(step, {})[index] = tensor[pos].detach().to(torch.float16).cpu()

    def pop_tensors(self) -> dict[str, torch.Tensor]:
        """One entry per saved step, shaped (blocks, heads, seq).

        Blocks without attention (hybrid architectures) are simply absent, so the first axis is the *recorded* blocks; `recorder.py` writes the block indices into the safetensors metadata.
        """
        out: dict[str, torch.Tensor] = {}
        for step, by_block in sorted(self._per_step.items()):
            blocks = sorted(by_block)
            out[f"step_{step:04d}"] = torch.stack([by_block[b] for b in blocks], dim=0)
        self._per_step.clear()
        return out

    def finalize(self) -> list[dict[str, Any]]:
        return []


class RawHiddenReducer(Reducer):
    """Dumps raw hidden states for a chosen subset of layers.

    Meant for before/after comparisons around pruning and quantization, where the reduced signals are not enough and you need the vectors themselves.

    This is the only signal that accepts a layer subset, because it is by far the most expensive per layer: at d=4096 in fp16 one layer costs 8 KB per position, and all layers cost ~264 KB per document.

    Args:
        layers: residual indices to keep, already normalised (deduplicated, ascending, range-checked) by `main.py`.
    """

    name = "raw_hidden"
    version = 1
    table = None
    layer_domain = "residual"
    comparable_across_vocab = True
    source = "residual"

    def __init__(self, layers: Sequence[int]) -> None:
        self._layers = list(layers)
        self._wanted = set(self._layers)
        self._collected: dict[int, list[torch.Tensor]] = {}

    def config(self) -> dict[str, Any]:
        return {"layers": self._layers, "dtype": "fp16"}

    def update(self, index: int, tensor: torch.Tensor, ctx: ForwardContext) -> None:
        if index not in self._wanted:
            return
        self._collected.setdefault(index, []).append(tensor.detach().to(torch.float16).cpu())

    def pop_tensors(self) -> dict[str, torch.Tensor]:
        """One entry per kept layer, shaped (total_steps, d)."""
        out = {
            f"layer_{layer:03d}": torch.cat(chunks, dim=0)
            for layer, chunks in sorted(self._collected.items())
        }
        self._collected.clear()
        return out

    def finalize(self) -> list[dict[str, Any]]:
        return []
