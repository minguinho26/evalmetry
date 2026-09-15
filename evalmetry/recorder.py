"""Hook plumbing: catch tensors, hand them to the reducers, write the rows.

The recorder is the only object that touches PyTorch hooks for signal capture.
`debug.py` registers its own, for a different purpose and on a different schedule; the two never share a handle.
It knows which document the model is currently running, slices the batched tensors down to the positions that matter, drives the reducers, and hands their rows to storage.

One capture path, used by both task kinds
-----------------------------------------
The residual stream always comes from the decoder module's own `output_hidden_states` output, i.e. the same L+1 tensors HF returns.
A forward *pre*-hook turns the flag on and a forward hook reads the result, which means loglikelihood and generate see identical tensors.

That uniformity matters.
HF's last hidden state has the final norm already applied, while a forward hook on the last block sees the pre-norm value; mixing the two would make the logit lens apply the final norm twice, but only on the last layer, and only on one of the two task kinds.
The plot would still look reasonable.
So the path is fixed here and written into the manifest.

Note this is *not* `generate(output_hidden_states=True)`, which would keep every layer of every step alive until generation ends.
Hooking the decoder gives us one step at a time, which the reducers consume and drop immediately.

Attention weights and value vectors do come from forward hooks on the modules the adapter points at, and need `attn_implementation="eager"`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch

from . import debug
from .adapters import ModelAdapter
from .reducers import ForwardContext, Reducer
from .storage import RunWriter


@dataclass(frozen=True)
class _GenerationSample:
    """The one document a generation belongs to, in the shape `debug.note_samples` reads.

    A generate task runs one forward per decoded token and there is no batch, so this carries the same three fields a `ForwardContext` does and a fixed `batch_row` of 0.
    Building a whole `ForwardContext` here is not possible: its positions are only known once the step has happened.
    """

    task_name: str
    doc_id: int
    choice_idx: int
    batch_row: int = 0


class Recorder:
    """Owns the hooks, the current-document context and the reducer calls.

    Kept separate from `backend.py` on purpose: the backend holds a copy of lm-eval's `_loglikelihood_tokens`, and mixing our own logic into that file would make the source-hash check and the upstream diff harder to read.

    Args:
        hooks: 선택적 HookSpec 목록. 입력·출력에서 토큰별 분석 지표를 추출한다.
        pass_name: evaluation 또는 collection. 이 pass에 해당하는 hook만 부착한다.
        adapter: resolved module locations for this model.
        reducers: the signal set for this run, in call order.
        writer: where rows and tensors go.
        tokenizer: used to render the `steps` table's token strings.
        write_steps: emit `steps` rows.
            False on the collection pass, where the first pass already recorded them for every document; writing them again would duplicate the rows of the collected documents.
        skip_recorded: skip documents the run directory already holds.
            A run that died leaves complete shards behind, and re-recording those documents on restart would duplicate their rows.
            Scoring still re-runs them - lm-eval needs every request to produce a score - so this skips the recording, not the forward pass.

    Example:
        >>> rec = Recorder(adapter, [lens, sim], writer, tokenizer)   # doctest: +SKIP
        >>> with rec.session():
        ...     rec.expect_loglikelihood([ctx])   # says what the next forward covers
        ...     logits = model(input_ids).logits  # hooks fire, reducers run
        ...     rec.flush()                       # rows are written
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        reducers: Sequence[Reducer],
        writer: RunWriter,
        tokenizer: Any,
        write_steps: bool = True,
        skip_recorded: bool = True,
        hooks: Sequence[Any] = (),
        pass_name: str = "evaluation",
    ) -> None:
        self.adapter = adapter
        self.reducers = list(reducers)
        self.writer = writer
        self.tokenizer = tokenizer
        self.write_steps = write_steps
        self.already_recorded = set(writer.already_recorded) if skip_recorded else set()
        # A collection pass re-runs recorded documents on purpose; a repeated collection
        # replaces tensor files but must not append table rows (attn_norm) a second time.
        self._rows_on_disk: dict[str, set[tuple[str, int, int]]] = {}
        if not skip_recorded:
            from .storage import existing_doc_keys
            self._rows_on_disk = {r.table: existing_doc_keys(writer.run_dir, r.table)
                                  for r in reducers if r.table}
        self.skipped = 0

        self._handles: list[Any] = []
        # What the forward pass now in flight is for.
        # None means "not tracing", which is how anything outside a declared request - a batch-size probe, or a document already recorded by an earlier run - produces no rows.
        self._plan: dict[str, Any] | None = None
        # Every context we built since the last flush, i.e. one document's worth for generate and one batch's worth for loglikelihood.
        self._recorded: list[ForwardContext] = []
        # Raw tensors caught during the current forward, cleared as soon as the reducers have consumed them.
        self._attention: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}
        from .hooks import HookRuntime, load_hooks
        self.custom = HookRuntime(self, load_hooks(None, {}, hooks), pass_name)

    # -- what the reducers ask for -------------------------------------------------

    @property
    def wants_attention(self) -> bool:
        """True when some reducer consumes attention weights."""
        return any(r.source == "attention" for r in self.reducers)

    @property
    def wants_values(self) -> bool:
        """True when some reducer consumes value vectors."""
        return any(r.source == "value" for r in self.reducers)

    # -- hook lifetime -------------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator["Recorder"]:
        """Register hooks for a block of evaluation, then always remove them.

        A hook left behind keeps firing during unrelated work and keeps references to tensors that should have been freed, so the pairing is enforced by the context manager rather than by discipline.
        """
        try:
            self.register_hooks()
            yield self
        except BaseException:
            self.custom.drain()
            self._recorded.clear()
            raise
        finally:
            self.remove_hooks()

    def register_hooks(self) -> None:
        """등록 실패 시 부분적으로 부착된 handle도 회수한다."""
        try:
            self._register_hooks()
        except BaseException:
            self.remove_hooks()
            raise

    def _register_hooks(self) -> None:
        if self._handles:
            return
        decoder = self.adapter.decoder
        self._handles.append(
            decoder.register_forward_pre_hook(self._request_hidden_states, with_kwargs=True)
        )
        self._handles.append(decoder.register_forward_hook(self._on_decoder_output))

        if self.wants_attention:
            for block_idx, module in enumerate(self.adapter.attn_modules):
                if module is not None:
                    self._handles.append(
                        module.register_forward_hook(self._make_attention_hook(block_idx))
                    )
        if self.wants_values:
            for block_idx, module in enumerate(self.adapter.v_projs):
                if module is not None:
                    self._handles.append(
                        module.register_forward_hook(self._make_value_hook(block_idx))
                    )

        self.custom.register()
        root = self.adapter.root_model
        if root is not None:
            self._handles.append(root.register_forward_pre_hook(self.custom.note_root, with_kwargs=True))
            self._handles.append(root.register_forward_hook(lambda *args: self.custom.end()))

    def remove_hooks(self) -> None:
        self.custom.close()
        self._attention.clear()
        self._values.clear()
        self._plan = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- hooks ---------------------------------------------------------------------

    def _request_hidden_states(self, module, args, kwargs):  # noqa: ANN001 - torch signature
        """Ask the decoder for its per-layer hidden states, one forward at a time."""
        self.custom.begin(args, kwargs)
        if self._plan is None:
            return None
        if any(r.source == "residual" for r in self.reducers):
            kwargs["output_hidden_states"] = True
        return args, kwargs

    def _on_decoder_output(self, module, args, output):  # noqa: ANN001 - torch signature
        """The forward pass is complete: build the context and run the reducers.

        Inner modules finish before their parent, so the attention and value hooks have already filled their buffers by the time this runs.
        """
        if self._plan is None:
            return
        hidden_states = getattr(output, "hidden_states", None)
        needs_residual = any(r.source == "residual" for r in self.reducers)
        if needs_residual:
            if hidden_states is None:
                raise RuntimeError(
                    "the decoder returned no hidden_states. The forward pre-hook that sets "
                    "output_hidden_states=True did not take effect; check the adapter's "
                    "decoder path for this architecture."
                )
            if len(hidden_states) != self.adapter.n_residual:
                raise ValueError(
                    f"expected {self.adapter.n_residual} hidden states (L+1) but got "
                    f"{len(hidden_states)}; the adapter's block list does not match the model"
                )
            sequence_length = hidden_states[0].shape[1]
        else:
            # Attention/value/custom hooks need token positions, not all residuals.
            last_hidden = getattr(output, "last_hidden_state", None)
            if last_hidden is None:
                raise RuntimeError("signal collection requires decoder.last_hidden_state for token positions")
            sequence_length = last_hidden.shape[1]
            hidden_states = ()
        # The logit lens decodes through the model's *own* final norm and head, so with a
        # module tracer attached those calls are module entries nested inside this hook.
        # The phase is what keeps them from being read as the model's own work.
        # Snapshot before reducers advance generation and invoke auxiliary head/norm calls.
        self.custom.contexts(sequence_length)
        self.custom.suspended = True
        try:
            with debug.phase("signal_collection"):
                for ctx in self._contexts_for_forward(sequence_length):
                    self._run_reducers(ctx, hidden_states)
                    self._recorded.append(ctx)
        finally:
            self.custom.suspended = False
        if self.adapter.root_model is None:
            self.custom.end()
        self._attention.clear()
        self._values.clear()

    def _make_attention_hook(self, block_idx: int):
        """Grab the (batch, heads, q, k) attention probabilities of one block."""

        def hook(module, args, output):  # noqa: ANN001 - torch signature
            if self._plan is None:
                return
            weights = _find_attention_weights(output)
            if weights is None:
                raise RuntimeError(
                    f"block {block_idx}: the attention module returned no weight matrix. "
                    "Attention capture needs attn_implementation='eager'; FlashAttention "
                    "and SDPA never materialise the probabilities."
                )
            self._attention[block_idx] = weights.detach()

        return hook

    def _make_value_hook(self, block_idx: int):
        """Grab the value projection output of one block."""

        def hook(module, args, output):  # noqa: ANN001 - torch signature
            if self._plan is None:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            # How to read values out of this projection is architecture specific, so the adapter owns it.
            self._values[block_idx] = self.adapter.extract_value(tensor.detach(), block_idx)

        return hook

    # -- declaring what a forward pass is for --------------------------------------

    def expect_loglikelihood(self, contexts: Sequence[ForwardContext]) -> None:
        """Declare the documents of the next loglikelihood forward, one per batch row.

        This is the step that keeps signals attached to the right document. lm-eval drops the `Instance` (and with it `doc_id`) before the model sees a request, and then reorders requests by length; `backend.py` re-establishes the link and states it here, right before the call.

        Documents already on disk from an earlier, interrupted run are dropped here.
        The forward pass still happens - lm-eval is scoring them - but nothing is recorded, so the shards gain no duplicates.
        """
        # The tracer is told about every context, not just the ones that will be recorded:
        # a document already on disk is still scored, and a forward that is not recorded is
        # exactly as able to run out of memory as one that is.
        debug.note_samples(contexts)
        wanted = [ctx for ctx in contexts if ctx.key not in self.already_recorded]
        self.skipped += len(contexts) - len(wanted)
        self._plan = {"kind": "loglikelihood", "contexts": wanted} if wanted else None

    def expect_generation(
        self, task_name: str, doc_id: int, prompt_length: int, choice_idx: int = 0
    ) -> None:
        """Declare that the next `model.generate()` belongs to one document.

        Generation runs one forward per token, so the contexts are built as the steps happen rather than up front: `step 0` is the last prefill position, and every decode step after that adds one.

        Args:
            prompt_length: number of prompt tokens, needed to turn a decoding step into an absolute sequence position.
        """
        debug.note_samples(
            [_GenerationSample(task_name, doc_id, choice_idx)]
        )
        if (task_name, doc_id, choice_idx) in self.already_recorded:
            self.skipped += 1
            self._plan = None
            return
        self._plan = {
            "kind": "generate",
            "task_name": task_name,
            "doc_id": doc_id,
            "choice_idx": choice_idx,
            "prompt_length": prompt_length,
            "next_step": 0,
        }

    def set_prompt_length(self, prompt_length: int) -> None:
        """Tell an in-flight generation how long its prompt is.

        Only known once lm-eval has tokenised and padded the prompt, which happens after `expect_generation` and before the first forward pass.
        Without it a decoding step cannot be turned into an absolute position.
        """
        if self._plan is not None and self._plan.get("kind") == "generate":
            self._plan["prompt_length"] = int(prompt_length)

    def _contexts_for_forward(self, seq_len: int) -> list[ForwardContext]:
        """Build the contexts describing the forward pass that just finished."""
        plan = self._plan
        assert plan is not None
        if plan["kind"] == "loglikelihood":
            return list(plan["contexts"])

        # generate: prefill carries the whole prompt and we score its last position; every later forward is a single decoded token.
        step = plan["next_step"]
        plan["next_step"] = step + 1
        seq_index = seq_len - 1
        position = plan["prompt_length"] - 1 + step
        return [
            ForwardContext(
                task_name=plan["task_name"],
                doc_id=plan["doc_id"],
                choice_idx=plan["choice_idx"],
                steps=[step],
                positions=[position],
                target_token_ids=None,   # a generate task has no gold token to rank
                n_residual=self.adapter.n_residual,
                n_blocks=self.adapter.n_blocks,
                task_kind="generate",
                batch_row=0,
                seq_indices=[seq_index],
                input_offset=(max(0, plan["original_prompt_length"] - plan["prompt_length"])
                              if "original_prompt_length" in plan else None),
            )
        ]

    def _run_reducers(self, ctx: ForwardContext, hidden_states: Sequence[torch.Tensor]) -> None:
        """Feed one document's slice of this forward pass to every reducer."""
        for reducer in self.reducers:
            if reducer.source == "residual":
                for layer, tensor in enumerate(hidden_states):
                    reducer.update(layer, _slice_positions(tensor, ctx), ctx)
            elif reducer.source == "attention":
                for block, tensor in sorted(self._attention.items()):
                    reducer.update(block, _slice_attention(tensor, ctx), ctx)
            elif reducer.source == "value":
                for block, tensor in sorted(self._values.items()):
                    reducer.update(block, _slice_positions(tensor, ctx), ctx)
            reducer.end_forward(ctx)
        ctx.shared.clear()

    # -- finishing documents -------------------------------------------------------

    def set_generated_tokens(self, token_ids: Sequence[int]) -> None:
        """Attach the tokens a generate call actually emitted to their steps.

        Only known once generation has finished, and needed because sampling can make the emitted token differ from the last layer's top-1.

        Example:
            >>> rec.set_generated_tokens([1820, 4320, 128009])       # doctest: +SKIP
        """
        for ctx in self._recorded:
            if ctx.task_kind != "generate":
                continue
            step = ctx.steps[0]
            ctx.step_token_ids = [int(token_ids[step])] if step < len(token_ids) else [None]

    def flush(self) -> int:
        """Finalize the reducers and write everything recorded since the last flush.

        Returns:
            How many documents were written.
        """
        contexts = self._recorded
        self._recorded = []
        self._plan = None
        if not contexts:
            return 0

        documents = _group_by_document(contexts)
        rows_by_table: dict[str, list[dict[str, Any]]] = self.custom.drain()
        if self.write_steps:
            rows_by_table["steps"] = []
            for group in documents:
                rows_by_table["steps"].extend(self.steps_rows(group))

        for reducer in self.reducers:
            rows = reducer.finalize()
            if rows and reducer.table:
                on_disk = self._rows_on_disk.get(reducer.table)
                if on_disk:
                    rows = [row for row in rows
                            if (row["task_name"], int(row["doc_id"]), int(row["choice_idx"])) not in on_disk]
                rows_by_table.setdefault(reducer.table, []).extend(rows)
            self._write_tensors(reducer, documents)

        self.writer.write_document(rows_by_table, documents=len(documents))
        return len(documents)

    def steps_rows(self, contexts: Sequence[ForwardContext]) -> list[dict[str, Any]]:
        """Build the `steps` rows for one document, across all of its forwards.

        The `steps` table cannot be replaced by the layer-L rows of `signals`: for loglikelihood the scored token is the gold continuation token rather than the model's prediction, and with sampling on, a generated token can differ from the last layer's top-1.

        Example:
            >>> rec.steps_rows([ctx])                              # doctest: +SKIP
            [{'task_name': 'xnli_ko', 'doc_id': 7, 'choice_idx': 1, 'step': 0,
              'token_id': 9891, 'token': 'Ġyes', 'position': 41}]
        """
        rows = []
        for ctx in contexts:
            token_ids = ctx.step_token_ids or []
            for index, step in enumerate(ctx.steps):
                token_id = token_ids[index] if index < len(token_ids) else None
                token_id = None if token_id is None else int(token_id)
                rows.append(
                    {
                        "task_name": ctx.task_name,
                        "doc_id": ctx.doc_id,
                        "choice_idx": ctx.choice_idx,
                        "step": step,
                        "token_id": token_id,
                        "token": self._decode_token(token_id),
                        "position": ctx.positions[index],
                    }
                )
        return rows

    def _decode_token(self, token_id: int | None) -> str | None:
        if token_id is None:
            return None
        try:
            return str(self.tokenizer.convert_ids_to_tokens(token_id))
        except Exception:
            return str(self.tokenizer.decode([token_id]))

    def _write_tensors(
        self, reducer: Reducer, documents: Sequence[Sequence[ForwardContext]]
    ) -> None:
        """Persist a reducer's opt-in tensor output, if it produced any.

        Tensor dumps are one file per document, so they are only defined for an unbatched forward - which is exactly how the collection path runs.
        """
        tensors = reducer.pop_tensors()
        if not tensors:
            return
        if len(documents) != 1:
            raise RuntimeError(
                f"{reducer.name} writes one tensor file per document, so it requires "
                f"batch size 1, but this flush covered {len(documents)} documents"
            )
        ctx = documents[0][0]
        store = self.writer.attention if reducer.source == "attention" else self.writer.raw_hidden
        meta = {
            "reducer": reducer.name,
            "version": str(reducer.version),
            "task_kind": ctx.task_kind,
        }
        store.save(ctx.task_name, ctx.doc_id, ctx.choice_idx, tensors, meta=meta)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _group_by_document(
    contexts: Sequence[ForwardContext],
) -> list[list[ForwardContext]]:
    """Group forward contexts by (task_name, doc_id, choice_idx), keeping order.

    A loglikelihood batch gives one context per document; a generation gives many contexts - one per decoding step - that all belong to one document.

    Example:
        >>> a = ForwardContext("t", 1, 0, [0], [3], None, 2, 1)
        >>> b = ForwardContext("t", 1, 0, [1], [4], None, 2, 1)
        >>> c = ForwardContext("t", 2, 0, [0], [3], None, 2, 1)
        >>> [len(g) for g in _group_by_document([a, b, c])]
        [2, 1]
    """
    grouped: dict[tuple[str, int, int], list[ForwardContext]] = {}
    for ctx in contexts:
        grouped.setdefault(ctx.key, []).append(ctx)
    return list(grouped.values())


def _slice_positions(tensor: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
    """Cut a (batch, seq, feature) tensor down to one document's scored positions.

    Right padding lives at the end of the sequence axis, so reading the tensor's last index instead of the document's own would silently pick up a pad token.
    Selecting explicit indices avoids that entirely.

    Example:
        >>> x = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        >>> ctx = ForwardContext("t", 0, 0, [0], [2], None, 2, 1, batch_row=1)
        >>> _slice_positions(x, ctx).tolist()
        [[18, 19, 20]]
    """
    index = torch.as_tensor(ctx.seq_indices, device=tensor.device, dtype=torch.long)
    return tensor[ctx.batch_row].index_select(0, index)


def _slice_attention(tensor: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
    """Cut a (batch, heads, q, k) attention tensor down to (n_pos, heads, k).

    Only the query rows we score are kept; the full q x k map is what makes attention unaffordable to store.

    Example:
        >>> attn = torch.zeros(1, 4, 6, 6)
        >>> ctx = ForwardContext("t", 0, 0, [0], [5], None, 2, 1)
        >>> _slice_attention(attn, ctx).shape
        torch.Size([1, 4, 6])
    """
    index = torch.as_tensor(ctx.seq_indices, device=tensor.device, dtype=torch.long)
    rows = tensor[ctx.batch_row].index_select(1, index)   # (heads, n_pos, k)
    return rows.transpose(0, 1)                           # (n_pos, heads, k)


def _find_attention_weights(output: Any) -> torch.Tensor | None:
    """Pick the attention probability matrix out of whatever the module returned.

    HF attention modules return `(attn_output, attn_weights)` and sometimes a third element, with `attn_weights` set to None unless the implementation materialises it.
    The probabilities are the only 4-dimensional element, so we look for that rather than for a tuple position that shifts between transformers versions.

    Example:
        >>> _find_attention_weights((torch.zeros(1, 3, 8), torch.zeros(1, 2, 3, 3))).shape
        torch.Size([1, 2, 3, 3])
    """
    candidates = output if isinstance(output, (tuple, list)) else [output]
    for item in candidates:
        if isinstance(item, torch.Tensor) and item.dim() == 4:
            return item
    return None
