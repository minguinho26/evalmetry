"""Everything that runs the model and makes forward passes happen.

Two entry points live here because they do the same job - drive the model so the hooks fire:

* `TracedHFLM`, an lm-eval backend that records signals while lm-eval scores.
* `ResearchDataCollector`, which re-feeds prompts from a finished run without going through lm-eval at all (the second pass of --save-attention / --save-hidden).

We do not fork lm-eval.
The backend is registered with `@register_model` and subclasses `HFLM`, so scoring stays exactly lm-eval's.

Attaching a signal to the right document
----------------------------------------
This is the part that fails silently if it is wrong: the numbers land on the wrong document, the plots look healthy, and nothing complains.
Two things go wrong on the way down from the evaluator to the hooks.

1.
   `doc_id` never reaches the model.
   `HFLM.loglikelihood` takes the arguments out of each `Instance` and passes on plain `((context, continuation), context_enc, continuation_enc)` tuples; the `Instance` - and with it `doc_id` and `task_name` - is not part of them.
   Upstream relies on list order to put results back.
2.
   That order changes again inside `_loglikelihood_tokens`, which sorts requests by length for batching efficiency and un-sorts only at the end.
   The order the hooks see is not document order.

So both methods are overridden.
`loglikelihood` is a thin layer over the upstream logic that additionally records request index -> (task_name, doc_id, choice_idx).
`_loglikelihood_tokens` is derived from upstream and keeps the sort permutation explicit, so the mapping can be looked up right before each forward pass.
Because that one is a copy, `check_upstream_source()` guards it: CI fails when lm-eval changes the original.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import os
import random
import sys
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import Collator

try:  # lm-eval >= 0.4.10 moved this helper into its own module
    from lm_eval.models.utils_hf import pad_and_concat
except ImportError:  # lm-eval 0.4.9.x
    from lm_eval.models.utils import pad_and_concat

from . import debug
from .reducers import ForwardContext
from .recorder import Recorder
from .storage import BACKEND_NAME, SAMPLING_SEED, read_samples


# --------------------------------------------------------------------------
# Upstream source check
# --------------------------------------------------------------------------

# SHA-256 hashes of reviewed upstream sources, keyed by lm-eval version.
# `_loglikelihood_tokens` is copied below; `loglikelihood` defines the request
# tuples we rebuild. Changes to either require reviewing the document mapping.
# Hash the whole Collator class because we also rely on its private index state
# (`_arr_with_indices`, `_reorder_indices`) to recover each batch row's document.
# An unlisted version produces a warning; a known version with changed source
# fails the strict test check. Package installation pins the default version.
UPSTREAM_SOURCE_HASHES: dict[str, dict[str, str]] = {
    "0.4.9.1": {
        "lm_eval.models.huggingface:HFLM._loglikelihood_tokens":
            "816339a43a1c145cd0194c29dd1c3d386f658c9ddeb38a75b8736e16c862eeac",
        "lm_eval.api.model:TemplateLM.loglikelihood":
            "7b6422e672f176bcb2a25d062a2c662bec5046b04d3de6b76cae94a79ed5eba0",
        "lm_eval.models.utils:Collator":
            "33da8a838a542df718a1835754cb03d10c7e139501ff874631a0aa2d27ef9241",
    },
    # 0.4.13 diff against 0.4.9.1, reviewed: `_loglikelihood_tokens` gains two assertions, `strict=True` on a zip, and a reset of cached auto batch sizes (mirrored below); `loglikelihood` gains a docstring and a progress bar over tokenisation; `Collator` is retyped to builtin generics, its logic untouched.
    # `pad_and_concat` moved to `lm_eval.models.utils_hf`.
    # None of it changes the scoring arithmetic.
    "0.4.13": {
        "lm_eval.models.huggingface:HFLM._loglikelihood_tokens":
            "994529f70fd23192506699a9e98fb83e6a63182494cc3e5997520582a28dc4f5",
        "lm_eval.api.model:TemplateLM.loglikelihood":
            "507df4a0fa1d0bae59b457b7624644fefcdf91c2f5fa3d234c97106b22580b5c",
        "lm_eval.models.utils:Collator":
            "2dc993c9a488146bd295a71e97f5bedf16a8a7d3405492a80a27d5742e53833a",
    },
}


def method_source(cls: type, method_name: str) -> str:
    """Return the source text of one method, located via the AST.

    Deliberately not `inspect.getsource`: that walks backwards from the code object and can pick up neighbouring comments, so the hash would move for reasons that have nothing to do with the code.

    Example:
        >>> method_source(HFLM, "_loglikelihood_tokens").splitlines()[0].strip()
        'def _loglikelihood_tokens('
    """
    path = inspect.getsourcefile(cls)
    if path is None:
        raise RuntimeError(f"cannot locate source file for {cls.__name__}")
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    tree = ast.parse("".join(lines))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return "".join(lines[item.lineno - 1 : item.end_lineno])
    raise LookupError(f"{cls.__name__}.{method_name} not found in {path}")


def class_source(cls: type) -> str:
    """Return the whole source text of a class, located via the AST.

    Used where we depend on more than one method of a class - reading private state, say - so that any change to it is caught rather than only a change to the one method we happened to name.
    """
    path = inspect.getsourcefile(cls)
    if path is None:
        raise RuntimeError(f"cannot locate source file for {cls.__name__}")
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    for node in ast.walk(ast.parse("".join(lines))):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return "".join(lines[node.lineno - 1 : node.end_lineno])
    raise LookupError(f"{cls.__name__} not found in {path}")


def upstream_source_hashes() -> dict[str, str]:
    """Hash the upstream methods as they exist in the installed lm-eval.

    Example:
        >>> upstream_source_hashes()["lm_eval.api.model:TemplateLM.loglikelihood"]  # doctest: +SKIP
        '2fbd0c...'
    """
    methods = {
        "lm_eval.models.huggingface:HFLM._loglikelihood_tokens": (HFLM, "_loglikelihood_tokens"),
        "lm_eval.api.model:TemplateLM.loglikelihood": (TemplateLM, "loglikelihood"),
    }
    found = {
        key: hashlib.sha256(method_source(cls, name).encode("utf-8")).hexdigest()
        for key, (cls, name) in methods.items()
    }
    found["lm_eval.models.utils:Collator"] = hashlib.sha256(
        class_source(Collator).encode("utf-8")
    ).hexdigest()
    return found


def _upstream_resets_batch_sizes() -> bool:
    """Whether the installed lm-eval clears cached auto batch sizes per request set.

    Read off the installed source rather than gated on a version string, so the mirror follows whatever is actually installed.
    """
    global _RESETS_BATCH_SIZES
    if _RESETS_BATCH_SIZES is None:
        _RESETS_BATCH_SIZES = (
            "self.batch_sizes = {}" in method_source(HFLM, "_loglikelihood_tokens")
        )
    return _RESETS_BATCH_SIZES


_RESETS_BATCH_SIZES: bool | None = None


def check_upstream_source(strict: bool = True) -> dict[str, str]:
    """Compare the installed lm-eval against the versions this file was written for.

    An editable checkout can have the same version string and different source, so the hash is the only reliable signal.

    Args:
        strict: raise on mismatch.
            Tests use True so CI fails first; a run uses False and only warns, because a changed upstream is not necessarily an incompatible one.

    Returns:
        The observed hashes, which also go into the manifest.

    Raises:
        RuntimeError: in strict mode, when a reviewed version's source has moved under it.
            An lm-eval version nobody has reviewed yet only warns: it is unverified, which is not the same as known-broken.
    """
    import lm_eval

    observed = upstream_source_hashes()
    expected = UPSTREAM_SOURCE_HASHES.get(lm_eval.__version__)
    if expected is None:
        message = (
            f"lm-eval {lm_eval.__version__} has not been reviewed against this backend. "
            f"Reviewed versions: {sorted(UPSTREAM_SOURCE_HASHES)}. It may work - diff "
            "`_loglikelihood_tokens` and `Collator` against backend.py, then add the "
            "hashes to UPSTREAM_SOURCE_HASHES."
        )
        print(f"warning: {message}", file=sys.stderr)
        return observed

    drifted = [key for key, value in expected.items() if observed.get(key) != value]
    if drifted and strict:
        raise RuntimeError(
            f"lm-eval {lm_eval.__version__} source changed for: "
            + ", ".join(drifted)
            + ". `_loglikelihood_tokens` is mirrored in backend.py; diff the upstream "
            "method against it, port any change, then update UPSTREAM_SOURCE_HASHES."
        )
    return observed


# --------------------------------------------------------------------------
# Traced backend
# --------------------------------------------------------------------------


@register_model(BACKEND_NAME)
class TracedHFLM(HFLM):
    """`HFLM` that reports which document each forward pass belongs to.

    Scoring is untouched - every number lm-eval reports comes from the upstream code path.
    The only additions are the document mapping and the hook session around the request loops.

    Example:
        >>> lm = TracedHFLM(pretrained="Qwen/Qwen3-1.7B", batch_size=1)  # doctest: +SKIP
        >>> lm.attach_recorder(recorder)
        >>> lm_eval.simple_evaluate(model=lm, tasks=["xnli_ko"], log_samples=True)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_transform = None
        self._recorder: Recorder | None = None
        # Request index -> (task_name, doc_id, choice_idx), rebuilt on every `loglikelihood` call.
        self._request_docs: list[tuple[str, int, int]] = []
        self._last_generated_tokens: list[int] = []
        # lm-eval's own defaults are left alone, `logits_cache` included.
        # It is what makes a score reproduce a published lm-eval number: the cache changes which requests share a forward pass, and a different batch shape changes the order a matmul accumulates in.
        # In fp16 that is enough to move a borderline document - stock lm-eval scores 0.66 / 0.66 / 0.64 at batch 1 / 4 / 8 on the same 50 MMLU documents.
        #
        # The cache does mean fewer forward passes than requests, which sounds like it would leave holes in the signals.
        # It does not: a cache group is a set of requests sharing `context + continuation[:-1]`, so they share the *input* and therefore the hidden states.
        # One forward pass supplies every member; only the gold token, and so `target_rank`, differs.
        # `_build_contexts` fans the pass out across the group.

    def prepare_context(self, context: str) -> str:
        """Apply a deterministic custom formatter once, before tokenization."""
        if self.prompt_transform is None:
            return context
        transformed = self.prompt_transform(context)
        if not isinstance(transformed, str):
            raise TypeError("prompt_transform must return str")
        return transformed

    def _prepare_requests(self, requests):
        # Keep evaluator-owned instances unchanged, including their saved prompt
        # hashes. Replay reads those originals and applies the same formatter.
        if self.prompt_transform is None:
            return requests
        import copy
        prepared = []
        for request in requests:
            item = copy.copy(request)
            item.arguments = (self.prepare_context(request.args[0]), *request.args[1:])
            prepared.append(item)
        return prepared

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        if self.prompt_transform is not None:
            raise ValueError("prompt_transform does not support loglikelihood_rolling; use a likelihood or generation task")
        if self._recorder is not None:
            raise ValueError("internal signals do not support loglikelihood_rolling; use signals=()")
        return super().loglikelihood_rolling(requests, disable_tqdm=disable_tqdm)

    def attach_recorder(self, recorder: Recorder) -> None:
        """Start recording.
Without this the backend behaves exactly like `HFLM`.

        Warns on `batch_size="auto"`, which is not safe here. lm-eval sizes the batch by running a forward pass and seeing what fits, but that probe registers no scored positions, so the reducers do nothing and the measurement is of a model that is not the one about to run. Recording then adds memory the chosen batch has no room for.

        Seen on OLMo-2-7B over boolq, whose contexts are long and whose vocabulary is 100,352: the stock backend settled on 45 and the traced one - the heavier of the two - probed *higher*, at 51, and died on the first real batch trying to allocate 2.76 GiB for the logits. Short contexts hide it, which is why arc_easy never showed it.
        """
        self._recorder = recorder
        if self.batch_size == "auto":
            print(
                "warning: batch_size=auto sizes the batch with a probe that does not "
                "record, so it can overshoot what recording leaves room for and "
                "fail with CUDA out of memory partway through. Pass an explicit "
                "batch size for a run you need to finish.",
                file=sys.stderr,
            )

    # -- loglikelihood -------------------------------------------------------------

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        """Same request tuples as upstream, plus the document mapping.

        Mirrors `TemplateLM.loglikelihood`; the only addition is `self._request_docs`, which records what each tuple came from before the `Instance` is dropped.

        `choice_idx` also comes from here: a multiple-choice document produces one consecutive `Instance` per choice, and the position within that run of identical `doc_id`s is the choice index.
        """
        requests = self._prepare_requests(requests)
        if self._recorder is None:
            return super().loglikelihood(requests, disable_tqdm=disable_tqdm)

        new_reqs = []
        self._request_docs = []
        seen_choices: dict[tuple[str, int], int] = {}
        for instance in requests:
            context, continuation = instance.args
            if context == "":
                # BOS or EOS as context, matching upstream.
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))

            key = (instance.task_name, instance.doc_id)
            choice_idx = seen_choices.get(key, 0)
            seen_choices[key] = choice_idx + 1
            self._request_docs.append((instance.task_name, instance.doc_id, choice_idx))

        with self._recorder.session():
            return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        """Upstream's batching loop with the document mapping threaded through.

        Derived from `HFLM._loglikelihood_tokens` (lm-eval 0.4.9.1) and pinned by `check_upstream_source()`.
        The scoring arithmetic is upstream's line for line, and the batching is upstream's `Collator` with upstream's parameters - not a re-derivation of it.
        That is deliberate: which requests share a forward pass determines the batch shape, the batch shape determines matmul accumulation order, and in fp16 that is enough to move a score.
        Reproducing a published lm-eval number means running the same batches, not merely the same formula.

        What is added:

        * before each forward, the documents that pass covers are declared to the recorder (`_build_contexts`);
        * after each forward, the reduced rows are flushed.

        Causal decoder models only: a seq2seq model scores on the decoder side, where the residual axis means something else.

        Returns:
            `(logprob, is_greedy)` per request in the original order, as upstream.
        """
        if self._recorder is None:
            return super()._loglikelihood_tokens(requests, disable_tqdm, override_bs)
        if self.backend != "causal":
            raise NotImplementedError(
                f"{BACKEND_NAME} traces causal decoder models only, got backend={self.backend!r}"
            )

        res = []

        def _collate(req: tuple[tuple[str, str], list[int], list[int]]):
            """Upstream's sort key: longest first, so an OOM happens on batch one."""
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: tuple[tuple[str, str], list[int], list[int]]):
            """Upstream's grouping key: context + continuation minus its last token."""
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.backend == "causal" and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
            else None
        )

        if batch_fn is not None and _upstream_resets_batch_sizes():
            # Mirrors lm-eval 0.4.10+: clear cached auto batch sizes so detection runs against this request set.
            # Done only when upstream does it, so the batching stays identical on either version - and identical batching is what keeps the scores identical.
            self.batch_sizes = {}

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        # How many items the collator has handed out so far.
        # Without grouping it records the original index of each one as it goes, so this counter is what turns a chunk row back into a document.
        emitted = 0

        for chunk in chunks:
            inps, cont_toks_list, inplens = [], [], []
            padding_len_inp = None

            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # CTX      CONT inp    0 1 2 3|4 5 6 7 8 9   <- last token dropped by [:-1] logits   1 2 3|4 5 6 7 8 9 scored         4 5 6 7 8 9 Too long for the window: truncate on the left, like upstream.
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape
                padding_len_inp = (
                    max(padding_len_inp, inplen) if padding_len_inp is not None else inplen
                )
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")

            self._recorder.expect_loglikelihood(
                self._build_contexts(chunk, inplens, cont_toks_list, batched_inps.shape[1],
                                     padding_len_inp, re_ord, emitted)
            )
            multi_logits = F.log_softmax(
                self._model_call(batched_inps), dim=-1, dtype=self.softmax_dtype
            )  # [batch, padding_len_inp, vocab]
            emitted += len(chunk)

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                contlen = len(cont_toks)
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]
                greedy_tokens = logits.argmax(dim=-1)

                # Fans a cache group back out into its members, or is a no-op when grouping is off.
                for request_str, cont_toks, logits in re_ord.get_cache(  # noqa: B020
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks).all()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)

                    answer = (float(logits.sum()), bool(max_equal))
                    res.append(answer)
                    if request_str is not None:
                        self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

            self._recorder.flush()

        pbar.close()
        return re_ord.get_original(res)

    def _build_contexts(
        self,
        chunk: Sequence[tuple[tuple[str, str], list[int], list[int]]],
        inplens: Sequence[int],
        cont_toks_list: Sequence[list[int]],
        logits_len: int,
        padding_len_inp: int,
        re_ord: Collator,
        emitted: int,
    ) -> list[ForwardContext]:
        """Describe, per batch row, which documents and positions this pass covers.

        lm-eval scores the continuation tokens, and the logits that produce them sit at input positions `[inplen - contlen, inplen)`: the last context token predicts the first continuation token, and it runs on from there.
        Those are the positions the signals are taken at - we do not offer a "which position?" option, we follow whatever lm-eval scores.

        When `logits_cache` is on, one row stands for a whole group of requests that share `context + continuation[:-1]`.
        They share the input, so they share the hidden states; each member gets its own context, differing only in `choice_idx` and in the gold tokens that `target_rank` is measured against.
        A member whose continuation is shorter than the group representative's scores the trailing part of the same span, matching upstream's `greedy_tokens[:, -len(cont):]`.

        Example: an MMLU document's four choices are " A".." D" after an identical context, so one forward pass covers all four, and this returns four contexts pointing at the same batch row and position.
        """
        contexts = []
        for row, (_, context_enc, continuation_enc) in enumerate(chunk):
            # Same correction as upstream, for models that insert virtual tokens (prompt/prefix tuning) ahead of the real input.
            ctx_len = inplens[row] + (logits_len - padding_len_inp)
            for request_index, member_cont in self._group_members(
                re_ord, context_enc, continuation_enc, emitted + row
            ):
                task_name, doc_id, choice_idx = self._request_docs[request_index]
                contlen = len(member_cont)
                contexts.append(
                    ForwardContext(
                        task_name=task_name,
                        doc_id=doc_id,
                        choice_idx=choice_idx,
                        steps=list(range(contlen)),
                        positions=list(range(ctx_len - contlen, ctx_len)),
                        target_token_ids=list(member_cont),
                        n_residual=self._recorder.adapter.n_residual,
                        n_blocks=self._recorder.adapter.n_blocks,
                        task_kind="loglikelihood",
                        batch_row=row,
                        input_length=inplens[row],
                        input_offset=max(0, len(context_enc) + len(continuation_enc) - 1 - inplens[row]),
                    )
                )
        return contexts

    def _group_members(
        self,
        re_ord: Collator,
        context_enc: list[int],
        continuation_enc: list[int],
        position: int,
    ) -> list[tuple[int, list[int]]]:
        """Every original request this batch row will produce an answer for.

        Two bookkeeping paths, because the collator tracks indices differently:

        * grouping on (`logits_cache` on) - the group is still sitting in the collator under its context key, and each entry carries its original index.
          `get_cache` pops it later, after this has read it.
        * grouping off - the collator records the original index of each item as it hands it out, so the item handed out `position`-th is `_reorder_indices[position]`.

        Both read the collator's own bookkeeping rather than recomputing it; the source-hash check covers `Collator` for exactly that reason.
        """
        if re_ord._group_by == "contexts":
            key = tuple(context_enc + continuation_enc[:-1])
            group = re_ord._arr_with_indices[key]
            return [(index, item[-1]) for index, item in group]
        return [(re_ord._reorder_indices[position], continuation_enc)]

    # -- phase labels --------------------------------------------------------------

    def _detect_batch_size(self, *args: Any, **kwargs: Any):
        """Keep auto-batch probes out of sample-level research statistics."""
        with debug.without_samples():
            return super()._detect_batch_size(*args, **kwargs)

    def _model_call(self, *args: Any, **kwargs: Any):
        """Upstream's forward, labelled as the model's own work.

        The label is what lets a module tracer tell three things apart that all run the same modules: the model computing what lm-eval scores, our reducers re-entering `final_norm` and `lm_head` for the logit lens, and everything else.
        `debug.phase` is a no-op when no tracer is attached, so this costs nothing in a normal run.

        Note what is deliberately *not* labelled: the `log_softmax` over `[batch, seq, vocab]` inside `_loglikelihood_tokens`. It is a mirror of upstream and is kept diffable, and it is not an `nn.Module` so no hook fires there anyway. A trace whose last event is the model exiting cleanly, followed by an error in phase `other`, is how that allocation announces itself.
        """
        with debug.phase("model_forward"):
            return super()._model_call(*args, **kwargs)

    # -- generate ------------------------------------------------------------------

    def generate_until(self, requests, disable_tqdm: bool = False):
        """One request at a time, so each generation maps to exactly one document.

        Batch 1 is forced.
        Batching generations would mix documents of different generated lengths in one tensor, and every hook call after the shortest one has hit EOS would be a padding step that has to be filtered out.
        Batched generation is a TODO.

        Generation length is not ours to choose: whatever the lm-eval task config asks for is what runs, and the manifest records it.
        """
        requests = self._prepare_requests(requests)
        if self._recorder is None:
            return super().generate_until(requests, disable_tqdm=disable_tqdm)

        # Its own bar over documents. The per-instance calls below pass `disable_tqdm=True` - a bar per document would be 1,531 bars - so without this a generate run prints nothing at all from start to finish, which on a multi-hour task is indistinguishable from hung.
        from tqdm import tqdm

        results: list[str] = []
        with self._recorder.session(), self._resolved_auto_batch_size():
            for instance in tqdm(requests, disable=disable_tqdm,
                                 desc="Running generate_until requests (traced)"):
                self._recorder.expect_generation(
                    task_name=instance.task_name, doc_id=instance.doc_id, prompt_length=0
                )
                if self._recorder._plan is not None:
                    original = generation_prompt_origin(self, instance.args[0])
                    if original is not None:
                        self._recorder._plan["original_prompt_length"] = original
                # `super()` runs the real generation path, hooks and all.
                output = super().generate_until([instance], disable_tqdm=True)
                self._recorder.set_generated_tokens(self._last_generated_tokens)
                self._recorder.flush()
                results.extend(output)
        return results

    @contextlib.contextmanager
    def _resolved_auto_batch_size(self):
        """Pin `batch_size="auto"` to one probed value for the duration of a generate loop.

        The loop above calls `super().generate_until()` once per document, and lm-eval re-runs
        `_detect_batch_size()` on every entry to that method while `batch_size` reads "auto".
        The probed value is then discarded, because a one-instance call batches one instance
        either way - so the probe is pure overhead, paid once per document. Measured on
        SmolLM2-135M over 10 gsm8k documents at bfloat16 on an RTX A5000: the traced pass takes
        8:07 with the probe repeated and 0:16 with it resolved once, against 0:17 for the same
        pass at an explicit batch of 1. All three produce identical generations.

        The probe is kept rather than forced to 1: it still decides what the scoring pass of a
        mixed run may use, and skipping it would change behaviour rather than only its cost.

        `HFLM.batch_size` is a read-only property over `batch_size_per_gpu`, so the pin is
        written there and restored afterwards, leaving the model as this found it.
        """
        if self.batch_size != "auto":
            yield
            return
        original = self.batch_size_per_gpu
        self.batch_size_per_gpu = self._detect_batch_size()
        try:
            yield
        finally:
            self.batch_size_per_gpu = original

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Note the prompt length, then keep the token ids that came out.

        The prompt length turns a decoding step into an absolute sequence position, and the emitted tokens fill the `steps` table - with sampling on they can differ from the last layer's top-1.
        """
        if self._recorder is not None:
            self._recorder.set_prompt_length(int(context.shape[1]))
        with debug.phase("model_forward"):
            output = super()._model_generate(context, max_length, stop, **generation_kwargs)
        if self._recorder is not None:
            self._last_generated_tokens = output[0, context.shape[1] :].tolist()
        return output


def generation_prompt_ids(lm: Any, prompt: str) -> list[int]:
    """The ids lm-eval's `generate_until` encodes one prompt to, before any truncation.

    That is `tok_batch_encode`, not `tok_encode`, and under lm-eval 0.4.13 the two decide on special tokens by different rules: the batch encoder skips them when the prompt starts with `tokenizer.bos_token`, `tok_encode` only when it starts with `decode(prefix_token_id)`.
    With a BOS-prepending tokenizer, a prompt that already starts with the BOS text and a `prefix_token_id` that is not the BOS, `tok_encode` gives one token more (`<s><s>...` against the scored `<s>...`).
    Collection replayed that longer prompt, and the first pass's `original_prompt_length` counted a truncated token that never was (`scripts/verify_bos_prompt_mismatch.py`).
    """
    input_ids, _ = lm.tok_batch_encode([prompt])
    return input_ids[0].tolist()


def generation_prompt_origin(lm: Any, prompt: str) -> int | None:
    """The prompt length that makes `input_offset` count only tokens removed from the left, or None when no such count exists.

    The recorder takes `input_offset` as this length minus the prompt length the model received, and `input_offset` means tokens removed from the left.
    Without `truncation`, `generate_until` only keeps the last `max_length - max_gen_toks` ids, so this is the untruncated length.
    With `truncation=True` the tokenizer first cuts at `model_max_length` on its `truncation_side`: a right cut removes nothing from the left, so the length is what the tokenizer kept; a left cut does, so it is the untruncated length again.
    A cut that is not a contiguous slice of the untruncated ids on that side - a tokenizer that puts BOS back after cutting from the left - removed no single left run of tokens, and None leaves `input_offset` unknown.
    Measured before this: a right-cut 87-token prompt kept its first 32 tokens and was recorded with `input_offset` 55.
    """
    full = generation_prompt_ids(lm, prompt)
    if not getattr(lm, "truncation", False):
        return len(full)
    input_ids, _ = lm.tok_batch_encode([prompt], truncation=True)
    kept = input_ids[0].tolist()
    if kept == full:
        return len(full)
    if getattr(lm.tokenizer, "truncation_side", "right") == "left":
        return len(full) if kept == full[len(full) - len(kept):] else None
    return len(kept) if kept == full[:len(kept)] else None


def refuse_truncated_generate_collection(lm: Any, samples: Sequence[dict[str, Any]]) -> None:
    """Refuse to collect a generate task from a model loaded with `truncation=True`.

    With it, `generate_until` asks the tokenizer to cut each prompt at its `model_max_length`, on its `truncation_side` (the right by default, so the end of the prompt goes) before lm-eval's own left truncation.
    The run records neither the cut nor the side, so the collection pass cannot feed the prompt that was scored: on a GPU a 32-token tokenizer scored the first 32 tokens of an 87-token prompt, collection fed all 87 plus the generation, and the teacher-forced audit failed.
    Loglikelihood requests do not go through that truncation, so only generate documents are refused, and before any forward pass.
    """
    if not getattr(lm, "truncation", False):
        return
    generate = sorted({sample["task_name"] for sample in samples
                       if not (sample.get("filtered_resps") and isinstance(sample["filtered_resps"][0], (list, tuple)))})
    if generate:
        raise ValueError(
            f"cannot collect generate tasks {generate} from a model loaded with truncation=True: the tokenizer "
            "cut their prompts at model_max_length on its truncation side while scoring, and the run does not "
            "record that cut, so the collection pass would feed tokens the model never scored. Run with "
            "truncation=False (the default) to collect generate tasks.")


# --------------------------------------------------------------------------
# Collection: run a finished run's prompts again, without lm-eval
# --------------------------------------------------------------------------


def select_documents_to_collect(
    samples: Sequence[dict[str, Any]], limit: int, seed: int = SAMPLING_SEED
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pick up to `limit` documents per task and correctness group reproducibly.

    Balanced on purpose: contrasting right against wrong answers is the main use of the attention and hidden-state dumps.
    Taking the first N documents instead would be worse than it looks - many datasets are ordered by subject or genre, so the head of the file is one narrow slice.

    Correctness comes from lm-eval's own grading in `samples.jsonl`; deciding it ourselves would mean reimplementing scoring, which is the one thing this tool does not do.
    That is also why collection is a second pass: at forward time during the first pass, nothing knows yet which documents were right.

    Args:
        samples: records from `samples.jsonl`.
        limit: N per group.
            Fewer are taken when a group is smaller, and the real counts are reported back for the manifest.

    Returns:
        (selected samples in document order, counts actually taken).

    Example:
        >>> picked, counts = select_documents_to_collect(samples, limit=500)   # doctest: +SKIP
        >>> counts
        {'correct': 500, 'incorrect': 233, 'unknown': 0}
    """
    from .storage import _extract_is_correct, primary_filters

    if limit < 0:
        raise ValueError("collection limit must be nonnegative")
    primary = primary_filters(list(samples))
    groups = {}
    seen = set()
    for sample in samples:
        task = sample["task_name"]
        if sample.get("filter") is not None and sample["filter"] != primary.get(task):
            continue
        identity = (task, int(sample["doc_id"]))
        if identity in seen:
            continue
        seen.add(identity)
        verdict = _extract_is_correct(sample)
        key = "unknown" if verdict is None else ("correct" if verdict else "incorrect")
        groups.setdefault((task, key), []).append(sample)

    rng = random.Random(seed)
    picked = []
    counts = {"correct": 0, "incorrect": 0, "unknown": 0}
    for (task, key), pool in sorted(groups.items()):
        pool.sort(key=lambda sample: int(sample["doc_id"]))
        take = min(limit, len(pool))
        picked.extend(rng.sample(pool, take))
        counts[key] += take
    picked.sort(key=lambda s: (s["task_name"], int(s["doc_id"])))
    return picked, counts


class ResearchDataCollector:
    """Feed a finished run's prompts through the model again to collect more signals.

    lm-eval is not involved: scoring finished in the first pass, and all that is left is one more forward pass with different hooks attached.

    Prompts are reused verbatim from `samples.jsonl` rather than rebuilt.
    Rebuilding them would change the few-shot examples: the few-shot sampler carries its random state forward, so evaluating a subset of documents hands each of them different examples than the first pass did.
    The attention maps would then belong to prompts the model never saw during scoring - and would look completely normal.
    Re-running lm-eval on a subset has the same problem plus renumbered `doc_id`s.

    Both passes must produce the same input tokens, which is checked three ways: identical tokenizer settings (taken from the run's manifest, never re-entered by the user), batch size 1 so padding cannot differ, and a prompt hash comparison against `samples.jsonl`.

    Being a second pass is not necessarily slower.
    `eager` attention only runs on at most 2N documents here, where a single-pass version would have to run the entire task in eager.

    Example:
        >>> runner = ResearchDataCollector(lm, recorder, run_dir)              # doctest: +SKIP
        >>> runner.run(limit=500)
        {'correct': 500, 'incorrect': 500, 'unknown': 0, 'documents': 1000}
    """

    def __init__(self, lm: HFLM, recorder: Recorder, run_dir: str | os.PathLike) -> None:
        self.lm = lm
        self.recorder = recorder
        self.run_dir = str(run_dir)
        self._emitted: dict[tuple[str, int], list[Any]] = {}
        self._has_steps = False

    def run(self, limit: int, seed: int = SAMPLING_SEED) -> dict[str, int]:
        """Collect from a balanced sample of the run's documents.

        Returns:
            The counts actually collectioned, to be written into the manifest.
        """
        samples = read_samples(self.run_dir)
        picked, counts = select_documents_to_collect(samples, limit=limit, seed=seed)
        refuse_truncated_generate_collection(self.lm, picked)
        self._emitted = self._recorded_generations(picked)

        documents = 0
        with self.recorder.session():
            for sample in picked:
                documents += self._collection_one(sample)
        counts["documents"] = documents
        return counts

    def _collection_one(self, sample: dict[str, Any]) -> int:
        """Collect from every forward pass of one document.

        Flushed once per (document, choice), not once per document: the tensor dumps are one file per forward pass, and a multiple-choice document runs one forward per choice.
        """
        arguments = sample.get("arguments") or []
        responses = sample.get("filtered_resps") or []
        is_loglikelihood = bool(responses) and isinstance(responses[0], (list, tuple))

        self._verify_prompt(sample, arguments)
        if is_loglikelihood:
            for choice_idx, args in enumerate(arguments):
                if not self.recorder.custom.begin_unit(sample["task_name"], int(sample["doc_id"]), choice_idx):
                    continue
                context, continuation = args[0], args[1]
                context = self.lm.prepare_context(context) if hasattr(self.lm, "prepare_context") and not sample.get("model_prompt_prepared") else context
                self._forward_loglikelihood(
                    sample["task_name"], int(sample["doc_id"]), choice_idx, context, continuation
                )
                self.recorder.flush()
                self.recorder.custom.finish_unit()
        else:
            if not self.recorder.custom.begin_unit(sample["task_name"], int(sample["doc_id"]), 0):
                return 1
            context = arguments[0][0] if arguments else ""
            context = self.lm.prepare_context(context) if hasattr(self.lm, "prepare_context") and not sample.get("model_prompt_prepared") else context
            generated = self._generated_tokens(sample)
            self._forward_generation(
                sample["task_name"], int(sample["doc_id"]), context, generated
            )
            self.recorder.flush()
            self.recorder.custom.finish_unit()
        return 1

    def _recorded_generations(self, picked: Sequence[dict[str, Any]]) -> dict[tuple[str, int], list[Any]]:
        """The token ids each picked generate document emitted, from the first pass's `steps` table.

        Collection teacher-forces these ids, not any text. It used to replay `filtered_resps`, which is what a task's filters made of the generation: for gsm8k the extracted answer, `[invalid]` when nothing matched. A document that generated 256 tokens was then collected over a three-token string, and every attention map, hidden-state dump and module statistic of the collection pass described that string. The raw `resps` text is closer but is still not the input - decoding and re-tokenising does not round-trip (283 tokens back from 256 emitted on a random Mixtral) - whereas the recorded ids are exactly what the model emitted and what the first pass's `steps` and signals describe.
        """
        from .storage import read_table

        wanted = set()
        for sample in picked:
            responses = sample.get("filtered_resps") or []
            if responses and not isinstance(responses[0], (list, tuple)):
                wanted.add((sample["task_name"], int(sample["doc_id"])))
        steps = read_table(self.run_dir, "steps")
        self._has_steps = bool(len(steps))
        emitted: dict[tuple[str, int], list[Any]] = {}
        if not wanted or not self._has_steps:
            return emitted
        rows = steps[steps.choice_idx == 0]
        for (task, doc), group in rows.groupby(["task_name", "doc_id"]):
            key = (str(task), int(doc))
            if key in wanted:
                emitted[key] = group.sort_values("step").token_id.tolist()
        return emitted

    def _generated_tokens(self, sample: dict[str, Any]) -> list[int]:
        """Token ids to teacher-force for one generate document; see `_recorded_generations`."""
        key = (sample["task_name"], int(sample["doc_id"]))
        if key in self._emitted:
            ids = self._emitted[key]
            if any(token is None or token != token for token in ids):
                raise RuntimeError(
                    f"the steps table has a missing token id for {key[0]} doc {key[1]}, so its "
                    "generation cannot be replayed exactly")
            return [int(token) for token in ids]
        if self._has_steps:
            raise RuntimeError(
                f"no generated tokens recorded for {key[0]} doc {key[1]} in the steps table; "
                "refusing to replay a different sequence than the one the first pass scored")
        # A run without internal signals writes no steps table, so lm-eval's raw generation - never the filtered answer - is the only record of what was generated.
        raw = (sample.get("resps") or [[""]])[0]
        raw = raw[0] if isinstance(raw, (list, tuple)) else raw
        return self.lm.tok_encode(str(raw), add_special_tokens=False)

    def _verify_prompt(self, sample: dict[str, Any], arguments: Sequence[Sequence[str]]) -> None:
        """Check the reused prompt is byte-identical to the one that was scored.

        `samples.jsonl` carries the sha256 of the first request's context; if it does not match, the two passes are looking at different prompts and the signals must not be attributed to this document.
        """
        from lm_eval.utils import hash_string

        expected = sample.get("prompt_hash")
        if not expected or not arguments:
            return
        actual = hash_string(arguments[0][0])
        if actual != expected:
            raise RuntimeError(
                f"prompt hash mismatch for {sample['task_name']} doc {sample['doc_id']}: "
                f"samples.jsonl says {expected[:12]}..., collection built {actual[:12]}.... "
                "The two passes are not running the same prompt."
            )

    def _forward_loglikelihood(
        self, task_name: str, doc_id: int, choice_idx: int, context: str, continuation: str
    ) -> None:
        """One (context, continuation) pair, tokenised exactly as the first pass was."""
        if context == "":
            context_enc, continuation_enc = [self.lm.prefix_token_id], self.lm.tok_encode(continuation)
        else:
            context_enc, continuation_enc = self.lm._encode_pair(context, continuation)

        inp = torch.tensor(
            (context_enc + continuation_enc)[-(self.lm.max_length + 1) :][:-1],
            dtype=torch.long,
            device=self.lm.device,
        ).unsqueeze(0)
        inplen = inp.shape[1]
        contlen = len(continuation_enc)
        positions = list(range(inplen - contlen, inplen))

        self.recorder.expect_loglikelihood(
            [
                ForwardContext(
                    task_name=task_name,
                    doc_id=doc_id,
                    choice_idx=choice_idx,
                    steps=list(range(contlen)),
                    positions=positions,
                    target_token_ids=list(continuation_enc),
                    n_residual=self.recorder.adapter.n_residual,
                    n_blocks=self.recorder.adapter.n_blocks,
                    task_kind="loglikelihood",
                    input_length=inplen,
                    input_offset=max(0, len(context_enc) + len(continuation_enc) - 1 - inplen),
                )
            ]
        )
        with torch.no_grad(), debug.phase("model_forward"):
            self.lm.model(inp)

    def _forward_generation(
        self, task_name: str, doc_id: int, context: str, generated_enc: Sequence[int]
    ) -> None:
        """Teacher-force the tokens the first pass generated, instead of generating again.

        Re-generating would produce different text whenever sampling is on, and the signals would then describe an output the run never scored.
        Feeding prompt + generation in one pass is also cheaper than decoding.

        Caveat, and it belongs in the output description as well: a single packed forward pass and an incremental KV-cache decode are not bitwise identical, so collectioned generate signals can differ slightly in the last digits from what the first pass saw.
        """
        context_enc = generation_prompt_ids(self.lm, context)
        generated_enc = [int(token) for token in generated_enc]
        if not generated_enc:
            return
        ids = (context_enc + generated_enc)[-self.lm.max_length :]
        prompt_len = len(ids) - len(generated_enc)
        inp = torch.tensor(ids, dtype=torch.long, device=self.lm.device).unsqueeze(0)

        # `step t` is the position that predicted generated token t, matching the first pass, where step 0 is the last prefill position.
        positions = list(range(prompt_len - 1, prompt_len - 1 + len(generated_enc)))
        self.recorder.expect_loglikelihood(
            [
                ForwardContext(
                    task_name=task_name,
                    doc_id=doc_id,
                    choice_idx=0,
                    steps=list(range(len(generated_enc))),
                    positions=positions,
                    target_token_ids=None,          # generate has no gold token to rank
                    step_token_ids=list(generated_enc),
                    n_residual=self.recorder.adapter.n_residual,
                    n_blocks=self.recorder.adapter.n_blocks,
                    task_kind="generate",
                    input_length=len(ids),
                    input_offset=max(0, len(context_enc) + len(generated_enc) - len(ids)),
                )
            ]
        )
        with torch.no_grad(), debug.phase("model_forward"):
            self.lm.model(inp)
