# Evalmetry 1.1.0 verification

## Result and scope

The traced backend matched stock lm-eval on all **50 model/task comparisons** in the CUDA verification sweep. Multiple-choice log probabilities matched bit for bit over three complete datasets. Generation strings matched on 200 documents per task **at the same actual batch size of 1**. This is evidence for the tested configurations, not a guarantee across hardware, precision, revisions or batch sizes.

“Collection off” below means the optional heavy collection pass is off. The traced scoring pass still has its recorder and default reducers attached. The question is whether observation changes the score, not whether an idle wrapper agrees with upstream.

| Task | Documents per model | Compared outputs per model | Result |
| --- | ---: | ---: | --- |
| ARC Easy | 2,376 (full test split) | 9,501 choice log probabilities | All bit-identical; max absolute difference 0 |
| HellaSwag | 10,042 (full validation split) | 40,168 choice log probabilities | All bit-identical; max absolute difference 0 |
| MMLU high school psychology | 545 (full test split) | 2,180 choice log probabilities | All bit-identical; max absolute difference 0 |
| GSM8K | First 200 test documents | 400 document/filter outputs | Raw generations and filtered outputs identical |
| TriviaQA | First 200 validation documents | 200 document/filter outputs | Raw generations and filtered outputs identical |

Ten checkpoints were used: HuggingFaceTB/SmolLM2-135M, Qwen/Qwen2.5-0.5B,
Qwen/Qwen3-0.6B, EleutherAI/pythia-160m, allenai/OLMo-2-0425-1B,
HuggingFaceTB/SmolLM3-3B, unsloth/gemma-2-2b,
microsoft/Phi-3-mini-4k-instruct,
hf-internal-testing/tiny-random-MistralForCausalLM and
hf-internal-testing/tiny-random-CohereForCausalLM. The two random checkpoints
exercise architecture paths; their scores are not capability measurements.
This covers ten adapter layouts, not every supported architecture or MoE.

The [per-run verdicts](evidence/parity_verdicts.json) retain comparison lines and
original log hashes. [Scores](evidence/parity_scores.tsv) retain repeated runs:
use `sample_len` and `config` to distinguish full-dataset, limited and collection
runs. The [request-level summary](evidence/score_parity_direct.tsv) also contains
older experiments; do not treat every row as part of this release sweep.

## Generation equality requires matching the batch that actually ran

A separate 24-run experiment used two models × two tasks × stock/traced ×
requested batches 1, 8 and auto, with 32 documents in each run. All twelve
comparisons of stock batch 1 against traced requested 1/8/auto matched raw
strings, filtered outputs and scores exactly (576 document/filter comparisons).
Every traced generation call actually used batch 1. Stock auto used batch 8 for
SmolLM2 and batch 1 for Qwen3 in this environment.

| Stock batch 1 versus stock batch 8 | Raw generations changed / 32 |
| --- | ---: |
| SmolLM2-135M, GSM8K | 22 |
| SmolLM2-135M, TriviaQA | 6 |
| Qwen3-0.6B, GSM8K | 17 |
| Qwen3-0.6B, TriviaQA | 6 |

Qwen3 GSM8K exact match changed from **15/32 (46.875%) to 14/32 (43.75%)**
for both strict and flexible extraction. All traced settings retained 15/32.
Input hashes matched. Batch-dependent floating-point execution is a plausible
mechanism, but this experiment did not isolate a particular kernel as the cause.
Equal requested batch labels do not establish an equal execution condition.
See the [complete experiment record](evidence/generation_batch_20260920.json).

## Heavy collection and distribution code

With attention dumps, hidden-state dumps, module statistics and an evaluation-pass
hook enabled, SmolLM2-135M, Pythia-160m and Phi-3-mini matched stock on ARC Easy
(200 documents) and GSM8K (4 documents). These six small comparisons target
hook/layout interactions. Phi-3 GSM8K scored 1.0 on both sides, so the generation
check is not supported solely by zero-score runs. Four documents are not a model
quality estimate.

An earlier 1.1.0 wheel was unpacked and tested on SmolLM2-135M and Qwen3-0.6B:
full ARC Easy and 200-document GSM8K all passed. Package sources matched the
public tree at `c25dbcb086ad74dc668789bb272af840e4b6c26c`.
The [distribution parity record](evidence/release_parity_20260920.json) describes
that run. It is distinct from the final upload artifacts and did not itself test
clean dependency installation. Final artifact installation and hashes belong in
the release artifact verification record; no new GPU parity run is implied.

## Comparison method and reproducibility

The GPU environment was RTX A5000 24 GB, driver 535.261.03, PyTorch 2.8.0+cu128,
Transformers 5.16.1 and lm-eval 0.4.13, using CUDA bfloat16. Construct upstream
`lm_eval.models.huggingface.HFLM` and `evalmetry.backend.TracedHFLM` from the same
model arguments. Attach Evalmetry's recorder and default reducers to the latter.
Call `lm_eval.simple_evaluate` with the same task, limit, seeds, task configuration
and tokenizer, `log_samples=True` and no response cache. Align samples by task,
document, choice and filter; compare raw responses before comparing aggregate
metrics. Require exact numeric and string equality; do not substitute a tolerance
or merely compare rounded scores. Multiple-choice requests compare each choice's
summed log probability. Generation compares raw strings and each task filter.

The main sweep requested auto batching (maximum probe 64); actual multiple-choice
batch was 64 except Gemma-2 MMLU at 32. Generation stock was explicitly aligned to
traced batch 1. The checker used task-default fewshot settings; the dedicated
batch experiment records GSM8K 5-shot and TriviaQA 0-shot, seeds 0/1234/1234/1234
(Python/NumPy/PyTorch/fewshot), no chat template, maximum 256 generated tokens and
no context-length override. Its JSON includes model revisions, full task configs,
dataset split fingerprints, software versions, execution observation and result
hashes. The source commit was `07f5527ea3e4663dde309fd2fb95a3690e29a9da`.

**Provenance limits:** the broad sweep's retained logs and TSV do not preserve a
complete immutable dataset/model revision manifest for every run. Do not infer
those revisions from today's upstream repositories. The dedicated batch record
is more complete; its dataset revision notes explicitly distinguish cached split
fingerprints from upstream revision information. Exact historical replay of all
50 combinations is therefore not fully specified by the surviving artifacts.
The comparisons themselves used the same inputs on both sides. Verification
scripts and large raw dumps are intentionally not distributed.

## Collection accuracy, cost and limits

[Chunk-invariance measurements](evidence/chunk_invariance_scale_20260920.json)
cover 15 tiny architectures and six real models. Statistics agreed between
65,536-element and 257-element accumulation chunks. This tests numerical
invariance of tensor reduction, not dataset loading or batch invariance.

[Axes/extremes cost](evidence/axes_extremes_cost_20260919.json) on Qwen3-8B,
ARC Easy, eight documents, bfloat16 and batch 8 reached about 277.5 MB/document
and 131.5 seconds of aggregation. The paired axes/extremes-off run separates
that option's additional cost. These are workload-specific measurements, not
storage or speed promises. [Debug numeric cost](evidence/debug_numeric_cost_20260919.json)
is recorded separately. Start with default signals and a small document limit.

MoE routing on a real large model remains unverified and is not a release gate.
Neither CPU installation checks nor the [historical research showcase](showcase.md)
substitute for GPU numerical evidence. Future verification should preserve full
revision manifests and compare actual execution batches whenever dependencies,
hardware or scoring paths change.
