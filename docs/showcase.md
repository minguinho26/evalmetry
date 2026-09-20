# Do more accurate models settle on a prediction earlier?

A benchmark score records how often a model answers correctly. Evalmetry's saved layer signals support a second measurement: the depth at which a model's token prediction stops changing.

In this exploratory study of seven models, higher accuracy went with **later settlement on ARC Challenge** and **earlier settlement on MMLU high school psychology**. The directions are opposite, so task format and token position need examination before prediction depth is used as a measure of model quality.

## Reading prediction depth

At each layer, the logit lens turns the model's hidden state into a vocabulary prediction using its final normalization and output head. The measurement takes the earliest layer whose top prediction agrees with the final layer and stays the same through the remaining layers. That layer number divided by the number of transformer blocks gives **settlement depth**: 0 is the embedding state and 1 is the final layer. Smaller values mean earlier settlement.

For example, if the last three layer predictions are `cat → dog → dog`, settlement begins at the second of those layers. The measure covers stability of the final token prediction. It does not establish when the model found the correct answer or finished reasoning, and a stable prediction can be wrong.

For each document, this study follows the **gold answer choice's first token** (`step == 0`) and takes the median depth across documents. Benchmark accuracy still comes from lm-eval's choice scoring, separately from the lens prediction. Each point below represents one model evaluated on 500 documents.

![Benchmark accuracy versus median first-token settlement depth for seven models on each of two tasks](prediction-depth.png)

The panels share their axes: farther right means later settlement, and higher means better benchmark accuracy. ARC uses length-normalized accuracy (`acc_norm`). MMLU uses accuracy (`acc`). Circles mark the four Qwen3 models and squares mark the other model families. The seven models were selected rather than sampled at random, so the figure presents observations without a fitted trend or confidence interval.

## The task changes the relationship

Across all seven models, the correlation between accuracy and median first-token settlement depth is **+0.878 on ARC** and **−0.798 on MMLU** (Pearson r). Restricting the comparison to the four Qwen3 models preserves the signs: +0.900 and −0.909, respectively.

| Qwen3 model | ARC accuracy | ARC settlement depth | MMLU accuracy | MMLU settlement depth |
| --- | ---: | ---: | ---: | ---: |
| 0.6B | 32.0% | 0.8571 | 57.0% | 0.8929 |
| 1.7B | 43.6% | 0.9643 | 78.2% | 0.8214 |
| 4B | 53.0% | 0.9722 | 88.0% | 0.6944 |
| 8B | 56.2% | 0.9722 | 91.6% | 0.7500 |

The Qwen3 comparison shows that the contrast persists within one model family. The pattern is not strict even there: on MMLU, 8B is more accurate than 4B but settles later. Neither correlation establishes that settling earlier or later causes a change in accuracy.

**Token position.** ARC's scored answer continuations average roughly 6–7 tokens here. MMLU's answer-letter continuations have one token. Averaging all ARC positions would mix first-token prediction with prediction inside an already supplied answer continuation. Selecting the first position gives a more focused comparison, but it does not control for differences in task content or answer format. This experiment does not establish what causes the opposite signs.

Exact values and scoring conditions are available in the [model/task summary](evidence/prediction_summary.tsv), [correlation table](evidence/depth_correlations.tsv) and [historical scores and model revisions](evidence/historical_scores.tsv).

## Reproducing the analysis from saved data

The repository includes [25,821 saved settlement observations](evidence/prediction_depth.parquet) with model, task, document, choice, token position, correctness and depth. Each row is one token position rather than one document, and several rows can come from the same document. The gold-choice selection has already been applied to this file.

The following runs from the repository root in an environment with Evalmetry's pandas and Parquet dependencies installed. It reads the saved data and needs no GPU or model download:

```python
import pandas as pd

rows = pd.read_parquet("docs/evidence/prediction_depth.parquet")
first = rows[rows["step"] == 0]

# Recover the model/task medians used in the figure.
print(first.groupby(["task", "model"])["depth"].median())

# Compare settlement depth for correct and incorrect answers.
print(first.groupby(["task", "model", "is_correct"])["depth"].median())
```

The second grouping uses the benchmark's correctness label. It relates prediction stability to task outcome without another inference run.

The same public readers and depth calculation apply to any completed multiple-choice run with default signals:

```python
from evalmetry.storage import load_signals, read_manifest
from evalmetry.report import prediction_depth


def first_token_depth(run_path):
    signals = load_signals(run_path, "signals")
    selected = signals[signals["is_target_choice"] & (signals["step"] == 0)]
    n_blocks = read_manifest(run_path)["n_blocks"]
    return prediction_depth(selected, n_blocks)
```

`first_token_depth` takes a run directory and returns per-document depths and correctness labels. The CLI's `report` command also provides grouping and comparison views. Default signals are enough for this analysis. Optional axes/extremes collection has a separate [measured cost](verification.md#collection-accuracy-cost-and-limits).

## Study scope and next experiment

These are historical observations from **September 10, 2026**: seven models, 500 documents per model/task, zero-shot direct prompting, CUDA bfloat16, requested batch 16 and lm-eval 0.4.9.1. The [1.1.0 verification report](verification.md) separately documents score agreement with stock lm-eval, including the requirement to match actual batch sizes for generation.

The retained per-position rows reproduce the settlement medians, and the summary reproduces the correlations. The original per-layer dumps and complete software, seed, dataset-revision and collection-command manifest are unavailable, so the files do not support an exact replay of the original experiment. A separate historical top-10 arrival metric is excluded from this showcase. Its correction and evidence limits are recorded in the [analysis audit note](verification.md#historical-analysis-correction).

A next experiment could hold the model, document selection and actual batch fixed while varying answer format, then compare first-token depth with uncertainty estimated at the document level. That design would test whether the task contrast survives the control. The observations here do not settle it.
