# Evalmetry

A research toolkit for measuring and analyzing model behavior.

Evalmetry evaluates Hugging Face language models with lm-eval and collects internal signals from the same forward passes used for scoring.

## Features

- Benchmark scores with per-layer logit lens and layer similarity.
- Optional attention, hidden states, custom hooks and module statistics.
- Custom benchmarks, LLM judge scoring and modified-model adapters.
- Resumable runs, saved-data readers and comparison reports.

Model evaluation supports one process on one CUDA GPU. CPU execution is for test fixtures and verification tools. Reports and saved-data readers do not require a GPU.

## Evidence and research example

Read the [1.1.0 verification report](https://github.com/minguinho26/evalmetry/blob/release-1.1.0/docs/verification.md) and the [prediction-depth research showcase](https://github.com/minguinho26/evalmetry/blob/release-1.1.0/docs/showcase.md). Generation parity is established at the same actual batch size; generation is not batch-invariant.

## Install

Install from PyPI in your Python environment:

```bash
pip install evalmetry
```

Evalmetry requires Python >=3.10. Use a compatible CUDA build of PyTorch. The Python import and CLI are both `evalmetry`.

To install this release explicitly, use `pip install evalmetry==1.1.0`.
For development, clone this repository and run `python -m pip install -e .`.

## Quick start

Use `run` to evaluate eight documents and save the default signals:

```bash
evalmetry run --model-args pretrained=Qwen/Qwen3-0.6B,dtype=bfloat16,device=cuda \
    --tasks arc_easy --num-fewshot 0 --limit 8 --batch-size 1 \
    --output results/quickstart
```

Use `report` to generate a report from that run:

```bash
evalmetry report results/quickstart --output report/quickstart
```

The first run downloads the model and dataset if needed. Reusing a run directory resumes its recorded configuration; use a different directory for a different experiment.

Other commands: `collect-research-data` adds optional tensors to a completed run, `debug` reads a saved module trace, and `module-stats` reads saved statistics. Traces and statistics must be enabled during collection. Use `evalmetry <command> --help` for options.

## Reading results

Evaluation results, manifests and collected signals are written under the selected output directory. Read saved signals with `evalmetry.load_signals(run_dir)`; `evalmetry.describe_schema()` describes the columns.

## License

Evalmetry is released under the MIT License.

This project is built on top of [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
