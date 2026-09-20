# Releases and publishing

## 1.1.0 — September 20, 2026

Published on [PyPI](https://pypi.org/project/evalmetry/1.1.0/).
The verified wheel and sdist were uploaded directly with Twine. GitHub Release
creation is still pending. The existing workflow uploads when a GitHub Release
is published; before creating the 1.1.0 GitHub Release, account for that trigger
so it does not attempt to rebuild and upload this already-published version.

### Breaking change

`--num-fewshot` is now required, and `RunConfig.num_fewshot` has no default.
Choose the shot count explicitly for the intended benchmark configuration:

```bash
evalmetry run --model-args pretrained=Qwen/Qwen3-0.6B,dtype=bfloat16,device=cuda \
    --tasks arc_easy --num-fewshot 0 --limit 8 --batch-size 1 \
    --output results/quickstart
```

Python callers must likewise pass `num_fewshot`, for example
`RunConfig(model_args="pretrained=...", tasks=["gsm8k"], num_fewshot=5)`.

### Fixes and additions

- Generation with automatic batching probes once per generation loop instead of
  repeating the probe for each document. The traced generation batch remains 1.
- Run manifests now record tool version 1.1.0, matching the distribution version.
  Version 1.0.0 incorrectly recorded tool version 0.1.0.
- `module-stats --pass collection` now explains which passes exist and how to
  create a missing collection pass.
- A package docstring no longer refers to development-only review files.
- Run manifests record phase timings for model loading, evaluation, collection
  and end-to-end execution; `collect-research-data` records collection timings.
- The repository includes a [verification report](docs/verification.md) and a
  [prediction-depth research showcase](docs/showcase.md) with supporting data.

### Verification and published artifacts

- Ten checkpoints across five tasks passed all 50 stock-versus-traced comparisons.
  Three multiple-choice tasks used their full datasets with bit-identical request
  log probabilities; two generation tasks used 200 documents each.
- Six additional comparisons exercised optional heavy collection across three
  architectures and two tasks.
- Generation equality is established **at the same actual batch size of 1**.
  A separate 24-run experiment confirmed all twelve matched-batch comparisons.
  Changing stock batching can change outputs and scores: Qwen3 GSM8K changed
  from 46.875% at batch 1 to 43.75% at batch 8 on 32 documents.
- Final wheel and sdist passed strict Twine validation. Rebuilding the sdist
  produced identical uncompressed wheel contents. Independent installation,
  dependency, import, CLI and offline local CPU scoring/collection checks passed.
  These installation checks are separate from the retained GPU parity evidence.
- The exact verified files were uploaded, then downloaded from PyPI and checked
  against their recorded SHA-256 values. See the [artifact record](docs/evidence/release_artifacts_20260920.json)
  for filenames, hashes, metadata and validation details.

## Publishing future releases

The `Publish to PyPI` workflow builds a wheel and source distribution, checks the
metadata, and verifies imports and CLI commands in a clean environment before
uploading to PyPI. Model evaluation still requires separate CUDA validation.

### One-time PyPI setup

Register a pending Trusted Publisher at
<https://pypi.org/manage/account/publishing/> with these values:

| Field | Value |
| --- | --- |
| PyPI project name | `evalmetry` |
| Owner | `minguinho26` |
| Repository name | `evalmetry` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

If the project already exists, add the publisher in its Publishing settings.
No long-lived PyPI API token is required.

### Release steps

1. Update `project.version` in `pyproject.toml`, `TOOL_VERSION` in
   `evalmetry/main.py`, the versioned README example and the release notes here.
2. Commit and push the changes, then create and push a matching `vX.Y.Z` tag.
3. Publish a GitHub Release for that tag. This starts the publishing workflow.
4. Confirm the workflow succeeded and install the version from PyPI in a fresh environment.

If the first run fails because PyPI setup is incomplete, finish the setup and
rerun the failed job. The workflow can also be started manually with an existing
release tag. PyPI does not allow replacing files in an uploaded release; publish
a new version for package changes.
