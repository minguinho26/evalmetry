# Publishing a release

The `Publish to PyPI` workflow builds a wheel and source distribution, checks the
metadata, and verifies imports and CLI commands in a clean environment before
uploading to PyPI. Model evaluation still requires separate CUDA validation.

## One-time PyPI setup

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

## Release steps

1. Update `project.version` in `pyproject.toml` and the versioned README example.
2. Commit and push the changes, then create and push a matching `vX.Y.Z` tag.
3. Publish a GitHub Release for that tag. This starts the publishing workflow.
4. Confirm the workflow succeeded and install the version from PyPI in a fresh environment.

If the first run fails because PyPI setup is incomplete, finish the setup and
rerun the failed job. The workflow can also be started manually with an existing
release tag. PyPI does not allow replacing files in an uploaded release; publish
a new version for package changes.
