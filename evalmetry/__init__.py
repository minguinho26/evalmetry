"""Per-layer internal signal collection alongside an lm-eval benchmark run.

Scoring is lm-eval's, unchanged.
This package only pulls extra research signals out of the same forward passes and writes them in a fixed, self-describing format.

The reading entry point is `load_signals`; `describe_schema` prints what the columns mean.
"""

from .storage import SCHEMA_VERSION, describe_schema, load_signals, read_manifest, read_sample_metrics

__all__ = ["SCHEMA_VERSION", "describe_schema", "load_signals", "read_manifest", "read_sample_metrics"]

from .hooks import HookSpec, HookContext

__all__ += ["HookSpec", "HookContext"]

from .models import ModelBundle, fingerprint_files

__all__ += ["ModelBundle", "fingerprint_files", "RunConfig", "run"]


def __getattr__(name):
    """Load evaluation entrypoints lazily so `python -m evalmetry.main` works."""
    if name in {"RunConfig", "run"}:
        from .main import RunConfig, cmd_run
        return {"RunConfig": RunConfig, "run": cmd_run}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
