"""
datasets_shim.py — Must be imported BEFORE sentence_transformers.

Injects a minimal stub for the 'datasets' package so that sentence_transformers
can be imported for inference on machines where pyarrow DLLs are blocked by
Application Control policy. Training helpers (sampler/trainer) are stubbed out;
encode() / cross_encode() work normally.
"""

import sys
import types
import importlib.machinery


class _FakeDatasetsModule(types.ModuleType):
    """Minimal stub for 'datasets' to bypass the pyarrow DLL block."""

    __file__ = "<stub>"
    __path__: list = []
    __version__ = "0.0.0"
    __spec__ = importlib.machinery.ModuleSpec("datasets", None)

    def __getattr__(self, name: str):
        # Allow dunder attribute lookup to fail normally
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Return a trivial no-op class for any other attribute
        return type(name, (), {"__init__": lambda self, *a, **kw: None})


def _install_shim() -> None:
    """Install the datasets stub into sys.modules if pyarrow is blocked."""
    if "datasets" in sys.modules:
        return  # already imported (real or stub)

    # Test whether pyarrow can actually load
    try:
        import pyarrow  # noqa: F401
        return  # pyarrow is fine — no shim needed
    except ImportError:
        pass

    stub = _FakeDatasetsModule("datasets")
    for sub in (
        "datasets",
        "datasets.arrow_dataset",
        "datasets.iterable_dataset",
        "datasets.utils",
        "datasets.utils.logging",
        "datasets.features",
    ):
        sys.modules[sub] = stub


_install_shim()
