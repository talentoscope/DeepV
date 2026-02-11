"""Compatibility shim for legacy `dataset_downloaders` package.

This module forwards imports to the new `dataset.downloaders.download_dataset` module.
It exists to make an incremental migration safe; it can be removed once callers
are updated to use `dataset.*` APIs.
"""
import sys
from importlib import import_module

try:
    _mod = import_module("dataset.downloaders.download_dataset")
except ImportError as e:
    print(f"Error: Failed to import dataset.downloaders.download_dataset: {e}", file=sys.stderr)
    sys.exit(1)

# Re-export common symbols
for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)

__all__ = [n for n in dir(_mod) if not n.startswith("_")]
