"""Top-level dataset helpers.

This package provides small adapters and processors for datasets. Download logic
is implemented in the existing `dataset_downloaders` package; processors live
under `dataset.processors` and standardize raw downloads into tidy
`data/vector/{dataset}` and `data/raster/{dataset}` folders.
"""

from . import downloaders, processors

__all__ = ["downloaders", "processors"]
