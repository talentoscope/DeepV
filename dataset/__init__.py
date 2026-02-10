"""Top-level dataset helpers.

This package provides small adapters and processors for datasets. Download logic
is implemented in the existing `dataset_downloaders` package; processors live
under `dataset.processors` and standardize raw downloads into tidy
`data/vector/{dataset}` and `data/raster/{dataset}` folders.

Submodules:
    downloaders: Dataset downloading utilities
    processors: Dataset processing and standardization

Example:
    from deepv.dataset import processors
    from deepv.dataset.processors import FloorPlanCADProcessor

    # Process FloorPlanCAD dataset
    processor = FloorPlanCADProcessor()
    processor.process_dataset("path/to/raw/data", "path/to/processed/data")
"""

from . import downloaders, processors

__all__: list[str] = ["downloaders", "processors"]
