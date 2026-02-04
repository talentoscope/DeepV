"""Dataset processors register and factory.

Processors are small objects that know how to standardize a raw download into
the repository's `data/vector/{name}` and `data/raster/{name}` layout.
"""
from .base import Processor
from .floorplancad import FloorPlanCADProcessor
from .deeppatent2 import DeepPatent2Processor

_REGISTRY = {
    'floorplancad': FloorPlanCADProcessor,
    'deeppatent2': DeepPatent2Processor,
}


def get_processor(name: str) -> Processor:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"No processor registered for: {name}")
    return cls()

__all__ = ['get_processor', 'Processor']
