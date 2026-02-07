"""Dataset processors register and factory.

Processors are small objects that know how to standardize a raw download into
the repository's `data/vector/{name}` and `data/raster/{name}` layout.
"""
from .base import Processor
from .floorplancad import FloorPlanCADProcessor
from .resplan import ResPlanProcessor
from .msd import MSDProcessor
from .sketchgraphs import SketchGraphsProcessor
from .cubicasa import CubiCasa5KProcessor
from .cadvgdrawing import CADVGDrawingProcessor
from .fplanpoly import FPLANPOLYProcessor
from .quickdraw import QuickDrawProcessor

_REGISTRY = {
    'floorplancad': FloorPlanCADProcessor,
    'resplan': ResPlanProcessor,
    'msd': MSDProcessor,
    'quickdraw': QuickDrawProcessor,
    'sketchgraphs': SketchGraphsProcessor,
    'cubicasa5k': CubiCasa5KProcessor,
    'cadvgdrawing': CADVGDrawingProcessor,
    'fplanpoly': FPLANPOLYProcessor,
}


def get_processor(name: str) -> Processor:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"No processor registered for: {name}")
    return cls()

__all__ = ['get_processor', 'Processor']
