"""Merging module for DeepV vectorization pipeline.

This module provides functionality for merging and consolidating primitive shapes
detected during the vectorization process. It handles both line and curve merging
to reduce over-segmentation and create clean vector representations.

Submodules:
    utils: Core merging functions and algorithms

Example:
    from deepv.merging import utils
    from deepv.merging.utils import merge_close_primitives

    # Merge primitives within tolerance
    merged_primitives = merge_close_primitives(primitives, tolerance=1.0)
"""

from . import utils

__all__ = ["utils"]
