#!/usr/bin/env python3
"""
Preprocessed Vector Data Module

PyTorch Dataset classes for preprocessed SVG vector data.
Combines SVG data loading with preprocessing utilities for training.

Features:
- PreprocessedSVG dataset class for standard preprocessing
- PreprocessedSVGPacked class for packed preprocessing
- Integration with patch extraction and primitive conversion

Used by training pipelines for SVG data loading.
"""

from util_files.data.preprocessing import PreprocessedBase, PreprocessedPacked
from util_files.data.vectordata.prepatched import PrepatchedSVG


class PreprocessedSVG(PrepatchedSVG, PreprocessedBase):
    def __init__(self, **kwargs):
        PreprocessedBase.__init__(self, **kwargs)
        PrepatchedSVG.__init__(self, **kwargs)

    def __getitem__(self, idx):
        return self.preprocess_sample(PrepatchedSVG.__getitem__(self, idx))


class PreprocessedSVGPacked(PrepatchedSVG, PreprocessedPacked):
    def __init__(self, **kwargs):
        PreprocessedPacked.__init__(self, **kwargs)
        PrepatchedSVG.__init__(self, **kwargs)

    def __getitem__(self, idx):
        return self.preprocess_sample(PrepatchedSVG.__getitem__(self, idx))
