#!/usr/bin/env python3
"""
Preprocessed Synthetic Data Module

PyTorch Dataset class for preprocessed synthetic handcrafted data.
Combines synthetic data generation with preprocessing utilities for training.

Features:
- PreprocessedSyntheticHandcrafted dataset class
- Integration of synthetic generation with preprocessing pipeline
- Sample preprocessing and normalization

Used by training pipelines for synthetic data loading.
"""

from util_files.data.preprocessing import PreprocessedBase
from util_files.data.syndata.datasets import SyntheticHandcraftedDataset


class PreprocessedSyntheticHandcrafted(SyntheticHandcraftedDataset, PreprocessedBase):
    def __init__(self, **kwargs):
        PreprocessedBase.__init__(self, **kwargs)
        SyntheticHandcraftedDataset.__init__(self, **kwargs)

    def __getitem__(self, idx):
        return self.preprocess_sample(SyntheticHandcraftedDataset.__getitem__(self, idx))
