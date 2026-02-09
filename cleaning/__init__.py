"""Cleaning module for DeepV vectorization pipeline.

This module provides data cleaning and preprocessing functionality for technical drawings.
It includes utilities for noise removal, artifact cleaning, and synthetic data generation.

Submodules:
    scripts: Command-line scripts for cleaning operations
    utils: Utility functions for data loading, loss functions, and synthetic data generation

Example:
    from deepv.cleaning import utils
    # Access cleaning utilities
"""

from . import scripts, utils

__all__ = ["scripts", "utils"]