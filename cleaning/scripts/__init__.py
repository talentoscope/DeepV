"""Cleaning scripts for DeepV pipeline.

This module contains command-line scripts for data cleaning operations including:
- Main cleaning pipeline (main_cleaning.py)
- Synthetic data generation (generate_synthetic_data.py)
- Model fine-tuning (fine_tuning.py)
- Pipeline execution (run.py)

Example:
    python -m deepv.cleaning.scripts.main_cleaning --help
"""

# Scripts are meant to be run as modules, not imported
__all__: list[str] = []