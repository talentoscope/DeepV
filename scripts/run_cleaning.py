#!/usr/bin/env python3
"""
DeepV Cleaning Model Training Runner

Runs training for the cleaning UNet model with predefined parameters.
Python equivalent of the shell script for cross-platform compatibility.

Features:
- Predefined training parameters for cleaning model
- Automatic directory navigation to cleaning module
- GPU device configuration
- Error handling and status reporting

Trains the cleaning model for improved artifact removal and image preprocessing.

Usage:
    python scripts/run_cleaning.py
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run the cleaning training with predefined parameters."""
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Get the script directory and navigate to cleaning
    script_dir = Path(__file__).parent
    cleaning_dir = script_dir.parent / "cleaning" / "scripts"

    # Change to cleaning directory
    os.chdir(cleaning_dir)

    # Run the training command
    cmd = [
        sys.executable,
        "main_cleaning.py",
        "--model",
        "UNET",
        "--n_epochs",
        "30",
        "--datadir",
        "/dataset/Cleaning/Synthetic/train",
        "--valdatadir",
        "/dataset/Cleaning/Synthetic/val",
        "--batch_size",
        "4",
    ]

    print("Running cleaning training...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {cleaning_dir}")

    try:
        result = subprocess.run(cmd, check=True)
        print("Cleaning training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Cleaning training failed with return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
