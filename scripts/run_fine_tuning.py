#!/usr/bin/env python3
"""
Run Fine-tuning Script

Python equivalent of run_fine_tuning.sh for fine-tuning the cleaning UNet model.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch


def main():
    """Run the fine-tuning with predefined parameters."""
    # Enforce GPU usage
    if not torch.cuda.is_available():
        print("Error: GPU is required for DeepV fine-tuning but CUDA is not available on this machine.")
        return 1

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Get the script directory and navigate to cleaning
    script_dir = Path(__file__).parent
    cleaning_dir = script_dir.parent / "cleaning" / "scripts"

    # Change to cleaning directory
    os.chdir(cleaning_dir)

    # Run the fine-tuning command
    cmd = [
        sys.executable,
        "main_cleaning.py",
        "--model",
        "UNET",
        "--n_epochs",
        "30",
        "--datadir",
        "/dataset/Cleaning/Real/train",
        "--valdatadir",
        "/dataset/Cleaning/Real/val",
        "--batch_size",
        "4",
        "--name",
        "UNET_test",
    ]

    print("Running fine-tuning...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {cleaning_dir}")

    try:
        result = subprocess.run(cmd, check=True)
        print("Fine-tuning completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Fine-tuning failed with return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("Fine-tuning interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
