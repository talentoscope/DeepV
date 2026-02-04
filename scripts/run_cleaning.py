#!/usr/bin/env python3
"""
Run Cleaning Training Script

Python equivalent of run_cleaning.sh for training the cleaning UNet model.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the cleaning training with predefined parameters."""
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Get the script directory and navigate to cleaning
    script_dir = Path(__file__).parent
    cleaning_dir = script_dir.parent / 'cleaning' / 'scripts'

    # Change to cleaning directory
    os.chdir(cleaning_dir)

    # Run the training command
    cmd = [
        sys.executable, 'main_cleaning.py',
        '--model', 'UNET',
        '--n_epochs', '30',
        '--datadir', '/dataset/Cleaning/Synthetic/train',
        '--valdatadir', '/dataset/Cleaning/Synthetic/val',
        '--batch_size', '4'
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