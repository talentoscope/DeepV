#!/usr/bin/env python3
"""
Create train/val/test split for FloorPlanCAD dataset.
Ignores folder structure and splits based on filenames only.
"""

import os
import random
from pathlib import Path

def create_train_val_test_split():
    """Create 80/10/10 train/val/test split from FloorPlanCAD filenames."""

    # Read all filenames
    with open('floorplancad_files.txt', 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    print(f"Total files: {len(filenames)}")

    # Shuffle the filenames for random split
    random.seed(42)  # For reproducibility
    random.shuffle(filenames)

    # Calculate split sizes
    total_files = len(filenames)
    train_size = int(0.8 * total_files)
    val_size = int(0.1 * total_files)
    test_size = total_files - train_size - val_size

    # Create splits
    train_files = filenames[:train_size]
    val_files = filenames[train_size:train_size + val_size]
    test_files = filenames[train_size + val_size:]

    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")

    # Write split files
    splits_dir = Path("data/splits/floorplancad")
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write_split_file(split_name, files):
        split_file = splits_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for filename in sorted(files):  # Sort for consistency
                f.write(f"{filename}\n")
        print(f"Created {split_file}")

    write_split_file("train", train_files)
    write_split_file("val", val_files)
    write_split_file("test", test_files)

    # Also create a combined file with all filenames for reference
    all_file = splits_dir / "all.txt"
    with open(all_file, 'w') as f:
        for filename in sorted(filenames):
            f.write(f"{filename}\n")
    print(f"Created {all_file}")

if __name__ == "__main__":
    create_train_val_test_split()