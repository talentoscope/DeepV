#!/usr/bin/env python3
"""
Regenerate FloorPlanCAD split files based on actual data structure.
"""

import os
import random
from pathlib import Path


def regenerate_splits():
    """Regenerate train/val/test splits based on actual FloorPlanCAD data."""

    data_dir = Path("data/vector/floorplancad")

    # Get files from each subdirectory
    train1_files = []
    train2_files = []
    test_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".svg"):
                rel_path = os.path.relpath(os.path.join(root, file), data_dir)
                if "train1" in rel_path:
                    train1_files.append(rel_path)
                elif "train2" in rel_path:
                    train2_files.append(rel_path)
                elif "test" in rel_path:
                    test_files.append(rel_path)

    print(f"Found {len(train1_files)} files in train1/")
    print(f"Found {len(train2_files)} files in train2/")
    print(f"Found {len(test_files)} files in test/")

    # Shuffle train2 for splitting into train/val
    random.seed(42)
    random.shuffle(train2_files)

    # Split train2: 80% for training, 20% for validation
    train2_split = int(0.8 * len(train2_files))
    train2_train = train2_files[:train2_split]
    train2_val = train2_files[train2_split:]

    # Combine for final splits
    train_files = train1_files + train2_train
    val_files = train2_val
    # Keep test as is

    print("Final splits:")
    print(f"Train: {len(train_files)} files (train1 + 80% of train2)")
    print(f"Val: {len(val_files)} files (20% of train2)")
    print(f"Test: {len(test_files)} files (test directory)")

    # Write split files
    splits_dir = Path("data/splits/floorplancad")
    splits_dir.mkdir(parents=True, exist_ok=True)

    def write_split_file(split_name, files):
        split_file = splits_dir / f"{split_name}.txt"
        with open(split_file, "w") as f:
            for filename in sorted(files):
                # Replace .svg with .png for the split file
                png_filename = filename.replace(".svg", ".png")
                f.write(f"{png_filename}\n")
        print(f"Created {split_file}")

    write_split_file("train", train_files)
    write_split_file("val", val_files)
    write_split_file("test", test_files)


if __name__ == "__main__":
    regenerate_splits()
