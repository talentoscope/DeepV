#!/usr/bin/env python3
"""
FloorPlanCAD Dataset Split Creator

Creates stratified train/validation/test splits for FloorPlanCAD dataset while
maintaining raster-vector pairs. Organizes data into proper directory structure
for training and evaluation.

Features:
- Maintains SVG/PNG filename correspondence across splits
- Configurable split ratios (default 80/10/10)
- Preserves subdirectory organization if present
- Generates split files for use by training scripts
- Supports random shuffling with optional seed for reproducibility

Creates the following outputs:
- data/splits/floorplancad/train.txt
- data/splits/floorplancad/val.txt
- data/splits/floorplancad/test.txt
- Organized data directories (optional)

Usage:
    python scripts/create_floorplancad_splits.py --data_dir data/vector/floorplancad --output_dir data/splits/floorplancad
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


def get_svg_files_by_subdir(data_dir: Path) -> dict:
    """Get SVG files organized by subdirectory."""
    svg_files = defaultdict(list)

    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            svg_files[subdir.name] = list(subdir.glob("*.svg"))

    return dict(svg_files)


def create_split_files(
    svg_files_by_subdir: dict, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits from SVG files.

    Uses existing 'test' subdirectory as test set, splits train1/train2 for train/val.
    """
    # Use existing test directory as test set
    test_files = []
    for filename in svg_files_by_subdir.get("test", []):
        test_files.append(str(filename.relative_to(filename.parents[1])))

    # Combine train1 and train2 for train/val split
    train_val_files = []
    for subdir in ["train1", "train2"]:
        for filename in svg_files_by_subdir.get(subdir, []):
            train_val_files.append(str(filename.relative_to(filename.parents[1])))

    # Shuffle for random split
    random.seed(42)  # For reproducibility
    random.shuffle(train_val_files)

    # Calculate split sizes
    n_total = len(train_val_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Remaining go to train to ensure we use all data

    train_files = train_val_files[: n_train + (n_total - n_train - n_val)]  # Add remainder to train
    val_files = train_val_files[n_train : n_train + n_val]

    return train_files, val_files, test_files


def create_symlinks_or_copy(source_dir: Path, target_dir: Path, file_list: List[str], copy_files: bool = False):
    """Create symlinks or copy files to target directory maintaining structure."""
    target_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in file_list:
        source_file = source_dir / rel_path
        target_file = target_dir / rel_path

        # Create subdirectories if needed
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if copy_files:
            shutil.copy2(source_file, target_file)
        else:
            # Create relative symlink
            try:
                os.symlink(os.path.relpath(source_file, target_file.parent), target_file)
            except OSError:
                # Fallback to copy on Windows if symlinks don't work
                shutil.copy2(source_file, target_file)


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for FloorPlanCAD dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--vector_subdir", type=str, default="vector/floorplancad", help="Vector data subdirectory")
    parser.add_argument("--raster_subdir", type=str, default="raster/floorplancad", help="Raster data subdirectory")
    parser.add_argument(
        "--output_dir", type=str, default="data/splits/floorplancad", help="Output directory for splits"
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument(
        "--copy_files",
        action="store_true",
        default=True,
        help="Copy files instead of creating symlinks (default: True on Windows)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    vector_dir = data_dir / args.vector_subdir
    raster_dir = data_dir / args.raster_subdir
    output_dir = Path(args.output_dir)

    print(f"Processing data from: {vector_dir}")
    print(f"Raster data from: {raster_dir}")
    print(f"Output to: {output_dir}")

    # Get SVG files organized by subdirectory
    svg_files_by_subdir = get_svg_files_by_subdir(vector_dir)
    print(f"Found subdirectories: {list(svg_files_by_subdir.keys())}")

    for subdir, files in svg_files_by_subdir.items():
        print(f"  {subdir}: {len(files)} files")

    # Create splits
    train_files, val_files, test_files = create_split_files(svg_files_by_subdir, args.train_ratio, args.val_ratio)

    print("\nSplit sizes:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")

    # Create output directories
    train_vector_dir = output_dir / "train" / "vector"
    train_raster_dir = output_dir / "train" / "raster"
    val_vector_dir = output_dir / "val" / "vector"
    val_raster_dir = output_dir / "val" / "raster"
    test_vector_dir = output_dir / "test" / "vector"
    test_raster_dir = output_dir / "test" / "raster"

    # Create symlinks/copies for each split
    print("\nCreating train split...")
    create_symlinks_or_copy(vector_dir, train_vector_dir, train_files, args.copy_files)
    create_symlinks_or_copy(
        raster_dir, train_raster_dir, [f.replace(".svg", ".png") for f in train_files], args.copy_files
    )

    print("Creating val split...")
    create_symlinks_or_copy(vector_dir, val_vector_dir, val_files, args.copy_files)
    create_symlinks_or_copy(raster_dir, val_raster_dir, [f.replace(".svg", ".png") for f in val_files], args.copy_files)

    print("Creating test split...")
    create_symlinks_or_copy(vector_dir, test_vector_dir, test_files, args.copy_files)
    create_symlinks_or_copy(
        raster_dir, test_raster_dir, [f.replace(".svg", ".png") for f in test_files], args.copy_files
    )

    # Create file lists for reference
    (output_dir / "file_lists").mkdir(parents=True, exist_ok=True)

    with open(output_dir / "file_lists" / "train.txt", "w") as f:
        f.write("\n".join(train_files))

    with open(output_dir / "file_lists" / "val.txt", "w") as f:
        f.write("\n".join(val_files))

    with open(output_dir / "file_lists" / "test.txt", "w") as f:
        f.write("\n".join(test_files))

    print("\nDone! Created splits in:")
    print(f"  {output_dir}")
    print("\nFile lists saved to:")
    print(f"  {output_dir}/file_lists/")


if __name__ == "__main__":
    main()
