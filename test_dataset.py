#!/usr/bin/env python3
"""
Test script to verify FloorPlanCADDataset SVG parsing and primitive extraction.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for local imports
sys.path.append(".")
sys.path.append("./scripts")

import torch
from train_floorplancad import FloorPlanCADDataset


def test_dataset_loading():
    """Test loading a few samples from the FloorPlanCAD dataset."""

    # Use the first train split file
    split_file = 'data/splits/floorplancad/train.txt'
    raster_dir = 'data/raster/floorplancad'
    vector_dir = 'data/vector/floorplancad'

    print("Creating FloorPlanCADDataset...")
    dataset = FloorPlanCADDataset(split_file, raster_dir, vector_dir)

    print(f"Dataset loaded with {len(dataset)} samples")

    # Test loading first few samples
    for i in range(min(5, len(dataset))):
        print(f"\nTesting sample {i}...")
        try:
            image, target = dataset[i]
            if hasattr(image, 'shape'):
                print(f"  Image shape: {image.shape}")
            else:
                print(f"  Image size: {image.size}")
            print(f"  Target shape: {target.shape}")
            print(f"  Target type: {target.dtype}")
            print(f"  Target range: [{target.min().item():.3f}, {target.max().item():.3f}]")

            # Check if target is not all zeros (indicating successful SVG parsing)
            non_zero_count = (target != 0).sum().item()
            print(f"  Non-zero elements: {non_zero_count}/{target.numel()}")

            if non_zero_count > 0:
                print("  ✓ Successfully parsed SVG and extracted primitives!")
            else:
                print("  ⚠ SVG parsing may have failed or produced empty primitives")

        except Exception as e:
            print(f"  ✗ Error loading sample {i}: {e}")

if __name__ == "__main__":
    test_dataset_loading()