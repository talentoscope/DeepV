#!/usr/bin/env python3
"""
Generate synthetic test datasets for DeepV benchmarking.

Creates simple vector graphics datasets in PNG+DXF format for testing.
"""

import os
import random
from pathlib import Path

def create_synthetic_dataset(output_dir: str = "synthetic_dataset", num_samples: int = 50):
    """
    Create synthetic dataset with PNG images and DXF ground truth.

    Args:
        output_dir: Output directory for the dataset
        num_samples: Number of samples to generate per split
    """
    print(f"Creating synthetic dataset in {output_dir}...")

    splits = ['train', 'val', 'test']
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(num_samples):
            # Create a simple placeholder image file
            image_filename = f"{i:04d}.png"
            image_path = os.path.join(split_dir, image_filename)

            # Create a minimal PNG-like file (just for testing)
            with open(image_path, 'wb') as f:
                # Write a minimal PNG header for compatibility
                f.write(b'\x89PNG\r\n\x1a\n')  # PNG signature
                f.write(b'dummy image data for testing')

            # Create corresponding DXF ground truth
            dxf_filename = f"{i:04d}.dxf"
            dxf_path = os.path.join(split_dir, dxf_filename)

            dxf_content = generate_simple_dxf()
            with open(dxf_path, 'w') as f:
                f.write(dxf_content)

    print(f"Created synthetic dataset with {num_samples} samples per split")

def generate_simple_dxf() -> str:
    """Generate a simple DXF file with basic entities."""
    dxf = "0\nSECTION\n2\nENTITIES\n"

    # Add a line
    dxf += "0\nLINE\n"
    dxf += "8\n0\n"  # Layer
    dxf += "10\n0.0\n20\n0.0\n30\n0.0\n"  # Start point
    dxf += "11\n100.0\n21\n100.0\n31\n0.0\n"  # End point

    # Add a circle
    dxf += "0\nCIRCLE\n"
    dxf += "8\n0\n"  # Layer
    dxf += "10\n50.0\n20\n50.0\n30\n0.0\n"  # Center
    dxf += "40\n25.0\n"  # Radius

    dxf += "0\nENDSEC\n0\nEOF\n"

    return dxf

def create_test_datasets(base_dir: str = "data"):
    """
    Create synthetic test datasets.

    Args:
        base_dir: Base directory for datasets
    """
    os.makedirs(base_dir, exist_ok=True)

    print("Creating synthetic test datasets for benchmarking...")

    # Create synthetic dataset
    create_synthetic_dataset(os.path.join(base_dir, "synthetic"), num_samples=20)

    print(f"\nSynthetic datasets created in {base_dir}")
    print("You can now test the benchmarking pipeline with:")
    print(f"python scripts/benchmark_pipeline.py --data-root {base_dir} --datasets synthetic")

if __name__ == "__main__":
    create_test_datasets()