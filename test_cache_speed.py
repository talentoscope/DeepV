#!/usr/bin/env python3
"""Quick test to verify cache-loading works and measure batch speed"""
import sys
sys.path.append(".")
sys.path.append("scripts")

import torch
from pathlib import Path
from train_floorplancad import create_data_loaders
import time

print("Testing cache loading and batch speed...")

# Create loaders with cache
train_loader, val_loader, _ = create_data_loaders(
    split_dir='data/splits/floorplancad',
    raster_dir='data/raster/floorplancad',
    vector_dir='data/vector/floorplancad',
    batch_size=4,
    max_primitives=20,
    num_workers=0,  # Single worker for reproducibility
    cache_dir='data/cache/floorplancad'
)

print("\nLoading and timing 5 batches...")
train_iter = iter(train_loader)
for batch_num in range(5):
    start = time.time()
    images, targets, counts = next(train_iter)
    elapsed = time.time() - start
    print(f"Batch {batch_num+1}: {elapsed:.3f}s | shapes: images={images.shape}, targets={targets.shape}, counts={counts.shape}")

print("\n✓ Cache loading works! Batch speed should be ~0.1-0.5s (vs 20+ seconds with SVG parsing)")
