import os
import sys
import time

import numpy as np

# Ensure repo root is on sys.path when running this test file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from merging.utils import merging_functions as mf


def test_benchmark_clip_to_box():
    """Benchmark clip_to_box function performance."""
    # Create test data - 1000 random lines
    np.random.seed(42)
    test_lines = np.random.rand(1000, 6) * 100  # random coordinates up to 100

    start_time = time.time()
    for _ in range(100):  # Run 100 times for better measurement
        for line in test_lines:
            mf.clip_to_box(line, box_size=(64, 64))
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_call = total_time / (100 * 1000)
    print(".6f")
    assert avg_time_per_call < 0.001  # Should be very fast (< 1ms per call)


def test_benchmark_assemble_vector_patches():
    """Benchmark _assemble_vector_patches function performance."""
    # Create test data - 10 patches with 50 primitives each
    np.random.seed(42)
    patches_vector = []
    patches_offsets = []

    for i in range(10):
        patch = np.random.rand(50, 6) * 64  # 50 primitives per patch
        patches_vector.append(patch)
        patches_offsets.append(np.array([i * 64, i * 64]))  # Different offsets

    start_time = time.time()
    for _ in range(100):  # Run 100 times
        mf._assemble_vector_patches(patches_vector, patches_offsets, x_indices=[0, 2], y_indices=[1, 3])
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_call = total_time / 100
    print(".6f")
    assert avg_time_per_call < 0.01  # Should be reasonably fast (< 10ms per call)


def test_benchmark_large_scale_assembly():
    """Benchmark with larger scale data similar to real usage."""
    # Simulate real-world scenario: 100 patches, 100 primitives each
    np.random.seed(42)
    patches_vector = []
    patches_offsets = []

    for i in range(100):
        patch = np.random.rand(100, 6) * 64
        patches_vector.append(patch)
        patches_offsets.append(np.array([i * 32, i * 32]))  # Overlapping patches

    start_time = time.time()
    assembled = mf._assemble_vector_patches(patches_vector, patches_offsets, x_indices=[0, 2], y_indices=[1, 3])
    end_time = time.time()

    total_time = end_time - start_time
    print(".6f")
    print(f"Assembled {assembled.shape[0]} patches with {assembled.shape[1]} primitives each")

    # Should complete in reasonable time and produce expected output shape
    assert total_time < 1.0  # Less than 1 second for large assembly
    assert assembled.shape[0] == 100  # Same number of patches
    assert assembled.shape[1] == 100  # Same number of primitives per patch
