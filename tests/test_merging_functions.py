import os
import sys

import numpy as np

# Ensure repo root is on sys.path when running this test file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from merging.utils import merging_functions as mf


def test_clip_to_box_inside():
    # line fully inside 64x64 box
    y_pred = np.array([10.0, 10.0, 50.0, 50.0, 1.0, 0.9])
    out = mf.clip_to_box(y_pred, box_size=(64, 64))
    assert not np.isnan(out).any()
    # coordinates should remain unchanged when inside box
    assert out[0] == 10.0 and out[1] == 10.0 and out[2] == 50.0 and out[3] == 50.0


def test_clip_to_box_edge_cases():
    """Test edge cases for clip_to_box."""
    # Line exactly on boundaries
    y_pred = np.array([0.0, 0.0, 64.0, 64.0, 1.0, 0.9])
    out = mf.clip_to_box(y_pred, box_size=(64, 64))
    assert not np.isnan(out).any()
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 64.0 and out[3] == 64.0

    # Zero width line
    y_pred = np.array([32.0, 32.0, 32.0, 32.0, 1.0, 0.9])
    out = mf.clip_to_box(y_pred, box_size=(64, 64))
    assert not np.isnan(out).any()

    # Very thin line
    y_pred = np.array([32.0, 32.0, 32.1, 32.1, 0.01, 0.9])
    out = mf.clip_to_box(y_pred, box_size=(64, 64))
    assert not np.isnan(out).any()


def test_assemble_vector_patches_no_mutation():
    patches_vector = [np.array([[1.0, 2.0, 3.0, 4.0, 0.5, 0.6]])]
    patches_offsets = [np.array([10, 20])]
    original = patches_vector[0].copy()
    assembled = mf._assemble_vector_patches(patches_vector, patches_offsets, x_indices=[0, 2], y_indices=[1, 3])
    assert assembled.shape[0] == 1
    # original patch should not be mutated
    assert np.allclose(patches_vector[0], original)
    # assembled coordinates should be offset by (y_offset,x_offset) -> (10,20)
    assert assembled[0][0][0] == original[0][0] + 20
    assert assembled[0][0][1] == original[0][1] + 10


def test_assemble_vector_patches_multiple_patches():
    """Test assembling multiple patches."""
    patches_vector = [np.array([[1.0, 2.0, 3.0, 4.0, 0.5, 0.6]]), np.array([[10.0, 20.0, 30.0, 40.0, 1.5, 1.6]])]
    patches_offsets = [np.array([0, 0]), np.array([100, 200])]
    assembled = mf._assemble_vector_patches(patches_vector, patches_offsets, x_indices=[0, 2], y_indices=[1, 3])

    # Should have 2 assembled patches
    assert assembled.shape[0] == 2

    # First patch should be offset by (0,0)
    assert assembled[0][0][0] == 1.0 + 0  # x1 + x_offset
    assert assembled[0][0][1] == 2.0 + 0  # y1 + y_offset

    # Second patch should be offset by (200,100) - note the order is (y_offset, x_offset)
    assert assembled[1][0][0] == 10.0 + 200  # x1 + x_offset
    assert assembled[1][0][1] == 20.0 + 100  # y1 + y_offset


def test_assemble_vector_patches_empty():
    """Test assembling with empty patches."""
    patches_vector = []
    patches_offsets = []
    assembled = mf._assemble_vector_patches(patches_vector, patches_offsets, x_indices=[0, 2], y_indices=[1, 3])
    assert assembled.shape[0] == 0
