import os
import sys

import numpy as np
import pytest

# Ensure repo root is on sys.path when running this test file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from refinement.our_refinement.utils.lines_refinement_functions import get_random_line


def test_get_random_line():
    """Test that get_random_line generates valid line parameters."""
    h, w = 64, 64
    line = get_random_line(h, w)

    # Should return 5 parameters: x1, y1, x2, y2, width
    assert len(line) == 5
    assert isinstance(line, np.ndarray)
    assert line.dtype == np.float32

    # All coordinates should be within bounds
    x1, y1, x2, y2, width = line
    assert 0 <= x1 <= w
    assert 0 <= y1 <= h
    assert 0 <= x2 <= w
    assert 0 <= y2 <= h
    assert width >= 1  # width should be at least 1


def test_get_random_line_with_max_width():
    """Test get_random_line with custom max_width."""
    h, w = 100, 100
    max_width = 5.0
    line = get_random_line(h, w, max_width=max_width)

    x1, y1, x2, y2, width = line
    assert width <= max_width + 1  # +1 because function adds 1 to the random value


def test_raster_coordinates_creation():
    """Test that raster coordinates are created properly at module import."""
    # Import the module to trigger raster_coordinates creation
    from refinement.our_refinement.utils import lines_refinement_functions as lrf

    # Check that raster_coordinates exists and is a tensor
    assert hasattr(lrf, "raster_coordinates")
    assert isinstance(lrf.raster_coordinates, torch.Tensor)

    # Should be on the correct device (CPU if CUDA not available)
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert lrf.raster_coordinates.device == expected_device
