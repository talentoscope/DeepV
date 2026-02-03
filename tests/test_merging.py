import numpy as np
import pytest

from merging.utils import merging_functions as mf


def test_merge_close_lines_basic():
    # Two overlapping collinear lines should produce a merged bounding line
    lines = np.array([
        [0.0, 0.0, 10.0, 0.0, 1.0, 1.0],
        [9.0, 0.0, 20.0, 0.0, 1.0, 1.0],
    ])

    merged = mf.merge_close_lines(lines)
    assert merged.shape == (4,) or merged.shape[0] == 4
    # merged bbox should span at least the min and max x of the inputs
    assert merged[0] <= 0.0 + 1e-6
    assert merged[2] >= 20.0 - 1e-6


def test_merge_with_graph_reduces_or_keeps_count():
    # Skip if NetworkX not available
    pytest.importorskip("networkx")

    lines = np.array([
        [0.0, 0.0, 10.0, 0.0, 1.0, 1.0],
        [9.0, 0.0, 20.0, 0.0, 1.0, 1.0],
        [100.0, 100.0, 110.0, 100.0, 1.0, 1.0],
    ])
    widths = np.array([1.0, 1.0, 1.0])

    merged = mf.merge_with_graph(lines, widths=widths, max_dist=5, max_angle=15)

    # merged should be a list of primitives and not more than input lines
    assert isinstance(merged, list)
    assert len(merged) <= len(lines)
