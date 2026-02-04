import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from merging.utils import merging_functions as mf


def test_clip_to_box_outside():
    # line completely outside box
    y = np.array([-10.0, -10.0, -5.0, -5.0, 0.1, 0.2])
    out = mf.clip_to_box(y, box_size=(64, 64))
    assert np.isnan(out).all()


def test_clip_to_box_partial():
    # line partially inside should clip to box bounds
    y = np.array([-10.0, 10.0, 20.0, 70.0, 0.5, 0.6])
    out = mf.clip_to_box(y, box_size=(64, 64))
    assert not np.isnan(out).any()
    # clipped coordinates lie within box
    assert 0 <= out[0] <= 64 and 0 <= out[1] <= 64 and 0 <= out[2] <= 64 and 0 <= out[3] <= 64


def test_assemble_preserves_input():
    patches_vector = [np.array([[0.0, 0.0, 1.0, 1.0, 0.3, 0.4]])]
    patches_offsets = [np.array([5, 10])]
    original = patches_vector[0].copy()
    assembled = mf._assemble_vector_patches(patches_vector, patches_offsets, x_indices=[0, 2], y_indices=[1, 3])
    # original should not be mutated
    assert np.allclose(patches_vector[0], original)
    # assembled should reflect offsets
    assert assembled[0][0][0] == original[0][0] + 10
    assert assembled[0][0][1] == original[0][1] + 5
