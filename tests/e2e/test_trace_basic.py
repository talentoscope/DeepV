import os
from pathlib import Path

import numpy as np

from analysis.tracing import Tracer


def test_tracer_writes_patch_and_model_output(tmp_path):
    base = tmp_path / "traces"
    image_id = "test_img"
    tracer = Tracer(enabled=True, base_dir=str(base), image_id=image_id)

    # create a dummy patch (64x64 grayscale)
    patch = (np.random.rand(64, 64) * 255).astype(np.uint8)
    tracer.save_patch("p0", patch, offset=(0, 0))

    # create a dummy model output
    model_out = {"vector": np.zeros((10, 4))}
    tracer.save_model_output("p0", model_out)

    # save pre and post refinement lists
    tracer.save_pre_refinement([{"patch_id": 0, "vector": [0, 0, 1, 1]}])
    tracer.save_post_refinement([{"primitive_idx": 0, "vector": [0, 0, 1, 1]}])

    # assert files exist
    pdir = base / image_id / "patches" / "p0"
    assert (pdir / "patch.png").exists() or (pdir / "patch.npz").exists()
    assert (pdir / "model_output.npz").exists()
    assert (base / image_id / "pre_refinement.json").exists()
    assert (base / image_id / "post_refinement.json").exists()
