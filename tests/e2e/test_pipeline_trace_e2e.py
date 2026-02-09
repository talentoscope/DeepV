"""
End-to-end trace validation test.

Runs a minimal pipeline with tracing enabled and asserts that all expected
trace artifacts (provenance, metrics, primitives, iterations) are produced
and well-formed.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_trace_artifacts_exist(tmp_path):
    """Test that enabling --trace produces expected artifact files."""
    from analysis.tracing import Tracer

    # Create a tracer in temp directory
    image_id = "test_synthetic_001"
    tracer = Tracer(enabled=True, base_dir=str(tmp_path), image_id=image_id, seed=42, device="cpu")

    # Simulate per-stage exports
    # 1. Save patch
    patch = (np.random.rand(64, 64) * 255).astype(np.uint8)
    tracer.save_patch("p0", patch, offset=(0, 0))

    # 2. Save model output
    tracer.save_model_output("p0", {"vector": np.zeros((10, 5))})

    # 3. Save pre-refinement
    tracer.save_pre_refinement([{"indices": [0, 1, 2]}])

    # 4. Save per-iteration snapshot
    tracer.save_iteration(0, lines_batch=np.zeros((1, 10, 5)), renderings=np.zeros((1, 64, 64)))
    tracer.save_iteration(10, lines_batch=np.zeros((1, 10, 5)), renderings=np.zeros((1, 64, 64)))

    # 5. Save post-refinement
    tracer.save_post_refinement([{"refined_indices": [0, 1]}])

    # 6. Save primitive history (stacked iterations)
    history = np.random.randn(5, 1, 10, 5).astype(np.float32)  # 5 iterations, 1 patch, 10 prims
    tracer.save_primitive_history(history)

    # 7. Save merge trace
    tracer.save_merge_trace({"clusters": [{"cluster": [0, 1], "merged": True}]})

    # 8. Save provenance
    tracer.save_provenance({"final_count": 5})

    # 9. Save metrics
    tracer.save_metrics({"final_iou": 0.85, "refinement_iterations": 100})

    # Assert all files exist and are well-formed
    trace_dir = tmp_path / image_id

    # Determinism metadata
    assert (trace_dir / "determinism.json").exists()
    with open(trace_dir / "determinism.json") as f:
        det = json.load(f)
        assert det["seed"] == 42
        assert det["device"] == "cpu"

    # Patches
    assert (trace_dir / "patches" / "p0" / "patch.png").exists() or (
        trace_dir / "patches" / "p0" / "patch.npz"
    ).exists()
    assert (trace_dir / "patches" / "p0" / "model_output.npz").exists()

    # Pre/post refinement
    assert (trace_dir / "pre_refinement.json").exists()
    assert (trace_dir / "post_refinement.json").exists()

    # Iterations
    assert (trace_dir / "iterations" / "meta_0.json").exists()
    assert (trace_dir / "iterations" / "meta_10.json").exists()

    # History and merge traces
    assert (trace_dir / "primitive_history.npz").exists()
    assert (trace_dir / "merge_trace.json").exists()

    # Provenance and metrics
    assert (trace_dir / "provenance.json").exists()
    with open(trace_dir / "provenance.json") as f:
        prov = json.load(f)
        assert prov["final_count"] == 5

    assert (trace_dir / "metrics.json").exists()
    with open(trace_dir / "metrics.json") as f:
        metrics = json.load(f)
        assert metrics["final_iou"] == 0.85
        assert metrics["refinement_iterations"] == 100


def test_trace_disabled_no_artifacts(tmp_path):
    """Test that disabling trace prevents artifact creation."""
    from analysis.tracing import Tracer

    tracer = Tracer(enabled=False, base_dir=str(tmp_path), image_id="disabled_test")

    # Try to save; should be no-ops
    patch = (np.random.rand(64, 64) * 255).astype(np.uint8)
    tracer.save_patch("p0", patch)
    tracer.save_metrics({"test": 1.0})

    # Assert nothing was created
    trace_dir = tmp_path / "disabled_test"
    assert not trace_dir.exists()


def test_primitive_history_stacking(tmp_path):
    """Test that history snapshots stack correctly."""
    from analysis.tracing import Tracer

    tracer = Tracer(enabled=True, base_dir=str(tmp_path), image_id="history_test")

    # Create and save history
    history = np.random.randn(20, 3, 15, 5).astype(np.float32)  # 20 iters, 3 patches, 15 prims
    tracer.save_primitive_history(history)

    # Load and validate shape
    history_file = tmp_path / "history_test" / "primitive_history.npz"
    assert history_file.exists()

    loaded = np.load(history_file)
    assert loaded["history"].shape == (20, 3, 15, 5)
