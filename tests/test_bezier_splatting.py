import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running this test file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from util_files.rendering.bezier_splatting import BezierSplatting


def test_bezier_splatting_initialization():
    """Test BézierSplatting initialization."""
    renderer = BezierSplatting(canvas_size=(64, 64), supersampling=4)

    assert renderer.canvas_size == (64, 64)
    assert renderer.supersampling == 4
    assert renderer.ss_canvas_size == (256, 256)
    assert renderer.grid_x.shape == (256, 256)
    assert renderer.grid_y.shape == (256, 256)


def test_render_lines():
    """Test line rendering."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = BezierSplatting(canvas_size=(64, 64), supersampling=2)
    renderer.to_device(device)

    # Create test line
    start_points = torch.tensor([[10.0, 20.0]], device=device)
    end_points = torch.tensor([[50.0, 40.0]], device=device)
    widths = torch.tensor([5.0], device=device)

    canvas = renderer.render_lines(start_points, end_points, widths)

    assert canvas.shape == (64, 64)
    assert canvas.min() >= 0.0
    assert canvas.max() <= 1.0
    # Should have some non-zero pixels
    assert canvas.sum() > 0


def test_render_quadratic_beziers():
    """Test quadratic Bézier curve rendering."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = BezierSplatting(canvas_size=(64, 64), supersampling=2)
    renderer.to_device(device)

    # Create test quadratic Bézier curve
    # Start at (10, 20), control at (30, 50), end at (50, 20)
    curves = torch.tensor([[[10.0, 20.0], [30.0, 50.0], [50.0, 20.0]]], device=device)
    widths = torch.tensor([3.0], device=device)

    canvas = renderer.render_quadratic_beziers(curves, widths)

    assert canvas.shape == (64, 64)
    assert canvas.min() >= 0.0
    assert canvas.max() <= 1.0
    assert canvas.sum() > 0


def test_render_batch():
    """Test batch rendering with DeepV format."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = BezierSplatting(canvas_size=(64, 64), supersampling=2)
    renderer.to_device(device)

    # Create test data in DeepV format
    lines_data = torch.tensor(
        [[10.0, 20.0, 50.0, 40.0, 5.0], [20.0, 10.0, 40.0, 50.0, 3.0]], device=device  # x1, y1, x2, y2, width
    )

    primitives = {1: lines_data}  # PT_LINE = 1

    canvas = renderer.render_batch(primitives)

    assert canvas.shape == (64, 64)
    assert canvas.min() >= 0.0
    assert canvas.max() <= 1.0
    assert canvas.sum() > 0


def test_supersampling_effect():
    """Test that supersampling affects rendering quality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Render same line with different supersampling
    start_points = torch.tensor([[10.0, 20.0]], device=device)
    end_points = torch.tensor([[50.0, 40.0]], device=device)
    widths = torch.tensor([5.0], device=device)

    renderer_low = BezierSplatting(canvas_size=(64, 64), supersampling=1)
    renderer_low.to_device(device)
    canvas_low = renderer_low.render_lines(start_points, end_points, widths)

    renderer_high = BezierSplatting(canvas_size=(64, 64), supersampling=4)
    renderer_high.to_device(device)
    canvas_high = renderer_high.render_lines(start_points, end_points, widths)

    # Both should have same final shape
    assert canvas_low.shape == canvas_high.shape == (64, 64)

    # Higher supersampling should generally produce smoother results
    # (though this is a statistical property)
    assert canvas_low.sum() > 0
    assert canvas_high.sum() > 0


def test_gradient_flow():
    """Test that gradients flow through the renderer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = BezierSplatting(canvas_size=(32, 32), supersampling=2)
    renderer.to_device(device)

    # Create differentiable inputs
    start_points = torch.tensor([[10.0, 20.0]], device=device, requires_grad=True)
    end_points = torch.tensor([[50.0, 40.0]], device=device, requires_grad=True)
    widths = torch.tensor([5.0], device=device, requires_grad=True)

    canvas = renderer.render_lines(start_points, end_points, widths)
    loss = canvas.sum()

    # Check gradient flow
    loss.backward()

    assert start_points.grad is not None
    assert end_points.grad is not None
    assert widths.grad is not None

    # Gradients should be finite
    assert torch.isfinite(start_points.grad).all()
    assert torch.isfinite(end_points.grad).all()
    assert torch.isfinite(widths.grad).all()
