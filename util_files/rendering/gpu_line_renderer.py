"""
GPU-accelerated differentiable line renderer for DeepV.

Provides fast GPU-based rendering as an alternative to Cairo rendering.
Based on antialiased differentiable rendering techniques.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class GPULineRenderer:
    """
    GPU-accelerated differentiable line renderer.

    Renders lines using antialiased differentiable rendering techniques,
    providing significant speedup over CPU-based Cairo rendering.
    """

    def __init__(self, canvas_size: Tuple[int, int], supersampling: int = 2):
        """
        Initialize GPU line renderer.

        Args:
            canvas_size: (height, width) of output canvas
            supersampling: Supersampling factor for anti-aliasing
        """
        self.canvas_size = canvas_size
        self.supersampling = supersampling
        self.ss_canvas_size = (canvas_size[0] * supersampling, canvas_size[1] * supersampling)

        # Create coordinate grids (will be moved to GPU when needed)
        self._create_coordinate_grids()

    def _create_coordinate_grids(self):
        """Create pixel coordinate grids for rendering."""
        h, w = self.ss_canvas_size

        # Pixel centers (0.5 offset for center sampling)
        y_coords = torch.linspace(0.5, h - 0.5, h)
        x_coords = torch.linspace(0.5, w - 0.5, w)

        # Use indexing='ij' for proper meshgrid behavior
        self.grid_y, self.grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    def to_device(self, device: torch.device):
        """Move grids to specified device."""
        self.grid_y = self.grid_y.to(device)
        self.grid_x = self.grid_x.to(device)
        return self

    def render_lines(self, lines: torch.Tensor, canvas_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Render lines using GPU-accelerated differentiable rendering.

        Args:
            lines: (N, 5) tensor of lines [x1, y1, x2, y2, width]
                  or (N, 6) tensor [x1, y1, x2, y2, width, opacity]
            canvas_size: Optional canvas size override

        Returns:
            (H, W) rendered image tensor
        """
        if canvas_size is None:
            canvas_size = self.canvas_size

        device = lines.device
        self.to_device(device)

        # Extract line parameters
        start_points = lines[:, :2]  # (N, 2)
        end_points = lines[:, 2:4]   # (N, 2)
        widths = lines[:, 4]         # (N,)
        if lines.shape[1] > 5:
            opacity = lines[:, 5]    # (N,)
        else:
            opacity = torch.ones_like(widths)

        # Scale coordinates to supersampled canvas
        scale_y, scale_x = self.ss_canvas_size[0] / canvas_size[0], self.ss_canvas_size[1] / canvas_size[1]

        start_points = start_points * torch.tensor([scale_x, scale_y], device=device)
        end_points = end_points * torch.tensor([scale_x, scale_y], device=device)
        widths = widths * torch.sqrt(torch.tensor(scale_x * scale_y, device=device))  # Scale width proportionally

        # Render all lines at once using vectorized operations
        canvas = self._render_lines_batch(start_points, end_points, widths, opacity)

        # Canvas is already (H, W) in supersampled space, downsample to target size
        if canvas.shape != canvas_size:
            canvas = canvas.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            canvas = F.interpolate(canvas, size=canvas_size, mode='bilinear', align_corners=False)
            canvas = canvas.squeeze(0).squeeze(0)  # (H, W)

        return canvas

    def _render_lines_batch(self, start_points: torch.Tensor, end_points: torch.Tensor,
                           widths: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
        """
        Render multiple lines by compositing individually rendered lines.
        """
        canvas = torch.zeros(self.ss_canvas_size, device=start_points.device)

        for i in range(len(start_points)):
            line_canvas = self._render_single_line(
                start_points[i], end_points[i], widths[i], opacity[i]
            )
            canvas = torch.maximum(canvas, line_canvas)

        return canvas

    def _render_single_line(self, start: torch.Tensor, end: torch.Tensor,
                           width: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
        """
        Render a single line using differentiable antialiased rendering.
        """
        # Vector from start to end
        line_vec = end - start
        line_length = torch.norm(line_vec)

        if line_length < 1e-6:
            # Degenerate line, render as point
            return self._render_point(start, width, opacity)

        # Unit vector along line
        line_dir = line_vec / line_length

        # Perpendicular vector
        perp_dir = torch.tensor([-line_dir[1], line_dir[0]], device=start.device)

        # Distance from each pixel to the line
        pixel_vecs = torch.stack([self.grid_x - start[0], self.grid_y - start[1]], dim=-1)  # (H, W, 2)

        # Project onto line direction and perpendicular
        proj_line = torch.sum(pixel_vecs * line_dir, dim=-1)  # Distance along line
        proj_perp = torch.sum(pixel_vecs * perp_dir, dim=-1)  # Distance perpendicular to line

        # Clamp projection along line to valid range
        proj_line = torch.clamp(proj_line, 0, line_length)

        # Distance to closest point on line segment
        closest_x = start[0] + proj_line * line_dir[0]
        closest_y = start[1] + proj_line * line_dir[1]
        closest_points = torch.stack([closest_x, closest_y], dim=-1)  # (H, W, 2)

        distances = torch.norm(pixel_vecs - closest_points, dim=-1)

        # Create antialiased line using smooth falloff
        half_width = width / 2
        # Use smoothstep-like falloff for anti-aliasing
        falloff_width = min(width * 0.1, 1.0)  # Anti-aliasing zone

        # Distance from line edge
        edge_distance = half_width - distances

        # Smooth falloff
        alpha = torch.clamp(edge_distance / falloff_width, 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)  # Smoothstep

        # Only render within line segment bounds (with small extension for anti-aliasing)
        extension = falloff_width
        along_line_mask = (proj_line >= -extension) & (proj_line <= line_length + extension)

        alpha = alpha * along_line_mask.float()

        return alpha * opacity


def _render_point(self, center: torch.Tensor, size: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    """
    Render a point (degenerate line) as a circle.
    """
    distances = torch.sqrt((self.grid_x - center[0])**2 + (self.grid_y - center[1])**2)
    half_size = size / 2
    falloff_width = min(size * 0.1, 1.0)

    edge_distance = half_size - distances
    alpha = torch.clamp(edge_distance / falloff_width, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)  # Smoothstep

    return alpha * opacity


# Add the method to the class
GPULineRenderer._render_point = _render_point


def render_lines_gpu(lines: torch.Tensor, canvas_size: Tuple[int, int] = (64, 64),
                    supersampling: int = 2) -> torch.Tensor:
    """
    Convenience function for GPU-accelerated line rendering.

    Args:
        lines: (N, 5) or (N, 6) tensor of lines
        canvas_size: Output canvas size
        supersampling: Supersampling factor

    Returns:
        Rendered image tensor
    """
    renderer = GPULineRenderer(canvas_size, supersampling)
    return renderer.render_lines(lines, canvas_size)


# Performance comparison utility
def benchmark_renderers():
    """Benchmark different rendering approaches."""
    import time
    import numpy as np

    # Create test data
    torch.manual_seed(42)
    lines = torch.randn(20, 6) * 32 + 32  # 20 random lines in 64x64 space
    lines[:, 4] = torch.abs(lines[:, 4]) * 2 + 1  # Positive widths 1-5
    lines[:, 5] = torch.clamp(lines[:, 5], 0.5, 1.0)  # Opacity 0.5-1.0

    print("Benchmarking rendering approaches...")
    print("=" * 50)

    # Benchmark GPU renderer
    if torch.cuda.is_available():
        lines_gpu = lines.cuda()
        renderer = GPULineRenderer((64, 64))

        # Warmup
        for _ in range(5):
            _ = renderer.render_lines(lines_gpu)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            result_gpu = renderer.render_lines(lines_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 100
        print(".4f")

    # Benchmark CPU Cairo renderer (if available)
    try:
        from util_files.data.graphics_primitives import PT_LINE
        from util_files.rendering.cairo import render

        lines_np = lines.numpy()
        lines_data = lines_np[:, :4]  # x1,y1,x2,y2
        widths = lines_np[:, 4:5]
        line_dict = {PT_LINE: np.concatenate((lines_data, widths), axis=1)}

        # Warmup
        for _ in range(5):
            _ = render(line_dict, (64, 64), data_representation="vahe")

        # Benchmark
        start = time.time()
        for _ in range(100):
            result_cpu = render(line_dict, (64, 64), data_representation="vahe")
        cpu_time = (time.time() - start) / 100
        print(".4f")

        if torch.cuda.is_available():
            speedup = cpu_time / gpu_time
            print(".1f")

    except ImportError:
        print("Cairo renderer not available for comparison")


if __name__ == "__main__":
    benchmark_renderers()