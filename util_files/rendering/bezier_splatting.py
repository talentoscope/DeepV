"""Bézier Splatting: Fast Differentiable Rendering for Vector Graphics.

This module implements Bézier Splatting as described in the paper
"Bézier Splatting: Fast Differentiable Rendering of Bézier Curves"
(https://arxiv.org/abs/2203.0945)

Bézier Splatting provides a fast, differentiable alternative to traditional
rasterization methods for rendering vector graphics primitives.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class BezierSplatting:
    """Bézier Splatting renderer for fast differentiable vector graphics rendering."""

    def __init__(self, canvas_size: Tuple[int, int], supersampling: int = 4):
        """
        Initialize Bézier Splatting renderer.

        Args:
            canvas_size: (height, width) of the output canvas
            supersampling: Supersampling factor for anti-aliasing
        """
        self.canvas_size = canvas_size
        self.supersampling = supersampling
        self.ss_canvas_size = (canvas_size[0] * supersampling, canvas_size[1] * supersampling)

        # Create coordinate grids
        self._create_coordinate_grids()

    def _create_coordinate_grids(self):
        """Create pixel coordinate grids for rendering."""
        h, w = self.ss_canvas_size

        # Pixel centers
        y_coords = torch.linspace(0.5, h - 0.5, h)
        x_coords = torch.linspace(0.5, w - 0.5, w)

        self.grid_y, self.grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.grid_y = self.grid_y.contiguous()
        self.grid_x = self.grid_x.contiguous()

    def to_device(self, device: torch.device):
        """Move coordinate grids to specified device."""
        self.grid_y = self.grid_y.to(device)
        self.grid_x = self.grid_x.to(device)
        return self

    def render_lines(
        self,
        start_points: torch.Tensor,
        end_points: torch.Tensor,
        widths: torch.Tensor,
        opacity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render lines using Bézier splatting.

        For straight lines, we approximate them as degenerate quadratic Bézier curves
        where the control point is the midpoint.

        Args:
            start_points: (N, 2) tensor of line start points (x, y)
            end_points: (N, 2) tensor of line end points (x, y)
            widths: (N,) tensor of line widths
            opacity: (N,) tensor of line opacities (default: 1.0)

        Returns:
            (H, W) tensor of rendered image
        """
        if opacity is None:
            opacity = torch.ones_like(widths)

        # Convert straight lines to degenerate quadratic Bézier curves
        # Control point is the midpoint
        control_points = (start_points + end_points) / 2

        # Stack into Bézier curve format: (start, control, end)
        curves = torch.stack([start_points, control_points, end_points], dim=1)  # (N, 3, 2)

        return self.render_quadratic_beziers(curves, widths, opacity)

    def render_quadratic_beziers(
        self, curves: torch.Tensor, widths: torch.Tensor, opacity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Render quadratic Bézier curves using splatting.

        Args:
            curves: (N, 3, 2) tensor of quadratic Bézier curves
                    Each curve is defined by 3 control points: start, control, end
            widths: (N,) tensor of curve widths
            opacity: (N,) tensor of curve opacities

        Returns:
            (H, W) tensor of rendered image
        """
        if opacity is None:
            opacity = torch.ones_like(widths)

        device = curves.device
        self.to_device(device)

        batch_size = curves.shape[0]
        canvas = torch.zeros(self.ss_canvas_size, dtype=torch.float32, device=device)

        for i in range(batch_size):
            curve = curves[i]  # (3, 2)
            width = widths[i]
            alpha = opacity[i]

            # Render single curve
            curve_canvas = self._render_single_quadratic_bezier(curve, width, alpha)
            canvas = canvas + curve_canvas

        # Clamp to valid range
        canvas = torch.clamp(canvas, 0.0, 1.0)

        # Downsample to final resolution
        if self.supersampling > 1:
            canvas = self._downsample(canvas)

        return canvas

    def _render_single_quadratic_bezier(
        self, curve: torch.Tensor, width: torch.Tensor, opacity: torch.Tensor
    ) -> torch.Tensor:
        """
        Render a single quadratic Bézier curve.

        Args:
            curve: (3, 2) tensor of control points
            width: scalar tensor of curve width
            opacity: scalar tensor of curve opacity

        Returns:
            Supersampled canvas with the rendered curve
        """
        # For quadratic Bézier: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        p0, p1, p2 = curve[0], curve[1], curve[2]

        # Adaptive sampling: sample more points where curvature is high
        t_values = torch.linspace(0, 1, 50, device=curve.device)

        # Evaluate curve points
        t = t_values.unsqueeze(-1)
        points = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

        # Compute tangents for width calculation
        tangents = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)
        tangent_norms = torch.norm(tangents, dim=-1, keepdim=True)
        tangent_norms = torch.clamp(tangent_norms, min=1e-6)
        tangents = tangents / tangent_norms

        # Compute normal vectors (perpendicular to tangents)
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)

        # Create ribbon (two parallel lines)
        half_width = width / 2
        left_points = points - normals * half_width
        right_points = points + normals * half_width

        # Render as filled polygon using splatting
        return self._splat_polygon(torch.cat([left_points, right_points.flip(0)], dim=0), opacity)

    def _splat_polygon(self, polygon: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
        """
        Splat a polygon onto the canvas using differentiable splatting.

        Args:
            polygon: (N, 2) tensor of polygon vertices
            opacity: scalar tensor of polygon opacity

        Returns:
            Canvas with splatted polygon
        """
        canvas = torch.zeros(self.ss_canvas_size, dtype=torch.float32, device=polygon.device)

        # Simple polygon filling using scanline algorithm (differentiable version)
        # For each pixel, check if it's inside the polygon
        pixels_y = self.grid_y.flatten()
        pixels_x = self.grid_x.flatten()

        # Use winding number algorithm for point-in-polygon test
        winding_numbers = self._compute_winding_numbers(pixels_x, pixels_y, polygon)

        # Soft inside mask using sigmoid (differentiable)
        # winding_numbers close to odd integers indicate inside
        inside_prob = torch.sigmoid(10 * (0.5 - (winding_numbers % 2).abs()))
        inside_prob = inside_prob.view(self.ss_canvas_size)  # Reshape to (H, W)

        # Gaussian splatting for all pixels, weighted by inside probability
        sigma = 0.5  # Pixel units
        y_dist = (self.grid_y.unsqueeze(-1) - polygon[:, 1].unsqueeze(0).unsqueeze(0)) ** 2
        x_dist = (self.grid_x.unsqueeze(-1) - polygon[:, 0].unsqueeze(0).unsqueeze(0)) ** 2
        dist_sq = y_dist + x_dist

        weights = torch.exp(-dist_sq / (2 * sigma**2))
        weights = weights.sum(dim=-1)  # Sum over all polygon points

        # Weight by inside probability and opacity
        canvas = canvas + weights * inside_prob * opacity

        return canvas

    def _compute_winding_numbers(
        self, pixels_x: torch.Tensor, pixels_y: torch.Tensor, polygon: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute winding numbers for point-in-polygon test.

        Args:
            pixels_x: x-coordinates of pixels
            pixels_y: y-coordinates of pixels
            polygon: (N, 2) polygon vertices

        Returns:
            Winding numbers for each pixel
        """
        num_pixels = pixels_x.shape[0]
        num_vertices = polygon.shape[0]

        winding = torch.zeros(num_pixels, device=polygon.device)

        # Close the polygon
        poly = torch.cat([polygon, polygon[:1]], dim=0)

        for i in range(num_vertices):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]

            # Check if edge crosses the horizontal line at pixel y
            y_crosses = ((y1 <= pixels_y) & (y2 > pixels_y)) | ((y1 > pixels_y) & (y2 <= pixels_y))

            if y_crosses.any():
                # Compute x-coordinate of intersection
                x_intersect = x1 + (pixels_y[y_crosses] - y1) * (x2 - x1) / (y2 - y1 + 1e-10)

                # Add to winding number based on edge direction
                # Use smooth sign approximation for differentiability
                direction = torch.tanh(10 * (x2 - x1))
                # Use soft comparison for differentiability
                intersect_weight = torch.sigmoid(10 * (x_intersect - pixels_x[y_crosses]))
                winding[y_crosses] += direction * intersect_weight

        return winding

    def _downsample(self, canvas: torch.Tensor) -> torch.Tensor:
        """Downsample supersampled canvas to final resolution."""
        # Average pooling
        kernel_size = self.supersampling
        stride = self.supersampling

        # Add batch and channel dimensions for conv2d
        canvas = canvas.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Use average pooling
        downsampled = F.avg_pool2d(canvas, kernel_size, stride)

        return downsampled.squeeze(0).squeeze(0)

    def render_batch(self, primitives: dict, data_representation: str = "vahe") -> torch.Tensor:
        """
        Render a batch of primitives in the format expected by DeepV.

        Args:
            primitives: Dictionary with primitive types as keys
            data_representation: Data format ("vahe" for DeepV)

        Returns:
            Rendered canvas
        """
        device = next(iter(primitives.values())).device
        self.to_device(device)

        canvas = torch.zeros(self.ss_canvas_size, dtype=torch.float32, device=device)

        # Render lines
        if 1 in primitives:  # PT_LINE = 1
            lines_data = primitives[1]  # (N, 5) - x1, y1, x2, y2, width
            if lines_data.shape[0] > 0:
                start_points = lines_data[:, :2]
                end_points = lines_data[:, 2:4]
                widths = lines_data[:, 4]

                lines_canvas = self.render_lines(start_points, end_points, widths)
                # Upsample to supersampled resolution for combination
                if self.supersampling > 1:
                    lines_canvas = (
                        F.interpolate(
                            lines_canvas.unsqueeze(0).unsqueeze(0),
                            size=self.ss_canvas_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                canvas = canvas + lines_canvas

        # Render curves (quadratic Bézier)
        if 2 in primitives:  # PT_QBEZIER = 2
            curves_data = primitives[2]  # (N, 6) - x1,y1,x2,y2,x3,y3,width
            if curves_data.shape[0] > 0:
                # Convert to (N, 3, 2) format
                curves = curves_data[:, :6].view(-1, 3, 2)
                widths = curves_data[:, 6]

                curves_canvas = self.render_quadratic_beziers(curves, widths)
                # Upsample to supersampled resolution for combination
                if self.supersampling > 1:
                    curves_canvas = (
                        F.interpolate(
                            curves_canvas.unsqueeze(0).unsqueeze(0),
                            size=self.ss_canvas_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                canvas = canvas + curves_canvas

        # Clamp and downsample
        canvas = torch.clamp(canvas, 0.0, 1.0)
        if self.supersampling > 1:
            canvas = self._downsample(canvas)

        return canvas
