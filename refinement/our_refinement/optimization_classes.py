"""
Refinement optimization classes for DeepV.

This module contains classes to refactor the monolithic render_optimization_hard function
into more maintainable, testable components.
"""

import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from refinement.our_refinement.utils.lines_refinement_functions import (
    NonanAdam,
    collapse_redundant_lines,
    constrain_parameters,
    dtype,
    h,
    mean_field_energy_lines,
    mean_vector_field_energy_lines,
    padding,
    reinit_excess_lines,
    render_lines_with_type,
    size_energy,
    snap_lines,
    w,
)
from util_files.metrics.iou import calc_iou__vect_image
from util_files.structured_logging import StructuredLogger

# Load refinement config
try:
    from omegaconf import OmegaConf
    refinement_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../../config/refinement/default.yaml'))
except:
    # Fallback
    class FallbackConfig:
        render_res = 64
        padding = 3
        learning_rate_lines = 0.1
        optimization_iters_n = 100
        batch_size = 300
        line_visibility_threshold = 0.5
        initial_probability = 2**-8
        width_min_threshold = 0.25
        iou_stop_threshold = 0.98
        probability_clip_min = 0.3
        reinit_interval = 20
        logging_interval = 20
        snap_interval = 20
    refinement_config = FallbackConfig()


class LineOptimizationState:
    """Manages the optimization state for line parameters during refinement."""

    def __init__(self, lines_batch: torch.Tensor, device: torch.device):
        """
        Initialize optimization state from line parameters.

        Args:
            lines_batch: Initial line parameters (batch, N, 5) [x1, y1, x2, y2, width]
            device: Torch device for computation
        """
        self.device = device

        # Convert to canonical parameters
        x1, y1, x2, y2, width = lines_batch[..., 0], lines_batch[..., 1], lines_batch[..., 2], lines_batch[..., 3], lines_batch[..., 4]

        X = x2 - x1
        Y = y2 - y1
        length = torch.sqrt(X**2 + Y**2)
        theta = torch.atan2(Y, X)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Store canonical parameters
        self.cx = cx.clone().detach().requires_grad_(True).to(device)
        self.cy = cy.clone().detach().requires_grad_(True).to(device)
        self.theta = theta.clone().detach().requires_grad_(True).to(device)
        self.length = length.clone().detach().requires_grad_(True).to(device)
        self.width = width.clone().detach().requires_grad_(True).to(device)

        # Initialize optimizers
        self.pos_optimizer = NonanAdam([self.cx, self.cy, self.theta], lr=refinement_config.learning_rate_lines)
        self.size_optimizer = NonanAdam([self.length, self.width], lr=refinement_config.learning_rate_lines)

        # Initialize gradients
        (self.cx + self.cy + self.theta + self.length + self.width).reshape(-1)[0].backward()
        self.pos_optimizer.zero_grad()
        self.pos_optimizer.step()
        self.size_optimizer.zero_grad()
        self.size_optimizer.step()

    def get_lines_batch(self) -> torch.Tensor:
        """Convert canonical parameters back to line format [x1, y1, x2, y2, width]."""
        x1 = self.cx - self.length * torch.cos(self.theta) / 2
        y1 = self.cy - self.length * torch.sin(self.theta) / 2
        x2 = self.cx + self.length * torch.cos(self.theta) / 2
        y2 = self.cy + self.length * torch.sin(self.theta) / 2
        return torch.stack([x1, y1, x2, y2, self.width], -1)

    def apply_constraints(self):
        """Apply parameter constraints to keep lines within bounds."""
        constrain_parameters(
            self.cx, self.cy, self.theta, self.length, self.width,
            canvas_width=w, canvas_height=h, size_optimizer=self.size_optimizer
        )

    def snap_lines(self):
        """Snap lines that have coinciding ends, close orientations and close widths."""
        snap_lines(self.cx, self.cy, self.theta, self.length, self.width,
                  pos_optimizer=self.pos_optimizer, size_optimizer=self.size_optimizer)


class BatchProcessor:
    """Handles batch processing for refinement optimization."""

    def __init__(self, patches_rgb: torch.Tensor, patches_vector: torch.Tensor,
                 batch_size: int = refinement_config.batch_size):
        """
        Initialize batch processor.

        Args:
            patches_rgb: RGB patches (batch, 1, H, W)
            patches_vector: Initial vector predictions (batch, N, 5)
            batch_size: Size of batches to process
        """
        self.patches_rgb = patches_rgb
        self.patches_vector = patches_vector
        self.batch_size = batch_size

    def get_valid_batches(self) -> List[int]:
        """Get indices of batches that contain non-empty patches."""
        valid_batches = []
        for it in range(0, self.patches_vector.shape[0], self.batch_size):
            batch_end = min(it + self.batch_size, self.patches_vector.shape[0])
            batch_indices = list(range(it, batch_end))

            # Check if any patch in batch has content
            batch_rgb = self.patches_rgb[batch_indices]
            if torch.mean(batch_rgb) != 0:
                valid_batches.extend(batch_indices)

        return valid_batches

    def get_batch_data(self, batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RGB and vector data for a batch."""
        rasters_batch = self.patches_rgb[batch_indices].type(dtype)
        initial_vector = self.patches_vector[batch_indices].cpu()
        return rasters_batch, initial_vector


class OptimizationLoop:
    """Manages the main optimization loop for line refinement."""

    def __init__(self, rasters_batch: torch.Tensor, initial_vector: np.ndarray,
                 device: torch.device, rendering_type: str, logger: StructuredLogger):
        """
        Initialize optimization loop.

        Args:
            rasters_batch: Target raster images (batch, H, W)
            initial_vector: Initial line parameters (batch, N, 5)
            device: Torch device
            rendering_type: Type of rendering ('hard' or 'bezier_splatting')
            logger: Logger instance
        """
        self.rasters_batch = torch.nn.functional.pad(rasters_batch, [padding, padding, padding, padding]).to(device)
        self.device = device
        self.rendering_type = rendering_type
        self.logger = logger

        # Initialize optimization state
        lines_batch = torch.from_numpy(initial_vector).type(dtype).to(device)
        self.opt_state = LineOptimizationState(lines_batch, device)

        # Initialize tracking variables
        self.patches_to_optimize = np.full(lines_batch.shape[0], True, bool)
        self.vector_rendering = render_lines_with_type(lines_batch.detach(), rendering_type)

        # Final parameter storage
        self.cx_final = torch.empty_like(self.opt_state.cx)
        self.cy_final = torch.empty_like(self.opt_state.cy)
        self.theta_final = torch.empty_like(self.opt_state.theta)
        self.length_final = torch.empty_like(self.opt_state.length)
        self.width_final = torch.empty_like(self.opt_state.width)
        self.lines_batch_final = torch.empty_like(lines_batch)

    def optimize_step(self, iteration: int) -> bool:
        """
        Perform one optimization step.

        Args:
            iteration: Current iteration number

        Returns:
            True if optimization should continue, False if stopped
        """
        try:
            # 1. Reinitialize excess predictions
            if iteration % refinement_config.reinit_interval == 0:
                self._reinitialize_excess(iteration)

            # 2. Optimize position parameters
            self._optimize_positions()

            # 3. Optimize size parameters (left-fixed)
            self._optimize_sizes_left_fixed()

            # 4. Optimize size parameters (right-fixed)
            self._optimize_sizes_right_fixed()

            # 5. Snap lines
            if (iteration + 1) % refinement_config.snap_interval == 0:
                self.opt_state.snap_lines()

            return True

        except KeyboardInterrupt:
            return False

    def _reinitialize_excess(self, iteration: int):
        """Reinitialize excess line predictions."""
        lines_batch = self.opt_state.get_lines_batch()
        lines_batch[..., -1][lines_batch[..., -1] < refinement_config.width_min_threshold] = 0

        self.vector_rendering[self.patches_to_optimize] = render_lines_with_type(
            lines_batch[self.patches_to_optimize].detach(), self.rendering_type
        )

        im = self.rasters_batch[self.patches_to_optimize].clone()
        im.masked_fill_(self.vector_rendering[self.patches_to_optimize] > 0, 0)

        reinit_excess_lines(
            self.opt_state.cx, self.opt_state.cy, self.opt_state.width,
            self.opt_state.length, im.reshape(im.shape[0], -1),
            patches_to_consider=self.patches_to_optimize
        )

    def _optimize_positions(self):
        """Optimize mean field energy w.r.t position parameters."""
        lines_batch = self.opt_state.get_lines_batch()

        mean_field_energy = mean_field_energy_lines(
            lines_batch[self.patches_to_optimize], self.rasters_batch[self.patches_to_optimize]
        )

        self.opt_state.pos_optimizer.zero_grad()
        mean_field_energy.backward()
        self.opt_state.pos_optimizer.step()
        self.opt_state.apply_constraints()

    def _optimize_sizes_left_fixed(self):
        """Optimize size parameters with left points and orientations fixed."""
        # Fix left points
        x1 = self.opt_state.cx.data - self.opt_state.length.data * torch.cos(self.opt_state.theta.data) / 2
        y1 = self.opt_state.cy.data - self.opt_state.length.data * torch.sin(self.opt_state.theta.data) / 2

        # Update right points based on current length
        x2 = x1 + self.opt_state.length * torch.cos(self.opt_state.theta.data)
        y2 = y1 + self.opt_state.length * torch.sin(self.opt_state.theta.data)

        lines_batch = torch.stack([x1, y1, x2, y2, self.opt_state.width], -1)

        excess_energy = size_energy(lines_batch[self.patches_to_optimize], self.rasters_batch[self.patches_to_optimize])
        collinearity_energy = mean_vector_field_energy_lines(lines_batch[self.patches_to_optimize])

        self.opt_state.size_optimizer.zero_grad()
        (excess_energy + collinearity_energy).backward()
        self.opt_state.size_optimizer.step()

        # Update centers
        self.opt_state.cx.data[self.patches_to_optimize] = (
            x1.data[self.patches_to_optimize]
            + self.opt_state.length.data[self.patches_to_optimize] * torch.cos(self.opt_state.theta.data[self.patches_to_optimize]) / 2
        )
        self.opt_state.cy.data[self.patches_to_optimize] = (
            y1.data[self.patches_to_optimize]
            + self.opt_state.length.data[self.patches_to_optimize] * torch.sin(self.opt_state.theta.data[self.patches_to_optimize]) / 2
        )
        self.opt_state.apply_constraints()

    def _optimize_sizes_right_fixed(self):
        """Optimize size parameters with right points and orientations fixed."""
        # Fix right points
        x2 = self.opt_state.cx.data + self.opt_state.length.data * torch.cos(self.opt_state.theta.data) / 2
        y2 = self.opt_state.cy.data + self.opt_state.length.data * torch.sin(self.opt_state.theta.data) / 2

        # Update left points based on current length
        x1 = x2 - self.opt_state.length * torch.cos(self.opt_state.theta.data)
        y1 = y2 - self.opt_state.length * torch.sin(self.opt_state.theta.data)

        lines_batch = torch.stack([x1, y1, x2, y2, self.opt_state.width], -1)

        excess_energy = size_energy(lines_batch[self.patches_to_optimize], self.rasters_batch[self.patches_to_optimize])
        collinearity_energy = mean_vector_field_energy_lines(lines_batch[self.patches_to_optimize])

        self.opt_state.size_optimizer.zero_grad()
        (excess_energy + collinearity_energy).backward()
        self.opt_state.size_optimizer.step()

        # Update centers
        self.opt_state.cx.data[self.patches_to_optimize] = (
            x2.data[self.patches_to_optimize]
            - self.opt_state.length.data[self.patches_to_optimize] * torch.cos(self.opt_state.theta.data[self.patches_to_optimize]) / 2
        )
        self.opt_state.cy.data[self.patches_to_optimize] = (
            y2.data[self.patches_to_optimize]
            - self.opt_state.length.data[self.patches_to_optimize] * torch.sin(self.opt_state.theta.data[self.patches_to_optimize]) / 2
        )
        self.opt_state.apply_constraints()

    def log_progress(self, iteration: int, patches_rgb_im: np.ndarray,
                    take_batches: List[int], iou_mass: List, mass_for_iou_one: List):
        """Log progress and update tracking variables."""
        if (iteration % refinement_config.logging_interval == 0):
            # Record current parameters
            self.cx_final[self.patches_to_optimize] = self.opt_state.cx.data[self.patches_to_optimize]
            self.cy_final[self.patches_to_optimize] = self.opt_state.cy.data[self.patches_to_optimize]
            self.theta_final[self.patches_to_optimize] = self.opt_state.theta.data[self.patches_to_optimize]
            self.length_final[self.patches_to_optimize] = self.opt_state.length.data[self.patches_to_optimize]
            self.width_final[self.patches_to_optimize] = self.opt_state.width.data[self.patches_to_optimize]

            # Collapse invisible lines
            self.width_final[self.width_final < refinement_config.width_min_threshold] = 0

            # Collapse redundant lines
            collapse_redundant_lines(
                self.cx_final, self.cy_final, self.theta_final,
                self.length_final, self.width_final,
                patches_to_consider=self.patches_to_optimize
            )

            # Convert back to line format
            x1 = (
                self.cx_final.data[self.patches_to_optimize]
                - self.length_final.data[self.patches_to_optimize] * torch.cos(self.theta_final.data[self.patches_to_optimize]) / 2
            )
            y1 = (
                self.cy_final.data[self.patches_to_optimize]
                - self.length_final.data[self.patches_to_optimize] * torch.sin(self.theta_final.data[self.patches_to_optimize]) / 2
            )
            x2 = (
                self.cx_final.data[self.patches_to_optimize]
                + self.length_final.data[self.patches_to_optimize] * torch.cos(self.theta_final.data[self.patches_to_optimize]) / 2
            )
            y2 = (
                self.cy_final.data[self.patches_to_optimize]
                + self.length_final.data[self.patches_to_optimize] * torch.sin(self.theta_final.data[self.patches_to_optimize]) / 2
            )

            self.lines_batch_final[self.patches_to_optimize] = torch.stack(
                [x1, y1, x2, y2, self.width_final[self.patches_to_optimize]], -1
            )

            # Update rendering
            self.vector_rendering[self.patches_to_optimize] = render_lines_with_type(
                self.lines_batch_final[self.patches_to_optimize], self.rendering_type
            )

            # Calculate IOU
            iou_val = calc_iou__vect_image(
                self.lines_batch_final.data.cpu() / 64, patches_rgb_im[take_batches]
            )
            # Ensure it's a scalar by taking mean if it's an array
            if hasattr(iou_val, '__len__') and len(iou_val) > 1:
                iou_val = float(np.mean(iou_val))
            iou_mass.append(iou_val)
            mass_for_iou_one.append(self.lines_batch_final.cpu().data.detach().numpy())

    def get_final_result(self) -> torch.Tensor:
        """Get the final optimized line parameters."""
        return self.lines_batch_final