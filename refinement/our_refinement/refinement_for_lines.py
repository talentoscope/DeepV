import argparse
import logging
import os
import signal
from typing import Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from refinement.our_refinement.optimization_classes import BatchProcessor, OptimizationLoop
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
from util_files.structured_logging import get_pipeline_logger
from util_files.metrics.iou import calc_iou__vect_image
from util_files.structured_logging import get_pipeline_logger

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


def register_sigint_flag(flag_list: List[bool]) -> None:
    """Register a SIGINT handler that sets flag_list[0] = True.

    `flag_list` should be a mutable sequence (e.g., `[False]`).
    This avoids defining inner functions repeatedly and centralizes signal handling.
    """

    def _handler(*_args):
        flag_list[0] = True

    signal.signal(signal.SIGINT, _handler)


def render_optimization_hard(
    patches_rgb: np.ndarray,
    patches_vector: torch.Tensor,
    device: torch.device,
    options: Any,
    name: str
) -> torch.Tensor:
    """
    Perform differentiable optimization to refine vector primitives for better raster matching.

    This function implements the core refinement algorithm for line primitives, using mean-field
    energy minimization with position and size parameter optimization. It processes patches in
    batches to handle large datasets efficiently.

    Args:
        patches_rgb: RGB patches as numpy arrays (batch, 3, H, W), values 0-255.
            Will be normalized to 0-1 internally.
        patches_vector: Initial vector predictions (batch, N, 5) where each vector is
            [x1, y1, x2, y2, width] for lines.
        device: Torch device for computation (cuda/cpu).
        options: Configuration object with attributes like rendering_type, diff_render_it.
        name: String identifier for logging and output file naming.

    Returns:
        Refined vector patches (batch, N, 6) with added probability channel.

    Notes:
        - Uses Bézier splatting for differentiable rendering if options.rendering_type == 'bezier_splatting'.
        - Configurable parameters loaded from refinement/default.yaml (learning rates, thresholds, intervals).
        - Optimization iterates over position and size parameters alternately.
        - Saves IOU metrics and refined vectors to numpy files in options.output_dir/arrays/.
    """
    # Initialize structured logger
    logger = get_pipeline_logger("refinement.lines")

    with logger.timing("refinement_optimization", logging.INFO):
        logger.log_pipeline_step("refinement", "started", component="lines", dataset=name)

        # Initialize data structures
        patches_rgb_im = np.copy(patches_rgb)
        patches_vector = torch.tensor(patches_vector)
        y_pred_rend = torch.zeros((patches_vector.shape[0], patches_vector.shape[1], patches_vector.shape[2] - 1))
        patches_rgb = 1 - torch.tensor(patches_rgb).squeeze(3).unsqueeze(1) / 255.0

        logger.info(f"Starting refinement optimization for dataset '{name}'",
                   extra={"dataset": name, "num_patches": patches_vector.shape[0],
                         "primitives_per_patch": patches_vector.shape[1]})

        logger.info(f"Random initialization: {options.init_random}",
                   extra={"init_random": options.init_random})

        if options.init_random:
            logger.info("Initializing with random vectors")
            patches_vector = torch.rand((patches_vector.shape)) * 64
            patches_vector[..., 4] = 1

        # Initialize batch processor
        batch_processor = BatchProcessor(patches_rgb.squeeze(1), patches_vector, refinement_config.batch_size)

        # Process batches
        first_encounter = True
        iou_all = None
        mass_for_iou = None
        total_batches = (patches_vector.shape[0] + refinement_config.batch_size - 1) // refinement_config.batch_size

        logger.info(f"Processing {total_batches} batches of size {refinement_config.batch_size}",
                   extra={"total_batches": total_batches, "batch_size": refinement_config.batch_size})

        for batch_idx, batch_start in enumerate(range(0, patches_vector.shape[0], refinement_config.batch_size)):
            batch_end = min(batch_start + refinement_config.batch_size, patches_vector.shape[0])
            batch_indices = list(range(batch_start, batch_end))

            batch_logger = logger.with_context(batch=batch_idx, batch_start=batch_start, batch_end=batch_end)

            # Skip empty batches
            batch_rgb = patches_rgb.squeeze(1)[batch_indices]
            if torch.mean(batch_rgb) == 0:
                batch_logger.debug("Skipping empty batch")
                continue

            batch_logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

            # Get batch data
            rasters_batch, initial_vector = batch_processor.get_batch_data(batch_indices)

            # Initialize lines with visibility threshold
            removed_lines = initial_vector[..., -1] < refinement_config.line_visibility_threshold * h
            rand_x1 = torch.rand_like(initial_vector[removed_lines, [0]]) * w
            rand_y1 = torch.rand_like(initial_vector[removed_lines, [1]]) * h
            initial_vector[removed_lines, [0]] = rand_x1
            initial_vector[removed_lines, [2]] = rand_x1 + 1
            initial_vector[removed_lines, [1]] = rand_y1
            initial_vector[removed_lines, [3]] = rand_y1 + 1
            initial_vector[removed_lines, [4]] = refinement_config.initial_probability
            initial_vector = initial_vector[..., :5].numpy()

            # Initialize optimization loop
            opt_loop = OptimizationLoop(rasters_batch, initial_vector, device, options.rendering_type, batch_logger)

            # Register signal handler for graceful interruption
            its_time_to_stop = [False]
            register_sigint_flag(its_time_to_stop)

            # Optimization loop
            iou_mass = []
            mass_for_iou_one = []
            early_stop_threshold = 0.001  # Stop if IOU improvement < 0.1% over last 3 measurements
            min_iterations = 20  # Run at least 20 iterations before checking early stopping

            with batch_logger.timing(f"batch_{batch_idx}_optimization"):
                for i in tqdm(range(options.diff_render_it), desc=f"Batch {batch_idx + 1}"):
                    if not opt_loop.optimize_step(i) or its_time_to_stop[0]:
                        batch_logger.warning("Optimization interrupted or stopped")
                        break

                    # Log progress
                    opt_loop.log_progress(i, patches_rgb_im, batch_indices, iou_mass, mass_for_iou_one)

                    # Check for early stopping based on IOU convergence
                    if (i >= min_iterations and 
                        (i + 1) % refinement_config.logging_interval == 0 and 
                        len(iou_mass) >= 3):
                        # Check if IOU improvement over last 3 measurements is minimal
                        recent_iou = iou_mass[-3:]
                        improvements = [recent_iou[j+1] - recent_iou[j] for j in range(len(recent_iou)-1)]
                        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
                        
                        if avg_improvement < early_stop_threshold:
                            batch_logger.info(f"Early stopping at iteration {i+1}: IOU improvement {avg_improvement:.6f} < {early_stop_threshold}")
                            break

            # Store results
            final_lines = opt_loop.get_final_result()
            y_pred_rend[batch_indices] = final_lines.cpu().detach()

            # Accumulate IOU data
            if first_encounter:
                first_encounter = False
                iou_all = np.array(iou_mass)
                mass_for_iou = np.array(mass_for_iou_one)
            else:
                iou_all = np.concatenate((iou_all, iou_mass), axis=1)
                mass_for_iou = np.concatenate((mass_for_iou, mass_for_iou_one), axis=1)

            batch_logger.info(f"Completed batch {batch_idx + 1}/{total_batches}")

        # Post-process results
        prd = y_pred_rend[:, :, -1].clone()
        prd[prd > 1] = 1
        prd[prd < refinement_config.probability_clip_min] = 0
        y_pred_rend = torch.cat((y_pred_rend, prd.unsqueeze(2)), dim=-1)

        # Save results
        os.makedirs(options.output_dir + "arrays/", exist_ok=True)
        if options.init_random:
            np.save(options.output_dir + "arrays/hard_optimization_iou_random_" + name, iou_all)
            np.save(options.output_dir + "arrays/hard_optimization_iou_mass_random_" + name, mass_for_iou)
        else:
            np.save(options.output_dir + "arrays/hard_optimization_iou_" + name, iou_all)
            np.save(options.output_dir + "arrays/hard_optimization_iou_mass_" + name, mass_for_iou)

        logger.log_pipeline_step("refinement", "completed", module="lines", dataset=name)
        logger.log_performance("refinement", {
            "total_patches": patches_vector.shape[0],
            "primitives_per_patch": patches_vector.shape[1],
            "iterations_per_patch": options.diff_render_it,
            "final_iou_mean": float(np.mean(iou_all)) if iou_all is not None else None
        })

    return y_pred_rend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="/vage/Download/testing_line/", help="dir to folder for output"
    )
    parser.add_argument("--diff_render_it", type=int, default=90, help="iteration count")
    parser.add_argument(
        "--init_random",
        action="store_true",
        default=False,
        dest="init_random",
        help="init model with random [default: False].",
    )
    parser.add_argument(
        "--rendering_type",
        type=str,
        default="bezier_splatting",
        help="hard - analytical rendering, bezier_splatting - fast differentiable rendering (recommended)",
    )
    return parser.parse_args()
