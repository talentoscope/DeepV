import os
import pickle
import sys
from argparse import ArgumentParser
from time import time

import h5py
import numpy as np
import torch
from tqdm import trange

sys.path.append(".")
from util_files.evaluation_utils import vector_image_from_patches
from util_files.optimization.optimizer.adam import Adam
from util_files.optimization.primitives.line_tensor import LineTensor
from util_files.optimization.primitives.quadratic_bezier_tensor import (
    QuadraticBezierTensor,
)
from util_files.simplification.join_qb import join_quad_beziers
from util_files.structured_logging import get_pipeline_logger


class DataLoader:
    """Handles data loading and preprocessing for refinement."""

    def __init__(self, options, logger):
        self.options = options
        self.logger = logger

    def load_data(self):
        """Load and preprocess the input data."""
        self.logger.info("1. Load data")

        if hasattr(self.options, "intermediate_output") and self.options.intermediate_output is not None:
            # Load from intermediate_output (legacy mode)
            intermediate_output = self.options.intermediate_output
            sample_name = intermediate_output["options"].sample_name[:-4]
            patch_offsets = torch.as_tensor(intermediate_output["patches_offsets"], dtype=self.options.dtype)
            model_output = torch.as_tensor(intermediate_output["patches_vector"], dtype=self.options.dtype)
            raster_patches = torch.as_tensor(intermediate_output["patches_rgb"], dtype=self.options.dtype)
            raster_patches = raster_patches.reshape(raster_patches.shape[:3])
            repatch_scale = None
            primitive_type_from_model = None
        else:
            # Load from pickle file (new mode)
            with open(f"{self.options.input_dir}/data.pkl", "rb") as f:
                data = pickle.load(f)

            sample_name = data["sample_name"]
            raster_patches = data["raster_patches"]
            patch_offsets = data["patch_offsets"]
            model_output = data["model_output"]
            repatch_scale = data["repatch_scale"]
            primitive_type_from_model = data["primitive_type_from_model"]

        self.logger.info(f"Sample name: {sample_name}")
        self.logger.info(f"Raster patches shape: {raster_patches.shape}")
        self.logger.info(f"Model output shape: {model_output.shape}")

        return (
            sample_name,
            raster_patches,
            patch_offsets,
            model_output,
            repatch_scale,
            primitive_type_from_model,
        )

    def preprocess_data(self, raster_patches, patch_offsets, model_output, repatch_scale):
        """Preprocess the loaded data."""
        self.logger.info("2. Preprocess data")

        # Handle repatching
        if hasattr(self.options, "intermediate_output") and self.options.intermediate_output is not None:
            # Legacy intermediate_output mode
            if not self.options.init_random:
                self.logger.info("2.5. Repatch")
                confident = model_output[:, :, -1] >= self.options.min_confidence
                widths = model_output[:, :, -2][confident].reshape(-1)
                the_width = np.percentile(widths, self.options.the_width_percentile)
                repatch_scale = int(round(the_width / 2))
                self.logger.info(f"\tthe width is {the_width}")
                self.logger.info(f"\tthe width percentile is {self.options.the_width_percentile}")
                self.logger.info(f"\trepatch scale is {repatch_scale}")
                raster_patches, patch_offsets, model_output = repatch(
                    raster_patches, patch_offsets, model_output, repatch_scale
                )
                model_output = merge_close(
                    model_output,
                    self.options.min_confidence,
                    self.options.the_width_percentile,
                )
        else:
            # New mode - repatch if needed
            if repatch_scale > 1:
                raster_patches, patch_offsets, model_output = repatch(
                    raster_patches, patch_offsets, model_output, scale=repatch_scale
                )

        # Merge close primitives
        if not hasattr(self.options, "intermediate_output") or self.options.intermediate_output is None:
            model_output = merge_close(model_output, self.options.min_confidence)

        # Convert raster from uint8 0-255 with white background to 0-1 with 0 background
        if hasattr(self.options, "intermediate_output") and self.options.intermediate_output is not None:
            raster_patches /= -255
            raster_patches += 1

        # Filter out empty patches
        self.logger.info("3. Filter out empty patches")
        patches_n_before = len(raster_patches)
        self.logger.info(f"\tfrom {patches_n_before} patches")
        nonempty = (raster_patches > 0).any(dim=-1).any(dim=-1)
        raster_patches = raster_patches[nonempty].contiguous()
        patch_offsets = patch_offsets[nonempty].contiguous()
        model_output = model_output[nonempty].contiguous()
        del nonempty
        patches_n = len(raster_patches)
        self.logger.info(f"\t{patches_n} patches left")

        # Group patches into batches
        if hasattr(self.options, "intermediate_output") and self.options.intermediate_output is not None:
            # Legacy mode
            if not self.options.init_random:
                self.logger.info("3.5. Sort patches")
                batches, patch_offsets_list = group_patches(
                    model_output,
                    raster_patches,
                    patch_offsets,
                    self.options.min_confidence,
                    self.options.primitives_n,
                    self.options.batch_size,
                )
                patch_offsets = torch.cat(patch_offsets_list, dim=0)
                patches_n = len(patch_offsets)
                self.options.primitives_n = batches[-1][0].shape[1]
                self.logger.info(
                    f"\t{patches_n} patches left with max " f"{self.options.primitives_n} primitives per patch"
                )
            else:
                batches = []
                for first_patch_i in range(0, patches_n, self.options.batch_size):
                    next_first_patch_i = min(patches_n, first_patch_i + self.options.batch_size)
                    batches.append(
                        (
                            (
                                next_first_patch_i - first_patch_i,
                                self.options.primitives_n,
                            ),
                            raster_patches[first_patch_i:next_first_patch_i],
                        )
                    )
        else:
            # New mode
            batches, patches_offset_batches = group_patches(
                model_output,
                raster_patches,
                patch_offsets,
                self.options.min_confidence,
                self.options.max_prims_n,
                self.options.max_batch_size,
            )

        self.logger.info(f"Number of batches: {len(batches)}")

        return (
            raster_patches,
            patch_offsets,
            model_output,
            batches,
            patches_offset_batches if "patches_offset_batches" in locals() else None,
        )


class MetricsLogger:
    """Handles metrics collection and logging during optimization."""

    def __init__(self, options, sample_name, patches_n, logger):
        self.options = options
        self.sample_name = sample_name
        self.patches_n = patches_n
        self.logger = logger
        self.metrics_file_path = f"{options.output_dir}/logs/{sample_name}.h5"

    def setup_metrics_file(self, optimization_iters_n, measure_period):
        """Set up the HDF5 file for metrics logging."""
        os.makedirs(os.path.dirname(self.metrics_file_path), exist_ok=True)
        self.logger.info(f"7. Prepare file with metrics at {self.metrics_file_path}")

        self.metrics_file = h5py.File(self.metrics_file_path, "w")
        log_iters_n = (optimization_iters_n - 1) // measure_period + 1
        self.intersection_array = self.metrics_file.create_dataset(
            "intersection", dtype="f", shape=[log_iters_n, self.patches_n]
        )
        self.union_array = self.metrics_file.create_dataset("union", dtype="f", shape=[log_iters_n, self.patches_n])

    def log_metrics(self, log_i, rasterization, binary_raster, first_patch_i, next_first_patch_i):
        """Log intersection and union metrics."""
        self.intersection_array[log_i, first_patch_i:next_first_patch_i] = (
            (rasterization & binary_raster).sum(dim=-1).sum(dim=-1)
        )
        self.union_array[log_i, first_patch_i:next_first_patch_i] = (
            (rasterization | binary_raster).sum(dim=-1).sum(dim=-1)
        )

    def close(self):
        """Close the metrics file."""
        if hasattr(self, "metrics_file"):
            self.metrics_file.close()


class Optimizer:
    """Handles the optimization process for primitive refinement."""

    def __init__(self, options, batches, patches_offset_batches, logger):
        self.options = options
        self.batches = batches
        self.patches_offset_batches = patches_offset_batches
        self.logger = logger

    def create_init_function(self):
        """Create the primitive initialization function."""
        return make_init_primitives(
            self.options.init_random,
            self.options.primitive_type,
            self.batches,
            self.options.patch_width,
            self.options.patch_height,
            torch.float32,
            self.options.device,
            self.options.primitive_type_from_model,
            self.logger,
        )

    def optimize_batch(self, batch_i, init_primitives, metrics_logger, first_patch_i):
        """Optimize a single batch of patches."""
        self.logger.info("\tInitialize")
        prim_ten = init_primitives(batch_i)
        q_raster = self.batches[batch_i][1]
        next_first_patch_i = first_patch_i + len(q_raster)

        aligner = Adam(prim_ten, q_raster, logger=self.logger, lr=self.options.lr)
        del prim_ten
        binary_raster = binarize(q_raster)
        del q_raster

        for i in trange(
            self.options.optimization_iters_n,
            desc=f"\toptimize patches {first_patch_i}-{next_first_patch_i - 1}",
            file=self.logger.info_stream,
            position=0,
            leave=True,
        ):
            # Measure and log metrics
            if i % self.options.measure_period == 0:
                log_i = i // self.options.measure_period
                rasterization = binarize(
                    aligner.prim_ten.render_with_cairo_total(
                        self.options.patch_width,
                        self.options.patch_height,
                        min_width=self.options.min_width,
                    )
                )
                metrics_logger.log_metrics(
                    log_i,
                    rasterization,
                    binary_raster,
                    first_patch_i,
                    next_first_patch_i,
                )
                del rasterization

            if i % self.options.merge_period == 0:
                aligner.prim_ten.merge_close()

            # Optimize
            aligner.step(
                i,
                reinit_period=(self.options.reinit_period if i <= self.options.max_reinit_iter else 1000000),
            )

        return aligner, next_first_patch_i

    def run_optimization(self, init_primitives, metrics_logger):
        """Run the complete optimization process."""
        self.logger.info("8. Optimization")
        optimization_start_time = time()

        primitives_after_optimization = torch.zeros(
            [
                len(self.patches_offset_batches),
                self.options.max_prims_n,
                self.options.parameters_n,
            ],
            dtype=torch.float32,
        )

        first_patch_i = 0
        for batch_i in trange(len(self.batches), file=self.logger.info_stream, desc="Optimize batches"):
            aligner, next_first_patch_i = self.optimize_batch(batch_i, init_primitives, metrics_logger, first_patch_i)

            assemble_primitives_to(
                aligner.prim_ten,
                primitives_after_optimization,
                first_patch_i,
                next_first_patch_i,
                self.options.min_width,
                self.options.min_confidence,
            )
            first_patch_i = next_first_patch_i

        self.logger.info(f"\tOptimization took {time() - optimization_start_time} seconds")
        return primitives_after_optimization

    def run_optimization_legacy(self, init_primitives, metrics_logger):
        """Run optimization in legacy mode (for intermediate_output)."""
        self.logger.info("8. Optimization")
        optimization_start_time = time()

        first_patch_i = 0
        for batch_i in trange(len(self.batches), file=self.logger.info_stream, desc="Optimize batches"):
            aligner, next_first_patch_i = self.optimize_batch_legacy(
                batch_i, init_primitives, metrics_logger, first_patch_i
            )
            first_patch_i = next_first_patch_i

        self.logger.info(f"\tOptimization took {time() - optimization_start_time} seconds")
        return self.options.primitives_after_optimization

    def optimize_batch_legacy(self, batch_i, init_primitives, metrics_logger, first_patch_i):
        """Optimize a single batch of patches in legacy mode."""
        self.logger.info("\tInitialize")
        prim_ten = init_primitives(batch_i)
        q_raster = self.batches[batch_i][1]
        next_first_patch_i = first_patch_i + len(q_raster)

        aligner = Adam(prim_ten, q_raster, logger=self.logger, lr=self.options.lr)
        del prim_ten
        binary_raster = binarize(q_raster)
        del q_raster

        for i in trange(
            self.options.optimization_iters_n,
            desc=f"\toptimize patches {first_patch_i}-{next_first_patch_i - 1}",
            file=self.logger.info_stream,
            position=0,
            leave=True,
        ):
            # Measure and log metrics
            if i % self.options.measure_period == 0:
                log_i = i // self.options.measure_period
                rasterization = binarize(
                    aligner.prim_ten.render_with_cairo_total(
                        self.options.patch_width,
                        self.options.patch_height,
                        min_width=self.options.min_width,
                    )
                )
                metrics_logger.log_metrics(
                    log_i,
                    rasterization,
                    binary_raster,
                    first_patch_i,
                    next_first_patch_i,
                )
                del rasterization

            if i % self.options.merge_period == 0:
                aligner.prim_ten.merge_close()

            # Optimize
            aligner.step(
                i,
                reinit_period=(self.options.reinit_period if i <= self.options.max_reinit_iter else 1000000),
            )

        assemble_primitives_to(
            aligner.prim_ten,
            self.options.primitives_after_optimization,
            first_patch_i,
            next_first_patch_i,
            self.options.min_width,
            self.options.min_confidence,
        )

        return aligner, next_first_patch_i


class OutputGenerator:
    """Handles output generation and saving."""

    def __init__(self, options, sample_name, logger):
        self.options = options
        self.sample_name = sample_name
        self.logger = logger

    def save_optimization_result(self, primitives_after_optimization):
        """Save the optimization result to SVG."""
        optimization_output_path = f"{self.options.output_dir}/after_optimization/{self.sample_name}.svg"
        self.logger.info(f"9. Save optimization result to {optimization_output_path}")
        os.makedirs(os.path.dirname(optimization_output_path), exist_ok=True)

        from util_files.evaluation_utils import get_vectorimage

        get_vectorimage(primitives_after_optimization).save(optimization_output_path)


class RefinementPipeline:
    """Main pipeline orchestrator for curve refinement."""

    def __init__(self, options):
        self.options = options
        self.logger = get_pipeline_logger("refinement.curves")

    def run(self):
        """Run the complete refinement pipeline."""
        # Data loading and preprocessing
        data_loader = DataLoader(self.options, self.logger)
        (
            sample_name,
            raster_patches,
            patch_offsets,
            model_output,
            repatch_scale,
            primitive_type_from_model,
        ) = data_loader.load_data()

        # Handle legacy mode setup
        if hasattr(self.options, "intermediate_output") and self.options.intermediate_output is not None:
            # Set up parameters for legacy mode
            intermediate_output = self.options.intermediate_output
            self.options.whole_image_size = intermediate_output["cleaned_image_shape"]
            self.options.patch_height, self.options.patch_width = raster_patches.shape[1:3]

            # Determine primitive types
            primitive_parameters_n = model_output.shape[2]
            if primitive_parameters_n == 6:
                self.options.primitive_type_from_model = "lines"
            elif primitive_parameters_n == 8:
                self.options.primitive_type_from_model = "qbeziers"
            else:
                raise NotImplementedError(f"Unknown number of parameters {primitive_parameters_n}")

            if self.options.primitive_type is None:
                self.options.primitive_type = self.options.primitive_type_from_model

            # Set up output directory
            outp_dir = self.options.output_dir
            init_dir_name = "random_initialization" if self.options.init_random else "model_initialization"
            if self.options.append_iters_to_outdir:
                self.options.output_dir = (
                    f"{self.options.output_dir}/" f"{self.options.optimization_iters_n}/" f"{init_dir_name}"
                )
            else:
                self.options.output_dir = f"{self.options.output_dir}/{init_dir_name}"

            # Initialize primitives tensor
            if self.options.primitive_type == "lines":
                primitive_parameters_n = 6
            elif self.options.primitive_type == "qbeziers":
                primitive_parameters_n = 8

            patches_n = len(patch_offsets)
            if not self.options.init_random:
                # This would need to be set from batches - for now assume it's available
                primitives_n = getattr(self.options, "primitives_n", model_output.shape[1])
                self.options.primitives_after_optimization = model_output.new_zeros(
                    [patches_n, primitives_n, primitive_parameters_n]
                )
            else:
                primitives_n = getattr(self.options, "primitives_n", model_output.shape[1])
                self.options.primitives_after_optimization = torch.zeros(
                    [patches_n, primitives_n, primitive_parameters_n],
                    dtype=self.options.dtype,
                )
        else:
            # Update options with loaded data
            self.options.primitive_type_from_model = primitive_type_from_model

        (
            raster_patches,
            patch_offsets,
            model_output,
            batches,
            patches_offset_batches,
        ) = data_loader.preprocess_data(raster_patches, patch_offsets, model_output, repatch_scale)

        # Optimization
        optimizer = Optimizer(self.options, batches, patches_offset_batches, self.logger)
        init_primitives = optimizer.create_init_function()

        # Metrics logging setup
        patches_n = len(patches_offset_batches) if patches_offset_batches is not None else len(patch_offsets)
        metrics_logger = MetricsLogger(self.options, sample_name, patches_n, self.logger)
        metrics_logger.setup_metrics_file(self.options.optimization_iters_n, self.options.measure_period)

        # Run optimization
        if hasattr(self.options, "intermediate_output") and self.options.intermediate_output is not None:
            # Legacy mode
            primitives_after_optimization = optimizer.run_optimization_legacy(init_primitives, metrics_logger)
        else:
            # New mode
            primitives_after_optimization = optimizer.run_optimization(init_primitives, metrics_logger)

        # Close metrics file
        metrics_logger.close()

        # Save output
        output_generator = OutputGenerator(self.options, sample_name, self.logger)
        output_generator.save_optimization_result(primitives_after_optimization)

        # Restore original output directory if in legacy mode
        if hasattr(self.options, "intermediate_output") and self.options.intermediate_output is not None:
            self.options.output_dir = outp_dir

        from util_files.evaluation_utils import get_vectorimage

        return (
            primitives_after_optimization,
            patch_offsets,
            repatch_scale,
            get_vectorimage(primitives_after_optimization),
        )


def build_vectorimage(
    patches,
    patch_offsets,
    image_size,
    control_points_n,
    patch_size,
    scale,
    min_width,
    min_confidence,
    min_length,
):
    """Top-level helper to build a VectorImage from per-patch primitives.

    This extracts the call to `vector_image_from_patches` out of the `main` scope so it
    can be reused or unit-tested more easily.
    """
    return vector_image_from_patches(
        primitives=patches,
        patch_offsets=patch_offsets,
        image_size=image_size,
        control_points_n=control_points_n,
        patch_size=patch_size,
        pixel_center_coodinates_are_integer=False,
        scale=scale,
        min_width=min_width,
        min_confidence=min_confidence,
        min_length=min_length,
    )


def main(
    options,
    intermediate_output=None,
    control_points_n=3,
    dtype=torch.float32,
    device="cuda",
    primitives_n=None,
    primitive_type=None,
    merge_period=60,
    lr=0.05,
    the_width_percentile=90,
    optimization_iters_n=None,
    batch_size=300,
    measure_period=20,
    reinit_period=20,
    max_reinit_iter=100,
    min_width=0.3,
    min_confidence=64 * 0.5,
    min_length=1.7,
    append_iters_to_outdir=True,
):
    """Legacy main function - now delegates to the RefinementPipeline."""
    # For backward compatibility, handle the intermediate_output parameter
    if intermediate_output is not None:
        # Extract options from intermediate_output and merge with provided options
        for k, v in options.__dict__.items():
            setattr(intermediate_output["options"], k, v)
        options = intermediate_output["options"]

        # Create a temporary options object for the pipeline
        class TempOptions:
            pass

        temp_options = TempOptions()
        # Copy all attributes from intermediate_output options
        for k, v in intermediate_output["options"].__dict__.items():
            setattr(temp_options, k, v)

        # Override with function parameters
        temp_options.intermediate_output = intermediate_output
        temp_options.control_points_n = control_points_n
        temp_options.dtype = dtype
        temp_options.device = device
        temp_options.primitives_n = primitives_n
        temp_options.primitive_type = primitive_type
        temp_options.merge_period = merge_period
        temp_options.lr = lr
        temp_options.the_width_percentile = the_width_percentile
        temp_options.optimization_iters_n = optimization_iters_n
        temp_options.batch_size = batch_size
        temp_options.measure_period = measure_period
        temp_options.reinit_period = reinit_period
        temp_options.max_reinit_iter = max_reinit_iter
        temp_options.min_width = min_width
        temp_options.min_confidence = min_confidence
        temp_options.min_length = min_length
        temp_options.append_iters_to_outdir = append_iters_to_outdir

        pipeline = RefinementPipeline(temp_options)
        return pipeline.run()
    else:
        raise ValueError("intermediate_output must be provided - job_tuples mode not supported")


def repatch(raster_patches, patch_offsets, model_output, scale=None, h=None, w=None):
    if h is None:
        h = scale
    if w is None:
        w = scale

    patches_n = len(patch_offsets)
    patches_n_hor = (patch_offsets[:, 0] != 0).numpy().argmax()
    assert patches_n % patches_n_hor == 0
    patches_n_vert = len(patch_offsets) // patches_n_hor

    if patches_n_vert % h == 0:
        pad_vert = 0
    else:
        pad_vert = h - (patches_n_vert % h)
    if patches_n_hor % w == 0:
        pad_hor = 0
    else:
        pad_hor = w - (patches_n_hor % w)

    patch_h, patch_w = raster_patches.shape[1:]
    primitives_n, parameters_n = model_output.shape[1:]
    raster_patches = raster_patches.reshape(patches_n_vert, patches_n_hor, patch_h, patch_w)
    patch_offsets = patch_offsets.reshape(patches_n_vert, patches_n_hor, 2)
    model_output = model_output.reshape(patches_n_vert, patches_n_hor, primitives_n, parameters_n)

    raster_patches = torch.nn.functional.pad(raster_patches, [0, 0, 0, 0, 0, pad_hor, 0, pad_vert], value=255)
    patch_offsets = torch.nn.functional.pad(patch_offsets, [0, 0, 0, pad_hor, 0, pad_vert])
    model_output = torch.nn.functional.pad(model_output, [0, 0, 0, 0, 0, pad_hor, 0, pad_vert])

    patches_n_vert, patches_n_hor = raster_patches.shape[:2]
    patches_n_vert = patches_n_vert // h
    patches_n_hor = patches_n_hor // w

    raster_patches = raster_patches.reshape(patches_n_vert, h, patches_n_hor, w, patch_h, patch_w)
    patch_offsets = patch_offsets.reshape(patches_n_vert, h, patches_n_hor, w, 2)
    model_output = model_output.reshape(patches_n_vert, h, patches_n_hor, w, primitives_n, parameters_n)

    raster_patches = raster_patches.permute(0, 2, 1, 4, 3, 5)
    model_output = model_output.permute(0, 2, 1, 3, 4, 5)
    patch_offsets = patch_offsets[:, 0, :, 0]

    shifts = model_output.new_zeros([h, w, 1, parameters_n])
    for i in range(h):
        shifts[i, :, :, 1:-2:2] += i * patch_h
    for j in range(w):
        shifts[:, j, :, 0:-2:2] += j * patch_w

    model_output = model_output + shifts

    patch_h = patch_h * h
    patch_w = patch_w * w

    raster_patches = raster_patches.reshape(-1, patch_h, patch_w)
    model_output = model_output.reshape(-1, primitives_n * h * w, parameters_n)
    patch_offsets = patch_offsets.reshape(-1, 2)

    if scale is not None:
        assert (patch_h % scale == 0) and (patch_w % scale == 0)
        patch_h //= scale
        patch_w //= scale
        patch_offsets /= scale
        model_output[:, :, :-1] /= scale
        raster_patches = (
            raster_patches.reshape(-1, patch_h, scale, patch_w, scale).permute(0, 1, 3, 2, 4).mean(dim=[-1, -2])
        )

    raster_patches = raster_patches.contiguous()
    model_output = model_output.contiguous()
    patch_offsets = patch_offsets.contiguous()

    return raster_patches, patch_offsets, model_output


def group_patches(
    model_output,
    raster_patches,
    patches_offset,
    min_confidence,
    max_prims_n,
    max_batch_size,
):
    parameters_n = model_output.shape[2]
    confident_primitives = model_output[:, :, -1] >= min_confidence
    prims_n = confident_primitives.sum(dim=-1)

    batches = []
    patches_offset_batches = []
    for prims_n_in_batch in range(1, max_prims_n + 1):
        mask = prims_n == prims_n_in_batch
        if not mask.any():
            continue
        masked_model_out = model_output[mask]
        masked_model_out = masked_model_out[confident_primitives[mask], :].reshape(-1, prims_n_in_batch, parameters_n)
        masked_raster_patches = raster_patches[mask]
        masked_patches_offset = patches_offset[mask]
        patches_n = len(masked_model_out)
        for first_patch_i in range(0, patches_n, max_batch_size):
            next_first_patch_i = min(first_patch_i + max_batch_size, patches_n)
            batches.append(
                [
                    masked_model_out[first_patch_i:next_first_patch_i],
                    masked_raster_patches[first_patch_i:next_first_patch_i],
                ]
            )
            patches_offset_batches.append(masked_patches_offset[first_patch_i:next_first_patch_i])
    return batches, patches_offset_batches


def make_init_primitives(
    random,
    primitive_type,
    batches,
    patch_width,
    patch_height,
    dtype,
    device,
    primitive_type_from_model,
    logger,
):
    if random:
        logger.info("\tinitialization is random")
        if primitive_type == "lines":

            def init_primitives(batch_i):
                n, primitives_n = batches[batch_i][0]
                return LineTensor(
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 1, primitives_n) + 1,
                    dtype=dtype,
                    device=device,
                )

        elif primitive_type == "qbeziers":

            def init_primitives(batch_i):
                n, primitives_n = batches[batch_i][0]
                return QuadraticBezierTensor(
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 1, primitives_n) + 1,
                    dtype=dtype,
                    device=device,
                )

    else:
        logger.info("\tinitialization is model")
        if primitive_type == "lines":
            if primitive_type_from_model == "lines":

                def init_primitives(batch_i):
                    model_output = batches[batch_i][0].permute(0, 2, 1).contiguous()
                    p1 = model_output[:, :2]
                    p2 = model_output[:, 2:4]
                    w = model_output[:, 4:5]
                    return LineTensor(p1, p2, w, dtype=dtype, device=device)

            else:
                raise NotImplementedError("Please implement conversion from curves to lines")
        elif primitive_type == "qbeziers":
            if primitive_type_from_model == "qbeziers":

                def init_primitives(batch_i):
                    model_output = batches[batch_i][0].permute(0, 2, 1).contiguous()
                    p1 = model_output[:, :2]
                    p2 = model_output[:, 2:4]
                    p3 = model_output[:, 4:6]
                    w = model_output[:, 6:7]
                    return QuadraticBezierTensor(p1, p2, p3, w, dtype=dtype, device=device)

            elif primitive_type_from_model == "lines":

                def init_primitives(batch_i):
                    model_output = batches[batch_i][0].permute(0, 2, 1).contiguous()
                    p1 = model_output[:, :2]
                    p3 = model_output[:, 2:4]
                    p2 = (p1 + p3) / 2
                    w = model_output[:, 4:5]
                    return QuadraticBezierTensor(p1, p2, p3, w, dtype=dtype, device=device)

    return init_primitives


def assemble_primitives_to(
    primitive_tensor,
    data_tensor,
    first_patch_i,
    next_first_patch_i,
    min_width,
    min_confidence,
):
    primitives_n = primitive_tensor.primitives_n
    good_confidence = min_confidence * 2
    if isinstance(primitive_tensor, LineTensor):
        p1 = primitive_tensor.p1.data.cpu()
        p2 = primitive_tensor.p2.data.cpu()
        width = primitive_tensor.width.data.cpu()
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, :2] = p1.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 2:4] = p2.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 4] = width[:, 0]
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 5] = width[:, 0] >= min_width
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 5] *= good_confidence
    elif isinstance(primitive_tensor, QuadraticBezierTensor):
        p1 = primitive_tensor.p1.data.cpu()
        p2 = primitive_tensor.p2.data.cpu()
        p3 = primitive_tensor.p3.data.cpu()
        width = primitive_tensor.width.cpu()
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, :2] = p1.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 2:4] = p2.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 4:6] = p3.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 6] = width[:, 0]
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 7] = width[:, 0] >= min_width
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 7] *= good_confidence


def binarize(raster):
    return raster > 0.5


def merge_close(model_output, min_confidence, width_percentile=90):
    patches_n, primitives_n, parameters_n = model_output.shape
    new_model_output = model_output.new_zeros([patches_n, primitives_n, parameters_n])
    max_primitives_in_patch = 0
    for i, patch in enumerate(model_output):
        confident = patch[:, -1] >= min_confidence
        if not confident.any():
            continue
        patch = patch[confident, :-1]
        common_width_in_patch = np.percentile(patch[:, -1], width_percentile)
        join_tol = 0.5 * common_width_in_patch
        fit_tol = 0.5 * common_width_in_patch
        w_tol = np.inf
        new_patch = join_quad_beziers(patch, join_tol=join_tol, fit_tol=fit_tol, w_tol=w_tol)
        new_patch = np.pad(new_patch, [[0, 0], [0, 1]], constant_values=min_confidence * 2)
        new_primitives_n = len(new_patch)
        max_primitives_in_patch = max(max_primitives_in_patch, new_primitives_n)
        new_model_output.numpy()[i, :new_primitives_n] = new_patch
    return new_model_output[:, :max_primitives_in_patch]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["synthetic", "custom"])
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--optimization_iters_n", type=int, default=100, help="iteration count")
    parser.add_argument("--init_random", action="store_true", help="init optimization randomly")

    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    main(options)
