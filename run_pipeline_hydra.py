"""DeepV Pipeline with Hydra Configuration.

This is a refactored version of run_pipeline.py that uses Hydra for configuration management.
It demonstrates how to use the new configuration system for reproducible experiments.
"""

import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from merging.merging_for_curves import main as curve_merging
from merging.merging_for_lines import postprocess
from refinement.our_refinement.refinement_for_curves import main as curve_refinement
from refinement.our_refinement.refinement_for_lines import render_optimization_hard
from util_files.patchify import patchify
from vectorization import load_model


def serialize(checkpoint):
    """Fix checkpoint keys for transformer models."""
    model_state_dict = checkpoint["model_state_dict"]
    keys = []
    for k in model_state_dict:
        if "hidden.transformer" in k:
            keys.append(k)

    for k in keys:
        new_key = "hidden.decoder.transformer" + k[len("hidden.transformer") :]
        model_state_dict[new_key] = model_state_dict[k]
        del model_state_dict[k]
    return checkpoint


def split_to_patches(rgb, patch_size, overlap=0):
    """Separates the input into square patches of specified size.

    :param rgb: input RGB image
    :type rgb: numpy.ndarray
    :param patch_size: size of patches in pixels (assuming square patches)
    :type patch_size: int
    :param overlap: amount in pixels of how much the patches
                    can overlap with each other (useful for merging)
    :type overlap: int

    :returns patches, patches_offsets
    :rtype Tuple[numpy.ndarray, numpy.ndarray]
    """
    return patchify(rgb, patch_size, overlap)


def vectorization_stage(options, model, json_path):
    """Run vectorization on patches."""
    # Load and preprocess image
    image_path = Path(options.data.data_dir) / options.data.image_name
    rgb = np.array(Image.open(image_path).convert("RGB"))

    # Split into patches
    patches, patches_offsets = split_to_patches(rgb, patch_size=64, overlap=options.pipeline.overlap)

    # Load model
    model = load_model(json_path, options.model.model_path)

    # Run vectorization
    patches_vector = []
    for patch in patches:
        patch_tensor = transforms.ToTensor()(patch).unsqueeze(0)
        if options.gpu is not None:
            patch_tensor = patch_tensor.cuda(options.gpu[0])

        with torch.no_grad():
            output = model(patch_tensor, n=options.pipeline.model_output_count)
            patches_vector.append(output.cpu().numpy())

    return patches_vector, patches_offsets


def refinement_stage(options, patches_vector, patches_offsets):
    """Run refinement on vectorized patches."""
    if options.pipeline.primitive_type == "line":
        return render_optimization_hard(
            patches_vector,
            patches_offsets,
            diff_render_it=options.pipeline.diff_render_it,
            rendering_type=options.pipeline.rendering_type,
        )
    elif options.pipeline.primitive_type == "curve":
        return curve_refinement(
            patches_vector,
            patches_offsets,
            diff_render_it=options.pipeline.diff_render_it,
        )
    else:
        raise ValueError(f"{options.pipeline.primitive_type} not implemented, please choose between line or curve")


def merging_stage(options, refinement_result):
    """Run merging on refined patches."""
    if options.pipeline.primitive_type == "line":
        return postprocess(
            refinement_result,
            max_angle_to_connect=options.pipeline.max_angle_to_connect,
            max_distance_to_connect=options.pipeline.max_distance_to_connect,
        )
    elif options.pipeline.primitive_type == "curve":
        return curve_merging(refinement_result)
    else:
        raise ValueError(f"{options.pipeline.primitive_type} not implemented, please choose between line or curve")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main pipeline function using Hydra configuration."""
    print(f"Running DeepV pipeline with config: {cfg.experiment_name}")
    print(f"Primitive type: {cfg.pipeline.primitive_type}")
    print(f"GPU: {cfg.gpu}")

    # Set random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Set GPU
    if cfg.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpu))

    # Run pipeline stages
    try:
        # Vectorization
        print("Running vectorization...")
        patches_vector, patches_offsets = vectorization_stage(cfg, None, cfg.model.json_path)

        # Refinement
        print("Running refinement...")
        refinement_result = refinement_stage(cfg, patches_vector, patches_offsets)

        # Merging
        print("Running merging...")
        merging_result = merging_stage(cfg, refinement_result)

        # Save results
        output_dir = Path(cfg.data.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{Path(cfg.data.image_name).stem}_result.npy"
        np.save(output_path, merging_result)

        print(f"Pipeline completed successfully! Results saved to {output_path}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
