#!/usr/bin/env python3
"""
DeepV cleaning and vectorization pipeline runner.

This script provides an end-to-end pipeline for cleaning and vectorizing
technical drawings using pre-trained models.
"""

import argparse
from itertools import product
from typing import Any, Tuple

import numpy as np
import skimage.io as skio
import torch

from util_files.patchify import patchify


def clean_image(rgb: np.ndarray, cleaning_model: Any) -> Any:
    """Run cleaning operation on the input line drawing,
    leaving clean (non-dirty, non-shaded) and
    full (without gaps) line drawing.

    :param rgb: raw input RGB image
    :type rgb: numpy.ndarray
    :param cleaning_model: an instance of the cleaning model
    :type cleaning_model: callable
    :returns cleaned_rgb -- a cleaner RGB image of the line drawing.
    :rtype cleaned_rgb: numpy.ndarray
    """
    # ensure the channel dimension is present and there's 3 channels
    if len(rgb.shape) < 3:
        rgb = np.repeat(rgb[None], 3, axis=0)

    # ensure the channel dimension is the first one
    if np.argmin(rgb.shape) != 0:
        # assuming it's (h,w,c)
        rgb = rgb.transpose([2, 0, 1])

    # ensure the input is [0,1] data
    rgb = (rgb / rgb.max()).astype(np.float32)

    # cleaning model wants the image dimensions to be divisible by 8
    h, w = rgb.shape[1:]
    pad_h = ((h - 1) // 8 + 1) * 8 - h
    pad_w = ((w - 1) // 8 + 1) * 8 - w
    input_np = np.pad(rgb, [(0, 0), (0, pad_h), (0, pad_w)], mode="constant", constant_values=1)

    input = torch.from_numpy(np.ascontiguousarray(input_np[None])).cuda()
    cleaned, _ = cleaning_model(input)
    result = cleaned[0, 0].cpu().detach().numpy()[..., None].astype(np.float32)
    return result


def split_to_patches(rgb: np.ndarray, patch_size: int, overlap: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Separates the input into square patches of specified size.

    :param rgb: input RGB image
    :type rgb: numpy.ndarray
    :param patch_size: size of patches in pixels (assuming
                        square patches)
    :type patch_size: int
    :param overlap: amount in pixels of how much the patches
                    can overlap with each other (useful for merging)
    :type overlap: int

    :returns patches, patches_offsets
    :rtype Tuple[numpy.ndarray, numpy.ndarray]
    """
    # TODO @artonson: add correct handling of rightmost patches (currently ignored)
    height, width, channels = rgb.shape
    assert patch_size > 0 and 0 <= overlap < patch_size
    patches = patchify(rgb, patch_size=(patch_size, patch_size, channels), step=patch_size - overlap)
    patches = patches.reshape((-1, patch_size, patch_size))
    height_offsets = np.arange(0, height, step=patch_size - overlap)
    width_offsets = np.arange(0, width, step=patch_size - overlap)
    patches_offsets = np.array(list(product(height_offsets, width_offsets)))
    return patches, patches_offsets


def save_output(output_vector: Any, output_filename: str) -> None:
    """Save vectorized output to file.

    :param output_vector: vectorized primitives to save
    :param output_filename: path to output file
    """
    with open(output_filename, "w") as output_file:
        for primitive in output_vector:
            primitive.write(output_file)


def vectorize(rgb: np.ndarray, vector_model: Any) -> Any:
    """Vectorize RGB image using the provided model.

    :param rgb: input RGB image
    :param vector_model: vectorization model
    :return: vectorized output
    """
    # TODO: Implement vectorization logic
    pass


def assemble_vector_patches(patches_vector: Any, patches_offsets: np.ndarray) -> Any:
    """Assemble vectorized patches into final output.

    :param patches_vector: list of vectorized patches
    :param patches_offsets: offsets for each patch
    :return: assembled vector output
    """
    for patch_vector, patch_offset in zip(patches_vector, patches_offsets):
        for primitive in patch_vector:
            primitive.offset(patch_offset)
    # TODO: Implement assembly logic
    pass


def load_vector_model(model_path: str) -> Any:
    """Load vectorization model from file.

    :param model_path: path to model file
    :return: loaded model
    """
    # TODO: Implement proper model loading
    return torch.load(model_path)


def main(options: argparse.Namespace) -> None:
    """Main function for running the cleaning and vectorization pipeline."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")

    input_rgb = skio.imread(options.input_filename)

    cleaned_rgb = input_rgb
    if options.use_cleaning:
        cleaning_model = torch.load(options.cleaning_model_filename)
        cleaning_model.eval()
        cleaned_rgb = clean_image(cleaned_rgb, cleaning_model)
        if options.cleaned_filename:
            skio.imsave(options.cleaned_filename, cleaned_rgb)

    if options.vectorize:
        vector_model = load_vector_model(options.vector_model_filename)
        if options.use_patches:
            patches_rgb, patches_offsets = split_to_patches(cleaned_rgb, options.patch_size)
            # patches_vector = []
            # for patch_idx, patch_rgb in enumerate(patches_rgb):
            #    patch_vector = vectorize(patch_rgb, vector_model)
            #    if options.vector_patch_path:
            #        patch_output_filename = \
            #            os.path.join(options.vector_patch_path,
            #                         f'patch_{patch_idx:02d}.svg')
            #        save_output(patch_vector, patch_output_filename)

            #    patches_vector.append(patch_vector)
            # output_vector = assemble_vector_patches(patches_vector, patches_offsets)
        else:
            output_vector = vectorize(cleaned_rgb, vector_model)

    if options.output_filename:
        save_output(output_vector, options.output_filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use [default: GPU 0].")
    parser.add_argument(
        "-np",
        "--no-patches",
        action="store_false",
        dest="use_patches",
        default=True,
        help="Set to disable vectorization via patches [default: use patches].",
    )
    parser.add_argument(
        "-s", "--patch-size", type=int, dest="patch_size", default=64, help="Patch size in pixels [default: 64]."
    )

    parser.add_argument(
        "-nc",
        "--no-cleaning",
        action="store_false",
        dest="use_cleaning",
        default=True,
        help="Set to disable cleaning [default: use cleaning].",
    )
    parser.add_argument(
        "-c",
        "--cleaning-model-file",
        dest="cleaning_model_filename",
        help="Path to cleaning model file [default: none].",
    )

    parser.add_argument(
        "-nv",
        "--no-vectorization",
        action="store_false",
        dest="vectorize",
        default=True,
        help="Set to disable vectorization [default: vectorize].",
    )
    parser.add_argument(
        "-v",
        "--vector-model-file",
        dest="vector_model_filename",
        help="Path to vectorization model file [default: none].",
    )

    parser.add_argument(
        "-i", "--input-file", required=True, dest="input_filename", help="Path to input image file [default: none]."
    )
    parser.add_argument(
        "-oc",
        "--cleaned-file",
        dest="cleaned_filename",
        help="Path to cleaned image file [default: none, meaning don't save the file].",
    )
    parser.add_argument(
        "-op",
        "--patch-output-path",
        dest="patch_output_filename",
        help="Path to directory containing vectorized patches [default: none, meaning don't save the files].",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=False,
        dest="output_filename",
        help="Path to input vector SVG file [default: none].",
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    main(options)
