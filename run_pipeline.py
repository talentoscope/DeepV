import sys
sys.path.append(".")  # Add current directory to path for local imports

import argparse
import os
from itertools import product

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from merging.merging_for_curves import main as curve_merging
from merging.merging_for_lines import postprocess
from refinement.our_refinement.refinement_for_curves import main as curve_refinement
from refinement.our_refinement.refinement_for_lines import render_optimization_hard
from util_files.patchify import patchify
from vectorization import load_model
from analysis.tracing import Tracer


def serialize(checkpoint):
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
    :param patch_size: size of patches in pixels (assuming
                        square patches)
    :type patch_size: int
    :param overlap: amount in pixels of how much the patches
                    can overlap with each other (useful for merging)
    :type overlap: int

    :returns patches, patches_offsets
    :rtype Tuple[numpy.ndarray, numpy.ndarray]
    """
    rgb = rgb.transpose(1, 2, 0)
    rgb_t = np.ones((rgb.shape[0] + 33, rgb.shape[1] + 33, rgb.shape[2])) * 255.0
    rgb_t[: rgb.shape[0], : rgb.shape[1], :] = rgb
    rgb = rgb_t

    height, width, channels = rgb.shape

    assert patch_size > 0 and 0 <= overlap < patch_size
    patches = patchify(rgb, patch_size=(patch_size, patch_size, channels), step=patch_size - overlap)
    patches = patches.reshape((-1, patch_size, patch_size, channels))
    height_offsets = np.arange(0, height - patch_size, step=patch_size - overlap)
    width_offsets = np.arange(0, width - patch_size, step=patch_size - overlap)
    patches_offsets = np.array(list(product(height_offsets, width_offsets)))
    return patches, patches_offsets, rgb


def preprocess_image(image):
    patch_height, patch_width = image.shape[1:3]
    image = torch.as_tensor(image).type(torch.float32).reshape(-1, patch_height, patch_width) / 255
    image = 1 - image  # 0 -- background
    mask = (image > 0).type(torch.float32)
    _xs = np.arange(1, patch_width + 1, dtype=np.float32)[None].repeat(patch_height, 0) / patch_width
    _ys = np.arange(1, patch_height + 1, dtype=np.float32)[..., None].repeat(patch_width, 1) / patch_height
    _xs = torch.from_numpy(_xs)[None]
    _ys = torch.from_numpy(_ys)[None]
    return torch.stack([image, _xs * mask, _ys * mask], dim=1)


def read_data(options, image_type="RGB"):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = []
    if options.image_name is None:
        image_names = os.listdir(options.data_dir)
        print(image_names)
        for image_name in image_names:
            if (image_name[-4:] != "jpeg" and image_name[-3:] != "png" and image_name[-3:] != "jpg") or image_name[
                0
            ] == ".":
                print(image_name[-4:])
                continue

            img = train_transform(Image.open(os.path.join(options.data_dir, image_name)).convert(image_type))
            print(img.shape)
            img_t = torch.ones(
                img.shape[0], img.shape[1] + (32 - img.shape[1] % 32), img.shape[2] + (32 - img.shape[2] % 32)
            )
            img_t[:, : img.shape[1], : img.shape[2]] = img
            dataset.append(img_t)
        options.image_name = image_names
    else:
        img = train_transform(Image.open(os.path.join(options.data_dir, options.image_name)).convert(image_type))
        print(img)
        print(img.shape)
        img_t = torch.ones(
            img.shape[0], img.shape[1] + (32 - img.shape[1] % 32), img.shape[2] + (32 - img.shape[2] % 32)
        )
        img_t[:, : img.shape[1], : img.shape[2]] = img
        dataset.append(img_t)
        options.image_name = [options.image_name]
    return dataset


def vector_estimation(patches_rgb, model, device, it, options):
    """
    :param image:
    :param model:
    :param device:
    :param it:
    :param options:
    :return:
    """
    model.eval()
    patches_vector = []
    print("--- Preprocessing BEGIN")
    patch_images = preprocess_image(patches_rgb)
    print("--- Preprocessing END")
    for it_batches in range(400, patch_images.shape[0] + 399, 400):
        it_start = it_batches - 400
        if it_batches > patch_images.shape[0]:
            it_batches = patch_images.shape[0]
        with torch.no_grad():
            # Check if model supports variable length output
            if hasattr(model.hidden, 'max_primitives'):
                # Variable length model - no need for model_output_count
                batch_output = model(patch_images[it_start:it_batches].to(device).float()).detach().cpu().numpy()
            else:
                # Fixed length model - use model_output_count
                batch_output = model(
                    patch_images[it_start:it_batches].to(device).float(),
                    options.model_output_count
                ).detach().cpu().numpy()

            if it_start == 0:
                patches_vector = batch_output
            else:
                patches_vector = np.concatenate((patches_vector, batch_output), axis=0)
    patches_vector = torch.tensor(patches_vector) * 64
    return patches_vector


class PipelineRunner:
    """Handles the execution of the DeepV vectorization pipeline."""

    def __init__(self, options):
        self.options = options
        self.device = self._setup_device()
        self.model = None

    def _setup_device(self):
        """Set up the appropriate compute device."""
        # Allow CPU fallback for testing
        if torch.cuda.is_available():
            # Use provided GPU ids if given, otherwise default to cuda:0
            if self.options.gpu is None or len(self.options.gpu) == 0:
                # Prefer the first visible CUDA device
                return torch.device("cuda:0")
            elif len(self.options.gpu) == 1:
                return torch.device(f"cuda:{self.options.gpu[0]}")
            else:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.options.gpu)
                return torch.device(f"cuda:{self.options.gpu[0]}")
        else:
            # Fallback to CPU for testing
            print("Warning: CUDA not available, using CPU. Performance will be slow.")
            return torch.device("cpu")

    def _load_model(self):
        """Load and initialize the ML model."""
        self.model = load_model(self.options.json_path).to(self.device)
        if not self.options.init_random:
            checkpoint = serialize(torch.load(self.options.model_path, map_location=self.device))
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def _process_images(self):
        """Process all images through the pipeline."""
        images = read_data(self.options, image_type="L")

        for it, image in enumerate(images):
            self._process_single_image(image, it)

    def _process_single_image(self, image, image_index):
        """Process a single image through the vectorization pipeline."""
        image_tensor = image.unsqueeze(0).to(self.device)
        self.options.sample_name = self.options.image_name[image_index]

        # Set RNG seeds for determinism if tracing
        trace_enabled = getattr(self.options, "trace", False)
        seed = 42  # Fixed seed for reproducibility in trace mode
        if trace_enabled:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Split image into patches
        patches_rgb, patches_offsets, input_rgb = split_to_patches(
            image_tensor.cpu().numpy()[0] * 255, 64, self.options.overlap
        )

        # Initialize tracer for this image (if enabled)
        tracer = Tracer(
            enabled=trace_enabled,
            base_dir=getattr(self.options, "trace_dir", "output/traces"),
            image_id=self.options.image_name[image_index],
            seed=seed if trace_enabled else None,
            device=str(self.device)
        )

        # Save per-patch images for inspection
        if tracer.enabled:
            for pidx in range(patches_rgb.shape[0]):
                try:
                    offset = patches_offsets[pidx].tolist() if pidx < patches_offsets.shape[0] else None
                except Exception:
                    offset = None
                tracer.save_patch(pidx, patches_rgb[pidx].astype(np.uint8), offset=offset)

        # Run vector estimation
        patches_vector = vector_estimation(patches_rgb, self.model, self.device, image_index, self.options)

        # Save raw model outputs per-patch (compact)
        try:
            patches_vector_np = patches_vector.detach().cpu().numpy() if hasattr(patches_vector, 'detach') else np.asarray(patches_vector)
        except Exception:
            patches_vector_np = np.asarray(patches_vector)
        if tracer.enabled:
            for pidx in range(patches_vector_np.shape[0]):
                tracer.save_model_output(pidx, {"vector": patches_vector_np[pidx]})

        # Save a lightweight pre-refinement assembly for debugging
        if tracer.enabled:
            prelist = []
            for pidx in range(patches_vector_np.shape[0]):
                prelist.append({"patch_id": int(pidx), "vector": patches_vector_np[pidx].tolist()})
            tracer.save_pre_refinement(prelist)

        # Run refinement and merging based on primitive type
        if self.options.primitive_type == "curve":
            self._process_curve_pipeline(patches_rgb, patches_vector, patches_offsets, input_rgb)
        elif self.options.primitive_type == "line":
            self._process_line_pipeline(patches_rgb, patches_vector, patches_offsets, input_rgb, image, tracer)
        else:
            raise ValueError(f"Unsupported primitive type: {self.options.primitive_type}")

    def _process_curve_pipeline(self, patches_rgb, patches_vector, patches_offsets, input_rgb):
        """Process image using curve primitives pipeline."""
        intermediate_output = {
            "options": self.options,
            "patches_offsets": patches_offsets,
            "patches_vector": patches_vector,
            "cleaned_image_shape": (patches_rgb.shape[1], patches_rgb.shape[2]),
            "patches_rgb": patches_rgb,
        }

        primitives_after_optimization, patch__optim_offsets, repatch_scale, optim_vector_image = curve_refinement(
            self.options, intermediate_output, optimization_iters_n=self.options.diff_render_it
        )

        merging_result = curve_merging(self.options, vector_image_from_optimization=optim_vector_image)
        return merging_result

    def _process_line_pipeline(self, patches_rgb, patches_vector, patches_offsets, input_rgb, image, tracer):
        """Process image using line primitives pipeline."""
        # Basic validation and logging
        print(f"[Pipeline] patches_rgb.shape={patches_rgb.shape}, patches_vector.shape={patches_vector.shape}")
        if patches_vector is None or patches_vector.shape[0] == 0:
            print("[Pipeline][Warning] Empty patches_vector received; skipping optimization")

        try:
            vector_after_opt = render_optimization_hard(
                patches_rgb, patches_vector, self.device, self.options, self.options.image_name[0]
            )
        except Exception as e:
            print(f"[Pipeline][Error] render_optimization_hard failed: {e}")
            raise

        # Save post-refinement primitives
        try:
            if tracer.enabled:
                try:
                    va_np = vector_after_opt.detach().cpu().numpy() if hasattr(vector_after_opt, 'detach') else np.asarray(vector_after_opt)
                except Exception:
                    va_np = np.asarray(vector_after_opt)
                postlist = [{"primitive_idx": int(i), "vector": va_np[i].tolist()} for i in range(va_np.shape[0])]
                tracer.save_post_refinement(postlist)
        except Exception as _:
            pass

        try:
            merging_result, rendered_merged_image = postprocess(
                vector_after_opt, patches_offsets, input_rgb, image, 0, self.options
            )
        except Exception as e:
            print(f"[Pipeline][Error] postprocess failed: {e}")
            raise

        # Save merge snapshot / basic trace
        if tracer.enabled:
            try:
                # merging_result may be array-like
                mr = np.asarray(merging_result)
                tracer.save_merge_trace({"num_primitives": int(mr.shape[0]) if mr.ndim > 0 else 0})
                tracer.save_provenance({"final_count": int(mr.shape[0]) if mr.ndim > 0 else 0})
            except Exception:
                pass

        # Ensure output directory exists and save results for inspection
        try:
            os.makedirs(self.options.output_dir, exist_ok=True)
            final_dir = os.path.join(self.options.output_dir, "final_renders")
            os.makedirs(final_dir, exist_ok=True)
            # Save rendered merged image (numpy array) if available
            if rendered_merged_image is not None:
                try:
                    img = Image.fromarray(rendered_merged_image)
                    out_path = os.path.join(final_dir, f"{self.options.image_name[0]}")
                    img.save(out_path)
                    print(f"[Pipeline] Saved rendered merged image to {out_path}")
                except Exception as e:
                    print(f"[Pipeline][Warning] Failed to save rendered image: {e}")

            # Save merged vectors as numpy for later inspection
            try:
                vectors_out = os.path.join(self.options.output_dir, f"{self.options.image_name[0]}.npy")
                np.save(vectors_out, merging_result)
                print(f"[Pipeline] Saved merged vectors to {vectors_out}")
            except Exception as e:
                print(f"[Pipeline][Warning] Failed to save merged vectors: {e}")
        except Exception:
            # Non-fatal; we've already produced the result
            pass

        return merging_result

    def run(self):
        """Execute the complete pipeline."""
        self._load_model()
        self._process_images()


def main(options):
    """Main entry point for the pipeline."""
    runner = PipelineRunner(options)
    return runner.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", action="append", help="GPU to use, can use multiple [default: use CPU].")
    parser.add_argument("-c", "--curve_count", type=int, default=10, help="curve count in patch [default: 10]")
    parser.add_argument("--primitive_type", type=str, default="line", help="line or curve")
    parser.add_argument(
        "--output_dir", type=str, default="logs/outputs/vectorization/lines/", help="dir to folder for output"
    )
    parser.add_argument("--diff_render_it", type=int, default=400, help="iteration count")
    parser.add_argument(
        "--init_random",
        action="store_true",
        default=False,
        dest="init_random",
        help="init model with random [default: False].",
    )
    parser.add_argument("--rendering_type", type=str, default="bezier_splatting", help="hard - analytical rendering, bezier_splatting - fast differentiable rendering")
    parser.add_argument("--data_dir", type=str, default="/data/synthetic/", help="dir to folder for input")
    parser.add_argument(
        "--image_name",
        type=str,
        default="00050000_99fd5beca7714bc586260b6a_step_000.png",
        help="Name of image. If None, will perform on all images in folder.",
    )
    parser.add_argument("--overlap", type=int, default=0, help="overlap in pixel")
    parser.add_argument("--model_output_count", type=int, default=10, help="max_model_output")
    parser.add_argument("--max_angle_to_connect", type=int, default=10, help="max_angle_to_connect in pixel")
    parser.add_argument("--max_distance_to_connect", type=int, default=3, help="max_distance_to_connect in pixel")
    parser.add_argument("--trace", action="store_true", default=False, help="Enable per-step tracing outputs for analysis")
    parser.add_argument("--trace_dir", type=str, default="output/traces", help="Directory to write trace artifacts")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model_lines.weights",
        help="Path to trained model",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=("vectorization/models/specs/"
                 "resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json"),
        help="Path to JSON model specification file",
    )
    options = parser.parse_args()

    return options


if __name__ == "__main__":
    options = parse_args()
    main(options)
