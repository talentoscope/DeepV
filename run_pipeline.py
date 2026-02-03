from PIL import Image
from torchvision import transforms
from itertools import product
import sys

sys.path.append('/code/Deep-Vectorization-of-Technical-Drawings/')

from vectorization import load_model
from util_files.patchify import patchify
import argparse
import os
import numpy as np
import torch
import logging

from refinement.our_refinement.refinement_for_curves import main as curve_refinement

from merging.merging_for_curves import main as curve_merging
from refinement.our_refinement.refinement_for_lines import render_optimization_hard
from merging.merging_for_lines import postprocess
from cleaning.scripts import run as cleaning_run


import multiprocessing

def serialize(checkpoint):
    model_state_dict = checkpoint['model_state_dict']
    keys = []
    for k in model_state_dict:
        if 'hidden.transformer' in k:
            keys.append(k)

    for k in keys:
        new_key = 'hidden.decoder.transformer' + k[len('hidden.transformer'):]
        model_state_dict[new_key] = model_state_dict[k]
        del model_state_dict[k]
    return checkpoint


def split_to_patches(rgb, patch_size, overlap=0):
    """Separates the input into square patches of specified size.

    :param rgb: input RGB image
    :type rgb: numpy.ndarray
    def process_image_worker(image_np, options, image_name, gpu_index=None):
        # Worker that loads model per-process. If gpu_index is None -> CPU, else use that CUDA device.
        if gpu_index is None:
            device_local = torch.device('cpu')
        else:
            device_local = torch.device(f'cuda:{gpu_index}')
    :param overlap: amount in pixels of how much the patches
                    can overlap with each other (useful for merging)
    :type overlap: int

    :returns patches, patches_offsets, padded_rgb

    rgb = rgb.transpose(1, 2, 0)

    height, width, channels = rgb.shape

    assert patch_size > 0 and 0 <= overlap < patch_size
    patches = patchify(rgb,
                       patch_size=(patch_size, patch_size, channels),
                       step=patch_size - overlap)
    patches = patches.reshape((-1, patch_size, patch_size, channels))
    height_offsets = np.arange(0, height - patch_size, step=patch_size - overlap)
    width_offsets = np.arange(0, width - patch_size, step=patch_size - overlap)
    patches_offsets = np.array(list(
        product(height_offsets, width_offsets)
    ))
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


def read_data(options, image_type='RGB'):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = []
    if options.image_name is None:
        image_names = os.listdir(options.data_dir)
        print(image_names)
        for image_name in image_names:
            if (image_name[-4:] != 'jpeg' and image_name[-3:] != 'png' and image_name[-3:] != 'jpg') or image_name[
                0] == '.':
                print(image_name[-4:])
                continue

            img = train_transform(Image.open(options.data_dir + image_name).convert(image_type))
            print(img.shape)
            img_t = torch.ones(img.shape[0], img.shape[1] + (32 - img.shape[1] % 32),
                               img.shape[2] + (32 - img.shape[2] % 32))
            img_t[:, :img.shape[1], :img.shape[2]] = img
            dataset.append(img_t)
        options.image_name = image_names
    else:
        img = train_transform(Image.open(options.data_dir + options.image_name).convert(image_type))
        print(img)
        print(img.shape)
        img_t = torch.ones(img.shape[0], img.shape[1] + (32 - img.shape[1] % 32),
                           img.shape[2] + (32 - img.shape[2] % 32))
        img_t[:, :img.shape[1], :img.shape[2]] = img
        dataset.append(img_t)
        options.image_name = [options.image_name]
    return dataset


def vector_estimation(patches_rgb, model, device, it, options):
    '''
    :param image:
    :param model:
    :param device:
    :param it:
    :param options:
    :return:
    '''
    model.eval()
    patches_vector = []
    print('--- Preprocessing BEGIN')
    patch_images = preprocess_image(patches_rgb)
    print('--- Preprocessing END')
    for it_batches in range(400, patch_images.shape[0] + 399, 400):
        it_start = it_batches - 400
        if it_batches > patch_images.shape[0]:
            it_batches = patch_images.shape[0]
        with torch.no_grad():
            if (it_start == 0):
                patches_vector = model(patch_images[it_start:it_batches].to(device).float(),
                                       options.model_output_count).detach().cpu().numpy()
            else:
                patches_vector = np.concatenate((patches_vector, model(patch_images[it_start:it_batches].to(device).float(),
                                                                       options.model_output_count).detach().cpu().numpy()),
                                                axis=0)
    patches_vector = torch.tensor(patches_vector) * 64
    return patches_vector


def main(options):
    if len(options.gpu) == 0:
        device = torch.device('cpu')
        prefetch_data = False
    elif len(options.gpu) == 1:
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(options.gpu)
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
        parallel = True

    # configure logging early
    log_level = getattr(options, 'log_level', 'INFO')
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s [%(process)d] %(message)s')

    # normalize and validate output directory early
    options.output_dir = os.path.abspath(options.output_dir)
    if getattr(options, 'save_outputs', True):
        try:
            os.makedirs(options.output_dir, exist_ok=True)
        except Exception as e:
            raise Exception(f"Cannot create or access output_dir '{options.output_dir}': {e}")

    ##reading images
    images = read_data(options, image_type='L')


    ##loading model (for sequential/GPU runs)
    model = None
    if len(options.gpu) > 0:
        model = load_model(options.json_path).to(device)
        checkpoint = serialize(torch.load(options.model_path))
        model.load_state_dict(checkpoint['model_state_dict'])

    # helper for CPU multiprocessing
    def process_image_worker(image_np, options, image_name, gpu_index=None):
        # Worker that loads model per-process. If gpu_index is None -> CPU, else use that CUDA device.
        if gpu_index is None:
            device_local = torch.device('cpu')
        else:
            device_local = torch.device(f'cuda:{gpu_index}')

        logger = logging.getLogger(__name__)
        logger.info(f"Worker start: image={image_name} device={device_local} pid={os.getpid()}")

        model_local = load_model(options.json_path).to(device_local)
        checkpoint_local = serialize(torch.load(options.model_path, map_location='cpu'))
        model_local.load_state_dict(checkpoint_local['model_state_dict'])
        model_local.eval()

        try:
            # image_np is CHW numpy
            img = torch.from_numpy(image_np)
            image_tensor = img.unsqueeze(0)
            orig_chw = image_tensor.numpy()[0]

            # set per-image names so downstream merging saves to correct filenames
            options.image_name = [image_name]
            options.sample_name = image_name

            # optional cleaning on CPU/GPU-local
            if getattr(options, 'use_cleaning', False):
                if not options.cleaning_model_path:
                    raise Exception('--use_cleaning set but --cleaning_model_path not provided')
                cleaning_model_local = torch.load(options.cleaning_model_path, map_location=device_local)
                cleaning_model_local.to(device_local)
                cleaning_model_local.eval()
                rgb = orig_chw.astype(np.float32)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                h, w = rgb.shape[1:]
                pad_h = ((h - 1) // 8 + 1) * 8 - h
                pad_w = ((w - 1) // 8 + 1) * 8 - w
                input_np = np.pad(rgb, [(0, 0), (0, pad_h), (0, pad_w)], mode='constant', constant_values=1)
                input_t = torch.from_numpy(np.ascontiguousarray(input_np[None])).to(device_local).float()
                with torch.no_grad():
                    cleaned, _ = cleaning_model_local(input_t)
                cleaned_np = cleaned[0, 0].cpu().numpy()
                cleaned_np = cleaned_np[:h, :w]
                cleaned_chw = np.expand_dims(cleaned_np, 0)
                patches_rgb, patches_offsets, input_rgb = split_to_patches(cleaned_chw * 255, 64, options.overlap)
            else:
                patches_rgb, patches_offsets, input_rgb = split_to_patches(orig_chw * 255, 64, options.overlap)

            patches_vector = vector_estimation(patches_rgb, model_local, device_local, 0, options)

            if options.primitive_type == "curve":
                intermediate_output = {'options': options, 'patches_offsets': patches_offsets,
                                       'patches_vector': patches_vector,
                                       'cleaned_image_shape': (image_tensor.shape[1], image_tensor.shape[2]),
                                       'patches_rgb': patches_rgb}
                primitives_after_optimization, patch__optim_offsets, repatch_scale, optim_vector_image = curve_refinement(
                    options, intermediate_output, optimization_iters_n=options.diff_render_it)
                merging_result = curve_merging(options, vector_image_from_optimization=optim_vector_image)
            elif options.primitive_type == "line":
                vector_after_opt = render_optimization_hard(patches_rgb, patches_vector, device_local, options, options.image_name[0])
                merging_result, rendered_merged_image = postprocess(vector_after_opt,patches_offsets,input_rgb,image_tensor,0,options)
            else:
                raise ( options.primitive_type+"not implemented, please choose between line or curve")

            return merging_result
        except Exception:
            logger.exception(f"Worker error for image={image_name}")
            raise
        finally:
            logger.info(f"Worker finish: image={image_name} device={device_local} pid={os.getpid()}")

    results = []
    # multiprocessing for CPU-only
    # GPU-aware multiprocessing: assign GPU indices to worker processes when GPUs provided
    if getattr(options, 'workers', 1) > 1 and len(options.gpu) > 0:
        # options.gpu contains GPU indices as strings; normalize to ints
        gpu_indices = [int(g) for g in options.gpu]
        # build args with round-robin GPU assignment across images
        image_args = []
        for i, (img, name) in enumerate(zip(images, options.image_name)):
            gpu_idx = gpu_indices[i % len(gpu_indices)]
            image_args.append((img.cpu().numpy(), options, name, gpu_idx))
        with multiprocessing.Pool(processes=options.workers) as pool:
            results = pool.starmap(process_image_worker, image_args)
        return results

    # CPU-only multiprocessing
    if getattr(options, 'workers', 1) > 1 and len(options.gpu) == 0:
        image_args = [(img.cpu().numpy(), options, name, None) for img, name in zip(images, options.image_name)]
        with multiprocessing.Pool(processes=options.workers) as pool:
            results = pool.starmap(process_image_worker, image_args)
        return results

    # ensure output directories exist when saving is enabled
    if getattr(options, 'save_outputs', True):
        subdirs = [
            'merging_output',
            'iou_postprocess',
            'lines_matching',
            'intermediate_output',
            'after_optimization'
        ]
        for sd in subdirs:
            try:
                os.makedirs(os.path.join(options.output_dir, sd), exist_ok=True)
            except Exception:
                # best-effort: continue if directory cannot be created
                pass

    # sequential processing (uses preloaded model if GPU selected)
    for it, image in enumerate(images):
        image_tensor = image.unsqueeze(0).to(device)
        options.sample_name = options.image_name[it]
        # prepare numpy image in CHW format (channels, H, W)
        orig_chw = image_tensor.cpu().numpy()[0]

        # optional cleaning step: run cleaning model before splitting to patches
        if getattr(options, 'use_cleaning', False):
            if not options.cleaning_model_path:
                raise Exception('--use_cleaning set but --cleaning_model_path not provided')
            cleaning_model = torch.load(options.cleaning_model_path)
            cleaning_model.to(device)
            cleaning_model.eval()

            def _clean_with_model(rgb_chw, cleaning_model, device):
                # rgb_chw: (C, H, W) or (H, W) / (H, W, C)
                if rgb_chw.ndim == 2:
                    rgb_chw = np.expand_dims(rgb_chw, 0)
                if rgb_chw.ndim == 3 and rgb_chw.shape[0] != 3 and rgb_chw.shape[0] != 1:
                    # maybe HWC
                    rgb_hwc = rgb_chw
                    rgb_chw = np.transpose(rgb_hwc, (2, 0, 1))

                # normalize to [0,1]
                rgb = rgb_chw.astype(np.float32)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0

                # pad to divisible by 8 as cleaning expects
                h, w = rgb.shape[1:]
                pad_h = ((h - 1) // 8 + 1) * 8 - h
                pad_w = ((w - 1) // 8 + 1) * 8 - w
                input_np = np.pad(rgb, [(0, 0), (0, pad_h), (0, pad_w)], mode='constant', constant_values=1)

                input_t = torch.from_numpy(np.ascontiguousarray(input_np[None])).to(device).float()
                with torch.no_grad():
                    cleaned, _ = cleaning_model(input_t)
                cleaned_np = cleaned[0, 0].cpu().numpy()  # Hpad x Wpad
                # crop to original size
                cleaned_np = cleaned_np[:h, :w]
                # return as CHW with single channel
                return np.expand_dims(cleaned_np, 0)

            cleaned_chw = _clean_with_model(orig_chw, cleaning_model, device)
            patches_rgb, patches_offsets, input_rgb = split_to_patches(cleaned_chw * 255, 64, options.overlap)
        else:
            # splitting image
            patches_rgb, patches_offsets, input_rgb = split_to_patches(orig_chw * 255, 64, options.overlap)
        patches_vector = vector_estimation(patches_rgb, model, device, it, options)

        if options.primitive_type == "curve":
            intermediate_output = {'options': options, 'patches_offsets': patches_offsets,
                                   'patches_vector': patches_vector,
                                   'cleaned_image_shape': (image_tensor.shape[1], image_tensor.shape[2]),
                                   'patches_rgb': patches_rgb}
            primitives_after_optimization, patch__optim_offsets, repatch_scale, optim_vector_image = curve_refinement(
                options, intermediate_output, optimization_iters_n=options.diff_render_it)
            merging_result = curve_merging(options, vector_image_from_optimization=optim_vector_image)
        elif options.primitive_type == "line":
            vector_after_opt = render_optimization_hard(patches_rgb, patches_vector, device, options, options.image_name[it])
            merging_result, rendered_merged_image = postprocess(vector_after_opt,patches_offsets,input_rgb,image,0,options)
        else:
            raise ( options.primitive_type+"not implemented, please choose between line or curve")

        results.append(merging_result)

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', action='append', help='GPU to use, can use multiple [default: use CPU].')
    parser.add_argument('-c', '--curve_count', type=int, default=10, help='curve count in patch [default: 10]')
    parser.add_argument('--primitive_type', type=str, default="line", help='line or curve')
    parser.add_argument('--output_dir', type=str, default="/logs/outputs/vectorization/lines/", help='dir to folder for output')
    parser.add_argument('--diff_render_it', type=int, default=400, help='iteration count')
    parser.add_argument('--init_random', action='store_true', default=False, dest='init_random',
                        help='init model with random [default: False].')
    parser.add_argument('--rendering_type', type=str, default='hard', help='hard -oleg,simple Alexey')
    parser.add_argument('--data_dir', type=str, default="/data/abc_png_svg/", help='dir to folder for input')
    parser.add_argument('--image_name', type=str, default="00050000_99fd5beca7714bc586260b6a_step_000.png",
                        help='Name of image.If None will perform to all images in '
                             'folder.[default: None]')
    parser.add_argument('--overlap', type=int, default=0, help='overlap in pixel')
    parser.add_argument('--model_output_count', type=int, default=10, help='max_model_output')
    parser.add_argument('--max_angle_to_connect', type=int, default=10, help='max_angle_to_connect in pixel')
    parser.add_argument('--max_distance_to_connect', type=int, default=3, help='max_distance_to_connect in pixel')
    parser.add_argument('--model_path', type=str,
                        default="/logs/models/vectorization/lines/model_lines.weights",
                        help='parth to trained model')
    parser.add_argument('--json_path', type=str,
                        default="/code/Deep-Vectorization-of-Technical-Drawings/vectorization/models/specs/resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json",
                        help='dir to folder for json file for transformer')
    parser.add_argument('--workers', type=int, default=1, help='Number of CPU workers for batch processing (CPU-only).')
    parser.add_argument('--use_cleaning', action='store_true', default=False, help='Run cleaning model before vectorization.')
    parser.add_argument('--cleaning_model_path', type=str, default=None, help='Path to cleaning model file (torch saved model).')
    parser.add_argument('--no-save-outputs', action='store_false', dest='save_outputs', default=True,
                        help='Disable writing merging/refinement outputs to disk (useful for dry runs).')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level: DEBUG, INFO, WARNING, ERROR')
    parser.add_argument('--use_graph_merging', action='store_true', default=False,
                        help='Enable NetworkX-based graph merging in merging stage (fallback to legacy if unavailable).')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    options = parse_args()
    main(options)
