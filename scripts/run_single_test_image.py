#!/usr/bin/env python3
"""Run the DeepV pipeline on a single test image (data/raw/test.png by default).

This helper constructs the options namespace expected by `run_pipeline.PipelineRunner`
and runs the pipeline. It is intended for quick single-image checks.
"""
import argparse
import sys
import os
import torch

sys.path.append(".")

from run_pipeline import PipelineRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline on a single image")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory containing the image")
    parser.add_argument("--image_name", type=str, default="test.png", help="Image filename to process")
    parser.add_argument("--primitive_type", type=str, default="line", help="line or curve")
    parser.add_argument("--model_path", type=str, help="Path to trained model (auto-selected if not provided)")
    parser.add_argument(
        "--json_path",
        type=str,
        default=(
            "vectorization/models/specs/"
            "resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json"
        ),
        help="Path to JSON model specification file",
    )
    parser.add_argument("--gpu", action="append", help="GPU id to use (can be repeated)")
    parser.add_argument("--init_random", action="store_true", help="Init model with random weights")
    parser.add_argument("--output_dir", type=str, default="logs/outputs/single_test/", help="Output dir")
    parser.add_argument("--diff_render_it", type=int, default=400, help="Differentiable render iterations")
    parser.add_argument("--model_output_count", type=int, default=10, help="Max primitives per patch")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels between patches")
    parser.add_argument("--curve_count", type=int, default=10, help="curve count in patch")
    parser.add_argument("--rendering_type", type=str, default="bezier_splatting", help="Rendering type")
    parser.add_argument("--max_angle_to_connect", type=int, default=10, help="max angle to connect")
    parser.add_argument("--max_distance_to_connect", type=int, default=3, help="max distance to connect")

    return parser.parse_args()


def build_options(args):
    # Auto-select model path based on primitive type if not provided
    if args.model_path is None:
        if args.primitive_type == "line":
            args.model_path = "models/model_lines.weights"
        elif args.primitive_type == "curve":
            args.model_path = "models/model_curves.weights"
        else:
            raise ValueError(f"Unknown primitive type: {args.primitive_type}")

    # Build a simple Namespace compatible with run_pipeline.parse_args output
    opts = argparse.Namespace()
    opts.gpu = args.gpu or []
    opts.curve_count = args.curve_count
    opts.primitive_type = args.primitive_type
    opts.output_dir = args.output_dir
    opts.diff_render_it = args.diff_render_it
    opts.init_random = args.init_random
    opts.rendering_type = args.rendering_type
    opts.data_dir = args.data_dir
    opts.image_name = args.image_name
    opts.overlap = args.overlap
    opts.model_output_count = args.model_output_count
    opts.max_angle_to_connect = args.max_angle_to_connect
    opts.max_distance_to_connect = args.max_distance_to_connect
    opts.model_path = args.model_path
    opts.json_path = args.json_path
    return opts


def main():
    args = parse_args()
    print(f"Running pipeline on: {os.path.join(args.data_dir, args.image_name)}")

    # Enforce GPU usage for this runner
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for DeepV but CUDA is not available on this machine.")

    options = build_options(args)
    runner = PipelineRunner(options)
    runner.run()


if __name__ == "__main__":
    main()
