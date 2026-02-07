#!/usr/bin/env python3
"""
Run the project's `run_pipeline.py` on a batch of images (samples random images
or reads a list), enforce GPU usage, and run the analysis wrapper for each
output. Writes per-image logs and metric JSONs into the specified output root.

Usage (example):
  python scripts/run_batch_pipeline.py --data_dir data/raster/floorplancad/test \
      --random_count 10 --output_root logs/outputs/batch_run --model_path models/model_lines.weights

This script shells out to `run_pipeline.py` to keep runtime behaviour identical
to the CLI entrypoint and avoid deep imports.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Directory with input images")
    p.add_argument("--images_file", help="File with newline-separated image filenames (optional)")
    p.add_argument("--random_count", type=int, default=10, help="Number of random images to sample if images_file not provided")
    p.add_argument("--output_root", default="logs/outputs/batch_run", help="Root output directory")
    p.add_argument("--model_path", default="models/model_lines.weights", help="Path to model checkpoint")
    p.add_argument("--json_path", default="vectorization/models/specs/resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json", help="Model json spec path")
    p.add_argument("--primitive_type", default="line", choices=("line","curve"))
    p.add_argument("--model_output_count", type=int, default=10)
    p.add_argument("--enforce_gpu", action="store_true", help="Require CUDA GPU and exit if unavailable")
    return p.parse_args()


def ensure_gpu(enforce_gpu: bool):
    if not enforce_gpu:
        return
    try:
        import torch
    except Exception:
        print("ERROR: PyTorch not available but --enforce_gpu was passed. Exiting.")
        sys.exit(2)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available but --enforce_gpu was passed. Exiting.")
        sys.exit(3)


def pick_images(data_dir: Path, images_file: str, random_count: int):
    imgs = []
    if images_file:
        with open(images_file, "r", encoding="utf-8") as f:
            imgs = [l.strip() for l in f if l.strip()]
    else:
        all_imgs = sorted([p.name for p in data_dir.glob("*.png")])
        if len(all_imgs) == 0:
            raise SystemExit(f"No PNG images found in {data_dir}")
        random.seed(42)
        imgs = random.sample(all_imgs, min(random_count, len(all_imgs)))
    return imgs


def run_for_image(repo_root: Path, data_dir: Path, img_name: str, output_root: Path, args):
    base = Path(img_name).stem
    output_dir = output_root / base
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_pipeline.log"

    cmd = [sys.executable, str(repo_root / "run_pipeline.py"),
           "--data_dir", str(data_dir),
           "--image_name", img_name,
           "--output_dir", str(output_dir),
           "--primitive_type", args.primitive_type,
           "--model_path", args.model_path,
           "--json_path", args.json_path,
           "--model_output_count", str(args.model_output_count)]

    print(f"Running pipeline for {img_name} -> {output_dir}")
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    duration = time.time() - start
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        f.write(f"\n# Exit code: {proc.returncode}\n# Duration: {duration:.2f}s\n")

    return proc.returncode, duration, output_dir


def run_analysis(repo_root: Path, output_dir: Path, original_path: Path, metrics_dir: Path):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir.name
    out_json = metrics_dir / f"{base}_summary.json"
    cmd = [sys.executable, str(repo_root / "scripts" / "analyze_outputs.py"),
           "--output_dir", str(output_dir),
           "--original", str(original_path),
           "--out", str(out_json)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # write analyzer stdout into the output_dir for reference
    with open(output_dir / "analyze_outputs.log", "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        f.write(f"\n# Exit code: {proc.returncode}\n")
    return proc.returncode, out_json


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    metrics_dir = output_root / "metrics"

    ensure_gpu(args.enforce_gpu)

    imgs = pick_images(data_dir, args.images_file, args.random_count)
    results = []
    for img in imgs:
        code, dur, out_dir = run_for_image(repo_root, data_dir, img, output_root, args)
        # attempt analysis; don't fail the whole run on analyzer failure
        try:
            a_code, a_json = run_analysis(repo_root, out_dir, data_dir / img, metrics_dir)
        except Exception as e:
            print(f"Analysis failed for {img}: {e}")
            a_code = -1
            a_json = None
        results.append({"image": img, "exit_code": code, "duration_s": dur, "analysis_exit": a_code, "analysis_json": str(a_json) if a_json else None})

    summary_path = output_root / "batch_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"Batch run complete. Summary: {summary_path}")


if __name__ == '__main__':
    main()
