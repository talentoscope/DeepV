#!/usr/bin/env python3
"""
DeepV Metrics Aggregation Script

Aggregates per-image JSON metric summaries into CSV format and computes
basic statistics for key performance metrics across multiple evaluations.

Features:
- JSON to CSV conversion for analysis
- Statistical summaries (mean, median, std dev)
- Missing value handling
- Sorted output for easy analysis

Combines individual image metrics into dataset-level summaries for
model evaluation and comparison.

Usage:
    python scripts/aggregate_metrics.py --metrics_dir logs/outputs/batch_run/metrics \
        --out_csv logs/metrics/summary.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", required=True)
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def load_jsons(metrics_dir: Path):
    files = sorted(metrics_dir.glob("*_summary.json"))
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        # pick common fields if present
        row = {
            "image": data.get("image_name") or f.stem.replace("_summary", ""),
            "iou": data.get("geometric", {}).get("iou"),
            "ssim": data.get("visual", {}).get("ssim"),
            "psnr": data.get("visual", {}).get("psnr"),
            "chamfer": data.get("geometric", {}).get("chamfer_distance"),
            "num_primitives": data.get("structural", {}).get("primitive_count"),
        }
        rows.append(row)
    return rows


def write_csv(rows, out_csv: Path):
    keys = ["image", "iou", "ssim", "psnr", "chamfer", "num_primitives"]
    with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_stats(rows):
    metrics = ["iou", "ssim", "psnr", "chamfer", "num_primitives"]
    print("Aggregated stats:")
    for m in metrics:
        vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
        if not vals:
            print(f" - {m}: no numeric values")
            continue
        mean = statistics.mean(vals)
        med = statistics.median(vals)
        stdev = statistics.pstdev(vals)
        print(f" - {m}: mean={mean:.4f}, median={med:.4f}, stdev={stdev:.4f} "
              f"(n={len(vals)})")


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_csv = Path(args.out_csv)
    rows = load_jsons(metrics_dir)
    if not rows:
        print(f"No metric JSONs found in {metrics_dir}")
        return
    write_csv(rows, out_csv)
    print(f"Wrote CSV: {out_csv}")
    print_stats(rows)


if __name__ == "__main__":
    main()
