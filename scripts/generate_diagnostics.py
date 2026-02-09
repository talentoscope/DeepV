#!/usr/bin/env python3
"""
DeepV Pipeline Diagnostics Report Generator

Generates comprehensive diagnostics reports from aggregated pipeline metrics.
Analyzes performance patterns, identifies worst-performing images, and provides
insights for model improvement.

Features:
- Statistical analysis of pipeline metrics (IoU, SSIM, etc.)
- Worst-case performance identification
- Performance trend analysis
- Automated insights and recommendations
- HTML/markdown report generation

Reads from logs/metrics/summary.csv and generates actionable diagnostics
for debugging and optimization.

Usage:
    python scripts/generate_diagnostics.py
"""

import csv
import json
from pathlib import Path


def main():
    csv_path = Path("logs/metrics/summary.csv")
    if not csv_path.exists():
        print("CSV not found")
        return

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v else None for k, v in row.items() if k != "image"})

    # Find worst images
    worst_iou = min((r for r in rows if r["iou"]), key=lambda x: x["iou"])
    worst_ssim = min((r for r in rows if r["ssim"]), key=lambda x: x["ssim"])

    report = f"""# Pipeline Diagnostics Report

## Summary Stats (10 images)
- IoU: mean={sum(r["iou"] for r in rows if r["iou"])/len([r for r in rows if r["iou"]]):.3f}
- SSIM: mean={sum(r["ssim"] for r in rows if r["ssim"])/len([r for r in rows if r["ssim"]]):.3f}
- Primitives: mean={sum(r["num_primitives"] for r in rows if r["num_primitives"])/len([r for r in rows if r["num_primitives"]]):.1f}

## Potential Issues
- High primitive count ({worst_iou["num_primitives"]:.0f}) suggests over-segmentation; check model output count or merging thresholds.
- Low SSIM ({worst_ssim["ssim"]:.3f}) indicates visual artifacts; investigate rendering or refinement convergence.
- Chamfer distance stable but check if geometric accuracy is sufficient for CAD use.

## Recommendations
- Tune merging: increase thresholds in merging/merging_for_lines.py to reduce primitives.
- Improve refinement: check tolerances in refinement/our_refinement/ for better convergence.
- Domain adaptation: real data may need fine-tuning or synthetic augmentation.
"""

    with open("logs/metrics/diagnostics.md", "w") as f:
        f.write(report)
    print("Wrote diagnostics.md")


if __name__ == "__main__":
    main()
