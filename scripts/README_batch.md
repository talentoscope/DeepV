Batch-run instructions
======================

This directory contains helper scripts to run the DeepV pipeline on a sampled
set of raster images and to aggregate metrics.

1) Sample & run 10 images (requires GPU by default if `--enforce_gpu` passed)

PowerShell example:

```powershell
# Run 10 random images from the FloorPlanCAD test raster folder
python scripts/run_batch_pipeline.py --data_dir data/raster/floorplancad/test --random_count 10 --output_root logs/outputs/batch_run --model_path models/model_lines.weights --enforce_gpu
```

This will create per-image directories under `logs/outputs/batch_run/<image_base>/`, logs and `analyze_outputs` results will be placed in `logs/outputs/batch_run/metrics/`.

2) Aggregate metrics into CSV

```powershell
python scripts/aggregate_metrics.py --metrics_dir logs/outputs/batch_run/metrics --out_csv logs/metrics/summary.csv
```

3) Generate diagnostics report

```powershell
python scripts/generate_diagnostics.py
```

Notes
- The batch runner shells out to `run_pipeline.py` and `scripts/analyze_outputs.py` to keep behavior consistent with project CLI.
- `--enforce_gpu` will cause the script to exit if CUDA is not available. Remove it to allow CPU runs (very slow).
- Metrics include IoU, SSIM, PSNR, Chamfer, primitive counts.
- Diagnostics highlights over-segmentation and visual quality issues as key improvement areas.
