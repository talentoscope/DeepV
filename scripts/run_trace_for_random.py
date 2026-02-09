#!/usr/bin/env python3
import random
import subprocess
import sys
from pathlib import Path

p = Path("data/raster/floorplancad/test")
files = [f.name for f in p.iterdir() if f.suffix.lower() == ".png"]
if not files:
    print("No test PNGs found in", p)
    sys.exit(1)
choice = random.choice(files)
print("Selected image:", choice)

cmd = [
    sys.executable,
    "run_pipeline.py",
    "--gpu",
    "0",
    "--data_dir",
    "data/raster/floorplancad/test",
    "--image_name",
    choice,
    "--primitive_type",
    "line",
    "--model_path",
    "models/model_lines.weights",
    "--trace",
    "--trace_dir",
    "./output/traces",
    "--output_dir",
    "logs/outputs/vectorization/lines/",
]
print("Running pipeline:", " ".join(cmd))
subprocess.run(cmd, check=True)

report_cmd = [sys.executable, "scripts/generate_trace_report.py", "--trace_dir", f"./output/traces/{choice}"]
print("Generating report:", " ".join(report_cmd))
subprocess.run(report_cmd, check=True)

print("Done. Report at:", f"output/traces/{choice}/report.html")
