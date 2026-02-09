#!/usr/bin/env python3
"""
FloorPlanCAD Dataset File Lister

Lists all files in the FloorPlanCAD raw dataset directory for verification and
inventory purposes. Useful for checking download completeness and dataset structure.

Outputs:
- TOTAL_FILES: Total count of files in dataset
- First 50 file paths for inspection
- MISSING: If dataset directory doesn't exist

Used for:
- Download verification
- Dataset inventory
- Debugging data loading issues

Usage:
    python scripts/list_floorplancad_files.py
"""

from pathlib import Path

p = Path("data/raw/floorplancad")
if not p.exists():
    print("MISSING")
    raise SystemExit(1)
files = list(p.rglob("*"))
files = [f for f in files if f.is_file()]
print("TOTAL_FILES", len(files))
# show first 50
for f in files[:50]:
    print(f)
