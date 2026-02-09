#!/usr/bin/env python3
"""CLI to run dataset processors.

Example:
  python -m dataset.run_processor --dataset floorplancad \
    --input-dir ./data/raw/floorplancad --output-base ./data --dry-run
"""

import argparse
import json
from pathlib import Path

from .processors import get_processor


def main():
    parser = argparse.ArgumentParser(description="Run dataset processor to standardize raw downloads")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., floorplancad)")
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Raw dataset directory (e.g., data/raw/floorplancad)"
    )
    parser.add_argument(
        "--output-base", type=Path, default=Path("./data"), help="Base output dir for vector/raster folders"
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not perform filesystem changes")

    args = parser.parse_args()

    proc = get_processor(args.dataset)
    res = proc.standardize(args.input_dir, args.output_base, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
