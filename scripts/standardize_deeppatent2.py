#!/usr/bin/env python3
"""Copy DeepPatent2 'Original' files into the standardized folder.

Features:
- Preserves relative paths under the source 'Original' folder.
- Skips existing files by default (use --overwrite to replace).
- Supports --dry-run to show what would be copied without doing it.
- Uses a ThreadPoolExecutor for parallel copying and a tqdm progress bar.

Example:
    python scripts/standardize_deeppatent2.py \
      --source "E:\\\\dv\\\\DeepV\\\\data\\\\raw\\\\deeppatent2\\\\Original" \
      --dest "E:\\\\dv\\\\DeepV\\\\data\\\\standardized" --dry-run
"""

from pathlib import Path
import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def copy_file(src: Path, dst: Path, overwrite: bool = False, dry_run: bool = False):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            return (src, dst, 'skipped')
        if dry_run:
            return (src, dst, 'dry')
        # Use copy2 to preserve metadata
        shutil.copy2(src, dst)
        return (src, dst, 'copied')
    except Exception as e:
        return (src, dst, f'error:{e}')


def gather_files(source_dir: Path, pattern: str = "**/*"):
    # Return all files matching pattern (files only)
    return [p for p in source_dir.glob(pattern) if p.is_file()]


def main():
    parser = argparse.ArgumentParser(description="Standardize DeepPatent2: copy Original -> standardized")
    parser.add_argument("--source", type=Path, required=True, help="Source 'Original' folder")
    parser.add_argument("--dest", type=Path, required=True, help="Destination standardized folder")
    parser.add_argument("--pattern", type=str, default="**/*", help="Glob pattern to select files (default: all)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel copy workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    parser.add_argument("--dry-run", action="store_true", help="Don't copy, just show what would be copied")
    parser.add_argument("--verbose", action="store_true", help="Show per-file actions")

    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()

    if not source.exists() or not source.is_dir():
        print(f"Error: source folder does not exist: {source}")
        sys.exit(2)

    # Ensure destination folder exists
    dest.mkdir(parents=True, exist_ok=True)

    files = gather_files(source, args.pattern)
    total = len(files)

    if total == 0:
        print(f"No files found in {source} with pattern '{args.pattern}'")
        return

    print(f"Found {total} files under {source}")
    if args.dry_run:
        print("Dry-run mode: no files will be copied")

    # Use ThreadPoolExecutor to copy files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for p in files:
            # Compute relative path under source and target path
            rel = p.relative_to(source)
            target = dest / rel
            futures[ex.submit(copy_file, p, target, args.overwrite, args.dry_run)] = p

        with tqdm(total=total, desc="Copying", unit="file") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if args.verbose:
                    src, dst, status = res
                    print(f"{status.upper():7} {src} -> {dst}")
                pbar.update(1)

    # Summarize
    copied = sum(1 for r in results if r[2] == 'copied')
    skipped = sum(1 for r in results if r[2] == 'skipped')
    dry = sum(1 for r in results if r[2] == 'dry')
    errors = [r for r in results if r[2].startswith('error:')]

    print(f"\nSummary: total={total} copied={copied} skipped={skipped} dry={dry} errors={len(errors)}")
    if errors:
        print("Errors (sample):")
        for e in errors[:10]:
            print(f"  {e[0]} -> {e[1]} : {e[2]}")

    if args.dry_run:
        print("Dry-run finished. Rerun without --dry-run to perform copying.")


if __name__ == '__main__':
    main()
