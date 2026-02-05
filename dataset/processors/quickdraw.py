"""Processor for QuickDraw Dataset (vector sketches)."""

import json
from pathlib import Path
from typing import Dict, Any

from .base import Processor


class QuickDrawProcessor(Processor):
    """Process QuickDraw dataset vector sketches into vector/raster format.

    QuickDraw contains 50M+ vector sketches as stroke sequences.
    Each drawing consists of multiple strokes, where each stroke contains
    sequences of (x,y) coordinates. Converts to SVG format for vectorization.
    """

    def standardize(self, input_dir: Path, output_base: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Process QuickDraw dataset files."""
        vector_dir = output_base / "vector" / "quickdraw"
        raster_dir = output_base / "raster" / "quickdraw"

        if not dry_run:
            vector_dir.mkdir(parents=True, exist_ok=True)
            raster_dir.mkdir(parents=True, exist_ok=True)

        svg_count = 0
        png_count = 0
        processed_files = 0
        skipped_files = 0

        # Look for NDJSON files (.ndjson) or other QuickDraw formats
        ndjson_files = list(input_dir.glob("*.ndjson"))
        parquet_files = list(input_dir.glob("*.parquet"))

        if not ndjson_files and not parquet_files:
            # Look in subdirectories
            for subdir in input_dir.iterdir():
                if subdir.is_dir():
                    ndjson_files.extend(list(subdir.glob("*.ndjson")))
                    parquet_files.extend(list(subdir.glob("*.parquet")))

        print(f"Found {len(ndjson_files)} NDJSON files and {len(parquet_files)} Parquet files")

        # Process NDJSON files (simplified format)
        max_drawings = 100 if dry_run else 1000  # Reasonable limit for actual processing
        drawings_processed = 0

        for ndjson_file in ndjson_files:
            try:
                print(f"Processing {ndjson_file.name}...")

                with open(ndjson_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if drawings_processed >= max_drawings:
                            break

                        try:
                            drawing_data = json.loads(line.strip())
                            drawing_id = f"{ndjson_file.stem}_{line_num:06d}"

                            # Create SVG from stroke data
                            svg_content = self._strokes_to_svg(drawing_data, drawing_id)

                            if svg_content:
                                if not dry_run:
                                    svg_path = vector_dir / f"{drawing_id}.svg"
                                    with open(svg_path, 'w', encoding='utf-8') as svg_file:
                                        svg_file.write(svg_content)
                                svg_count += 1
                                processed_files += 1
                                drawings_processed += 1

                                if drawings_processed % 100 == 0:
                                    print(f"Processed {drawings_processed} drawings...")

                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num} in {ndjson_file}: {e}")
                            skipped_files += 1
                            continue

            except Exception as e:
                print(f"Error processing {ndjson_file}: {e}")
                skipped_files += 1
                continue

        # Process Parquet files if NDJSON not found
        if not ndjson_files and parquet_files:
            print("No NDJSON files found, trying Parquet files...")
            try:
                import pandas as pd

                for parquet_file in parquet_files[:1]:  # Just process first file for now
                    print(f"Processing {parquet_file.name}...")

                    df = pd.read_parquet(parquet_file)

                    for idx, row in df.iterrows():
                        if drawings_processed >= max_drawings:
                            break

                        drawing_id = f"{parquet_file.stem}_{idx:06d}"

                        # Assume the drawing data is in a 'drawing' column
                        if 'drawing' in row:
                            svg_content = self._strokes_to_svg({'drawing': row['drawing']}, drawing_id)

                            if svg_content:
                                if not dry_run:
                                    svg_path = vector_dir / f"{drawing_id}.svg"
                                    with open(svg_path, 'w', encoding='utf-8') as svg_file:
                                        svg_file.write(svg_content)
                                svg_count += 1
                                processed_files += 1
                                drawings_processed += 1

            except ImportError:
                print("pandas not available for Parquet processing")
                skipped_files += len(parquet_files)
            except Exception as e:
                print(f"Error processing Parquet files: {e}")
                skipped_files += len(parquet_files)

        return {
            "dataset": "quickdraw",
            "input_dir": str(input_dir),
            "vector_dir": str(vector_dir),
            "raster_dir": str(raster_dir),
            "ndjson_files_found": len(ndjson_files),
            "parquet_files_found": len(parquet_files),
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "svg_count": svg_count,
            "png_count": png_count,
            "note": "QuickDraw processor converts stroke sequences to SVG vector format",
        }

    def _strokes_to_svg(self, drawing_data: Dict, drawing_id: str) -> str:
        """Convert QuickDraw stroke data to SVG format.

        Args:
            drawing_data: Dict containing 'drawing' key with stroke sequences
            drawing_id: Unique identifier for the drawing

        Returns:
            SVG string representation of the drawing
        """
        try:
            strokes = drawing_data.get('drawing', [])

            if not strokes:
                return ""

            # Calculate bounds for SVG viewBox
            all_x = []
            all_y = []

            for stroke in strokes:
                if len(stroke) >= 2:
                    x_coords, y_coords = stroke[0], stroke[1]
                    all_x.extend(x_coords)
                    all_y.extend(y_coords)

            if not all_x or not all_y:
                return ""

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # Add some padding
            padding = 10
            width = max_x - min_x + 2 * padding
            height = max_y - min_y + 2 * padding

            # Create SVG paths for each stroke
            paths = []
            for stroke in strokes:
                if len(stroke) >= 2:
                    x_coords, y_coords = stroke[0], stroke[1]

                    if len(x_coords) > 0 and len(y_coords) > 0:
                        # Create path string
                        path_data = f"M {x_coords[0] - min_x + padding},{y_coords[0] - min_y + padding}"
                        for x, y in zip(x_coords[1:], y_coords[1:]):
                            path_data += f" L {x - min_x + padding},{y - min_y + padding}"

                        paths.append(
                            f'<path d="{path_data}" fill="none" stroke="black" '
                            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                        )

            if not paths:
                return ""

            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <g id="{drawing_id}">
    {"".join(paths)}
  </g>
</svg>'''

            return svg_content

        except Exception as e:
            print(f"Error converting strokes to SVG for {drawing_id}: {e}")
            return ""
