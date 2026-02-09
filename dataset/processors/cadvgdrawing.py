import shutil
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from .base import Processor


class CADVGDrawingProcessor(Processor):
    """Processor for CAD-VGDrawing (Drawing2CAD) dataset.

    Copies SVG vector files to the vector directory and optionally
    generates raster PNG versions for training.
    """

    def standardize(self, input_dir: Path, output_base: Path, dry_run: bool = True) -> Dict:
        input_dir = Path(input_dir)
        output_base = Path(output_base)
        vec_dir = output_base / "vector" / "cadvgdrawing"
        ras_dir = output_base / "raster" / "cadvgdrawing"

        # Find SVG files in the nested directory structure
        svg_files = list(input_dir.rglob("*.svg"))

        # Filter out any non-SVG files and ensure we're in the svg_raw directory
        svg_files = [f for f in svg_files if f.suffix.lower() == ".svg" and "svg_raw" in str(f)][
            :10000
        ]  # Limit to 10,000

        if dry_run:
            return {
                "dataset": "cadvgdrawing",
                "svg_count": len(svg_files),
                "vec_dir": str(vec_dir),
                "ras_dir": str(ras_dir),
                "dry_run": True,
            }

        vec_dir.mkdir(parents=True, exist_ok=True)
        ras_dir.mkdir(parents=True, exist_ok=True)

        svg_copied = 0
        png_count = 0

        for svg_file in tqdm(svg_files, desc="Processing CADVGDrawing files"):
            # Create a flattened filename from the path
            # e.g., svg_raw/0000/00000007/00000007_Front.svg -> 00000007_Front.svg
            parts = svg_file.parts
            # Find the svg_raw directory and take the last few parts
            try:
                svg_raw_idx = parts.index("svg_raw")
                # Get the last directory name and filename
                model_dir = parts[svg_raw_idx + 2]  # e.g., '00000007'
                filename = parts[-1]  # e.g., '00000007_Front.svg'
                # Create new filename: model_view.svg
                new_filename = f"{model_dir}_{filename}"
            except (IndexError, ValueError):
                # Fallback to original filename if path parsing fails
                new_filename = svg_file.name

            dest_path = vec_dir / new_filename

            # Copy SVG file
            if not dest_path.exists():
                shutil.copy2(svg_file, dest_path)
                svg_copied += 1

                # Render PNG from SVG
                try:
                    import cairosvg

                    png_path = ras_dir / new_filename.replace(".svg", ".png")
                    with open(svg_file, "r", encoding="utf-8") as f:
                        svg_content = f.read()
                    cairosvg.svg2png(
                        bytestring=svg_content.encode("utf-8"),
                        write_to=str(png_path),
                        output_width=1000,
                        output_height=1000,
                    )
                    png_count += 1
                except Exception as e:
                    print(f"Error rendering PNG for {new_filename}: {e}")

        return {
            "dataset": "cadvgdrawing",
            "svg_count": len(svg_files),
            "svg_copied": svg_copied,
            "png_count": png_count,
        }
