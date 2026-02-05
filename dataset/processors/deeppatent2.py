from pathlib import Path
import shutil
import json
from .base import Processor
from typing import Dict, List, Any


class DeepPatent2Processor(Processor):
    """Standardizer for DeepPatent2.

    Processes PNG images and JSON metadata files to create vector representations.
    Extracts bounding boxes from JSON annotations and converts them to SVG format.
    Moves PNGs into `data/raster/deeppatent2` and creates SVGs in `data/vector/deeppatent2`.
    """

    def standardize(self, raw_dir: Path, output_base: Path, dry_run: bool = True) -> Dict:
        raw_dir = Path(raw_dir)
        output_base = Path(output_base)
        ras_dir = output_base / "raster" / "deeppatent2"
        vec_dir = output_base / "vector" / "deeppatent2"

        # Find all PNG and JSON files
        pngs = list(raw_dir.rglob("*.png"))
        jsons = list(raw_dir.rglob("*.json"))

        # Filter out metadata.json files
        jsons = [j for j in jsons if j.name != "metadata.json"]

        if dry_run:
            return {
                "dataset": "deeppatent2",
                "png_count": len(pngs),
                "json_count": len(jsons),
                "ras_dir": str(ras_dir),
                "vec_dir": str(vec_dir),
                "dry_run": True,
            }

        ras_dir.mkdir(parents=True, exist_ok=True)
        vec_dir.mkdir(parents=True, exist_ok=True)

        # Process PNG files
        png_moved = 0
        for png_file in pngs:
            dest = ras_dir / png_file.name
            if not dest.exists():
                shutil.copy2(png_file, dest)
                png_moved += 1

        # Process JSON files to create SVG vectors
        svg_created = 0
        json_processed = 0

        for json_file in jsons:
            try:
                svg_created += self._process_json_to_svg(json_file, vec_dir)
                json_processed += 1
            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
                continue

        return {
            "dataset": "deeppatent2",
            "png_count": len(pngs),
            "png_moved": png_moved,
            "json_count": len(jsons),
            "json_processed": json_processed,
            "svg_created": svg_created,
        }

    def _process_json_to_svg(self, json_file: Path, vec_dir: Path) -> int:
        """Process a JSON file to create SVG vector representation.

        Returns the number of SVG files created (0 or 1).
        """
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {json_file}")
                return 0

        # Create SVG from JSON data
        svg_content = self._create_svg_from_annotations(data, json_file.stem)

        if svg_content:
            svg_file = vec_dir / f"{json_file.stem}.svg"
            with open(svg_file, "w", encoding="utf-8") as f:
                f.write(svg_content)
            return 1

        return 0

    def _create_svg_from_annotations(self, data: Dict[str, Any], filename: str) -> str:
        """Create SVG content from JSON annotations.

        Handles various JSON formats that might contain bounding boxes, objects, etc.
        """
        # Try to extract image dimensions and objects
        width, height = self._extract_image_dimensions(data)
        objects = self._extract_objects(data)

        if not objects:
            return ""

        # Create SVG with bounding boxes as rectangles
        svg_elements = []

        for i, obj in enumerate(objects):
            bbox = obj.get("bbox") or obj.get("bounding_box")
            if bbox and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                # Ensure positive dimensions
                w = max(w, 1)
                h = max(h, 1)

                # Get object label/category
                label = obj.get("object", obj.get("class", obj.get("category", f"object_{i}")))

                # Create rectangle element
                rect = (
                    f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                    f'fill="none" stroke="black" stroke-width="2" '
                    f'title="{label}"/>'
                )
                svg_elements.append(rect)

        if not svg_elements:
            return ""

        # Create SVG wrapper
        svg_width = width or 1000  # Default width if not found
        svg_height = height or 1000  # Default height if not found

        svg_content = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg width="{svg_width}" height="{svg_height}" '
            f'xmlns="http://www.w3.org/2000/svg">\n'
            f"  <title>{filename}</title>\n"
            f'  {"  ".join(svg_elements)}\n'
            f"</svg>"
        )

        return svg_content

    def _extract_image_dimensions(self, data: Dict[str, Any]) -> tuple:
        """Extract image width and height from JSON data."""
        # Try various possible keys for image dimensions
        width = data.get("width") or data.get("image_width") or data.get("img_width")
        height = data.get("height") or data.get("image_height") or data.get("img_height")

        # Try nested structures
        if not width and "image" in data:
            img_data = data["image"]
            if isinstance(img_data, dict):
                width = img_data.get("width")
                height = img_data.get("height")

        return width, height

    def _extract_objects(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract object annotations from JSON data."""
        objects = []

        # Try various possible keys for objects
        possible_keys = ["objects", "annotations", "items", "elements", "shapes", "regions"]

        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                objects.extend(data[key])

        # Handle case where objects are at root level
        if not objects and isinstance(data, dict):
            # Check if the data itself represents an object
            if any(k in data for k in ["bbox", "bounding_box", "object", "class", "category"]):
                objects.append(data)

        return objects
