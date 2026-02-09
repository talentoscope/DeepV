"""Processor for CubiCasa5K Dataset."""

import base64
import shutil
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from .base import Processor


class CubiCasa5KProcessor(Processor):
    """Process CubiCasa5K dataset into vector/raster format.

    Extracts SVG polygon annotations and high-res raster images from
    individual floorplan directories for noisy raster-to-vector tasks.
    """

    def standardize(self, input_dir: Path, output_base: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Process CubiCasa5K floorplan directories."""
        vector_dir = output_base / "vector" / "cubicasa5k"
        raster_dir = output_base / "raster" / "cubicasa5k"

        if not dry_run:
            vector_dir.mkdir(parents=True, exist_ok=True)
            raster_dir.mkdir(parents=True, exist_ok=True)

        svg_count = 0
        png_count = 0

        # Look for high_quality directory with floorplan subdirectories
        high_quality_dir = input_dir / "cubicasa5k" / "high_quality"
        if not high_quality_dir.exists():
            print(f"High quality directory not found at: {high_quality_dir}")
            return {
                "dataset": "cubicasa5k",
                "svg_count": 0,
                "png_count": 0,
                "error": "High quality directory not found",
                "dry_run": dry_run,
            }

        print(f"Found high quality directory at: {high_quality_dir}")

        # Get all floorplan directories
        floorplan_dirs = [d for d in high_quality_dir.iterdir() if d.is_dir()]
        print(f"Found {len(floorplan_dirs)} floorplan directories")

        # Process floorplans (limit for dry run)
        max_floorplans = 5 if dry_run else float('inf')

        for i, floorplan_dir in enumerate(floorplan_dirs):
            if i >= max_floorplans:
                break

            floorplan_id = floorplan_dir.name
            print(f"Processing floorplan {floorplan_id} ({i+1}/{min(len(floorplan_dirs), max_floorplans)})")

            try:
                # Check for required files
                original_png = floorplan_dir / "F1_original.png"
                scaled_png = floorplan_dir / "F1_scaled.png"
                model_svg = floorplan_dir / "model.svg"

                if not model_svg.exists():
                    print(f"  Warning: model.svg not found in {floorplan_dir}")
                    continue

                # Copy raster image (prefer original, fallback to scaled)
                raster_source = original_png if original_png.exists() else scaled_png
                if raster_source.exists() and not dry_run:
                    raster_dest = raster_dir / f"{floorplan_id}.png"
                    shutil.copy2(raster_source, raster_dest)
                    png_count += 1

                # Process SVG annotations
                if not dry_run:
                    svg_content = self._process_svg_annotations(model_svg, floorplan_id)
                    if svg_content:
                        svg_dest = vector_dir / f"{floorplan_id}.svg"
                        with open(svg_dest, 'w', encoding='utf-8') as f:
                            f.write(svg_content)
                        svg_count += 1

            except Exception as e:
                print(f"Error processing floorplan {floorplan_id}: {e}")
                continue

        return {
            "dataset": "cubicasa5k",
            "svg_count": svg_count,
            "png_count": png_count,
            "vec_dir": str(vector_dir),
            "ras_dir": str(raster_dir),
            "dry_run": dry_run,
        }

    def _save_raster_image(self, image_data: dict, raster_dir: Path, key_str: str) -> Path:
        """Save raster image from CubiCasa data."""
        try:
            # CubiCasa stores images as base64 encoded data or numpy arrays
            if 'data' in image_data:
                # Assume base64 encoded image
                image_bytes = base64.b64decode(image_data['data'])
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)

                # Decode image
                if 'shape' in image_data:
                    height, width, channels = image_data['shape']
                    image = image_array.reshape((height, width, channels))
                else:
                    # Try to decode as PNG/JPEG
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image is not None:
                    # Convert BGR to RGB if needed
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Save as PNG
                    png_path = raster_dir / f"{key_str}.png"
                    pil_image = Image.fromarray(image)
                    pil_image.save(png_path)
                    return png_path

            elif 'path' in image_data:
                # Image stored as file path
                image_path = Path(image_data['path'])
                if image_path.exists():
                    # Copy to raster directory
                    png_path = raster_dir / f"{key_str}.png"
                    shutil.copy2(image_path, png_path)
                    return png_path

            return None

        except Exception as e:
            print(f"Error saving raster image for {key_str}: {e}")
            return None

    def _create_svg_from_annotations(self, svg_data: dict, key_str: str) -> str:
        """Create SVG from CubiCasa polygon annotations."""
        try:
            # CubiCasa SVG data contains polygon annotations for rooms, walls, etc.
            # Structure: {'polygons': [...], 'labels': [...], 'metadata': {...}}

            polygons = svg_data.get('polygons', [])
            labels = svg_data.get('labels', [])

            if not polygons:
                return None

            # Calculate bounds for SVG canvas
            all_points = []
            for polygon in polygons:
                if isinstance(polygon, list):
                    all_points.extend(polygon)

            if not all_points:
                return None

            # Flatten points and find bounds
            flat_points = []
            for point in all_points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    flat_points.extend(point)

            if len(flat_points) % 2 != 0:
                print(f"Warning: Odd number of coordinates in {key_str}")
                return None

            xs = flat_points[::2]
            ys = flat_points[1::2]

            if not xs or not ys:
                return None

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Add padding
            padding = 50
            width = int(max_x - min_x + 2 * padding)
            height = int(max_y - min_y + 2 * padding)

            if width <= 0 or height <= 0:
                return None

            # Create SVG elements
            svg_elements = []

            # Process each polygon
            for i, polygon in enumerate(polygons):
                if not isinstance(polygon, list) or len(polygon) < 6:  # Need at least 3 points (6 coords)
                    continue

                # Get label if available
                label = labels[i] if i < len(labels) else f"polygon_{i}"

                # Convert polygon to SVG path
                path_data = self._polygon_to_svg_path(polygon, min_x - padding, min_y - padding)

                if path_data:
                    # Color based on label (simplified)
                    fill_color = self._get_color_for_label(label)
                    svg_elements.append(
                        f'<path d="{path_data}" fill="{fill_color}" '
                        'stroke="black" stroke-width="1" opacity="0.7"/>'
                    )

                    # Add label text at centroid
                    centroid = self._calculate_centroid(polygon)
                    if centroid:
                        cx = centroid[0] - (min_x - padding)
                        cy = centroid[1] - (min_y - padding)
                        svg_elements.append(
                            f'<text x="{cx}" y="{cy}" font-size="12" '
                            'text-anchor="middle" fill="black">{label}</text>'
                        )

            if not svg_elements:
                return None

            # Create SVG content
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  {"".join(svg_elements)}
  <text x="10" y="20" font-size="12" fill="black">CubiCasa5K {key_str}</text>
</svg>'''

            return svg_content

        except Exception as e:
            print(f"Error creating SVG for {key_str}: {e}")
            return None

    def _polygon_to_svg_path(self, polygon: list, offset_x: float, offset_y: float) -> str:
        """Convert polygon coordinates to SVG path data."""
        try:
            if len(polygon) < 6 or len(polygon) % 2 != 0:
                return None

            # Extract x,y coordinates
            coords = []
            for i in range(0, len(polygon), 2):
                x = polygon[i] - offset_x
                y = polygon[i + 1] - offset_y
                coords.append((x, y))

            if len(coords) < 3:
                return None

            # Create SVG path
            path_parts = [f"M {coords[0][0]} {coords[0][1]}"]
            for x, y in coords[1:]:
                path_parts.append(f"L {x} {y}")
            path_parts.append("Z")  # Close path

            return " ".join(path_parts)

        except Exception as e:
            print(f"Error converting polygon to path: {e}")
            return None

    def _calculate_centroid(self, polygon: list) -> tuple:
        """Calculate centroid of polygon."""
        try:
            if len(polygon) < 6 or len(polygon) % 2 != 0:
                return None

            xs = polygon[::2]
            ys = polygon[1::2]

            centroid_x = sum(xs) / len(xs)
            centroid_y = sum(ys) / len(ys)

            return (centroid_x, centroid_y)

        except Exception:
            return None

    def _get_color_for_label(self, label: str) -> str:
        """Get color for semantic label."""
        # Simplified color mapping for common room types
        color_map = {
            'living_room': '#FFB6C1',  # Light pink
            'kitchen': '#FFA07A',     # Light salmon
            'bedroom': '#98FB98',     # Pale green
            'bathroom': '#87CEEB',    # Sky blue
            'hallway': '#DDA0DD',     # Plum
            'wall': '#000000',        # Black
            'door': '#8B4513',        # Saddle brown
            'window': '#00CED1',      # Dark turquoise
        }

        # Default color
        return color_map.get(label.lower().replace(' ', '_'), '#CCCCCC')

    def _process_svg_annotations(self, svg_path: Path, floorplan_id: str) -> str:
        """Process SVG annotations from CubiCasa model.svg file.

        Extracts wall and space polygons and converts them to line segments
        for vectorization training.

        Args:
            svg_path: Path to the model.svg file
            floorplan_id: Identifier for the floorplan

        Returns:
            SVG content as string with extracted line segments
        """
        try:
            import xml.etree.ElementTree as ET

            # Parse SVG file
            tree = ET.parse(svg_path)
            root = tree.getroot()

            # Define namespaces
            ns = {'svg': 'http://www.w3.org/2000/svg'}

            # Get SVG dimensions
            width = root.get('width', '1000')
            height = root.get('height', '800')

            # Extract line segments from polygons
            line_segments = []

            # Find all g elements with polygons
            for g_element in root.findall('.//svg:g', ns):
                class_attr = g_element.get('class', '')
                if 'Wall' in class_attr or 'Space' in class_attr:
                    # Find polygons in this g element
                    polygons = g_element.findall('svg:polygon', ns)
                    for polygon in polygons:
                        # Get polygon points
                        points_str = polygon.get('points', '')
                        if points_str:
                            # Parse points string
                            coords = []
                            for point in points_str.strip().split():
                                try:
                                    x, y = map(float, point.split(','))
                                    coords.append((x, y))
                                except ValueError:
                                    continue

                            # Convert polygon to line segments
                            if len(coords) >= 2:
                                for i in range(len(coords)):
                                    start = coords[i]
                                    end = coords[(i + 1) % len(coords)]  # Close the polygon
                                    line_segments.append((start, end))

            # Create SVG with line segments
            svg_elements = []
            for (start, end) in line_segments:
                x1, y1 = start
                x2, y2 = end
                svg_elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="1"/>')

            if not svg_elements:
                print(f"  Warning: No line segments extracted from {svg_path}")
                return None

            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
{chr(10).join(svg_elements)}
</svg>'''

            return svg_content

        except Exception as e:
            print(f"Error processing SVG annotations for {floorplan_id}: {e}")
            return None 
