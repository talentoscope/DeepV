"""Processor for ResPlan Dataset (residential floorplans)."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Processor


class ResPlanProcessor(Processor):
    """Process ResPlan dataset residential floorplans into vector/raster format.

    Extracts vector primitives from pickled Shapely geometries containing
    walls, doors, windows, balconies, and room spaces. Converts geometric
    data into colored SVG representations with PNG raster renderings.

    The dataset contains residential floorplan geometries with different
    element types (walls, doors, windows, rooms) stored as Shapely objects.

    Args:
        input_dir: Directory containing ResPlan.pkl file (or extracted/ subdirectory)
        output_base: Base directory for processed output
        dry_run: If True, only analyze and report without processing files

    Returns:
        Dict containing processing metadata (SVG/PNG counts, directory paths)
    """

    def standardize(self, input_dir: Path, output_base: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Process ResPlan dataset files.

        Loads pickled floorplan geometries and converts them to SVG vector
        format with corresponding PNG raster renderings using CairoSVG.
        """
        vector_dir = output_base / "vector" / "resplan"
        raster_dir = output_base / "raster" / "resplan"

        if not dry_run:
            vector_dir.mkdir(parents=True, exist_ok=True)
            raster_dir.mkdir(parents=True, exist_ok=True)

        svg_count = 0
        png_count = 0

        # Look for the pickle file
        pkl_file = input_dir / "extracted" / "ResPlan.pkl"
        if not pkl_file.exists():
            pkl_file = input_dir / "ResPlan.pkl"

        if pkl_file.exists():
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)

                print(f"Loaded {len(data)} floorplans from pickle file")

                for i, plan in enumerate(data[:10] if dry_run else data[:10000]):  # Limit to 10,000
                    plan_id = f"plan_{i:05d}"

                    # Create SVG from geometry data
                    svg_content = self._create_svg_from_resplan(plan, plan_id)

                    if svg_content and not dry_run:
                        svg_path = vector_dir / f"{plan_id}.svg"
                        with open(svg_path, "w") as f:
                            f.write(svg_content)
                        svg_count += 1

                        # Render PNG from SVG
                        try:
                            import cairosvg

                            png_path = raster_dir / f"{plan_id}.png"
                            cairosvg.svg2png(
                                bytestring=svg_content.encode("utf-8"),
                                write_to=str(png_path),
                                output_width=1000,
                                output_height=1000,
                            )
                            png_count += 1
                        except Exception as e:
                            print(f"Error rendering PNG for {plan_id}: {e}")

            except Exception as e:
                print(f"Error processing pickle file: {e}")
        else:
            print(f"ResPlan.pkl not found at {pkl_file}")

        return {
            "dataset": "resplan",
            "svg_count": svg_count,
            "png_count": png_count,
            "vec_dir": str(vector_dir),
            "ras_dir": str(raster_dir),
            "dry_run": dry_run,
        }

    def _create_svg_from_resplan(self, plan: Dict, plan_id: str) -> Optional[str]:
        """Create SVG from ResPlan Shapely geometry data.

        Converts floorplan geometries (walls, doors, windows, rooms) into
        a colored SVG representation with appropriate styling for each element type.

        Args:
            plan: Dictionary containing Shapely geometries keyed by element type
            plan_id: Unique identifier for the floorplan

        Returns:
            SVG content as string, or None if conversion fails
        """
        try:
            from shapely.geometry import MultiPolygon, Polygon

            svg_elements = []
            width, height = 1000, 800

            # Room types to visualize with different colors
            room_types = {
                "wall": ("black", 2),
                "door": ("brown", 1),
                "window": ("cyan", 1),
                "balcony": ("orange", 1),
                "bathroom": ("blue", 0.5),
                "bedroom": ("green", 0.5),
                "kitchen": ("red", 0.5),
                "living": ("yellow", 0.5),
                "inner": ("gray", 0.5),
            }

            bounds_min_x, bounds_min_y = float("inf"), float("inf")
            bounds_max_x, bounds_max_y = float("-inf"), float("-inf")

            # First pass: collect all bounds
            for room_type in room_types.keys():
                if room_type in plan:
                    geom = plan[room_type]
                    if hasattr(geom, "bounds"):
                        minx, miny, maxx, maxy = geom.bounds
                        bounds_min_x = min(bounds_min_x, minx)
                        bounds_min_y = min(bounds_min_y, miny)
                        bounds_max_x = max(bounds_max_x, maxx)
                        bounds_max_y = max(bounds_max_y, maxy)

            if bounds_min_x == float("inf"):
                return None

            # Calculate scale to fit in SVG
            scale = min(900 / (bounds_max_x - bounds_min_x), 700 / (bounds_max_y - bounds_min_y))
            offset_x = -bounds_min_x * scale + 50
            offset_y = -bounds_min_y * scale + 50

            # Second pass: create SVG elements
            for room_type, (color, stroke_width) in room_types.items():
                if room_type in plan:
                    geom = plan[room_type]

                    if geom.is_empty:
                        continue

                    # Convert geometry to SVG paths
                    if isinstance(geom, (Polygon, MultiPolygon)):
                        paths = self._geometry_to_svg_paths(geom, scale, offset_x, offset_y)
                        for path_data in paths:
                            if room_type in ["wall", "door", "window"]:
                                # Lines/strokes for structural elements
                                svg_elements.append(
                                    f'<path d="{path_data}" fill="none" '
                                    f'stroke="{color}" stroke-width="{stroke_width}"/>'
                                )
                            else:
                                # Filled areas for rooms
                                svg_elements.append(
                                    f'<path d="{path_data}" fill="{color}" fill-opacity="0.3" '
                                    f'stroke="{color}" stroke-width="0.5"/>'
                                )

            if not svg_elements:
                return None

            svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  {"".join(svg_elements)}
</svg>"""

            return svg

        except Exception as e:
            print(f"Error creating SVG for {plan_id}: {e}")
            return None

    def _geometry_to_svg_paths(self, geom, scale: float, offset_x: float, offset_y: float) -> List[str]:
        """Convert Shapely geometry to SVG path data.

        Transforms Shapely Polygon/MultiPolygon objects into SVG path strings
        with proper coordinate scaling and translation.

        Args:
            geom: Shapely geometry object (Polygon or MultiPolygon)
            scale: Scaling factor to fit geometry in SVG viewport
            offset_x: X-axis translation offset
            offset_y: Y-axis translation offset

        Returns:
            List of SVG path data strings
        """
        paths = []

        def coords_to_path(coords):
            if len(coords) < 2:
                return ""
            path = f"M {coords[0][0]*scale + offset_x},{coords[0][1]*scale + offset_y}"
            for x, y in coords[1:]:
                path += f" L {x*scale + offset_x},{y*scale + offset_y}"
            path += " Z"
            return path

        if hasattr(geom, "geoms"):  # MultiPolygon
            for poly in geom.geoms:
                if hasattr(poly, "exterior"):
                    coords = list(poly.exterior.coords)
                    if coords:
                        paths.append(coords_to_path(coords))
        else:  # Single Polygon
            if hasattr(geom, "exterior"):
                coords = list(geom.exterior.coords)
                if coords:
                    paths.append(coords_to_path(coords))

        return paths
