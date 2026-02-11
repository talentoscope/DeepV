"""Processor for Modified Swiss Dwellings (MSD) Dataset."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from .base import Processor


class MSDProcessor(Processor):
    """Process Modified Swiss Dwellings dataset into vector/raster format.

    Extracts vector primitives from NetworkX graphs containing room geometries
    as Shapely polygons, and raster images from numpy arrays. The dataset
    contains floor plan graphs with room type annotations and structural layouts.

    Processes two types of data:
    - graph_out/*.pickle: NetworkX graphs with room geometries and types
    - struct_in/*.npy: 3D numpy arrays with structural information

    Args:
        input_dir: Directory containing graph_out/ and struct_in/ subdirectories
        output_base: Base directory for processed output
        dry_run: If True, only analyze and report without processing files

    Returns:
        Dict containing processing metadata (SVG/PNG counts, directory paths)
    """

    def standardize(self, input_dir: Path, output_base: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Process MSD dataset files.

        Extracts SVG floor plans from NetworkX graphs and PNG structural images
        from numpy arrays, organizing them into standardized vector/raster directories.
        """
        vector_dir = output_base / "vector" / "msd"
        raster_dir = output_base / "raster" / "msd"

        if not dry_run:
            vector_dir.mkdir(parents=True, exist_ok=True)
            raster_dir.mkdir(parents=True, exist_ok=True)

        svg_count = 0
        png_count = 0

        # Look for graph_out directory with pickle files
        graph_out_dir = input_dir / "graph_out"
        if graph_out_dir.exists():
            pickle_files = list(graph_out_dir.glob("*.pickle"))
            print(f"Found {len(pickle_files)} graph_out pickle files")

            for pickle_file in tqdm(
                pickle_files[:10] if dry_run else pickle_files[:10000], desc="Processing MSD graphs"
            ):  # Limit to 10,000
                plan_id = pickle_file.stem  # e.g., "0", "1", etc.

                try:
                    # Load NetworkX graph with room geometries
                    with open(pickle_file, "rb") as f:
                        graph = pickle.load(f)

                    # Create SVG from graph geometries
                    svg_content = self._create_svg_from_msd_graph(graph, plan_id)

                    if svg_content and not dry_run:
                        svg_path = vector_dir / f"{plan_id}.svg"
                        with open(svg_path, "w") as f:
                            f.write(svg_content)
                        svg_count += 1

                except Exception as e:
                    print(f"Error processing {pickle_file}: {e}")
                    continue
        else:
            print(f"graph_out directory not found at {graph_out_dir}")

        # Also process struct_in for raster images
        struct_in_dir = input_dir / "struct_in"
        if struct_in_dir.exists():
            npy_files = list(struct_in_dir.glob("*.npy"))
            print(f"Found {len(npy_files)} struct_in npy files")

            for npy_file in npy_files[:10] if dry_run else npy_files:  # Limit for dry run
                plan_id = npy_file.stem

                try:
                    # Create PNG from structural numpy array
                    png_content = self._create_png_from_msd_struct(npy_file, plan_id)

                    if png_content and not dry_run:
                        png_path = raster_dir / f"{plan_id}.png"
                        with open(png_path, "wb") as f:
                            f.write(png_content)
                        png_count += 1

                except Exception as e:
                    print(f"Error processing {npy_file}: {e}")
                    continue
        else:
            print(f"struct_in directory not found at {struct_in_dir}")

        return {
            "dataset": "msd",
            "svg_count": svg_count,
            "png_count": png_count,
            "vec_dir": str(vector_dir),
            "ras_dir": str(raster_dir),
            "dry_run": dry_run,
        }

    def _create_svg_from_msd_graph(self, graph, plan_id: str) -> Optional[str]:
        """Create SVG from MSD NetworkX graph containing room geometries.

        Converts NetworkX graph nodes with Shapely polygon geometries into
        colored SVG paths with room type labels. Each room type gets a
        distinct color and abbreviated label.

        Args:
            graph: NetworkX graph with node data containing 'geometry' and 'roomtype'
            plan_id: Identifier for the floor plan

        Returns:
            SVG content as string, or None if conversion fails
        """
        try:
            svg_elements = []
            width, height = 1000, 800

            # Room type colors
            room_colors = {
                "Bedroom": "#FF6B6B",
                "Livingroom": "#4ECDC4",
                "Kitchen": "#45B7D1",
                "Dining": "#FFA07A",
                "Corridor": "#98D8C8",
                "Stairs": "#F7DC6F",
                "Storeroom": "#BB8FCE",
                "Bathroom": "#85C1E9",
                "Balcony": "#F8C471",
                "Structure": "#34495E",
            }

            bounds_min_x, bounds_min_y = float("inf"), float("inf")
            bounds_max_x, bounds_max_y = float("-inf"), float("-inf")

            # First pass: collect all bounds
            for node_id, node_data in graph.nodes(data=True):
                if "geometry" in node_data:
                    geom = node_data["geometry"]
                    if hasattr(geom, "bounds"):
                        minx, miny, maxx, maxy = geom.bounds
                        bounds_min_x = min(bounds_min_x, minx)
                        bounds_min_y = min(bounds_min_y, miny)
                        bounds_max_x = max(bounds_max_x, maxx)
                        bounds_max_y = max(bounds_max_y, maxy)

            if bounds_min_x == float("inf"):
                return None  # No valid geometries

            # Calculate scaling
            geom_width = bounds_max_x - bounds_min_x
            geom_height = bounds_max_y - bounds_min_y
            scale = min(width / geom_width, height / geom_height) * 0.8  # 80% of available space

            # Second pass: create SVG elements
            for node_id, node_data in graph.nodes(data=True):
                if "geometry" in node_data and "roomtype" in node_data:
                    geom = node_data["geometry"]
                    room_type = node_data["roomtype"]

                    color = room_colors.get(room_type, "#95A5A6")  # Default gray

                    if hasattr(geom, "exterior"):
                        # It's a Polygon
                        exterior = geom.exterior
                        coords = list(exterior.coords)

                        # Scale and translate coordinates
                        scaled_coords = []
                        for x, y in coords:
                            scaled_x = (x - bounds_min_x) * scale + 50  # 50px margin
                            scaled_y = height - ((y - bounds_min_y) * scale + 50)  # Flip Y axis
                            scaled_coords.append(f"{scaled_x},{scaled_y}")

                        if scaled_coords:
                            path_data = (
                                f"M {scaled_coords[0]} " + " ".join(f"L {coord}" for coord in scaled_coords[1:]) + " Z"
                            )
                            svg_elements.append(
                                f'<path d="{path_data}" fill="{color}" '
                                'stroke="black" stroke-width="1" opacity="0.7"/>'
                            )

                            # Add room type label
                            centroid = geom.centroid
                            label_x = (centroid.x - bounds_min_x) * scale + 50
                            label_y = height - ((centroid.y - bounds_min_y) * scale + 50)
                            svg_elements.append(
                                f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
                                f'font-size="10" fill="black">{room_type[:3]}</text>'
                            )

            if not svg_elements:
                return None

            svg_content = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
                '  <rect width="100%" height="100%" fill="white"/>\n'
                f'  {"".join(svg_elements)}\n'
                f'  <text x="10" y="20" font-size="12" fill="black">MSD Plan {plan_id}</text>\n'
                "</svg>"
            )

            return svg_content

        except Exception as e:
            print(f"Error creating SVG for plan {plan_id}: {e}")
            return None

    def _create_png_from_msd_struct(self, npy_file: Path, plan_id: str) -> Optional[bytes]:
        """Create PNG from MSD structural numpy array.

        Converts 3D numpy array (512x512x3) structural data into a binary
        PNG image where structural elements are black and background is white.

        Args:
            npy_file: Path to .npy file containing 3D structural array
            plan_id: Identifier for the floor plan

        Returns:
            PNG image data as bytes, or None if conversion fails
        """
        try:
            import numpy as np
            from PIL import Image

            # Load the 3D numpy array (512, 512, 3)
            stack = np.load(npy_file)

            # Get structural component as binary mask (first channel)
            struct_mask = stack[..., 0].astype(np.uint8)

            # Convert to RGB image (structure in black, background in white)
            rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)
            rgb_image[struct_mask == 0] = [0, 0, 0]  # Structure in black
            rgb_image[struct_mask == 1] = [255, 255, 255]  # Background in white

            # Create PIL Image and save to bytes
            img = Image.fromarray(rgb_image)
            from io import BytesIO

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()

        except Exception as e:
            print(f"Error creating PNG for plan {plan_id}: {e}")
            return None
