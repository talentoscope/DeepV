#!/usr/bin/env python3
"""
Test script to demonstrate vector-to-raster conversion using the DeepV rendering system.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import svgpathtools
from PIL import Image

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from util_files.data.graphics_primitives import PT_CBEZIER, PT_LINE, PT_QBEZIER
from util_files.rendering.cairo import render


def parse_svg_to_primitives(svg_path):
    """Parse SVG file and extract geometric primitives."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Get SVG dimensions
    width = float(root.get('width', 1000))
    height = float(root.get('height', 1000))

    primitives = []

    # Find all path elements
    for path_elem in root.iter('{http://www.w3.org/2000/svg}path'):
        if 'd' not in path_elem.attrib:
            continue

        d_string = path_elem.attrib['d']
        try:
            path = svgpathtools.parse_path(d_string)

            for segment in path:
                if isinstance(segment, svgpathtools.Line):
                    x1, y1 = segment.start.real, segment.start.imag
                    x2, y2 = segment.end.real, segment.end.imag

                    stroke_width = 0.1
                    if 'stroke-width' in path_elem.attrib:
                        try:
                            stroke_width = float(path_elem.attrib['stroke-width'])
                        except ValueError:
                            pass

                    line = np.array([x1, y1, x2, y2, stroke_width])
                    primitives.append((PT_LINE, line))

        except Exception as e:
            print(f"Warning: Failed to parse path: {e}")
            continue

    return primitives, (width, height)

def convert_primitives_to_render_format(primitives):
    """Convert primitives to render format."""
    primitive_sets = {PT_LINE: [], PT_CBEZIER: [], PT_QBEZIER: []}

    for prim_type, prim_data in primitives:
        if prim_type in primitive_sets:
            primitive_sets[prim_type].append(prim_data)

    for prim_type in primitive_sets:
        if primitive_sets[prim_type]:
            primitive_sets[prim_type] = np.array(primitive_sets[prim_type])
        else:
            primitive_sets[prim_type] = np.array([]).reshape(0, 5 if prim_type == PT_LINE else 9)

    return primitive_sets

def test_single_file(svg_path):
    """Test vector-to-raster conversion on a single SVG file."""
    print(f"\n=== Testing: {svg_path.name} ===")

    try:
        primitives, dimensions = parse_svg_to_primitives(svg_path)
        print(f"Parsed {len(primitives)} primitives")
        print(f"SVG dimensions: {dimensions}")

        if not primitives:
            print("No primitives found")
            return

        primitive_sets = convert_primitives_to_render_format(primitives)

        for prim_type, prims in primitive_sets.items():
            if len(prims) > 0:
                print(f"  {prim_type}: {len(prims)} primitives")

        width, height = dimensions
        scale_factor = min(800 / width, 800 / height, 1.0)
        render_width = int(width * scale_factor)
        render_height = int(height * scale_factor)

        print(f"Rendering at {render_width}x{render_height}")

        scaled_primitive_sets = {}
        for prim_type, prims in primitive_sets.items():
            if len(prims) > 0:
                scaled_prims = prims.copy()
                coord_cols = 4 if prim_type == PT_LINE else 8
                scaled_prims[:, :coord_cols] *= scale_factor
                scaled_primitive_sets[prim_type] = scaled_prims
            else:
                scaled_primitive_sets[prim_type] = prims

        rendered_image = render(
            scaled_primitive_sets,
            (render_width, render_height),
            data_representation="vahe"
        )

        print(f"Rendered image shape: {rendered_image.shape}")
        print(f"Rendered image dtype: {rendered_image.dtype}")

        if rendered_image.dtype != np.uint8:
            img_min, img_max = rendered_image.min(), rendered_image.max()
            if img_max > img_min:
                rendered_image = ((rendered_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                rendered_image = (rendered_image * 255).astype(np.uint8)

        if len(rendered_image.shape) == 3:
            if rendered_image.shape[2] == 1:
                rendered_image = rendered_image.squeeze(axis=2)
            elif rendered_image.shape[2] == 3:
                rendered_image = np.mean(rendered_image, axis=2).astype(np.uint8)

        pil_image = Image.fromarray(rendered_image, mode='L')
        output_path = Path(f"test_render_{svg_path.stem}.png")
        pil_image.save(output_path)
        print(f"Saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    base_path = Path("data/raw/floorplancad/train1")
    test_files = [
        base_path / "0107-0027.svg",
        base_path / "0110-0004.svg",
    ]

    for svg_file in test_files:
        if svg_file.exists():
            test_single_file(svg_file)
        else:
            print(f"File not found: {svg_file}")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">test_vector_to_raster.py