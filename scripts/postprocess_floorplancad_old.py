#!/usr/bin/env python3
"""
Post-process FloorPlanCAD vectors and rasters to convert to black lines on white background.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import cairosvg
from tqdm import tqdm


def process_svg(svg_path: Path, output_path: Path):
    """Process SVG to make lines black on white background."""
    try:
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Remove any existing background rects or defs
        for elem in list(root):
            if elem.tag.endswith(('rect', 'defs')) and 'width' in elem.attrib and elem.attrib.get('width') == '100%':
                root.remove(elem)

        # Add white background rectangle as first child
        bg_rect = ET.Element('rect')
        bg_rect.set('width', '100%')
        bg_rect.set('height', '100%')
        bg_rect.set('fill', 'white')
        bg_rect.set('stroke', 'none')
        root.insert(0, bg_rect)

        # Process the root
        def process_element(elem, bg_rect):
            if elem is bg_rect:
                return elem  # Skip the background rect
            if elem.tag.endswith('text'):
                return None  # Mark for removal
            if elem.tag.endswith(('path', 'line', 'polyline', 'polygon', 'circle', 'ellipse', 'rect')):
                # Set stroke to black and width to 0.1
                elem.set('stroke', 'black')
                elem.set('stroke-width', '0.1')
                elem.set('fill', 'none')
                # Remove any color attributes
                for attr in ['stroke', 'fill']:
                    if attr in elem.attrib:
                        if attr == 'stroke':
                            elem.set(attr, 'black')
                        elif attr == 'fill':
                            elem.set(attr, 'none')
            # Process children
            children_to_remove = []
            for child in list(elem):
                result = process_element(child, bg_rect)
                if result is None:
                    children_to_remove.append(child)
            # Remove marked children
            for child in children_to_remove:
                elem.remove(child)
            return elem
        
        # Process the root
        process_element(root, bg_rect)

        # Write processed SVG
        tree.write(output_path, encoding='unicode', xml_declaration=True)

    except Exception as e:
        print(f"Error processing {svg_path}: {e}")


def process_png(svg_path: Path, output_path: Path):
    """Render SVG to PNG using Cairo."""
    try:
        # Read the processed SVG content
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # Convert SVG to PNG using cairosvg
        # Set output size to 1000x1000
        cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=str(output_path), output_width=1000, output_height=1000)
        
    except Exception as e:
        print(f"Error rendering {svg_path} to PNG: {e}")


def main():
    parser = argparse.ArgumentParser(description="Post-process FloorPlanCAD vectors and rasters")
    parser.add_argument('--vector_dir', type=Path, required=True, help='Directory containing SVG files')
    parser.add_argument('--raster_dir', type=Path, required=True, help='Directory containing PNG files')
    parser.add_argument('--output_vector_dir', type=Path, help='Output directory for processed SVGs (default: overwrite)')
    parser.add_argument('--output_raster_dir', type=Path, help='Output directory for processed PNGs (default: overwrite)')

    args = parser.parse_args()

    vector_dir = args.vector_dir
    raster_dir = args.raster_dir
    output_vector_dir = args.output_vector_dir or vector_dir
    output_raster_dir = args.output_raster_dir or raster_dir

    # Create output directories if needed
    output_vector_dir.mkdir(parents=True, exist_ok=True)
    output_raster_dir.mkdir(parents=True, exist_ok=True)

    # Process SVGs
    svg_files = list(vector_dir.glob('*.svg'))
    print(f"Processing {len(svg_files)} SVG files...")
    for svg_file in tqdm(svg_files, desc="Processing SVGs"):
        output_path = output_vector_dir / svg_file.name
        process_svg(svg_file, output_path)

    # Render PNGs from processed SVGs
    processed_svg_files = list(output_vector_dir.glob('*.svg'))
    print(f"Rendering {len(processed_svg_files)} PNG files from SVGs...")
    for svg_file in tqdm(processed_svg_files, desc="Rendering PNGs"):
        png_path = output_raster_dir / svg_file.with_suffix('.png').name
        process_png(svg_file, png_path)

    print("Post-processing complete!")


if __name__ == '__main__':
    main()