#!/usr/bin/env python3
"""
Process FloorPlanCAD dataset directly from raw data to create clean vector/raster files.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import cairosvg
from tqdm import tqdm


def process_svg_content(svg_content):
    """Process SVG content to make lines black on white background and remove text."""
    try:
        # Parse SVG
        root = ET.fromstring(svg_content)

        # Change background color to white if it exists
        if 'style' in root.attrib:
            if 'background-color' in root.attrib['style']:
                root.set('style', root.attrib['style'].replace('background-color: #000;', 'background-color: #fff;'))
            else:
                root.set('style', root.attrib['style'] + '; background-color: #fff;')
        else:
            root.set('style', 'background-color: #fff;')

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

        # Process all elements to make strokes black and remove fills
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

        # Convert back to string
        processed_svg = ET.tostring(root, encoding='unicode', method='xml')
        return processed_svg

    except Exception as e:
        print(f"Error processing SVG content: {e}")
        return svg_content


def main():
    parser = argparse.ArgumentParser(description="Process FloorPlanCAD dataset from raw data")
    parser.add_argument('--input_dir', type=Path, required=True, help='Directory containing raw FloorPlanCAD data (SVG files)')
    parser.add_argument('--output_vector_dir', type=Path, required=True, help='Output directory for processed SVGs')
    parser.add_argument('--output_raster_dir', type=Path, required=True, help='Output directory for rendered PNGs')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_vector_dir = args.output_vector_dir
    output_raster_dir = args.output_raster_dir

    # Create output directories
    output_vector_dir.mkdir(parents=True, exist_ok=True)
    output_raster_dir.mkdir(parents=True, exist_ok=True)

    # Find all SVG files in input directory (recursively)
    svg_files = list(input_dir.rglob("*.svg"))
    
    if not svg_files:
        raise ValueError(f"No SVG files found in {input_dir}")

    print(f"Found {len(svg_files)} SVG files to process")

    for svg_path in tqdm(svg_files, desc="Processing SVG files"):
        try:
            # Read SVG content
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()

            # Process SVG
            processed_svg = process_svg_content(svg_content)
            
            # Generate output paths (preserve relative structure)
            relative_path = svg_path.relative_to(input_dir)
            sample_id = relative_path.stem  # filename without extension
            
            # Create subdirectories if needed
            vector_subdir = output_vector_dir / relative_path.parent
            raster_subdir = output_raster_dir / relative_path.parent
            vector_subdir.mkdir(parents=True, exist_ok=True)
            raster_subdir.mkdir(parents=True, exist_ok=True)
            
            # Save processed SVG
            output_svg_path = vector_subdir / f"{sample_id}.svg"
            with open(output_svg_path, 'w', encoding='utf-8') as f:
                f.write(processed_svg)

            # Render PNG
            output_png_path = raster_subdir / f"{sample_id}.png"
            cairosvg.svg2png(bytestring=processed_svg.encode('utf-8'), write_to=str(output_png_path), output_width=1000, output_height=1000)

        except Exception as e:
            print(f"Error processing {svg_path}: {e}")
            continue

    print("FloorPlanCAD processing complete!")


if __name__ == '__main__':
    main()