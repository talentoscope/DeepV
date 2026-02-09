#!/usr/bin/env python3
"""
Extract ground truth vectors from FloorPlanCAD SVG files for evaluation.

This script parses FloorPlanCAD SVG files and extracts geometric primitives
(lines, arcs, curves) from relevant layers (walls, structural elements) to
create ground truth vectors in the same format as DeepV outputs.
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import svgpathtools


def parse_svg_path_d(d_string: str) -> List[Tuple[str, List[float]]]:
    """Parse SVG path 'd' attribute into commands and coordinates."""
    # Split by command letters, keeping the letters
    commands = re.findall(r"[MLHVCSQTAZ][^MLHVCSQTAZ]*", d_string.upper())

    parsed_commands = []
    for cmd in commands:
        cmd_type = cmd[0]
        # Remove command letter and split by whitespace/comma
        coords_str = cmd[1:].strip()
        if coords_str:
            # Split by whitespace and commas, convert to float
            coords = []
            for part in re.split(r"[,\s]+", coords_str):
                part = part.strip()
                if part:
                    try:
                        coords.append(float(part))
                    except ValueError:
                        continue
            parsed_commands.append((cmd_type, coords))
        else:
            parsed_commands.append((cmd_type, []))

    return parsed_commands


def extract_line_segments_from_path(path_elem) -> List[np.ndarray]:
    """Extract line segments from an SVG path element."""
    lines = []
    if "d" not in path_elem.attrib:
        return lines

    d_string = path_elem.attrib["d"]
    try:
        # Use svgpathtools to parse the path
        path = svgpathtools.parse_path(d_string)

        # Extract line segments from the path
        current_pos = None
        for segment in path:
            if isinstance(segment, svgpathtools.Line):
                x1, y1 = segment.start.real, segment.start.imag
                x2, y2 = segment.end.real, segment.end.imag

                # Get stroke width, default to 0.1 if not specified
                stroke_width = 0.1
                if "stroke-width" in path_elem.attrib:
                    try:
                        stroke_width = float(path_elem.attrib["stroke-width"])
                    except ValueError:
                        pass

                line = np.array([x1, y1, x2, y2, stroke_width])
                lines.append(line)

            elif isinstance(segment, svgpathtools.Arc):
                # For arcs, approximate with line segments or skip for now
                # Could implement arc to line conversion later
                continue

            elif isinstance(segment, svgpathtools.CubicBezier) or isinstance(segment, svgpathtools.QuadraticBezier):
                # For curves, approximate with line segments or skip
                continue

    except Exception as e:
        print(f"Warning: Failed to parse path: {e}")
        return lines

    return lines


def extract_ground_truth_vectors(svg_file: str, relevant_layers: List[str] = None) -> np.ndarray:
    """
    Extract ground truth vectors from FloorPlanCAD SVG file.

    Args:
        svg_file: Path to SVG file
        relevant_layers: List of layer IDs to extract from (e.g., ['A-WALL', 'DIM_LEAD'])
                         If None, extracts from all layers

    Returns:
        Array of shape (N, 5) with [x1, y1, x2, y2, width] for each line segment
    """
    if relevant_layers is None:
        # Default to structural layers
        relevant_layers = ["A-WALL", "DOOR_FIRE", "WINDOW"]

    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Define namespaces
        ns = {"inkscape": "http://www.inkscape.org/namespaces/inkscape", "svg": "http://www.w3.org/2000/svg"}

        # Debug: print all layer IDs
        all_layers = root.findall(
            ".//{http://www.w3.org/2000/svg}g[@{http://www.inkscape.org/namespaces/inkscape}groupmode='layer']"
        )
        print(f"Found {len(all_layers)} layers total:")
        for layer in all_layers:
            layer_id = layer.get("{http://www.w3.org/2000/svg}id", "no-id")
            label = layer.get("{http://www.inkscape.org/namespaces/inkscape}label", "no-label")
            print(f"  Layer ID: '{layer_id}', Label: '{label}'")

        all_lines = []

        # Find all path elements in relevant layers
        for layer_label in relevant_layers:
            # Find layer groups by label (since id is empty)
            layer_groups = root.findall(
                f".//{{http://www.w3.org/2000/svg}}g[@{{http://www.inkscape.org/namespaces/inkscape}}label='{layer_label}']"
            )
            print(f"Found {len(layer_groups)} groups with label '{layer_label}'")
            for layer_group in layer_groups:
                # Find all path elements in this layer
                paths = layer_group.findall(".//{http://www.w3.org/2000/svg}path")
                print(f"Found {len(paths)} path elements in layer {layer_label}")
                for path_elem in paths:
                    lines = extract_line_segments_from_path(path_elem)
                    all_lines.extend(lines)

        if not all_lines:
            print(f"Warning: No lines found in relevant layers {relevant_layers}")
            return np.array([]).reshape(0, 5)

        # Convert to numpy array
        vectors = np.array(all_lines)

        # Normalize coordinates if needed (SVG viewBox is 0-100)
        # For now, keep in SVG coordinate system

        print(f"Extracted {len(vectors)} line segments from {svg_file}")
        return vectors

    except Exception as e:
        print(f"Error parsing SVG {svg_file}: {e}")
        return np.array([]).reshape(0, 5)


def convert_svg_coords_to_pixel(
    vectors: np.ndarray, svg_viewbox: Tuple[float, float, float, float], target_size: Tuple[int, int] = (1000, 1000)
) -> np.ndarray:
    """
    Convert SVG coordinates to pixel coordinates.

    Args:
        vectors: Array of shape (N, 5) with SVG coordinates
        svg_viewbox: (min_x, min_y, width, height) of SVG viewBox
        target_size: Target pixel dimensions (width, height)

    Returns:
        Vectors in pixel coordinates
    """
    if vectors.size == 0:
        return vectors

    min_x, min_y, svg_width, svg_height = svg_viewbox
    target_width, target_height = target_size

    # Scale factors
    scale_x = target_width / svg_width
    scale_y = target_height / svg_height

    # Convert coordinates
    pixel_vectors = vectors.copy()
    pixel_vectors[:, 0] = (vectors[:, 0] - min_x) * scale_x  # x1
    pixel_vectors[:, 1] = (vectors[:, 1] - min_y) * scale_y  # y1
    pixel_vectors[:, 2] = (vectors[:, 2] - min_x) * scale_x  # x2
    pixel_vectors[:, 3] = (vectors[:, 3] - min_y) * scale_y  # y2

    # Scale stroke widths proportionally to maintain visual consistency
    pixel_vectors[:, 4] *= min(scale_x, scale_y)

    return pixel_vectors


def get_svg_viewbox(svg_file: str) -> Tuple[float, float, float, float]:
    """Extract viewBox from SVG file."""
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        if "viewBox" in root.attrib:
            viewbox_str = root.attrib["viewBox"]
            parts = viewbox_str.split()
            if len(parts) == 4:
                return tuple(float(x) for x in parts)

        # Default viewBox if not found
        return (0, 0, 100, 100)

    except Exception as e:
        print(f"Warning: Could not extract viewBox: {e}")
        return (0, 0, 100, 100)


def main():
    parser = argparse.ArgumentParser(description="Extract ground truth vectors from FloorPlanCAD SVG")
    parser.add_argument("--svg_file", required=True, help="Path to SVG file")
    parser.add_argument("--output_file", required=True, help="Output numpy file path")
    parser.add_argument("--layers", nargs="+", default=["A-WALL", "DOOR_FIRE", "WINDOW"], help="Layers to extract from")
    parser.add_argument(
        "--target_size", nargs=2, type=int, default=[1000, 1000], help="Target pixel dimensions (width height)"
    )

    args = parser.parse_args()

    # Extract ground truth vectors
    gt_vectors = extract_ground_truth_vectors(args.svg_file, args.layers)

    if gt_vectors.size == 0:
        print("No vectors extracted, creating empty file")
        np.save(args.output_file, np.array([]).reshape(0, 5))
        return

    # Get SVG viewBox
    viewbox = get_svg_viewbox(args.svg_file)
    print(f"SVG viewBox: {viewbox}")

    # Convert to pixel coordinates
    pixel_vectors = convert_svg_coords_to_pixel(gt_vectors, viewbox, tuple(args.target_size))

    # Add confidence column (1.0 for ground truth)
    confidence = np.ones((pixel_vectors.shape[0], 1))
    final_vectors = np.hstack([pixel_vectors, confidence])

    # Save to file
    np.save(args.output_file, final_vectors)
    print(f"Saved {len(final_vectors)} ground truth vectors to {args.output_file}")
    print(f"Shape: {final_vectors.shape}")
    print(f"Sample vectors:")
    print(final_vectors[:5])


if __name__ == "__main__":
    main()
