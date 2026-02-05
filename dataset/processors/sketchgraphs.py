"""Processor for SketchGraphs Dataset."""

from pathlib import Path
from typing import Dict, Any

from .base import Processor


class SketchGraphsProcessor(Processor):
    """Process SketchGraphs dataset into vector/raster format.

    Extracts geometric primitives from constraint graphs and converts
    them to SVG format for vectorization tasks.
    """

    def standardize(self, input_dir: Path, output_base: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Process SketchGraphs dataset files."""
        vector_dir = output_base / "vector" / "sketchgraphs"
        raster_dir = output_base / "raster" / "sketchgraphs"

        if not dry_run:
            vector_dir.mkdir(parents=True, exist_ok=True)
            raster_dir.mkdir(parents=True, exist_ok=True)

        svg_count = 0
        png_count = 0

        # Look for .npy sequence files
        npy_files = list(input_dir.glob("*.npy"))
        print(f"Found {len(npy_files)} .npy sequence files")

        for npy_file in npy_files[:1] if dry_run else npy_files:  # Limit for dry run
            split_name = npy_file.stem  # e.g., "sg_t16_train", "sg_t16_validation", etc.
            print(f"Processing {split_name}...")

            try:
                # Load the sequence data
                from sketchgraphs.data import flat_array
                data = flat_array.load_dictionary_flat(str(npy_file))

                sequences = data['sequences']
                sequence_lengths = data['sequence_lengths']
                sketch_ids = data['sketch_ids']

                print(f"Loaded {len(sequences)} sequences from {split_name}")

                # Process a subset of sequences (first 100 per file for dry run, more for full run)
                max_sequences = 100 if dry_run else min(1000, len(sequences))

                for i in range(max_sequences):
                    sequence = sequences[i]
                    seq_length = sequence_lengths[i]
                    sketch_id = sketch_ids[i]

                    # Create a unique ID for this sketch
                    sketch_unique_id = f"{split_name}_{sketch_id[0]}_{sketch_id[1]}_{sketch_id[2]}"

                    try:
                        # Convert sequence to sketch
                        # Reconstruct the sketch from the sequence
                        # This is a simplified approach - in practice, you'd need to
                        # properly decode the sequence using the sketchgraphs library
                        sketch = self._sequence_to_sketch(sequence[:seq_length])

                        if sketch:
                            # Create SVG from sketch entities
                            svg_content = self._create_svg_from_sketch(sketch, sketch_unique_id)

                            if svg_content and not dry_run:
                                svg_path = vector_dir / f"{sketch_unique_id}.svg"
                                with open(svg_path, 'w') as f:
                                    f.write(svg_content)
                                svg_count += 1

                    except Exception as e:
                        print(f"Error processing sequence {i}: {e}")
                        continue

            except Exception as e:
                print(f"Error processing {npy_file}: {e}")
                continue

        return {
            "dataset": "sketchgraphs",
            "svg_count": svg_count,
            "png_count": png_count,
            "vec_dir": str(vector_dir),
            "ras_dir": str(raster_dir),
            "dry_run": dry_run,
        }

    def _sequence_to_sketch(self, sequence):
        """Convert a sequence back to a Sketch object."""
        try:
            from sketchgraphs.data.sequence import sketch_from_sequence
            sketch = sketch_from_sequence(sequence)
            return sketch
        except Exception as e:
            print(f"Error converting sequence to sketch: {e}")
            return None

    def _create_svg_from_sketch(self, sketch, sketch_id: str) -> str:
        """Create SVG from SketchGraphs Sketch object."""
        try:
            svg_elements = []
            width, height = 1000, 800

            # Collect all points to calculate bounds
            all_points = []

            for entity in sketch.entities.values():
                entity_type = type(entity).__name__

                if entity_type == 'Line':
                    # Line has start_point and end_point
                    start = entity.start_point
                    end = entity.end_point
                    all_points.extend([start, end])

                elif entity_type == 'Circle':
                    # Circle has center and radius
                    center = (entity.xCenter, entity.yCenter)
                    radius = entity.radius
                    # Add points on the circle for bounds calculation
                    all_points.append((center[0] - radius, center[1] - radius))
                    all_points.append((center[0] + radius, center[1] + radius))

                elif entity_type == 'Point':
                    # Point has coordinates (need to check attribute name)
                    if hasattr(entity, 'x') and hasattr(entity, 'y'):
                        all_points.append((entity.x, entity.y))
                    elif hasattr(entity, 'coordinates'):
                        all_points.append(entity.coordinates)

            if not all_points:
                return None

            # Calculate bounds
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            bounds_min_x, bounds_max_x = min(xs), max(xs)
            bounds_min_y, bounds_max_y = min(ys), max(ys)

            # Calculate scaling
            geom_width = bounds_max_x - bounds_min_x
            geom_height = bounds_max_y - bounds_min_y

            if geom_width == 0 or geom_height == 0:
                return None

            scale = min(width / geom_width, height / geom_height) * 0.8  # 80% of available space

            # Calculate offset to center the drawing
            offset_x = bounds_min_x
            offset_y = bounds_min_y

            # Create SVG elements for each entity
            for entity in sketch.entities.values():
                element = self._entity_to_svg_element(entity, offset_x, offset_y, scale, width, height)
                if element:
                    svg_elements.append(element)

            if not svg_elements:
                return None

            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>
  {"".join(svg_elements)}
  <text x="10" y="20" font-size="12" fill="black">SketchGraphs {sketch_id}</text>
</svg>'''

            return svg_content

        except Exception as e:
            print(f"Error creating SVG for sketch {sketch_id}: {e}")
            return None

    def _entity_to_svg_element(self, entity, offset_x, offset_y, scale, svg_width, svg_height):
        """Convert a SketchGraphs entity to an SVG element."""
        try:
            entity_type = type(entity).__name__

            if entity_type == 'Line':
                # Convert line to SVG
                start = entity.start_point
                end = entity.end_point

                # Transform coordinates
                x1 = (start[0] - offset_x) * scale + svg_width * 0.1
                y1 = svg_height - ((start[1] - offset_y) * scale + svg_height * 0.1)  # Flip Y
                x2 = (end[0] - offset_x) * scale + svg_width * 0.1
                y2 = svg_height - ((end[1] - offset_y) * scale + svg_height * 0.1)  # Flip Y

                return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="2"/>'

            elif entity_type == 'Circle':
                # Convert circle to SVG
                cx = (entity.xCenter - offset_x) * scale + svg_width * 0.1
                cy = svg_height - ((entity.yCenter - offset_y) * scale + svg_height * 0.1)  # Flip Y
                r = entity.radius * scale

                return f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="black" stroke-width="2" fill="none"/>'

            elif entity_type == 'Point':
                # Convert point to SVG
                if hasattr(entity, 'x') and hasattr(entity, 'y'):
                    px = (entity.x - offset_x) * scale + svg_width * 0.1
                    py = svg_height - ((entity.y - offset_y) * scale + svg_height * 0.1)  # Flip Y
                elif hasattr(entity, 'coordinates'):
                    px = (entity.coordinates[0] - offset_x) * scale + svg_width * 0.1
                    py = svg_height - ((entity.coordinates[1] - offset_y) * scale + svg_height * 0.1)  # Flip Y
                else:
                    return None

                return f'<circle cx="{px}" cy="{py}" r="3" fill="red"/>'

            # Skip other entity types for now
            return None

        except Exception as e:
            print(f"Error converting entity {type(entity).__name__} to SVG: {e}")
            return None
