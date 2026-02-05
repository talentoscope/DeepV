# DeepV CAD Export Module

This module provides comprehensive CAD export functionality for DeepV vectorized primitives, enabling seamless integration with professional CAD software and design workflows.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Supported Formats](#supported-formats)
- [API Reference](#api-reference)
- [File Structure](#file-structure)
- [Integration](#integration)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## Features

- **DXF Export**: Export line primitives to Drawing Exchange Format (DXF) for CAD software
- **SVG Export**: Export to Scalable Vector Graphics format for web and vector editors
- **Multi-primitive Support**: Lines, curves, arcs, and Bézier splines
- **Validation**: Built-in validation of exported CAD files
- **Batch Processing**: Efficient export of multiple primitives
- **Extensible Framework**: Easy to add support for new primitive types and formats

## Installation

### Required Dependencies

```bash
# Core CAD export dependencies
pip install ezdxf      # For DXF export
pip install svgwrite   # For SVG export (if not in requirements)

# Optional but recommended
pip install numpy      # For array operations
```

### Verification

```python
# Test CAD export functionality
python -c "from cad.export import export_to_dxf; print('CAD export ready')"
```

## Quick Start

### Basic Export Example

```python
from cad.export import create_cad_from_primitives
import numpy as np

# Create line primitives (format: [x1, y1, x2, y2, width])
lines = np.array([
    [10.0, 20.0, 100.0, 20.0, 2.0],  # horizontal line
    [50.0, 10.0, 50.0, 100.0, 1.5],  # vertical line
])

# Export to both DXF and SVG
results = create_cad_from_primitives({'lines': lines}, output_dir='output/')
print(results)
```

### Command Line Usage

```bash
# Export from pipeline results
python run_pipeline.py --export_cad --output_dir results/
```

## Usage

### Basic Export

```python
from cad.export import create_cad_from_primitives
import numpy as np

# Line primitives: [x1, y1, x2, y2, width]
lines = np.array([
    [10.0, 20.0, 100.0, 20.0, 2.0],
    [50.0, 10.0, 50.0, 100.0, 1.5],
])

# Curve primitives: [x1, y1, cx1, cy1, cx2, cy2, x2, y2, width]
curves = np.array([
    [0.0, 0.0, 50.0, -25.0, 100.0, 25.0, 150.0, 0.0, 2.0],
])

# Export all primitives
primitives = {'lines': lines, 'curves': curves}
results = create_cad_from_primitives(primitives, output_dir='output/')
```

### Direct Export Functions

```python
from cad.export import export_to_dxf, export_to_svg

# Export lines to DXF
export_to_dxf(lines, 'drawing.dxf', width=256, height=256)

# Export curves to SVG
export_to_svg(curves, 'drawing.svg', width=256, height=256, primitive_type='curves')
```

### Advanced Usage

```python
# Custom export options
from cad.export import CADExporter

exporter = CADExporter(width=512, height=512, scale_factor=2.0)
exporter.add_lines(lines)
exporter.add_curves(curves)
exporter.export_all('output/', formats=['dxf', 'svg'])
```

## Supported Formats

### DXF (Drawing Exchange Format)
- **Compatibility**: AutoCAD, FreeCAD, DraftSight, and most CAD software
- **Features**: Geometric entities as lines, polylines, and splines
- **Coordinate System**: Origin at bottom-left, Y-axis pointing up
- **Layers**: Automatic layer assignment by primitive type

### SVG (Scalable Vector Graphics)
- **Compatibility**: Web browsers, Inkscape, Adobe Illustrator
- **Features**: Stroke width, colors, and vector scaling
- **Advantages**: Web-native, human-readable XML format
- **Limitations**: No native CAD metadata support

### Format Comparison

| Feature | DXF | SVG |
|---------|-----|-----|
| CAD Software | ✅ Native | ❌ Limited |
| Web Viewing | ❌ Requires viewer | ✅ Native |
| File Size | Compact | Moderate |
| Metadata | Rich | Basic |
| Curves | Full support | Good support |

## API Reference

### Main Functions

#### `create_cad_from_primitives(primitives, output_dir, formats=['dxf', 'svg'])`
Creates CAD files from primitive dictionaries.

**Parameters:**
- `primitives`: Dict with keys 'lines', 'curves', etc. containing numpy arrays
- `output_dir`: Directory to save output files
- `formats`: List of formats to export ('dxf', 'svg')

**Returns:** Dict with export results and file paths

#### `export_to_dxf(primitives, filename, width=256, height=256)`
Direct DXF export function.

#### `export_to_svg(primitives, filename, width=256, height=256, primitive_type='lines')`
Direct SVG export function.

### Classes

#### `CADExporter`
Advanced exporter class with fine-grained control.

```python
exporter = CADExporter(width=512, height=512)
exporter.add_lines(line_array)
exporter.add_curves(curve_array)
exporter.export_all(output_dir)
```

## File Structure

```
cad/
├── export.py          # Main export functions and CADExporter class
└── README.md          # This documentation
```

## Integration

The CAD export module integrates seamlessly with other DeepV components:

### Pipeline Integration
```python
# From run_pipeline.py
from cad.export import create_cad_from_primitives

# After pipeline processing
cad_results = create_cad_from_primitives(
    pipeline_output.primitives,
    output_dir='results/cad/'
)
```

### Web UI Integration
- Provides DXF/SVG download options in the Gradio interface
- Real-time export during interactive sessions
- Batch export for multiple processed images

### Validation Integration
- Automatic file validation after export
- Error reporting for malformed primitives
- Compatibility checks for target CAD software

## Troubleshooting

### Common Issues

**DXF files not opening in CAD software**:
- Check coordinate system (DeepV uses bottom-left origin)
- Verify primitive data format
- Try different CAD software (FreeCAD vs AutoCAD)

**SVG scaling issues**:
- Check viewBox attribute in SVG header
- Verify width/height parameters match image dimensions
- Use vector editors that support SVG scaling

**Memory errors with large exports**:
- Process primitives in batches
- Use `CADExporter` for memory-efficient streaming
- Reduce export resolution if needed

### Validation

```python
# Validate exported files
from cad.export import validate_cad_file

is_valid = validate_cad_file('output/drawing.dxf', format='dxf')
print(f"File valid: {is_valid}")
```

## Future Enhancements

- [ ] **Advanced Primitives**: Full arc, circle, and ellipse support
- [ ] **Text Annotations**: Embedded text and dimensioning
- [ ] **Layer Management**: Custom layers and grouping
- [ ] **DWG Export**: Native AutoCAD format support
- [ ] **Parametric CAD**: Conversion to solid modeling formats
- [ ] **3D Export**: Extrusion and 3D CAD formats (STEP, IGES)
- [ ] **Color and Styling**: Advanced line styles and colors
- [ ] **Metadata**: Embedded processing information and timestamps