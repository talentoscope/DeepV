# DeepV CAD Export

This module provides CAD export functionality for DeepV vectorized primitives.

## Features

- **DXF Export**: Export line primitives to Drawing Exchange Format (DXF) for CAD software
- **SVG Export**: Export to Scalable Vector Graphics format
- **Validation**: Built-in validation of exported CAD files
- **Extensible**: Framework for adding support for curves, arcs, and other primitives

## Installation

```bash
pip install ezdxf  # For DXF export
```

## Usage

### Basic Export

```python
from cad.export import create_cad_from_primitives
import numpy as np

# Create some line primitives
lines = np.array([
    [10.0, 20.0, 100.0, 20.0, 2.0],  # horizontal line
    [50.0, 10.0, 50.0, 100.0, 1.5],  # vertical line
])

# Export to CAD formats
results = create_cad_from_primitives({'lines': lines}, output_dir='output')

print(results)
# {'dxf': {'path': 'output/primitives.dxf', 'success': True},
#  'svg': {'path': 'output/primitives.svg', 'success': True}}
```

### Direct Export Functions

```python
from cad.export import export_to_dxf, export_to_svg

# Export to DXF
export_to_dxf(lines, 'drawing.dxf', width=256, height=256)

# Export to SVG
export_to_svg(lines, 'drawing.svg', width=256, height=256)
```

## Supported Formats

### DXF (Drawing Exchange Format)
- Widely supported by CAD software (AutoCAD, FreeCAD, etc.)
- Contains geometric entities as lines
- Coordinate system: Origin at bottom-left, Y-axis flipped

### SVG (Scalable Vector Graphics)
- Web-compatible vector format
- Can be viewed in browsers and most vector editors
- Includes stroke width information

## File Structure

```
cad/
├── export.py          # Main export functions
└── README.md          # This file
```

## Future Enhancements

- Support for curves and Bézier splines
- Arc and circle primitives
- Text annotations
- Layer support
- DWG format export
- Parametric CAD conversion

## Integration

The CAD export module integrates with:

- **Web UI**: Provides DXF/SVG download options
- **Pipeline**: Can be called from `run_pipeline.py` for batch export
- **Validation**: Includes file validation to ensure export integrity

## Dependencies

- `ezdxf`: For DXF file creation
- `numpy`: For array operations (optional, for torch compatibility)