# DeepV Utilities

This directory contains shared utility functions and helpers used throughout the DeepV codebase.

## Modules

### Core Utilities

#### `file_utils.py`
File system operations and path management.

**Key Functions:**
- `require_empty()`: Ensure directory is empty
- `safe_makedirs()`: Create directories safely
- `find_files()`: Recursively find files by pattern

**Usage:**
```python
from util_files import file_utils as fu

# Create directory if it doesn't exist
fu.safe_makedirs('output/results')

# Find all PNG files
png_files = fu.find_files('data/', '*.png')
```

#### `config_manager.py`
Hydra-based configuration management.

**Features:**
- Hierarchical configuration files
- Environment variable overrides
- Runtime configuration merging

**Usage:**
```python
from util_files.config_manager import get_config

# Load configuration
cfg = get_config()

# Load with overrides
cfg = get_config(overrides=['gpu=0', 'batch_size=16'])
```

### Rendering

#### `rendering/cairo.py`
Cairo-based rendering for vector primitives.

**Supported Primitives:**
- Lines
- Quadratic Bézier curves
- Cubic Bézier curves
- Arcs and circles

**Usage:**
```python
from util_files.rendering.cairo import render_primitives

# Render primitives to image
image = render_primitives(primitives, width=256, height=256)
```

#### `rendering/bezier_splatting.py`
Fast Bézier splatting rendering using Gaussian kernels.

**Advantages:**
- GPU-accelerated
- Differentiable
- Fast rendering for optimization

### Optimization

#### `optimization/primitives/`
Primitive-specific optimization classes.

**Available Primitives:**
- `line_tensor.py`: Line primitive optimization
- `curve_tensor.py`: Curve primitive optimization
- `arc_tensor.py`: Arc primitive optimization

**Usage:**
```python
from util_files.optimization.primitives.line_tensor import LineTensor

# Create line tensor for optimization
lines = LineTensor(p1, p2, width)
```

#### `optimization/energy/`
Energy computation for differentiable rendering.

**Methods:**
- Analytical rendering
- Bézier splatting
- Hybrid approaches

### Loss Functions

#### `loss_functions/supervised.py`
Supervised loss functions for training.

**Loss Types:**
- Primitive prediction losses
- Geometric constraint losses
- Regularization terms

#### `loss_functions/semi_supervised.py`
Semi-supervised and unsupervised losses.

**Applications:**
- Self-supervised pretraining
- Domain adaptation
- Weak supervision

### Data Processing

#### `patchify.py`
Image patching utilities for processing large images.

**Features:**
- Overlapping patches
- Memory-efficient processing
- Batch processing support

**Usage:**
```python
from util_files.patchify import patchify

# Split image into patches
patches = patchify(image, patch_size=(64, 64))
```

## Development Notes

### File Organization
- Keep utility functions focused and reusable
- Add comprehensive docstrings
- Include type hints where possible
- Write unit tests for critical functions

### Import Conventions
```python
# Preferred imports
from util_files import file_utils as fu
from util_files.config_manager import get_config

# Avoid shadowing stdlib
# DON'T: from util_files.os import *  # Old pattern
```

### Adding New Utilities
1. Create new module in appropriate subdirectory
2. Add comprehensive docstrings and type hints
3. Write unit tests
4. Update this README
5. Update imports in dependent modules

## Testing

Run utility tests:
```bash
pytest tests/test_file_utils.py
pytest tests/test_config_manager.py
```

## Performance Considerations

- Profile utility functions for bottlenecks
- Use vectorized operations where possible
- Consider GPU acceleration for rendering functions
- Optimize memory usage for large datasets

## Contributing

When modifying utilities:
- Maintain backward compatibility
- Update dependent code
- Add tests for new functionality
- Update documentation