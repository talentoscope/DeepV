# DeepV Web UI

A Gradio-based web interface for demonstrating DeepV's vectorization capabilities.

## Current Status

The web UI framework is implemented but currently has deployment issues due to environment conflicts with Gradio and the conda environment. A command-line demo is available as a workaround.

## Features

- **Interactive Demo**: Upload technical drawings and see rendering results
- **Rendering Comparison**: Compare analytical vs Bézier splatting rendering methods
- **SVG Export**: Generate SVG output for CAD software
- **Real-time Processing**: Fast rendering using optimized algorithms

## Installation

1. Install the development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run the web interface:
```bash
python run_web_ui.py
```

Note: Due to environment conflicts, the Gradio interface may not start. Use the command-line demo instead.

## Command-Line Demo

Use the command-line demo for testing functionality:

```bash
python run_web_ui_demo.py --image path/to/image.png --output output.svg --method bezier
```

### Demo Parameters

- `--image, -i`: Path to input image (required)
- `--output, -o`: Path to save SVG output (optional)
- `--method, -m`: Vectorization method (`analytical` or `bezier`, default: `analytical`)

### Example

```bash
python run_web_ui_demo.py --image test_image.png --output result.svg --method bezier
```

## Usage

1. Upload a technical drawing image (PNG/JPG)
2. Select rendering method:
   - **Hard (Analytical)**: Traditional exact geometry rendering
   - **Bézier Splatting**: Fast differentiable rendering with anti-aliasing
3. Choose primitive type (currently lines only in demo)
4. Click "Render Demo" to see the result
5. View the rendered image and download SVG output

## Demo vs Full Pipeline

This web UI provides a **demonstration** of DeepV's rendering capabilities using synthetic primitives. The full DeepV pipeline requires:

- Trained machine learning models
- Specific data formats
- GPU acceleration for optimal performance

For production vectorization, use the main `run_pipeline.py` script.

## Architecture

The web UI demonstrates:
- Bézier Splatting rendering (Phase 2 optimization)
- SVG generation from vector primitives
- Real-time image processing
- Gradio-based interactive interface

## Future Enhancements

- Integration with full ML pipeline
- Support for curves and arcs
- Batch processing
- Model selection and configuration
- Performance metrics display
- Resolution of environment deployment issues