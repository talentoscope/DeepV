# DeepV Web UI

A Gradio-based web interface for demonstrating DeepV's vectorization and rendering capabilities.

## Table of Contents

- [Current Status](#current-status)
- [Features](#features)
- [Installation](#installation)
- [Command-Line Demo](#command-line-demo)
- [Web Interface Usage](#web-interface-usage)
- [Demo vs Full Pipeline](#demo-vs-full-pipeline)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## Current Status

The web UI framework is implemented but currently has deployment issues due to environment conflicts with Gradio and the conda environment. A command-line demo is available as a workaround.

**Status**: ⚠️ Functional but with deployment limitations

## Features

- **Interactive Demo**: Upload technical drawings and see rendering results
- **Rendering Comparison**: Compare analytical vs Bézier splatting rendering methods
- **SVG Export**: Generate SVG output for CAD software
- **Real-time Processing**: Fast rendering using optimized algorithms
- **Multiple Rendering Methods**: Hard analytical vs soft Bézier splatting
- **Primitive Support**: Lines, curves, and Bézier primitives

## Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (recommended)
- Gradio and related dependencies

### Setup Steps

1. **Install development dependencies**:
```bash
pip install -r requirements-dev.txt
```

2. **Verify environment**:
```bash
python scripts/validate_env.py
```

3. **Run the web interface**:
```bash
python run_web_ui.py
```

**Note**: Due to environment conflicts, the Gradio interface may not start. Use the command-line demo instead.

## Command-Line Demo

Use the command-line demo for testing functionality when the web UI has deployment issues:

```bash
python run_web_ui_demo.py --image path/to/image.png --output output.svg --method bezier
```

### Demo Parameters

| Parameter | Short | Description | Required |
|-----------|-------|-------------|----------|
| `--image` | `-i` | Path to input image (PNG/JPG) | Yes |
| `--output` | `-o` | Path to save SVG output | No |
| `--method` | `-m` | Vectorization method (`analytical` or `bezier`) | No (default: `analytical`) |

### Examples

**Basic usage**:
```bash
python run_web_ui_demo.py --image test_image.png
```

**With Bézier rendering**:
```bash
python run_web_ui_demo.py --image test_image.png --output result.svg --method bezier
```

**Batch processing**:
```bash
for img in images/*.png; do
  python run_web_ui_demo.py -i "$img" -o "output/$(basename "$img" .png).svg" -m bezier
done
```

## Web Interface Usage

1. **Upload Image**: Select a technical drawing image (PNG/JPG format)
2. **Select Rendering Method**:
   - **Hard (Analytical)**: Traditional exact geometry rendering
   - **Bézier Splatting**: Fast differentiable rendering with anti-aliasing
3. **Choose Primitive Type**: Currently lines only in demo (full pipeline supports curves/arcs)
4. **Render**: Click "Render Demo" to process the image
5. **View Results**: See the rendered image and download SVG output

## Demo vs Full Pipeline

This web UI provides a **demonstration** of DeepV's rendering capabilities using synthetic primitives. The full DeepV pipeline requires:

- Trained machine learning models
- Specific data formats
- GPU acceleration for optimal performance
- Complete cleaning → vectorization → refinement → merging pipeline

### Comparison Table

| Feature | Web UI Demo | Full Pipeline |
|---------|-------------|----------------|
| Input | Single image upload | Batch processing, datasets |
| Models | Pre-trained demo models | Custom trained models |
| Output | SVG rendering demo | Complete CAD export (DXF/SVG) |
| Performance | Real-time rendering | Production vectorization |
| Primitives | Lines only | Lines, curves, arcs, splines |

For production vectorization, use the main `run_pipeline.py` script.

## Architecture

The web UI demonstrates:
- **Bézier Splatting rendering**: Phase 2 optimization for fast differentiable rendering
- **SVG generation**: From vector primitives to CAD-compatible output
- **Real-time processing**: Optimized image processing pipeline
- **Gradio interface**: Interactive web-based demonstration

### Key Components
- `run_web_ui.py`: Main Gradio interface
- `run_web_ui_demo.py`: Command-line demo fallback
- Bézier splatting renderer (GPU-accelerated)
- SVG export utilities

## Troubleshooting

### Common Issues

**Gradio interface won't start**:
- Use command-line demo: `python run_web_ui_demo.py`
- Check conda environment conflicts
- Try running in fresh virtual environment

**CUDA/GPU issues**:
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Import errors**:
- Run environment validation: `python scripts/validate_env.py`
- Check dependencies: `pip install -r requirements-dev.txt`

### Performance Tips
- Use GPU for faster rendering
- Reduce image resolution for quicker processing
- Bézier splatting is faster than analytical rendering

## Future Enhancements

- [ ] Integration with full ML pipeline
- [ ] Support for curves and arcs in demo
- [ ] Batch processing interface
- [ ] Model selection and configuration options
- [ ] Performance metrics display
- [ ] Resolution of Gradio deployment issues
- [ ] Mobile-responsive interface
- [ ] Custom primitive type selection