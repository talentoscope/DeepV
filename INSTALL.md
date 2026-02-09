# Installation Guide for DeepV

## Important: Current Status (February 2026)

⚠️ **Before Installing**: DeepV is currently in Phase 4 development (~70% complete). Here's what you should know:

**What Works Well**:
- ✅ Pipeline runs successfully on technical drawings
- ✅ Fast processing (70% speed improvement, 23s per image)
- ✅ CAD export to DXF/SVG formats
- ✅ Interactive Gradio demo for visualization

**Known Limitations**:
- ⚠️ **FloorPlanCAD Performance** (**#1 Priority**): Model shows poor performance on FloorPlanCAD dataset (IoU: 0.010, high over-segmentation)
  - See [DEVELOPMENT.md - Critical Priority](DEVELOPMENT.md#-critical-priority-floorplancad-performance-gap) for detailed analysis and improvement plan
- ⚠️ Degraded/noisy scanned images may produce poor results (active improvement area - see DEVELOPMENT.md)
- ⚠️ Web UI has deployment issues with some Gradio/conda combinations
- ⚠️ Color drawings are converted to grayscale

**Best Use Cases** (currently):
- Technical drawings and diagrams
- Research and experimentation
- Testing vectorization improvements

**Active Development**:
- Architecture improvements (Non-Autoregressive Transformer)
- Training optimization on FloorPlanCAD data
- Reducing over-segmentation and improving geometric accuracy

See [README.md](README.md) and [docs/PLAN.md](docs/PLAN.md) for detailed status and improvement roadmap.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Install](#quick-install)
- [System Dependencies](#system-dependencies)
- [GPU Support](#gpu-support)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Running the Pipeline](#running-the-pipeline)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Python**: 3.10 or later
- **Operating System**: Windows/Linux/macOS
- **Disk Space**: ~2GB for dependencies and models
- **RAM**: 8GB+ recommended for training, 4GB minimum

## Quick Install

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/DeepV.git
cd DeepV
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# (Optional) Install development dependencies
pip install -r requirements-dev.txt
```

## System Dependencies

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install libcairo2-dev pkg-config python3-dev build-essential
```

### macOS

```bash
# Using Homebrew
brew install cairo pkg-config

# Or using MacPorts
sudo port install cairo pkgconfig
```

### Windows

```bash
# Using conda (recommended)
conda install -c conda-forge cairo

# Or using vcpkg
vcpkg install cairo
```

## GPU Support

For CUDA acceleration, install PyTorch with CUDA support:

```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Note**: Match PyTorch CUDA version with your system's CUDA installation.

## Development Setup

For contributors and developers:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run environment validation
python scripts/validate_env.py
```

## Running Tests

```bash
# Run full test suite
python -m pytest

# Run specific tests
python -m pytest tests/test_file_utils.py

# Run with coverage
python -m pytest --cov=deepv

# Windows PowerShell helper
.\scripts\run_tests_local.ps1
```

## Running the Pipeline

After installation, test the pipeline:

```bash
# Download a sample model (replace with actual URLs)
wget -O models/sample_model.weights "https://example.com/model.weights"

# Run basic pipeline
python run_pipeline.py \
  --model_path models/sample_model.weights \
  --data_dir sample_data/ \
  --primitive_type line

# Run web demo
python run_web_ui_demo.py --image sample.png
```

See `DEVELOPER.md` for detailed usage examples.

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### CUDA/GPU Issues
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Cairo Rendering Issues
```bash
# Install system cairo
# Ubuntu/Debian:
sudo apt-get install libcairo2-dev

# macOS:
brew install cairo

# Windows:
conda install -c conda-forge cairo
```

#### Memory Issues
- Reduce batch size in configuration
- Use smaller patch sizes (64x64 instead of 128x128)
- Enable mixed precision training

#### Permission Errors
```bash
# Fix script permissions
chmod +x scripts/*.sh
chmod +x scripts/*.ps1
```

### Getting Help

- **Documentation**: Check module-specific READMEs
- **Issues**: [GitHub Issues](https://github.com/your-repo/DeepV/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/DeepV/discussions)

### Environment Validation

Run the environment validator to check your setup:

```bash
python scripts/validate_env.py
```

This will verify:
- Python version compatibility
- Required packages installation
- CUDA availability (if applicable)
- Basic import functionality