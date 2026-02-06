# DeepV: Deep Vectorization of Technical Drawings

[![Paper](https://img.shields.io/badge/arXiv-2003.05471-b31b1b.svg)](https://arxiv.org/abs/2003.05471)
[![Video](https://img.shields.io/badge/YouTube-Demo-red)](https://www.youtube.com/watch?v=lnQNzHJOLvE)
[![Slides](https://img.shields.io/badge/Google%20Slides-Presentation-blue)](https://drive.google.com/file/d1ZrykQeA2PE4_8yf1JwuEBk9sS4OP8KeM/view?usp=sharing)

**DeepV** - Modern PyTorch implementation of deep vectorization for technical drawings. Based on the ECCV 2020 paper: **Deep Vectorization of Technical Drawings**

![DeepV Pipeline](https://drive.google.com/uc?export=view&id=191r0QAaNhOUIaHPOlPWH5H4Jg7qxCMRA)

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarking & Evaluation](#benchmarking--evaluation)
- [Models](#models)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

DeepV is a deep learning framework for converting raster technical drawings into structured vector representations. The pipeline consists of four main modules:

1. **Cleaning**: Noise removal and artifact reduction using UNet-based models
2. **Vectorization**: Neural network prediction of geometric primitives from image patches
3. **Refinement**: Differentiable optimization to improve primitive accuracy
4. **Merging**: Consolidation of primitives and CAD export (DXF/SVG)

### Key Features

- **Extended Primitives**: Lines, quadratic/cubic Béziers, arcs, splines
- **Variable Length**: Up to 20 primitives per patch via autoregressive transformer
- **CAD Export**: Direct export to DXF and SVG formats
- **Interactive UI**: Gradio-based web interface with real-time rendering
- **Distributed Training**: Multi-GPU support with PyTorch distributed
- **Modern Tooling**: Hydra configuration, pytest testing, pre-commit hooks

## Repository Structure

The repository follows a modular structure for easy development and testing:

```
DeepV/
├── cleaning/           # UNet-based noise removal and preprocessing
├── vectorization/      # Neural network models for primitive prediction
├── refinement/         # Differentiable optimization of primitives
├── merging/           # Primitive consolidation and CAD export
├── dataset/           # Dataset downloaders and processors
├── notebooks/         # Jupyter notebooks for demos and experiments
├── util_files/        # Shared utilities and rendering functions
├── web_ui/            # Gradio-based interactive interface
├── scripts/           # Training, evaluation, and benchmarking scripts
├── config/            # Hydra configuration files
├── cad/               # CAD export functionality (DXF/SVG)
└── tests/             # Unit and integration tests
```

Each module contains its own README with detailed usage instructions.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support (recommended)
- Linux/Windows/macOS

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-repo/DeepV.git
cd DeepV

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install development dependencies
pip install -r requirements-dev.txt
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libcairo2-dev pkg-config python3-dev

# macOS
brew install cairo pkg-config

# Windows (via conda)
conda install cairo
```

## Benchmarking & Evaluation

DeepV includes comprehensive benchmarking against state-of-the-art baselines:

### Quick Benchmark

```bash
# Evaluate on synthetic dataset
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --deepv-model-path /path/to/model \
  --datasets synthetic

# Compare against baselines
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --models deepv baseline1 baseline2 \
  --datasets floorplancad archcad
```

### Supported Metrics

- **Vector Metrics**: F1 Score, IoU, Hausdorff Distance, Chamfer Distance
- **Raster Metrics**: PSNR, MSE, SSIM
- **CAD Metrics**: Parametric accuracy, topological correctness

See `scripts/README_benchmarking.md` for detailed usage.

## Models

### Pre-trained Models

| Model | Primitives | Download | Size |
|-------|------------|----------|------|
| Lines | Lines only | [Download](https://disk.yandex.ru/d/FKJuMvNJuy-K9g) | ~500MB |
| Curves | Lines + Béziers | [Download](https://disk.yandex.ru/d/yOZzCSrd-QSACA) | ~750MB |

### Model Architecture

- **Encoder**: ResNet-18 backbone with feature extraction
- **Decoder**: Autoregressive transformer (up to 20 primitives/patch)
- **Loss**: Supervised loss with geometric constraints
- **Training**: Distributed training with mixed precision

## Notebooks

Interactive Jupyter notebooks demonstrating key functionality:

1. **Rendering Examples** (`notebooks/Rendering_example.ipynb`)
   - Cairo-based rendering of primitives
   - Bézier splatting vs analytical rendering comparison

2. **Model Training** (`notebooks/Data_loading_and_model_training.ipynb`)
   - Dataset loading and preprocessing
   - Model training pipeline
   - Loss function visualization

3. **Pretrained Evaluation** (`notebooks/pretrain_model_loading_and_evaluation_for_lines.ipynb`)
   - Loading and evaluating pretrained models
   - Inference pipeline walkthrough

## Benchmarking and Evaluation

DeepV includes a comprehensive benchmarking pipeline for evaluating vectorization models across multiple datasets and comparing against state-of-the-art baselines.

### Quick Benchmarking

```bash
# Run evaluation on synthetic dataset
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --deepv-model-path /path/to/trained/model \
  --datasets synthetic

# Run comprehensive benchmark across multiple datasets
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --deepv-model-path /path/to/trained/model \
  --datasets dataset1 dataset2 dataset3 \
  --output-dir benchmark_results
```

### Supported Dataset Formats
- PNG + DXF format pairs (image + ground truth)
- SVG vector graphics
- PDF technical drawings
- Any custom dataset following standard directory structure

### Evaluation Metrics
- **Vector Metrics**: F1 Score, IoU, Hausdorff Distance, Chamfer Distance
- **Raster Metrics**: PSNR, MSE, SSIM
- **Comprehensive Reports**: Automated comparison against baselines

See `scripts/README_benchmarking.md` for detailed usage.

## Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for details on:

- Development setup and workflow
- Code quality standards
- Testing requirements
- Pull request process

### Quick Developer Setup

```bash
# Validate environment
python scripts/validate_env.py

# Run tests
pytest -q

# Format code
black .
isort .
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@InProceedings{egiazarian2020deep,
  title="Deep Vectorization of Technical Drawings",
  author="Egiazarian, Vage and Voynov, Oleg and Artemov, Alexey and Volkhonskiy, Denis and Safin, Aleksandr and Taktasheva, Maria and Zorin, Denis and Burnaev, Evgeny",
  booktitle="Computer Vision -- ECCV 2020",
  year="2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="582--598",
  isbn="978-3-030-58601-0"
}
```

---

## Requirements

**System Requirements:**
- Linux system
- Python 3.8+
- CUDA-compatible GPU (recommended)

**Key Dependencies:**
- PyTorch 2.0+
- torchvision
- cairo==1.14.12
- pycairo==1.19.1
- chamferdist==1.0.0
- ezdxf (for CAD export)
- gradio (for web UI)

See `requirements.txt` for complete list.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and virtual environment is activated
2. **CUDA Issues**: Check PyTorch CUDA compatibility with your GPU drivers
3. **Memory Errors**: Reduce batch size or use smaller patch sizes
4. **Rendering Issues**: Install system cairo library (`libcairo2-dev` on Ubuntu)

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-repo/DeepV/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/DeepV/discussions)
- **Documentation**: See module-specific READMEs and docstrings

---

*DeepV is actively maintained. For questions about specific modules, see the README in each subdirectory.*
