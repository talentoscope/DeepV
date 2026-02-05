# DeepV Benchmarking Pipeline

A comprehensive evaluation framework for benchmarking DeepV vectorization models against state-of-the-art baselines across multiple datasets and metrics.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Supported Datasets](#supported-datasets)
- [Supported Models](#supported-models)
- [Metrics](#metrics)
- [Output Structure](#output-structure)
- [Usage Examples](#usage-examples)
- [Extending the Pipeline](#extending-the-pipeline)
- [Integration](#integration)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)

---

## Overview

The benchmarking pipeline provides:

- **Multi-dataset evaluation**: Support for various technical drawing datasets
- **Model comparison**: Compare DeepV against baseline and competitor models
- **Comprehensive metrics**: Vector and raster-based evaluation metrics
- **Automated reporting**: Generate detailed performance reports and visualizations
- **Extensible framework**: Easy to add new datasets, models, and metrics

### Key Components

- `benchmark_pipeline.py`: Main benchmarking runner
- `evaluation_suite.py`: Core evaluation framework
- `benchmark_config.json`: Configuration for datasets and models
- `util_files/metrics/`: Metric calculation functions

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements-dev.txt
pip install seaborn matplotlib

# Verify environment
python scripts/validate_env.py
```

### Basic Benchmark Run

```bash
# Benchmark DeepV on a single dataset
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --deepv-model-path /path/to/deepv/model \
  --output-dir results \
  --datasets floorplancad
```

### Comprehensive Evaluation

```bash
# Compare multiple models across datasets
python scripts/benchmark_pipeline.py \
  --data-root /datasets \
  --deepv-model-path /models/deepv \
  --output-dir benchmark_results \
  --datasets floorplancad archcad synthetic \
  --models deepv_current vectorgraphnet deepsvg
```

### Generate Synthetic Data

```bash
# Create synthetic evaluation dataset
python scripts/benchmark_pipeline.py \
  --generate-synthetic \
  --synthetic-samples 1000 \
  --output-dir synthetic_data
```

## Configuration

### Benchmark Config (`benchmark_config.json`)

The configuration file defines:

```json
{
  "datasets": {
    "floorplancad": {
      "type": "png_dxf",
      "path": "FloorPlanCAD",
      "description": "Floor plan CAD dataset"
    }
  },
  "models": {
    "deepv_current": {
      "type": "deepv",
      "path": "models/deepv",
      "description": "Current DeepV implementation"
    }
  },
  "metrics": ["f1", "iou", "hausdorff", "chamfer"],
  "evaluation": {
    "raster_resolution": 256,
    "timeout_seconds": 300,
    "parallel_evaluation": true
  }
}
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-root` | Root directory for datasets | Required |
| `--deepv-model-path` | Path to DeepV model | Required |
| `--output-dir` | Output directory | `benchmark_results` |
| `--datasets` | List of datasets to evaluate | `floorplancad` |
| `--models` | List of models to compare | `deepv_current` |
| `--generate-synthetic` | Generate synthetic dataset | `false` |
| `--synthetic-samples` | Number of synthetic samples | `1000` |

## Supported Datasets

### PNG + DXF Format
**Description**: Raster images with CAD ground truth
- **Input**: PNG raster images
- **Ground Truth**: DXF CAD files
- **Use Case**: Technical drawings, floor plans
- **Metrics**: F1, IoU, Hausdorff, Chamfer distance

### SVG Format
**Description**: Vector graphics datasets
- **Input**: SVG vector files
- **Ground Truth**: SVG paths and primitives
- **Use Case**: Web graphics, icons
- **Metrics**: Vector accuracy, path similarity

### Custom Formats
**Description**: Extensible format support
- **Input**: Any image format
- **Ground Truth**: SVG or custom vector format
- **Use Case**: Domain-specific datasets
- **Metrics**: Configurable metric suite

### Precision Floorplan Dataset
**Description**: Architectural floor plans
- **Input**: PDF documents
- **Ground Truth**: DXF floor plans
- **Use Case**: Architecture evaluation
- **Metrics**: Architectural accuracy metrics

### Dataset Requirements

Each dataset directory should contain:
```
dataset_name/
├── images/          # Input images (PNG, SVG, PDF)
├── ground_truth/    # Ground truth files (DXF, SVG)
├── metadata.json    # Dataset metadata (optional)
└── splits.json      # Train/val/test splits (optional)
```

## Supported Models

### DeepV Models
- **Current Implementation**: Latest DeepV model with all enhancements
- **Ablation Variants**: Different primitive types, architectures
- **Checkpoint Support**: Load from training checkpoints

### Baseline Models
- **VectorGraphNET**: State-of-the-art vectorization method
- **DeepSVG**: SVG generation model
- **Custom Baselines**: Any model with prediction interface

### Model Interface

Models must implement:
```python
class ModelInterface:
    def load(self, path: str) -> None:
        """Load model from path"""

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict primitives from image"""
```

## Metrics

### Vector Metrics

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **F1 Score** | Harmonic mean of precision/recall | 0-1 | 1.0 |
| **IoU** | Intersection over Union | 0-1 | 1.0 |
| **Hausdorff Distance** | Max distance between point sets | 0-∞ | 0.0 |
| **Chamfer Distance** | Average point set distance | 0-∞ | 0.0 |

### Raster Metrics

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | 0-∞ | ∞ |
| **MSE** | Mean Squared Error | 0-∞ | 0.0 |
| **SSIM** | Structural Similarity Index | 0-1 | 1.0 |

### Metric Calculation

```python
from util_files.metrics.vector_metrics import calculate_vector_metrics

predictions = model.predict(image)
ground_truth = load_ground_truth(dxf_file)

metrics = calculate_vector_metrics(predictions, ground_truth)
print(f"F1: {metrics['f1']:.3f}, IoU: {metrics['iou']:.3f}")
```

## Output Structure

```
benchmark_results/
├── benchmark_results.json          # Raw results data
├── benchmark_summary.md           # Comprehensive report
├── benchmark_summary.csv          # Tabular results
├── plots/
│   ├── f1_comparison.png          # Metric comparison plots
│   ├── dataset_performance.png    # Dataset-wise results
│   └── model_comparison.png       # Model comparison charts
├── datasets/
│   └── floorplancad/
│       └── deepv_current/
│           ├── results/           # Per-sample detailed results
│           ├── visualizations/    # Evaluation plots
│           ├── predictions/       # Model predictions
│           └── reports/          # Dataset-specific reports
└── config_used.json              # Configuration used for run
```

## Usage Examples

### Single Model Evaluation

```bash
# Evaluate DeepV on FloorPlanCAD
python scripts/benchmark_pipeline.py \
  --data-root /data \
  --deepv-model-path /models/deepv_line \
  --datasets floorplancad \
  --output-dir eval_line
```

### Multi-Model Comparison

```bash
# Compare line vs curve models
python scripts/benchmark_pipeline.py \
  --data-root /data \
  --models deepv_line deepv_curve vectorgraphnet \
  --datasets floorplancad archcad \
  --output-dir comparison_results
```

### Ablation Study

```bash
# Test different primitive counts
python scripts/benchmark_pipeline.py \
  --data-root /data \
  --models deepv_10_primitives deepv_20_primitives deepv_30_primitives \
  --datasets synthetic \
  --output-dir ablation_study
```

### Custom Dataset Evaluation

```bash
# Evaluate on custom dataset
python scripts/benchmark_pipeline.py \
  --data-root /custom_data \
  --deepv-model-path /models/trained \
  --datasets my_custom_dataset \
  --output-dir custom_eval
```

## Extending the Pipeline

### Adding New Datasets

1. **Create dataset directory** with required structure
2. **Add to benchmark_config.json**:
```json
{
  "datasets": {
    "my_dataset": {
      "type": "png_dxf",
      "path": "my_dataset",
      "description": "My custom dataset"
    }
  }
}
```
3. **Implement loader** in `evaluation_suite.py` if needed

### Adding New Models

1. **Add to benchmark_config.json**:
```json
{
  "models": {
    "my_model": {
      "type": "custom",
      "path": "/path/to/model",
      "description": "My custom model"
    }
  }
}
```
2. **Implement interface** in `ModelLoader` class

### Adding New Metrics

1. **Implement metric function**:
```python
def my_custom_metric(predictions, ground_truth):
    # Calculate custom metric
    return score
```

2. **Add to metrics registry** in `evaluation_suite.py`

## Integration

### With Training Pipeline

```bash
# Train model
python run_pipeline.py \
  --model_path /models/trained \
  --data_dir /training_data \
  --primitive_type line

# Evaluate trained model
python scripts/benchmark_pipeline.py \
  --deepv-model-path /models/trained \
  --data-root /eval_data
```

### With CI/CD

```yaml
# .github/workflows/benchmark.yml
- name: Run Benchmarks
  run: |
    python scripts/benchmark_pipeline.py \
      --data-root test_data \
      --output-dir benchmark_results
```

### With Experiment Tracking

```python
# Log results to experiment tracker
import wandb

wandb.init(project="deepv-benchmarking")
wandb.log(results)
```

## Troubleshooting

### Common Issues

**Dataset Not Found**:
```
Dataset 'floorplancad' not found in /data
```
- Verify dataset directory exists
- Check dataset name in config
- Ensure correct `--data-root` path

**Model Loading Errors**:
```
Failed to load model: checkpoint not found
```
- Verify model path exists
- Check model file format
- Ensure model type matches config

**Memory Issues**:
```
CUDA out of memory
```
- Reduce batch size in config
- Use CPU evaluation: `--gpu -1`
- Decrease raster resolution

**Timeout Errors**:
```
Evaluation timeout for sample_001
```
- Increase timeout in config
- Check for infinite loops in evaluation
- Optimize model prediction speed

### Debug Mode

```bash
# Run with debug output
python scripts/benchmark_pipeline.py \
  --debug \
  --datasets floorplancad \
  --models deepv_current
```

## Performance Optimization

### Parallel Processing

```bash
# Enable parallel evaluation
python scripts/benchmark_pipeline.py \
  --parallel-evaluation true \
  --max-workers 8
```

### Memory Optimization

- Use appropriate raster resolution
- Batch predictions when possible
- Clear GPU cache between evaluations

### Speed Optimization

- Pre-load models when evaluating multiple datasets
- Cache ground truth parsing
- Use faster metrics for quick iterations

## Contributing

### Adding New Features

1. **Follow existing patterns** in the codebase
2. **Add comprehensive tests** in `tests/test_benchmarking.py`
3. **Update documentation** with new capabilities
4. **Ensure backward compatibility**

### Code Standards

- Use type hints for all functions
- Add docstrings with examples
- Follow existing naming conventions
- Include error handling

### Testing

```bash
# Run benchmarking tests
pytest tests/test_benchmarking.py -v

# Test with synthetic data
python scripts/benchmark_pipeline.py \
  --generate-synthetic \
  --synthetic-samples 100 \
  --run-tests
```

### Documentation Updates

When adding new features:
- Update this README with usage examples
- Add configuration examples
- Document any new dependencies
- Include troubleshooting information