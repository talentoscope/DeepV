# DeepV Benchmarking Pipeline

This directory contains the comprehensive benchmarking pipeline for evaluating DeepV vectorization models across multiple datasets and comparing against state-of-the-art baselines.

## Files

- `benchmark_pipeline.py` - Main benchmarking runner with dataset loaders, model interfaces, and comprehensive reporting
- `evaluation_suite.py` - Core evaluation framework with DatasetEvaluator and BenchmarkEvaluator classes
- `benchmark_config.json` - Configuration file defining datasets, models, and evaluation settings

## Quick Start

### Basic Usage

```bash
# Run benchmark on custom dataset with DeepV model
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --deepv-model-path /path/to/deepv/model \
  --output-dir benchmark_results \
  --datasets your_dataset

# Run comprehensive benchmark across multiple datasets
python scripts/benchmark_pipeline.py \
  --data-root /path/to/datasets \
  --deepv-model-path /path/to/deepv/model \
  --datasets dataset1 dataset2 dataset3 \
  --models deepv_current baseline_model1 baseline_model2
```

### Generate Synthetic Dataset

```bash
# Generate synthetic dataset for evaluation
python scripts/benchmark_pipeline.py \
  --generate-synthetic \
  --synthetic-samples 1000
```

## Configuration

The `benchmark_config.json` file defines:

- **Datasets**: Custom datasets (PNG + DXF pairs)
- **Models**: DeepV current implementation + baseline models
- **Metrics**: F1, IoU, Hausdorff distance, Chamfer distance, PSNR, MSE
- **Evaluation Settings**: Raster resolution, timeouts, parallel processing

## Output Structure

```
benchmark_results/
├── benchmark_results.json          # Raw results data
├── benchmark_summary.md           # Comprehensive report
├── dataset1/
│   └── model_name/
│       ├── results/               # Per-sample results
│       ├── visualizations/        # Evaluation plots
│       └── reports/              # Dataset-specific reports
└── dataset2/
    └── model_name/
        ├── results/
        ├── visualizations/
        └── reports/
```

## Supported Dataset Formats

### PNG + DXF Format
- **Input**: PNG raster images
- **Ground Truth**: DXF CAD files
- **Evaluation**: Vector reconstruction metrics (F1, IoU, Hausdorff, Chamfer)

### SVG Format
- **Input**: SVG vector graphics
- **Ground Truth**: Vector paths and primitives
- **Evaluation**: Full vector-based metrics

### Custom Formats
- Support for additional formats can be added to the evaluation suite
- **Ground Truth**: SVG vector graphics
- **Evaluation**: Vector accuracy metrics

### Precision Floorplan Dataset
- **Format**: PDF documents
- **Ground Truth**: DXF floorplans
- **Evaluation**: Architectural accuracy metrics

## Metrics

### Vector Metrics
- **F1 Score**: Precision and recall for vector primitives
- **IoU Score**: Intersection over Union for vector elements
- **Hausdorff Distance**: Maximum distance between point sets
- **Chamfer Distance**: Average distance between point sets

### Raster Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **MSE**: Mean Squared Error
- **SSIM**: Structural Similarity Index

## Extending the Pipeline

### Adding New Datasets

1. Add dataset configuration to `benchmark_config.json`
2. Implement loader method in `DatasetLoader` class
3. Add ground truth parsing logic

### Adding New Models

1. Add model configuration to `benchmark_config.json`
2. Implement model loading in `ModelLoader` class
3. Add prediction interface in evaluation framework

### Adding New Metrics

1. Implement metric function in `util_files/metrics/vector_metrics.py`
2. Add to `METRICS_BY_NAME` dictionary
3. Include in benchmark configuration

## Dependencies

Install additional dependencies:

```bash
pip install seaborn==0.12.2
```

## Integration with DeepV Pipeline

The benchmarking pipeline integrates with the main DeepV pipeline:

```bash
# Train model
python run_pipeline.py --model_path model_output --data_dir training_data

# Evaluate trained model
python scripts/benchmark_pipeline.py \
  --deepv-model-path model_output \
  --data-root evaluation_data
```

## Troubleshooting

### Common Issues

1. **Missing Datasets**: Ensure dataset directories exist at `--data-root`
2. **Model Loading Errors**: Verify model paths and checkpoint formats
3. **Memory Issues**: Reduce batch sizes or use CPU evaluation
4. **Import Errors**: Install missing dependencies from `requirements-dev.txt`

### Performance Optimization

- Use `--parallel-evaluation true` for multi-core processing
- Set appropriate timeouts to avoid hanging on complex samples
- Configure raster resolution based on available memory

## Contributing

When adding new evaluation capabilities:

1. Update `benchmark_config.json` with new configurations
2. Add comprehensive tests in `tests/test_benchmarking.py`
3. Update this documentation
4. Ensure backward compatibility with existing results