# Dataset Management - DeepV

## Current Status: Clean Slate

All original dataset downloads have been removed due to broken links and corrupted files. The repository now starts with a clean dataset state.

## Available Options

### 1. Synthetic Dataset Generation (Recommended)

Generate synthetic datasets for testing and benchmarking:

```bash
# Generate all synthetic datasets
python scripts/create_test_datasets.py
```

This creates test datasets in `data/` directory with PNG and DXF files for benchmarking.

### 2. Real Dataset Acquisition

For production use with real datasets, see [DATA_SOURCES.md](DATA_SOURCES.md) for a comprehensive catalog of publicly available datasets suitable for deep vectorization tasks, including detailed information about formats, access links, licenses, and preprocessing tips.

## Repository Hygiene

- ✅ `.gitignore` excludes `data/` and `dataset/` directories
- ✅ No large binary files committed
- ✅ Clean, reproducible synthetic generation
- ✅ Proper data management practices

## Usage

```bash
# Generate synthetic datasets
python scripts/create_test_datasets.py

# Run benchmarking
python scripts/benchmark_pipeline.py --data-root data --datasets [your_datasets]

# Your data goes here
# data/custom_dataset/ (PNG + DXF files)
```

## Future Dataset Integration

When real datasets become available again:
1. Update download scripts with working links
2. Add proper validation and checksums
3. Document sources and licensing
4. Update this README with acquisition instructions

## Future Dataset Integration

When real datasets become available again:
1. Update download scripts with working links
2. Add proper validation and checksums
3. Document sources and licensing
4. Update this README with acquisition instructions