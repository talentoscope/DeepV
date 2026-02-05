# DeepV Cleaning Module

The cleaning module handles preprocessing of technical drawings to remove noise, artifacts, and improve image quality before vectorization.

## Overview

This module uses UNet-based architectures for:
- **Noise removal**: Eliminating scanning artifacts and digital noise
- **Gap inpainting**: Filling missing regions in damaged drawings
- **Artifact reduction**: Removing stains, folds, and other imperfections
- **Quality enhancement**: Improving contrast and clarity

## Current Status

⚠️ **Under Development**: The cleaning module is partially implemented but requires testing and refinement. The UNet components exist but may have integration issues with the rest of the pipeline.

## Architecture

### UNet Model
- **Input**: Grayscale or RGB technical drawings
- **Output**: Cleaned images with reduced artifacts
- **Backbone**: Convolutional UNet with skip connections
- **Loss**: Combination of MSE and perceptual losses

### Training Data
- **Synthetic**: Generated corrupted versions of clean drawings
- **Real**: Paired noisy/clean technical drawing datasets
- **Augmentation**: Various noise types (Gaussian, salt&pepper, scratches)

## Usage

### Training

```bash
# Train cleaning model
python cleaning/scripts/main_cleaning.py \
  --model UNet \
  --datadir /path/to/training/data \
  --valdatadir /path/to/validation/data \
  --n_epochs 50 \
  --batch_size 8 \
  --name cleaning_experiment
```

### Parameters

- `--model`: Model architecture (currently only UNet supported)
- `--datadir`: Training data directory
- `--valdatadir`: Validation data directory
- `--n_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--name`: Experiment name for logging

### Data Format

Training data should be organized as:
```
data/
├── train/
│   ├── clean/
│   └── noisy/
└── val/
    ├── clean/
    └── noisy/
```

## Integration with Pipeline

The cleaning module integrates with the main DeepV pipeline:

```python
from cleaning.model import CleaningModel

# Load trained model
cleaner = CleaningModel.load('path/to/model.pth')

# Clean image
cleaned_image = cleaner.process(noisy_image)
```

## Known Issues

- Module rearrangement may have introduced import errors
- Requires testing with current codebase
- May need updates for compatibility with PyTorch versions

## Future Improvements

- [ ] Test and fix integration issues
- [ ] Add more advanced architectures (SegFormer, etc.)
- [ ] Implement diffusion-based inpainting
- [ ] Add multi-class cleaning (symbols, text, annotations)
- [ ] Support for domain-specific augmentations

## Contributing

When working on the cleaning module:

1. Test changes with the training script
2. Verify compatibility with main pipeline
3. Update this README with any new features
4. Add appropriate tests

## References

- UNet: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Image Inpainting: [Pathak et al., 2016](https://arxiv.org/abs/1607.07539)