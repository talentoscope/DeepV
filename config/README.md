# Configuration Management

DeepV uses [Hydra](https://hydra.cc/) for hierarchical configuration management, enabling reproducible experiments and flexible parameter overrides across different environments.

## Table of Contents

- [Overview](#overview)
- [Configuration Structure](#configuration-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration Files](#configuration-files)
- [Key Features](#key-features)
- [Migration Guide](#migration-guide)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

Hydra provides structured configuration management with:

- **Hierarchical configs**: Compose settings from multiple YAML files
- **Command-line overrides**: Modify any parameter without code changes
- **Environment-specific**: Different configs for local development vs production
- **Reproducible experiments**: Save exact configuration used for each run

## Configuration Structure

```
config/
├── config.yaml              # Main configuration
├── local.yaml               # Local development overrides
├── pipeline/
│   ├── default.yaml         # Pipeline parameters
│   └── overrides.yaml       # Pipeline-specific overrides
├── model/
│   ├── default.yaml         # Model architecture and paths
│   └── specs.yaml           # Model specifications
├── data/
│   └── default.yaml         # Dataset paths and settings
├── training/
│   ├── default.yaml         # Training hyperparameters
│   └── schedules.yaml       # Learning rate schedules
├── refinement/
│   └── default.yaml         # Refinement optimization settings
└── paths/
    ├── default.yaml         # Path configurations
    └── local.yaml           # Local development paths
```

## Quick Start

### Basic Usage

```python
from util_files.config_manager import get_config

# Load default configuration
cfg = get_config()

# Access nested parameters
print(f"Primitive type: {cfg.pipeline.primitive_type}")
print(f"GPU device: {cfg.gpu}")
```

### Command Line

```bash
# Run with defaults
python run_pipeline_hydra.py

# Override parameters
python run_pipeline_hydra.py pipeline.primitive_type=curve gpu=1

# Use local config
python run_pipeline_hydra.py --config-name local
```

## Usage

### Loading Configurations

```python
from util_files.config_manager import get_config

# Default config
cfg = get_config()

# With command-line style overrides
cfg = get_config(overrides=["pipeline.primitive_type=curve", "gpu=0"])

# Specific config file
cfg = get_config(config_name="local")
```

### Accessing Parameters

```python
# Nested access
model_path = cfg.model.path
data_dir = cfg.data.data_dir
learning_rate = cfg.training.lr

# Dictionary-style access
model_config = cfg["model"]
pipeline_config = cfg.pipeline
```

### Command Line Examples

```bash
# Basic pipeline run
python run_pipeline_hydra.py

# Override primitive type and model
python run_pipeline_hydra.py \
  pipeline.primitive_type=curve \
  model.path=/custom/model.pth

# Experiment with different settings
python run_pipeline_hydra.py \
  training.lr=0.001 \
  training.batch_size=16 \
  pipeline.curve_count=25 \
  seed=42

# Local development
python run_pipeline_hydra.py --config-name local

# Multirun (sweep parameters)
python run_pipeline_hydra.py --multirun \
  training.lr=0.001,0.0001 \
  pipeline.primitive_type=line,curve
```

## Configuration Files

### Main Configuration (`config.yaml`)
Primary configuration with production defaults:

```yaml
# Example structure
defaults:
  - pipeline: default
  - model: default
  - data: default
  - training: default
  - paths: default

gpu: 0
seed: 12345
output_dir: /logs
```

### Local Configuration (`local.yaml`)
Development overrides with local paths:

```yaml
# Local development settings
defaults:
  - paths: local
  - _self_

gpu: 0
data:
  data_dir: ./data
model:
  path: ./models
output_dir: ./output
```

### Component Configurations

**Pipeline Config** (`pipeline/default.yaml`):
```yaml
primitive_type: line
model_output_count: 10
overlap: 0
curve_count: 15
```

**Model Config** (`model/default.yaml`):
```yaml
path: /logs/models/vectorization/lines/model_lines.weights
json_path: /code/vectorization/models/specs/resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json
```

## Key Features

### Hierarchical Composition
- **Composition**: Build configs from multiple files using `defaults`
- **Overrides**: Command-line parameters override any setting
- **Inheritance**: Child configs inherit from parents

### Environment Handling
- **Path management**: Different paths for local vs production
- **Device configuration**: GPU/CPU settings per environment
- **Resource allocation**: Memory and compute settings

### Experiment Reproducibility
- **Config saving**: Automatic saving of exact config used
- **Version control**: Track configuration changes
- **Reproduction**: Exact recreation of experiments

## Migration Guide

### From Argparse to Hydra

**Old approach** (`run_pipeline.py`):
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='/logs/models/...')
args = parser.parse_args()
```

**New approach** (`run_pipeline_hydra.py`):
```python
from util_files.config_manager import get_config

cfg = get_config()
model_path = cfg.model.path
```

### Key Changes

1. **Configuration externalized**: Move hardcoded defaults to YAML files
2. **Environment separation**: Different configs for local vs production
3. **Override flexibility**: Change any parameter from command line
4. **Structured access**: Dot notation instead of attribute access

### Migration Steps

1. **Identify parameters**: List all argparse arguments
2. **Create YAML configs**: Move defaults to appropriate YAML files
3. **Update code**: Replace `args.param` with `cfg.section.param`
4. **Test thoroughly**: Verify all parameter combinations work

## Advanced Usage

### Multirun (Parameter Sweeps)

```bash
# Sweep learning rates and primitive types
python run_pipeline_hydra.py --multirun \
  training.lr=0.001,0.0001,0.00001 \
  pipeline.primitive_type=line,curve \
  training.batch_size=8,16
```

### Custom Config Groups

```python
# Create experiment-specific config
python run_pipeline_hydra.py \
  +experiment=baseline \
  training.lr=0.001 \
  pipeline.primitive_type=line
```

### Configuration Validation

```python
from util_files.config_manager import validate_config

cfg = get_config()
errors = validate_config(cfg)
if errors:
    print("Configuration errors:", errors)
```

## Troubleshooting

### Common Issues

**Configuration not found**:
```
Config file not found: config/local.yaml
```
- Check file exists in `config/` directory
- Verify file naming and path

**Override not working**:
```bash
python run_pipeline_hydra.py pipeline.primitive_type=curve
# But cfg.pipeline.primitive_type is still 'line'
```
- Check YAML syntax in config files
- Verify parameter path is correct
- Use `cfg.get()` for optional parameters

**Path resolution issues**:
- Ensure `paths/local.yaml` exists for local development
- Check path mounts for production runs
- Verify absolute paths in production configs

### Debugging

```python
# Print full configuration
from util_files.config_manager import get_config
cfg = get_config()
print(cfg.pretty())

# Check specific section
print(cfg.pipeline)
print(cfg.model)
```

### Environment-Specific Issues

**Local development**:
- Use `--config-name local`
- Check relative paths in `paths/local.yaml`
- Verify local GPU availability

**Production**:
- Default config uses production paths
- Check volume mounts
- Verify GPU passthrough

## Best Practices

### Configuration Organization

1. **Group related settings**: Keep related parameters together
2. **Use defaults**: Leverage Hydra's composition system
3. **Document parameters**: Add comments to YAML files
4. **Version configs**: Track config changes with experiments

### Development Workflow

1. **Local testing**: Use `local.yaml` for development
2. **Override for experiments**: Use command-line overrides
3. **Save configs**: Always save config with experiment results
4. **Version control**: Commit config changes

### Parameter Management

1. **Type hints**: Use appropriate types in YAML
2. **Validation**: Add runtime validation for critical parameters
3. **Documentation**: Document parameter meanings and ranges
4. **Defaults**: Choose sensible defaults for all parameters

### Examples

**Well-structured config**:
```yaml
# training/default.yaml
lr: 0.001          # Learning rate
batch_size: 16     # Batch size for training
n_epochs: 100      # Number of training epochs
optimizer: adam    # Optimizer type
```

**Command-line experimentation**:
```bash
# Quick experiments
python train.py training.lr=0.0001,0.001 training.optimizer=adam,sgd

# Ablation study
python train.py model.backbone=resnet18,resnet50 training.batch_size=8,16,32
```