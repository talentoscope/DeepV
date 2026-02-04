# Configuration Management

DeepV uses [Hydra](https://hydra.cc/) for configuration management, providing hierarchical, reproducible configurations for experiments.

## Configuration Structure

```
config/
├── config.yaml          # Main configuration file
├── local.yaml           # Local development overrides
├── pipeline/
│   └── default.yaml     # Pipeline parameters
├── model/
│   └── default.yaml     # Model paths and specs
├── data/
│   └── default.yaml     # Data paths and settings
├── training/
│   └── default.yaml     # Training parameters
└── paths/
    ├── default.yaml     # Docker/container paths
    └── local.yaml       # Local development paths
```

## Usage

### Basic Usage

```python
from util_files.config_manager import get_config

# Load default configuration
cfg = get_config()

# Load with overrides
cfg = get_config(overrides=["pipeline.primitive_type=curve", "gpu=0"])
```

### Command Line Usage

```bash
# Run with default config
python run_pipeline_hydra.py

# Override settings
python run_pipeline_hydra.py pipeline.primitive_type=curve gpu=0

# Use local config
python run_pipeline_hydra.py --config-name local
```

### Configuration Files

- **`config.yaml`**: Main configuration with defaults for all components
- **`local.yaml`**: Local development configuration with relative paths
- **Component configs**: Modular configuration for different pipeline stages

## Key Features

- **Hierarchical**: Compose configurations from multiple files
- **Override**: Command-line overrides for experiments
- **Environment-specific**: Different configs for local vs Docker
- **Reproducible**: Save exact configuration used for each experiment
- **Type-safe**: Structured configuration with validation

## Migration from Argparse

The old `run_pipeline.py` uses hardcoded argparse defaults. The new `run_pipeline_hydra.py` demonstrates the Hydra approach:

- Configuration is externalized to YAML files
- Environment-specific settings are separated
- Experiments can be reproduced with exact config files
- Command-line overrides are more flexible

## Examples

### Local Development
```bash
python run_pipeline_hydra.py --config-name local pipeline.primitive_type=line
```

### Docker/Production
```bash
python run_pipeline_hydra.py  # Uses Docker paths by default
```

### Experiment with Overrides
```bash
python run_pipeline_hydra.py \
  pipeline.primitive_type=curve \
  pipeline.curve_count=20 \
  training.n_epochs=50 \
  seed=12345
```