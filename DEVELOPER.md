# Developer Guide

## Table of Contents

- [Quick Start](#quick-start)
- [Current Development Focus](#current-development-focus)
- [Development Environment](#development-environment)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Pipeline Development](#pipeline-development)
- [Contributing](#contributing)
- [Common Tasks & Troubleshooting](#common-tasks--troubleshooting)

## Current Development Focus

### Phase 4: Production-Ready & Robustness (~70% Complete)

**Top Priorities (February 2026):**

1. **🔴 Critical Issue**: FloorPlanCAD Performance Improvement
   - Current performance: IoU 0.010, high over-segmentation (~5120 primitives)
   - Primary focus: Architecture improvements (Non-Autoregressive Transformer), training on FloorPlanCAD data
   - Target: 20-40% primitive reduction, 50x IoU improvement

2. **Documentation & Type Safety**: 
   - ~70% of functions missing docstrings (especially refinement/merging)
   - Type hint coverage at ~20-30% in critical areas
   - Target: 80%+ documentation coverage

3. **Code Quality & Refactoring**:
   - Refinement module: 400+ line functions need breaking apart
   - Magic numbers scattered throughout (tolerances, thresholds)
   - Tensor broadcasting issues in batch processing

4. **Testing & Validation**:
   - Current: 14+ tests, pre-commit hooks active
   - Target: 70-80% unit test coverage, full integration tests
   - Need regression tests for pipeline outputs

**Recent Wins** 🎉:
- 90% speedup in greedy merging (53s → 5s)
- 387x faster Bézier splatting (2.5s → 0.0064s per patch)
- 70% overall pipeline improvement (77s → 23s)
- Large image support fixed (handles 1121×771px)
- Comprehensive metrics framework implemented

**Where to Contribute:**
- Architecture research (Non-Autoregressive Transformer for primitive reduction)
- Docstring additions (focus: refinement/merging)
- Integration tests for full pipeline
- Type hint improvements
- Performance profiling and optimization

## Quick Start

### Local Development Setup (Windows)

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install minimal dev dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

# Run environment validator and test suite
.\scripts\run_tests_local.ps1
```

### Key Notes

- The repository targets local development; heavy ML packages (PyTorch, torchvision) are optional for quick iteration
- Tests skip heavy ML components when packages aren't available
- For full experiments, install PyTorch matching your CUDA configuration from [pytorch.org](https://pytorch.org/get-started/locally/)

## Development Environment

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (venv/conda)
- (Optional) CUDA-compatible GPU

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-repo/DeepV.git
cd DeepV

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Validate setup
python scripts/validate_env.py
```

### IDE Setup

#### VS Code
- Install Python extension
- Select virtual environment as interpreter
- Install recommended extensions: Pylance, Python, Jupyter

#### PyCharm
- Open project folder
- Configure virtual environment in settings
- Enable pytest as test runner

## Code Quality

### Formatting & Linting

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type check
mypy --ignore-missing-imports --no-strict-optional .
```

### Pre-commit Hooks

```bash
# Install hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Code Style Guidelines

- **Line length**: 127 characters (Black default)
- **Imports**: Sort with `isort`, group by type
- **Docstrings**: Use Google-style docstrings
- **Types**: Add type hints where possible
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes

## Testing

### Test Structure

```
tests/
├── test_*.py              # Unit tests
├── integration/           # Integration tests
├── fixtures/              # Test data
└── conftest.py           # Pytest configuration
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_file_utils.py

# With coverage
pytest --cov=deepv --cov-report=html

# Lightweight tests only
pytest -k "not heavy"

# Windows helper
.\scripts\run_tests_local.ps1
```

### Writing Tests

```python
import pytest
from deepv.module import function_to_test

def test_function_basic():
    """Test basic functionality."""
    result = function_to_test(input_data)
    assert result == expected_output

def test_function_edge_cases():
    """Test edge cases and error conditions."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)
```

### Test Coverage Goals

- Unit tests: 80%+ coverage
- Integration tests: Full pipeline coverage
- Performance benchmarks: Key functions profiled

## Pipeline Development

### Understanding the Pipeline

DeepV processes images through four main stages:

1. **Cleaning**: Noise removal and preprocessing (UNet-based)
2. **Vectorization**: Neural network primitive prediction (ResNet + Transformer)
3. **Refinement**: Differentiable optimization with Bézier splatting rendering
4. **Merging**: Primitive consolidation and CAD export (DXF/SVG)

**Key Insight**: The pipeline needs improvement on FloorPlanCAD data. When developing:
- Test on FloorPlanCAD dataset to validate improvements
- Monitor IoU, Dice coefficient, SSIM, and CAD compliance metrics
- Use `scripts/comprehensive_analysis.py` for full quality assessment
- Focus on reducing over-segmentation (current: ~5120 primitives)

### Performance Characteristics

**Current Bottlenecks** (profiled February 2026):
- Rendering: 0.04-0.05s per iteration (Cairo-based; Bézier splatting optimized)
- Refinement: Needs further profiling; tensor broadcasting issues
- Merging: Spatial indexing could be improved (currently R-tree with 30px window)

**Optimization Opportunities**:
- Adaptive step sizes in refinement
- Better spatial indexing in merging
- Multi-scale processing for complex drawings
- Batch processing improvements

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes with tests
# Edit code...
# Add tests...

# 3. Run quality checks
black .
isort .
flake8 .
pytest

# 4. Test pipeline
python run_pipeline.py --test-mode

# 5. Commit changes
git add .
git commit -m "feat: add your feature"

# 6. Create PR
git push origin feature/your-feature
```

### Adding New Primitives

1. **Define primitive structure** in `util_files/optimization/primitives/`
2. **Update vectorization model** to predict new primitive types
3. **Add refinement logic** in `refinement/our_refinement/`
4. **Update merging** in `merging/`
5. **Add CAD export** in `cad/export.py`

### Performance Profiling

```bash
# Profile pipeline
python -m cProfile run_pipeline.py --profile

# Memory profiling
python -c "from util_files.profiling import profile_memory; profile_memory()"

# Benchmark specific functions
python scripts/benchmark_performance.py
```

## Contributing

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes with tests
4. **Run** all quality checks
5. **Submit** a pull request

### Commit Messages

```
feat: add new vectorization model
fix: resolve memory leak in refinement
docs: update installation guide
test: add integration tests for pipeline
refactor: simplify primitive merging logic
```

### Code Review Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No breaking changes
- [ ] Performance impact assessed

### Getting Help

- **Documentation**: Check module READMEs
- **Issues**: Search existing issues first
- **Discussions**: Use GitHub Discussions for questions
- **Code**: Read existing implementations for patterns

## Common Tasks & Troubleshooting

### Quick Reference Commands

```bash
# Environment validation
python scripts/validate_env.py

# Run comprehensive analysis on outputs
python scripts/comprehensive_analysis.py --results-dir logs/outputs/your_experiment

# Profile performance
python scripts/benchmark_performance.py
python scripts/profile_refinement_bottlenecks.py

# Benchmark against baselines
python scripts/benchmark_pipeline.py --data-root ./data --datasets floorplancad

# Extract ground truth from FloorPlanCAD SVGs
python scripts/extract_floorplancad_ground_truth.py

# Quick smoke test
python run_web_ui_demo.py
```

### Debugging Tips

**Tensor Shape Issues** (common in refinement):
```python
# Always validate tensor dimensions when modifying batch processing
print(f"Tensor shape: {tensor.shape}")
assert tensor.shape == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"
```

**Import Issues**:
```python
# Prefer explicit imports from util_files
from util_files import file_utils as fu
# NOT: from util_files.os import ...  (old pattern)
```

**Performance Issues**:
- Check if Cairo is using system library (not Python fallback)
- Profile with: `python -m cProfile -o profile.stats run_pipeline.py`
- Analyze with: `python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"`

**Poor Results on Real Data**:
- This is expected and under active research
- Try data augmentation with scanning artifacts during training
- Consider fine-tuning on small real dataset
- Check PLAN.md for improvement strategies

### Working with Datasets

**FloorPlanCAD** (14,625 real drawings - primary dataset):
- Location: `data/vector/floorplancad/` (SVG) and `data/raster/floorplancad/` (PNG)
- Status: Pre-processed and ready for training/evaluation
- Current performance: IoU 0.010, high over-segmentation (~5120 primitives)
- Focus: Architecture improvements and training optimization

**Creating Custom Splits**:
```bash
python scripts/create_floorplancad_splits.py --data-dir data/raster/floorplancad --output data/splits/
```

### Understanding Metrics

Run comprehensive analysis to get full quality report:
```bash
python scripts/comprehensive_analysis.py --results-dir logs/outputs/test_nes
```

**Key Metrics to Watch**:
- **IoU (Intersection over Union)**: Geometric overlap; target >0.5 (currently 0.010 on FloorPlanCAD)
- **Dice Coefficient**: Similar to IoU; target >0.9
- **SSIM (Structural Similarity)**: Visual quality; target >0.7 (currently low)
- **CAD Angle Compliance**: Percentage at standard angles (0\u00b0, 90\u00b0, etc.); target >80%
- **Chamfer Distance**: Point cloud similarity; lower is better

---

*This guide is maintained by the DeepV development team. For the latest information, check the main README.md and CONTRIBUTING.md.*
