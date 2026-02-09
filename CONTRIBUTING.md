# Contributing to DeepV

Thank you for your interest in contributing to DeepV! This document provides comprehensive guidelines for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Areas for Contribution](#areas-for-contribution)

## Getting Started

### Prerequisites

- Python 3.8 or later
- Git
- Familiarity with PyTorch and differentiable rendering

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/DeepV.git
   cd DeepV
   ```

3. Set up the development environment:
   ```bash
   # On Windows (recommended)
   .\scripts\run_tests_local.ps1

   # Or manually
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. Run the tests to ensure everything works:
   ```bash
   pytest
   ```

## Development Workflow

### Branching Strategy

- `master`/`main`: Main development branch
- `draft/<topic>`: Short-lived feature branches for experimentation
- Create a descriptive branch from `main`/`master`:

```bash
git checkout -b draft/your-feature-name
```

### Making Changes

1. Create a feature branch
2. Make your changes following the [code quality guidelines](#code-quality)
3. Add tests for new functionality
4. Run the full test suite and quality checks
5. Commit with clear, descriptive messages

## Code Quality

### Formatting & Linting

This project uses automated tools to maintain code quality:

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type check (optional)
mypy --ignore-missing-imports --no-strict-optional .
```

### Pre-commit Hooks

Install pre-commit hooks to automatically run checks:

```bash
pip install pre-commit
pre-commit install
```

### Code Style Guidelines

- **Line length**: 127 characters (Black default)
- **Functions/Methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Files**: `snake_case.py`
- Use docstrings for all public functions/classes
- Keep comments concise and informative

## Testing

### Test Structure

- Unit tests in `tests/` directory
- Test files named `test_*.py`
- Use pytest framework

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run lightweight tests only (for quick validation)
pytest -k "not heavy"
```

### Writing Tests

- Use descriptive test names
- Test both success and failure cases
- Keep tests fast and isolated
- Aim for good test coverage

## Submitting Changes

### Pull Request Process

1. Ensure all tests pass and code quality checks pass
2. Update documentation if needed
3. Write a clear PR description
4. Request review from maintainers

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add new vectorization model
fix: resolve memory leak in refinement
docs: update installation guide
```

## Areas for Contribution

### 🔴 Highest Value (February 2026)

**CRITICAL: Real-world performance** - The #1 blocking issue
- Research and implement domain adaptation techniques
- Create data augmentation pipelines with realistic scanning artifacts
- Implement geometric regularization during refinement (parallelism, perpendicularity)
- Add architectural priors for floor plans (right angles, grid alignment, symmetry)
- Experiment with perceptual and geometric loss functions

**Why this matters**: Model performs 13x worse on real data vs synthetic. This blocks production deployment.

### 🟡 High Priority

**Documentation & Type Safety**
- Add docstrings to undocumented functions (especially refinement/merging - ~50% missing)
- Add type hints to function signatures (currently ~60%, target 80%+)
- Update module READMEs with recent changes

**Testing & Validation**
- Write integration tests for full pipeline
- Add regression tests comparing outputs to baselines
- Create performance benchmarks with time/memory targets

**Code Quality**
- Extract remaining magic numbers to Hydra configs
- Improve error handling and validation
- Refactor complex functions (>100 lines) into smaller units

### 🟢 Medium Priority

**New Features & Improvements**
- Multi-scale processing for complex drawings
- Enhanced CAD export with parametric constraints
- Additional primitive types and rendering methods
- Web UI improvements and deployment fixes

**Research Areas**
- Attention mechanisms for better spatial relationships
- Adaptive primitive selection based on local characteristics
- Robust handling of different degradation types
- Multi-term loss functions (reconstruction + geometric + perceptual)

### Getting Started with Contributing

**For First-Time Contributors**:
1. Start with documentation improvements (add docstrings, fix typos)
2. Add tests for existing functionality
3. Fix small bugs or TODOs marked "good first issue"

**For Experienced Contributors**:
1. Work on the real-world performance gap (domain adaptation, geometric priors)
2. Implement advanced features (multi-scale processing, new primitives)
3. Conduct research experiments and share findings

**For Researchers**:
1. Experiment with domain adaptation techniques on FloorPlanCAD dataset
2. Propose and implement improved loss functions
3. Benchmark against state-of-the-art vectorization methods
4. Write up findings in issues or discussions

---

## Quick Reference (Original)

Utilities and gotchas
- `util_files/file_utils.py` provides `require_empty()` — it was renamed from the old `util_files/os.py` to avoid shadowing the stdlib.
- Many scripts use absolute defaults for paths (`/code`, `/data`, `/logs`). Prefer overriding `argparse` flags rather than changing defaults in code.

Docker / Windows / WSL
- The provided `docker/Dockerfile` can be used to build a reproducible environment. Example run on Windows with WSL or Docker Desktop:

```powershell
docker build -t deepv:latest .
docker run --rm -it --shm-size 128G \
  --mount type=bind,source="C:/path/to/DeepV",target=/code \
  --mount type=bind,source="C:/path/to/data",target=/data \
  --mount type=bind,source="C:/path/to/logs",target=/logs \
  --name deepv-container deepv:latest /bin/bash
```

Pushing and PRs
- Push your branch and open a PR against `main` / `master`. Use a concise title and the PR template will automatically populate with the standard format.

Maintenance notes
- Upgrading `torch` requires selecting a wheel matching your CUDA runtime. Use the official PyTorch install selector (https://pytorch.org/get-started/locally/).

Where to get help
- For questions about specific modules see:
  - `cleaning/` — cleaning & UNet training
  - `vectorization/` — model specs and training
  - `refinement/` and `merging/` — optimization + postprocessing
  - `util_files/` — shared utilities
