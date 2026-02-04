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

### High Priority
- Bug fixes and performance improvements
- Additional test coverage
- Documentation improvements

### Medium Priority
- New primitive types (arcs, splines)
- CAD export functionality
- Web UI improvements

### Research Areas
- Transformer/diffusion models for vectorization
- Multimodal vectorization approaches
- Advanced differentiable rendering techniques

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
- Push your branch and open a PR against `main` / `master`. Use a concise title and paste the contents of `PR_DESCRIPTION.md` as the PR body when applicable.

Maintenance notes
- Upgrading `torch` requires selecting a wheel matching your CUDA runtime. Use the official PyTorch install selector (https://pytorch.org/get-started/locally/).

Where to get help
- For questions about specific modules see:
  - `cleaning/` — cleaning & UNet training
  - `vectorization/` — model specs and training
  - `refinement/` and `merging/` — optimization + postprocessing
  - `util_files/` — shared utilities
