# Developer Guide

## Table of Contents

- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Pipeline Development](#pipeline-development)
- [Contributing](#contributing)

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
- Use provided `docker/Dockerfile` for reproducible environments

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

1. **Cleaning**: Noise removal and preprocessing
2. **Vectorization**: Neural network primitive prediction
3. **Refinement**: Differentiable optimization
4. **Merging**: Primitive consolidation and CAD export

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

---

*This guide is maintained by the DeepV development team. For the latest information, check the main README.md and CONTRIBUTING.md.*
