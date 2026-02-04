# Installation Guide for DeepV

## Prerequisites
- Python 3.10+
- Windows/Linux/macOS (WSL recommended on Windows)

## Quick Install

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/DeepV.git
   cd DeepV
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support (if you have CUDA):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. (Optional) Install dev dependencies for testing/linting:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Running Tests
```bash
# Run the full test suite
.\scripts\run_tests_local.ps1  # Windows PowerShell

# Or manually:
python -m pytest -q
```

## Running the Pipeline
See `DEVELOPER.md` for examples of running the vectorization pipeline.

## Troubleshooting
- If you encounter import errors, ensure all dependencies are installed.
- For GPU issues, check CUDA compatibility with PyTorch versions.
- Use the provided Docker setup for isolated environments.