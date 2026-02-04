DEVELOPER GUIDE
===============

Quick local workflow for development and testing (Windows)

- Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- Install minimal dev dependencies (fast local test loop):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

- Run the environment validator and full test suite (helper provided):

```powershell
.\scripts\run_tests_local.ps1
```

- Notes:
  - The repository is intended for local development; heavy ML packages (PyTorch, torchvision) are optional for quick iteration. Tests that require them are skipped when not present.
  - For full experiments you can install PyTorch using the official instructions for your CUDA/CPU configuration at https://pytorch.org/get-started/locally/.
  - If you prefer isolation, use the provided `docker/Dockerfile` as a reproducible environment.
