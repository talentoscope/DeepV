# Copilot instructions for DeepV (Deep Vectorization)

Short, actionable guidance to get an AI coding agent productive quickly in this repo.

## Big picture
- Pipeline modules: **cleaning → vectorization → refinement → merging**. Flow: raster image → cleaning (denoise/pad/patchify) → vectorization (predict primitives per patch) → refinement (differentiable optimization) → merging (consolidate primitives, export DXF/SVG).
- Implemented in PyTorch; datasets & preprocessing live in `dataset/`; demos and experiments in `notebooks/`.

## Where to start (quick links)
- Full pipeline runner: [run_pipeline.py](run_pipeline.py)
- Hydra variant: [run_pipeline_hydra.py](run_pipeline_hydra.py)
- Cleaning entrypoint: [cleaning/scripts/main_cleaning.py](cleaning/scripts/main_cleaning.py)
- Web demo: [run_web_ui_demo.py](run_web_ui_demo.py) and [web_ui/](web_ui/)
- Notebooks: [notebooks/Rendering_example.ipynb](notebooks/Rendering_example.ipynb)
- Utilities: [util_files/file_utils.py](util_files/file_utils.py) and [util_files/patchify.py](util_files/patchify.py)

## How to run (concrete examples)
- Run pipeline (example):

```bash
python run_pipeline.py \
  --model_path /logs/models/vectorization/lines/model_lines.weights \
  --json_path /code/vectorization/models/specs/<spec>.json \
  --data_dir /data/synthetic/ \
  --primitive_type line \
  --model_output_count 10 \
  --overlap 0
```

- Train cleaning UNet (example):

```bash
python cleaning/scripts/main_cleaning.py \
  --model UNET --datadir /data/synth/ --valdatadir /data/val/ \
  --n_epochs 10 --batch_size 8 --name exp1
```

- Local Windows dev (powershell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
.\scripts\run_tests_local.ps1
```

## Project-specific conventions & gotchas
- Many scripts use `argparse` with defaults assuming repo mounted at `/code`, datasets at `/data`, logs at `/logs`. Prefer overriding flags rather than editing defaults.
- Patch & padding: patches are square (commonly 64px) and images are padded to multiples of 32 (see `cleaning/scripts/main_cleaning.py` and `run_pipeline.py` read/padding logic).
- `util_files/os.py` previously shadowed stdlib `os`. Use [util_files/file_utils.py](util_files/file_utils.py) imports: `from util_files import file_utils as fu`.
- Checkpoints & logging: defaults use `/logs/...` and `tensorboardX` (SummaryWriter). Always set `--output_dir` and TB dir for reproducibility.

## Integration points & dependencies
- Native/C bindings: `cairo`, `pycairo` used for rendering. Ensure system `cairo` is installed for the Python bindings to work.
- Custom ops: `chamferdist` may require compilation or binary wheel matching your PyTorch version.
- Deep learning: PyTorch + torchvision; specific versions may be pinned in `requirements*.txt`.

## Tests, validation & debugging
- Validate environment: `python scripts/validate_env.py`.
- Run local tests (Windows helper): `.\scripts\run_tests_local.ps1` or `pytest -q` after installing `requirements-dev.txt`.
- Not all tests require GPU; tests skip heavy ML parts when CUDA/PyTorch missing.

## Where code is fragile / good refactor targets
- Long, mixed-responsibility functions exist in `refinement/` and `merging/` — prefer small, focused refactors.
- Hardcoded paths and magic numbers across scripts (batch sizes, primitive counts) — make config-driven when changing behavior.

## Search shortcuts for an agent
- Find entrypoints: search for `if __name__ == '__main__'` (e.g., `run_pipeline.py`, `cleaning/scripts/main_cleaning.py`).
- Inspect `util_files/`, `notebooks/`, and `dataset/` for I/O formats and examples.
- Model specs are JSON under `vectorization/models/specs/` — useful when modifying model I/O.

## Quick checklist for common tasks
- Run a quick smoke inference: run `run_web_ui_demo.py` (small demo) to sanity-check model imports and rendering path.
- Debug imports: prefer `from util_files import file_utils as fu` to avoid name clashes.
- Reproduce an experiment: always supply `--data_dir`, `--model_path`, and `--output_dir` to pipeline scripts.

If you want, I can expand this with a short example for Docker on Windows/WSL, add exact lines to change when renaming utilities, or include common pytest commands. Which area should I expand? 
