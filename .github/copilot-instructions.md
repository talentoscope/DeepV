# Copilot instructions for DeepV (Deep Vectorization)

Short, actionable guidance to get an AI coding agent productive quickly in this repo.

## Big picture
- Pipeline modules: **cleaning → vectorization → refinement → merging**. Flow: raster image → cleaning (denoise/pad/patchify) → vectorization (predict primitives per patch) → refinement (differentiable optimization) → merging (consolidate primitives, export DXF/SVG).
- Implemented in PyTorch; datasets & preprocessing live in `dataset/`; demos and experiments in `notebooks/`.
- Supports extended primitives: lines, quadratic/cubic Béziers, arcs, splines with variable counts (up to 20 per patch).
- Configuration managed via Hydra (see `config/`); supports both argparse (legacy) and Hydra configs.
- Unified pipeline interface in `pipeline_unified.py` consolidates line/curve processing.

## Where to start (quick links)
- Full pipeline runner: [run_pipeline.py](run_pipeline.py) (argparse) or [run_pipeline_hydra.py](run_pipeline_hydra.py) (Hydra config)
- Unified pipeline: [pipeline_unified.py](pipeline_unified.py)
- Cleaning entrypoint: [cleaning/scripts/main_cleaning.py](cleaning/scripts/main_cleaning.py)
- Web demo: [run_web_ui_demo.py](run_web_ui_demo.py) and [web_ui/](web_ui/)
- Notebooks: [notebooks/Rendering_example.ipynb](notebooks/Rendering_example.ipynb)
- Utilities: [util_files/file_utils.py](util_files/file_utils.py) and [util_files/patchify.py](util_files/patchify.py)

## How to run (concrete examples)
- Run pipeline (argparse example):

```bash
python run_pipeline.py \
  --model_path /logs/models/vectorization/lines/model_lines.weights \
  --json_path /code/vectorization/models/specs/resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json \
  --data_dir /data/synthetic/ \
  --primitive_type line \
  --model_output_count 10 \
  --overlap 0
```

- Run pipeline (Hydra example):

```bash
python run_pipeline_hydra.py \
  pipeline.primitive_type=line \
  model.path=/logs/models/vectorization/lines/model_lines.weights \
  data.data_dir=/data/synthetic/
```

- Train cleaning UNet:

```bash
python cleaning/scripts/main_cleaning.py \
  --model UNET --datadir /data/synth/ --valdatadir /data/val/ \
  --n_epochs 10 --batch_size 8 --name exp1
```

- Local Windows dev (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
.\scripts\run_tests_local.ps1
```

- Docker on Windows/WSL (example):

```powershell
# Build image
docker build -t deepv:latest .

# Run container with Windows paths
docker run --rm -it --shm-size 128G -p 4045:4045 `
  --mount type=bind,source="C:/path/to/DeepV",target=/code `
  --mount type=bind,source="C:/path/to/data",target=/data `
  --mount type=bind,source="C:/path/to/logs",target=/logs `
  --name deepv-container deepv:latest /bin/bash
```

Inside container:
```bash
. /opt/.venv/vect-env/bin/activate/
```

## Project-specific conventions & gotchas
- Many scripts use `argparse` with defaults assuming repo mounted at `/code`, datasets at `/data`, logs at `/logs`. Prefer overriding flags rather than editing defaults.
- Patch & padding: patches are square (commonly 64px) and images are padded to multiples of 32 (see `cleaning/scripts/main_cleaning.py` and `run_pipeline.py` read/padding logic).
- `util_files/os.py` previously shadowed stdlib `os`. Use [util_files/file_utils.py](util_files/file_utils.py) imports: `from util_files import file_utils as fu`.
- When renaming utilities, update imports from `from util_files.os import ...` to `from util_files.file_utils import ...`.
- Checkpoints & logging: defaults use `/logs/...` and `tensorboardx` (SummaryWriter). Always set `--output_dir` and TB dir for reproducibility.
- Model specs: JSON configs under `vectorization/models/specs/` define network architectures (e.g., ResNet + Transformer decoder).
- Primitive types: Use "line" or "curve"; curves include Béziers, arcs, splines.
- Rendering: Uses `cairo`/`pycairo` for vector-to-raster; ensure system cairo installed.

## Integration points & dependencies
- Native/C bindings: `cairo`, `pycairo` used for rendering. Ensure system `cairo` is installed for the Python bindings to work.
- Custom ops: `chamferdist` may require compilation or binary wheel matching your PyTorch version.
- Deep learning: PyTorch + torchvision; specific versions pinned in `requirements*.txt`. Use `requirements-updated.txt` for latest compatible versions.
- Config system: Hydra configs in `config/` with composition (e.g., `config.yaml` composes pipeline, model, data configs).
- CAD export: DXF/SVG output via `ezdxf`, `svgwrite`; parametric conversion supported.

## Tests, validation & debugging
- Validate environment: `python scripts/validate_env.py`.
- Run local tests (Windows helper): `.\scripts\run_tests_local.ps1` or `pytest -q` after installing `requirements-dev.txt`.
- Common pytest commands: `pytest tests/test_smoke.py -v` (smoke tests), `pytest tests/ -k "integration"` (integration tests), `pytest --tb=short` (shorter tracebacks).
- Not all tests require GPU; tests skip heavy ML parts when CUDA/PyTorch missing.
- Benchmarking: `scripts/benchmark_pipeline.py` for evaluation against baselines.

## Where code is fragile / good refactor targets
- Long, mixed-responsibility functions exist in `refinement/` and `merging/` — prefer small, focused refactors.
- Hardcoded paths and magic numbers across scripts (batch sizes, primitive counts) — make config-driven when changing behavior.
- Separate line/curve pipelines partially unified; use `UnifiedPipeline` class for consistency.

## Search shortcuts for an agent
- Find entrypoints: search for `if __name__ == '__main__'` (e.g., `run_pipeline.py`, `cleaning/scripts/main_cleaning.py`).
- Inspect `util_files/`, `notebooks/`, and `dataset/` for I/O formats and examples.
- Model specs are JSON under `vectorization/models/specs/` — useful when modifying model I/O.
- Rendering examples: See `notebooks/Rendering_example.ipynb` for cairo usage.

## Quick checklist for common tasks
- Run a quick smoke inference: run `run_web_ui_demo.py` (small demo) to sanity-check model imports and rendering path.
- Debug imports: prefer `from util_files import file_utils as fu` to avoid name clashes.
- Reproduce an experiment: always supply `--data_dir`, `--model_path`, and `--output_dir` to pipeline scripts.
- For new features: check Hydra configs first, then argparse fallbacks. Use unified pipeline for line/curve agnostic code. 
