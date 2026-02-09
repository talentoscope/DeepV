# Copilot instructions for DeepV (Deep Vectorization)

Short, actionable guidance to get an AI coding agent productive quickly in this repo.

## Big picture
- **Core mission**: Convert degraded raster technical drawings (scanned books, patents, faded photocopies, architectural plans) into clean, editable CAD-quality vector representations through intelligent reconstruction—not just pixel tracing, but idealizing geometry by enforcing constraints, removing noise, and inferring design intent.
- **Pipeline modules**: **cleaning → vectorization → refinement → merging**. Flow: raster image → cleaning (denoise/pad/patchify) → vectorization (predict primitives per patch) → refinement (differentiable optimization) → merging (consolidate primitives, export DXF/SVG).
- **Current phase**: Phase 4 (Production-Ready & Robustness) ~70% complete. Phase 3 enhancements (CAD export, Gradio UI, Bézier splatting) fully implemented. Focus now: documentation, refactoring, testing, and **improving FloorPlanCAD performance through architecture changes**.
- Implemented in PyTorch; datasets & preprocessing live in `dataset/`; demos and experiments in `notebooks/`.
- Supports extended primitives: lines, quadratic/cubic Béziers, arcs, splines with variable counts (up to 20 per patch).
- Configuration managed via Hydra (see `config/`); supports both argparse (legacy) and Hydra configs.
- Unified pipeline interface in `pipeline_unified.py` consolidates line/curve processing.

## Critical context (READ FIRST)
**MAJOR PERFORMANCE ISSUE**: FloorPlanCAD dataset shows poor performance (IoU: 0.010, Overall: 0.043/1.000) with high over-segmentation (~5120 primitives per image). This is the #1 priority issue. Focus on architecture improvements and training optimization. See [PLAN.md](PLAN.md) for baseline metrics and [TODO.md](TODO.md) for improvement strategies.

**Recent wins** (celebrate these!):
- 90% speedup in greedy merging algorithm (53s → 5s)
- 387x faster Bézier splatting (2.5s → 0.0064s per patch)
- 70% overall pipeline improvement (77s → 23s)
- Large image support fixed (handles 1121x771px NES images)
- Comprehensive metrics framework implemented ✅

**Current bottlenecks**: Refinement still needs profiling; real data domain gap; magic numbers scattered in refinement/merging; incomplete docstrings (~70% missing in refinement/merging).

## Where to start (quick links)
- Full pipeline runner: [run_pipeline.py](run_pipeline.py) (argparse) or [run_pipeline_hydra.py](run_pipeline_hydra.py) (Hydra config)
- Unified pipeline: [pipeline_unified.py](pipeline_unified.py)
- Cleaning entrypoint: [cleaning/scripts/main_cleaning.py](cleaning/scripts/main_cleaning.py)
- Web demo: [run_web_ui_demo.py](run_web_ui_demo.py) and [web_ui/](web_ui/) (⚠️ has deployment issues with Gradio/conda)
- Notebooks: [notebooks/Rendering_example.ipynb](notebooks/Rendering_example.ipynb)
- Utilities: [util_files/file_utils.py](util_files/file_utils.py) and [util_files/patchify.py](util_files/patchify.py)
- Analysis scripts: [scripts/comprehensive_analysis.py](scripts/comprehensive_analysis.py) (metrics), [scripts/benchmark_pipeline.py](scripts/benchmark_pipeline.py) (evaluation)

Note: Datasets have been prepared and placed under `data/` (raster/vector). Use the verification scripts below to confirm integrity before running experiments.

## How to run (concrete examples)
- Run pipeline (argparse example):
 - Run pipeline (argparse example):

```bash
python run_pipeline.py \
  --model_path ./logs/models/vectorization/lines/model_lines.weights \
  --json_path ./vectorization/models/specs/resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json \
  --data_dir ./data/raster/floorplancad/ \
  --primitive_type line \
  --model_output_count 10 \
  --overlap 0
```

- Run pipeline (Hydra example):

```bash
python run_pipeline_hydra.py \
  pipeline.primitive_type=line \
  model.path=./logs/models/vectorization/lines/model_lines.weights \
  data.data_dir=./data/raster/floorplancad/
```

- Train cleaning UNet:
 - Train cleaning UNet:

```bash
python cleaning/scripts/main_cleaning.py \
  --model UNET --datadir ./data/synth/ --valdatadir ./data/val/ \
  --n_epochs 10 --batch_size 8 --name exp1
```

- Local Windows dev (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
.\scripts\run_tests_local.ps1

Verification & dataset notes:
- Validate environment: run `python scripts/validate_env.py` to check system dependencies.
- Verify FloorPlanCAD dataset and splits: run `python scripts/download_and_verify_floorplancad.py` or `python create_floorplancad_splits.py` if you produced splits locally.
- Quick smoke inference (small demo): run `python run_web_ui_demo.py` to sanity-check model imports and rendering.
```

## Project-specific conventions & gotchas
- Many scripts use `argparse` with defaults assuming local paths. Prefer overriding flags rather than editing defaults.
- Patch & padding: patches are square (commonly 64px) and images are padded to multiples of 32 (see `cleaning/scripts/main_cleaning.py` and `run_pipeline.py` read/padding logic).
- `util_files/os.py` previously shadowed stdlib `os`. Use [util_files/file_utils.py](util_files/file_utils.py) imports: `from util_files import file_utils as fu`.
- When renaming utilities, update imports from `from util_files.os import ...` to `from util_files.file_utils import ...`.

If datasets are already in place:
- Confirm `data/raster/` and `data/vector/` contain the expected FloorPlanCAD files listed in `floorplancad_files.txt`.
- Use `scripts/extract_floorplancad_ground_truth.py` to regenerate or validate SVG ground truth parsing.
- Checkpoints & logging: defaults use `logs/` and `tensorboardx` (SummaryWriter). Always set `--output_dir` and TB dir for reproducibility.
- Model specs: JSON configs under `vectorization/models/specs/` define network architectures (e.g., ResNet + Transformer decoder).
- Primitive types: Use "line" or "curve"; curves include Béziers, arcs, splines.
- Rendering: Uses `cairo`/`pycairo` for vector-to-raster; ensure system cairo installed.
- **Data quality**: FloorPlanCAD dataset works but shows poor performance (IoU: 0.010). Over-segmentation and geometric accuracy are the key improvement areas.
- **Magic numbers**: Refinement tolerances, merging thresholds, and patch overlap are hardcoded in many places—check carefully before modifying.
- **Tensor shapes**: Refinement has broadcasting issues—always validate tensor dimensions when modifying batch processing.
- **Performance**: Rendering is the bottleneck (0.04-0.05s per iteration); Bézier splatting is optimized but Cairo-based operations are slow.

## Integration points & dependencies
- Native/C bindings: `cairo`, `pycairo` used for rendering. Ensure system `cairo` is installed for the Python bindings to work.
- Custom ops: `chamferdist` may require compilation or binary wheel matching your PyTorch version.
- Deep learning: PyTorch + torchvision; specific versions pinned in `requirements*.txt`. Use `requirements-updated.txt` for latest compatible versions.
- Config system: Hydra configs in `config/` with composition (e.g., `config.yaml` composes pipeline, model, data configs).
- CAD export: DXF/SVG output via `ezdxf`, `svgwrite`; parametric conversion supported.

## Datasets & data processing
- **FloorPlanCAD**: 14,625 real architectural drawings (SVG vectors + PNG rasters) in `data/vector/floorplancad/` and `data/raster/floorplancad/`. Primary benchmark dataset. **WARNING**: Model performs very poorly on this (IoU: 0.010).
- **Data pipeline**: `dataset/processors/` contains converters for various formats (CubiCasa5K, SketchGraphs, ResPlan, etc.).
- **Ground truth**: Use `scripts/extract_floorplancad_ground_truth.py` to parse SVG ground truth for comparison.
- **Processing**: Images are padded to multiples of 32, split into 64×64 patches with configurable overlap (default: 0).
- **Splits**: Train/val/test splits defined in `data/splits/` as JSON files with image paths.

If you've placed datasets into `data/` already, run `python scripts/comprehensive_analysis.py` (or `python scripts/benchmark_pipeline.py`) on a small subset to confirm the tooling and metric pipelines are working before a full-scale run.

## Tests, validation & debugging
- Validate environment: `python scripts/validate_env.py`.
- Run local tests (Windows helper): `.\scripts\run_tests_local.ps1` or `pytest -q` after installing `requirements-dev.txt`.

Small checklist after dataset placement:
- Run `python scripts/validate_env.py`.
- Run `python scripts/download_and_verify_floorplancad.py` (if needed) or `python scripts/create_floorplancad_splits.py` to generate splits.
- Execute `python run_web_ui_demo.py` for a smoke run.
- Common pytest commands: `pytest tests/test_smoke.py -v` (smoke tests), `pytest tests/ -k "integration"` (integration tests), `pytest --tb=short` (shorter tracebacks).
- Not all tests require GPU; tests skip heavy ML parts when CUDA/PyTorch missing.
- Benchmarking: `scripts/benchmark_pipeline.py` for evaluation against baselines.
- **Comprehensive metrics**: Run `python scripts/comprehensive_analysis.py` to get full quality report (geometric, visual, CAD compliance).
- **Profiling**: Use `scripts/profile_refinement_bottlenecks.py` and `scripts/benchmark_performance.py` for performance analysis.

## Where code is fragile / good refactor targets
- **Refinement module** (`refinement/our_refinement/`): Long functions (400+ lines), tensor broadcasting issues, hardcoded tolerances. Recently refactored but still needs work.
- **Merging module** (`merging/`): Separate line/curve paths with duplicated logic; magic threshold values; spatial indexing can be improved.
- **Hardcoded paths and magic numbers**: Batch sizes, primitive counts, merging tolerances, refinement step sizes scattered throughout—make config-driven when changing behavior.
- **FloorPlanCAD performance**: Model needs training on FloorPlanCAD data; over-segmentation and geometric accuracy are the key improvement areas.
- **Error handling**: Many functions lack proper exception handling; tensor shape mismatches can cause obscure errors.
- **Type hints**: Only ~20-30% coverage in critical areas; gradual typing improvements ongoing.
- **Documentation**: ~70% of functions missing docstrings, especially in refinement/merging modules.

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

## Improving real-world performance (key research directions)
The FloorPlanCAD performance gap is the main open problem. Consider these approaches:
- **Architecture improvements**: Switch to Non-Autoregressive Transformer to reduce over-segmentation; train directly on FloorPlanCAD data.
- **Geometric constraints**: Enforce parallelism, perpendicularity, equal spacing during refinement; add architectural priors (walls meet at right angles, rooms are rectangular).
- **Multi-scale processing**: Hierarchical vectorization for complex drawings; coarse-to-fine refinement; context-aware primitive detection.
- **Better loss functions**: Perceptual losses (LPIPS); geometric losses (parallelism, symmetry); CAD-specific losses (angle snapping, grid alignment).
- **Adaptive refinement**: Per-primitive step sizes; region-specific tolerances; confidence-based optimization.
- **Post-processing regularization**: Snap to grid, merge nearly-parallel lines, detect and group repeated patterns.

See [PLAN.md](PLAN.md) for detailed improvement strategies and [TODO.md](TODO.md) for concrete implementation tasks. 
