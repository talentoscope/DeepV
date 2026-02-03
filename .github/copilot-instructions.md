# Copilot / AI Agent Instructions — Deep Vectorization (concise)

This file contains short, actionable guidance to help an AI coding agent be productive in this repository.

**Big picture**
- The repo implements a pipeline to convert raster technical drawings into vector primitives. Major modules: `cleaning/`, `vectorization/`, `refinement/`, `merging/` (see [README.md](README.md)).
- Data flow: images -> patchify -> `vectorization` NN -> vector primitives -> `refinement` (opt) -> `merging` -> final vector output. See `run_pipeline.py` for the orchestration and example flow.

**Where to look first (entry points)**
- Project overview: [README.md](README.md).
- Pipeline runner and orchestration: `run_pipeline.py` — shows patching (`patchify`), model loading (`vectorization.load_model`), `serialize()` checkpoint tweaks, and calls into refinement/merging.
- Patch utilities and data representation notes: [util_files/README.md](util_files/README.md) and `util_files/patchify.py`.

**Project-specific conventions and gotchas**
- Module-like layout: each top-level folder (e.g., `cleaning`, `vectorization`) houses its own scripts, models, and README — inspect folder READMEs for module-specific instructions.
- Data representations: two main modes documented in `util_files/README.md`: `paths` (graphics-style path lists) and `vahe` (dictionary with `PT_LINE` and `PT_BEZIER` arrays). Use the correct mode when calling rendering/loss code.
- Patching rules: images are padded to multiples of 32 and then split into square patches (see `run_pipeline.split_to_patches` and `util_files/patchify.py`). Respect the `patch_size` and `overlap` parameters when modifying pipeline code.
- Checkpoint loading: some checkpoints require key renaming — `run_pipeline.serialize()` rewrites `hidden.transformer` -> `hidden.decoder.transformer` before `load_state_dict`. If you add models or change transformer layers, mirror this behavior or update `serialize()` accordingly.
- Output scaling: network outputs are multiplied by 64 in `vector_estimation` before downstream use — don't forget the scale when interpreting `patches_vector`.

**Developer workflows & commands**
- Environment: Linux (Docker recommended). See `requirements.txt` and Dockerfile at `docker/Dockerfile`.
- Build/run: preferred quick run is via `scripts/run_pipeline.sh` (or `run_pipeline.py` directly). Typical steps: install deps, download pretrained models, then run `python run_pipeline.py --model_path <path> --json_path <spec> --data_dir <dir>`.
- GPU handling: `run_pipeline.parse_args()` accepts `-g/--gpu` (appendable). When multiple GPUs are provided the script sets `CUDA_VISIBLE_DEVICES` and uses `cuda:0` for device selection.

**Where code commonly changes**
- Model specs and loader: `vectorization/*` and JSON model specs referenced by `run_pipeline.py` (`--json_path`).
- Refinement and merging: changes here alter final vector output format; see `refinement/our_refinement/*` and `merging/*`.
- Dataset scripts: `dataset/` contains download and preprocessing utilities; modify these when adding new datasets or patching logic.

**Examples / concrete references**
- Checkpoint key rename: see `run_pipeline.py` function `serialize(checkpoint)`.
- Patch splitting: see `run_pipeline.py::split_to_patches` and `util_files/patchify.py`.
- Rendering/data representation hint: `util_files/README.md` (search for `data_representation='paths'` and `data_representation='vahe'`).

**When you make a change**
- Run a small end-to-end local test using a single sample image (use `run_pipeline.py` with `--image_name`) to validate patching, vector output shape and scaling.
- If you modify checkpoint keys or model shapes, add a short note in the corresponding module README explaining compatibility expectations.

If anything above is unclear or you'd like me to expand specific sections (examples for refactoring, tests, or CI suggestions), tell me which area to elaborate on.
