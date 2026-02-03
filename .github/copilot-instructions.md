# Copilot instructions for DeepV (Deep Vectorization)

Purpose: give an AI coding agent the minimal, actionable context needed to be productive in this repository.

- **Big picture:** The pipeline has four main modules: cleaning, vectorization, refinement, and merging. Inputs flow: raw raster -> cleaning (denoise) -> vectorization (predict primitives per patch) -> refinement (differentiable optimization) -> merging (consolidate primitives). See [README.md](README.md) and [PLAN.md](PLAN.md) for longer descriptions.

- **Where to start (files):**
  - **Run full pipeline:** [run_pipeline.py](run_pipeline.py)
  - **Cleaning training / entrypoint:** [cleaning/scripts/main_cleaning.py](cleaning/scripts/main_cleaning.py)
  - **Module code:** `cleaning/`, `vectorization/`, `refinement/`, `merging/` folders
  - **Utilities & gotchas:** [util_files/os.py](util_files/os.py) (note: name shadows stdlib `os`)

- **How to run (examples)** — most scripts use `argparse` and have hardcoded defaults that assume the repository is mounted under `/code`, datasets under `/data`, and logs under `/logs` (Docker defaults in README). Use these or override flags.

  - Run the vectorization/refinement/merge pipeline (example):

  ```bash
  python run_pipeline.py \
    --model_path /logs/models/vectorization/lines/model_lines.weights \
    --json_path /code/.../vectorization/models/specs/<spec>.json \
    --data_dir /data/abc_png_svg/ \
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

- **Conventions & patterns (project-specific)**
  - Most scripts rely on `argparse` defaults with absolute paths (e.g., `/data`, `/logs`, `/code`) — prefer overriding flags rather than changing defaults.
  - Patching strategy: images are padded to multiples of 32 (see `cleaning/scripts/main_cleaning.py` and `run_pipeline.py`'s `read_data`/padding logic).
  - Vectorization works on square patches (often 64px) extracted using `util_files/patchify.py` (`patchify` call in `run_pipeline.py`).
  - Checkpoint & tensorboard locations: many defaults point to `/logs/...` and the code uses `tensorboardX` (SummaryWriter). Configure `--output_dir`/`--model_path` and TB dir explicitly for reproducibility.
  - Some utility filenames shadow stdlib modules (e.g., `util_files/os.py`) — import by package path (e.g., `from util_files import os as uos`) or rename locally when editing.

- **Developer workflows & environment**
  - Recommended: use the provided Dockerfile (see `docker/Dockerfile`) or run inside Linux/WSL. The README includes Docker build/run examples and notes about mounted paths.
  - On Windows, use WSL or Docker for compatibility (dataset download scripts and shell helpers assume Linux).
  - There is no automated test suite; run small experiments and validate with the notebooks in `notebooks/` when iterating.

- **Common pitfalls to watch for**
  - Hardcoded default paths may cause silent failures if files are not at expected locations.
  - Some code expects GPU/CPU behavior via `--gpu` flags and sets CUDA env vars manually (`CUDA_VISIBLE_DEVICES`) — be explicit when testing multi-GPU.
  - Long functions and mixed responsibilities exist in `refinement/` and `merging/` — prefer small, local refactors and add unit tests for helpers.

- **Search shortcuts for the agent**
  - Look for entrypoints with `if __name__ == '__main__'` to find scripts to run (e.g., `run_pipeline.py`, `cleaning/scripts/main_cleaning.py`).
  - Inspect `util_files/`, `notebooks/`, and `dataset/` for examples of expected input formats and dataset layout.

If anything here looks incomplete or you want more examples (e.g., a canonical `docker run` for Windows/WSL, or exact lines to change when renaming `util_files/os.py`), tell me which area to expand and I'll iterate.
