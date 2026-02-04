# DeepV TODO Checklist

This file is a comprehensive, actionable checklist derived from `PLAN.md`. Use it as a living checklist for local development. Items marked [x] are complete; [ ] are pending.

## Phase 1 — Foundation (high priority)
- [x] Update core dependencies and create a pinned `requirements.txt` / `constraints.txt` for releases. (See `requirements-dev.txt` for dev pins.)
- [x] Add `requirements-dev.txt` with pinned dev packages for fast local testing.
- [x] Add Windows test runner `scripts/run_tests_local.ps1` to create a venv, install dev deps, run validator, and run pytest.
- [x] Add `DEVELOPER.md` with quick venv/run instructions.
- [x] Rename conflicting util: `util_files/os.py` → `util_files/file_utils.py` and update imports.
- [x] Add a small suite of unit tests for `merging` helpers and `util_files` utilities.
- [x] Add a lightweight refinement smoke test that avoids importing heavy ML libs.
- [x] Run the full local pytest suite and iterate on dependency fixes until green.

## Phase 1 — Additional action items
- [x] Refactor `cleaning/scripts/main_cleaning.py` to use consistent argparse/config pattern; add unit tests for data loaders and synthetic data generator.
- [x] Add unit tests for vectorization components (model loading, small inference).
- [x] Add targeted unit tests for refinement helper functions (canonicalization, constraints).
- [x] Expand merging tests with edge cases and performance benchmarks.
- [x] Add linting, formatting, and pre-commit hooks (`black`, `flake8`, optional `mypy`).
- [x] Add Sphinx docs and an API reference; migrate useful notebook content into docs/examples.
- [x] Add configuration management (Hydra or centralized config object) for reproducible experiments.

## Phase 2 — Optimization
- [x] Profile refinement and rendering hotspots (use `torch.profiler` and small benchmarks).
- [x] Implement faster differentiable rendering (e.g., Bézier Splatting) prototypes to accelerate refinement.
- [x] Optimize merging algorithms with spatial indices (R-tree) to reduce O(n^2) behavior.

## Phase 3 — Enhancements
- [ ] Add support for arcs, splines, and variable primitive counts per patch.
- [ ] Upgrade vectorization with transformer/diffusion models for generative and multimodal vectorization.
- [x] Add CAD export and sequence-to-CAD conversion for parametric downstream usage.
- [x] Build a simple web UI (Gradio) to visualize and test results interactively.

## Phase 4 — Maintenance & Community
- [x] Add automated dependency security scans (`safety`) and periodic update cadence.
- [x] Add an issues/labels template and contribution guide for external contributors.
- [x] Optionally add CI with a separation of lightweight vs heavy tests (local-first policy recommended).

## Quick wins (done)
- [x] Tests: small unit tests for merging and util_files added.
- [x] Dev helper: `requirements-dev.txt` and `scripts/run_tests_local.ps1` added.
- [x] `DEVELOPER.md` added with concise local workflow.
- [x] Dependencies: Updated `requirements.txt` with modern pinned versions (torch 2.5.1, etc.).
- [x] INSTALL.md: Created with installation guide and troubleshooting.
- [x] Linting: Added black, isort, flake8 configs and scripts; applied formatting to codebase.
- [x] Tests: All 14 tests pass, including smoke tests for pipeline import.