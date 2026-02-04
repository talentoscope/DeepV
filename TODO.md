# DeepV TODO Checklist (Updated February 2026)

This file is a comprehensive, actionable checklist derived from `PLAN.md`. Use it as a living checklist for local development. Items marked [x] are complete; [ ] are pending. Prioritized based on current state (Phase 3 completed, Phase 4 active).

## Phase 3 — Enhancements (COMPLETED ✓)
- [x] Add support for arcs, splines, and variable primitive counts per patch.
- [x] Upgrade vectorization with autoregressive transformer decoder (up to 20 primitives/patch).
- [x] Add CAD export (DXF/SVG) and parametric CAD conversion.
- [x] Build Gradio-based web UI with Bézier splatting rendering.
- [x] Improve metrics (added curve-based Hausdorff distance).
- [x] Add distributed training support (torch.distributed).
- [x] Fix svgpathtools Python 3.10+ compatibility.

## Phase 4 — Production-Ready & Robustness (ACTIVE, High Priority)
### Documentation & Type Safety (HIGH PRIORITY)
- [x] Add comprehensive docstrings to all functions (focus: refinement/merging modules, e.g., lines_refinement_functions.py).
- [x] Add type hints to function signatures (target 80%+ coverage across codebase). [PARTIALLY COMPLETE - Added to key functions in merging and refinement modules]
- [x] Improve error handling with specific exception types (refinement/merging operations).
- [x] Add structured logging to critical paths (training, inference, refinement).

### Code Quality & Refactoring (HIGH PRIORITY)
- [x] Refactor long functions in refinement (400+ lines) into smaller units with clear responsibilities.
- [x] Extract magic numbers and hardcoded values into configurable parameters (Hydra configs).
- [x] Add configuration files for tolerance-based merging and refinement hyperparameters.
- [x] Consolidate separate line/curve refinement and merging pipelines into unified interfaces.

### Testing & Validation
- [x] Expand unit test coverage to 70%+ (focus: refinement pipeline, merging logic). [PARTIALLY COMPLETE - Added basic tests for refactored functions, existing tests verified working]
- [x] Add integration tests for full pipeline (cleaning → vectorization → refinement → merging).
- [x] Add regression tests comparing outputs against baseline results.
- [x] Add performance benchmarks with expected time/memory targets.

### Performance & Optimization
- [x] Profile and optimize refinement bottlenecks (target: <2s per 64x64 patch). [COMPLETED - Identified rendering as bottleneck, created 3.5x faster GPU renderer]
- [x] Add mixed-precision training support for memory-efficient large model training.
- [x] Implement checkpoint resumption to enable long training runs without interruption.
- [x] Add early stopping validation on training script.

### Dataset & Evaluation
- [ ] Add benchmarking pipeline for ArchCAD, CAD-VGDrawing datasets.
- [ ] Implement comprehensive evaluation suite (compare F1, IoU, Hausdorff vs SOTA).
- [ ] Add dataset-specific evaluation reports and visualization.
- [ ] Support for synthetic dataset generation with variable complexity.

## Phase 5 — Advanced & Next-Gen (Ongoing, 3–9 months)
- [ ] Explore diffusion-transformer models for generative vectorization (text+image conditioning).
- [ ] Add panoptic symbol spotting + vector merging.
- [ ] Implement multimodal inputs (text prompts, style references).
- [ ] Explore VLM distillation (OmniSVG-inspired) for complex SVGs.
- [ ] Community: HF model hub upload, public demo Spaces.
- [ ] Regular security/dependency maintenance.

## Recommended Immediate Next Steps (Priority Order)
### 1–2 weeks Quick Wins
- [x] Extract all refinement/merging magic numbers to Hydra configs.
- [x] Add docstrings to 10–15 most complex functions.
- [x] Extend type hints in refinement pipeline.

### 2–6 weeks Medium Term
- [ ] Refactor refinement long functions → classes/modules.
- [ ] Build basic end-to-end integration test suite.
- [ ] Add structured logging + exception hierarchy.

### 2–4 months Long Term
- [ ] Reach high type-hint coverage + strict mypy.
- [ ] Full performance profiling + targeted optimizations.
- [ ] Integrate ArchCAD-400K + symbol spotting evaluation.
- [ ] Explore diffusion-transformer prototype for generative mode.

## Maintenance & Community (Ongoing)
- [x] Add automated dependency security scans and periodic updates.
- [x] Add issues/labels template and contribution guide.
- [x] Local-first testing policy (no CI, run tests locally).

## Completed Foundation Items (Phases 1-2)
- [x] Update core dependencies and create pinned requirements.
- [x] Add requirements-dev.txt and Windows test runner.
- [x] Add DEVELOPER.md with venv/run instructions.
- [x] Rename util_files/os.py → file_utils.py and update imports.
- [x] Add unit tests for merging, util_files, refinement smoke tests.
- [x] Refactor main_cleaning.py with consistent argparse.
- [x] Add linting/formatting (black, flake8, pre-commit).
- [x] Add Sphinx docs and API reference.
- [x] Add configuration management (Hydra).
- [x] Profile and optimize rendering (Bézier Splatting implemented).
- [x] Optimize merging with spatial indices.