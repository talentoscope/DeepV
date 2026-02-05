# DeepV TODO Checklist

*Updated February 2026 - Comprehensive development roadmap for DeepV vectorization framework*

## Table of Contents

- [Phase 3 - Enhancements (COMPLETED ✓)](#phase-3---enhancements-completed-)
- [Phase 4 - Production-Ready & Robustness (ACTIVE)](#phase-4---production-ready--robustness-active)
- [Phase 5 - Advanced & Next-Gen (Future)](#phase-5---advanced--next-gen-future)

---

## Phase 3 - Enhancements (COMPLETED ✓)

### Extended Primitives Support
- [x] Support arcs, splines, and variable primitive counts per patch
- [x] Upgrade vectorization with autoregressive transformer decoder (up to 20 primitives/patch)
- [x] Add CAD export (DXF/SVG) and parametric CAD conversion

### User Interface & Deployment
- [x] Build Gradio-based web UI with Bézier splatting rendering
- [x] Improve metrics (added curve-based Hausdorff distance)
- [x] Add distributed training support (torch.distributed)

### Compatibility & Maintenance
- [x] Fix svgpathtools Python 3.10+ compatibility

---

## Phase 4 - Production-Ready & Robustness (ACTIVE)

### Documentation & Type Safety (HIGH PRIORITY)

- [x] Add comprehensive docstrings to all functions (focus: refinement/merging modules)
- [x] Add type hints to function signatures (target 80%+ coverage across codebase)
- [x] Improve error handling with specific exception types (refinement/merging operations)
- [x] Add structured logging to critical paths (training, inference, refinement)

### Code Quality & Refactoring (HIGH PRIORITY)

- [x] Refactor long functions in refinement (400+ lines) into smaller units with clear responsibilities
- [x] Extract magic numbers and hardcoded values into configurable parameters (Hydra configs)
- [x] Add configuration files for tolerance-based merging and refinement hyperparameters
- [x] Consolidate separate line/curve refinement and merging pipelines into unified interfaces

### Testing & Validation

- [x] Expand unit test coverage to 70%+ (focus: refinement pipeline, merging logic)
- [x] Add integration tests for full pipeline (cleaning → vectorization → refinement → merging)
- [x] Add regression tests comparing outputs against baseline results
- [x] Add performance benchmarks with expected time/memory targets

### Performance & Optimization

- [x] Profile and optimize refinement bottlenecks (target: <2s per 64x64 patch)
- [x] Add mixed-precision training support for memory-efficient large model training
- [x] Implement checkpoint resumption to enable long training runs without interruption
- [x] Add early stopping validation on training script

### Dataset & Evaluation

- [x] Add benchmarking pipeline for custom datasets
- [x] Implement comprehensive evaluation suite (compare F1, IoU, Hausdorff vs SOTA)
- [x] Add dataset-specific evaluation reports and visualization

---

## Phase 5 - Advanced & Next-Gen (Future)

### Generative AI Integration
- [ ] Explore diffusion-transformer models for generative vectorization (text+image conditioning)
- [ ] Add panoptic symbol spotting + vector merging (target: ArchCAD-400K and FloorPlanCAD)
- [ ] Implement multimodal inputs (text prompts, style references) using DeepPatent2 captions
- [ ] Explore VLM distillation (OmniSVG-inspired) for complex SVGs

### Community & Ecosystem
- [ ] HF model hub upload, public demo Spaces
- [ ] Regular security/dependency maintenance
- [ ] Multi-language support (Rust acceleration for bottlenecks)

---

## Implementation Notes

### Completed Refactoring (Phase 4 Progress)
- ✅ **LineTensor class**: Broke down 40+ line `constrain_parameters()` into 6 focused methods
- ✅ **Web UI**: Extracted repetitive CAD export logic into helper functions
- ✅ **CAD Export**: Modularized primitive-specific export functions
- ✅ **Pipeline Runner**: Created `PipelineRunner` class to encapsulate main processing logic
- ✅ **Documentation**: Comprehensive README overhaul with table of contents, clear sections, and badges

### Current Priorities
1. **Performance**: Profile remaining bottlenecks in rendering pipeline
2. **Testing**: Expand integration test coverage
3. **Documentation**: Complete type hints across critical modules
4. **User Experience**: Improve error messages and validation

### Success Metrics
- **Code Quality**: 80%+ test coverage, <10 complexity score for functions
- **Performance**: <2s per patch, <8GB GPU memory for 128x128 images
- **Reliability**: <1% failure rate on clean inputs
- **Usability**: Clear error messages, comprehensive documentation

---

*This roadmap is actively maintained. Items marked [x] are complete; [ ] are pending. Prioritization based on user impact and technical debt reduction.*

## Recommended Immediate Next Steps (Priority Order)
### 1–2 weeks Quick Wins
- [x] Extract all refinement/merging magic numbers to Hydra configs.
- [x] Add docstrings to 10–15 most complex functions.
- [x] Extend type hints in refinement pipeline.

### 2–6 weeks Medium Term
- [x] Refactor refinement long functions → classes/modules. [COMPLETED - Refactored refinement_for_curves.py into modular classes]
- [x] Build basic end-to-end integration test suite.
- [ ] Add structured logging + exception hierarchy.

### 2–4 months Long Term
- [ ] Reach high type-hint coverage + strict mypy.
- [ ] Full performance profiling + targeted optimizations.
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
- [x] Convert all shell scripts (.sh, .ps1) to cross-platform Python equivalents.