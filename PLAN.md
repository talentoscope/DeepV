# Deep Vectorization Codebase Analysis and Improvement Plan

## Project Overview

The Deep Vectorization project aims to convert raster technical drawings (e.g., floor plans, engineering diagrams) into clean vector graphics representations using deep learning. The pipeline consists of four main modules: cleaning (noise removal), vectorization (primitive prediction), refinement (optimization), and merging (consolidation). The codebase is implemented in PyTorch and supports various datasets like ABC, PFP, and synthetic data.

**Key Strengths:**
- Modular architecture facilitating independent development
- Comprehensive pipeline from raster to vector
- Support for multiple primitive types (lines, quadratic Beziers)
- Jupyter notebooks for demonstration
- Potential for integration with emerging transformer and diffusion models for unsupervised or multimodal vectorization

**Overall Assessment:**
- Codebase is functional but shows signs of academic/research origins: inconsistent code quality, outdated dependencies, limited testing, and incomplete documentation.
- Performance and maintainability could be significantly improved.
- Opportunities for extensibility and accuracy enhancements exist, including "smarter" accuracy via hybrid models (e.g., combining transformers with Gaussian splatting for faster, more precise rendering) and expansion to new domains like mechanical parts (beyond floor plans) or symbolic elements (e.g., text/symbol spotting in drawings).
- Prioritize features that enable end-to-end CAD generation (e.g., from raster to parametric CAD sequences).

## Module-by-Module Analysis

### Cleaning Module
**Purpose:** Preprocess noisy technical drawings to remove artifacts before vectorization.

**Code Quality:**
- Mixed Python styles; some functions lack type hints.
- Hardcoded paths and magic numbers (e.g., batch sizes).
- Inconsistent error handling.

**Performance:**
- UNet models may benefit from modern architectures (e.g., UNet++).
- Training loops could be optimized with mixed precision.

**Features:**
- Supports synthetic data generation, which is good.
- Limited to binary cleaning; could extend to multi-class segmentation.

**Accuracy:**
- Uses IoU for evaluation; consider additional metrics like F1-score.

**Bugs/Issues:**
- README notes potential errors due to untested refactoring.
- Possible memory leaks in data loading.

**Documentation:**
- Sparse; README is brief and mentions untested state.

**Dependencies:**
- Relies on standard PyTorch; no major issues.

**Testing:**
- No unit tests; relies on manual validation.

**Suggestions:**
1. Refactor `main_cleaning.py` to use argparse consistently and add configuration files.
2. Implement unit tests for UNet models and data loaders.
3. Add validation for synthetic data generation quality.
4. Update to use `torchvision` transforms for data augmentation.
5. Profile training and optimize bottlenecks.
6. Extend to multi-class segmentation for handling symbols/text (e.g., doors, annotations in technical drawings).
7. Integrate vision transformers (ViTs) or data-efficient variants like DeiT for better handling of degraded inputs (e.g., photos/scans with varying noise).
8. Add unsupervised cleaning via diffusion models for inpainting gaps.

### Vectorization Module
**Purpose:** Predict vector primitives (lines, curves) from cleaned patches.

**Code Quality:**
- Complex model specifications in JSON; could be simplified.
- Long training script with many parameters; consider config classes.

**Performance:**
- Models use older PyTorch features; update to modern APIs.
- Potential for GPU memory optimization in batch processing.

**Features:**
- Supports multiple model architectures; extensible.
- Limited to specific primitive counts per patch.

**Accuracy:**
- Uses custom loss functions; validate against standard metrics.
- Evaluation relies on rasterization; ensure fidelity.

**Bugs/Issues:**
- Hardcoded device selection; improve multi-GPU support.
- Possible overfitting due to lack of regularization checks.

**Documentation:**
- Model specs need better explanation.

**Dependencies:**
- Tied to specific PyTorch version.

**Testing:**
- No automated tests for model loading/training.

**Suggestions:**
1. Refactor `train_vectorization.py` into smaller functions/classes.
2. Add hyperparameter tuning support (e.g., via Optuna).
3. Implement model checkpointing with best validation loss.
4. Add unit tests for model components.
5. Profile inference time and optimize for real-time use.
6. Support adaptive primitive counts per patch based on image complexity.
7. Upgrade transformer-based prediction to include arcs and splines.
8. Incorporate diffusion transformers for generative vectorization, enabling text-to-vector or style transfer.
9. Add symbol spotting via panoptic segmentation.

### Refinement Module
**Purpose:** Optimize predicted primitives using differentiable rendering.

**Code Quality:**
- Very long functions (400+ lines); break into smaller units.
- Mixed concerns: loading, processing, optimization.

**Performance:**
- Optimization loops may be slow; parallelize where possible.
- Memory-intensive for large images.

**Features:**
- Supports curves and lines separately; could unify.
- Uses Adam optimizer; consider alternatives.

**Accuracy:**
- Relies on rendering quality; validate against ground truth.

**Bugs/Issues:**
- Potential numerical instabilities in optimization.
- Hardcoded tolerances; make configurable.

**Documentation:**
- Inline comments sparse; add docstrings.

**Dependencies:**
- Uses custom optimization primitives; ensure compatibility.

**Testing:**
- No tests; hard to validate correctness.

**Suggestions:**
1. Refactor into classes with clear responsibilities.
2. Add early stopping and convergence checks.
3. Implement parallel optimization for multiple patches.
4. Add unit tests for primitive operations.
5. Profile and optimize rendering bottlenecks.
6. Adopt faster differentiable rendering like Bézier Splatting for speedup.
7. Integrate rendering-aware RL for feedback-driven refinement.
8. Support 3D-aware refinement, allowing conversion to parametric CAD.
9. Add adaptive parameterization for curves to preserve details in complex shapes.

### Merging Module
**Purpose:** Merge overlapping or similar primitives to simplify output.

**Code Quality:**
- Similar issues to refinement: long functions, mixed concerns.

**Performance:**
- Joining algorithms may be O(n^2); optimize for large n.

**Features:**
- Separate for curves/lines; could generalize.

**Accuracy:**
- Tolerance-based merging; tune for datasets.

**Bugs/Issues:**
- Possible loss of detail in aggressive merging.

**Documentation:**
- Minimal; explain algorithms.

**Testing:**
- None.

**Suggestions:**
1. Refactor to use data structures for efficient merging (e.g., R-trees).
2. Add configurable merging strategies.
3. Implement validation against original raster.
4. Add tests for merging logic.
5. Generalize to merge symbols/text with primitives using graph neural networks.
6. Enhance with transformer-based grouping for semantic similarity.
7. Enable sequence-to-CAD merging to convert vectors to parametric operations.

### Dataset Module
**Purpose:** Download and preprocess datasets.

**Code Quality:**
- Shell scripts; consider Python alternatives for portability.

**Performance:**
- Downloading large files; add resume support.

**Features:**
- Supports multiple datasets; good coverage.

**Accuracy:**
- Preprocessing assumes specific formats; validate robustness.

**Bugs/Issues:**
- Hardcoded URLs; may break.

**Documentation:**
- README is helpful but could detail preprocessing steps.

**Dependencies:**
- Uses wget; ensure availability.

**Testing:**
- No validation of downloaded data integrity.

**Suggestions:**
1. Replace shell scripts with Python using `requests` for downloads.
2. Add checksum validation for datasets.
3. Implement data versioning.
4. Add unit tests for preprocessing functions.
5. Integrate new large-scale datasets: ArchCAD, CAD-VGDrawing, ArchCAD-400k.
6. Add synthetic generation for arcs/symbols.
7. Support multimodal preprocessing (e.g., align raster with SVG).

### Util Files
**Purpose:** Shared utilities for data, rendering, metrics, etc.

**Code Quality:**
- Large collection; some files outdated (e.g., Python 2 remnants).
- Inconsistent naming (e.g., `os.py` shadows stdlib).

**Performance:**
- Rendering with Cairo; ensure efficient.

**Features:**
- Comprehensive; covers most needs.

**Accuracy:**
- Metrics implementations; verify correctness.

**Bugs/Issues:**
- Potential import issues due to naming conflicts.

**Documentation:**
- README exists but incomplete.

**Dependencies:**
- Many; some pinned to old versions.

**Testing:**
- Sparse.

**Suggestions:**
1. Rename conflicting files (e.g., `os.py` to `file_utils.py`).
2. Refactor into subpackages for better organization.
3. Update deprecated code.
4. Add comprehensive unit tests.
5. Document utility functions with examples.
6. Add utils for differentiable rendering (e.g., Bézier Splatting) and transformer losses.

### Notebooks
**Purpose:** Demonstrate usage and experiments.

**Code Quality:**
- Mix of code and markdown; some cells long.
- Hardcoded paths; not reproducible.

**Performance:**
- For exploration; not optimized.

**Features:**
- Good for tutorials.

**Accuracy:**
- Demonstrations; ensure outputs match expectations.

**Bugs/Issues:**
- May not run due to path dependencies.

**Documentation:**
- Self-documenting via markdown.

**Testing:**
- N/A.

**Suggestions:**
1. Refactor into scripts with config files.
2. Add error handling and assertions.
3. Make paths configurable.
4. Convert to proper tutorials with nbdev or similar.
5. Convert to interactive demos (e.g., via Gradio for text-to-vector experiments).

## Cross-Cutting Concerns

### Dependencies
- Many packages outdated (e.g., PyTorch 1.7.1 → 2.x, NumPy 1.20 → 1.24).
- Security vulnerabilities possible in old versions.
- Suggestions: Update to latest stable versions, use `poetry` or `pip-tools` for management, add security scanning (e.g., `safety`). Add diffusers library for diffusion-transformers.

### Testing
- No test suite; critical for reliability.
- Suggestions: Add `pytest`, aim for 70%+ coverage, and include integration tests for the pipeline. Run tests and linters locally (see `CONTRIBUTING.md`) and optionally add CI later if you want automated runs on a remote service.

### Documentation
- READMEs are basic; missing API docs, installation guides.
- Suggestions: Use Sphinx for docs, add docstrings to all functions, create user guide, update with latest changes. Sphinx docs with examples from research papers.

### Code Quality
- Inconsistent style, no enforced standards.
- Suggestions: Add pre-commit hooks with black, flake8, mypy; enforce PEP8; use type hints throughout.

### Performance
- No profiling; potential bottlenecks in rendering/optimization.
- Suggestions: Use `torch.profiler`, optimize data loading with DataLoader improvements, consider ONNX for inference speedup. Aim for 30x speedup via Bézier Splatting.

### Features
- Limited to specific primitives/datasets.
- Suggestions: Add support for arcs, splines; web UI for visualization; API for integration; multi-modal inputs. Multimodal inputs (e.g., text prompts for guided vectorization). Real-time vectorization via fast rendering. CAD integration: End-to-end raster-to-parametric CAD. Unsupervised modes: Vector synthesis without labels. Web UI/API: For visualization/editing.

### Accuracy
- Metrics may not cover all cases.
- Suggestions: Add comprehensive evaluation suite, compare against baselines, validate on diverse datasets. Add benchmarks on new datasets (e.g., ArchCAD for symbol spotting); compare to SOTA like VectorGraphNET (89% F1).

### Bugs
- Potential issues in untested areas.
- Suggestions: Add logging/monitoring, use issue tracker, implement error recovery.

### Security
- No obvious issues, but old deps risky.
- Suggestions: Audit dependencies, avoid shell execution where possible.

### Scalability/Extensibility
- Hardcoded limits (e.g., primitive counts).
- Suggestions: Make parameters configurable, support larger images, add plugin architecture. Plugin for new primitives (arcs/splines); distributed training on large datasets like ArchCAD-400k.

### Error Handling/Logging
- Minimal; hard to debug failures.
- Suggestions: Add structured logging, exception handling, progress bars.

### Configuration Management
- Scattered configs.
- Suggestions: Use Hydra or similar for centralized config.

## Roadmap

### Phase 1: Foundation (1 month)
- Update all dependencies to latest versions.
 - Set up testing and linting (run locally; optionally add CI later if desired).
- Add basic unit tests for core functions.
- Fix critical bugs and improve error handling.
- Refactor obvious code quality issues (naming, structure).
- Add quick-win feature: Arcs/splines support in vectorization.

### Phase 2: Optimization (1-2 months)
- Profile and optimize performance bottlenecks.
- Implement comprehensive testing (unit, integration).
- Improve documentation and add tutorials.
- Refactor long functions and modules.
- Add configuration management.
- Integrate Bézier Splatting for refinement speedup.

### Phase 3: Enhancement (2-4 months) - COMPLETED ✓
- [x] **Extended Primitive Support**: Added arcs, splines (quadratic/cubic Bézier curves), and variable primitive counts per patch
- [x] **Variable-Length Models**: Implemented autoregressive transformer decoder supporting up to 20 primitives per patch
- [x] **Variable-Length Models**: Implemented autoregressive transformer decoder supporting up to 20 primitives per patch
- [x] **CAD Export**: Added DXF/SVG export for all primitive types with parametric CAD conversion
- [x] **Web UI**: Built Gradio-based interactive demo with Bézier splatting rendering for all primitive types
- [x] **Dependency Fixes**: Resolved svgpathtools Python 3.10+ compatibility issues with optional imports
- [x] **Better Metrics**: Added curve-based Hausdorff distance metric for more accurate evaluation
- [x] **Distributed Training**: Added torch.distributed support for multi-GPU/multi-node training
- Improve accuracy with better metrics/models.
- Add scalability features (larger images, distributed training).
- Community: Open-source contributions, issue tracking.
- New primitives, datasets (ArchCAD/CAD-VGDrawing), multimodal (text-to-vector), CAD output. Prioritize diffusion-transformers for generative features.

### Phase 4: Maintenance (Ongoing)
- Regular dependency updates.
- Performance monitoring.
- Feature requests and bug fixes.
- Feature monitoring; community contributions for extensions like 3D projection.

**Estimated Effort:** 6-12 months for full overhaul, depending on team size.
**Priority:** Start with testing and dependencies for stability, then code quality and features. New and smarter features should be the priority.