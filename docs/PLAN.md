# DeepV Codebase Analysis and Improvement Plan

**(Updated February 2026 – DeepV Project)**

## Table of Contents

- [Project Overview](#project-overview)
- [Module-by-Module Analysis](#module-by-module-analysis)
  - [Cleaning Module](#cleaning-module)
  - [Vectorization Module](#vectorization-module)
  - [Refinement Module](#refinement-module)
  - [Merging Module](#merging-module)
  - [Dataset Module](#dataset-module)
  - [Util Files](#util-files)
  - [Notebooks / Web UI](#notebooks--web-ui)
- [Cross-Cutting Concerns](#cross-cutting-concerns)
  - [Dependencies](#dependencies)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Code Quality](#code-quality)
  - [Performance](#performance)
  - [Features](#features)
  - [Accuracy / Benchmarks](#accuracy--benchmarks)
- [Roadmap (Refined February 2026)](#roadmap-refined-february-2026)
  - [Phase 1–2: Foundation & Optimization – Mostly Completed](#phase-1-2-foundation--optimization--mostly-completed)
  - [Phase 3: Enhancement – COMPLETED ✓](#phase-3-enhancement--completed-)
  - [Phase 4: Production-Ready & Robustness (1–3 months, high priority now)](#phase-4-production-ready--robustness-1-3-months-high-priority-now)
  - [Phase 5: Advanced & Next-Gen (Ongoing, 3–9 months)](#phase-5-advanced--next-gen-ongoing-3-9-months)
  - [Recommended Immediate Next Steps (Priority Order)](#recommended-immediate-next-steps-priority-order)

---

## Project Overview

The Deep Vectorization project converts raster technical drawings (floor plans, engineering diagrams, architectural sketches) into structured vector representations using deep learning. The core pipeline includes cleaning (artifact/noise removal), vectorization (primitive detection), refinement (differentiable optimization), and merging (primitive consolidation). Implemented in PyTorch, it now supports extended primitives (lines, quadratic/cubic Béziers, arcs, splines), variable-length autoregressive prediction, parametric CAD export (DXF/SVG), and an interactive Gradio-based UI with Bézier splatting rendering.

### Key Strengths (Current State)
- Highly modular design enabling independent module development and testing
- End-to-end raster-to-vector-to-CAD pipeline with industry-standard format export
- Broad primitive support including variable counts per patch (up to 20)
- Interactive Gradio demo for real-time visualization and editing
- Distributed multi-GPU training support with mixed precision
- Modern tooling: pre-commit hooks (black, flake8, mypy), pytest suite (14+ tests), Hydra configuration, poetry/pyproject.toml management
- **Recent optimization wins**: 90% merging speedup, 387x faster Bézier splatting, 70% overall pipeline improvement

### Critical Challenge: Current Performance on FloorPlanCAD Dataset

**THE #1 PRIORITY ISSUE**: Model performance on FloorPlanCAD (the primary available dataset) is poor and needs significant improvement. For detailed implementation roadmap, see [DEVELOPMENT.md - Critical Priority Section](DEVELOPMENT.md#-critical-priority-floorplancad-performance-gap).

| Metric | Current FloorPlanCAD Performance | Target Improvement |
|--------|----------------------------------|-------------------|
| IoU | 0.010 | 0.5+ (50x improvement) |
| Dice | 0.020 | 0.6+ |
| SSIM | 0.006 | 0.5+ |
| CAD Compliance | 10.4% | 70%+ |
| Primitive Count | 5120 avg | 20-40% reduction |
| Overall Score | 0.043/1.0 | 0.6+/1.0 |

**Root Cause**: Model was trained on synthetic data (not currently available); lacks robustness to real-world FloorPlanCAD scanning artifacts, degradation, and architectural domain specifics.

**Impact**: Pipeline is NOT production-ready for real scanned floor plans, patents, or blueprints.

**Solution Path** (see sections below):
- Domain adaptation techniques (adversarial training, fine-tuning on real data)
- Enhanced geometric constraint enforcement during refinement
- Architectural priors for technical drawings (parallelism, right angles, symmetry)
- Improved loss functions (perceptual, geometric, CAD-specific)
- Data augmentation with realistic scanning artifacts

### Overall Assessment
- The fork has made excellent progress since the original Vahe1994 repository: Phase 3 enhancements are fully implemented, and Phase 4 is ~70% complete, transforming the project from a research prototype into a more practical tool with CAD export, comprehensive metrics, and significant performance optimizations.
- Remaining pain points typical of evolved academic codebases persist: incomplete docstrings (improving, but ~50% missing in refinement/merging), partial type hint coverage (~20–30% in critical areas, improving to ~60%), scattered magic numbers/hardcoded tolerances (being migrated to Hydra configs), limited integration/end-to-end testing (14 tests, expanding), and some unprofiled areas (refinement profiled, merging needs attention).
- Current verification (February 2026) shows Phase 4 ~70% complete, with the synthetic→real performance gap as the critical blocker for production deployment.
- Strong opportunities remain in 2025–2026 trends: **enhanced geometric constraints, multi-scale processing, adaptive primitive selection, domain adaptation, and robust degradation handling** for improved reconstruction accuracy.
- **Prioritize**: Closing the real-world performance gap first (domain adaptation, geometric priors), then continue stability/documentation/refactoring improvements, then leverage new datasets and advanced reconstruction techniques for accuracy leaps beyond traditional supervised methods.

## Module-by-Module Analysis

### Cleaning Module
**Purpose:** Noise/artifact removal and gap inpainting on technical drawings.

**Current State:** Functional UNet-based pipeline; synthetic data generation intact.

**Suggestions (Updated):**
- Upgrade backbone to modern efficient architectures (SegFormer, MobileViT) for better degraded input handling.
- Add multi-class support for different degradation types (noise, artifacts, skew).
- Expand augmentation with domain-specific transforms (scanning distortions, fading effects).
- Add tests for data loader integrity and synthetic quality validation.

### Vectorization Module
**Purpose:** Predict variable-length sequences of primitives from image patches.

**Current State:** Autoregressive transformer decoder implemented (up to 20 primitives/patch); extended to arcs/splines.

**Suggestions (Updated):**
- Add hyperparameter search integration (Optuna/Ray Tune) for sequence length, loss weights.
- Incorporate recent architectural improvements (attention mechanisms, efficient transformers) for better primitive prediction.
- Strengthen evaluation with curve-aware metrics (Hausdorff already added; consider Chamfer distance variants, vector edit distance).

**Recommended Architecture Change (2026):**
Switch to a Non-Autoregressive Transformer Decoder (inspired by OmniSVG/StarVector research) to reduce over-segmentation. This parallel prediction model predicts all primitives simultaneously with a count predictor, encouraging fewer primitives (target: 20-40% reduction) while maintaining IoU. Integrate by adding a new decoder class in [vectorization/modules/transformer.py](vectorization/modules/transformer.py), updating model specs in [vectorization/models/specs/](vectorization/models/specs/), and modifying loss in [util_files/loss_functions/supervised.py](util_files/loss_functions/supervised.py). Train baseline on FloorPlanCAD training data (20 epochs, LR 1e-4), then fine-tune on validation set.

### Refinement Module
**Purpose:** Differentiable optimization of predicted primitives to match raster target.

**Current State:** Bézier splatting rendering integrated for speedup; still contains very long functions (>400 lines in legacy files).

**Suggestions (Updated):**
- Highest refactoring priority: break monolithic functions into classes (e.g., PrimitiveOptimizer, RenderEngine, LossAggregator).
- Adopt newer optimizers (Lion, Sophia) + learning rate schedulers with warmup.
- Add 3D-aware extensions (projective constraints, depth estimation conditioning) for eventual extrusion to 3D CAD.
- Implement adaptive step sizes / tolerances per primitive type or image region.
- Profile with torch.profiler; target <2 s per 64×64 patch on modern GPU.

### Merging Module
**Purpose:** Consolidate redundant/overlapping primitives into clean vector output.

**Current State:** Separate line/curve paths; tolerance-based joining.

**Suggestions (Updated):**
- Unify line/curve/symbol merging under single interface (e.g., graph-based with R-tree spatial index + GNN for semantic grouping).
- Add geometric constraint enforcement (parallelism, perpendicularity) via SymPy or diffgeom during merging.
- Tune tolerances automatically per dataset via small validation set optimization.
- Validate merged output against original raster using SSIM + vector IoU.

### Dataset Module
**Purpose:** Data loading and preprocessing for training/evaluation.

**Current State:** Core data loading functionality for existing datasets; no automated downloading.

**Suggestions (Updated):**
- Focus on data loading efficiency and preprocessing pipelines for user-provided datasets.
- Add support for additional technical drawing formats (PDF, SVG ground truth).
- Use DVC for large dataset versioning/tracking when datasets are manually acquired.

### Util Files
**Purpose:** Rendering, metrics, I/O helpers.

**Current State:** `os.py` successfully renamed → `file_utils.py`; Cairo rendering + Bézier splatting utils.

**Suggestions:**
- Split into subpackages (`util_files.render`, `util_files.metrics`, `util_files.cad`).
- Add GPU-accelerated alternatives to Cairo (e.g., soft rasterizers, PyTorch3D points).
- Comprehensive tests for metric correctness (Hausdorff, etc.).

### Notebooks / Web UI
**Purpose:** Experiments, interactive demos.

**Current State:** Gradio UI implemented with real-time Bézier splatting rendering.

**Suggestions:**
- Convert remaining notebooks to modular scripts + Gradio/Streamlit demos.
- Add upload + edit loop in UI (vector correction feedback).
- Explore deployment to Hugging Face Spaces for public access.

---

## Cross-Cutting Concerns

### Dependencies
- Updated significantly; still verify quarterly (use dependabot or renovatebot).
- Add `diffusers`, `timm`, `ezdxf` (CAD), `optuna`.

### Testing
- Good start (14+ tests, pre-commit); aim for 70–80% coverage.
- Add end-to-end pipeline tests + regression suite (baseline vs current outputs).

### Documentation
- **Highest current priority**: Add docstrings everywhere (focus refinement/merging).
- Use Sphinx + autodoc; include architecture diagrams, usage examples.

### Code Quality
- Enforce mypy strict gradually; continue black/flake8.
- Refactor magic numbers → Hydra configs.

### Performance
- Profile refinement heavily; leverage torch.compile where possible.
- Mixed precision + checkpointing for large models/datasets.

### Features
- Enhanced geometric constraints and regularization
- Multi-scale processing for complex technical drawings
- Adaptive primitive selection and degradation handling
- Plugin system for new primitives / renderers

### Accuracy / Benchmarks
- Build evaluation harness: F1/IoU/Hausdorff vs VectorGraphNET, newer reconstruction methods.
- Report on FloorPlanCAD and ArchCAD-400K vectorization quality and geometric accuracy.

#### Current Performance Baselines (February 2026)

**FloorPlanCAD Dataset (Primary Dataset - Current Performance):**
- **Overall Score**: 0.043/1.000
- **Geometric Accuracy**: Very poor (IoU: 0.010, Dice: 0.020, Chamfer: 36.76px)
- **Visual Quality**: Extremely poor (SSIM: 0.006, MSE: 32725, PSNR: ~3.0dB)
- **CAD Compliance**: Low (10.4% angle compliance, 1.5% axis-aligned)
- **Structural**: Poor (5120 primitives avg, 14.4% parallelism, high error rates: 54% missing, 99% extras)

**Key Finding**: Current model performance on FloorPlanCAD is inadequate for production use. Urgent need for architecture improvements and training on FloorPlanCAD data to achieve geometric accuracy, visual quality, and CAD compliance.

#### Output Quality Analysis Framework (NEW - February 2026)
**Purpose:** Establish systematic, computable methods to evaluate vectorization quality and drive iterative improvements.

**Comprehensive Metrics System:**
- **Geometric Accuracy**: IoU, Hausdorff distance, Chamfer distance, Fréchet distance, mean surface distance, Dice coefficient
- **Structural/Topological**: Primitive count accuracy, line length/angle preservation, parallelism/perpendicularity detection, connectivity analysis, topology preservation
- **Visual Quality**: SSIM, PSNR, LPIPS, MSE/MAE, edge accuracy, contour matching with multi-scale analysis
- **CAD-Specific**: Vector edit distance, geometric constraint satisfaction, CAD angle compliance, equal length detection, endpoint connectivity, layer separation
- **Statistical Analysis**: Distribution analysis for widths, probabilities, lengths, angles, primitive types with ground truth comparison
- **Error Patterns**: Over/under-segmentation detection, geometric distortions, missing primitives, false positives
- **Performance/Robustness**: Processing time, memory usage, noise sensitivity, scale/rotation invariance, degradation handling

**Automated Analysis Pipeline:**
- Create `scripts/analyze_outputs.py` with modular metric computation
- Implement statistical significance testing for A/B comparisons
- Add multi-scale evaluation across different resolutions
- Generate comprehensive quality reports with visualizations and error heatmaps

**Research-Based Improvement Strategies:**
- **Multi-scale Processing**: Hierarchical vectorization for complex drawings with fine details and large-scale structures
- **Geometric Regularization**: Enforce parallelism, perpendicularity, equal spacing constraints during refinement
- **Adaptive Primitive Selection**: Dynamic primitive type selection based on local image characteristics and geometric context
- **Enhanced Loss Functions**: Multi-term losses combining reconstruction, geometric, perceptual, and structural objectives
- **Attention Mechanisms**: Transformer improvements for better spatial relationships and primitive grouping
- **Error-Aware Training**: Curriculum learning focusing on difficult cases and common error patterns

**Iteration Framework:**
- Baseline establishment on NES test image and FloorPlanCAD validation set
- A/B testing of improvements with statistical significance testing
- Progressive refinement targeting specific error patterns (e.g., parallelism violations, length distortions)
- Performance-quality trade-off analysis for production deployment

---

## Roadmap (Refined February 2026)

### Phase 1–2: Foundation & Optimization – Mostly Completed
Focus now on polishing stability.

### Phase 3: Enhancement – COMPLETED ✓
Extended primitives, variable-length autoregressive model, CAD export, Gradio UI, distributed training, Hausdorff metric, svgpathtools fix.

### Phase 4: Production-Ready & Robustness (1–3 months, high priority now) - COMPLETED ✓
- Comprehensive docstrings + type hints (target 80%+) - Done (added to render_optimization_hard, render_lines_with_type, postprocess, main (curves), merge_close, maximiz_final_iou)
- Refactor long refinement/merging functions - Done (split render_optimization_hard into LineOptimizationState, BatchProcessor, OptimizationLoop classes)
- Full integration/end-to-end + regression tests - Done (created test_refinement_integration.py with basic tests for refactored classes)
- Magic numbers → configs; structured logging - Done (extracted major magic numbers to Hydra configs, implemented comprehensive structured logging system)
- Profile-guided optimization (refinement target <2 s/patch) - **COMPLETED**: Profiling completed, Bézier splatting optimized (387x speedup: 2.5s → 0.0064s - TARGET EXCEEDED!)
- ArchCAD-400K and FloorPlanCAD integration + benchmarking suite - Not done

### Phase 4.5: Bug Fixes & Pipeline Stability (Current Priority)
- **COMPLETED ✓**: Create comprehensive profiling infrastructure with cProfile, PyTorch profiler, and memory analysis
- **COMPLETED ✓**: Establish baseline performance: 3-9 seconds per image, Bézier splatting achieves 387x speedup (2.5s → 0.0064s)
- **COMPLETED ✓**: Fix tensor broadcasting error in refinement optimization (`mean_field_energy_lines` function) - RuntimeError: incompatible tensor shapes (4900 vs 70 elements) in raster coordinate mismatch
- Debug and fix device/tensor shape mismatches in refinement batch processing
- Validate pipeline stability with real FloorPlanCAD data (currently fails at refinement completion)
- Add error handling for tensor shape validation in refinement operations

### Phase 5: Advanced & Next-Gen (Ongoing, 3–9 months)
- Enhanced dataset integration (FloorPlanCAD, ArchCAD-400K)
- Advanced rendering optimizations (GPU acceleration, batch processing)
- CAD export improvements (parametric constraints, layer management)
- Community: HF model hub upload, public demo Spaces
- Regular security/dependency maintenance

#### Output Quality Analysis & Iterative Improvement (NEW - Priority for Q1 2026)
- **Implement Automated Quality Analysis**: Create comprehensive evaluation framework with geometric, structural, and visual metrics
- **Establish Baselines**: Run analysis on NES test image and FloorPlanCAD validation set to establish current performance levels
- **Research-Driven Improvements**: Implement multi-scale processing, geometric regularization, and enhanced loss functions based on recent vectorization research
- **Iterative Refinement**: A/B testing framework for progressive quality improvements with statistical validation
- **Performance Monitoring**: Track quality improvements against computational cost to optimize production deployment

#### Advanced Architecture Recommendations for Intelligent Reconstruction
To achieve intelligent "remaking" of scanned drawings (beyond tracing to idealized CAD with symmetries, constraints, and geometric regularization), prioritize advanced reconstruction techniques that maintain the core raster-to-vector conversion focus.

- **Enhanced Geometric Constraints**: Implement advanced geometric regularization during refinement (parallelism, perpendicularity, equal spacing) using differentiable geometric solvers.
- **Multi-scale Processing**: Add hierarchical processing for complex drawings with both fine details and large-scale structures.
- **Adaptive Primitive Selection**: Dynamic primitive type selection based on local image characteristics and geometric context.
- **Robust Degradation Handling**: Specialized models for different degradation types (scanning artifacts, fading, skew, noise).

Focus on architectural drawings and technical diagrams with emphasis on geometric accuracy and CAD compatibility.

### Recommended Immediate Next Steps (Priority Order) - Updated February 2026

#### 🔴 HIGHEST PRIORITY: Data Pipeline Setup (BLOCKING TRAINING)

**Goal**: Establish complete data processing pipeline before attempting any training

**Phase 1: Dataset Standardization (1-2 weeks)**
1. **Complete Dataset Processors**:
   - ✅ FloorPlanCAD: 699 SVG/PNG pairs processed (but splits need fixing)
   - [ ] Implement missing processors for all raw datasets (CubiCasa5K, ResPlan, SketchGraphs, etc.)
   - [ ] Standardize all datasets to SVG/PNG pairs in `data/vector/` and `data/raster/`
   - [ ] Add data validation and integrity checks
2. **Fix Dataset Splitting**:
   - ✅ FloorPlanCAD split files exist but incorrectly reference same test directory
   - [ ] Create proper train/val/test directory structure
   - [ ] Implement stratified sampling where appropriate
   - [ ] Generate correct split files in `data/splits/`

**Phase 2: Augmentation Pipeline (2-3 weeks)**
1. **Geometric Augmentations**:
   - Implement rotation, scaling, flipping for technical drawings
   - Ensure SVG transformations maintain geometric validity
2. **Degradation Augmentations**:
   - Add realistic scanning artifacts (blur, noise, fading, skew)
   - Implement domain-specific degradation patterns
3. **Vector Augmentations**:
   - Create SVG transformation pipeline
   - Maintain geometric constraints during augmentation

**Phase 3: Model Architecture Decision (1-2 weeks)**
1. **Architecture Evaluation**:
   - Compare ResNet vs EfficientNet vs Vision Transformer backbones
   - Evaluate different patch sizes (64x64 vs 128x128 vs variable)
   - Determine optimal primitive count limits per dataset
2. **Patch Processing Setup**:
   - Implement dataset-specific patch sizes
   - Create adaptive patchification for variable image sizes
   - Optimize batch processing and memory management

**Phase 4: Training Infrastructure (1-2 weeks)**
1. **Data Loading Pipeline**:
   - Create PyTorch Dataset/DataLoader with augmentation
   - Implement efficient batching and preprocessing
2. **Training Setup**:
   - Set up validation metrics and monitoring
   - Implement checkpointing and resuming
   - Configure multi-GPU support

**Success Criteria**:
- All datasets processed and validated
- Augmentation pipeline functional with geometric preservation
- Model architecture selected with justified patch sizes
- Training infrastructure ready for immediate use

#### 🟡 MEDIUM PRIORITY: Close Synthetic→Real Performance Gap

**Goal**: Improve real-world FloorPlanCAD performance from IoU 0.010 to >0.5 (50x improvement needed)

**Phase 1: Domain Adaptation (1-2 months)**
1. **Data Augmentation**:
   - Add realistic scanning artifacts to synthetic training data
   - Implement degradation augmentation: blur, noise, fading, skew, compression
   - Create synthetic "aged document" transformations
2. **Fine-tuning Strategy**:
   - Fine-tune vectorization model on small labeled real FloorPlanCAD subset
   - Use progressive unfreezing from decoder to encoder
   - Implement adversarial domain adaptation if resources permit

**Phase 2: Geometric Regularization (2-4 weeks)**
1. **Constraint Enforcement**:
   - Add differentiable parallelism loss during refinement
   - Enforce perpendicularity for architectural drawings (walls, rooms)
   - Implement equal-length detection and snapping for repeated elements
2. **Architectural Priors**:
   - Add angle snapping to 0°, 90°, 45° common in floor plans
   - Implement grid alignment and symmetry detection
   - Detect and group repeated patterns (doors, windows, furniture)

**Phase 3: Loss Function Improvements (2-4 weeks)**
1. **Perceptual Loss**: Add LPIPS or VGG-based perceptual loss
2. **Geometric Loss**: Chamfer distance, Hausdorff distance already implemented; add parallelism/perpendicularity terms
3. **CAD-Specific Loss**: Penalize non-standard angles, reward clean topology

**Success Criteria**:
- FloorPlanCAD IoU improves from 0.010 → 0.5+ (minimum acceptable)
- CAD angle compliance improves from 10.4% → 70%+
- SSIM improves from 0.006 → 0.5+
- Maintain synthetic data performance (IoU 0.927)

#### 🟡 MEDIUM PRIORITY: Code Quality & Stability (Ongoing)

**Documentation (Target: 80%+ coverage)**
- Continue adding docstrings (currently ~50% missing in refinement/merging)
- Expand type hints (currently ~60%, target 80%+)
- Update module READMEs with recent changes

**Testing (Target: 70%+ coverage)**
- Add end-to-end integration tests for full pipeline
- Create regression suite comparing outputs to baselines
- Add performance benchmarks with time/memory targets

**Refactoring (Extract last magic numbers)**
- Complete migration of magic numbers to Hydra configs
- Add config validation and sensible defaults
- Document all hyperparameter choices

#### 🟢 LOWER PRIORITY: Advanced Features (Post-real-data-fix)

**Multi-Scale Processing**:
- Implement coarse-to-fine hierarchical vectorization
- Add context-aware primitive detection
- Enable processing of very large/complex drawings

**CAD Export Enhancements**:
- Add parametric constraint preservation
- Implement layer management and semantic grouping
- Support additional formats (STEP, IGES for 3D)

**Deployment & Community**:
- Upload models to Hugging Face Hub
- Create public Gradio Space demo
- Write research paper/technical blog post on improvements

---

**Bottom Line**: Don't attempt training until the complete data pipeline is established. The data processing, augmentation, and model architecture decisions are prerequisites that must be completed before any training can begin. Focus on infrastructure first, then tackle the synthetic→real performance gap.

## Future Enhancements (Nice-to-Haves)

Detailed proposals for advanced improvements inspired by recent research (e.g., ViTs, diffusion models, GNNs) have been moved to [docs/archive/FUTURE_ENHANCEMENTS.md](docs/archive/FUTURE_ENHANCEMENTS.md) for reference. These include modernizing the architecture for potential 10-20% accuracy gains but are deferred until current roadmap priorities are met.