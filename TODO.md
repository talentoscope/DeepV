# DeepV TODO Checklist

*Updated February 2026 - Comprehensive development roadmap for DeepV vectorization framework*

## 🔴 CRITICAL PRIORITY: FloorPlanCAD Performance Improvement

**THE BLOCKING ISSUE**: DeepV performs very poorly on FloorPlanCAD (the primary available dataset). This is the #1 priority that must be addressed before other features.

| Dataset | IoU | Dice | SSIM | Overall Score |
|---------|-----|------|------|---------------|
| **FloorPlanCAD** | 0.010 | 0.020 | 0.006 | 0.043/1.0 |

**Root Cause**: Model was trained on synthetic data (not currently available); lacks robustness to real-world FloorPlanCAD scanning artifacts, degradation, and architectural domain specifics.

**Impact**: Pipeline is NOT production-ready for real scanned floor plans, patents, or blueprints.

**Action Items** (see [PLAN.md](PLAN.md) for details):
- [ ] **Architecture Improvement**: Implement Non-Autoregressive Transformer for reduced over-segmentation (HIGH PRIORITY - Start Here)
- [ ] **Training on FloorPlanCAD**: Train/fine-tune model directly on FloorPlanCAD data with domain-specific augmentations
- [ ] **Geometric Regularization**: Enforce parallelism, perpendicularity, angle snapping during refinement
- [ ] **Data Augmentation**: Add realistic scanning artifacts, blur, noise, fading to FloorPlanCAD training data
- [ ] **Loss Function Improvements**: Add perceptual loss (LPIPS), geometric losses, CAD-specific penalties

**Success Criteria**:
- **Phase 1 (Architecture)**: Non-autoregressive decoder implemented and trains successfully on FloorPlanCAD data
- **Phase 2 (Performance)**: Primitive count reduces by 20-40% (from ~5120 to ~3000-4000 avg) while maintaining IoU ≥0.010
- **Phase 3 (Quality)**: FloorPlanCAD IoU improves from 0.010 → 0.5+ (minimum 50x improvement), CAD angle compliance improves from 10.4% → 70%+, SSIM improves from 0.006 → 0.5+

**DO NOT PROCEED** with advanced features, optimizations, or deployment until this issue is resolved.

**Implementation Timeline** (6-8 days total):
- **Days 1-2**: Architecture design and loss function updates ✅ COMPLETED
- **Days 3-5**: Training setup and baseline training on FloorPlanCAD
- **Days 6-8**: Integration testing and performance validation

---

## Table of Contents

- [🔴 Critical Priority: Synthetic→Real Gap](#-critical-priority-syntheticreal-performance-gap)
- [Phase 3 - Enhancements (COMPLETED ✓)](#phase-3---enhancements-completed-)
- [Phase 4 - Production-Ready & Robustness (ACTIVE)](#phase-4---production-ready--robustness-active)
- [Phase 5 - Advanced & Next-Gen (Future)](#phase-5---advanced--next-gen-future)

---

## Phase 3 - Enhancements (COMPLETED ✓)

### Performance Optimizations
- [x] **MAJOR BOTTLENECK FIXED**: Optimized `maximiz_final_iou` greedy algorithm - reduced from 53s to 5s (90% improvement)
- [x] **RENDERING OPTIMIZED**: Limited IOU optimization to 50 candidate lines instead of all ~582 lines
- [x] **SPATIAL INDEXING FIXED**: Corrected bounding box ordering in `merge_close` and reduced window size from 200 to 30 pixels
- [x] **PIPELINE SPEEDUP**: Total pipeline time reduced from 77s to 23s (70% overall improvement)
- [x] **LARGE IMAGE SUPPORT**: Fixed tensor size mismatch in `reinit_excess_lines` - pipeline now handles large images (1121x771px NES image successfully processed)
- [x] Pipeline profiling established with memory/cProfile analysis

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

- [x] Profile refinement bottlenecks (target: <2s per 64x64 patch) - **COMPLETED**: Identified rendering as bottleneck (0.04-0.05s), Bézier splatting needs optimization
- [x] Optimize Bézier splatting rendering performance (387x speedup achieved! 2.5s → 0.0064s - TARGET MET!)
- [x] Add mixed-precision training support for memory-efficient large model training
- [x] Implement checkpoint resumption to enable long training runs without interruption
- [x] Add early stopping validation on training script

### Critical Bug Fixes (IMMEDIATE PRIORITY)

- [x] **COMPLETED**: Create comprehensive profiling script for pipeline performance analysis
- [x] **COMPLETED**: Establish baseline performance metrics (3-9s per image, Bézier splatting 0.0064s/patch)
- [x] **COMPLETED**: Fix tensor broadcasting error in refinement optimization - RuntimeError in `mean_field_energy_lines` function: "The size of tensor a (4900) must match the size of tensor b (70) at non-singleton dimension 2"
- [ ] Debug tensor shape mismatches in `torch.broadcast_tensors` call during refinement batch processing
- [ ] Add tensor shape validation and error handling in refinement operations
- [ ] Validate pipeline stability with real FloorPlanCAD data (currently fails at refinement completion)

### Output Quality Analysis & Iterative Improvement (NEW - HIGH PRIORITY)

#### Comprehensive Metrics Framework Implementation
- [x] **Geometric Accuracy Metrics**: IoU, Dice coefficient, Chamfer distance ✅
- [x] **Structural/Topological Metrics**: Primitive count, line length/angle statistics, CAD angle compliance, parallelism detection ✅
- [x] **Visual Quality Metrics**: MSE, MAE, RMSE, SSIM (structural similarity) ✅
- [x] **CAD-Specific Metrics**: Equal length detection, endpoint connectivity analysis ✅
- [x] **Statistical Distribution Analysis**: Width/probability/length/angle distributions with percentiles ✅
- [x] **Error Pattern Detection**: Over-segmentation, missing/extra primitives analysis ✅
- [x] **Overall Quality Scoring**: Weighted combination framework (geometric 40%, visual 50%, structural 10%) ✅
- [x] **Comprehensive Analysis Script**: `scripts/comprehensive_analysis.py` with full framework ✅
- [x] **Ground Truth Extraction**: FloorPlanCAD SVG parser for structural primitives (`scripts/extract_floorplancad_ground_truth.py`) ✅
- [x] **Vector-to-Vector Comparison**: Direct geometric comparison between ground truth and predicted vectors ✅
- [ ] **Performance & Robustness Metrics**: Processing time, memory usage, noise sensitivity analysis
- [ ] **Multi-Scale Evaluation**: Hierarchical quality assessment across different scales

#### Baseline Establishment & Research
- [x] **FloorPlanCAD Batch Analysis**: Complete comprehensive analysis on 10 random images (IoU: 0.010, SSIM: 0.006, CAD compliance: 10.4%, Overall: 0.043/1.000) ✅
- [x] **Analysis Report Generation**: JSON report saved to `logs/outputs/single_test/comprehensive_analysis.json` ✅
- [ ] Analyze FloorPlanCAD validation set for comprehensive baseline metrics
- [ ] Research recent vectorization literature for improvement strategies
- [ ] Identify specific error patterns (over-segmentation, geometric distortions, missing primitives)
- [ ] **Implement Non-Autoregressive Transformer Architecture**: Switch vectorization decoder to parallel prediction model (OmniSVG/StarVector inspired) to reduce over-segmentation. Target 20-40% primitive count reduction while maintaining IoU. Update [vectorization/modules/transformer.py](vectorization/modules/transformer.py), model specs, and loss functions. Train baseline on FloorPlanCAD training data (20 epochs, LR 1e-4), then fine-tune on validation set.
  - [x] **Architecture Design (1-2 days)**: Analyze current autoregressive decoder in [vectorization/modules/transformer.py](vectorization/modules/transformer.py), design parallel prediction head with primitive count estimator, update model specs in [vectorization/models/specs/](vectorization/models/specs/)
  - [x] **Loss Function Updates (1 day)**: Modify [util_files/loss_functions/supervised.py](util_files/loss_functions/supervised.py) to handle parallel predictions, add count prediction loss component, ensure backward compatibility
  - [ ] **Training Setup (2-3 days)**: Create FloorPlanCAD training script using existing data loaders, set up 20-epoch baseline training (LR 1e-4) on FloorPlanCAD train split, implement validation monitoring for primitive count reduction
  - [ ] **Integration & Testing (2 days)**: Update pipeline integration in [pipeline_unified.py](pipeline_unified.py), add unit tests for new decoder components, validate primitive count reduction (target: 3000-4000 primitives vs current 5120)

### Current Performance Baseline (FloorPlanCAD - February 2026)
**Geometric Accuracy**: Very poor (IoU: 0.010, Dice: 0.020, Chamfer: 36.76px)  
**Visual Quality**: Extremely poor (SSIM: 0.006, MSE: 32725, MAE: 140)  
**CAD Compliance**: Very low (10.4% angle compliance, only 104/5120 primitives)  
**Structural Preservation**: Poor (5120 primitives, low parallelism 14.4%, axis-aligned: 1.5%)  
**Error Patterns**: High over-segmentation (10%), massive missing primitives (54%), excessive extras (99%)  
**Overall Score**: 0.043/1.000 (needs significant improvement in all areas)

#### FloorPlanCAD Validation Results (February 2026)
**Geometric Accuracy**: Very poor (IoU: 0.010, Dice: 0.020, Chamfer: 36.76px)  
**Visual Quality**: Extremely poor (SSIM: 0.006, MSE: 32725, MAE: 140)  
**CAD Compliance**: Very low (10.4% angle compliance, only 104/5120 primitives)  
**Structural Preservation**: Poor (5120 primitives, low parallelism 14.4%, axis-aligned: 1.5%, connectivity: 2.8%)  
**Error Patterns**: High over-segmentation (10%), massive missing primitives (54%), excessive extras (99%)  
**Overall Score**: 0.043/1.000

**Vector-to-Vector Ground Truth Comparison** (NEW - February 2026):
**Ground Truth Primitives**: 6 structural wall lines from FloorPlanCAD SVG
**Predicted Primitives**: 2459 (410x over-segmentation!)
**Geometric Error**: Chamfer distance 369.9px, max error 1358.9px
**Stroke Width Error**: Mean difference 4.3px, MAE 3.3px  
**Length Error**: Mean difference 29.7px, MAE 43.4px
**Quality Score with Ground Truth**: 0.097/1.000 (2.3x improvement over self-comparison)

**Key Finding**: DeepV performs poorly on FloorPlanCAD data with major gaps in geometric accuracy, visual quality, and structural validity. Ground truth comparison reveals 410x over-segmentation and significant geometric distortions. Architecture improvements and training on FloorPlanCAD data are urgently needed.

#### Research-Based Improvements Implementation
- [ ] Implement multi-scale processing for complex technical drawings
- [ ] Add geometric regularization (parallelism, perpendicularity constraints)
- [ ] Enhance loss functions with multi-term objectives (reconstruction + geometric + perceptual)
- [ ] Improve attention mechanisms for better spatial relationships
- [ ] Add adaptive primitive selection based on local image characteristics

#### Iterative Testing & Validation
- [ ] Build A/B testing framework for comparing improvement iterations
- [ ] Implement statistical significance testing for quality improvements
- [ ] Create automated regression testing against baseline metrics
- [ ] Track performance-quality trade-offs for production optimization

### Code Modernization & Refactoring

**📋 Dedicated tracking moved to [REFACTOR.md](REFACTOR.md)**

*Active systematic codebase improvement initiative covering:*
- ✅ **Completed**: Project branding, PEP 8 compliance (11+ files), import organization
- 🔄 **In Progress**: String formatting modernization, exception handling standardization
- 🎯 **Future**: Architecture modernization, performance optimization, testing infrastructure

*See [REFACTOR.md](REFACTOR.md) for detailed progress tracking and implementation notes.*

---

## Implementation Notes

### Refactoring Priorities (Code Modernization)
1. **Identity & Branding**: Establish clear DeepV identity separate from original project
2. **Code Quality**: Achieve high standards of code quality and maintainability
3. **Architecture**: Modernize design patterns and architectural decisions
4. **Performance**: Optimize for both speed and resource usage
5. **Testing**: Build comprehensive, maintainable test infrastructure
6. **Documentation**: Ensure code is well-documented and self-explanatory

### Success Metrics (Code Modernization)
- **Code Quality**: <10 complexity score, 90%+ PEP 8 compliance, zero critical linting issues
- **Maintainability**: <50 lines per function, clear separation of concerns, minimal code duplication
- **Testability**: 80%+ test coverage, fast test execution, reliable CI pipeline
- **Developer Experience**: Clear documentation, helpful error messages, good IDE support

### Refactoring Guidelines
- **Incremental Changes**: Make small, focused changes that can be easily reviewed
- **Backward Compatibility**: Maintain API compatibility unless explicitly breaking changes are needed
- **Testing First**: Ensure comprehensive tests exist before refactoring
- **Documentation Updates**: Update documentation to reflect code changes
- **Performance Monitoring**: Track performance impact of refactoring changes

### Advanced Reconstruction Features
- [ ] Enhanced geometric constraints (parallelism, perpendicularity, equal spacing)
- [ ] Multi-scale processing for complex drawings
- [ ] Adaptive primitive selection based on local characteristics
- [ ] Robust degradation handling for different input types
- [ ] Differentiable geometric regularization during refinement
- [ ] Multi-term loss functions (reconstruction + geometric + perceptual)
- [ ] Attention-enhanced spatial relationship modeling

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
### IMMEDIATE CRITICAL FIXES (Blocker)
- [ ] **CRITICAL**: Fix tensor broadcasting error in refinement optimization - RuntimeError in `mean_field_energy_lines` function preventing pipeline completion
- [ ] Debug tensor shape mismatches in `torch.broadcast_tensors` during refinement batch processing
- [ ] Add tensor shape validation and error handling in refinement operations

### Quality Analysis & Improvement (NEW - High Priority)
- [ ] Create comprehensive output analysis framework (`scripts/analyze_outputs.py`)
- [ ] Establish quality baselines on NES test image and FloorPlanCAD validation set
- [ ] Implement research-based improvements (multi-scale processing, geometric regularization)
- [ ] Build A/B testing infrastructure for iterative improvement validation

### 1–2 weeks Quick Wins
- [x] Extract all refinement/merging magic numbers to Hydra configs.
- [x] Add docstrings to 10–15 most complex functions.
- [x] Extend type hints in refinement pipeline.

### 2–6 weeks Medium Term
- [x] Refactor refinement long functions → classes/modules. [COMPLETED - Refactored refinement_for_curves.py into modular classes]
- [x] Build basic end-to-end integration test suite.
- [x] Add structured logging + exception hierarchy.

### 2–4 months Long Term
- [x] Reach high type-hint coverage + strict mypy. **COMPLETED**: Added type hints to key functions in refinement utils (reinit_excess_lines, get_random_line, render functions, energy functions) and improved import coverage.
- [ ] Full performance profiling + targeted optimizations.
- [ ] Enhanced CAD export with parametric constraints and layer management.
- [ ] Multi-scale processing for complex technical drawings.

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