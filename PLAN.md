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
- End-to-end raster-to-vector-to-CAD pipeline
- Broad primitive support including variable counts per patch (up to 20)
- Interactive Gradio demo for real-time visualization and editing
- Distributed multi-GPU training support
- Modern tooling: pre-commit hooks (black, flake8, mypy), pytest suite (14+ tests), Hydra configuration, poetry/pyproject.toml management

### Overall Assessment
- The fork has made excellent progress since the original Vahe1994 repository: Phase 3 enhancements are fully implemented, transforming the project from a research prototype into a more practical tool with CAD export and user-facing demo.
- Remaining pain points typical of evolved academic codebases persist: incomplete docstrings (especially in refinement/merging), partial type hint coverage (~20–30% in critical areas), scattered magic numbers/hardcoded tolerances, limited integration/end-to-end testing, and unprofiled performance bottlenecks (particularly refinement).
- Current verification (February 2026) shows Phase 4 ~60% complete, with refactoring, documentation, and testing as immediate priorities; Phase 5 remains untouched.
- Strong opportunities remain in 2025–2026 trends: diffusion transformers for generative vectorization, multimodal (text+image) conditioned synthesis, panoptic symbol spotting on datasets like ArchCAD-400K and FloorPlanCAD, and tighter integration with emerging SVG generation models (e.g., OmniSVG-style VLM+distillation approaches or LayerTracer diffusion transformers).
- Prioritize stability/documentation/refactoring first, then leverage new datasets and generative techniques for accuracy leaps beyond traditional supervised methods.

## Module-by-Module Analysis

### Cleaning Module
**Purpose:** Noise/artifact removal and gap inpainting on technical drawings.

**Current State:** Functional UNet-based pipeline; synthetic data generation intact.

**Suggestions (Updated):**
- Upgrade backbone to modern efficient architectures (SegFormer, MobileViT, or SAM 2 adapter) for better degraded input handling.
- Add multi-class support (symbols, text, annotations) via panoptic heads.
- Integrate diffusion-based inpainting (e.g., via diffusers library) for unsupervised gap filling.
- Expand augmentation with domain-specific transforms (scanning distortions, handwriting overlays).
- Add tests for data loader integrity and synthetic quality validation.

### Vectorization Module
**Purpose:** Predict variable-length sequences of primitives from image patches.

**Current State:** Autoregressive transformer decoder implemented (up to 20 primitives/patch); extended to arcs/splines.

**Suggestions (Updated):**
- Add hyperparameter search integration (Optuna/Ray Tune) for sequence length, loss weights.
- Explore conditioning on text prompts via CLIP or modern VLMs for guided vectorization.
- Incorporate recent generative approaches (diffusion transformers, score distillation) for zero-shot or few-shot adaptation.
- Add panoptic symbol spotting head (leveraging ArchCAD-400K and FloorPlanCAD annotations) to detect/ vectorize discrete elements (doors, windows, furniture symbols).
- Strengthen evaluation with curve-aware metrics (Hausdorff already added; consider Chamfer distance variants, vector edit distance).

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
- Add multimodal loaders (raster + SVG + text descriptions where available).
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
- Multimodal / text-guided vectorization
- Symbol spotting + vectorization on ArchCAD-400K and FloorPlanCAD datasets
- Plugin system for new primitives / renderers

### Accuracy / Benchmarks
- Build evaluation harness: F1/IoU/Hausdorff vs VectorGraphNET, newer diffusion-based methods.
- Report on ArchCAD-400K and CAD-VGDrawing symbol spotting + vectorization quality.

---

## Roadmap (Refined February 2026)

### Phase 1–2: Foundation & Optimization – Mostly Completed
Focus now on polishing stability.

### Phase 3: Enhancement – COMPLETED ✓
Extended primitives, variable-length autoregressive model, CAD export, Gradio UI, distributed training, Hausdorff metric, svgpathtools fix.

### Phase 4: Production-Ready & Robustness (1–3 months, high priority now) - ~60% complete
- Comprehensive docstrings + type hints (target 80%+)
- Refactor long refinement/merging functions
- Full integration/end-to-end + regression tests
- Magic numbers → configs; structured logging
- Profile-guided optimization (refinement target <2 s/patch)
- ArchCAD-400K and FloorPlanCAD integration + benchmarking suite

### Phase 5: Advanced & Next-Gen (Ongoing, 3–9 months)
- Diffusion-transformer generative vectorization (text+image conditioning)
- Panoptic symbol spotting + vector merging
- Multimodal inputs (text prompts, style references)
- Explore VLM distillation (OmniSVG-inspired) for complex SVGs
- Community: HF model hub upload, public demo Spaces
- Regular security/dependency maintenance

#### Advanced Architecture Recommendations for Intelligent Reconstruction
To achieve intelligent "remaking" of scanned drawings (beyond tracing to idealized CAD with symmetries, constraints, and symbol recognition), prioritize hybrid architectures incorporating semantic understanding and generative capabilities. These align with the core objective of producing clean, editable vectors from degraded rasters.

- **Hybrid VLM + Diffusion Transformer**: Use VLM (e.g., LLaVA-Next) for semantic interpretation and description generation, then diffusion model (e.g., Hugging Face Diffusers) to generate idealized primitives. Adapt open-source like Vectra2D or Shap-E for 2D CAD reconstruction. Train/fine-tune on DeepPatent2 (2M+ patent drawings) and scanned datasets (e.g., Kaggle docs). Integration: Add to vectorization module with Hydra configs; benchmark on Hausdorff/vector IoU for 15-25% improvements.
- **Dual-Decoder Transformer with Panoptic Symbol Detection**: Sequence-to-sequence with dual decoders for CAD commands and params (e.g., Drawing2CAD). Add panoptic head for symbol spotting (gears, circuits). Use CAD-VGDrawing dataset. Refactor merging for constraints; test on patents/books.
- **AI Agent for CAD Recreation**: Prompt-to-CAD pipeline (e.g., MIT CAD Agent + OpenSCAD). Automate "human-like" redrawing via UI actions. Fine-tune on VideoCAD dataset. Add as optional mode for complex inputs.

Start with VLM-diffusion prototype (1-2 weeks); extend to multimodal for patents/books.

### Recommended Immediate Next Steps (Priority Order) - Current Status: ~20-30% complete

#### 1–2 weeks Quick Wins
- Extract all refinement/merging magic numbers to Hydra configs - Done (major ones extracted, config loading added)
- Add docstrings to 10–15 most complex functions - Done (added to render_optimization_hard, render_lines_with_type, postprocess, main (curves), merge_close, maximiz_final_iou)
- Extend type hints in refinement pipeline - Done (added to key functions in refinement and merging modules)
- Refactor long functions (render_optimization_hard) into classes/modules - Done (split into LineOptimizationState, BatchProcessor, OptimizationLoop classes)

#### 2–6 weeks Medium Term
- Refactor refinement long functions → classes/modules - Not done
- Build basic end-to-end integration test suite - Partial
- Add structured logging + exception hierarchy - Partial

#### 2–4 months Long Term
- Reach high type-hint coverage + strict mypy - Not done
- Full performance profiling + targeted optimizations - Partial
- Integrate ArchCAD-400K and CAD-VGDrawing + symbol spotting evaluation - Not done
- Explore diffusion-transformer prototype for generative mode - Not done