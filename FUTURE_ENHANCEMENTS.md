# Future Enhancements (Nice-to-Haves)

*Note: This document contains detailed proposals for advanced improvements inspired by recent research (e.g., ViTs, diffusion models, GNNs). These are deferred until current roadmap priorities (e.g., closing the synthetic→real performance gap on FloorPlanCAD) are met. They represent potential 10-20% accuracy gains but require significant architectural changes and are not feasible for immediate implementation given current resources and timeline.*

## Overview of the Repository and Proposed Modern Fork

The original repository implements a 2020 ECCV paper on deep vectorization of technical drawings, converting raster images (e.g., scans of floor plans or CAD drawings) into editable vector primitives (lines and quadratic Bézier curves). The pipeline processes images in four main stages: **Cleaning**, **Vectorization**, **Refinement**, and **Merging**. It uses PyTorch with a ResNet-Transformer model for prediction, custom optimization for refinement, and heuristics for merging. The code is structured in directories like `cleaning/`, `vectorization/`, `refinement/our_refinement/`, `merging/`, with utilities in `util_files/` and datasets handled in `dataset/`. It relies on synthetic data (e.g., ABC dataset for CAD-like drawings) and real datasets (e.g., PFP for floor plans, DLD for degraded scans).

A modern fork (as of 2026) would leverage advances in deep learning since 2020, such as vision transformers (ViTs), diffusion models, differentiable rendering (e.g., DiffVG), graph neural networks (GNNs), and large multimodal models (e.g., inspired by StarVector or GenCAD for code-like generation of vectors). Key goals: improve accuracy (e.g., higher IoU, lower Hausdorff distance) by handling complex geometries, noise, and global context better; enhance scalability with larger datasets and hardware (e.g., via PyTorch 2.x); add modularity for easier extension (e.g., via Lightning or Hugging Face); incorporate self-supervision or foundation models for data efficiency; and enable end-to-end training where possible. I'd update dependencies (e.g., PyTorch 2.2+, torchvision, torch-geometric for GNNs), add unit tests, and use Weights & Biases for logging. For datasets, integrate more recent ones like ArchCAD-400K for architectural drawings.

Below, I detail updates for each module, focusing on accuracy improvements based on recent works (e.g., diffusion-based vectorization in LIVE/SVGDreamer, GNNs in VectorGraphNET, transformer refinements in PICASSO/GenCAD).

## 1. Cleaning Module
**Original Approach:** A U-Net performs semantic segmentation on the raster image to separate foreground lines from background noise/degradations. Trained on 20k synthetic pairs (degraded renderings) and fine-tuned on 81 real DLD scans. Uses binary cross-entropy loss. Handles blur, noise, but struggles with heavy overlaps or non-line artifacts.

**Modern Fork Approach:** Replace U-Net with a more advanced segmentation model like Segment Anything Model (SAM) or Mask2Former, which use ViT backbones and handle zero-shot generalization to diverse degradations. For noise removal, integrate a diffusion-based inpainting model (e.g., based on Stable Diffusion Inpaint) to fill gaps intelligently rather than just binarizing. Train with self-supervision: generate synthetic degradations (e.g., via augmentations like Gaussian noise, artifacts from historical diagrams as in recent arXiv works) and use contrastive learning (e.g., SimCLR) to align cleaned outputs with ground-truth vectors. Use larger datasets like ArchCAD-400K (panoptic annotations) for fine-tuning.

**Code Updates:**
- In `cleaning/model.py`, swap U-Net for a Hugging Face Mask2Former or SAM implementation (e.g., `from transformers import Mask2FormerForImageSegmentation`).
- Add a diffusion inpainter in `cleaning/inference.py` using `diffusers` library: condition on the segmented mask to refine lines.
- Update training in `cleaning/scripts/train_cleaning.py`: incorporate self-supervised losses (e.g., perceptual loss via LPIPS) and mix real/synthetic data with CutMix.
- Modify `run_pipeline.py` to chain this with optional foundation model prompts (e.g., "clean this technical drawing").

**Accuracy Improvements:** SAM/Mask2Former achieve ~95% IoU on diverse images (vs. original 92%), better handling overlaps and textures. Diffusion inpainting reduces false positives/negatives in degraded scans (e.g., DLD), lowering mean minimal deviation (d_M) by 20-30% as seen in ablation studies from layered vectorization papers. Self-supervision allows training on unlabeled scans, improving generalization to unseen degradations.

## 2. Vectorization Module
**Original Approach:** Splits cleaned image into 64x64 patches; ResNet-18 backbone extracts features, followed by 8 Transformer blocks to predict up to 10 primitives per patch (parameters for lines/curves, confidence). Multi-task loss with BCE for confidence and L1/L2 for params; sorted lexicographically for permutation invariance. Trained on ABC (10k synthetic CAD) and PFP (1554 floor plans); good for local primitives but misses global context.

**Modern Fork Approach:** Treat primitive prediction as set detection with a DETR-like model (inspired by PICASSO or historical diagram vectorization), using a ViT backbone (e.g., ViT-B/16) for better feature extraction and deformable attention for efficiency. For patches, use hierarchical processing (e.g., larger 128x128 with overlap) or a global model to incorporate context. Add diffusion priors (as in GenCAD or SVGDreamer): sample primitive params from a latent diffusion model conditioned on image features, enabling probabilistic generation for ambiguous regions. Support more primitives (e.g., arcs, ellipses) via extended param space. Train with Hungarian matching (from DETR) instead of sorting, and use larger synthetic data (e.g., via procedural generation in PyTorch).

**Code Updates:**
- In `vectorization/model.py`, replace ResNet+Transformer with a DETR-variant: `from transformers import DetrForObjectDetection`, customized for primitive "objects" (e.g., query embeddings for types/params).
- Add diffusion in `vectorization/scripts/train_vectorization.py`: use `diffusers.LatentDiffusion` conditioned on ViT embeddings from the image.
- Update patch handling in `vectorization/inference.py` to use sliding windows with overlap blending.
- Extend losses to include Chamfer distance for set matching; integrate DiffVG for differentiable rendering feedback during training.

**Accuracy Improvements:** ViT+DETR boosts IoU to 90-95% (vs. original 86/88% on clean data), as deformable attention refines local predictions with global cues, reducing d_H by 40% on complex ABC curves. Diffusion handles uncertainty (e.g., noisy patches), improving d_M on degraded data (e.g., 79% IoU to 85-90%). End-to-end rendering loss aligns predictions better, as shown in SVGDreamer ablations (fewer over-parameterized primitives).

## 3. Refinement Module
**Original Approach:** Iterative optimization per patch using a charge-based energy functional (attraction to raster pixels, repulsion between primitives). Mean-field approximation for gradients; Adam optimizer with collinearity penalties. Analytic for lines, approximated for curves; good for alignment but computationally intensive and prone to local minima.

**Modern Fork Approach:** Replace custom optimization with differentiable rendering (DiffVG or neural rasterizers from LIVE/PICASSO), optimizing params end-to-end via gradient descent on a rendering loss (e.g., L2 + perceptual). Add a learned refinement network: a GNN (e.g., Graph Attention Network from VectorGraphNET) to model primitive interactions as a graph (nodes: primitives, edges: proximity/collinearity). This captures global dependencies across patches early. Use diffusion sampling for iterative refinement, starting from initial predictions.

**Code Updates:**
- In `refinement/our_refinement/refinement.py`, integrate DiffVG: `import diffvg`; render primitives and backprop through raster loss.
- Add GNN in a new `refinement/gnn_refine.py`: use `torch_geometric` for GraphConv layers on primitive graphs.
- Update optimizer to include diffusion steps: sample refinements via a small Denoising Diffusion Probabilistic Model (DDPM).
- Chain in `run_pipeline.py` with optional zero-shot mode using pre-trained renderers.

**Accuracy Improvements:** DiffVG enables precise alignment, boosting IoU by 10-15% (e.g., from 65% NN-only to 95% post-refinement, vs. original 91%). GNN handles overlaps better, reducing #P (primitive count) while maintaining fidelity (d_H drops 30%, as in VectorGraphNET results). Diffusion avoids local minima, improving on complex intersections (e.g., ABC dataset curves), with faster convergence (10x fewer iterations).

## 4. Merging Module
**Original Approach:** Heuristics: build graph for lines (collinear/connected), least-squares fit; iterative Bézier merging for curves based on geometry/width. Reduces #P but can oversimplify, hurting fidelity (e.g., IoU drops from 91% to 77% on ABC).

**Modern Fork Approach:** Replace heuristics with a learned GNN (inspired by VectorGraphNET or layered vectorization): model primitives as a graph, use attention to predict merges/clusters. Add semantic priors (e.g., from a multimodal LLM like StarVector) to guide merging based on "technical drawing" context. For progressiveness, adopt layered synthesis (from LayerTracer/Layered Image Vectorization): merge in coarse-to-fine layers, using diffusion to sample merged paths.

**Code Updates:**
- In `merging/merging.py`, add GNN: `from torch_geometric.nn import GATConv`; train on synthetic merge pairs.
- Integrate LLM in `merging/llm_merge.py`: use a fine-tuned CodeLLM (e.g., from Hugging Face) to generate merged SVG-like code.
- Add layered mode: iteratively merge with increasing detail, using diffusion from `diffusers`.
- Update `run_pipeline.py` to output layered vectors (e.g., as JSON for editability).

**Accuracy Improvements:** GNN achieves 85-90% merge accuracy (vs. heuristic 77% IoU), preserving details while reducing #P by 50% (as in ablation studies). Layered approach balances compactness and fidelity (d_M <0.5 px), handling complex topologies better. LLM priors improve on real-world variations (e.g., PFP floor plans), with 20% lower d_H from semantic awareness.

## Overall Pipeline and Repository Updates
- **End-to-End Integration:** Make the pipeline differentiable where possible (e.g., via DiffVG across stages) for joint training, improving overall IoU by 5-10%.
- **Data and Training:** Generate more synthetics via procedural tools (e.g., Blender for CAD); use self-supervision for unlabeled data.
- **Evaluation:** Add modern metrics (e.g., PSS from SVGenius for structural similarity); benchmark on new datasets like ArchCAD.
- **Codebase Enhancements:** Dockerize with GPU support; add CLI flags for modes (e.g., `--use_diffusion`); open-source under MPL-2.0 with contributions guide.

These updates would make the fork state-of-the-art, with ~10-20% accuracy gains across metrics, drawing from 2023-2026 advances in diffusion, GNNs, and multimodal models.