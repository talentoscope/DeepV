# Datasets for DeepV

This document compiles a comprehensive list of publicly available datasets suitable for training, evaluating, or benchmarking DeepV vectorization models, with a primary focus on technical drawings (e.g., CAD, floorplans, mechanical schematics, patents). Vectorization tasks typically involve converting raster inputs (e.g., PNG scans) to vector outputs (e.g., SVG primitives, parametric sequences, graphs) or related processes like primitive extraction, symbol spotting, and generative modeling. Datasets are categorized into **real-world** (scanned or industry-sourced, often noisy) and **synthetic/processed** (generated, cleaner for augmentat...

## DeepV Data Processing Philosophy

**Clean Vector Ground Truth + Augmented Dirty Training Data**

DeepV follows a strict preprocessing pipeline designed to create optimal training data for vectorization:

1. **Extract Only Vector Primitives**: Remove all text, symbols, annotations, and non-geometric elements from source datasets
2. **Standardize Line Properties**: Convert all lines to 0.1px thickness (ultra-thin) for consistent geometric representation
3. **Generate Clean Vector Ground Truth**: Store pristine SVG/DXF files with only geometric primitives
4. **Create Augmented Dirty Training Images**: Apply realistic degradation (noise, blur, compression artifacts, paper texture, scanning distortions) to simulate real-world scanned documents
5. **Maintain Raster-Vector Alignment**: Ensure perfect correspondence between degraded training images and clean vector targets

This approach ensures models learn to recover clean geometry from noisy inputs, mimicking real-world vectorization of degraded technical drawings.

Inclusion criteria prioritize datasets with:
- Raster-vector pairs or derivable alignments.
- Relevance to technical/architectural/mechanical/patent/sketches.
- Recent updates or releases (up to early 2026).
- Public accessibility (e.g., Hugging Face, GitHub, arXiv-linked).

For each dataset, details include size, formats, key features, access links, license, and suitability notes. Preprocessing tips are provided where relevant (e.g., rasterizing vectors via Cairo or OpenCV). Licenses must be verified for commercial use.

## Table of Contents
- [Introduction](#datasets-for-deep-vectorization)
- [DeepV Data Processing Philosophy](#deepv-data-processing-philosophy)
- [Evaluation Status](#evaluation-status)
- [Suitable Datasets](#suitable-datasets)
  - [Real-World Datasets](#real-world-datasets)
  - [Synthetic/Processed Datasets](#syntheticprocessed-datasets)
- [Unsuitable Datasets](#unsuitable-datasets)
  - [Real-World Datasets](#real-world-datasets-1)
  - [Synthetic/Processed Datasets](#syntheticprocessed-datasets-1)
- [Internal Dataset Pipeline](#internal-dataset-pipeline)

## Evaluation Status
All datasets listed in this document have been thoroughly evaluated for suitability in **DeepV** as of February 5, 2026. Evaluations focused on the presence of vector geometric primitives (e.g., lines, arcs, curves in formats like SVG, DXF, or parametric sequences) needed for technical drawing vectorization. Unsuitable datasets are marked with "(UNSUITABLE for DeepV)" followed by detailed reasoning. Suitable datasets include highly suitable ones (e.g., FPLAN-POLY) with direct vector primitives, and secondary ones (e.g., QuickDraw, SketchGraphs) with vector data for related tasks.

## Suitable Datasets

These datasets contain vector geometric primitives or derivable vector data suitable for DeepV tasks.

### Real-World Datasets

These datasets often include real-world variations like noise, distortions, or multi-view elements, ideal for robust model training.

- **FloorPlanCAD**  
  Size: 15,663 CAD drawings (expanded from ~10k).  
  Formats: Original dataset: SVG vectors with PNG rasterizations and COCO annotations. Hugging Face version: PNG images with FiftyOne/COCO-style annotations (bounding boxes, segmentation masks). Parquet (auto-converted).  
  Features: 30+ categories (walls, symbols); panoptic spotting; residential/commercial/hospitals; 3D derivable. Supports object detection, instance/semantic segmentation. Privacy-protected (cropped, text removed).  
  Access: [Hugging Face](https://huggingface.co/datasets/Voxel51/FloorPlanCAD) (FiftyOne format, PNG + annotations); project site [floorplancad.github.io](https://floorplancad.github.io/) (original SVGs via Google Drive).  
  License: CC BY-NC 4.0.  
  Suitability: CAD vectorization benchmark; raster derivable from annotations. Preprocessing: Extract geometric primitives only (walls, structural elements); standardize to 0.1px lines; generate augmented dirty training images with realistic scanning artifacts. Last updated: November 2025 (FiftyOne dataset).  
  **Status**: ✅ Fully processed for DeepV vectorization training. Cleaned dataset available: 14,625 SVG vectors (black-on-white, text removed, 0.1px lines) in `data/vector/floorplancad/` and corresponding augmented dirty PNG rasters in `data/raster/floorplancad/`. Raw data in `data/raw/floorplancad/`.

- **ResPlan**  
  Size: 17,000 residential floorplans.  
  Formats: JSON (vectors/geometries/semantics/graphs), PNG previews; NetworkX graphs; PKL (pickled data).  
  Features: Elements (walls, doors, windows, balconies) and spaces (rooms); connectivity graphs; realistic layouts. Unit-level; 3D convertible.  
  Access: [GitHub](https://github.com/m-agour/ResPlan) (ResPlan.zip); [arXiv (2508.14006)](https://arxiv.org/abs/2508.14006). Open-source pipeline for geometry cleaning/alignment.  
  License: MIT.  
  Suitability: Architectural vector-graph tasks; generative. Preprocessing: Extract geometric primitives (walls, structural elements); remove semantic labels; standardize to 0.1px lines; generate augmented dirty training images.  
  **Status**: ✅ Implemented in DeepV pipeline (17,107 SVG files generated).

- **MSD (Modified Swiss Dwellings)**  
  Size: 5,372 floor plans (17.4 GB).  
  Formats: Raster images (512x512x3 .npy), vector geometries (Shapely polygons in .pickle graphs), CSV dataframe; derived from Swiss Dwellings database.  
  Features: Multi-floor residential complexes; access graphs; train/test split by buildings. Includes cleaned geometries (rooms, walls, structural elements). Designed for floor plan auto-completion tasks (boundary + constraints → full plan).  
  Access: [Kaggle](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings); [4TU.ResearchData](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9/1); original [Swiss Dwellings](https://zenodo.org/record/7788422).  
  License: CC BY-SA 4.0.  
  Suitability: Complex multi-unit vector-graph. Preprocessing: Extract geometries from CSV for vector primitives.  
  **Status**: Processor available in DeepV pipeline.

- **SketchGraphs**  
  Size: 15M CAD sketches (from real-world CAD models).  
  Formats: Constraint graphs (JSON/serialized); nodes (geometric primitives like lines/arcs/circles), edges (constraints like parallel/perpendicular); construction sequences (custom binary).  
  Features: Large-scale dataset for modeling relational geometry in CAD; sketches as graphs with primitives and constraints; supports generative modeling and autoconstrain tasks. Extracted from Onshape platform.  
  Access: [GitHub](https://github.com/PrincetonLIPS/SketchGraphs); data downloads [here](https://sketchgraphs.cs.princeton.edu/); paper [arXiv:2007.08506](https://arxiv.org/abs/2007.08506).  
  License: MIT (code); research (data per Onshape Terms).  
  Suitability: Relational vector tasks; pair with rasters for raster-to-vector pipelines. Preprocessing: Render graphs to rasters for input.
  **Status**: ✅ Implemented in DeepV pipeline (processor converts constraint graphs to SVG).

- **CubiCasa5K**  
  Size: 5,000 scanned floorplans.  
  Formats: High-res raster (up to 6k px), SVG annotations (polygon vectors); LMDB database (~105 GB).  
  Features: 80+ semantic labels (rooms, furniture, walls); Finnish real estate CAD-sourced. Dense polygon annotations for object separation.  
  Access: [GitHub](https://github.com/CubiCasa/CubiCasa5k); [Zenodo](https://zenodo.org/record/2613548) (5.5 GB zip).  
  License: CC BY-NC-SA 4.0.  
  Suitability: Noisy raster-to-vector. Preprocessing: Align pairs; parse SVG for vector primitives.  
  **Status**: ✅ Implemented in DeepV pipeline (992 PNG + 992 SVG files processed).

### Synthetic/Processed Datasets

These provide controlled data for initial training or augmentation.

- **CAD-VGDrawing (Drawing2CAD)**  
  Size: ~157k–161k SVG-to-CAD pairs (from CAD models; 4 views: Front, Top, Right, Isometric).  
  Formats: SVG vectors, raster PNG (derived), JSON/sequences for parametric CAD commands, .npy, .h5.  
  Features: Aligns vector primitives (lines, arcs, curves) with editable CAD operations; preserves geometry and design intent; path reordering/normalization. Original CAD models sourced from the rundiwu/DeepCAD project; dataset conversion and packaging in lllssc/Drawing2CAD repository. Conversion/export to viewable SVGs via FreeCAD. Split: 90% train, 5% val/test.  
  Access: [Google Drive](https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU?usp=sharing).  
  Repositories: [lllssc/Drawing2CAD](https://github.com/lllssc/Drawing2CAD) (dataset + conversion scripts), [rundiwu/DeepCAD](https://github.com/rundiwu/DeepCAD) (original CAD models).  
  License: MIT.  
  Suitability: **HIGHLY SUITABLE** - Vector-to-parametric tasks; rasterize SVGs for full raster-to-vector pipelines. Preprocessing: Limit sequences to 100 primitives. Last updated: December 2025.  
  **Status**: ✅ Implemented in DeepV pipeline (processor copies SVG vectors with flattened filenames).

- **FPLAN-POLY**  
  Size: 42 floorplans + 38 symbol models.  
  Formats: DXF vector files (POLYLINE primitives).  
  Features: Real floorplans converted from raster to vector; symbol spotting ground truth.  
  Access: [Archived CVC site download](https://web.archive.org/web/20130621114030/http://www.cvc.uab.es/~marcal/FPLAN-POLY/img/FPLAN-POLY.zip).  
  License: Research.  
  Suitability: **HIGHLY SUITABLE** - Contains vector geometric primitives (polylines) for floorplan analysis. Perfect for vectorization tasks.  
  **Status**: ✅ Implemented in DeepV pipeline (processor copies 80 DXF vector files).

- **QuickDraw**  
  Size: 50M+ vector sketches (345 classes).  
  Formats: Stroke-3 (vector), raster.  
  Features: Human-drawn; 100k+/class.  
  Access: [Google](https://quickdraw.withgoogle.com/data).  
  License: CC BY 4.0.  
  Suitability: **Secondary dataset** - Contains genuine vector stroke sequences ((x,y) point coordinates), suitable for training stroke vectorization models, though focused on everyday objects rather than technical drawings.
  **Status**: ✅ Implemented in DeepV pipeline (processor converts stroke sequences to SVG). Raw data downloaded: 10 classes (~670MB NDJSON files) in `data/raw/quickdraw/`.

## Internal Dataset Pipeline

This section documents the project's internal pipeline for downloading and preprocessing datasets used in **DeepV**.

### Downloaders
Dataset downloaders are located in `dataset/downloaders/`. These scripts handle fetching datasets from their original sources (e.g., Hugging Face, Google Drive, GitHub releases).

- `download_dataset.py`: General downloader utility for various datasets.

### Processors
Preprocessing scripts to convert raw datasets into usable formats (SVG vectors or raster images) are in `dataset/processors/`. Each processor is typically dataset-specific and follows the DeepV philosophy of extracting clean geometric primitives and generating augmented dirty training images.

**Processing Pipeline:**
1. Extract geometric primitives only (lines, arcs, curves) - remove text, symbols, annotations
2. Standardize all lines to 0.1px thickness for consistent representation
3. Generate clean vector ground truth (SVG/DXF)
4. Apply realistic augmentations to create dirty training images (noise, blur, compression, paper texture, scanning artifacts)

- `base.py`: Base processor class with common functionality.
- `cadvgdrawing.py`: Processor for CAD-VGDrawing dataset.
- `cubicasa.py`: Processor for CubiCasa5K dataset.
- `floorplancad.py`: Processor for FloorPlanCAD dataset.
- `fplanpoly.py`: Processor for FPLAN-POLY dataset.
- `msd.py`: Processor for Modified Swiss Dwellings (MSD) dataset.
- `quickdraw.py`: Processor for QuickDraw dataset.
- `resplan.py`: Processor for ResPlan dataset.
- `sketchgraphs.py`: Processor for SketchGraphs dataset.

### Data Storage Structure
All processed data is stored in the `/data/` directory with the following structure:
- `/data/raw/dataset_name/`: Raw downloaded data.
- `/data/vector/dataset_name/`: Clean vector ground truth (SVG/DXF) with only geometric primitives (0.1px lines).
- `/data/raster/dataset_name/`: Augmented dirty training images (PNG) with realistic degradation artifacts.

This organization ensures clean separation between pristine vector targets and noisy training inputs, facilitating robust vectorization model training.

## Unsuitable Datasets

These datasets do not contain vector geometric primitives or derivable vector data suitable for DeepV vectorization tasks.

### Real-World Datasets

- **IMPACT (AI4Patents)**  
  Size: 500,000+ design patents with 3.61 million figures.  
  Formats: Patent images (PNG/JPG figures), CSV metadata files, text captions.  
  Features: Design patent figures with generated captions; multimodal patent analysis dataset.  
  Access: [Hugging Face](https://huggingface.co/datasets/AI4Patents/IMPACT); [GitHub](https://github.com/AI4Patents/IMPACT).  
  License: CC BY-SA 4.0.  
  Suitability: **UNSUITABLE for DeepV** - Contains design patent images and captions, but no vector geometric primitives. Focuses on product design patents (ornamental designs) rather than technical drawings requiring vectorization. Better suited for multimodal patent analysis tasks.  
  **Status**: ❌ Evaluated February 7, 2026 - Not suitable for vectorization training.