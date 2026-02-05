# Datasets for Deep Vectorization

This document compiles a comprehensive list of publicly available datasets suitable for training, evaluating, or benchmarking deep vectorization models, with a primary focus on technical drawings (e.g., CAD, floorplans, mechanical schematics, patents). Vectorization tasks typically involve converting raster inputs (e.g., PNG scans) to vector outputs (e.g., SVG primitives, parametric sequences, graphs) or related processes like primitive extraction, symbol spotting, and generative modeling. Datasets are categorized into **real-world** (scanned or industry-sourced, often noisy) and **synthetic/processed** (generated, cleaner for augmentation).

Inclusion criteria prioritize datasets with:
- Raster-vector pairs or derivable alignments.
- Relevance to technical/architectural/mechanical/patent/sketches.
- Recent updates or releases (up to early 2026).
- Public accessibility (e.g., Hugging Face, GitHub, arXiv-linked).

For each dataset, details include size, formats, key features, access links, license, and suitability notes. Preprocessing tips are provided where relevant (e.g., rasterizing vectors via Cairo or OpenCV). Licenses must be verified for commercial use.

## Table of Contents
- [Introduction](#datasets-for-deep-vectorization)
- [Evaluation Status](#evaluation-status)
- [Suitable Datasets](#suitable-datasets)
  - [Real-World Datasets](#real-world-datasets)
  - [Synthetic/Processed Datasets](#syntheticprocessed-datasets)
- [Unsuitable Datasets](#unsuitable-datasets)
  - [Real-World Datasets](#real-world-datasets-1)
  - [Synthetic/Processed Datasets](#syntheticprocessed-datasets-1)
- [Internal Dataset Pipeline](#internal-dataset-pipeline)

## Evaluation Status
All datasets listed in this document have been thoroughly evaluated for suitability in DeepV (Deep Vectorization) as of February 5, 2026. Evaluations focused on the presence of vector geometric primitives (e.g., lines, arcs, curves in formats like SVG, DXF, or parametric sequences) needed for technical drawing vectorization. Unsuitable datasets are marked with "(UNSUITABLE for DeepV)" followed by detailed reasoning. Suitable datasets include highly suitable ones (e.g., FPLAN-POLY, RPLAN) with direct vector primitives, and secondary ones (e.g., QuickDraw, SketchGraphs) with vector data for related tasks.

## Suitable Datasets

These datasets contain vector geometric primitives or derivable vector data suitable for DeepV tasks.

### Real-World Datasets

These datasets often include real-world variations like noise, distortions, or multi-view elements, ideal for robust model training.

- **FloorPlanCAD**  
  Size: 15,663 CAD drawings (expanded from ~10k).  
  Formats: SVG vectors with PNG rasterizations; COCO annotations. Parquet (auto-converted).  
  Features: 30+ categories (walls, symbols); panoptic spotting; residential/commercial/hospitals; 3D derivable. Supports object detection, instance/semantic segmentation. Privacy-protected (cropped, text removed).  
  Access: [Hugging Face](https://huggingface.co/datasets/Voxel51/FloorPlanCAD); project site [floorplancad.github.io](https://floorplancad.github.io/).  
  License: CC BY-NC 4.0.  
  Suitability: CAD vectorization benchmark; raster derivable. Preprocessing: Patch large drawings. Last updated: November 2025 (FiftyOne dataset).

- **ArchCAD-400K**  
  Size: 413,062 chunks from 5,538 drawings (~26x larger than FloorPlanCAD); 40k samples on HF.  
  Formats: SVG vectors (labeled/instances/RGB), raster aligned, JSON, Q&A, point clouds.  
  Features: Multimodal; 30+ categories; diverse buildings/scales; panoptic (semantic/instance); vectorized annotation workflow. Each sample corresponds to a 14m × 14m area.  
  Access: [Hugging Face](https://huggingface.co/datasets/jackluoluo/ArchCAD) (requires approval for non-commercial use); [GitHub](https://github.com/ArchiAI-LAB/ArchCAD); paper [arXiv:2503.22346](https://arxiv.org/abs/2503.22346).  
  License: CC BY-NC 4.0.  
  Suitability: Large-scale end-to-end vectorization; precise primitives. Preprocessing: Auto + human correction.

- **ABC**  
  Size: 1M CAD models (~10k with vector projections/drawings).  
  Formats: Parametric curves/surfaces (Step, Parasolid, Features YAML), OBJ meshes, PNG images; vector boundaries derivable.  
  Features: Explicitly parametrized curves and surfaces; ground truth for differential quantities, patch segmentation, geometric features. For geometric deep learning.  
  Access: [GitHub](https://github.com/deep-geometry/abc-dataset); [Website](https://deep-geometry.github.io/abc-dataset/); paper [arXiv:1812.06216](https://arxiv.org/pdf/1812.06216.pdf).  
  License: Research (Onshape Terms for data).  
  Suitability: Mechanical curves/lines; parametric CAD primitives. Preprocessing: Extract 2D projections from boundaries.

- **ResPlan**  
  Size: 17,000 residential floorplans.  
  Formats: JSON (vectors/geometries/semantics/graphs), PNG previews; NetworkX graphs; PKL (pickled data).  
  Features: Elements (walls, doors, windows, balconies) and spaces (rooms); connectivity graphs; realistic layouts. Unit-level; 3D convertible.  
  Access: [GitHub](https://github.com/m-agour/ResPlan) (ResPlan.zip); [arXiv (2508.14006)](https://arxiv.org/abs/2508.14006). Open-source pipeline for geometry cleaning/alignment.  
  License: MIT.  
  Suitability: Architectural vector-graph tasks; generative. Preprocessing: Python cleaning/alignment.

- **MSD (Modified Swiss Dwellings)**  
  Size: 5,372 floor plans (17.4 GB).  
  Formats: Raster images, vector geometries (CSV dataframe), graphs; derived from Swiss Dwellings database.  
  Features: Multi-floor residential complexes; access graphs; train/test split by buildings. Includes cleaned geometries (rooms, walls, structural elements). Designed for floor plan auto-completion tasks (boundary + constraints → full plan).  
  Access: [Kaggle](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings); [4TU.ResearchData](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9/1); original [Swiss Dwellings](https://zenodo.org/record/7788422).  
  License: CC BY-SA 4.0.  
  Suitability: Complex multi-unit vector-graph. Preprocessing: Extract geometries from CSV for vector primitives.

- **SketchGraphs**  
  Size: 15M CAD sketches (from real-world CAD models).  
  Formats: Constraint graphs (JSON/serialized); nodes (geometric primitives like lines/arcs/circles), edges (constraints like parallel/perpendicular); construction sequences (custom binary).  
  Features: Large-scale dataset for modeling relational geometry in CAD; sketches as graphs with primitives and constraints; supports generative modeling and autoconstrain tasks. Extracted from Onshape platform.  
  Access: [GitHub](https://github.com/PrincetonLIPS/SketchGraphs); data downloads [here](https://sketchgraphs.cs.princeton.edu/); paper [arXiv:2007.08506](https://arxiv.org/abs/2007.08506).  
  License: MIT (code); research (data per Onshape Terms).  
  Suitability: Relational vector tasks; pair with rasters for raster-to-vector pipelines. Preprocessing: Render graphs to rasters for input.

- **CubiCasa5K**  
  Size: 5,000 scanned floorplans.  
  Formats: High-res raster (up to 6k px), SVG annotations (polygon vectors); LMDB database (~105 GB).  
  Features: 80+ semantic labels (rooms, furniture, walls); Finnish real estate CAD-sourced. Dense polygon annotations for object separation.  
  Access: [GitHub](https://github.com/CubiCasa/CubiCasa5k); [Zenodo](https://zenodo.org/record/2613548) (5.5 GB zip).  
  License: CC BY-NC-SA 4.0.  
  Suitability: Noisy raster-to-vector. Preprocessing: Align pairs; parse SVG for vector primitives.

- **DeepPatent2**  
  Size: >2.7M technical drawings (2M full patents, 2.8M segmented figures).  
  Formats: PNG (original/segmented), JSON (metadata/semantics), CSV (distributions). Organized by year (2007-2020).  
  Features: Patent drawings with object names (132k unique), viewpoints (22k), captions, bounding boxes; semantic extraction via NN (e.g., "object": "chair"). Compound drawings segmented.  
  Access: [OneDrive (2020 data - Original_2020.tar.gz)](https://olddominion-my.sharepoint.com/personal/j1wu_odu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020%2FOriginal%5F2020%2Etar%2Egz&viewid=7828cbdf%2D98fd%2D45c8%2D9fbf%2D337e03d13638&parent=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020) or [OSF (2007 subset)](https://osf.io/kv4xa).  
  License: CC BY-NC 2.0.  
  Suitability: Technical/patent vectorization; coords for primitive extraction. Preprocessing: Use JSON for spatial augmentation.

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

- **FPLAN-POLY**  
  Size: 42 floorplans + 38 symbol models.  
  Formats: DXF vector files (POLYLINE primitives).  
  Features: Real floorplans converted from raster to vector; symbol spotting ground truth.  
  Access: [Archived CVC site download](https://web.archive.org/web/20130621114030/http://www.cvc.uab.es/~marcal/FPLAN-POLY/img/FPLAN-POLY.zip).  
  License: Research.  
  Suitability: **HIGHLY SUITABLE** - Contains vector geometric primitives (polylines) for floorplan analysis. Perfect for vectorization tasks.

- **RPLAN**  
  Size: 80k+ floorplans.  
  Formats: Raster, graphs (vector derivable).  
  Features: Room layouts; generative DL.  
  Access: [Site](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html); request via form.  
  License: Free.  
  Suitability: **HIGHLY SUITABLE** - Contains vector floorplan data with boundaries, room boxes, doors, windows as geometric primitives. Graph-to-vector conversion possible. Requires access request.

- **QuickDraw**  
  Size: 50M+ vector sketches (345 classes).  
  Formats: Stroke-3 (vector), raster.  
  Features: Human-drawn; 100k+/class.  
  Access: [Google](https://quickdraw.withgoogle.com/data).  
  License: CC BY 4.0.  
  Suitability: **Secondary dataset** - Contains genuine vector stroke sequences ((x,y) point coordinates), suitable for training stroke vectorization models, though focused on everyday objects rather than technical drawings.

## Unsuitable Datasets

These datasets lack vector geometric primitives or are otherwise unsuitable for DeepV technical drawing vectorization.

### Real-World Datasets

- **BlendedNet** (UNSUITABLE for DeepV)  
  Size: 999 blended wing body geometries (~9 conditions each; 8,830 cases).  
  Formats: CSV metadata, HDF5 point clouds with aerodynamic coefficients, VTK meshes.  
  Features: CFD surface meshes with pressure/skin friction coefficients; aerodynamic predictions.  
  Access: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VJT9EP); [GitHub](https://github.com/nicksungg/clarc_blended_wing_body).  
  License: Apache 2.0 variant.  
  Suitability: CFD aerodynamic prediction, not vectorization - contains point cloud meshes with scalar fields rather than vector geometric primitives.

- **CAD/BIM Collection** (UNSUITABLE for DeepV)  
  Size: 4,596 IFC, 6,471 RVT, 156,024 DWG.  
  Formats: Vector CAD/BIM files.  
  Features: Automated from architectural sites.  
  Access: Contact authors (site unavailable).  
  License: Varies (public).  
  Suitability: **NOT SUITABLE** - Dataset not publicly accessible (site unavailable, requires contacting authors), and contains 3D BIM models rather than 2D vector geometric primitives needed for DeepV technical drawing vectorization.

- **CADSketchNet** (UNSUITABLE for DeepV)  
  Size: 1k+ annotated sketches.  
  Formats: Sketches paired with 3D CAD.  
  Features: 3D retrieval.  
  Access: Contact authors (Computers & Graphics paper); no public repo.  
  License: Research.  
  Suitability: **NOT SUITABLE** - Dataset not publicly accessible (requires contacting authors), and designed for 3D CAD model retrieval from 2D sketches rather than vectorizing technical drawings into geometric primitives.

- **CFP (Comprehensive Floor Plan)** (UNSUITABLE for DeepV)  
  Size: 100k+ elements (high-res images).  
  Formats: Raster with vector points (quartets for shapes).  
  Features: Diverse plans; sparse boundaries.  
  Access: Contact authors (no public repo).  
  License: Research.  
  Suitability: **NOT SUITABLE** - Dataset not publicly accessible (requires contacting authors, no public repository), though it contains vector points for floor plan shapes which could be suitable for vectorization if available.

- **CVC-FP** (UNSUITABLE for DeepV)  
  Size: 122 floorplans (4 subsets).  
  Formats: Raster with masks; vector derivable.  
  Features: Semantic segmentation; architectural.  
  Access: [CVC-FP.zip](http://dag.cvc.uab.es/DATASETS/CVC-FP.zip); evaluation scripts [Evaluation.zip](http://dag.cvc.uab.es/DATASETS/Evaluation.zip).  
  License: Research.  
  Suitability: **NOT SUITABLE** - Contains semantic segmentation polygons, not vector primitives. Use for evaluation only after conversion to parametric format.

- **DeepPatent** (UNSUITABLE for DeepV)  
  Size: >350k patent drawings.  
  Formats: PNG (from TIF), XML metadata.  
  Features: Design patents (2018-2019); multi-view.  
  Access: [GitHub](https://github.com/GoFigure-LANL/DeepPatent-dataset); Google Drive (compressed parts).  
  License: BSD-3-Clause.  
  Suitability: **NOT SUITABLE** - Contains patent drawings as raster images for image retrieval and recognition, not vector geometric primitives needed for DeepV technical drawing vectorization.

- **DLD (Degraded Line Drawings)** (UNSUITABLE for DeepV)  
  Size: 81 photos/scans of floorplans.  
  Formats: Raster (raw/cleaned), vector targets.  
  Features: Real noise/text; inpainted lines. Resolution ~1300x1000.  
  Access: Contact authors (ECCV 2020 paper "Deep Vectorization of Technical Drawings"); no public repo.  
  License: Research.  
  Suitability: **NOT SUITABLE** - Dataset not publicly accessible (requires contacting authors), though it contains raster images with vector targets for degraded line drawing vectorization.

- **DrivAerNet++** (UNSUITABLE for DeepV)  
  Size: 8,150+ car meshes/simulations.  
  Formats: 3D surface meshes (VTK), point clouds with aerodynamic coefficients, CFD volumetric fields.  
  Features: High-fidelity CFD simulations with pressure/velocity fields; multimodal automotive aerodynamics.  
  Access: [GitHub](https://github.com/Mohamedelrefaie/DrivAerNet).  
  License: CC BY-NC 4.0.  
  Suitability: CFD aerodynamic prediction, not vectorization - contains 3D surface meshes and scalar fields rather than vector geometric primitives.

- **Engineering Symbols** (UNSUITABLE for DeepV)  
  Size: 2,432 instances (multi-class imbalanced).  
  Formats: 100x100 pixel binary images (CSV flattened).  
  Features: Engineering symbols from P&ID diagrams for CNN classification.  
  Access: [GitHub](https://github.com/heyad/Eng_Diagrams).  
  License: Research.  
  Suitability: Symbol spotting/classification, not vectorization - contains raster images of symbols rather than vector geometric primitives.

- **HoliCity** (UNSUITABLE for DeepV)  
  Size: City-scale 3D models (6,300 panoramas).  
  Formats: CAD (DXF/IFC/OBJ/STL), panoramas, depth/normal maps, plane parameters, semantic segmentation.  
  Features: Urban buildings/parcels/roads; 3D holistic structures (planes, surfaces, wireframes).  
  Access: [Project site](https://holicity.io/); sample data on GitHub.  
  License: Commercial/research.  
  Suitability: **NOT SUITABLE** - Contains 3D city CAD models and derived 3D data (depth maps, normals, planes) for holistic 3D structure learning, not 2D vector geometric primitives needed for DeepV technical drawing vectorization.

- **IMPACT** (UNSUITABLE for DeepV)  
  Size: 500k patents, 3.61M figures (2007–2022).  
  Formats: Multimodal (figures, captions, metadata); CSV files.  
  Features: Viewpoint-coherent captions; retrieval/classification.  
  Access: [GitHub](https://github.com/AI4Patents/IMPACT); [Hugging Face](https://huggingface.co/datasets/AI4Patents/IMPACT).  
  License: Open.  
  Suitability: **NOT SUITABLE** - Contains patent figures as images with captions and metadata for multimodal retrieval and classification, not vector geometric primitives needed for DeepV technical drawing vectorization.

- **PatentDesc-355K** (UNSUITABLE for DeepV)  
  Size: ~355k figures from 60k+ documents.  
  Formats: Images, captions, metadata.  
  Features: Multimodal descriptions; sparse illustrations.  
  Access: Code and data publicly available (arXiv 2501.15074).  
  License: Research.  
  Suitability: **NOT SUITABLE** - Contains patent figures as images with textual descriptions for multimodal description generation, not vector geometric primitives needed for DeepV technical drawing vectorization.

- **PDTW150K** (UNSUITABLE for DeepV)  
  Size: 150k+ patents, 850k+ drawings.  
  Formats: Images, text metadata, bounding boxes.  
  Features: Multi-view figures; object/viewpoint info.  
  Access: [GitHub](https://github.com/ncyuMARSLab/PDTW150K); Google Drive (compressed parts).  
  License: Open Government Data License v1.0.  
  Suitability: **NOT SUITABLE** - Contains patent drawings as images with bounding boxes for views, not vector geometric primitives needed for DeepV technical drawing vectorization.

- **VideoCAD** (UNSUITABLE for DeepV)  
  Size: 41k+ CAD UI videos.  
  Formats: Videos, 3D models, UI traces.  
  Features: Long-horizon UI for generation.  
  Access: [GitHub](https://github.com/ghadinehme/VideoCAD); [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WX8PCK).  
  License: Open.  
  Suitability: **NOT SUITABLE** - Contains videos of CAD UI interactions and 3D CAD models for behavior cloning and CAD generation, not 2D raster images of technical drawings for vectorization.

### Synthetic/Processed Datasets

- **IAM Handwriting** (UNSUITABLE for DeepV)  
  Size: 115k word images (offline), 13k lines (online; 70k words).  
  Formats: Raster (offline), point coordinates (online).  
  Features: Sequence recognition.  
  Access: [Site](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database); register for download.  
  License: Research.  
  Suitability: **NOT SUITABLE** - Contains handwriting strokes for text recognition, not geometric primitives needed for DeepV.

- **SESYD** (UNSUITABLE for DeepV)  
  Size: 1,000 floorplans (10 classes, 100 each).  
  Formats: Synthetic documents with symbols.  
  Features: Furniture layouts/variations; symbol spotting.  
  Access: [Original site](http://mathieu.delalandre.free.fr/projects/sesyd/) (currently unavailable).  
  License: Free.  
  Suitability: **NOT SUITABLE** - Contains synthetic documents for symbol spotting evaluation, not vector geometric primitives needed for DeepV. Symbols are likely raster instances rather than parametric vector elements.

- **TU-Berlin** (UNSUITABLE for DeepV)  
  Size: 20k sketches (250 categories, 80 each).  
  Formats: Raster/vector.  
  Features: Freehand sketches.  
  Access: [Hugging Face](https://huggingface.co/datasets/kmewhort/tu-berlin-png); or university archives.  
  License: Research.  
  Suitability: **NOT SUITABLE** - Contains raster PNG images of object sketches for classification, not vector primitives needed for DeepV.

## Internal Dataset Pipeline

This section documents the project's internal pipeline for downloading and preprocessing datasets used in Deep Vectorization.

### Downloaders
Dataset downloaders are located in `dataset/downloaders/`. These scripts handle fetching datasets from their original sources (e.g., Hugging Face, Google Drive, GitHub releases).

- `download_dataset.py`: General downloader utility for various datasets.

### Processors
Preprocessing scripts to convert raw datasets into usable formats (SVG vectors or raster images) are in `dataset/processors/`. Each processor is typically dataset-specific.

- `base.py`: Base processor class with common functionality.
- `deeppatent2.py`: Processor for DeepPatent2 dataset.
- `floorplancad.py`: Processor for FloorPlanCAD dataset.

### Data Storage Structure
All processed data is stored in the `/data/` directory with the following structure:
- `/data/raw/dataset_name/`: Raw downloaded data.
- `/data/vector/dataset_name/`: Vector format data (e.g., SVG).
- `/data/raster/dataset_name/`: Raster format data (e.g., PNG).

This organization ensures clean separation of data stages and facilitates pipeline reproducibility.