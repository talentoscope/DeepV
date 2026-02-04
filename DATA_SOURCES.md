# Datasets for Deep Vectorization

This document compiles a comprehensive list of publicly available datasets suitable for training, evaluating, or benchmarking deep vectorization models, with a primary focus on technical drawings (e.g., CAD, floorplans, mechanical schematics, patents). Vectorization tasks typically involve converting raster inputs (e.g., PNG scans) to vector outputs (e.g., SVG primitives, parametric sequences, graphs) or related processes like primitive extraction, symbol spotting, and generative modeling. Datasets are categorized into **real-world** (scanned or industry-sourced, often noisy) and **synthetic/processed** (generated, cleaner for augmentation). 

Inclusion criteria prioritize datasets with:
- Raster-vector pairs or derivable alignments.
- Relevance to technical/architectural/mechanical/patent/sketches.
- Recent updates or releases (up to early 2026).
- Public accessibility (e.g., Hugging Face, GitHub, arXiv-linked).

For each dataset, details include size, formats, key features, access links, license, and suitability notes. Preprocessing tips are provided where relevant (e.g., rasterizing vectors via Cairo or OpenCV). Licenses must be verified for commercial use.

## Real-World Datasets

These datasets often include real-world variations like noise, distortions, or multi-view elements, ideal for robust model training.

- **CAD-VGDrawing (Drawing2CAD)**  
  Size: ~157k–161k SVG-to-CAD pairs (from CAD models; 4 views: Front, Top, Right, Isometric).  
  Formats: SVG vectors, raster PNG (derived), JSON/sequences for parametric CAD commands, .npy, .h5.  
  Features: Aligns vector primitives (lines, arcs, curves) with editable CAD operations; preserves geometry and design intent; path reordering/normalization. Original CAD models sourced from the rundiwu/DeepCAD project; dataset conversion and packaging in lllssc/Drawing2CAD repository. Conversion/export to viewable SVGs via FreeCAD. Split: 90% train, 5% val/test.  
  Access: [Google Drive](https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU?usp=sharing).  
  Repositories: [lllssc/Drawing2CAD](https://github.com/lllssc/Drawing2CAD) (dataset + conversion scripts), [rundiwu/DeepCAD](https://github.com/rundiwu/DeepCAD) (original CAD models).  
  License: MIT.  
  Suitability: Vector-to-parametric tasks; rasterize SVGs for full raster-to-vector pipelines. Preprocessing: Limit sequences to 100 primitives. Last updated: December 2025.

- **DeepPatent2**  
  Size: >2.7M technical drawings (2M full patents, 2.8M segmented figures).  
  Formats: PNG (original/segmented), JSON (metadata/semantics), CSV (distributions). Organized by year (2007-2020).  
  Features: Patent drawings with object names (132k unique), viewpoints (22k), captions, bounding boxes; semantic extraction via NN (e.g., "object": "chair"). Compound drawings segmented.  
  Access: [OneDrive (2020 data - Original_2020.tar.gz)](https://olddominion-my.sharepoint.com/personal/j1wu_odu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020%2FOriginal%5F2020%2Etar%2Egz&viewid=7828cbdf%2D98fd%2D45c8%2D9fbf%2D337e03d13638&parent=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020) or [OSF (2007 subset)](https://osf.io/kv4xa).  
  License: CC BY-NC 2.0.  
  Suitability: Technical/patent vectorization; coords for primitive extraction. Preprocessing: Use JSON for spatial augmentation.

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
  Access: [Hugging Face](https://huggingface.co/datasets/jackluoluo/ArchCAD).  
  License: CC BY-NC 4.0.  
  Suitability: Large-scale end-to-end vectorization; precise primitives. Preprocessing: Auto + human correction.

- **ResPlan**  
  Size: 17,000 residential floorplans.  
  Formats: JSON (vectors/geometries/semantics/graphs), PNG previews; NetworkX graphs.  
  Features: Elements (walls, doors) and spaces (rooms); connectivity graphs; realistic layouts. Unit-level; 3D convertible.  
  Access: [arXiv (2508.14006)](https://arxiv.org/abs/2508.14006); request access for dataset via form or contact authors. Open-source pipeline for geometry cleaning/alignment.  
  License: CC BY 4.0.  
  Suitability: Architectural vector-graph tasks; generative. Preprocessing: Python cleaning/alignment.

- **CubiCasa5K**  
  Size: 5,000 scanned floorplans.  
  Formats: High-res raster (up to 6k px), SVG annotations; LMDB database (~105 GB).  
  Features: 80+ semantic labels (rooms, furniture, walls); Finnish real estate CAD-sourced. Dense polygon annotations.  
  Access: [GitHub](https://github.com/CubiCasa/CubiCasa5k); [Zenodo](https://zenodo.org/record/2613548).  
  License: CC BY-NC 4.0 (dataset open on Zenodo).  
  Suitability: Noisy raster-to-vector. Preprocessing: Align pairs.

- **ABC**  
  Size: ~10k vector mechanical drawings (from 1M+ CAD models).  
  Formats: Vector projections (boundaries), raster derivable.  
  Features: Parametric CAD with edges/surfaces; geometric DL.  
  Access: [GitHub](https://github.com/deep-geometry/abc-dataset).  
  License: CC BY 4.0.  
  Suitability: Mechanical curves/lines. Preprocessing: 2D views from boundaries.

- **MSD (Modified Swiss Dwellings)**  
  Size: 5,372 floor plans (17.4 GB).  
  Formats: Raster, vector, graphs; CSV (geometry data).  
  Features: Access graphs; multi-floor; Swiss dwellings; split into train/test. Includes cleaned Pandas dataframe with geometries (rooms, walls).  
  Access: [Kaggle](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings).  
  License: CC BY-SA 4.0.  
  Suitability: Complex multi-unit vector-graph. Preprocessing: Extract units.

- **CVC-FP**  
  Size: 122 floorplans (4 subsets).  
  Formats: Raster with masks; vector derivable.  
  Features: Semantic segmentation; architectural.  
  Access: [CVC-FP.zip](http://dag.cvc.uab.es/DATASETS/CVC-FP.zip); evaluation scripts [Evaluation.zip](http://dag.cvc.uab.es/DATASETS/Evaluation.zip).  
  License: Research.  
  Suitability: Baseline vector traces. Preprocessing: Pair masks.

- **DeepPatent**  
  Size: >350k patent drawings.  
  Formats: PNG (from TIF), XML metadata.  
  Features: Design patents (2018-2019); multi-view.  
  Access: [GitHub](https://github.com/GoFigure-LANL/DeepPatent-dataset); Google Drive (compressed parts).  
  License: BSD-3-Clause.  
  Suitability: Patent retrieval/vectorization. Preprocessing: Query sampling.

- **DLD (Degraded Line Drawings)**  
  Size: 81 photos/scans of floorplans.  
  Formats: Raster (raw/cleaned), vector targets.  
  Features: Real noise/text; inpainted lines. Resolution ~1300x1000.  
  Access: Contact authors (ECCV 2020 paper "Deep Vectorization of Technical Drawings"); no public repo.  
  License: Research.  
  Suitability: Degraded raster-to-vector.

- **SketchGraphs**  
  Size: 15M CAD sketches.  
  Formats: Constraint graphs (JSON/serialized); nodes (primitives), edges (constraints).  
  Access: [GitHub](https://github.com/PrincetonLIPS/SketchGraphs).  
  License: Open/research (per Onshape Terms).  
  Suitability: Relational vector tasks; pair with rasters.

- **PDTW150K**  
  Size: 150k+ patents, 850k+ drawings.  
  Formats: Images, text metadata, bounding boxes.  
  Features: Multi-view figures; object/viewpoint info.  
  Access: [GitHub](https://github.com/ncyuMARSLab/PDTW150K); Google Drive (compressed parts).  
  License: Open Government Data License v1.0.  
  Suitability: Patent retrieval/vectorization. Preprocessing: Segment figures.

- **PatentDesc-355K**  
  Size: ~355k figures from 60k+ documents.  
  Formats: Images, captions, metadata.  
  Features: Multimodal descriptions; sparse illustrations.  
  Access: Code and data publicly available (arXiv 2501.15074).  
  License: Research.  
  Suitability: Image-to-vector with captions.

- **IMPACT**  
  Size: 500k patents, 3.61M figures (2007–2022).  
  Formats: Multimodal (figures, captions, metadata); CSV files.  
  Features: Viewpoint-coherent captions; retrieval/classification.  
  Access: [GitHub](https://github.com/AI4Patents/IMPACT); [Hugging Face](https://huggingface.co/datasets/AI4Patents/IMPACT).  
  License: Open.  
  Suitability: Patent vectorization; VQA/3D.

- **CADSketchNet**  
  Size: 1k+ annotated sketches.  
  Formats: Sketches paired with 3D CAD.  
  Features: 3D retrieval.  
  Access: Contact authors (Computers & Graphics paper); no public repo.  
  License: Research.  
  Suitability: Sketch-to-vector/CAD.

- **VideoCAD**  
  Size: 41k+ CAD UI videos.  
  Formats: Videos, 3D models, UI traces.  
  Features: Long-horizon UI for generation.  
  Access: [GitHub](https://github.com/ghadinehme/VideoCAD); [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WX8PCK).  
  License: Open.  
  Suitability: Dynamic vectorization/UI-CAD.

- **CFP (Comprehensive Floor Plan)**  
  Size: 100k+ elements (high-res images).  
  Formats: Raster with vector points (quartets for shapes).  
  Features: Diverse plans; sparse boundaries.  
  Access: Contact authors (no public repo).  
  License: Research.  
  Suitability: Minimal-point vectorization.

- **CAD/BIM Collection**  
  Size: 4,596 IFC, 6,471 RVT, 156,024 DWG.  
  Formats: Vector CAD/BIM files.  
  Features: Automated from architectural sites.  
  Access: Contact authors (site unavailable).  
  License: Varies (public).  
  Suitability: Large-scale vector analysis. Preprocessing: For ML.

- **HoliCity**  
  Size: City-scale 3D models (6,300 panoramas).  
  Formats: CAD (DXF/IFC/OBJ/STL).  
  Features: Urban buildings/parcels/roads; vector maps.  
  Access: [Project site](https://holicity.io/); sample data on GitHub.  
  License: Commercial/research.  
  Suitability: Large-area technical vectorization; 2D/3D.

- **DrivAerNet++**  
  Size: 8,150+ car meshes/simulations.  
  Formats: 3D meshes, aerodynamic data (pressure/velocity fields).  
  Features: Multimodal automotive aerodynamics; point clouds/parts.  
  Access: [GitHub](https://github.com/Mohamedelrefaie/DrivAerNet).  
  License: CC BY-NC 4.0.  
  Suitability: Mechanical/automotive vectorization; 3D-to-vector.

- **BlendedNet**  
  Size: 999 blended wing body geometries (~9 conditions each; 8,830 cases).  
  Formats: Geometries, RANS simulations (Spalart-Allmaras).  
  Features: Aerodynamic; 9-14M cells/case.  
  Access: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VJT9EP); [GitHub](https://github.com/nicksungg/clarc_blended_wing_body).  
  License: Apache 2.0 variant.  
  Suitability: Aerospace mechanical vectorization.

- **Engineering Symbols**  
  Size: 2,432 instances (multi-class imbalanced).  
  Formats: Images/symbols.  
  Features: Symbols from Oil & Gas/construction/mechanical.  
  Access: [GitHub](https://github.com/heyad/Eng_Diagrams).  
  License: Research.  
  Suitability: Symbol spotting in drawings.

## Synthetic/Processed Datasets

These provide controlled data for initial training or augmentation.

- **SESYD**  
  Size: 1,000 floorplans (10 classes, 100 each).  
  Formats: Vector (raster renderable).  
  Features: Furniture layouts/variations; symbol spotting.  
  Access: [Site](https://mathieu.delalandre.free.fr/projects/sesyd/).  
  License: Free.  
  Suitability: Controlled experiments. Preprocessing: Render rasters.

- **FPLAN-POLY**  
  Size: ~48 polygonal floorplans.  
  Formats: Vector.  
  Features: Room shapes/connectivity.  
  Access: [CVC site](http://www.cvc.uab.es/~marcal/FPLAN-POLY/index.html).  
  License: Research.  
  Suitability: Simple vector-graph. Preprocessing: Render inputs.

- **RPLAN**  
  Size: 80k+ floorplans.  
  Formats: Raster, graphs (vector derivable).  
  Features: Room layouts; generative DL.  
  Access: [Site](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html); request via form.  
  License: Free.  
  Suitability: Graph-to-vector. Preprocessing: Conversion.

- **QuickDraw**  
  Size: 50M+ vector sketches (345 classes).  
  Formats: Stroke-3 (vector), raster.  
  Features: Human-drawn; 100k+/class.  
  Access: [Google](https://quickdraw.withgoogle.com/data).  
  License: CC BY 4.0.  
  Suitability: Sketch vectorization; large-scale.

- **TU-Berlin**  
  Size: 20k sketches (250 categories, 80 each).  
  Formats: Raster/vector.  
  Features: Freehand sketches.  
  Access: [Hugging Face](https://huggingface.co/datasets/kmewhort/tu-berlin-png); or university archives.  
  License: Research.  
  Suitability: Sketch recognition/vectorization.

- **IAM Handwriting**  
  Size: 115k word images (offline), 13k lines (online; 70k words).  
  Formats: Raster (offline), point coordinates (online).  
  Features: Sequence recognition.  
  Access: [Site](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database); register for download.  
  License: Research.  
  Suitability: Handwriting vectorization; rasterize online for pairs.

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