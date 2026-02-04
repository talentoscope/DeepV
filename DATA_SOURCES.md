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
  Size: 157k–161k SVG-to-CAD pairs (from CAD models; 4 views: Front, Top, Right, Isometric).  
  Formats: SVG vectors, raster PNG (derived), JSON/sequences for parametric CAD commands.  
  Features: Aligns vector primitives (lines, arcs, curves) with editable CAD operations; preserves geometry and design intent; path reordering/normalization. Derived from DeepCAD models via FreeCAD. Split: 90% train, 5% val/test.  
  Access: [Google Drive](https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU?usp=sharing).  
  License: Academic/research.  
  Suitability: Vector-to-parametric tasks; rasterize SVGs for full raster-to-vector pipelines. Preprocessing: Limit sequences to 100 primitives.

- **DeepPatent2**  
  Size: >2.7M technical drawings (2M full patents, 2.8M segmented figures).  
  Formats: PNG (original/segmented), JSON (metadata/semantics), CSV (distributions). Organized by year (2007-2020).  
  Features: Patent drawings with object names (132k unique), viewpoints (22k), captions, bounding boxes; semantic extraction via NN (e.g., "object": "chair"). Compound drawings segmented.  
  Access: [OneDrive](https://bit.ly/deeppatent2-onedrive); 2007 subset on [OSF](https://osf.io/kv4xa).  
  License: CC BY-NC 2.0.  
  Suitability: Technical/patent vectorization; coords for primitive extraction. Preprocessing: Use JSON for spatial augmentation.

- **FloorPlanCAD**  
  Size: 15,663 CAD drawings (expanded from ~10k).  
  Formats: SVG vectors with PNG rasterizations; COCO annotations.  
  Features: 30+ categories (walls, symbols); panoptic spotting; residential/commercial/hospitals; 3D derivable.  
  Access: [Hugging Face](https://huggingface.co/datasets/Voxel51/FloorPlanCAD); project site (floorplancad.github.io).  
  License: Research (contact authors).  
  Suitability: CAD vectorization benchmark; raster derivable. Preprocessing: Patch large drawings.

- **ArchCAD-400K**  
  Size: 413,062 chunks from 5,538 drawings (~26x larger than FloorPlanCAD).  
  Formats: SVG vectors (labeled/instances/RGB), raster aligned, JSON, Q&A, point clouds.  
  Features: Multimodal; 30+ categories; diverse buildings/scales; panoptic (semantic/instance); vectorized annotation workflow.  
  Access: [Hugging Face](https://huggingface.co/datasets/jackluoluo/ArchCAD).  
  License: Research/open.  
  Suitability: Large-scale end-to-end vectorization; precise primitives. Preprocessing: Auto + human correction.

- **ResPlan**  
  Size: 17,000 residential floorplans.  
  Formats: JSON (vectors/geometries/semantics/graphs), PNG previews; NetworkX graphs.  
  Features: Elements (walls, doors) and spaces (rooms); connectivity graphs; realistic layouts. Unit-level; 3D convertible.  
  Access: [arXiv (2508.14006)](https://arxiv.org/abs/2508.14006); GitHub pipeline.  
  License: CC BY 4.0.  
  Suitability: Architectural vector-graph tasks; generative. Preprocessing: Python cleaning/alignment.

- **CubiCasa5K**  
  Size: 5,000 scanned floorplans.  
  Formats: High-res raster (up to 6k px), SVG annotations.  
  Features: 80+ semantic labels (rooms, furniture, walls); Finnish real estate CAD-sourced.  
  Access: [GitHub](https://github.com/CubiCasa/CubiCasa5k).  
  License: CC BY-NC 4.0.  
  Suitability: Noisy raster-to-vector. Preprocessing: Align pairs.

- **ABC**  
  Size: ~10k vector mechanical drawings (from 1M+ CAD models).  
  Formats: Vector projections (boundaries), raster derivable.  
  Features: Parametric CAD with edges/surfaces; geometric DL.  
  Access: [GitHub](https://deep-geometry.github.io/abc-dataset).  
  License: CC BY 4.0.  
  Suitability: Mechanical curves/lines. Preprocessing: 2D views from boundaries.

- **MSD (Modified Swiss Dwellings)**  
  Size: 5,000+ building complexes.  
  Formats: Raster, vector, graphs.  
  Features: Access graphs; multi-floor; Swiss dwellings.  
  Access: [Kaggle](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings).  
  License: CC0.  
  Suitability: Complex multi-unit vector-graph. Preprocessing: Extract units.

- **CVC-FP**  
  Size: ~122 floorplans (4 subsets).  
  Formats: Raster with masks; vector derivable.  
  Features: Semantic segmentation; architectural.  
  Access: [CVC Site](http://dag.cvc.uab.es/floorplan/).  
  License: Research.  
  Suitability: Baseline vector traces. Preprocessing: Pair masks.

- **DeepPatent**  
  Size: 350k+ patent drawings.  
  Formats: PNG (from TIF), XML metadata.  
  Features: Design patents (2018-2019); multi-view.  
  Access: Google Drive (per GitHub).  
  License: CC0.  
  Suitability: Patent retrieval/vectorization. Preprocessing: Query sampling.

- **DLD (Degraded Line Drawings)**  
  Size: 81 photos/scans of floorplans.  
  Formats: Raster (raw/cleaned), vector targets.  
  Features: Real noise/text; inpainted lines.  
  Access: Contact authors (ECCV 2020).  
  License: Research.  
  Suitability: Degraded raster-to-vector.

- **SketchGraphs**  
  Size: 15M CAD sketches.  
  Formats: Constraint graphs (JSON/serialized); nodes (primitives), edges (constraints).  
  Access: [GitHub](https://github.com/PrincetonLIPS/SketchGraphs).  
  License: Open/research.  
  Suitability: Relational vector tasks; pair with rasters.

- **PDTW150K**  
  Size: 150k+ patents, 850k+ drawings.  
  Formats: Images, text metadata, bounding boxes.  
  Features: Multi-view figures; object/viewpoint info.  
  Access: MMM 2024 paper; contact authors.  
  License: Not specified.  
  Suitability: Patent retrieval/vectorization. Preprocessing: Segment figures.

- **PatentDesc-355K**  
  Size: ~355k figures from 60k+ documents.  
  Formats: Images, captions, metadata.  
  Features: Multimodal descriptions; sparse illustrations.  
  Access: arXiv 2501.15074; release pending.  
  License: Research.  
  Suitability: Image-to-vector with captions.

- **IMPACT**  
  Size: 500k patents, 3.61M figures (2007–2022).  
  Formats: Multimodal (figures, captions, metadata).  
  Features: Viewpoint-coherent captions; retrieval/classification.  
  Access: [GitHub](https://github.com/AI4Patents/IMPACT).  
  License: Open.  
  Suitability: Patent vectorization; VQA/3D.

- **CADSketchNet**  
  Size: 1k+ annotated sketches.  
  Formats: Sketches paired with 3D CAD.  
  Features: 3D retrieval.  
  Access: Contact authors (Computers & Graphics).  
  License: Research.  
  Suitability: Sketch-to-vector/CAD.

- **VideoCAD**  
  Size: 41k+ CAD UI videos.  
  Formats: Videos, 3D models, UI traces.  
  Features: Long-horizon UI for generation.  
  Access: OpenReview; code available.  
  License: Open.  
  Suitability: Dynamic vectorization/UI-CAD.

- **CFP (Comprehensive Floor Plan)**  
  Size: 100k+ elements (high-res images).  
  Formats: Raster with vector points (quartets for shapes).  
  Features: Diverse plans; sparse boundaries.  
  Access: Contact authors.  
  License: Research.  
  Suitability: Minimal-point vectorization.

- **CAD/BIM Collection**  
  Size: 4,596 IFC, 6,471 RVT, 156,024 DWG.  
  Formats: Vector CAD/BIM files.  
  Features: Automated from architectural sites.  
  Access: DataDrivenConstruction.io.  
  License: Varies (public).  
  Suitability: Large-scale vector analysis. Preprocessing: For ML.

- **HoliCity**  
  Size: City-scale 3D models.  
  Formats: CAD (DXF/IFC/OBJ/STL).  
  Features: Urban buildings/parcels/roads; vector maps.  
  Access: AccuCities/TopoExport.  
  License: Commercial/research.  
  Suitability: Large-area technical vectorization; 2D/3D.

- **DrivAerNet++**  
  Size: 8,000+ car meshes/simulations.  
  Formats: 3D meshes, aerodynamic data (pressure/velocity fields).  
  Features: Multimodal automotive aerodynamics; point clouds/parts.  
  Access: [GitHub](https://github.com/Mohamedelrefaie/DrivAerNet).  
  License: Open.  
  Suitability: Mechanical/automotive vectorization; 3D-to-vector.

- **BlendedNet**  
  Size: 999 blended wing body geometries (~9 conditions each; 8,830 cases).  
  Formats: Geometries, RANS simulations (Spalart-Allmaras).  
  Features: Aerodynamic; 9-14M cells/case.  
  Access: Public (per paper).  
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
  Access: Contact authors (ECCV).  
  License: Research.  
  Suitability: Simple vector-graph. Preprocessing: Render inputs.

- **RPLAN**  
  Size: 80k+ floorplans.  
  Formats: Raster, graphs (vector derivable).  
  Features: Room layouts; generative DL.  
  Access: [Site](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html).  
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
  Access: Public.  
  License: Research.  
  Suitability: Sketch recognition/vectorization.

- **IAM Handwriting**  
  Size: 115k word images (offline), 13k lines (online; 70k words).  
  Formats: Raster (offline), point coordinates (online).  
  Features: Sequence recognition.  
  Access: IAM site.  
  License: Research.  
  Suitability: Handwriting vectorization; rasterize online for pairs.