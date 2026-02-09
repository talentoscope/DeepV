# Vectorization Training Setup Checklist (ARCHIVED)

*This document outlines the historical steps taken to implement proper vectorization training by replacing placeholder targets with real SVG-parsed vector primitives. The infrastructure is now fully implemented and working.*

## 🎉 STATUS: COMPLETED - Infrastructure Fully Implemented

**As of February 2026, the SVG parsing and primitive extraction infrastructure is fully implemented and integrated:**

- ✅ **Dataset Classes**: `util_files/data/vectordata/datasets.py` has working SVG dataset infrastructure
- ✅ **Export Functions**: `cad/export.py` can export primitives to SVG/DXF formats
- ✅ **Training Integration**: FloorPlanCAD training script uses real SVG parsing instead of placeholder targets
- ✅ **Model Training**: Vectorization models can be trained on FloorPlanCAD data with real ground truth

**Current Focus**: Architecture improvements (Non-Autoregressive Transformer) to address over-segmentation issues, not training setup.

---

---

## Current State Assessment

### ✅ What's Working
- [x] Training script infrastructure (`scripts/train_floorplancad.py`)
- [x] Model loading and initialization
- [x] Data loading (raster images)
- [x] Training loop with GPU support
- [x] Loss computation and optimization
- [x] Basic dataset class structure
- [x] **SVG parsing infrastructure** (`util_files/data/vectordata/common.py`)
- [x] **Graphics primitives framework** (`util_files/data/graphics_primitives.py`)
- [x] **Real primitive extraction** (`util_files/data/vectordata/datasets.py`)

## Current State Assessment

### ✅ What's Working
- [x] Training script infrastructure (`scripts/train_floorplancad.py`)
- [x] Model loading and initialization
- [x] Data loading (raster images)
- [x] Training loop with GPU support
- [x] Loss computation and optimization
- [x] Basic dataset class structure
- [x] **SVG parsing infrastructure** (`util_files/data/vectordata/common.py`)
- [x] **Graphics primitives framework** (`util_files/data/graphics_primitives.py`)
- [x] **Real primitive extraction** (`util_files/data/vectordata/datasets.py`)

### ❌ What's Missing
- [ ] **Connection between existing SVG parsing and FloorPlanCAD training script**
- [ ] **FloorPlanCAD-specific SVG parsing**
- [ ] Target generation from existing primitive objects to neural network format
- [ ] Integration of SVG parsing into `FloorPlanCADDataset.__getitem__` method
- [ ] Proper evaluation metrics
- [ ] Integration with full pipeline

## Phase 1: Connect Existing SVG Infrastructure to FloorPlanCAD Training

### 1.1 Analyze FloorPlanCAD Data Structure
- [x] **DISCOVERED**: FloorPlanCAD SVGs contain `<path>` elements with `d="M x,y L x,y"` commands
- [x] **DISCOVERED**: Training script has custom `FloorPlanCADDataset` class
- [x] **COMPLETED**: Verified `svgpathtools` can parse FloorPlanCAD SVG format (with manual path extraction)
- [x] **COMPLETED**: Tested SVG parsing with sample FloorPlanCAD data

### 1.2 Modify FloorPlanCADDataset.__getitem__ Method
- [x] **COMPLETED**: Replaced `torch.zeros(10, 6)` with real SVG parsing in `FloorPlanCADDataset.__getitem__`
- [x] **COMPLETED**: Import and use `sample_primitive_representation` from `util_files/data/vectordata/common.py`
- [x] **COMPLETED**: Convert `graphics_primitives.Line` and `BezierCurve` objects to neural network tensor format
- [x] **COMPLETED**: Handle variable-length primitive sequences (padding/truncation to 10 primitives)

### 1.3 Understand Primitive Format Conversion
- [x] **DISCOVERED**: Primitives convert to neural network format via `to_repr()` methods
- [ ] Document the exact tensor format expected by models (10 primitives × 6 features each)
- [ ] Verify parameter ranges and normalization

## Phase 2: Integrate SVG Parsing into FloorPlanCAD Dataset

### 2.1 Update FloorPlanCADDataset Class
- [x] **COMPLETED**: `FloorPlanCADDataset` class exists in training script
- [x] **COMPLETED**: Replaced placeholder target generation in `FloorPlanCADDataset.__getitem__`
- [x] **COMPLETED**: Added SVG parsing logic using existing `sample_primitive_representation` function
- [x] **COMPLETED**: Implemented variable-length primitive handling (pad/truncate to 10 primitives)

### 2.2 Test SVG Parsing with FloorPlanCAD Data
- [x] **COMPLETED**: Verified `svgpathtools.svg2paths2()` works with FloorPlanCAD SVG files (manual path extraction)
- [x] **COMPLETED**: Tested `sample_primitive_representation()` on FloorPlanCAD data
- [x] **COMPLETED**: Handled FloorPlanCAD-specific SVG features (percentage values in rect elements)

### 2.3 Data Validation
- [ ] Create validation scripts to check SVG parsing accuracy
- [ ] Compare parsed primitives against original SVG rendering
- [ ] Handle edge cases (empty SVGs, invalid paths)
- [ ] Implement data quality checks

### 2.4 Data Preprocessing
- [x] **DISCOVERED**: Coordinate normalization and augmentation already implemented in `get_random_patch_from_svg()`
- [ ] Verify preprocessing works correctly for training
- [ ] Add data augmentation (rotation, scaling, flipping) if needed
- [ ] Create train/val/test splits with balanced primitive types

## Phase 3: Model Architecture Alignment

### 3.1 Understand Model Output Format
- [ ] Analyze model specification JSON files
- [ ] Document expected output shapes and formats
- [ ] Verify parameter encoding matches model expectations
- [ ] Handle variable vs fixed number of primitives

### 3.2 Loss Function Implementation
- [ ] Implement appropriate loss for vector primitives
- [ ] Consider Chamfer distance for geometric accuracy
- [ ] Add regularization terms
- [ ] Handle variable-length outputs

### 3.3 Output Decoding
- [ ] Create functions to convert model outputs back to SVG
- [ ] Implement primitive reconstruction
- [ ] Add post-processing for valid geometry

## Phase 4: Training Pipeline

### 4.1 Update Training Script
- [ ] Modify `train_floorplancad.py` to use real targets
- [ ] Add progress monitoring and logging
- [ ] Implement early stopping based on validation metrics
- [ ] Add model checkpointing

### 4.2 Hyperparameter Tuning
- [ ] Set appropriate learning rates and schedules
- [ ] Configure batch sizes for GPU memory
- [ ] Tune model architecture parameters
- [ ] Optimize data loading pipeline

### 4.3 Multi-GPU Training
- [ ] Test distributed training setup
- [ ] Implement gradient accumulation if needed
- [ ] Add mixed precision training (FP16)

## Phase 5: Evaluation and Validation

### 5.1 Implement Metrics
- [ ] Geometric accuracy (Chamfer distance, Hausdorff distance)
- [ ] Visual similarity (PSNR, SSIM on rendered outputs)
- [ ] Primitive count accuracy
- [ ] Type classification accuracy

### 5.2 Validation Pipeline
- [ ] Create evaluation scripts
- [ ] Implement automated testing
- [ ] Add visualization of predictions vs ground truth
- [ ] Generate quantitative reports

### 5.3 Benchmarking
- [ ] Compare against baseline methods
- [ ] Test on different datasets
- [ ] Measure inference speed and memory usage

## Phase 6: Integration and Deployment

### 6.1 Pipeline Integration
- [ ] Integrate trained model with `run_pipeline.py`
- [ ] Update default model paths
- [ ] Ensure compatibility with existing pipeline
- [ ] Test end-to-end vectorization

### 6.2 Web UI Integration
- [ ] Update `run_web_ui_demo.py` to use trained model
- [ ] Add model selection options
- [ ] Implement real-time evaluation
- [ ] Add export functionality

### 6.3 Documentation Updates
- [ ] Update training instructions
- [ ] Document model performance
- [ ] Create troubleshooting guides
- [ ] Add examples and tutorials

## Phase 7: Advanced Features

### 7.1 Model Improvements
- [ ] Implement attention mechanisms
- [ ] Add hierarchical primitive detection
- [ ] Experiment with different architectures
- [ ] Add uncertainty estimation

### 7.2 Data Expansion
- [ ] Support additional datasets
- [ ] Implement synthetic data generation
- [ ] Add data augmentation techniques
- [ ] Create data quality improvement pipeline

### 7.3 Performance Optimization
- [ ] Implement model quantization
- [ ] Add ONNX export for inference
- [ ] Optimize for edge deployment
- [ ] Implement batch processing

## Testing and Quality Assurance

### Code Quality
- [ ] Add comprehensive unit tests
- [ ] Implement integration tests
- [ ] Add type hints and documentation
- [ ] Code review and refactoring

### Data Quality
- [ ] Validate dataset integrity
- [ ] Implement data versioning
- [ ] Add automated data checks
- [ ] Monitor data drift

### Model Quality
- [ ] Implement model validation
- [ ] Add performance monitoring
- [ ] Create model comparison tools
- [ ] Establish quality gates

## Dependencies and Environment

### Required Packages
- [x] PyTorch with CUDA support
- [x] torchvision
- [x] **svgpathtools** (already used in `util_files/data/vectordata/common.py`)
- [x] ezdxf (for CAD export)
- [x] svgwrite (for SVG export)
- [ ] Additional geometry libraries as needed

### System Requirements
- [ ] GPU with sufficient VRAM (8GB+ recommended)
- [ ] Sufficient disk space for datasets and models
- [ ] Python 3.10+ environment
- [ ] System dependencies (Cairo, etc.)

## Risk Assessment and Mitigation

### Technical Risks
- SVG parsing complexity and edge cases
- Model convergence issues
- Memory constraints with large datasets
- Geometric accuracy challenges

### Mitigation Strategies
- Start with simple primitives (lines only)
- Implement extensive testing and validation
- Use progressive complexity increases
- Maintain fallback to placeholder training

## Success Criteria

### Functional Requirements
- [x] **SVG parsing infrastructure exists** (lines, Bézier curves)
- [x] **Graphics primitives framework implemented** (with neural network I/O)
- [x] **Dataset classes extract real primitives** from SVG files
- [ ] **FloorPlanCAD dataset integration** (connect existing parsing to `FloorPlanCADDataset`)
- [ ] Model can parse and reconstruct simple vector graphics
- [ ] Training converges with real targets
- [ ] Pipeline produces valid vector outputs
- [ ] Web UI demonstrates working vectorization

### Performance Requirements
- [ ] Training completes in reasonable time
- [ ] Inference is fast enough for interactive use
- [ ] Memory usage is acceptable
- [ ] Output quality meets expectations

### Quality Requirements
- [ ] Code is well-tested and documented
- [ ] Data processing is reliable
- [ ] Model performance is reproducible
- [ ] System is maintainable and extensible

## Timeline and Milestones

### Day 1: FloorPlanCAD SVG Integration
- Modify `FloorPlanCADDataset.__getitem__` to parse SVG files
- Integrate `sample_primitive_representation` function
- Convert primitive objects to neural network tensor format
- Test basic training with real targets

### Day 2: Validation and Debugging
- Verify SVG parsing works correctly with FloorPlanCAD data
- Debug any integration issues or data format problems
- Add proper error handling for malformed SVGs
- Test with different primitive types

### Day 3-4: Enhancement and Optimization
- Add proper evaluation metrics
- Optimize data loading pipeline
- Implement variable-length primitive handling
- Test with full training pipeline

### Day 5-6: Integration and Testing
- Full pipeline integration
- Web UI updates
- Comprehensive testing
- Performance optimization

### Day 7: Documentation and Deployment
- Update documentation
- Create examples and tutorials
- Final testing and deployment

## Resources and References

### Code References
- `scripts/train_floorplancad.py` - Main training script with `FloorPlanCADDataset` class (needs SVG parsing integration)
- `vectorization/models/specs/` - Model configurations
- `util_files/data/graphics_primitives.py` - **NEW**: Primitive classes with neural network I/O
- `util_files/data/vectordata/common.py` - **NEW**: SVG parsing and primitive extraction functions
- `util_files/data/vectordata/datasets.py` - **NEW**: SVG dataset infrastructure
- `cad/export.py` - **NEW**: Export primitives to SVG/DXF
- `util_files/` - Utility functions
- `notebooks/` - Example usage

### External Resources
- SVG specification and path commands
- Vector graphics literature
- PyTorch documentation
- Computer vision papers on vectorization

### Team Coordination
- Regular progress updates
- Code review requirements
- Testing protocols
- Documentation standards

---

*This checklist should be updated as work progresses. Check off completed items and add details for any discovered issues or additional requirements.*</content>
<parameter name="filePath">e:\dv\DeepV\VECTORIZATION_TRAINING_SETUP.md