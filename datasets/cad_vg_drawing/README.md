# CAD-VGDrawing Dataset Integration

This directory contains the CAD-VGDrawing dataset downloaded from:
https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU

## Dataset Structure
- `train_data.json`: Training data with SVG-to-CAD pairs
- `val_data.json`: Validation data
- `test_data.json`: Test data
- SVG files: Vector drawings
- PNG files: Rasterized versions (if available)

## Dataset Statistics
- Total samples: ~161,000
- Format: SVG-to-parametric CAD conversion
- Use case: Parametric CAD generation from vector drawings

## Usage
After downloading, run the preprocessing script:
```bash
python dataset/preprocess_cad_vg_drawing.py
```

This will convert the dataset into the format expected by DeepV's training pipeline.

## Source
Paper: Drawing2CAD - "Drawing2CAD: Automated Conversion of Mechanical Drawings to CAD Models"
