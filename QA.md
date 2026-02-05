# DeepV Q&A

Frequently asked questions about **DeepV** - Deep Vectorization of Technical Drawings.

## Table of Contents

- [General Questions](#general-questions)
- [Technical Issues](#technical-issues)
- [Model & Training](#model--training)
- [Usage & Applications](#usage--applications)
- [Contributing](#contributing)

## General Questions

### <b>Code Availability</b>

<b>Answer:</b> We're actively working to release code components. Check the main README.md for current availability status and recent updates.

### <b>Can I use your code?</b>

<b>Answer:</b> Yes, you can use our code. Please cite our paper if you use it in your research or applications.

### <b>Where should I ask questions?</b>

<b>Answer:</b> You can ask questions here on GitHub (Issues/Discussions) or contact the authors directly via email.

## Technical Issues

### <b>Vertical lines appearing dashed in results?</b>

<b>Answer:</b> This may occur due to data augmentation issues where vertical lines are underrepresented in training data, or other implementation details. All metrics are calculated on the actual results. If you identify and fix the issue, please report it so we can improve the code.

### <b>Memory issues during processing?</b>

<b>Answer:</b> Try reducing batch size, using smaller patch sizes (64x64 instead of 128x128), or enabling mixed precision. For very large images, consider processing in tiles.

### <b>CUDA/GPU compatibility issues?</b>

<b>Answer:</b> Ensure PyTorch CUDA version matches your system's CUDA installation. Check with `nvidia-smi` and reinstall PyTorch if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### <b>Import errors or missing dependencies?</b>

<b>Answer:</b> Ensure your virtual environment is activated and all dependencies are installed:

```bash
pip install -r requirements.txt
python scripts/validate_env.py
```

## Model & Training

### <b>How do I train on custom data?</b>

<b>Answer:</b> Prepare your data in the expected format (PNG+DXF pairs or SVG), then use the training scripts in `vectorization/scripts/`. See the training documentation for detailed instructions.

### <b>What primitive types are supported?</b>

<b>Answer:</b> DeepV supports lines, quadratic Bézier curves, cubic Bézier curves, arcs, and splines. Variable counts per patch (up to 20) are supported via autoregressive prediction.

### <b>How accurate is the vectorization?</b>

<b>Answer:</b> Accuracy depends on the input quality and model training. Typical metrics include F1 scores >0.8, IoU >0.7, and Hausdorff distances <2 pixels for well-trained models on clean data.

## Usage & Applications

### <b>What file formats are supported?</b>

<b>Answer:</b>
- **Input**: PNG, JPEG, SVG, PDF, DXF
- **Output**: SVG, DXF, parametric sequences
- **Intermediate**: PyTorch tensors, NumPy arrays

### <b>Can it handle colored drawings?</b>

<b>Answer:</b> The current implementation works with grayscale/binary images. Color information is converted to grayscale during preprocessing.

### <b>What's the maximum image size?</b>

<b>Answer:</b> No hard limit, but practical limits depend on GPU memory. Large images are automatically split into patches for processing.

### <b>Real-time processing possible?</b>

<b>Answer:</b> For small patches (64x64), processing can be near real-time on modern GPUs. Larger images take longer due to patch-based processing.

## Contributing

### <b>How can I contribute?</b>

<b>Answer:</b> See CONTRIBUTING.md for detailed guidelines. We welcome bug fixes, feature additions, documentation improvements, and test contributions.

### <b>Reporting bugs?</b>

<b>Answer:</b> Use GitHub Issues with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)

### <b>Feature requests?</b>

<b>Answer:</b> Open a GitHub Issue with the "enhancement" label. Include:
- Use case description
- Proposed implementation approach
- Potential impact

---

*This Q&A is maintained by the DeepV development team. For the latest information, check GitHub Issues and Discussions.*