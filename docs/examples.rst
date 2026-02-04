Examples
========

Basic Rendering Example
-----------------------

Here's how to render basic vector primitives using DeepV's rendering utilities:

.. code-block:: python

   import numpy as np
   from util_files.rendering.cairo import render
   from util_files.data.graphics_primitives import PT_LINE, PT_QBEZIER

   # Create some random lines and curves
   lines = np.random.rand(3, 4) * 128  # 3 lines, coordinates scaled to 128x128
   line_widths = np.random.rand(3) * 7  # Random widths 0-7 pixels

   curves = np.random.rand(2, 6) * 128  # 2 quadratic Bezier curves
   curve_widths = np.random.rand(2) * 7

   # Render to image
   rendered_image = render(
       lines=lines,
       line_widths=line_widths,
       curves=curves,
       curve_widths=curve_widths,
       canvas_size=(128, 128)
   )

Model Loading and Inference
---------------------------

Load a pre-trained vectorization model and run inference:

.. code-block:: python

   import json
   import torch
   from vectorization.models import GenericVectorizationNet

   # Load model specification
   with open('path/to/model_spec.json', 'r') as f:
       model_spec = json.load(f)

   # Create and load model
   model = GenericVectorizationNet(**model_spec)
   model.load_state_dict(torch.load('path/to/model_weights.pth'))
   model.eval()

   # Run inference on a patch
   with torch.no_grad():
       input_patch = torch.randn(1, 3, 64, 64)  # Example input
       output = model(input_patch, n=10)  # Generate 10 primitives

Data Loading
------------

Load and preprocess training data:

.. code-block:: python

   from util_files.data.line_drawings_dataset import LineDrawingsDataset
   from torch.utils.data import DataLoader

   # Create dataset
   dataset = LineDrawingsDataset(
       root_dir='path/to/data',
       split='train',
       patch_size=64
   )

   # Create data loader
   dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

For more detailed examples, see the Jupyter notebooks in the ``notebooks/`` directory.