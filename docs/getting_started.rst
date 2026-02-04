Getting Started
===============

Installation
------------

DeepV requires Python 3.8+ and can be installed from source:

.. code-block:: bash

   git clone https://github.com/your-repo/deepv.git
   cd deepv
   pip install -r requirements.txt

For development, also install the dev dependencies:

.. code-block:: bash

   pip install -r requirements-dev.txt

Quick Start
-----------

The main entry point for running the full pipeline is ``run_pipeline.py``:

.. code-block:: bash

   python run_pipeline.py \
     --model_path /path/to/model \
     --json_path /path/to/model_spec.json \
     --data_dir /path/to/data \
     --primitive_type line \
     --model_output_count 10

Pipeline Overview
-----------------

DeepV processes raster images through four stages:

1. **Cleaning**: Denoise and preprocess input images
2. **Vectorization**: Extract primitive shapes using deep learning models
3. **Refinement**: Optimize primitives through differentiable rendering
4. **Merging**: Consolidate overlapping primitives into final vector output

Each stage can be run independently or as part of the full pipeline.