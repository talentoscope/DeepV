"""Example runner for DeepV showing CPU and GPU batch runs (dry-run).

Edit paths below to point to your models and example images. This script calls
`run_pipeline.main(options)` with a constructed options namespace so you can
validate the pipeline behavior without using the CLI.

Usage: python scripts/example_run.py
"""
from types import SimpleNamespace
import os

from run_pipeline import main as pipeline_main


def example_cpu_batch():
    # Update these paths to point to actual files in your environment
    opts = SimpleNamespace()
    opts.gpu = []  # CPU run
    opts.model_path = os.path.join('logs', 'models', 'vectorization', 'lines', 'model_lines.weights')
    opts.json_path = os.path.join('vectorization', 'models', 'specs', 'resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json')
    opts.data_dir = os.path.join('examples', 'images')
    opts.image_name = None  # process all images in data_dir
    opts.overlap = 0
    opts.model_output_count = 10
    opts.primitive_type = 'line'
    opts.output_dir = os.path.join('outputs')
    opts.workers = 2
    opts.use_cleaning = False
    opts.cleaning_model_path = None
    opts.save_outputs = False  # dry-run: don't write files
    opts.diff_render_it = 10

    print('Running CPU batch dry-run (workers=2)')
    results = pipeline_main(opts)
    print('Processed', len(results), 'images (CPU)')


def example_gpu_single():
    # Update GPU index and paths as needed
    opts = SimpleNamespace()
    opts.gpu = ['0']
    opts.model_path = os.path.join('logs', 'models', 'vectorization', 'lines', 'model_lines.weights')
    opts.json_path = os.path.join('vectorization', 'models', 'specs', 'resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json')
    opts.data_dir = os.path.join('examples', 'images')
    opts.image_name = None
    opts.overlap = 0
    opts.model_output_count = 10
    opts.primitive_type = 'line'
    opts.output_dir = os.path.join('outputs')
    opts.workers = 1
    opts.use_cleaning = False
    opts.cleaning_model_path = None
    opts.save_outputs = False
    opts.diff_render_it = 10

    print('Running single-GPU sequential dry-run (gpu=0)')
    results = pipeline_main(opts)
    print('Processed', len(results), 'images (GPU)')


if __name__ == '__main__':
    # Run the CPU example by default. Uncomment to run GPU example.
    example_cpu_batch()
    # example_gpu_single()
