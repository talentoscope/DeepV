#!/usr/bin/env python3
"""
DeepV Pipeline Performance Profiler

Comprehensive profiling script for the DeepV raster-to-vector pipeline.
Uses PyTorch profiler and Python cProfile to identify performance bottlenecks.
"""

import argparse
import cProfile
import io
import os
import pstats
import time
from pathlib import Path
from typing import Dict, Any, Optional
import gc

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PipelineRunner lazily inside profiling methods to avoid heavy
# top-level imports (torchvision, etc.) during script startup.


class DeepVProfiler:
    """Comprehensive profiler for DeepV pipeline performance analysis."""

    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Profiling results storage
        self.timing_results: Dict[str, float] = {}
        self.memory_stats: Dict[str, Any] = {}
        self.pytorch_profile_results = None

    def profile_pipeline_with_cprofile(self, options) -> Dict[str, Any]:
        """Profile pipeline using Python's cProfile."""
        print("🔍 Running cProfile analysis...")

        pr = cProfile.Profile()
        pr.enable()

        # Run pipeline
        start_time = time.time()
        try:
            from run_pipeline import PipelineRunner
            runner = PipelineRunner(options)
            runner.run()
            end_time = time.time()
        except Exception as e:
            pr.disable()
            print(f"❌ Pipeline failed during cProfile: {e}")
            return {}
        finally:
            pr.disable()

        # Process results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()

        total_time = end_time - start_time

        # Save cProfile results
        cprofile_file = self.output_dir / "cprofile_results.txt"
        with open(cprofile_file, 'w') as f:
            f.write("DeepV Pipeline cProfile Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total execution time: {total_time:.3f} seconds\n\n")
            f.write("Profile statistics (sorted by cumulative time):\n")
            f.write("-" * 50 + "\n")
            f.write(s.getvalue())

        print(f"✅ cProfile results saved to {cprofile_file}")

        return {
            'profile_stats': s.getvalue(),
            'total_time': total_time,
            'functions': self._extract_function_stats(ps)
        }

    def profile_pipeline_with_pytorch(self, options) -> Dict[str, Any]:
        """Profile pipeline using PyTorch profiler."""
        print("🔍 Running PyTorch profiler analysis...")

        # Configure profiler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,  # Reduce memory usage
            profile_memory=True,
            with_stack=False  # Reduce memory usage
        ) as prof:

            with record_function("deepv_pipeline"):
                try:
                    from run_pipeline import PipelineRunner
                    runner = PipelineRunner(options)
                    runner.run()
                except Exception as e:
                    print(f"❌ Pipeline failed during PyTorch profiling: {e}")
                    return {}

        # Save profiling results
        profile_file = self.output_dir / "pytorch_profile.json"
        prof.export_chrome_trace(str(profile_file))

        # Generate summary
        summary_file = self.output_dir / "pytorch_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PyTorch Profiler Summary\n")
            f.write("=" * 50 + "\n\n")

            # CPU time breakdown
            f.write("CPU Time Breakdown:\n")
            f.write("-" * 20 + "\n")
            cpu_stats = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
            f.write(str(cpu_stats))
            f.write("\n\n")

            # Memory usage
            f.write("Memory Usage Summary:\n")
            f.write("-" * 20 + "\n")
            memory_stats = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20)
            f.write(str(memory_stats))
            f.write("\n\n")

            # CUDA time (if available)
            if torch.cuda.is_available():
                f.write("CUDA Time Breakdown:\n")
                f.write("-" * 20 + "\n")
                cuda_stats = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                f.write(str(cuda_stats))

        print(f"✅ PyTorch profiling results saved to {profile_file} and {summary_file}")
        return {'profiler': prof}

    def profile_memory_usage(self, options) -> Dict[str, Any]:
        """Profile memory usage during pipeline execution."""
        print("🔍 Profiling memory usage...")

        # Reset peak memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            from run_pipeline import PipelineRunner
            runner = PipelineRunner(options)
            runner.run()
        except Exception as e:
            print(f"❌ Pipeline failed during memory profiling: {e}")
            return {}

        # Collect memory statistics
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        memory_stats = {
            'initial_memory_mb': initial_memory / 1024 / 1024,
            'final_memory_mb': final_memory / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'memory_increase_mb': (final_memory - initial_memory) / 1024 / 1024,
            'cuda_available': torch.cuda.is_available()
        }

        self.memory_stats = memory_stats

        # Save memory stats
        memory_file = self.output_dir / "memory_profile.txt"
        with open(memory_file, 'w') as f:
            f.write("DeepV Pipeline Memory Profile\n")
            f.write("=" * 40 + "\n\n")
            for key, value in memory_stats.items():
                if 'mb' in key:
                    f.write(f"{key}: {value:.2f} MB\n")
                else:
                    f.write(f"{key}: {value}\n")

        print(f"✅ Memory profiling results saved to {memory_file}")
        return memory_stats

    def run_comprehensive_profiling(self, options) -> Dict[str, Any]:
        """Run all profiling analyses."""
        print("🚀 Starting comprehensive DeepV pipeline profiling...")
        print(f"📁 Results will be saved to: {self.output_dir}")

        results = {}

        # 1. Memory profiling (lightweight, run first)
        results['memory'] = self.profile_memory_usage(self._create_options_copy(options))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. cProfile analysis
        results['cprofile'] = self.profile_pipeline_with_cprofile(self._create_options_copy(options))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. PyTorch profiling (disabled for now to avoid OOM)
        # results['pytorch'] = self.profile_pipeline_with_pytorch(self._create_options_copy(options))
        results['pytorch'] = {}
        print("🔍 Skipping PyTorch profiling to avoid OOM")

        # Generate summary report
        self._generate_summary_report(results)

        print("✅ Comprehensive profiling completed!")
        return results

    def _create_options_copy(self, options):
        """Create a fresh copy of options to avoid state pollution between profiling runs."""
        from argparse import Namespace
        new_options = Namespace()
        for key, value in vars(options).items():
            if isinstance(value, list):
                setattr(new_options, key, value.copy())
            else:
                setattr(new_options, key, value)
        return new_options

    def _extract_function_stats(self, ps: pstats.Stats, limit: int = 20):
        """Extract top functions from a pstats.Stats object.

        Returns a list of dicts with keys: function, filename, line, ncalls, tottime, cumtime.
        """
        stats = []
        # ps.stats is a dict: key -> (cc, nc, tt, ct, callers)
        for func, stat in ps.stats.items():
            filename, line, funcname = func
            cc, nc, tt, ct, callers = stat
            stats.append({
                'function': funcname,
                'filename': filename,
                'line': line,
                'ncalls': nc,
                'tottime': tt,
                'cumtime': ct,
            })

        # Sort by cumulative time descending
        stats = sorted(stats, key=lambda x: x['cumtime'], reverse=True)
        return stats[:limit]

    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive summary report."""
        summary_file = self.output_dir / "profiling_summary.md"

        with open(summary_file, 'w') as f:
            f.write("# DeepV Pipeline Performance Profile Summary\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall timing
            if 'cprofile' in results and results['cprofile']:
                total_time = results['cprofile'].get('total_time', 0)
                f.write("## Overall Performance\n\n")
                f.write(f"- **Total pipeline time:** {total_time:.3f} seconds\n")

                # Performance assessment
                if total_time < 1.0:
                    f.write("- **Performance rating:** Excellent (< 1 second)\n")
                elif total_time < 5.0:
                    f.write("- **Performance rating:** Good (1-5 seconds)\n")
                elif total_time < 15.0:
                    f.write("- **Performance rating:** Acceptable (5-15 seconds)\n")
                else:
                    f.write("- **Performance rating:** Needs optimization (> 15 seconds)\n")

                f.write("\n")

            # Memory usage
            if 'memory' in results and results['memory']:
                mem = results['memory']
                f.write("## Memory Usage\n\n")
                f.write(f"- **Peak memory:** {mem.get('peak_memory_mb', 0):.2f} MB\n")
                f.write(f"- **Memory increase:** {mem.get('memory_increase_mb', 0):.2f} MB\n")
                f.write(f"- **CUDA available:** {mem.get('cuda_available', False)}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### Immediate Actions\n")
            f.write("- Review `cprofile_results.txt` for top time-consuming functions\n")
            f.write("- Analyze `pytorch_profile.json` in Chrome tracing for detailed GPU/CPU breakdown\n")
            f.write("- Check `memory_profile.txt` for memory optimization opportunities\n\n")

            f.write("### Optimization Targets\n")
            f.write("- Target: < 2 seconds per 64×64 patch (current Bézier splatting: 0.0064s OK)\n")
            f.write("- Memory: < 8GB peak usage for 128×128 images\n")
            f.write("- CPU/GPU balance: Minimize data transfers between devices\n\n")

            f.write("### Files Generated\n")
            f.write("- `cprofile_results.txt` - Python function timing analysis\n")
            f.write("- `pytorch_profile.json` - Detailed GPU/CPU profiling (open in Chrome)\n")
            f.write("- `pytorch_summary.txt` - PyTorch profiler summary tables\n")
            f.write("- `memory_profile.txt` - Memory usage statistics\n")

        print(f"📊 Summary report saved to {summary_file}")


def main():
    """Main profiling entry point."""
    parser = argparse.ArgumentParser(description="Profile DeepV pipeline performance")
    parser.add_argument('--data_dir', type=str, default='data/raw/floorplancad/train1',
                       help='Directory containing test images')
    parser.add_argument('--image_name', type=str, default='0109-0037.png',
                       help='Test image filename')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Full path to test image (overrides data_dir and image_name)')
    parser.add_argument('--primitive_type', type=str, default='line',
                       help='Primitive type to process')
    parser.add_argument('--output_dir', type=str, default='profiling_results',
                       help='Directory to save profiling results')
    parser.add_argument('--model_output_count', type=int, default=5,
                       help='Number of primitives to generate')
    parser.add_argument('--init_random', action='store_true',
                       help='Use random initialization instead of trained model')
    parser.add_argument('--model_path', type=str, default='models/model_lines.weights',
                       help='Path to trained model')
    parser.add_argument('--json_path', type=str, 
                       default='vectorization/models/specs/resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json',
                       help='Path to JSON model specification file')
    parser.add_argument('--diff_render_it', type=int, default=100,
                       help='Number of refinement iterations (early stopping enabled)')
    parser.add_argument('--early_stop_threshold', type=float, default=0.001,
                       help='Early stopping IOU improvement threshold')
    parser.add_argument('--gpu', action='append', help='GPU to use')

    args = parser.parse_args()

    # Create options object for PipelineRunner (argparse.Namespace)
    from argparse import Namespace
    options = Namespace()
    
    # Handle image path specification
    if args.image_path:
        # Use full image path - extract directory and filename
        image_path = Path(args.image_path)
        options.data_dir = str(image_path.parent)
        options.image_name = image_path.name
        print(f"Using image: {args.image_path}")
    else:
        # Use traditional data_dir + image_name
        options.data_dir = args.data_dir
        options.image_name = args.image_name
    
    options.primitive_type = args.primitive_type
    options.output_dir = f"{args.output_dir}/pipeline_output"
    options.model_output_count = args.model_output_count
    options.init_random = args.init_random
    options.gpu = args.gpu if args.gpu else []
    options.model_path = args.model_path
    options.json_path = args.json_path
    options.curve_count = 10  # default
    # Reduce iterations for profiling to keep runtime reasonable
    options.diff_render_it = args.diff_render_it  # Use argument
    options.rendering_type = 'bezier_splatting'
    options.overlap = 0
    options.max_angle_to_connect = 10
    options.max_distance_to_connect = 3
    options.gpu = args.gpu if args.gpu else []

    # Run profiling
    profiler = DeepVProfiler(args.output_dir)
    results = profiler.run_comprehensive_profiling(options)

    print("\n🎯 Profiling completed! Key insights:")
    if 'cprofile' in results and results['cprofile']:
        total_time = results['cprofile'].get('total_time', 0)
        print(".2f")
    if 'memory' in results and results['memory']:
        mem = results['memory']
        print(".2f")


if __name__ == "__main__":
    main()