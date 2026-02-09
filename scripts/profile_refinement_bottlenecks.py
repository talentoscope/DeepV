#!/usr/bin/env python3
"""
DeepV Refinement Bottleneck Profiler

Detailed profiling of refinement pipeline components to identify and optimize
performance bottlenecks. Target performance: <2 seconds per 64x64 patch.

Features:
- Component-level timing analysis
- Memory usage profiling
- Bottleneck identification and ranking
- Optimization recommendations
- Synthetic test data generation

Uses cProfile and custom timing to provide detailed performance insights
for refinement optimization.

Usage:
    python scripts/profile_refinement_bottlenecks.py
"""

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


class RefinementProfiler:
    """Detailed profiler for refinement pipeline bottlenecks."""

    def __init__(self):
        self.results = {}

    def create_test_patch_data(
        self, num_patches: int = 4, primitives_per_patch: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create realistic test data for refinement profiling.

        Args:
            num_patches: Number of patches to create
            primitives_per_patch: Number of primitives per patch

        Returns:
            Tuple of (patches_rgb, patches_vector)
        """
        np.random.seed(42)

        # Create RGB patches (64x64x1 grayscale)
        patches_rgb = []
        patches_vector = []

        for i in range(num_patches):
            # Create a simple patch with some lines
            patch_rgb = np.ones((64, 64, 1), dtype=np.float32) * 255

            # Add some line-like features
            for _ in range(3):
                x1, y1 = np.random.randint(10, 54, 2)
                x2, y2 = np.random.randint(10, 54, 2)
                # Draw simple line
                if x1 != x2 and y1 != y2:
                    # Simple line drawing
                    length = max(abs(x2 - x1), abs(y2 - y1))
                    for t in range(length):
                        x = int(x1 + (x2 - x1) * t / length)
                        y = int(y1 + (y2 - y1) * t / length)
                        if 0 <= x < 64 and 0 <= y < 64:
                            patch_rgb[y, x] = 0

            patches_rgb.append(patch_rgb)

            # Create corresponding vector predictions
            primitives = []
            for _ in range(primitives_per_patch):
                # [x1, y1, x2, y2, width, opacity]
                primitive = np.random.rand(6)
                primitive[:4] *= 64  # Scale coordinates to patch size
                primitive[4] = primitive[4] * 3 + 1  # Width 1-4
                primitive[5] = 0.8 + primitive[5] * 0.2  # Opacity 0.8-1.0
                primitives.append(primitive)

            patches_vector.append(np.array(primitives))

        return np.array(patches_rgb), np.array(patches_vector)

    def profile_function_detailed(self, func, *args, name: str = "", **kwargs) -> Dict[str, Any]:
        """
        Profile a function with detailed timing and memory analysis.

        Args:
            func: Function to profile
            name: Name for the profiling result
            *args, **kwargs: Arguments for the function

        Returns:
            Profiling results dictionary
        """
        print(f"\nProfiling {name}...")

        # Memory usage before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = 0

        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Time the function
        start_time = time.time()
        try:
            func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            error = str(e)
        end_time = time.time()

        profiler.disable()

        # Memory usage after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            memory_after = 0
            peak_memory = 0

        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions
        profile_output = s.getvalue()

        result_dict = {
            "name": name,
            "success": success,
            "total_time": end_time - start_time,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "peak_memory": peak_memory,
            "memory_delta": memory_after - memory_before,
            "profile_output": profile_output[:2000],  # Limit output size
        }

        if not success:
            result_dict["error"] = error

        self.results[name] = result_dict

        if success:
            print(".4f")
            if torch.cuda.is_available():
                memory_delta = result_dict["memory_delta"]
                peak_memory = result_dict["peak_memory"]
                print(f"  Memory: {memory_delta / 1024 / 1024:.1f} MB " f"(peak: {peak_memory / 1024 / 1024:.1f} MB)")
        else:
            print(f"  FAILED: {error}")

        return result_dict

    def profile_refinement_components(self):
        """Profile individual refinement components."""
        print("=== Profiling Refinement Components ===")

        # Create test data
        patches_rgb, patches_vector = self.create_test_patch_data(num_patches=1, primitives_per_patch=10)

        # Profile rendering (used heavily in refinement)
        try:
            import torch

            from refinement.our_refinement.utils.lines_refinement_functions import (
                render_lines_with_type,
            )

            def render_test():
                lines_tensor = torch.tensor(patches_vector[0], dtype=torch.float32)
                return render_lines_with_type(lines_tensor, "bezier_splatting")

            self.profile_function_detailed(render_test, name="render_64x64_patch_bezier")

        except ImportError as e:
            print(f"  Skipping Bézier render profiling: {e}")

        # Profile GPU-accelerated rendering
        try:
            from util_files.rendering.gpu_line_renderer import GPULineRenderer

            def gpu_render_test():
                renderer = GPULineRenderer((64, 64))
                lines_tensor = torch.tensor(patches_vector[0], dtype=torch.float32)
                return renderer.render_lines(lines_tensor)

            self.profile_function_detailed(gpu_render_test, name="gpu_line_renderer_64x64")

        except Exception as e:
            print(f"  Skipping GPU renderer profiling: {e}")

        # Profile energy computation (core refinement logic)
        try:
            from refinement.our_refinement.utils.lines_refinement_functions import (
                MeanFieldEnergyComputer,
            )

            def energy_test():
                computer = MeanFieldEnergyComputer()
                # Mock the required inputs
                rendered = torch.randn(1, 64, 64)
                mask = torch.ones(1, 64, 64)
                return computer._weight_visible_excess_charge(rendered, mask)

            self.profile_function_detailed(energy_test, name="energy_computation")

        except Exception as e:
            print(f"  Skipping energy computation profiling: {e}")

    def profile_full_refinement_pipeline(self):
        """Profile the full refinement pipeline."""
        print("\n=== Profiling Full Refinement Pipeline ===")

        # Create test data
        patches_rgb, patches_vector = self.create_test_patch_data(num_patches=4, primitives_per_patch=10)

        # Mock device and options for testing
        device = torch.device("cpu")  # Use CPU for profiling

        class MockOptions:
            def __init__(self):
                self.diff_render_it = 50  # Reduced iterations for testing
                self.sample_name = "test_profile"
                self.init_random = False
                self.max_angle_to_connect = 10
                self.max_distance_to_connect = 3
                self.rendering_type = "bezier_splatting"

        options = MockOptions()

        # Profile full refinement
        try:
            from refinement.our_refinement.refinement_for_lines import (
                render_optimization_hard,
            )

            def full_refinement_test():
                return render_optimization_hard(patches_rgb, patches_vector, device, options, "test")

            result = self.profile_function_detailed(full_refinement_test, name="full_refinement_pipeline")

            # Check if we meet the target (<2s per patch)
            if result["success"]:
                time_per_patch = result["total_time"] / 4  # 4 patches
                target_time = 2.0
                if time_per_patch < target_time:
                    print(f"  ✓ MEETS TARGET: {time_per_patch:.3f}s < " f"{target_time}s per patch")
                else:
                    print(f"  ✗ EXCEEDS TARGET: {time_per_patch:.3f}s > " f"{target_time}s per patch")
                    print("    Optimization needed!")

        except Exception as e:
            print(f"  Skipping full refinement profiling: {e}")

    def identify_bottlenecks(self) -> List[str]:
        """Identify the main performance bottlenecks."""
        bottlenecks = []

        # Check rendering performance
        if "render_64x64_patch" in self.results:
            render_time = self.results["render_64x64_patch"]["total_time"]
            if render_time > 0.01:  # >10ms
                bottlenecks.append(f"Rendering: {render_time:.4f}s (target: <0.01s)")

        # Check full pipeline performance
        if "full_refinement_pipeline" in self.results:
            full_time = self.results["full_refinement_pipeline"]["total_time"]
            per_patch = full_time / 4
            if per_patch > 2.0:
                bottlenecks.append(f"Full pipeline: {per_patch:.3f}s per patch (target: <2.0s)")

        return bottlenecks

    def generate_optimization_report(self) -> str:
        """Generate a report with optimization recommendations."""
        report = "# Refinement Performance Profiling Report\n\n"

        if not self.results:
            report += "No profiling results available.\n"
            return report

        report += "## Performance Results\n\n"

        for name, data in self.results.items():
            report += f"### {name.replace('_', ' ').title()}\n\n"
            if data["success"]:
                report += f"- **Time**: {data['total_time']:.4f}s\n"
                if data.get("peak_memory", 0) > 0:
                    report += "- **Peak Memory**: " + f"{data['peak_memory'] / 1024 / 1024:.1f} MB\n"
            else:
                report += f"- **Status**: FAILED - {data.get('error', 'Unknown error')}\n"
            report += "\n"

        # Bottlenecks
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            report += "## Identified Bottlenecks\n\n"
            for bottleneck in bottlenecks:
                report += f"- {bottleneck}\n"
            report += "\n"

        # Optimization recommendations
        report += "## Optimization Recommendations\n\n"

        if "render_64x64_patch" in self.results:
            render_time = self.results["render_64x64_patch"]["total_time"]
            if render_time > 0.01:
                report += "### Rendering Optimization\n"
                report += "- Consider using Bézier splatting instead of Cairo rendering\n"
                report += "- Implement GPU-accelerated rendering\n"
                report += "- Cache rendered results for similar primitives\n\n"

        if "full_refinement_pipeline" in self.results:
            full_time = self.results["full_refinement_pipeline"]["total_time"]
            per_patch = full_time / 4
            if per_patch > 2.0:
                report += "### Pipeline Optimization\n"
                report += "- Reduce differentiable rendering iterations\n"
                report += "- Implement early stopping based on convergence\n"
                report += "- Use more efficient optimizers (Lion, Sophia)\n"
                report += "- Add gradient checkpointing for memory efficiency\n\n"

        report += "### General Recommendations\n"
        report += "- Profile with torch.profiler for GPU kernel-level optimization\n"
        report += "- Consider mixed-precision training (FP16)\n"
        report += "- Implement patch-level parallel processing\n"
        report += "- Add caching for repeated computations\n"

        return report


def main():
    """Run comprehensive refinement profiling."""
    print("DeepV Refinement Performance Profiling")
    print("=" * 50)

    profiler = RefinementProfiler()

    # Profile individual components
    profiler.profile_refinement_components()

    # Profile full pipeline
    profiler.profile_full_refinement_pipeline()

    # Generate report
    report = profiler.generate_optimization_report()

    # Save report
    output_dir = Path("logs") / "profiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "refinement_optimization_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print(f"\nReport preview:\n{report}")

    # Print bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    if bottlenecks:
        print("\n🚨 Critical Bottlenecks Identified:")
        for bottleneck in bottlenecks:
            print(f"  - {bottleneck}")
    else:
        print("\n✅ All performance targets met!")


if __name__ == "__main__":
    main()
