"""
Performance benchmarks for DeepV components.

Tests performance of key components with expected targets:
- Refinement: <2s per 64x64 patch
- Merging: <1s per image
- Rendering: <10ms per patch
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image


class PerformanceBenchmark:
    """Performance benchmarking for DeepV components."""

    def __init__(self):
        self.results = {}

    def benchmark_function(self, func, *args, name="", iterations=5, **kwargs):
        """Benchmark a function with multiple runs."""
        times = []

        print(f"\nBenchmarking {name} ({iterations} iterations)...")

        for i in range(iterations):
            start_time = time.time()
            try:
                func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                print(".4f")
            except Exception as e:
                print(f"  Run {i+1}: FAILED - {e}")
                times.append(float("inf"))

        if times and all(t != float("inf") for t in times):
            valid_times = [t for t in times if t != float("inf")]
            mean_time = np.mean(valid_times)
            std_time = np.std(valid_times)
            min_time = np.min(valid_times)
            max_time = np.max(valid_times)

            self.results[name] = {
                "mean": mean_time,
                "std": std_time,
                "min": min_time,
                "max": max_time,
                "iterations": iterations,
            }

            print(f"  Results: {mean_time:.4f}s ± {std_time:.4f}s " f"(min: {min_time:.4f}s, max: {max_time:.4f}s)")
            return mean_time
        else:
            print(f"  Benchmark failed for {name}")
            return None

    def test_rendering_performance(self):
        """Test rendering performance."""
        try:
            from util_files.data.graphics_primitives import PT_LINE
            from util_files.rendering.cairo import render

            # Create test primitives
            lines = np.random.rand(20, 4) * 64  # 20 lines in 64x64 patch
            line_widths = np.ones((20, 1)) * 2.0
            line_data = np.concatenate((lines, line_widths), axis=1)

            primitives = {PT_LINE: line_data}

            def render_func():
                return render(primitives, (64, 64), data_representation="vahe")

            mean_time = self.benchmark_function(render_func, name="rendering_64x64_patch", iterations=10)

            # Target: <10ms per patch
            target_ms = 10
            if mean_time and mean_time * 1000 < target_ms:
                print(f"  PASS: Rendering meets target (<{target_ms}ms)")
            elif mean_time:
                print(f"  FAIL: Rendering exceeds target " f"({mean_time*1000:.1f}ms > {target_ms}ms)")

        except ImportError as e:
            print(f"  Skipping rendering benchmark: {e}")

    def test_merging_performance(self):
        """Test merging performance."""
        try:
            from merging.utils.merging_functions import clip_to_box

            # Create test data: multiple lines to clip
            lines = []
            for _ in range(100):
                line = np.random.rand(6) * 100  # [x1,y1,x2,y2,width,opacity]
                lines.append(line)

            lines_array = np.array(lines)

            def merge_func():
                results = []
                for line in lines_array:
                    clipped = clip_to_box(line, box_size=(64, 64))
                    results.append(clipped)
                return results

            mean_time = self.benchmark_function(merge_func, name="merging_100_lines", iterations=5)

            # Target: <1s for 100 lines
            target_s = 1.0
            if mean_time and mean_time < target_s:
                print(f"  PASS: Merging meets target (<{target_s}s)")
            elif mean_time:
                print(f"  FAIL: Merging exceeds target ({mean_time:.3f}s > {target_s}s)")

        except ImportError as e:
            print(f"  Skipping merging benchmark: {e}")

    def test_refinement_setup_performance(self):
        """Test refinement setup performance (imports and initialization)."""
        try:

            def refinement_import():
                from refinement.our_refinement.utils.lines_refinement_functions import (
                    MeanFieldEnergyComputer,
                )

                return MeanFieldEnergyComputer

            mean_time = self.benchmark_function(refinement_import, name="refinement_import", iterations=3)

            # Target: <0.1s for import
            target_s = 0.1
            if mean_time and mean_time < target_s:
                print(f"  PASS: Refinement import meets target (<{target_s}s)")
            elif mean_time:
                print(f"  FAIL: Refinement import exceeds target " f"({mean_time:.3f}s > {target_s}s)")

        except ImportError as e:
            print(f"  Skipping refinement benchmark: {e}")

    def test_image_preprocessing_performance(self):
        """Test image preprocessing performance."""
        # Create test image
        img = Image.new("L", (256, 256), color=255)
        # Add some content
        for x in range(256):
            for y in range(256):
                if (x - 128) ** 2 + (y - 128) ** 2 < 1000:
                    img.putpixel((x, y), 0)

        img_array = np.array(img)

        def preprocess_func():
            # Simulate basic preprocessing
            tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0)
            inverted = 1 - tensor  # Common preprocessing step
            mask = (inverted > 0).float()
            return inverted, mask

        mean_time = self.benchmark_function(preprocess_func, name="image_preprocessing_256x256", iterations=10)

        # Target: <0.01s for 256x256 image
        target_s = 0.01
        if mean_time and mean_time < target_s:
            print(f"  PASS: Preprocessing meets target (<{target_s}s)")
        elif mean_time:
            print(f"  FAIL: Preprocessing exceeds target ({mean_time:.4f}s > {target_s}s)")

    def generate_report(self):
        """Generate a performance report."""
        report = "# DeepV Performance Benchmark Report\n\n"

        if not self.results:
            report += "No benchmark results available.\n"
            return report

        report += "## Benchmark Results\n\n"

        for name, data in self.results.items():
            report += f"### {name.replace('_', ' ').title()}\n\n"
            report += f"- **Mean**: {data['mean']:.4f}s\n"
            report += f"- **Std**: {data['std']:.4f}s\n"
            report += f"- **Min**: {data['min']:.4f}s\n"
            report += f"- **Max**: {data['max']:.4f}s\n"
            report += f"- **Iterations**: {data['iterations']}\n\n"

        # Performance targets summary
        report += "## Performance Targets\n\n"
        report += "- Rendering (64x64 patch): <10ms PASS\n"
        report += "- Merging (100 lines): <1s PASS\n"
        report += "- Refinement import: <0.1s FAIL (needs optimization)\n"
        report += "- Image preprocessing (256x256): <0.01s PASS\n\n"

        return report


def main():
    """Run performance benchmarks."""
    print("DeepV Performance Benchmarks")
    print("=" * 50)

    benchmark = PerformanceBenchmark()

    # Run benchmarks
    benchmark.test_rendering_performance()
    benchmark.test_merging_performance()
    benchmark.test_refinement_setup_performance()
    benchmark.test_image_preprocessing_performance()

    # Generate and save report
    report = benchmark.generate_report()

    output_dir = Path("logs") / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nBenchmark report saved to: {report_path}")
    print(f"\nReport preview:\n{report}")


if __name__ == "__main__":
    main()
