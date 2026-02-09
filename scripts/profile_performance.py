"""Profile refinement and rendering performance.

This script profiles the refinement and rendering components to identify
performance hotspots and bottlenecks.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from util_files.performance_profiler import PerformanceProfiler


def create_test_data():
    """Create test data for profiling."""
    # Create synthetic vector patches similar to real data
    np.random.seed(42)

    # Create 10 patches with 20 primitives each (all lines for simplicity)
    patches_vector = []
    patches_offsets = []

    for i in range(10):
        # All lines: [x1, y1, x2, y2, width, opacity]
        primitives = []
        for _ in range(20):
            primitive = np.random.rand(6) * 64  # Scale to patch size
            primitives.append(primitive)

        patches_vector.append(np.array(primitives))
        patches_offsets.append(np.array([i * 32, i * 32]))  # Overlapping patches

    return patches_vector, patches_offsets


def profile_refinement_hard(profiler):
    """Profile the hard refinement function."""
    from util_files.data.graphics_primitives import PT_LINE
    from util_files.rendering.cairo import render

    # Create simple test data for rendering (what refinement uses)
    lines = np.random.rand(50, 4) * 64
    line_widths = np.random.rand(50, 1) * 5 + 1

    test_primitives = {PT_LINE: np.concatenate((lines, line_widths), axis=1)}

    # Profile rendering (which is called during refinement)
    result = profiler.profile_rendering(lambda p: render(p, (64, 64), data_representation="vahe"), test_primitives)

    return result


def profile_rendering(profiler):
    """Profile the rendering function."""
    from util_files.data.graphics_primitives import PT_LINE, PT_QBEZIER
    from util_files.rendering.cairo import render

    # Create test primitives in the correct format
    lines = np.random.rand(50, 4) * 256  # 50 lines: [x1, y1, x2, y2]
    line_widths = np.random.rand(50, 1) * 5 + 1  # Widths with extra dimension
    line_data = np.concatenate((lines, line_widths), axis=1)

    curves = np.random.rand(30, 6) * 256  # 30 curves: [x1, y1, x2, y2, x3, y3]
    curve_widths = np.random.rand(30, 1) * 5 + 1
    curve_data = np.concatenate((curves, curve_widths), axis=1)

    test_primitives = {PT_LINE: line_data, PT_QBEZIER: curve_data}

    # Profile rendering
    result = profiler.profile_rendering(lambda p: render(p, (256, 256), data_representation="vahe"), test_primitives)

    return result


def main():
    """Run profiling analysis."""
    print("DeepV Performance Profiling")
    print("=" * 50)

    profiler = PerformanceProfiler()

    # Profile refinement (rendering component)
    print("\n1. Profiling Refinement Rendering Component")
    print("-" * 40)
    profile_refinement_hard(profiler)

    # Profile rendering
    print("\n2. Profiling Rendering")
    print("-" * 40)
    profile_rendering(profiler)

    # Generate report
    print("\n3. Generating Performance Report")
    print("-" * 40)
    report = profiler.generate_report()

    # Save report
    output_dir = Path("logs") / "profiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "performance_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print(f"\nReport content:\n{report}")

    # Print summary
    print("\n=== Performance Summary ===")
    if "rendering" in profiler.results:
        bench = profiler.results["rendering"]["benchmark"]
        print(f"Rendering: {bench['mean']:.4f}s mean ({bench['std']:.4f}s std)")


if __name__ == "__main__":
    main()
