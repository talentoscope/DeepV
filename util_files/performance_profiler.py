"""Profiling utilities for DeepV performance analysis.

This module provides profiling tools to identify performance hotspots
in the refinement and rendering pipeline.
"""

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function


@contextmanager
def time_block(name: str):
    """Context manager for timing code blocks."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{name}: {end_time - start_time:.4f}s")


def profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile a function using cProfile and return statistics."""
    pr = cProfile.Profile()
    pr.enable()

    result = func(*args, **kwargs)

    pr.disable()

    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()

    return {"result": result, "profile_stats": s.getvalue(), "profiler": pr}


def torch_profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile a function using PyTorch profiler."""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            result = func(*args, **kwargs)

    # Print profiler results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return {"result": result, "profiler": prof}


def benchmark_function(
    func: Callable, *args, num_runs: int = 10, warmup_runs: int = 2, **kwargs
) -> Dict[str, float]:
    """Benchmark a function's performance over multiple runs."""
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)

    times = np.array(times)
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
        "num_runs": num_runs,
    }


def memory_profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile memory usage of a function."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory profiling")
        return {"result": func(*args, **kwargs), "memory_stats": None}

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    start_memory = torch.cuda.memory_allocated()

    result = func(*args, **kwargs)

    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()

    return {
        "result": result,
        "memory_stats": {
            "start_memory_mb": start_memory / 1024 / 1024,
            "end_memory_mb": end_memory / 1024 / 1024,
            "peak_memory_mb": peak_memory / 1024 / 1024,
            "memory_increase_mb": (end_memory - start_memory) / 1024 / 1024,
        },
    }


class PerformanceProfiler:
    """Comprehensive performance profiler for DeepV components."""

    def __init__(self):
        self.results = {}

    def profile_refinement(
        self, refinement_func: Callable, test_data: Any
    ) -> Dict[str, Any]:
        """Profile refinement function performance."""
        print("=== Profiling Refinement Function ===")

        # Time profiling
        with time_block("refinement_timing"):
            result = refinement_func(test_data)

        # CPU profiling
        print("\n=== CPU Profiling ===")
        profile_result = profile_function(refinement_func, test_data)

        # Benchmarking
        print("\n=== Benchmarking (10 runs) ===")
        benchmark_result = benchmark_function(refinement_func, test_data, num_runs=10)

        # Memory profiling (if CUDA available)
        print("\n=== Memory Profiling ===")
        memory_result = memory_profile_function(refinement_func, test_data)

        combined_result = {
            "result": result,
            "cpu_profile": profile_result,
            "benchmark": benchmark_result,
            "memory": memory_result,
        }

        self.results["refinement"] = combined_result
        return combined_result

    def profile_rendering(
        self, render_func: Callable, test_primitives: Any
    ) -> Dict[str, Any]:
        """Profile rendering function performance."""
        print("=== Profiling Rendering Function ===")

        # Time profiling
        with time_block("rendering_timing"):
            result = render_func(test_primitives)

        # CPU profiling
        print("\n=== CPU Profiling ===")
        profile_result = profile_function(render_func, test_primitives)

        # Benchmarking
        print("\n=== Benchmarking (10 runs) ===")
        benchmark_result = benchmark_function(render_func, test_primitives, num_runs=10)

        combined_result = {
            "result": result,
            "cpu_profile": profile_result,
            "benchmark": benchmark_result,
        }

        self.results["rendering"] = combined_result
        return combined_result

    def generate_report(self) -> str:
        """Generate a performance report."""
        report = ["# DeepV Performance Profiling Report\n"]

        for component, results in self.results.items():
            report.append(f"## {component.title()} Performance\n")

            if "benchmark" in results:
                bench = results["benchmark"]
                report.append("### Benchmark Results\n")
                report.append(f"- Mean: {bench['mean']:.4f}s\n")
                report.append(f"- Std: {bench['std']:.4f}s\n")
                report.append(f"- Min: {bench['min']:.4f}s\n")
                report.append(f"- Max: {bench['max']:.4f}s\n")
                report.append(f"- Median: {bench['median']:.4f}s\n")

            if "memory" in results and results["memory"]["memory_stats"]:
                mem = results["memory"]["memory_stats"]
                report.append("### Memory Usage\n")
                report.append(f"- Start: {mem['start_memory_mb']:.2f} MB\n")
                report.append(f"- End: {mem['end_memory_mb']:.2f} MB\n")
                report.append(f"- Peak: {mem['peak_memory_mb']:.2f} MB\n")
                report.append(f"- Increase: {mem['memory_increase_mb']:.2f} MB\n")
        return "\n".join(report)
