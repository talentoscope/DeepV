#!/usr/bin/env python3
"""
DeepV Output Analysis Wrapper

Convenient wrapper script for running comprehensive DeepV output analysis and
generating compact JSON summaries. Provides simplified interface to the full
comprehensive_analysis.py functionality.

Features:
- Automatic analysis execution on pipeline outputs
- JSON summary generation for CI/CD integration
- Optional original image comparison
- Configurable device selection (CPU/GPU)
- Structured logging integration

Generates analysis_summary.json with key metrics and quality assessments.

Usage:
    python scripts/analyze_outputs.py --output_dir logs/outputs/single_test/ --original data/raw/test.png --out summary.json
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import torch

from util_files.structured_logging import get_pipeline_logger
logger = get_pipeline_logger("analysis.outputs")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive analysis and save JSON summary")
    parser.add_argument("--output_dir", required=True, help="Pipeline output directory")
    parser.add_argument("--original", required=False, help="Path to original image (optional)")
    parser.add_argument("--device", default="cpu", help="Device string for analyzer")
    parser.add_argument("--out", default="analysis_summary.json", help="Output JSON file")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    comp_path = script_dir / "comprehensive_analysis.py"
    if not comp_path.exists():
        logger.error(f"comprehensive_analysis.py not found at {comp_path}")
        raise SystemExit(1)

    logger.info(f"Loading comprehensive analyzer from {comp_path}")
    globals_dict = runpy.run_path(str(comp_path))
    if "ComprehensiveQualityAnalyzer" not in globals_dict:
        logger.error("ComprehensiveQualityAnalyzer not found in comprehensive_analysis.py")
        raise SystemExit(1)

    AnalyzerClass = globals_dict["ComprehensiveQualityAnalyzer"]
    analyzer = AnalyzerClass(device=args.device)

    logger.info("Loading outputs...")
    data = analyzer.load_outputs(args.output_dir, original_path=args.original)

    logger.info("Computing metrics...")
    summary = {}
    try:
        summary["geometric"] = analyzer.geometric_accuracy_metrics(data["original"], data["rendered"])
    except Exception as e:
        logger.warning(f"geometric metrics failed: {e}")
        summary["geometric"] = {}

    try:
        summary["visual"] = analyzer.visual_quality_metrics(data["original"], data["rendered"])
    except Exception as e:
        logger.warning(f"visual metrics failed: {e}")
        summary["visual"] = {}

    try:
        summary["structural"] = analyzer.structural_topological_metrics(data.get("vectors", []))
    except Exception as e:
        logger.warning(f"structural metrics failed: {e}")
        summary["structural"] = {}

    if data.get("ground_truth") is not None:
        try:
            summary["vector_to_vector"] = analyzer.vector_to_vector_comparison(
                data["ground_truth"], data.get("vectors", [])
            )
        except Exception as e:
            logger.warning(f"vector-to-vector comparison failed: {e}")
            summary["vector_to_vector"] = {}
    else:
        summary["vector_to_vector"] = {"note": "no ground truth available"}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Analysis summary written to {out_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
DeepV Output Quality Analysis Framework

Comprehensive analysis of vectorization pipeline outputs with multiple quality metrics.
Supports geometric accuracy, structural preservation, visual quality, and CAD-specific evaluation.

Usage:
    python scripts/analyze_outputs.py --output_dir logs/outputs/single_test/ --original data/raw/test.png
    python scripts/analyze_outputs.py --batch_dir logs/outputs/batch_results/ --save_report analysis_report.json
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import optional dependencies
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy.spatial.distance import cdist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import merging
    import refinement
    import vectorization
    from util_files import file_utils as fu
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and all dependencies are installed")
    sys.exit(1)

# Set up logging (already done above)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """Comprehensive analyzer for vectorization output quality."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        logger.info(f"Using device: {device}")

    def load_outputs(self, output_dir: str, original_path: Optional[str] = None) -> Dict[str, Any]:
        """Load pipeline outputs from directory."""
        output_path = Path(output_dir)

        results = {}

        # Load original image
        if original_path:
            orig_path = Path(original_path)
            if orig_path.exists():
                if HAS_PIL:
                    results["original"] = np.array(Image.open(orig_path).convert("L"))
                else:
                    logger.warning("PIL not available, cannot load images")
                    raise ImportError("PIL required for image loading")
            else:
                raise FileNotFoundError(f"Original image not found: {original_path}")
        else:
            # Try to find original in output directory
            orig_path = output_path / "original.png"
            if orig_path.exists():
                if HAS_PIL:
                    results["original"] = np.array(Image.open(orig_path).convert("L"))
                else:
                    logger.warning("PIL not available, cannot load images")
                    raise ImportError("PIL required for image loading")
            else:
                raise FileNotFoundError("Original image not found. Please specify --original path")

        # Load rendered output
        render_path = output_path / "final_renders" / "test.png"
        if render_path.exists():
            if HAS_PIL:
                results["rendered"] = np.array(Image.open(render_path).convert("L"))
            else:
                logger.warning("PIL not available, cannot load images")
                raise ImportError("PIL required for image loading")
        else:
            raise FileNotFoundError(f"Rendered image not found: {render_path}")

        # Load vectors
        vectors_path = output_path / "test.png.npy"
        if vectors_path.exists():
            results["vectors"] = np.load(vectors_path)
        else:
            raise FileNotFoundError(f"Vector data not found: {vectors_path}")

        # Load intermediate results if available
        for file in output_path.glob("*.npy"):
            if file.name != "test.png.npy":
                results[file.stem] = np.load(file)

        return results

    def geometric_accuracy_metrics(self, original: np.ndarray, rendered: np.ndarray) -> Dict[str, float]:
        """Calculate geometric accuracy metrics."""
        metrics = {}

        # Resize images to same dimensions for comparison
        from PIL import Image as PILImage

        orig_pil = PILImage.fromarray(original)
        render_pil = PILImage.fromarray(rendered)

        # Use the smaller dimensions
        min_width = min(orig_pil.width, render_pil.width)
        min_height = min(orig_pil.height, render_pil.height)

        orig_resized = np.array(orig_pil.resize((min_width, min_height), PILImage.LANCZOS))
        render_resized = np.array(render_pil.resize((min_width, min_height), PILImage.LANCZOS))

        # IoU (Intersection over Union)
        intersection = np.logical_and(orig_resized > 127, render_resized > 127).sum()
        union = np.logical_or(orig_resized > 127, render_resized > 127).sum()
        metrics["iou"] = intersection / union if union > 0 else 0.0

        # Hausdorff distance (approximate usingChamfer distance with sampling)
        if not HAS_SCIPY:
            logger.warning("SciPy not available, skipping Chamfer distance")
            metrics["chamfer_distance"] = float("inf")
            return metrics

        orig_points = np.array(np.where(orig_resized > 127)).T
        render_points = np.array(np.where(render_resized > 127)).T

        # Sample points to avoid memory issues
        max_points = 5000
        if len(orig_points) > max_points:
            orig_indices = np.random.choice(len(orig_points), max_points, replace=False)
            orig_points = orig_points[orig_indices]
        if len(render_points) > max_points:
            render_indices = np.random.choice(len(render_points), max_points, replace=False)
            render_points = render_points[render_indices]

        if len(orig_points) > 0 and len(render_points) > 0:
            # Chamfer distance as proxy for Hausdorff
            dist_matrix = cdist(orig_points, render_points)
            chamfer_orig_to_render = np.mean(np.min(dist_matrix, axis=1))
            chamfer_render_to_orig = np.mean(np.min(dist_matrix, axis=0))
            metrics["chamfer_distance"] = (chamfer_orig_to_render + chamfer_render_to_orig) / 2
        else:
            metrics["chamfer_distance"] = float("inf")

        return metrics

    def structural_preservation_metrics(self, vectors: np.ndarray, original: np.ndarray) -> Dict[str, Any]:
        """Analyze structural preservation of primitives."""
        metrics = {}

        if len(vectors) == 0:
            return {"error": "No vectors found"}

        # Basic statistics
        metrics["primitive_count"] = len(vectors)
        metrics["width_stats"] = {
            "mean": float(vectors[:, 4].mean()),
            "std": float(vectors[:, 4].std()),
            "min": float(vectors[:, 4].min()),
            "max": float(vectors[:, 4].max()),
        }
        metrics["probability_stats"] = {
            "mean": float(vectors[:, 5].mean()),
            "std": float(vectors[:, 5].std()),
            "min": float(vectors[:, 5].min()),
            "max": float(vectors[:, 5].max()),
        }

        # Line length analysis
        lengths = np.sqrt((vectors[:, 2] - vectors[:, 0]) ** 2 + (vectors[:, 3] - vectors[:, 1]) ** 2)
        metrics["length_stats"] = {
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
            "min": float(lengths.min()),
            "max": float(lengths.max()),
        }

        # Angle analysis (horizontal/vertical detection)
        angles = np.arctan2(vectors[:, 3] - vectors[:, 1], vectors[:, 2] - vectors[:, 0])
        angles_deg = np.rad2deg(angles) % 180  # Normalize to 0-180

        # Detect axis-aligned lines (within 5 degrees)
        axis_aligned = np.abs(np.abs(angles_deg - 90) - 90) < 5
        metrics["axis_aligned_ratio"] = float(axis_aligned.mean())

        # Parallelism analysis (simplified)
        if len(vectors) > 1:
            angle_diffs = cdist(
                angles.reshape(-1, 1), angles.reshape(-1, 1), lambda u, v: min(abs(u[0] - v[0]), 180 - abs(u[0] - v[0]))
            )
            parallel_pairs = (angle_diffs < 5).sum() - len(vectors)  # Subtract self-pairs
            metrics["parallel_ratio"] = parallel_pairs / (len(vectors) * (len(vectors) - 1))
        else:
            metrics["parallel_ratio"] = 0.0

        return metrics

    def visual_quality_metrics(self, original: np.ndarray, rendered: np.ndarray) -> Dict[str, float]:
        """Calculate visual quality metrics."""
        metrics = {}

        if not HAS_PIL:
            logger.warning("PIL not available, skipping visual quality metrics")
            metrics["ssim"] = 0.0
            metrics["psnr"] = 0.0
            metrics["mse"] = float("inf")
            return metrics

        # Resize images to same dimensions for comparison
        from PIL import Image as PILImage

        orig_pil = PILImage.fromarray(original)
        render_pil = PILImage.fromarray(rendered)

        # Use the smaller dimensions
        min_width = min(orig_pil.width, render_pil.width)
        min_height = min(orig_pil.height, render_pil.height)

        orig_resized = np.array(orig_pil.resize((min_width, min_height), PILImage.LANCZOS))
        render_resized = np.array(render_pil.resize((min_width, min_height), PILImage.LANCZOS))

        # SSIM (Structural Similarity Index)
        if HAS_SKIMAGE:
            try:
                metrics["ssim"], _ = ssim(orig_resized, render_resized, full=True, data_range=255)
            except Exception as e:
                logger.warning(f"SSIM calculation failed: {e}")
                metrics["ssim"] = 0.0

            # PSNR (Peak Signal-to-Noise Ratio)
            try:
                metrics["psnr"] = psnr(orig_resized, render_resized, data_range=255)
            except Exception as e:
                logger.warning(f"PSNR calculation failed: {e}")
                metrics["psnr"] = 0.0
        else:
            logger.warning("scikit-image not available, skipping SSIM and PSNR")
            metrics["ssim"] = 0.0
            metrics["psnr"] = 0.0

        # MSE (Mean Squared Error)
        mse = np.mean((orig_resized.astype(float) - render_resized.astype(float)) ** 2)
        metrics["mse"] = float(mse)

        # Simple perceptual metrics
        diff = np.abs(orig_resized.astype(float) - render_resized.astype(float))
        metrics["mean_absolute_error"] = float(np.mean(diff))
        metrics["max_absolute_error"] = float(np.max(diff))

        return metrics

    def cad_specific_metrics(self, vectors: np.ndarray) -> Dict[str, Any]:
        """CAD-specific quality metrics."""
        metrics = {}

        if len(vectors) == 0:
            return {"error": "No vectors found"}

        # Detect potential CAD constraints violations
        angles = np.arctan2(vectors[:, 3] - vectors[:, 1], vectors[:, 2] - vectors[:, 0])
        angles_deg = np.rad2deg(angles) % 180

        # Check for common CAD angles (0, 30, 45, 60, 90 degrees)
        cad_angles = [0, 30, 45, 60, 90, 120, 135, 150]
        cad_compliance = np.zeros(len(angles_deg))

        for cad_angle in cad_angles:
            cad_compliance = np.maximum(cad_compliance, 1 - np.abs(angles_deg - cad_angle) / 5)

        metrics["cad_angle_compliance"] = float(cad_compliance.mean())

        # Length ratios (detect equal lengths, common ratios)
        lengths = np.sqrt((vectors[:, 2] - vectors[:, 0]) ** 2 + (vectors[:, 3] - vectors[:, 1]) ** 2)
        if len(lengths) > 1:
            length_ratios = cdist(lengths.reshape(-1, 1), lengths.reshape(-1, 1))
            equal_length_pairs = (np.abs(length_ratios - np.diag(lengths)) < 2).sum() - len(lengths)
            metrics["equal_length_ratio"] = equal_length_pairs / (len(lengths) * (len(lengths) - 1))
        else:
            metrics["equal_length_ratio"] = 0.0

        # Endpoint connectivity (simplified)
        endpoints = vectors[:, :4].reshape(-1, 2)  # All endpoints
        if len(endpoints) > 0:
            endpoint_distances = cdist(endpoints, endpoints)
            connected_endpoints = (endpoint_distances < 3).sum() - len(endpoints)  # Subtract self-distances
            metrics["endpoint_connectivity"] = connected_endpoints / (len(endpoints) * (len(endpoints) - 1))
        else:
            metrics["endpoint_connectivity"] = 0.0

        return metrics

    def analyze_output_quality(self, output_dir: str, original_path: Optional[str] = None) -> Dict[str, Any]:
        """Run complete quality analysis on pipeline outputs."""
        logger.info(f"Analyzing outputs from: {output_dir}")

        # Load outputs
        outputs = self.load_outputs(output_dir, original_path)

        if "original" not in outputs or "rendered" not in outputs:
            raise ValueError("Missing original or rendered images in output directory")

        if "vectors" not in outputs:
            raise ValueError("Missing vector data in output directory")

        analysis = {
            "timestamp": str(np.datetime64("now")),
            "output_dir": output_dir,
            "image_dimensions": {"original": outputs["original"].shape, "rendered": outputs["rendered"].shape},
        }

        # Geometric accuracy
        logger.info("Calculating geometric accuracy metrics...")
        analysis["geometric_accuracy"] = self.geometric_accuracy_metrics(outputs["original"], outputs["rendered"])

        # Structural preservation
        logger.info("Analyzing structural preservation...")
        analysis["structural_preservation"] = self.structural_preservation_metrics(
            outputs["vectors"], outputs["original"]
        )

        # Visual quality
        logger.info("Calculating visual quality metrics...")
        analysis["visual_quality"] = self.visual_quality_metrics(outputs["original"], outputs["rendered"])

        # CAD-specific metrics
        logger.info("Analyzing CAD-specific metrics...")
        analysis["cad_specific"] = self.cad_specific_metrics(outputs["vectors"])

        # Overall quality score (weighted combination)
        analysis["overall_score"] = self.compute_overall_score(analysis)

        logger.info("Analysis complete!")
        return analysis

    def compute_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Compute overall quality score from individual metrics."""
        try:
            geo_score = analysis["geometric_accuracy"].get("iou", 0) * 0.4
            visual_score = (
                analysis["visual_quality"].get("ssim", 0) * 0.3
                + min(analysis["visual_quality"].get("psnr", 0) / 50, 1) * 0.2
            )
            struct_score = (
                analysis["structural_preservation"].get("axis_aligned_ratio", 0) * 0.05
                + analysis["structural_preservation"].get("parallel_ratio", 0) * 0.05
            )

            return geo_score + visual_score + struct_score
        except KeyError:
            return 0.0

    def save_report(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Report saved to: {output_path}")

    def print_summary(self, analysis: Dict[str, Any]):
        """Print human-readable summary of analysis."""
        print("\n" + "=" * 60)
        print("DEEPV OUTPUT QUALITY ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"Output Directory: {analysis['output_dir']}")
        print(f"Image Dimensions: {analysis['image_dimensions']}")

        print(f"\nOverall Quality Score: {analysis.get('overall_score', 0):.3f}")

        print("\nGEOMETRIC ACCURACY:")
        geo = analysis.get("geometric_accuracy", {})
        print(".3f")
        print(".2f")

        print("\nVISUAL QUALITY:")
        vis = analysis.get("visual_quality", {})
        print(".3f")
        print(".2f")
        print(".2f")

        print("\nSTRUCTURAL PRESERVATION:")
        struct = analysis.get("structural_preservation", {})
        print(f"  Primitives: {struct.get('primitive_count', 0)}")
        print(".2f")
        print(".2f")
        print(".3f")
        print(".3f")

        print("\nCAD-SPECIFIC METRICS:")
        cad = analysis.get("cad_specific", {})
        print(".3f")
        print(".3f")
        print(".3f")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze DeepV pipeline output quality")
    parser.add_argument("--output_dir", required=True, help="Directory containing pipeline outputs")
    parser.add_argument("--original", help="Path to original image (if not in output_dir)")
    parser.add_argument("--save_report", help="Save analysis report to JSON file")
    parser.add_argument("--batch_dir", help="Analyze all subdirectories in batch mode")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    analyzer = QualityAnalyzer()

    if args.batch_dir:
        # Batch analysis mode
        batch_path = Path(args.batch_dir)
        results = {}

        for subdir in batch_path.iterdir():
            if subdir.is_dir():
                try:
                    logger.info(f"Analyzing {subdir.name}...")
                    analysis = analyzer.analyze_output_quality(str(subdir), args.original)
                    results[subdir.name] = analysis

                    if not args.quiet:
                        analyzer.print_summary(analysis)

                except Exception as e:
                    logger.error(f"Failed to analyze {subdir.name}: {e}")
                    results[subdir.name] = {"error": str(e)}

        if args.save_report:
            analyzer.save_report(results, args.save_report)
            print(f"\nBatch report saved to: {args.save_report}")

    else:
        # Single analysis mode
        try:
            analysis = analyzer.analyze_output_quality(args.output_dir, args.original)

            if not args.quiet:
                analyzer.print_summary(analysis)

            if args.save_report:
                analyzer.save_report(analysis, args.save_report)

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
