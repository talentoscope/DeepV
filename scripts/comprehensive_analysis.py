#!/usr/bin/env python3
"""
Comprehensive DeepV Output Quality Analysis Framework

Computes extensive metrics for vectorization quality assessment including:
- Geometric accuracy (IoU, Dice, Chamfer distance)
- Structural/topological metrics (lengths, angles, parallelism)
- Visual quality (SSIM, PSNR, MSE, MAE)
- CAD-specific metrics (angle compliance, connectivity, equal lengths)
- Statistical distributions (widths, probabilities, angles)
- Error pattern analysis

Usage:
    python scripts/comprehensive_analysis.py --output_dir logs/outputs/single_test/ --original data/raw/test.png
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

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
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveQualityAnalyzer:
    """Comprehensive analyzer for vectorization output quality with extensive metrics."""

    def __init__(self, device: str = 'cpu'):
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
                    results['original'] = np.array(Image.open(orig_path).convert('L'))
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
                    results['original'] = np.array(Image.open(orig_path).convert('L'))
                else:
                    logger.warning("PIL not available, cannot load images")
                    raise ImportError("PIL required for image loading")
            else:
                raise FileNotFoundError("Original image not found. Please specify --original path")

        # Load rendered output
        # Try multiple possible locations for rendered image
        render_paths = [
            output_path / "final_renders" / "test.png",
            output_path / "final_renders" / "0490-0079.png",  # Specific to our test
            output_path / "0490-0079.png",  # Direct in output dir
        ]
        
        rendered_found = False
        for render_path in render_paths:
            if render_path.exists():
                if HAS_PIL:
                    results['rendered'] = np.array(Image.open(render_path).convert('L'))
                    rendered_found = True
                    break
        
        if not rendered_found:
            raise FileNotFoundError(f"Rendered image not found in any expected location")

        # Load vectors
        # Try multiple possible locations for vector data
        vector_paths = [
            Path(output_dir.replace('merging_output', 'arrays')) / "hard_optimization_iou_mass_0490-0079.png.npy",
            output_path / "0490-0079.png.npy",
            output_path / "test.png.npy",
            Path(output_dir.replace('merging_output', 'arrays')) / "hard_optimization_iou_0490-0079.png.npy",
        ]
        
        vectors_found = False
        for vectors_path in vector_paths:
            if vectors_path.exists():
                vectors_data = np.load(vectors_path)
                print(f"DEBUG: Loaded vectors from {vectors_path} with shape {vectors_data.shape}")
                # Reshape from (batch, patches, primitives, features) to (total_primitives, features)
                if vectors_data.ndim == 4:
                    results['vectors'] = vectors_data.reshape(-1, vectors_data.shape[-1])
                    print(f"DEBUG: Reshaped to {results['vectors'].shape}")
                else:
                    results['vectors'] = vectors_data
                vectors_found = True
                break
        
        if not vectors_found:
            raise FileNotFoundError(f"Vector data not found in any expected location")

        # Load ground truth vectors if available
        # Try to find ground truth based on the image name
        gt_paths = []
        for vectors_path in vector_paths:
            if vectors_path.exists():
                # Extract base name (e.g., 0490-0079 from 0490-0079.png.npy)
                base_name = vectors_path.stem.replace('.png', '').replace('.npy', '')
                gt_path = Path(f"data/raw/floorplancad/test/{base_name}_gt.npy")
                gt_paths.append(gt_path)
        
        gt_found = False
        for gt_path in gt_paths:
            if gt_path.exists():
                results['ground_truth'] = np.load(gt_path)
                print(f"DEBUG: Loaded ground truth from {gt_path} with shape {results['ground_truth'].shape}")
                gt_found = True
                break
        
        if not gt_found:
            print("DEBUG: No ground truth found, analysis will compare outputs against themselves")
            results['ground_truth'] = None

        return results

    def vector_to_vector_comparison(self, ground_truth: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
        """Compare ground truth vectors against predicted vectors."""
        metrics = {}
        
        # Basic counts
        metrics['ground_truth_count'] = len(ground_truth)
        metrics['predicted_count'] = len(predicted)
        
        # Extract coordinates (assuming format: x1, y1, x2, y2, stroke_width, confidence)
        gt_coords = ground_truth[:, :4]  # x1, y1, x2, y2
        pred_coords = predicted[:, :4]
        
        # Chamfer distance between vector endpoints
        if HAS_SCIPY and len(gt_coords) > 0 and len(pred_coords) > 0:
            # Create point sets from line endpoints
            gt_points = np.vstack([gt_coords[:, :2], gt_coords[:, 2:]])  # All endpoints
            pred_points = np.vstack([pred_coords[:, :2], pred_coords[:, 2:]])
            
            # Remove duplicates
            gt_points = np.unique(gt_points, axis=0)
            pred_points = np.unique(pred_points, axis=0)
            
            # Chamfer distance
            dist_gt_to_pred = cdist(gt_points, pred_points).min(axis=1)
            dist_pred_to_gt = cdist(pred_points, gt_points).min(axis=1)
            
            metrics['chamfer_distance'] = (dist_gt_to_pred.mean() + dist_pred_to_gt.mean()) / 2
            metrics['max_distance'] = max(dist_gt_to_pred.max(), dist_pred_to_gt.max())
        else:
            metrics['chamfer_distance'] = float('inf')
            metrics['max_distance'] = float('inf')
        
        # Stroke width comparison
        if len(ground_truth) > 0 and len(predicted) > 0:
            gt_widths = ground_truth[:, 4]
            pred_widths = predicted[:, 4]
            
            metrics['mean_stroke_width_diff'] = abs(gt_widths.mean() - pred_widths.mean())
            metrics['stroke_width_mae'] = np.mean(np.abs(gt_widths - pred_widths[:len(gt_widths)])) if len(pred_widths) >= len(gt_widths) else float('inf')
        
        # Length comparison
        gt_lengths = np.sqrt((gt_coords[:, 2] - gt_coords[:, 0])**2 + (gt_coords[:, 3] - gt_coords[:, 1])**2)
        pred_lengths = np.sqrt((pred_coords[:, 2] - pred_coords[:, 0])**2 + (pred_coords[:, 3] - pred_coords[:, 1])**2)
        
        if len(gt_lengths) > 0 and len(pred_lengths) > 0:
            metrics['mean_length_diff'] = abs(gt_lengths.mean() - pred_lengths.mean())
            metrics['length_mae'] = np.mean(np.abs(gt_lengths - pred_lengths[:len(gt_lengths)])) if len(pred_lengths) >= len(gt_lengths) else float('inf')
        
        return metrics

    def geometric_accuracy_metrics(self, original: np.ndarray, rendered: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive geometric accuracy metrics."""
        metrics = {}

        if not HAS_PIL:
            logger.warning("PIL not available, skipping geometric metrics")
            metrics['iou'] = 0.0
            metrics['dice'] = 0.0
            metrics['chamfer_distance'] = float('inf')
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

        # IoU (Intersection over Union)
        intersection = np.logical_and(orig_resized > 127, render_resized > 127).sum()
        union = np.logical_or(orig_resized > 127, render_resized > 127).sum()
        metrics['iou'] = intersection / union if union > 0 else 0.0

        # Dice Coefficient
        orig_sum = np.sum(orig_resized > 127)
        render_sum = np.sum(render_resized > 127)
        metrics['dice'] = 2 * intersection / (orig_sum + render_sum) if (orig_sum + render_sum) > 0 else 0.0

        # Chamfer distance approximation
        if not HAS_SCIPY:
            logger.warning("SciPy not available, skipping Chamfer distance")
            metrics['chamfer_distance'] = float('inf')
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
            metrics['chamfer_distance'] = (chamfer_orig_to_render + chamfer_render_to_orig) / 2
        else:
            metrics['chamfer_distance'] = float('inf')

        return metrics

    def structural_topological_metrics(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Analyze structural and topological properties of primitives."""
        metrics = {}

        if len(vectors) == 0:
            return {'error': 'No vectors found'}

        # Sample a subset for computational efficiency
        max_primitives = 1000
        if len(vectors) > max_primitives:
            indices = np.random.choice(len(vectors), max_primitives, replace=False)
            vectors_sample = vectors[indices]
        else:
            vectors_sample = vectors

        metrics['primitive_count'] = len(vectors_sample)

        # Line lengths analysis (assuming format: [type, x1, y1, x2, y2])
        lengths = np.sqrt((vectors_sample[:, 3] - vectors_sample[:, 1])**2 + (vectors_sample[:, 4] - vectors_sample[:, 2])**2)
        metrics['length_stats'] = {
            'mean': float(lengths.mean()),
            'std': float(lengths.std()),
            'min': float(lengths.min()),
            'max': float(lengths.max()),
            'median': float(np.median(lengths))
        }

        # Angle analysis
        angles = np.arctan2(vectors_sample[:, 4] - vectors_sample[:, 2], vectors_sample[:, 3] - vectors_sample[:, 1])
        angles_deg = np.rad2deg(angles) % 180

        metrics['angle_stats'] = {
            'mean': float(angles_deg.mean()),
            'std': float(angles_deg.std()),
            'min': float(angles_deg.min()),
            'max': float(angles_deg.max()),
            'median': float(np.median(angles_deg))
        }

        # CAD angle compliance (0, 30, 45, 60, 90, 120, 135, 150 degrees)
        cad_angles = [0, 30, 45, 60, 90, 120, 135, 150]
        cad_compliance = np.zeros(len(angles_deg))
        for cad_angle in cad_angles:
            cad_compliance = np.maximum(cad_compliance, 1 - np.abs(angles_deg - cad_angle) / 5)

        metrics['cad_angle_compliance'] = {
            'ratio': float(np.mean(cad_compliance > 0.8)),  # Within 4 degrees
            'count': int(np.sum(cad_compliance > 0.8)),
            'total': len(vectors)
        }

        # Axis-aligned detection (horizontal/vertical within 5 degrees)
        axis_aligned = np.abs(np.abs(angles_deg - 90) - 90) < 5
        metrics['axis_aligned_ratio'] = float(axis_aligned.mean())
        metrics['axis_aligned_count'] = int(np.sum(axis_aligned))

        # Parallelism analysis
        if len(vectors_sample) > 1 and HAS_SCIPY:
            angle_diffs = cdist(angles.reshape(-1, 1), angles.reshape(-1, 1),
                               lambda u, v: min(abs(u[0]-v[0]), np.pi-abs(u[0]-v[0])))
            parallel_pairs = (angle_diffs < np.deg2rad(5)).sum() - len(vectors_sample)  # Subtract self-pairs
            metrics['parallel_ratio'] = parallel_pairs / (len(vectors_sample) * (len(vectors_sample) - 1))
        else:
            metrics['parallel_ratio'] = 0.0

        return metrics

    def visual_quality_metrics(self, original: np.ndarray, rendered: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive visual quality metrics."""
        metrics = {}

        if not HAS_PIL:
            logger.warning("PIL not available, skipping visual quality metrics")
            metrics['mse'] = float('inf')
            metrics['mae'] = float('inf')
            metrics['ssim'] = 0.0
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

        # Basic error metrics
        mse = np.mean((orig_resized.astype(float) - render_resized.astype(float)) ** 2)
        mae = np.mean(np.abs(orig_resized.astype(float) - render_resized.astype(float)))
        metrics['mse'] = float(mse)
        metrics['mae'] = float(mae)
        metrics['rmse'] = float(np.sqrt(mse))
        metrics['max_absolute_error'] = float(np.max(np.abs(orig_resized.astype(float) - render_resized.astype(float))))

        # SSIM (Structural Similarity Index)
        if HAS_SKIMAGE:
            try:
                metrics['ssim'], _ = ssim(orig_resized, render_resized, full=True, data_range=255)
            except Exception as e:
                logger.warning(f"SSIM calculation failed: {e}")
                metrics['ssim'] = 0.0

            # PSNR (Peak Signal-to-Noise Ratio)
            try:
                metrics['psnr'] = psnr(orig_resized, render_resized, data_range=255)
            except Exception as e:
                logger.warning(f"PSNR calculation failed: {e}")
                metrics['psnr'] = 0.0
        else:
            # Simple SSIM approximation
            def simple_ssim(img1, img2):
                C1 = (0.01 * 255)**2
                C2 = (0.03 * 255)**2

                img1 = img1.astype(float)
                img2 = img2.astype(float)

                mu1 = np.mean(img1)
                mu2 = np.mean(img2)
                sigma1 = np.var(img1)
                sigma2 = np.var(img2)
                sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]

                numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
                denominator = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)

                return numerator / denominator if denominator > 0 else 0.0

            metrics['ssim'] = simple_ssim(orig_resized, render_resized)
            # Simple PSNR approximation
            if mse > 0:
                metrics['psnr'] = 20 * np.log10(255 / np.sqrt(mse))
            else:
                metrics['psnr'] = float('inf')

        return metrics

    def cad_specific_metrics(self, vectors: np.ndarray) -> Dict[str, Any]:
        """CAD-specific quality metrics."""
        metrics = {}

        if len(vectors) == 0:
            return {'error': 'No vectors found'}

        # Sample a subset for computational efficiency
        max_primitives = 1000
        if len(vectors) > max_primitives:
            indices = np.random.choice(len(vectors), max_primitives, replace=False)
            vectors_sample = vectors[indices]
        else:
            vectors_sample = vectors

        # Equal length detection (assuming format: [type, x1, y1, x2, y2])
        lengths = np.sqrt((vectors_sample[:, 3] - vectors_sample[:, 1])**2 + (vectors_sample[:, 4] - vectors_sample[:, 2])**2)
        if len(lengths) > 1 and HAS_SCIPY:
            length_diffs = cdist(lengths.reshape(-1, 1), lengths.reshape(-1, 1))
            equal_length_pairs = (length_diffs < 2).sum() - len(lengths)  # Subtract self-pairs
            metrics['equal_length_ratio'] = equal_length_pairs / (len(lengths) * (len(lengths) - 1))
        else:
            metrics['equal_length_ratio'] = 0.0

        # Endpoint connectivity analysis
        endpoints = vectors_sample[:, 1:5].reshape(-1, 2)  # All endpoints
        if len(endpoints) > 0 and HAS_SCIPY:
            endpoint_distances = cdist(endpoints, endpoints)
            connected_endpoints = (endpoint_distances < 3).sum() - len(endpoints)  # Subtract self-distances
            metrics['endpoint_connectivity'] = connected_endpoints / (len(endpoints) * (len(endpoints) - 1))
        else:
            metrics['endpoint_connectivity'] = 0.0

        # Perpendicularity detection
        if len(vectors_sample) > 1 and HAS_SCIPY:
            angles = np.arctan2(vectors_sample[:, 4] - vectors_sample[:, 2], vectors_sample[:, 3] - vectors_sample[:, 1])
            angle_diffs = cdist(angles.reshape(-1, 1), angles.reshape(-1, 1),
                               lambda u, v: min(abs(u[0]-v[0]), np.pi-abs(u[0]-v[0])))
            perpendicular_pairs = (np.abs(angle_diffs - np.pi/2) < np.deg2rad(5)).sum() - len(vectors_sample)
            metrics['perpendicular_ratio'] = perpendicular_pairs / (len(vectors_sample) * (len(vectors_sample) - 1))
        else:
            metrics['perpendicular_ratio'] = 0.0

        return metrics

    def statistical_distribution_analysis(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Analyze statistical distributions of vector properties."""
        metrics = {}

        if len(vectors) == 0:
            return {'error': 'No vectors found'}

        # Width distribution (assuming format: [type, x1, y1, x2, y2] - no width column)
        # Use line length as proxy for "width"
        lengths = np.sqrt((vectors[:, 3] - vectors[:, 1])**2 + (vectors[:, 4] - vectors[:, 2])**2)
        metrics['width_distribution'] = {
            'mean': float(lengths.mean()),
            'std': float(lengths.std()),
            'min': float(lengths.min()),
            'max': float(lengths.max()),
            'median': float(np.median(lengths)),
            'q25': float(np.percentile(lengths, 25)),
            'q75': float(np.percentile(lengths, 75))
        }

        # No probability distribution available in this format
        metrics['probability_distribution'] = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'q25': 0.0,
            'q75': 0.0
        }

        # Length distribution
        metrics['length_distribution'] = {
            'mean': float(lengths.mean()),
            'std': float(lengths.std()),
            'min': float(lengths.min()),
            'max': float(lengths.max()),
            'median': float(np.median(lengths)),
            'q25': float(np.percentile(lengths, 25)),
            'q75': float(np.percentile(lengths, 75))
        }

        # Angle distribution
        angles = np.arctan2(vectors[:, 3] - vectors[:, 1], vectors[:, 2] - vectors[:, 0])
        angles_deg = np.rad2deg(angles) % 180
        metrics['angle_distribution'] = {
            'mean': float(angles_deg.mean()),
            'std': float(angles_deg.std()),
            'min': float(angles_deg.min()),
            'max': float(angles_deg.max()),
            'median': float(np.median(angles_deg)),
            'q25': float(np.percentile(angles_deg, 25)),
            'q75': float(np.percentile(angles_deg, 75))
        }

        return metrics

    def error_pattern_analysis(self, vectors: np.ndarray, original: np.ndarray, rendered: np.ndarray) -> Dict[str, Any]:
        """Analyze common error patterns in vectorization."""
        metrics = {}

        if len(vectors) == 0:
            return {'error': 'No vectors found'}

        # Over-segmentation detection (very short lines)
        lengths = np.sqrt((vectors[:, 2] - vectors[:, 0])**2 + (vectors[:, 3] - vectors[:, 1])**2)
        short_lines = lengths < np.percentile(lengths, 10)  # Bottom 10%
        metrics['over_segmentation_ratio'] = float(short_lines.mean())

        # Missing primitive detection (gaps in rendered vs original)
        if HAS_PIL:
            from PIL import Image as PILImage
            orig_pil = PILImage.fromarray(original)
            render_pil = PILImage.fromarray(rendered)
            min_w, min_h = min(orig_pil.width, render_pil.width), min(orig_pil.height, render_pil.height)
            orig_resized = np.array(orig_pil.resize((min_w, min_h), PILImage.LANCZOS))
            render_resized = np.array(render_pil.resize((min_w, min_h), PILImage.LANCZOS))

            # Areas in original but not in render (missing primitives)
            missing_pixels = np.logical_and(orig_resized > 127, render_resized <= 127).sum()
            total_orig_pixels = np.sum(orig_resized > 127)
            metrics['missing_primitive_ratio'] = missing_pixels / total_orig_pixels if total_orig_pixels > 0 else 0.0

            # Extra primitives (hallucinations)
            extra_pixels = np.logical_and(render_resized > 127, orig_resized <= 127).sum()
            total_render_pixels = np.sum(render_resized > 127)
            metrics['extra_primitive_ratio'] = extra_pixels / total_render_pixels if total_render_pixels > 0 else 0.0
        else:
            metrics['missing_primitive_ratio'] = 0.0
            metrics['extra_primitive_ratio'] = 0.0

        return metrics

    def compute_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Compute overall quality score from individual metrics."""
        try:
            # Geometric score (40%)
            geo = analysis.get('geometric_accuracy', {})
            geometric_score = geo.get('iou', 0) * 0.4

            # Visual score (30%)
            vis = analysis.get('visual_quality', {})
            ssim_score = vis.get('ssim', 0) * 0.3
            psnr_score = min(vis.get('psnr', 0) / 50, 1) * 0.2  # Normalize PSNR
            visual_score = ssim_score + psnr_score

            # Structural score (20%)
            struct = analysis.get('structural_topological', {})
            cad_compliance = struct.get('cad_angle_compliance', {}).get('ratio', 0) * 0.1
            axis_aligned = struct.get('axis_aligned_ratio', 0) * 0.05
            parallel = struct.get('parallel_ratio', 0) * 0.05
            structural_score = cad_compliance + axis_aligned + parallel

            # CAD-specific score (10%)
            cad = analysis.get('cad_specific', {})
            connectivity = cad.get('endpoint_connectivity', 0) * 0.05
            equal_lengths = cad.get('equal_length_ratio', 0) * 0.05
            cad_score = connectivity + equal_lengths

            return geometric_score + visual_score + structural_score + cad_score
        except KeyError:
            return 0.0

    def analyze_output_quality(self, output_dir: str, original_path: Optional[str] = None) -> Dict[str, Any]:
        """Run complete comprehensive quality analysis on pipeline outputs."""
        logger.info(f"Running comprehensive analysis on: {output_dir}")

        # Load outputs
        outputs = self.load_outputs(output_dir, original_path)

        analysis = {
            'timestamp': str(np.datetime64('now')),
            'output_dir': output_dir,
            'image_dimensions': {
                'original': outputs['original'].shape if 'original' in outputs else None,
                'rendered': outputs['rendered'].shape if 'rendered' in outputs else None
            },
            'has_ground_truth': outputs['ground_truth'] is not None
        }

        # If ground truth is available, compare vectors directly
        if outputs['ground_truth'] is not None:
            logger.info("Ground truth available - performing vector-to-vector comparison")
            analysis['vector_comparison'] = self.vector_to_vector_comparison(
                outputs['ground_truth'], outputs['vectors']
            )
        else:
            logger.info("No ground truth available - comparing rendered output to original")

        # Geometric accuracy (original vs rendered)
        logger.info("Calculating geometric accuracy metrics...")
        analysis['geometric_accuracy'] = self.geometric_accuracy_metrics(
            outputs['original'], outputs['rendered']
        )

        # Structural and topological
        logger.info("Analyzing structural and topological metrics...")
        analysis['structural_topological'] = self.structural_topological_metrics(outputs['vectors'])

        # Visual quality
        logger.info("Calculating visual quality metrics...")
        analysis['visual_quality'] = self.visual_quality_metrics(
            outputs['original'], outputs['rendered']
        )

        # CAD-specific metrics
        logger.info("Analyzing CAD-specific metrics...")
        analysis['cad_specific'] = self.cad_specific_metrics(outputs['vectors'])

        # Statistical distributions
        logger.info("Analyzing statistical distributions...")
        analysis['statistical_distributions'] = self.statistical_distribution_analysis(outputs['vectors'])

        # Error patterns
        logger.info("Analyzing error patterns...")
        analysis['error_patterns'] = self.error_pattern_analysis(
            outputs['vectors'], outputs['original'], outputs['rendered']
        )

        # Overall quality score
        analysis['overall_score'] = self.compute_overall_score(analysis)

        logger.info("Comprehensive analysis complete!")
        return analysis

    def save_report(self, analysis: Dict[str, Any], output_path: str):
        """Save comprehensive analysis report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Comprehensive report saved to: {output_path}")

    def print_summary(self, analysis: Dict[str, Any]):
        """Print human-readable summary of comprehensive analysis."""
        print("\n" + "="*80)
        print("COMPREHENSIVE DEEPV OUTPUT QUALITY ANALYSIS SUMMARY")
        print("="*80)

        print(f"Output Directory: {analysis['output_dir']}")
        print(f"Image Dimensions: {analysis['image_dimensions']}")

        print(f"\nOverall Quality Score: {analysis.get('overall_score', 0):.4f}/1.000")

        # Geometric Accuracy
        print("\n1. GEOMETRIC ACCURACY:")
        geo = analysis.get('geometric_accuracy', {})
        print(".4f")
        print(".4f")
        print(".2f")

        # Structural/Topological
        print("\n2. STRUCTURAL/TOPOLOGICAL:")
        struct = analysis.get('structural_topological', {})
        print(f"   Primitives: {struct.get('primitive_count', 0)}")
        length_stats = struct.get('length_stats', {})
        print(".1f")
        cad_compliance = struct.get('cad_angle_compliance', {})
        print(".3f")
        print(".3f")
        print(".3f")

        # Visual Quality
        print("\n3. VISUAL QUALITY:")
        vis = analysis.get('visual_quality', {})
        print(".2f")
        print(".2f")
        print(".4f")
        print(".2f")

        # CAD-Specific
        print("\n4. CAD-SPECIFIC:")
        cad = analysis.get('cad_specific', {})
        print(".3f")
        print(".3f")
        print(".3f")

        # Statistical Distributions
        print("\n5. STATISTICAL DISTRIBUTIONS:")
        stats = analysis.get('statistical_distributions', {})
        width_dist = stats.get('width_distribution', {})
        print(".2f")
        prob_dist = stats.get('probability_distribution', {})
        print(".3f")
        angle_dist = stats.get('angle_distribution', {})
        print(".1f")

        # Error Patterns
        print("\n6. ERROR PATTERNS:")
        errors = analysis.get('error_patterns', {})
        print(".3f")
        print(".3f")
        print(".3f")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive DeepV pipeline output quality analysis")
    parser.add_argument('--output_dir', required=True, help='Directory containing pipeline outputs')
    parser.add_argument('--original', help='Path to original image (if not in output_dir)')
    parser.add_argument('--save_report', help='Save comprehensive analysis report to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')

    args = parser.parse_args()

    analyzer = ComprehensiveQualityAnalyzer()

    try:
        analysis = analyzer.analyze_output_quality(args.output_dir, args.original)

        if not args.quiet:
            analyzer.print_summary(analysis)

        if args.save_report:
            analyzer.save_report(analysis, args.save_report)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()