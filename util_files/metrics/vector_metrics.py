#!/usr/bin/env python3
"""
Vector Metrics Module

Metrics for evaluating vector graphics quality and comparison.
Provides functions for computing similarity between predicted and reference vector primitives.

Features:
- Vector-to-vector distance computation
- Primitive matching and assignment
- Parameter-wise comparison
- Raster conversion utilities

Used by evaluation pipelines for vector quality assessment.
"""

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances

import util_files.metrics.raster_metrics as r
from util_files.data.graphics_primitives import (
    PT_QBEZIER,
    GraphicsPrimitive,
    repr_len_by_type,
)
from util_files.rendering.cairo import render

VectorImage = Dict[str, List[GraphicsPrimitive]]


def _maybe_vector_to_raster(image, raster_res, data_representation="vahe"):
    if isinstance(image, Dict):
        raster = render(image, raster_res, data_representation)
    elif isinstance(image, List):
        raster = np.array([render(vector, raster_res, data_representation) for vector in image])
    elif isinstance(image, np.ndarray):
        raster = image
    else:
        raise TypeError(f"Parameter 'image' must be a Dict, List, or np.ndarray, got {type(image).__name__}")
    return raster


def hausdorff_score(image_true, image_pred, raster_res, **kwargs):
    """Computes Hausdorff metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.hausdorff_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.hausdorff_score(raster_true, raster_pred, **kwargs)


def psnr_score(image_true, image_pred, raster_res, **kwargs):
    """Computes PSNR metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.psnr_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.psnr_score(raster_true, raster_pred, **kwargs)


def f1_score(image_true, image_pred, raster_res, **kwargs):
    """Computes F1 metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.psnr_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.f1_score(raster_true, raster_pred, **kwargs)


def precision_score(image_true, image_pred, raster_res, **kwargs):
    """Computes Precision metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.precision_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.precision_score(raster_true, raster_pred, **kwargs)


def recall_score(image_true, image_pred, raster_res, **kwargs):
    """Computes Precision metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.recall_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.recall_score(raster_true, raster_pred, **kwargs)


def emd_score(image_true, image_pred, raster_res, **kwargs):
    """Computes IoU metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.iou_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.emd_score(raster_true, raster_pred, **kwargs)


def cd_score(image_true, image_pred, raster_res, **kwargs):
    """Computes IoU metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.iou_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.cd_score(raster_true, raster_pred, **kwargs)


def iou_score(image_true, image_pred, raster_res, **kwargs):
    """Computes IoU metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.iou_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.iou_score(raster_true, raster_pred, **kwargs)


# TODO sasha fix this PT_LINE should be default
def batch_numpy_to_vector(batch_numpy, raster_res, primitive_type=PT_QBEZIER):
    max_x, max_y = raster_res
    assert max_x == max_y
    repr_len = repr_len_by_type[primitive_type]  # TODO fix this

    def is_really_line(item):
        return round(item[repr_len]) == 1.0

    return [
        {primitive_type: np.array([item[:repr_len] * max_x for item in filter(is_really_line, array)])}
        for array in batch_numpy
    ]


# TODO fix PT_LINE should be default
def _vector_to_numpy(image, primitive_type=PT_QBEZIER):
    if isinstance(image, Dict):
        image_array = np.array(image[primitive_type])
    elif isinstance(image, List):
        image_array = np.array([vector[primitive_type] for vector in image])
    elif isinstance(image, np.ndarray):
        image_array = image
    else:
        raise TypeError(f"Parameter 'image' must be a Dict, List, or np.ndarray, got {type(image).__name__}")
    return image_array


def endpoint_score(image_true, image_pred, kind="full", metric="euclidean", average=None, **kwargs):
    """Computes metric between endpoints of ground-truth and predicted vectors."""

    def _metric(image_true, image_pred, kind="full", metric="euclidean"):
        if image_true.size == 0 or image_pred.size == 0:
            return np.nan

        image_true, image_pred = image_true[:4], image_pred[:4]
        cost_matrix = pairwise_distances(image_true, image_pred, metric=metric)

        if kind == "full":  # compute max(N, M) correspondences
            pred_match_idx = cost_matrix.argmin(axis=1)  # indexes of preds that true primitives match to
            diffsq1 = cost_matrix[range(len(image_true)), pred_match_idx]

            true_match_idx = cost_matrix.argmin(axis=0)  # indexes of true primitives that preds match to
            left_pred_idx = np.array([idx for idx in range(len(image_pred)) if idx not in pred_match_idx]).astype(int)

            diffsq2 = cost_matrix[true_match_idx[left_pred_idx], left_pred_idx]
            diffsq = np.hstack((diffsq1, diffsq2))

        elif kind == "bijection":  # compute min(N, M) correspondences (one-to-one matching)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            diffsq = cost_matrix[row_ind, col_ind]

        else:
            raise ValueError(f"Unknown kind '{kind}'. Supported kinds are 'full' and 'bijection'.")

        return np.mean(diffsq)

    tensor_true, tensor_pred = r.ensure_tensor(_vector_to_numpy(image_true)), r.ensure_tensor(
        _vector_to_numpy(image_pred)
    )
    value = np.array(
        [
            _metric(image_true, image_pred, kind=kind, metric=metric)
            for image_true, image_pred in zip(tensor_true, tensor_pred)
        ]
    )
    value = value[~np.isnan(value)]
    if average == "mean":
        value = np.mean(value, axis=0)

    return value


def mse_score(image_true, image_pred, kind="full", average=None, **kwargs):
    """Computes MSE metric between endpoints of ground-truth and predicted vectors."""
    return endpoint_score(image_true, image_pred, metric="euclidean", kind=kind, average=average, **kwargs)


def mae_score(image_true, image_pred, kind="full", average=None, **kwargs):
    """Computes MAE metric between endpoints of ground-truth and predicted vectors."""
    return endpoint_score(image_true, image_pred, metric="cityblock", kind=kind, average=average, **kwargs)


def curve_hausdorff_distance(curve1, curve2, n_samples=100):
    """Compute Hausdorff distance between two curves by sampling points."""
    if len(curve1) < 4 or len(curve2) < 4:
        # Not enough points for curve, fall back to endpoint distance
        return np.linalg.norm(np.array(curve1[:2]) - np.array(curve2[:2]))

    # Sample points along the curves
    if len(curve1) == 4:  # Line: (x1,y1,x2,y2)
        points1 = np.array([curve1[:2], curve1[2:4]])
    elif len(curve1) == 6:  # Quadratic Bézier: (x1,y1,x2,y2,x3,y3)
        t = np.linspace(0, 1, n_samples)
        p0, p1, p2 = curve1[:2], curve1[2:4], curve1[4:6]
        points1 = np.array([(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2 for t in t])
    elif len(curve1) == 8:  # Cubic Bézier: (x1,y1,x2,y2,x3,y3,x4,y4)
        t = np.linspace(0, 1, n_samples)
        p0, p1, p2, p3 = curve1[:2], curve1[2:4], curve1[4:6], curve1[6:8]
        points1 = np.array(
            [(1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3 for t in t]
        )
    else:
        # Unknown curve type, use endpoints
        points1 = np.array([curve1[:2], curve1[-2:]])

    if len(curve2) == 4:  # Line
        points2 = np.array([curve2[:2], curve2[2:4]])
    elif len(curve2) == 6:  # Quadratic Bézier
        t = np.linspace(0, 1, n_samples)
        p0, p1, p2 = curve2[:2], curve2[2:4], curve2[4:6]
        points2 = np.array([(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2 for t in t])
    elif len(curve2) == 8:  # Cubic Bézier
        t = np.linspace(0, 1, n_samples)
        p0, p1, p2, p3 = curve2[:2], curve2[2:4], curve2[4:6], curve2[6:8]
        points2 = np.array(
            [(1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3 for t in t]
        )
    else:
        points2 = np.array([curve2[:2], curve2[-2:]])

    # Compute Hausdorff distance between point sets
    from scipy.spatial.distance import directed_hausdorff

    return max(directed_hausdorff(points1, points2)[0], directed_hausdorff(points2, points1)[0])


def curve_score(image_true, image_pred, kind="full", metric="hausdorff", average=None, **kwargs):
    """Computes curve-based metric between ground-truth and predicted vectors using full curve geometry."""

    def _curve_metric(image_true, image_pred, kind="full", metric="hausdorff"):
        if image_true.size == 0 or image_pred.size == 0:
            return np.nan

        if metric == "hausdorff":
            distance_func = curve_hausdorff_distance
        else:
            # Fallback to endpoint distance
            def distance_func(c1, c2):
                return np.linalg.norm(np.array(c1[:2]) - np.array(c2[:2]))

        cost_matrix = np.zeros((len(image_true), len(image_pred)))
        for i, curve_true in enumerate(image_true):
            for j, curve_pred in enumerate(image_pred):
                cost_matrix[i, j] = distance_func(curve_true, curve_pred)

        if kind == "full":  # compute max(N, M) correspondences
            if len(image_true) == 0 and len(image_pred) == 0:
                return 0.0
            elif len(image_true) == 0 or len(image_pred) == 0:
                return np.inf

            pred_match_idx = cost_matrix.argmin(axis=1)  # indexes of preds that true primitives match to
            diffsq1 = cost_matrix[range(len(image_true)), pred_match_idx]

            true_match_idx = cost_matrix.argmin(axis=0)  # indexes of true primitives that preds match to
            left_pred_idx = np.array([idx for idx in range(len(image_pred)) if idx not in pred_match_idx]).astype(int)

            if len(left_pred_idx) > 0:
                diffsq2 = cost_matrix[true_match_idx[left_pred_idx], left_pred_idx]
                diffsq = np.hstack((diffsq1, diffsq2))
            else:
                diffsq = diffsq1

        elif kind == "bijection":  # compute min(N, M) correspondences (one-to-one matching)
            if len(image_true) == 0 or len(image_pred) == 0:
                return np.inf
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            diffsq = cost_matrix[row_ind, col_ind]

        else:
            raise ValueError(f"Unknown kind '{kind}'. Supported kinds are 'full' and 'bijection'.")

        return np.mean(diffsq)

    tensor_true, tensor_pred = _vector_to_numpy(image_true), _vector_to_numpy(image_pred)

    value = np.array(
        [
            _curve_metric(image_true, image_pred, kind=kind, metric=metric)
            for image_true, image_pred in zip(tensor_true, tensor_pred)
        ]
    )
    value = value[~np.isnan(value)]
    if average == "mean":
        value = np.mean(value, axis=0)

    return value


def nerror_score(image_true, image_pred, average=None, **kwargs):
    """Computes the stupid variant of the accuracy metric (how accurate is the number of predicted lines)
    between ground-truth and predicted vectors."""
    tensor_true, tensor_pred = _vector_to_numpy(image_true), _vector_to_numpy(image_pred)

    nerrors = [np.abs(len(image_true) - len(image_pred)) for image_true, image_pred in zip(tensor_true, tensor_pred)]

    if average == "mean":
        nerrors = np.mean(nerrors, axis=0)

    return nerrors


METRICS_BY_NAME = {
    "f1_score": f1_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "iou_score": iou_score,
    # 'emd_score':emd_score,
    "cd_score": cd_score,
    "psnr_score": psnr_score,
    "hausdorff_score": hausdorff_score,
    "mse_score": mse_score,
    "mae_score": mae_score,
    "curve_score": curve_score,
    "nerror_score": nerror_score,
}


__all__ = [
    "f1_score",
    "precision_score",
    "recall_score",
    "iou_score",
    "psnr_score",
    "cd_score",
    "emd_score",
    "hausdorff_score",
    "mse_score",
    "mae_score",
    "curve_score",
    "nerror_score",
    "batch_numpy_to_vector",
    "METRICS_BY_NAME",
]
