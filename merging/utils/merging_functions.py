#!/usr/bin/env python3
"""
Merging Functions Module

Core merging algorithms for consolidating vector primitives.
Provides functions for merging overlapping and nearby primitives into clean vector output.

Features:
- Primitive merging based on proximity and similarity
- Line and curve merging algorithms
- Tolerance-based consolidation
- Spatial indexing for efficient merging

Used by merging pipelines for primitive consolidation.
"""

from __future__ import division

import os
import sys
from typing import List, Tuple

sys.path.append("..")
sys.path.append(os.path.join(os.getcwd(), ".."))

import math
import os

import numpy as np
from PIL import Image
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm

import util_files.data.graphics_primitives as graphics_primitives
from util_files.data.graphics_primitives import PT_LINE
from util_files.exceptions import ClippingError
from util_files.geometric import liang_barsky_screen
from util_files.rendering.cairo import render, render_with_skeleton


def ordered(line: np.ndarray) -> np.ndarray:
    """Reorder line coordinates to ensure min_x, min_y, max_x, max_y order for spatial indexing.

    Args:
        line: Array of [x1, y1, x2, y2, width, opacity]

    Returns:
        Array of [min_x, min_y, max_x, max_y] for bounding box
    """
    min_x = min(line[0], line[2])
    min_y = min(line[1], line[3])
    max_x = max(line[0], line[2])
    max_y = max(line[1], line[3])

    return np.array([min_x, min_y, max_x, max_y])


def clip_to_box(y_pred: np.ndarray, box_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Clip line coordinates to fit within the specified box using Liang-Barsky algorithm.

    Args:
        y_pred: Array of [x1, y1, x2, y2, width, opacity]
        box_size: Tuple of (width, height) for the box

    Returns:
        Clipped line array or NaN array if clipping fails
    """
    width, height = box_size
    bbox = (0, 0, width, height)
    point1, point2 = y_pred[:2], y_pred[2:4]
    try:
        clipped_point1, clipped_point2, is_drawn = liang_barsky_screen(point1, point2, bbox)
    except Exception as e:
        raise ClippingError(f"Line clipping failed: {e}") from e

    if clipped_point1 and clipped_point2:
        return np.asarray([clipped_point1, clipped_point2, y_pred[4:]]).ravel()
    else:
        return np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def assemble_vector_patches_curves(patches_vector, patches_offsets):
    return _assemble_vector_patches(patches_vector, patches_offsets, x_indices=[0, 2, 4], y_indices=[1, 3, 5])


def assemble_vector_patches_lines(patches_vector, patches_offsets):
    return _assemble_vector_patches(patches_vector, patches_offsets, x_indices=[0, 2], y_indices=[1, 3])


def _assemble_vector_patches(patches_vector, patches_offsets, x_indices, y_indices):
    """Assemble per-patch primitive coordinates into global coordinates.

    - `patches_vector` is an iterable of (n_primitives, params) arrays for each patch.
    - `patches_offsets` is an iterable of (y_offset, x_offset) pairs for each patch.
    - `x_indices`/`y_indices` specify which columns in the primitive array correspond to x/y coordinates.

    This helper copies per-patch arrays before offsetting to avoid mutating caller data.
    """
    primitives = []
    for patch_vector, patch_offset in zip(patches_vector, patches_offsets):
        pv = patch_vector.copy()
        pv[:, x_indices] += patch_offset[1]
        pv[:, y_indices] += patch_offset[0]
        primitives.append(pv)
    return np.array(primitives)


def tensor_vector_graph_numpy(y_pred_render, patches_offsets, options):

    nump = np.array(list(map(clip_to_box, y_pred_render.reshape(-1, 6).cpu().detach().numpy())))

    nump = assemble_vector_patches_lines(
        np.array((nump.reshape(-1, options.model_output_count, 6))), np.array(patches_offsets)
    )

    nump = nump.reshape(-1, 6)

    nump = nump[~np.isnan(nump).any(axis=1)]

    nump = nump[(nump[:, -2] > 0.3)]
    nump = nump[(nump[:, -1] > 0.5)]

    nump = nump[((nump[:, 0] - nump[:, 2]) ** 2 + (nump[:, 1] - nump[:, 3]) ** 2 >= 3)]

    return nump


def merge_close_lines(lines: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Merge collinear lines that are close together using RANSAC regression.

    Args:
        lines: Array of shape (n_lines, 6) with [x1, y1, x2, y2, width, opacity]
        threshold: Slope threshold to distinguish vertical lines

    Returns:
        Merged line array of [x1, y1, x2, y2]
    """
    #     min_x = min(lines[:,0].min(), lines[:,2].min())
    #     max_x = max(lines[:,0].max(), lines[:,2].max())
    #     min_y = min(lines[:,1].min(), lines[:,3].min())
    #     max_y = max(lines[:,1].max(), lines[:,3].max())

    lr = LinearRegression()
    ransac = RANSACRegressor()

    dt = np.hstack((lines[:, 0], lines[:, 2]))
    y_t = np.hstack((lines[:, 1], lines[:, 3]))
    #     if lines.shape[0] == 1:
    #         return np.array(lines)
    if lines.shape[0] == 2:
        if (lines[0, 0] <= lines[0, 2] and lines[0, 1] <= lines[0, 3]) or (
            lines[0, 2] <= lines[0, 0] and lines[0, 3] <= lines[0, 1]
        ):
            return np.array([np.min(dt), np.min(y_t), np.max(dt), np.max(y_t)])
        else:
            return np.array([np.min(dt), np.max(y_t), np.max(dt), np.min(y_t)])
    try:
        ransac.fit(dt.reshape(-1, 1), y_t)
        inlier_mask = ransac.inlier_mask_

        lr.fit(dt[inlier_mask].reshape(-1, 1), y_t[inlier_mask])
    except Exception:
        # Fallback to simple linear regression if RANSAC fails
        lr.fit(dt.reshape(-1, 1), y_t)

    if abs(lr.coef_) >= threshold:  # vertical line
        lr = LinearRegression()

        dt = np.hstack((lines[:, 1], lines[:, 3]))
        y_t = np.hstack((lines[:, 0], lines[:, 2]))

        lr.fit(dt.reshape(-1, 1), y_t)

        dt = np.sort(dt)

        y_pred = lr.predict(dt.reshape(-1, 1))
        return np.array([y_pred[0], dt[0], y_pred[-1], dt[-1]])

    lr = LinearRegression()
    lr.fit(dt.reshape(-1, 1), y_t)
    dt = np.sort(dt)
    y_pred = lr.predict(dt.reshape(-1, 1))

    return np.array([dt[0], y_pred[0], dt[-1], y_pred[-1]])


def point_to_line_distance(point: Tuple[float, float], line: Tuple[float, float, float, float]) -> float:
    """Calculate the perpendicular distance from a point to an infinite line.

    Args:
        point: Tuple (px, py)
        line: Tuple (x1, y1, x2, y2)

    Returns:
        Distance from point to line
    """
    px, py = point
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment's just a point
        return math.hypot(px - x1, py - y1)
    return np.abs(dy * px - dx * py + x2 * y1 - x1 * y2) / np.sqrt(dy**2 + dx**2)


def point_segment_distance(point, line):
    """Calculate the minimum distance from a point to a line segment.

    Args:
        point: Tuple (px, py)
        line: Tuple (x1, y1, x2, y2)

    Returns:
        Minimum distance to the segment
    """
    px, py = point
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment's just a point
        return math.hypot(px - x1, py - y1)

    # Calculate the t that minimizes the distance.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x1
        dy = py - y1
    elif t > 1:
        dx = px - x2
        dy = py - y2
    else:
        near_x = x1 + t * dx
        near_y = y1 + t * dy
        dx = px - near_x
        dy = py - near_y

    return math.hypot(dx, dy)


def dist(line0: np.ndarray, line1: np.ndarray) -> float:
    """Calculate the minimum distance between two lines, or 9999 if too far apart.

    Args:
        line0, line1: Arrays of [x1, y1, x2, y2, width, opacity]

    Returns:
        Minimum distance between endpoints and segments, or 9999 if lines are too far apart
    """
    if (
        point_to_line_distance(line0[:2], line1[:4]) >= 2
        or point_to_line_distance(line0[2:4], line1[:4]) >= 2
        or point_to_line_distance(line1[:2], line0[:4]) >= 2
        or point_to_line_distance(line1[2:4], line0[:4]) >= 2
    ):
        return 9999

    return min(
        distance.euclidean(line0[:2], line1[:2]),
        distance.euclidean(line0[2:4], line1[:2]),
        distance.euclidean(line0[:2], line1[2:4]),
        distance.euclidean(line0[2:4], line1[2:4]),
        point_segment_distance(line0[:2], line1[:4]),
        point_segment_distance(line0[2:4], line1[:4]),
        point_segment_distance(line1[:2], line0[:4]),
        point_segment_distance(line1[2:4], line0[:4]),
    )


def dfs(graph, start):
    """Perform depth-first search on a graph.

    Args:
        graph: Dictionary representing adjacency list
        start: Starting node

    Returns:
        Set of visited nodes
    """
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(set(graph[vertex]) - visited)
    return visited


def line_length(line):
    """Calculate the length of a line segment.

    Args:
        line: Array of [x1, y1, x2, y2, ...]

    Returns:
        Euclidean distance between endpoints
    """
    return np.sqrt((line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2)


def intersect(line0, line1):
    """Find intersection points of two line segments.

    Args:
        line0, line1: Arrays of [x1, y1, x2, y2, ...]

    Returns:
        List of intersection points as (t1, t2) parameters, or empty list
    """
    # Solve the system [p1-p0, q1-q0]*[t1, t2]^T = q0 - p0
    # where line0 = (p0, p1) and line1 = (q0, q1)

    denom = (line0[2] - line0[0]) * (line1[1] - line1[3]) - (line0[3] - line0[1]) * (line1[0] - line1[2])

    if np.isclose(denom, 0):
        return []
    t1 = (
        line1[0] * (line0[1] - line1[3]) - line1[2] * (line0[1] - line1[1]) - line0[0] * (line1[1] - line1[3])
    ) / denom
    t2 = (
        -(line0[2] * (line0[1] - line1[1]) - line0[0] * (line0[3] - line1[1]) - line1[0] * (line0[1] - line0[3]))
        / denom
    )
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return [(t1, t2)]
    return []


def angle_radians(pt1, pt2):
    """Calculate the angle in radians between two vectors.

    Args:
        pt1, pt2: Tuples (x, y) representing vectors

    Returns:
        Angle in radians
    """
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product / (len1 * len2))


def normalize(x):
    """Normalize a vector to unit length.

    Args:
        x: Numpy array

    Returns:
        Normalized vector
    """
    return x / np.linalg.norm(x)


def compute_angle(line0, line1):
    pt1 = normalize([line0[2] - line0[0], line0[3] - line0[1]])
    pt2 = normalize([line1[2] - line1[0], line1[3] - line1[1]])

    try:
        angle = math.degrees(angle_radians(pt1, pt2))
    except Exception:
        # Return 0 angle if angle calculation fails
        angle = 0
    if angle >= 90 and angle <= 270:
        angle = np.abs(180 - angle)
    elif angle > 270 and angle <= 360:
        angle = 360 - angle
    return angle


def merge_close(
    lines: np.ndarray,
    idx,
    widths: np.ndarray,
    tol: float = 1e-3,
    max_dist: float = 5,
    max_angle: float = 15,
    window_width: int = 100,
    tracer=None,
) -> List[np.ndarray]:
    """
    Merge lines that are close, intersecting, or nearly parallel using spatial indexing.

    This function identifies groups of lines that should be merged based on proximity,
    intersection, and angular similarity, then consolidates them into single lines.

    Args:
        lines: Array of shape (N, 6) with [x1, y1, x2, y2, width, opacity].
        idx: R-tree spatial index for efficient proximity queries.
        widths: Array of line widths (N,).
        tol: Tolerance for merging operations.
        max_dist: Maximum distance between lines to consider merging.
        max_angle: Maximum angle difference (degrees) for parallel lines.
        window_width: Search window size for spatial queries.

    Returns:
        List of merged lines, each as [x1, y1, x2, y2, width, opacity].

    Notes:
        - Uses DFS to find connected components of mergeable lines.
        - Skips lines shorter than 3 units.
        - Merges groups into single representative lines.
    """
    n = len(lines)
    close = [[] for _ in range(n)]
    merge_trace = []

    for i in tqdm(range(n)):
        #         if (line_length(lines[i, :4]) < 3):
        #             continue
        # Create properly ordered bounding box for intersection query
        query_bbox = ordered(lines[i, :4]) + np.array([-window_width, -window_width, window_width, window_width])
        for j in idx.intersection(query_bbox):
            if i == j:
                continue

            if line_length(lines[j, :4]) < 3:
                continue
            if (
                dist(lines[i], lines[j]) < max_dist or intersect(lines[i], lines[j])  # lines are close
            ) and compute_angle(  # lines intersect
                lines[i], lines[j]
            ) < max_angle:  # the angle is less than threshold
                close[i].append(j)
    result = []
    merged = set()

    for i in range(n):

        if line_length(lines[i, :4]) < 3:
            continue

        elif (close[i]) and (i not in merged):
            path = list(dfs(close, i))
            width = widths[path].mean(keepdims=True)
            new_line = merge_close_lines(lines[path])
            merged_line = np.concatenate((new_line, width, np.ones(width.shape)))
            result.append(merged_line)
            merged.update(path)
            # record merge cluster for tracing
            try:
                merge_trace.append({"cluster": [int(p) for p in path], "merged_line": merged_line.tolist()})
            except Exception:
                pass
        elif i not in merged:
            result.append((lines[i]))

    # Save merge trace if tracer provided
    try:
        if tracer is not None and getattr(tracer, "enabled", False):
            tracer.save_merge_trace({"clusters": merge_trace})
    except Exception:
        pass

    return result


def draw_with_skeleton(lines, drawing_scale=1, skeleton_line_width=0, skeleton_node_size=0, max_x=64, max_y=64):
    scaled_primitives = lines.copy()
    scaled_primitives[..., :-1] *= drawing_scale
    return render_with_skeleton(
        {graphics_primitives.PrimitiveType.PT_LINE: scaled_primitives},
        (max_x * drawing_scale, max_y * drawing_scale),
        data_representation="vahe",
        line_width=skeleton_line_width,
        node_size=skeleton_node_size,
    )


def maximiz_final_iou(nump: np.ndarray, input_rgb: np.ndarray, tracer=None) -> List[np.ndarray]:
    """
    Optimize final line set by maximizing IOU with original image through iterative removal.

    This function performs a greedy optimization by removing lines that don't contribute
    to the overall raster match, measured by MSE against the input image.

    Args:
        nump: Array of lines (N, 6) with [x1, y1, x2, y2, width, opacity].
        input_rgb: Original RGB image for comparison.

    Returns:
        Optimized array of lines with improved IOU.

    Notes:
        - Iteratively tests removing each line and keeps removals that improve MSE.
        - Uses skeleton rendering for line visualization.
        - Can be computationally expensive for large line counts.
        - OPTIMIZED: Only tests removal of short/low-opacity lines, limits iterations.
    """
    if len(nump) == 0:
        return nump

    # Pre-compute reference rendering and MSE
    aa = (draw_with_skeleton(nump, max_x=input_rgb.shape[1], max_y=input_rgb.shape[0]) / 255.0)[..., 0]
    k = input_rgb[..., 0] / 255.0
    mse_ref = ((np.array(aa) - np.array(k)) ** 2).mean()

    lines = list(nump)

    # Only test removal of potentially redundant lines:
    # - Short lines (length < 10 pixels)
    # - Low opacity lines (< 0.7)
    # - Thin lines (width < 1.0)
    line_lengths = np.array([line_length(line) for line in lines])
    short_lines = line_lengths < 10
    low_opacity = np.array([line[5] for line in lines]) < 0.7
    thin_lines = np.array([line[4] for line in lines]) < 1.0

    # Prioritize removal candidates: short + low opacity + thin
    candidates = np.where(short_lines | low_opacity | thin_lines)[0]

    # Limit to top 50 candidates to avoid excessive computation
    if len(candidates) > 50:
        # Sort by combined score (shorter + lower opacity + thinner = higher priority)
        scores = (
            line_lengths[candidates]
            + np.array([lines[i][5] for i in candidates])
            + np.array([lines[i][4] for i in candidates])
        )
        candidates = candidates[np.argsort(scores)][:50]

    # Test removal of candidate lines
    removed_count = 0
    removal_trace = []
    for idx in candidates:
        if idx >= len(lines):  # Line might have been removed already
            continue

        # Remove the line temporarily
        poped_line = lines.pop(idx)

        trace_entry = {"candidate_idx": int(idx), "line": poped_line.tolist()}

        # Re-render with line removed
        tmp_scr = (draw_with_skeleton(np.array(lines), max_x=input_rgb.shape[1], max_y=input_rgb.shape[0]) / 255.0)[
            ..., 0
        ]
        tmp_mse = ((np.array(tmp_scr) - np.array(k)) ** 2).mean()

        if tmp_mse > mse_ref:
            # MSE got worse, put the line back
            lines.insert(idx, poped_line)
            trace_entry["removed"] = False
        else:
            # MSE improved, keep the removal
            mse_ref = tmp_mse
            removed_count += 1
            trace_entry["removed"] = True

        removal_trace.append(trace_entry)

    print(f"IOU optimization: tested {len(candidates)} lines, removed {removed_count} redundant lines")
    # Save removal trace if tracer provided
    try:
        if tracer is not None and getattr(tracer, "enabled", False):
            tracer.save_merge_trace(
                {
                    "iou_optimization": {
                        "tested": int(len(candidates)),
                        "removed_count": int(removed_count),
                        "removals": removal_trace,
                    }
                }
            )
    except Exception:
        pass

    return lines


def two_point_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt((np.square(p1 - p2)).sum())


def line(p1):
    A = p1[1] - p1[3]
    B = p1[2] - p1[0]
    C = p1[0] * p1[3] - p1[2] * p1[1]
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return []


def lines_matching(lines, frac=0.01):
    """
    lines: lines
    frac: fraction of line
    """
    from rtree import index

    # Build spatial index for efficient intersection candidate finding
    idx = index.Index()
    for i, line_data in enumerate(lines):
        bbox = ordered(line_data)  # [min_x, min_y, max_x, max_y]
        idx.insert(i, bbox)

    line_inter = [[(np.inf, np.inf), (np.inf, np.inf)] for i in range(len(lines))]

    # Use spatial index to find candidate pairs instead of checking all pairs
    checked_pairs = set()

    for idx_1 in range(len(lines)):
        # Query for lines with overlapping bounding boxes
        bbox = ordered(lines[idx_1])
        # Expand bbox slightly to catch potential intersections
        expanded_bbox = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)

        for idx_2 in idx.intersection(expanded_bbox):
            if idx_2 <= idx_1:  # Avoid duplicate checks
                continue

            pair = (min(idx_1, idx_2), max(idx_1, idx_2))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            intr = intersection(line(lines[idx_1]), line(lines[idx_2]))
            if intr:
                # Update closest intersection for line idx_1
                if two_point_dist(intr, lines[idx_1][:2]) < two_point_dist(line_inter[idx_1][0], lines[idx_1][:2]):
                    line_inter[idx_1][0] = intr
                if two_point_dist(intr, lines[idx_1][2:4]) < two_point_dist(line_inter[idx_1][1], lines[idx_1][2:4]):
                    line_inter[idx_1][1] = intr

                # Update closest intersection for line idx_2
                if two_point_dist(intr, lines[idx_2][:2]) < two_point_dist(line_inter[idx_2][0], lines[idx_2][:2]):
                    line_inter[idx_2][0] = intr
                if two_point_dist(intr, lines[idx_2][2:4]) < two_point_dist(line_inter[idx_2][1], lines[idx_2][2:4]):
                    line_inter[idx_2][1] = intr

    for idx in range(len(lines)):
        if two_point_dist(lines[idx][:2], lines[idx][2:4]) * frac >= two_point_dist(line_inter[idx][0], lines[idx][:2]):
            lines[idx][:2] = list(line_inter[idx][0])

        if two_point_dist(lines[idx][:2], lines[idx][2:4]) * frac >= two_point_dist(
            line_inter[idx][1], lines[idx][2:4]
        ):
            lines[idx][2:4] = list(line_inter[idx][1])

    return lines


def save_svg(result_vector, size, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if len(size) == 2:
        size = (1, size[0], size[1])
    a = {PT_LINE: np.concatenate((result_vector[..., :-1], result_vector[..., -1][..., None]), axis=1)}
    rendered_image = render(a, (size[2], size[1]), data_representation="vahe", linecaps="round")
    Image.fromarray(rendered_image).save(output_dir + name)
    return rendered_image
