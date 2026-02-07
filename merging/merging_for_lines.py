import numpy as np
import torch
from typing import Tuple, Any
from rtree import index

from merging.utils.merging_functions import (
    lines_matching,
    maximiz_final_iou,
    merge_close,
    ordered,
    save_svg,
    tensor_vector_graph_numpy,
)
from analysis.tracing import Tracer


def postprocess(
    y_pred_render: torch.Tensor,
    patches_offsets: np.ndarray,
    input_rgb: np.ndarray,
    cleaned_image: np.ndarray,
    it: int,
    options: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Postprocess refined vector predictions by merging overlapping lines and optimizing final output.

    This function consolidates line primitives from multiple patches into a clean, non-redundant set
    suitable for CAD export. It uses spatial indexing for efficient merging of close/parallel lines.

    Args:
        y_pred_render: Refined vector predictions (patches, N, 6) with [x1,y1,x2,y2,width,prob].
        patches_offsets: Patch offsets in original image (patches, 2).
        input_rgb: Original RGB image for final IOU optimization.
        cleaned_image: Cleaned input image for SVG export.
        it: Current image index for naming.
        options: Config object with merging parameters (max_angle_to_connect, etc.).

    Returns:
        Tuple of (merged_lines, rendered_image) where merged_lines is (N, 6) array.

    Notes:
        - Uses R-tree spatial index for efficient line proximity queries.
        - Merges lines within angular/distance thresholds to reduce redundancy.
        - Saves SVG output to options.output_dir/merging_output/.
    """
    nump = tensor_vector_graph_numpy(y_pred_render, patches_offsets, options)

    # Initialize tracer for merging if enabled
    tracer = Tracer(enabled=getattr(options, "trace", False), base_dir=getattr(options, "trace_dir", "output/traces"), image_id=options.image_name[it])

    lines = nump.copy()
    lines = np.array(lines)
    lines = lines[lines[:, 1].argsort()[::-1]]
    idx = index.Index()
    widths = lines[:, 4]  # storing widths and coordinates separately
    ordered_lines = []
    for i, line in enumerate(lines):
        ordered_line = ordered(line)
        idx.insert(i, ordered_line)
        ordered_lines.append(ordered_line)

    result = np.array(
        merge_close(
            lines,
            idx,
            widths,
            max_angle=options.max_angle_to_connect,
            window_width=30,  # Reduced from 200 to 30 for better performance
            max_dist=5,  # Fixed: was using angle instead of distance
            tracer=tracer,
        )
    )
    save_svg(result, cleaned_image.shape, options.image_name[it], options.output_dir + "merging_output/")
    result_tuning = np.array(maximiz_final_iou(result, input_rgb, tracer=tracer))
    save_svg(result_tuning, cleaned_image.shape, options.image_name[it], options.output_dir + "iou_postprocess/")
    result_tuning = lines_matching(result_tuning, frac=0.07)
    # Save final merge provenance summary and metrics
    try:
        if tracer.enabled:
            prov = {"pre_merge_count": int(len(nump)), "post_merge_count": int(len(result_tuning))}
            tracer.save_provenance(prov)
            # Compute and save final metrics
            metrics_dict = {
                "pre_merge_lines": int(len(nump)),
                "post_merge_lines": int(len(result_tuning)),
                "merge_compression_ratio": float(len(result_tuning) / max(len(nump), 1))
            }
            tracer.save_metrics(metrics_dict)
    except Exception:
        pass
    tuned_image = save_svg(
        result_tuning, cleaned_image.shape, options.image_name[it], options.output_dir + "lines_matching/"
    )
    return result_tuning, tuned_image
