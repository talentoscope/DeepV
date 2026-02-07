#!/usr/bin/env python3
"""
Generate an HTML report from trace artifacts.

Reads output/traces/<image_id>/ and produces an interactive HTML report
with metrics, thumbnails, and links to detailed outputs.
"""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import sys

# ensure scripts/ is on sys.path so we can import local helpers
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from report_utils import select_patch_ids, make_patch_composite


def load_json_safe(path):
    """Load JSON file or return empty dict on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_patch_thumbnails(trace_dir, max_count=4):
    """Collect up to max_count patch PNG thumbnails."""
    patches_dir = trace_dir / "patches"
    if not patches_dir.exists():
        return []
    
    thumbnails = []
    for patch_dir in sorted(patches_dir.iterdir())[:max_count]:
        patch_file = patch_dir / "patch.png"
        if patch_file.exists():
            # Return relative path for embedding relative to the trace_dir
            rel_path = str(patch_file.relative_to(trace_dir)).replace('\\','/')
            thumbnails.append((patch_dir.name, rel_path))
    return thumbnails


def get_iteration_thumbnails(trace_dir, max_count=3):
    """Collect sample iteration render thumbnails."""
    iters_dir = trace_dir / "iterations"
    if not iters_dir.exists():
        return []
    
    renders = []
    for render_file in sorted(iters_dir.glob("render_*.png"))[:max_count]:
        rel_path = str(render_file.relative_to(trace_dir)).replace('\\','/')
        renders.append((render_file.stem, rel_path))
    return renders


def generate_html_report(trace_dir, output_file=None):
    """Generate HTML report from trace directory.
    
    Args:
        trace_dir: Path to trace directory (e.g., output/traces/image_001)
        output_file: Where to write HTML (default: trace_dir/report.html)
    
    Returns:
        HTML string
    """
    trace_dir = Path(trace_dir)
    image_id = trace_dir.name
    
    if output_file is None:
        output_file = trace_dir / "report.html"
    
    # Load all metadata
    determinism = load_json_safe(trace_dir / "determinism.json")
    metrics = load_json_safe(trace_dir / "metrics.json")
    pre_refinement = load_json_safe(trace_dir / "pre_refinement.json")
    post_refinement = load_json_safe(trace_dir / "post_refinement.json")
    merge_trace = load_json_safe(trace_dir / "merge_trace.json")
    provenance = load_json_safe(trace_dir / "provenance.json")
    
    # Collect artifacts
    patch_thumbs = get_patch_thumbnails(trace_dir)
    iter_thumbs = get_iteration_thumbnails(trace_dir)
    
    # Count iterations
    iters_dir = trace_dir / "iterations"
    num_iterations = len(list(iters_dir.glob("meta_*.json"))) if iters_dir.exists() else 0
    
    # Check for history
    has_history = (trace_dir / "primitive_history.npz").exists()

    # Prepare assets directory and selected patches
    assets_dir = trace_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    sel_patch_ids = select_patch_ids(trace_dir, count=20)
    patch_metrics = []
    # load primitive history if present
    hist_file = trace_dir / "primitive_history.npz"
    history = None
    if hist_file.exists():
        try:
            data = np.load(hist_file)
            if "history" in data:
                history = data["history"]
        except Exception:
            history = None

    for pid in sel_patch_ids:
        out_img, metrics_p = make_patch_composite(trace_dir, pid, assets_dir, history_iters=history, patch_size=128)
        # store relative path for HTML relative to the trace directory
        try:
            rel = out_img.relative_to(trace_dir)
        except Exception:
            rel = out_img
        rel = str(rel).replace('\\','/')
        patch_metrics.append((pid, rel, metrics_p))
    
    # Build HTML (header and metadata)
    html = f"""
    <html>
    <head>
        <title>Trace Report: {image_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .metric {{ background: #f0f0f0; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .thumbnail {{ max-width: 400px; max-height: 400px; margin: 5px; border: 1px solid #ccc; }}
            .legend {{ font-size: 13px; color: #222; margin: 6px 0 12px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Trace Report: {image_id}</h1>
        
        <h2>Determinism & Metadata</h2>
        <div class="metric">
            <strong>Seed:</strong> {determinism.get("seed", "N/A")}<br>
            <strong>Device:</strong> {determinism.get("device", "N/A")}<br>
            <strong>Timestamp:</strong> {determinism.get("timestamp", "N/A")[:19]}
        </div>
        
        <h2>Pipeline Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
    """
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            html += f"<tr><td>{key}</td><td>{value:.6f}</td></tr>\n"
        else:
            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
    
    html += """
        </table>
        
        <h2>Refinement Progress</h2>
        <div class="metric">
    """
    html += f"<strong>Total Iterations:</strong> {num_iterations}<br>\n"
    html += f"<strong>Primitive History Saved:</strong> {'Yes' if has_history else 'No'}<br>\n"
    
    html += """
        </div>
        
        <h2>Merging Summary</h2>
        <div class="metric">
    """
    html += f"<strong>Pre-Merge Lines:</strong> {provenance.get('pre_merge_count', 'N/A')}<br>\n"
    html += f"<strong>Post-Merge Lines:</strong> {provenance.get('post_merge_count', 'N/A')}<br>\n"
    
    html += """
        </div>
        
        <h2>Selected Patch Comparisons</h2>
        <div class="legend">Each composite shows three panels (left → ground-truth raster patch; middle → model's predicted primitives rasterized; right → refined/final primitives rasterized). Numeric metrics below each image are: Pixel IOU(model→GT) and Pixel IOU(final→GT).</div>
        <div>
    """
    for pid, rel_path, metrics_p in patch_metrics:
        rel_str = str(rel_path).replace('\\','/')
        html += f'<div style="display:inline-block; margin:8px;">'
        html += f'<img src="{rel_str}" class="thumbnail" alt="patch_{pid}" title="patch_{pid}">'
        html += f'<div style="font-size:12px;">Patch {pid}: IOU model={metrics_p["iou_model"]:.3f}, IOU final={metrics_p["iou_final"]:.3f}</div>'
        html += '</div>\n'

    html += """
        </div>

        <h2>Sample Patch Inputs</h2>
        <div>
    """
    for patch_id, rel_path in patch_thumbs:
        html += f'<img src="{rel_path}" class="thumbnail" alt="Patch {patch_id}" title="Patch {patch_id}">\n'
    
    html += """
        </div>
        
        <h2>Sample Iteration Renders</h2>
        <div>
    """
    for iter_name, rel_path in iter_thumbs:
        html += f'<img src="{rel_path}" class="thumbnail" alt="{iter_name}" title="{iter_name}">\n'
    
    html += """
        </div>
        
        <h2>Trace Files</h2>
        <ul>
    """
    
    # List all JSON files in trace
    for json_file in sorted(trace_dir.glob("*.json")):
        html += f"<li><a href=\"{json_file.name}\">{json_file.name}</a></li>\n"
    
    html += """
        </ul>
        
        <hr>
        <p>Generated from trace artifacts in <code>{}</code></p>
    </body>
    </html>
    """.format(trace_dir)
    
    # Write report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Report written to {output_file}")
    return html


def main():
    """Generate reports for all trace directories."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate HTML trace reports")
    parser.add_argument("--trace_dir", type=str, default="output/traces", help="Base traces directory")
    args = parser.parse_args()
    
    trace_base = Path(args.trace_dir)
    if not trace_base.exists():
        print(f"Trace directory {trace_base} does not exist")
        return

    # If the provided path is a single trace directory (contains metrics.json), process it
    if (trace_base / "metrics.json").exists():
        print(f"Generating report for {trace_base.name}...")
        generate_html_report(trace_base)
        return

    # Otherwise, iterate child directories
    for image_dir in trace_base.iterdir():
        if image_dir.is_dir():
            print(f"Generating report for {image_dir.name}...")
            generate_html_report(image_dir)


if __name__ == "__main__":
    main()
