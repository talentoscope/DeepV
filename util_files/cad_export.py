"""
CAD Export Utilities for DeepV

This module provides functionality to export vectorized primitives to CAD formats
like DXF (Drawing Exchange Format) for use in CAD software.
"""

import os


# Lazy imports to avoid conflicts
def _get_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _get_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def export_to_dxf(lines, filename: str, width: int = 256, height: int = 256) -> bool:
    """
    Export line primitives to DXF format.

    Args:
        lines: Array-like of shape (N, 5) where each row is [x1, y1, x2, y2, width]
        filename: Output DXF filename
        width: Image width for coordinate scaling
        height: Image height for coordinate scaling

    Returns:
        bool: True if export successful
    """
    try:
        import ezdxf
    except ImportError:
        print("ezdxf not installed. Install with: pip install ezdxf")
        return False

    # Create a new DXF document
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Convert lines to numpy if needed
    torch_module = _get_torch()
    np_module = _get_numpy()

    if torch_module is not None and isinstance(lines, torch_module.Tensor):
        lines_np = lines.detach().cpu().numpy()
    elif np_module is not None and isinstance(lines, np_module.ndarray):
        lines_np = lines
    else:
        lines_np = lines  # Assume it's already array-like

    for line in lines_np:
        x1, y1, x2, y2, line_width = line

        # Convert to DXF coordinates (flip Y axis)
        dxf_x1, dxf_y1 = x1, height - y1
        dxf_x2, dxf_y2 = x2, height - y2

        # Add line to DXF
        msp.add_line((dxf_x1, dxf_y1), (dxf_x2, dxf_y2))

    # Save the DXF file
    try:
        doc.saveas(filename)
        print(f"DXF exported to: {filename}")
        return True
    except Exception as e:
        print(f"Error saving DXF: {e}")
        return False


def export_to_svg(lines, filename: str, width: int = 256, height: int = 256) -> bool:
    """
    Export line primitives to SVG format.

    Args:
        lines: Array-like of shape (N, 5) where each row is [x1, y1, x2, y2, width]
        filename: Output SVG filename
        width: Image width
        height: Image height

    Returns:
        bool: True if export successful
    """
    # Convert lines to numpy if needed
    torch_module = _get_torch()
    np_module = _get_numpy()

    if torch_module is not None and isinstance(lines, torch_module.Tensor):
        lines_np = lines.detach().cpu().numpy()
    elif np_module is not None and isinstance(lines, np_module.ndarray):
        lines_np = lines
    else:
        lines_np = lines  # Assume it's already array-like

    svg_lines = []
    for line in lines_np:
        x1, y1, x2, y2, line_width = line
        svg_lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{line_width}"/>')

    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
{chr(10).join(svg_lines)}
</svg>"""

    try:
        with open(filename, "w") as f:
            f.write(svg_content)
        print(f"SVG exported to: {filename}")
        return True
    except Exception as e:
        print(f"Error saving SVG: {e}")
        return False


def create_cad_from_primitives(primitives: dict, output_dir: str = "output") -> dict:
    """
    Create CAD files from vectorized primitives.

    Args:
        primitives: Dictionary containing different primitive types
            Expected keys: 'lines', 'curves', etc.
        output_dir: Output directory for CAD files

    Returns:
        dict: Dictionary with file paths and success status
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Export lines to DXF
    if "lines" in primitives and len(primitives["lines"]) > 0:
        dxf_path = os.path.join(output_dir, "primitives.dxf")
        svg_path = os.path.join(output_dir, "primitives.svg")

        dxf_success = export_to_dxf(primitives["lines"], dxf_path)
        svg_success = export_to_svg(primitives["lines"], svg_path)

        results["dxf"] = {"path": dxf_path, "success": dxf_success}
        results["svg"] = {"path": svg_path, "success": svg_success}

    # Future: Add support for curves, arcs, etc.
    if "curves" in primitives:
        print("Curve export not yet implemented")

    return results


def validate_cad_export(filename: str) -> bool:
    """
    Validate that a CAD export file was created successfully.

    Args:
        filename: Path to the CAD file

    Returns:
        bool: True if file exists and is valid
    """
    if not os.path.exists(filename):
        return False

    # Basic validation - check file size
    if os.path.getsize(filename) == 0:
        return False

    # For DXF files, check if it contains basic structure
    if filename.endswith(".dxf"):
        try:
            with open(filename, "r") as f:
                content = f.read()
                # Check for basic DXF structure
                return "SECTION" in content and "ENDSEC" in content
        except Exception:
            return False

    # For SVG files, check for basic XML structure
    if filename.endswith(".svg"):
        try:
            with open(filename, "r") as f:
                content = f.read()
                return "<svg" in content and "</svg>" in content
        except Exception:
            return False

    return True


# Example usage and testing
if __name__ == "__main__":
    np_module = _get_numpy()
    if np_module is None:
        print("NumPy not available. Cannot run example.")
        exit(1)

    # Create some sample line primitives
    sample_lines = np_module.array(
        [
            [10.0, 20.0, 100.0, 20.0, 2.0],  # horizontal line
            [50.0, 10.0, 50.0, 100.0, 1.5],  # vertical line
            [10.0, 10.0, 100.0, 100.0, 1.0],  # diagonal line
        ]
    )

    # Export to CAD formats
    results = create_cad_from_primitives({"lines": sample_lines})

    print("CAD Export Results:")
    for format_name, result in results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"{format_name.upper()}: {status} - {result['path']}")

        # Validate the export
        if result["success"]:
            valid = validate_cad_export(result["path"])
            print(f"  Validation: {'✅ Valid' if valid else '❌ Invalid'}")
