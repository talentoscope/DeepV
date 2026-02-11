"""
CAD Export Utilities for DeepV

This module provides functionality to export vectorized primitives to CAD formats
like DXF (Drawing Exchange Format) for use in CAD software.
"""

import os
from typing import Any, Dict, Union

try:
    import numpy as np
except ImportError:
    np = None


def export_to_dxf(primitives: Union[Dict[str, Any], Any], filename: str, width: int = 256, height: int = 256) -> bool:
    """
    Export primitives to DXF format.

    Args:
        primitives: Dictionary with primitive types as keys, or array of lines
                   If dict: {'lines': array(N,5), 'curves': array(N,7), 'arcs': array(N,6)}
                   If array: legacy format array(N,5) for lines only
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

    if np is None:
        print("NumPy not available")
        return False

    # Handle legacy format (array of lines)
    if not isinstance(primitives, dict):
        primitives = {"lines": np.asarray(primitives)}

    # Create a new DXF document
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Export different primitive types
    _export_lines_to_dxf(msp, primitives, height)
    _export_curves_to_dxf(msp, primitives, height)
    _export_cubic_curves_to_dxf(msp, primitives, height)
    _export_arcs_to_dxf(msp, primitives, height)

    # Save the DXF file
    try:
        doc.saveas(filename)
        print(f"DXF exported to: {filename}")
        return True
    except Exception as e:
        print(f"Error saving DXF: {e}")
        return False


def _export_lines_to_dxf(msp, primitives, height: int) -> None:
    """Export line primitives to DXF modelspace."""
    if "lines" not in primitives or len(primitives["lines"]) == 0:
        return

    lines_np = np.asarray(primitives["lines"])
    for line in lines_np:
        x1, y1, x2, y2, line_width = line[:5]  # Handle variable length
        # Convert to DXF coordinates (flip Y axis)
        dxf_x1, dxf_y1 = float(x1), float(height - y1)
        dxf_x2, dxf_y2 = float(x2), float(height - y2)
        # Add line to DXF
        msp.add_line((dxf_x1, dxf_y1), (dxf_x2, dxf_y2))


def _export_curves_to_dxf(msp, primitives, height: int) -> None:
    """Export quadratic Bézier curves to DXF modelspace."""
    if "curves" not in primitives or len(primitives["curves"]) == 0:
        return

    curves_np = np.asarray(primitives["curves"])
    for curve in curves_np:
        if len(curve) >= 7:  # x1,y1,x2,y2,x3,y3,width
            x1, y1, x2, y2, x3, y3 = curve[:6]
            # Convert to DXF coordinates (flip Y axis)
            points = [(float(x1), float(height - y1)), (float(x2), float(height - y2)), (float(x3), float(height - y3))]
            # Add spline to DXF
            msp.add_spline(points, degree=2)


def _export_cubic_curves_to_dxf(msp, primitives, height: int) -> None:
    """Export cubic Bézier curves to DXF modelspace."""
    if "cubic_curves" not in primitives or len(primitives["cubic_curves"]) == 0:
        return

    curves_np = np.asarray(primitives["cubic_curves"])
    for curve in curves_np:
        if len(curve) >= 9:  # x1,y1,x2,y2,x3,y3,x4,y4,width
            x1, y1, x2, y2, x3, y3, x4, y4 = curve[:8]
            # Convert to DXF coordinates (flip Y axis)
            points = [
                (float(x1), float(height - y1)),
                (float(x2), float(height - y2)),
                (float(x3), float(height - y3)),
                (float(x4), float(height - y4)),
            ]
            # Add spline to DXF
            msp.add_spline(points, degree=3)


def _export_arcs_to_dxf(msp, primitives, height: int) -> None:
    """Export arc primitives to DXF modelspace."""
    if "arcs" not in primitives or len(primitives["arcs"]) == 0:
        return

    arcs_np = np.asarray(primitives["arcs"])
    for arc in arcs_np:
        if len(arc) >= 6:  # cx,cy,radius,angle1,angle2,width
            cx, cy, radius, angle1, angle2 = arc[:5]
            # Convert angles to degrees and handle coordinate system
            start_angle = float(np.degrees(angle1))
            end_angle = float(np.degrees(angle2))
            # Flip Y axis for center
            dxf_cx, dxf_cy = float(cx), float(height - cy)
            # Add arc to DXF
            msp.add_arc((dxf_cx, dxf_cy), float(radius), start_angle, end_angle)


def export_to_svg(primitives: Union[Dict[str, Any], Any], filename: str, width: int = 256, height: int = 256) -> bool:
    """
    Export primitives to SVG format.

    Args:
        primitives: Dictionary with primitive types as keys, or array of lines
                   If dict: {'lines': array(N,5), 'curves': array(N,7), 'arcs': array(N,6)}
                   If array: legacy format array(N,5) for lines only
        filename: Output SVG filename
        width: Image width
        height: Image height

    Returns:
        bool: True if export successful
    """
    if np is None:
        print("NumPy not available")
        return False

    # Handle legacy format (array of lines)
    if not isinstance(primitives, dict):
        primitives = {"lines": np.asarray(primitives)}

    svg_elements = []

    # Export lines
    if "lines" in primitives and len(primitives["lines"]) > 0:
        lines_np = np.asarray(primitives["lines"])
        for line in lines_np:
            x1, y1, x2, y2, line_width = line[:5]
            svg_elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{line_width}"/>'
            )

    # Export quadratic Bézier curves
    if "curves" in primitives and len(primitives["curves"]) > 0:
        curves_np = np.asarray(primitives["curves"])
        for curve in curves_np:
            if len(curve) >= 7:  # x1,y1,x2,y2,x3,y3,width
                x1, y1, x2, y2, x3, y3, width = curve[:7]
                # SVG quadratic Bézier: M x1 y1 Q x2 y2 x3 y3
                path_data = f"M {x1} {y1} Q {x2} {y2} {x3} {y3}"
                svg_elements.append(f'<path d="{path_data}" stroke="black" stroke-width="{width}" fill="none"/>')

    # Export cubic Bézier curves
    if "cubic_curves" in primitives and len(primitives["cubic_curves"]) > 0:
        curves_np = np.asarray(primitives["cubic_curves"])
        for curve in curves_np:
            if len(curve) >= 9:  # x1,y1,x2,y2,x3,y3,x4,y4,width
                x1, y1, x2, y2, x3, y3, x4, y4, width = curve[:9]
                # SVG cubic Bézier: M x1 y1 C x2 y2 x3 y3 x4 y4
                path_data = f"M {x1} {y1} C {x2} {y2} {x3} {y3} {x4} {y4}"
                svg_elements.append(f'<path d="{path_data}" stroke="black" stroke-width="{width}" fill="none"/>')

    # Export arcs
    if "arcs" in primitives and len(primitives["arcs"]) > 0:
        arcs_np = np.asarray(primitives["arcs"])
        for arc in arcs_np:
            if len(arc) >= 6:  # cx,cy,radius,angle1,angle2,width
                cx, cy, radius, angle1, angle2, width = arc[:6]
                # Convert angles to degrees
                start_angle = float(np.degrees(angle1))
                end_angle = float(np.degrees(angle2))
                # Calculate arc flags (large arc flag, sweep flag)
                angle_diff = (end_angle - start_angle) % 360
                large_arc = 1 if angle_diff > 180 else 0
                sweep = 1  # clockwise
                # Calculate end point
                end_x = cx + radius * np.cos(np.radians(end_angle))
                end_y = cy + radius * np.sin(np.radians(end_angle))
                # Calculate start point
                start_x = cx + radius * np.cos(np.radians(start_angle))
                start_y = cy + radius * np.sin(np.radians(start_angle))
                # SVG arc: A rx ry x-axis-rotation large-arc-flag sweep-flag x y
                path_data = f"M {start_x} {start_y} A {radius} {radius} 0 {large_arc} {sweep} {end_x} {end_y}"
                svg_elements.append(f'<path d="{path_data}" stroke="black" stroke-width="{width}" fill="none"/>')

    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
{chr(10).join(svg_elements)}
</svg>"""

    try:
        with open(filename, "w") as f:
            f.write(svg_content)
        print(f"SVG exported to: {filename}")
        return True
    except Exception as e:
        print(f"Error saving SVG: {e}")
        return False


def create_cad_from_primitives(primitives: Dict[str, Any], output_dir: str = "output") -> Dict[str, Dict[str, Any]]:
    """
    Create CAD files from vectorized primitives.

    Args:
        primitives: Dictionary containing different primitive types
            Expected keys: 'lines', 'curves', 'cubic_curves', 'arcs', etc.
        output_dir: Output directory for CAD files

    Returns:
        dict: Dictionary with file paths and success status
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Export all primitives to DXF and SVG
    if primitives and any(len(v) > 0 for v in primitives.values() if hasattr(v, "__len__")):
        dxf_path = os.path.join(output_dir, "primitives.dxf")
        svg_path = os.path.join(output_dir, "primitives.svg")

        dxf_success = export_to_dxf(primitives, dxf_path)
        svg_success = export_to_svg(primitives, svg_path)

        results["dxf"] = {"path": dxf_path, "success": dxf_success}
        results["svg"] = {"path": svg_path, "success": svg_success}
    else:
        print("No primitives to export")

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
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                # Check for basic DXF structure
                return "SECTION" in content and "ENDSEC" in content
        except (OSError, UnicodeDecodeError):
            return False

    # For SVG files, check for basic XML structure
    if filename.endswith(".svg"):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                return "<svg" in content and "</svg>" in content
        except (OSError, UnicodeDecodeError):
            return False

    return True


# Example usage and testing
if __name__ == "__main__":
    if np is None:
        print("NumPy not available. Cannot run example.")
        exit(1)

    # Create some sample line primitives
    sample_lines = np.array(
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
