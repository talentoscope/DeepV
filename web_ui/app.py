"""
DeepV Web UI - Interactive Vectorization Interface

A Gradio-based web interface for the DeepV vectorization pipeline.
Allows users to upload images and see real-time vectorization results.
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

# Import CAD export functionality
try:
    from cad.export import export_to_dxf, export_to_svg
except ImportError:
    export_to_dxf = None
    export_to_svg = None

# Import DeepV components
from util_files.rendering.bezier_splatting import BezierSplatting
from refinement.our_refinement.utils.lines_refinement_functions import render_primitives_with_type


def demo_vectorize_image(input_image, rendering_type="bezier_splatting", primitive_type="line"):
    """
    Demo vectorization using rendering capabilities.

    This is a simplified demo that shows the rendering pipeline
    without requiring trained ML models.

    Args:
        input_image: PIL Image
        rendering_type: "hard" or "bezier_splatting"
        primitive_type: "line", "curve", or "arc"

    Returns:
        tuple: (processed_image, svg_content, dxf_content, status_message)
    """
    try:
        # Convert to grayscale and threshold to create binary image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        gray_image = input_image.convert('L')
        binary_array = np.array(gray_image) < 128  # Simple thresholding

        # Find connected components or edges to simulate vector primitives
        # For demo purposes, create some synthetic primitives
        height, width = binary_array.shape

        # Create synthetic primitive data (in real implementation, this would come from ML model)
        primitives = {}

        if primitive_type == "line":
            # Create some horizontal and vertical lines
            lines = []
            num_lines = 5
            for i in range(num_lines):
                x1 = np.random.randint(10, width - 10)
                y1 = np.random.randint(10, height - 10)
                x2 = np.random.randint(10, width - 10)
                y2 = np.random.randint(10, height - 10)
                width_val = np.random.uniform(1.0, 3.0)
                lines.append([x1, y1, x2, y2, width_val])
            primitives['lines'] = torch.tensor(lines, dtype=torch.float32)

        elif primitive_type == "curve":
            # Create quadratic Bézier curves
            curves = []
            num_curves = 3
            for i in range(num_curves):
                x1 = np.random.randint(10, width - 10)
                y1 = np.random.randint(10, height - 10)
                x2 = np.random.randint(10, width - 10)  # control point
                y2 = np.random.randint(10, height - 10)
                x3 = np.random.randint(10, width - 10)
                y3 = np.random.randint(10, height - 10)
                width_val = np.random.uniform(1.0, 3.0)
                curves.append([x1, y1, x2, y2, x3, y3, width_val])
            primitives['curves'] = torch.tensor(curves, dtype=torch.float32)

        elif primitive_type == "arc":
            # Create circular arcs
            arcs = []
            num_arcs = 3
            for i in range(num_arcs):
                cx = np.random.randint(20, width - 20)
                cy = np.random.randint(20, height - 20)
                radius = np.random.randint(10, 30)
                angle1 = np.random.uniform(0, 2 * np.pi)
                angle2 = angle1 + np.random.uniform(np.pi/4, np.pi)  # arc of 45-180 degrees
                width_val = np.random.uniform(1.0, 3.0)
                arcs.append([cx, cy, radius, angle1, angle2, width_val])
            primitives['arcs'] = torch.tensor(arcs, dtype=torch.float32)

        primitives_batch = [primitives]

        # Render using selected renderer
        rendered = render_primitives_with_type(primitives_batch, rendering_type)

        # Convert to PIL Image
        rendered_image = Image.fromarray((rendered[0].numpy() * 255).astype(np.uint8))

        # Create CAD representations
        svg_content = create_svg_from_primitives(primitives, width, height)
        dxf_content = create_dxf_from_primitives(primitives, width, height)

        return rendered_image, svg_content, dxf_content, f"Demo completed with {rendering_type} rendering and {primitive_type} primitives!"

    except Exception as e:
        error_msg = f"Error during demo: {str(e)}"
        print(error_msg)
        error_image = Image.new('L', (256, 256), color=128)
        return error_image, "<svg></svg>", "", error_msg


def create_dxf_from_lines(lines, width, height):
    """Create DXF from line data using CAD export module."""
    if export_to_dxf is None:
        return "CAD export module not available"

    # Create temporary file for DXF
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.dxf', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Export to temporary DXF file
        success = export_to_dxf(lines, tmp_path, width, height)
        if not success:
            return "DXF export failed"

        # Read the DXF content
        with open(tmp_path, 'r') as f:
            dxf_content = f.read()

        return dxf_content
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def create_dxf_from_primitives(primitives, width, height):
    """Create DXF from primitives data using CAD export module."""
    if export_to_dxf is None:
        return "CAD export module not available"

    # Create temporary file for DXF
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.dxf', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Export to temporary DXF file
        success = export_to_dxf(primitives, tmp_path, width, height)
        if not success:
            return "DXF export failed"

        # Read the DXF content
        with open(tmp_path, 'r') as f:
            dxf_content = f.read()

        return dxf_content
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def create_svg_from_primitives(primitives, width, height):
    """Create SVG from primitives data using CAD export module."""
    if export_to_svg is None:
        return "SVG export module not available"

    # Create temporary file for SVG
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.svg', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Export to temporary SVG file
        success = export_to_svg(primitives, tmp_path, width, height)
        if not success:
            return "SVG export failed"

        # Read the SVG content
        with open(tmp_path, 'r') as f:
            svg_content = f.read()

        return svg_content
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="DeepV - Technical Drawing Vectorization", theme=gr.themes.Soft()) as interface:

        gr.Markdown("""
        # 🖼️ DeepV - Technical Drawing Vectorization Demo

        **Demo Interface** - This is a simplified demonstration of DeepV's rendering capabilities.

        Upload a technical drawing and see how different rendering methods work!
        (Note: This demo uses synthetic primitives for illustration)

        **Features:**
        - Compare analytical vs Bézier splatting rendering
        - Real-time rendering preview
        - SVG and DXF export for CAD software
        - Support for line primitives
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Technical Drawing",
                    type="pil",
                    height=300
                )

                with gr.Row():
                    rendering_type = gr.Radio(
                        choices=["hard", "bezier_splatting"],
                        value="bezier_splatting",
                        label="Rendering Method",
                        info="Analytical rendering vs fast Bézier splatting"
                    )

                    primitive_type = gr.Radio(
                        choices=["line", "curve", "arc"],
                        value="line",
                        label="Primitive Type",
                        info="Type of geometric primitives to render"
                    )

                submit_btn = gr.Button(
                    "🚀 Render Demo",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Rendered Result",
                    height=300
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready for demo!"
                )

        with gr.Row():
            with gr.Column():
                svg_output = gr.Code(
                    label="SVG Output (Web/CAD)",
                    language="xml",
                    lines=10,
                    show_label=True
                )

            with gr.Column():
                dxf_output = gr.Code(
                    label="DXF Output (CAD Software)",
                    language="plaintext",
                    lines=10,
                    show_label=True
                )

        # Connect the interface
        submit_btn.click(
            fn=demo_vectorize_image,
            inputs=[input_image, rendering_type, primitive_type],
            outputs=[output_image, svg_output, dxf_output, status_text]
        )

        # Add some info
        gr.Markdown("""
        ### About the Rendering Methods

        - **Hard (Analytical)**: Traditional analytical line rendering with exact geometry
        - **Bézier Splatting**: Fast differentiable rendering using Gaussian splatting on Bézier curves

        ### Note
        This is a demo interface showing rendering capabilities. Full vectorization requires trained ML models.
        """)

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )