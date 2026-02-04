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

# Import DeepV components
from util_files.rendering.bezier_splatting import BezierSplatting
from refinement.our_refinement.utils.lines_refinement_functions import render_lines_with_type


def demo_vectorize_image(input_image, rendering_type="bezier_splatting", primitive_type="line"):
    """
    Demo vectorization using rendering capabilities.

    This is a simplified demo that shows the rendering pipeline
    without requiring trained ML models.

    Args:
        input_image: PIL Image
        rendering_type: "hard" or "bezier_splatting"
        primitive_type: "line" or "curve"

    Returns:
        tuple: (processed_image, svg_content, status_message)
    """
    try:
        # Convert to grayscale and threshold to create binary image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        gray_image = input_image.convert('L')
        binary_array = np.array(gray_image) < 128  # Simple thresholding

        # Find connected components or edges to simulate vector primitives
        # For demo purposes, create some synthetic lines
        height, width = binary_array.shape

        # Create synthetic line data (in real implementation, this would come from ML model)
        lines_batch = []

        if primitive_type == "line":
            # Create some horizontal and vertical lines based on image content
            num_lines = 5

            for i in range(num_lines):
                # Random line parameters
                x1 = np.random.randint(10, width - 10)
                y1 = np.random.randint(10, height - 10)
                x2 = np.random.randint(10, width - 10)
                y2 = np.random.randint(10, height - 10)
                width_val = np.random.uniform(1.0, 3.0)

                lines_batch.append([x1, y1, x2, y2, width_val])

        lines_batch = torch.tensor([lines_batch], dtype=torch.float32)

        # Render using selected renderer
        rendered = render_lines_with_type(lines_batch, rendering_type)

        # Convert to PIL Image
        rendered_image = Image.fromarray((rendered[0].numpy() * 255).astype(np.uint8))

        # Create simple SVG representation
        svg_content = create_svg_from_lines(lines_batch[0], width, height)

        return rendered_image, svg_content, f"Demo completed with {rendering_type} rendering!"

    except Exception as e:
        error_msg = f"Error during demo: {str(e)}"
        print(error_msg)
        error_image = Image.new('L', (256, 256), color=128)
        return error_image, "<svg></svg>", error_msg


def create_svg_from_lines(lines, width, height):
    """Create simple SVG from line data."""
    svg_lines = []
    for line in lines:
        x1, y1, x2, y2, w = line.tolist()
        svg_lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{w}"/>')

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
{chr(10).join(svg_lines)}
</svg>'''

    return svg_content


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
        - SVG export
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
                        choices=["line"],
                        value="line",
                        label="Primitive Type",
                        info="Currently only lines are supported in demo"
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
            svg_output = gr.Code(
                label="SVG Output",
                language="xml",
                lines=15,
                show_label=True
            )

        # Connect the interface
        submit_btn.click(
            fn=demo_vectorize_image,
            inputs=[input_image, rendering_type, primitive_type],
            outputs=[output_image, svg_output, status_text]
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