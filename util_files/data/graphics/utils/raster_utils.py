#!/usr/bin/env python3
"""
Raster Graphics Utilities

Utilities for handling raster images in graphics processing.
Provides image encoding and conversion functions for embedded graphics.

Features:
- Image to data URI conversion
- JPEG encoding with quality control
- Base64 encoding for SVG embedding

Used by raster-embedded graphics processing.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def image_to_datauri(image, format="jpeg"):
    image = Image.fromarray(np.asarray(image))
    buffer = BytesIO()
    if format == "jpeg":
        image.save(buffer, format=format, quality=95)
        encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoding}"
    else:
        raise NotImplementedError(f"image_to_datauri is not implemented for {format} format")
