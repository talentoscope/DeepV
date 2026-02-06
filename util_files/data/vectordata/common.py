"""
SVG Vector Data Processing Utilities

This module provides utilities for processing SVG files and extracting vector primitives
for use in vectorization training. It handles SVG parsing, patch extraction, and primitive
parameterization using svgpathtools.

Key Functions:
- get_random_patch_from_svg: Extract random patches from SVG files with augmentation
- sample_primitive_representation: Convert SVG paths to graphics primitives
- to_primitive_representation: Convert individual SVG segments to primitives

Used by: SVG dataset classes for training data preparation.
Dependencies: svgpathtools for SVG parsing.
"""

import random
import warnings
from math import pi

import svgpathtools

from util_files.data import graphics_primitives, vectortools


def get_random_patch_from_svg(
    svg_file_location,
    patch_size,
    margin=1,
    new_ppi_to_old_ppi=1,
    scale_width=1,
    rotate=True,
    per_path_translation_amplitude=0,
    per_path_rotation_amplitude=0,
    mutually_exclusive_transforms=True,
    jitter_translation_amplitude=0,
    jitter_radius_amplitude=0,
    jitter_rotation_amplitude=0,
    random_width=False,
    remove_filled=True,
    outline_filled=False,
    outline_style=None,
    all_black=False,
):
    """Extract a random patch from an SVG file with data augmentation.

    This function loads an SVG file, extracts a random rectangular patch, and applies
    various augmentations to create diverse training data for vectorization.

    Args:
        svg_file_location (str): Path to the SVG file
        patch_size (tuple): (width, height) of the patch in pixels
        margin (float): Margin factor for patch extraction (affects augmentation area)
        new_ppi_to_old_ppi (float): PPI scaling factor for coordinate transformation
        scale_width (float): Width scaling factor for path strokes
        rotate (bool): Whether to apply random rotation augmentation
        per_path_translation_amplitude (complex): Translation amplitude per path
        per_path_rotation_amplitude (float): Rotation amplitude per path in degrees
        mutually_exclusive_transforms (bool): Whether transforms are mutually exclusive
        jitter_translation_amplitude (complex): Jitter translation amplitude
        jitter_radius_amplitude (float): Jitter radius scaling factor
        jitter_rotation_amplitude (float): Jitter rotation amplitude in degrees
        random_width (bool): Whether to randomize stroke widths
        remove_filled (bool): Whether to remove filled paths
        outline_filled (bool): Whether to convert filled paths to outlines
        outline_style (dict): Stroke attributes for outlined filled paths
        all_black (bool): Whether to make all paths black

    Returns:
        tuple: (paths, attribute_dicts, svg_attributes) for the extracted patch
    """
    assert "patch sizes and canvas sizes are in pixels"

    # read svg
    paths, attribute_dicts, svg_attributes = svgpathtools.svg2paths2(svg_file_location)

    # get canvas size
    if "viewBox" in svg_attributes:
        svg_width, svg_height = svg_attributes["viewBox"].split()[-2:]
        svg_width, svg_height = float(svg_width), float(svg_height)
    else:
        if svg_attributes["width"].endswith("px"):
            svg_width = float(svg_attributes["width"][:-2])
        else:
            svg_width = float(svg_attributes["width"])
        if svg_attributes["height"].endswith("px"):
            svg_height = float(svg_attributes["height"][:-2])
        else:
            svg_height = float(svg_attributes["height"])

    # get patch size
    patch_width, patch_height = patch_size

    # convert patch_size to old ppi
    patch_width_op = patch_width
    patch_height_op = patch_height
    if new_ppi_to_old_ppi != 1:
        patch_width_op /= new_ppi_to_old_ppi
        patch_height_op /= new_ppi_to_old_ppi

    # randomly choose upper left corner of the patch
    patch_ox, patch_oy = random.uniform(0, svg_width - patch_width_op), random.uniform(0, svg_height - patch_height_op)

    # get area of interest -- (1 + 2*margin)^2 around our patch
    left = patch_ox - patch_width_op
    top = patch_oy - patch_height_op
    right = left + patch_width_op * 3
    bottom = top + patch_height_op * 3

    # get paths of interest
    paths, attribute_dicts = vectortools.remove_empty_paths(paths, attribute_dicts, remove_filled=remove_filled)
    paths, attribute_dicts = vectortools.flatten_transforms(paths, attribute_dicts)
    paths, attribute_dicts = vectortools.get_graphics_in_rect(paths, attribute_dicts, (left, top, right, bottom))

    # augment
    if rotate:
        deg = random.uniform(0, pi * 2)
        origin = patch_ox + patch_width_op / 2 + (patch_oy + patch_height_op) * 1j
        paths = vectortools.transforms.rotate(paths, deg, origin)

    def jittering_transform(paths):
        return vectortools.transforms.random_jitter(
            paths, jitter_translation_amplitude / new_ppi_to_old_ppi, jitter_radius_amplitude, jitter_rotation_amplitude
        )

    def translation_transform(paths):
        return vectortools.transforms.random_translate(paths, per_path_translation_amplitude / new_ppi_to_old_ppi)

    def rotation_transform(paths):
        return vectortools.transforms.random_rotate(paths, per_path_rotation_amplitude)

    if mutually_exclusive_transforms:
        transforms = []
        if jitter_translation_amplitude != 0 or jitter_radius_amplitude != 0 or jitter_rotation_amplitude != 0:
            transforms.append(jittering_transform)
        if per_path_translation_amplitude != 0:
            transforms.append(translation_transform)
        if per_path_rotation_amplitude > 0:
            transforms.append(rotation_transform)

        if len(transforms) > 0:
            paths = transforms[random.randint(0, len(transforms) - 1)](paths)
    else:
        if jitter_translation_amplitude != 0 or jitter_radius_amplitude != 0 or jitter_rotation_amplitude != 0:
            paths = jittering_transform(paths)
        if per_path_translation_amplitude != 0:
            paths = translation_transform(paths)
        if per_path_rotation_amplitude > 0:
            paths = rotation_transform(paths)

    # cut out margins
    left = patch_ox
    top = patch_oy
    right = left + patch_width_op
    bottom = top + patch_height_op

    paths, attribute_dicts = vectortools.remove_empty_paths(paths, attribute_dicts)
    paths, attribute_dicts = vectortools.get_graphics_in_rect(paths, attribute_dicts, (left, top, right, bottom))

    # set origin to the upper left corner of the patch, adjust ppi, and draw outlines of the filled paths
    out_paths = []
    out_attributes = []
    for path, attributes in zip(paths, attribute_dicts):
        path = path.translated(-(left + 1j * top))

        attributes = vectortools.clean_attributes(attributes)

        if new_ppi_to_old_ppi != 1 or scale_width != 1:
            path, attributes = vectortools.adjust_path_ppi(path, attributes, new_ppi_to_old_ppi, scale_width)

        if outline_filled:
            attributes = vectortools.outline_filled(attributes, outline_style=outline_style)

        if random_width:
            if "stroke-width" in attributes:
                new_width = float(attributes["stroke-width"])
                new_width *= 1 + random.gauss(0, 1 / 3)
                new_width = max(min(new_width, min(patch_width, patch_height) / 10 * 0.75), 0.75)  # 1 px = 0.75 pt
                attributes["stroke-width"] = str(new_width)

        if all_black:
            if vectortools.path_is_outlined(attributes):
                attributes["stroke"] = "#000000"
            if vectortools.path_is_filled(attributes):
                attributes["fill"] = "#000000"

        out_paths.append(path)
        out_attributes.append(attributes)

    # set patch canvas attributes
    svg_attributes = dict()
    svg_attributes["width"] = str(patch_width) + "px"
    svg_attributes["height"] = str(patch_height) + "px"
    svg_attributes["viewBox"] = "0 0 {} {}".format(patch_width, patch_height)
    svg_attributes["x"] = svg_attributes["y"] = "0px"

    return out_paths, out_attributes, svg_attributes


def sample_parametric_representation(paths, attribute_dicts, max_lines_n=None, max_arc_n=0, max_beziers_n=None):
    warnings.warn("legacy", DeprecationWarning)
    assert max_arc_n == 0, (
        "Arc has rotation as its parameter, but a CNN is unlikely to learn finding such parameter from an image. "
        "Need to convert arcs to bezier curves."
    )

    # TODO implement random sampling instead of taking first max_x_count
    lines = []
    arcs = []
    beziers = []

    lines_n = 0
    beziers_n = 0

    for path, attributes in zip(paths, attribute_dicts):
        for seg in path:
            if isinstance(seg, svgpathtools.Line):
                if max_lines_n is not None and lines_n == max_lines_n:
                    continue
                lines.append(to_parametric_representation(seg, attributes))
                lines_n += 1
            elif isinstance(seg, (svgpathtools.QuadraticBezier, svgpathtools.CubicBezier)):
                if max_beziers_n is not None and beziers_n == max_beziers_n:
                    continue
                beziers.append(to_parametric_representation(seg, attributes))
                beziers_n += 1

    lines.sort()
    beziers.sort()

    return lines, arcs, beziers


def sample_primitive_representation(
    paths, attribute_dicts, max_lines_n=None, max_arc_n=0, max_beziers_n=None, sample_primitives_randomly=True
):
    """Sample graphics primitives from SVG paths for training targets.

    Converts SVG path segments into graphics_primitives.Line and graphics_primitives.BezierCurve
    objects that can be used as training targets for the vectorization network.

    Args:
        paths (list): List of svgpathtools Path objects
        attribute_dicts (list): List of attribute dictionaries for each path
        max_lines_n (int): Maximum number of lines to sample (None for unlimited)
        max_arc_n (int): Maximum number of arcs (must be 0, arcs not supported)
        max_beziers_n (int): Maximum number of Bézier curves to sample (None for unlimited)
        sample_primitives_randomly (bool): Whether to sample randomly or take first N

    Returns:
        tuple: (lines, arcs, beziers) where each is a list of graphics_primitives objects
    """
    assert max_arc_n == 0, (
        "Arc has rotation as its parameter, but a CNN is unlikely to learn finding such parameter from an image. "
        "Need to convert arcs to bezier curves."
    )

    lines = []
    arcs = []
    beziers = []

    lines_n = 0
    beziers_n = 0

    for path, attributes in zip(paths, attribute_dicts):
        for seg in path:
            if isinstance(seg, svgpathtools.Line):
                if max_lines_n is not None and lines_n == max_lines_n and not sample_primitives_randomly:
                    continue
                lines.append(to_primitive_representation(seg, attributes))
                lines_n += 1
            elif isinstance(seg, (svgpathtools.QuadraticBezier, svgpathtools.CubicBezier)):
                if max_beziers_n is not None and beziers_n == max_beziers_n and not sample_primitives_randomly:
                    continue
                beziers.append(to_primitive_representation(seg, attributes))
                beziers_n += 1

    if sample_primitives_randomly:
        if lines_n > max_lines_n:
            primitive_ids = random.sample(range(lines_n), max_lines_n)
            lines = [lines[idx] for idx in primitive_ids]
        if beziers_n > max_beziers_n:
            primitive_ids = random.sample(range(beziers_n), max_beziers_n)
            beziers = [beziers[idx] for idx in primitive_ids]

    return lines, arcs, beziers


def to_parametric_representation(seg, attributes):
    warnings.warn("legacy", DeprecationWarning)
    assert not isinstance(seg, svgpathtools.Arc), (
        "Not implemented. Arc has rotation as its parameter, but a CNN is unlikely to learn "
        "finding such parameter from an image. Need to convert arcs to bezier curves."
    )

    if isinstance(
        seg,
        (
            svgpathtools.Line,
            svgpathtools.CubicBezier
        )
    ):
        points = seg.bpoints()
    # convert quadratic bezier to cubic bezier
    if isinstance(seg, svgpathtools.QuadraticBezier):
        """source https://stackoverflow.com/a/3162732"""
        q0, q1, q2 = seg.bpoints()
        points = [q0, q0 + (q1 - q0) * 2 / 3, q2 + (q1 - q2) * 2 / 3, q2]

    reversed_points = reversed(points)
    coords = [coord for point in points for coord in (point.real, point.imag)]
    reversed_coords = [coord for point in reversed_points for coord in (point.real, point.imag)]

    if coords < reversed_coords:
        return coords + [
            float(attributes["stroke-width"]),
        ]
    else:
        return reversed_coords + [
            float(attributes["stroke-width"]),
        ]


def to_primitive_representation(seg, attributes):
    """Convert an SVG path segment to a graphics primitive object.

    Takes an individual SVG path segment (Line, QuadraticBezier, or CubicBezier) and
    converts it to the corresponding graphics_primitives.Line or graphics_primitives.BezierCurve
    object with the appropriate stroke width.

    Args:
        seg: svgpathtools path segment (Line, QuadraticBezier, or CubicBezier)
        attributes (dict): SVG attributes dictionary containing stroke-width

    Returns:
        graphics_primitives.Line or graphics_primitives.BezierCurve: The primitive object
    """
    assert not isinstance(seg, svgpathtools.Arc), (
        "Not implemented. Arc has rotation as its parameter, but a CNN is unlikely to learn "
        "finding such parameter from an image. Need to convert arcs to bezier curves."
    )

    if isinstance(
        seg,
        (
            svgpathtools.Line,
            svgpathtools.CubicBezier
        )
    ):
        points = seg.bpoints()
    # convert quadratic bezier to cubic bezier
    if isinstance(seg, svgpathtools.QuadraticBezier):
        """source https://stackoverflow.com/a/3162732"""
        q0, q1, q2 = seg.bpoints()
        points = [q0, q0 + (q1 - q0) * 2 / 3, q2 + (q1 - q2) * 2 / 3, q2]

    points = [(pt.real, pt.imag) for pt in points]

    if isinstance(seg, svgpathtools.Line):
        return graphics_primitives.Line(*points, float(attributes["stroke-width"]))
    elif isinstance(seg, (svgpathtools.CubicBezier, svgpathtools.QuadraticBezier)):
        return graphics_primitives.BezierCurve(*points, float(attributes["stroke-width"]))
