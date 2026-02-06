"""
Graphics Primitives Module

This module defines the core vector graphics primitive classes used throughout the DeepV vectorization pipeline.
It provides parametric representations for lines, Bézier curves, and arcs that can be predicted by neural networks
and exported to CAD formats like SVG and DXF.

Key Classes:
- GraphicsPrimitive: Base class for all vector primitives
- Line: Straight line segment with start/end points and stroke width
- BezierCurve: Cubic Bézier curve with 4 control points and stroke width
- Arc: Circular arc with center, radius, angles, and stroke width

The primitives support conversion to/from neural network representations and SVG export.
Used by: vectorization training targets, CAD export, dataset processing.
"""

from enum import Enum, auto

# from vectran.renderers.cairo import cairo_line, cairo_bezier, cairo_arc
from util_files.geometric import liang_barsky_screen


class PrimitiveType(Enum):
    """Enumeration of supported vector primitive types."""
    PT_LINE = auto()
    PT_CBEZIER = auto()
    PT_ARC = auto()
    PT_POINT = auto()
    PT_QBEZIER = auto()
    PT_QBEZIER_B = auto()


PT_ARC = PrimitiveType.PT_ARC
PT_CBEZIER = PrimitiveType.PT_CBEZIER
PT_QBEZIER = PrimitiveType.PT_QBEZIER
PT_QBEZIER_B = PrimitiveType.PT_QBEZIER_B
PT_LINE = PrimitiveType.PT_LINE
PT_POINT = PrimitiveType.PT_POINT


repr_len_by_type = {
    PT_LINE: 5,
    PT_CBEZIER: 9,
    PT_QBEZIER: 7,
    PT_QBEZIER_B: 7,
    PT_ARC: 6,
}


class GraphicsPrimitive(object):
    """Base class for all vector graphics primitives.

    Provides common interface for drawing, representation conversion, and clipping.
    All primitives inherit from this class and implement the required methods.
    """

    def __init__(self):
        self.is_drawn = True

    def draw(self, ctx):
        """Draw the primitive using the provided Cairo context."""
        raise NotImplementedError

    @classmethod
    def from_repr(cls, line_repr):
        """Create a primitive instance from its neural network representation."""
        raise NotImplementedError

    def to_repr(self):
        """Convert the primitive to its neural network representation."""
        raise NotImplementedError

    def clip_to_box(self, ctx):
        """Clip the primitive to a bounding box using the provided Cairo context."""
        raise NotImplementedError


class Line(GraphicsPrimitive):
    """Straight line segment primitive.

    Represents a line from point1 to point2 with a given stroke width.
    Used for vectorizing straight edges in technical drawings and CAD files.

    Attributes:
        point1 (tuple): (x, y) coordinates of line start point
        point2 (tuple): (x, y) coordinates of line end point
        width (float): stroke width for rendering
    """

    def __init__(self, point1, point2, width):
        """Initialize a line primitive.

        Args:
            point1 (tuple): (x, y) coordinates of start point
            point2 (tuple): (x, y) coordinates of end point
            width (float): stroke width
        """
        self.point1 = point1
        self.point2 = point2
        self.width = width
        super().__init__()

    def draw(self, ctx):
        """Draw the line using Cairo context (currently commented out)."""
        """cairo_line(ctx, self.to_repr())"""

    def clip_to_box(self, box_size):
        """Clip the line to fit within a bounding box.

        Args:
            box_size (tuple): (width, height) of the bounding box
        """
        width, height = box_size
        bbox = (0, 0, width, height)
        clipped_point1, clipped_point2, self.is_drawn = liang_barsky_screen(self.point1, self.point2, bbox)
        if self.is_drawn:
            self.point1, self.point2 = clipped_point1, clipped_point2

    @classmethod
    def from_repr(cls, line_repr):
        """Create a Line from neural network representation.

        Args:
            line_repr (list): [x1, y1, x2, y2, width] neural network output

        Returns:
            Line: reconstructed line primitive
        """
        assert len(line_repr) == repr_len_by_type[PrimitiveType.PT_LINE]
        return cls(tuple(line_repr[0:2]), tuple(line_repr[2:4]), line_repr[4])

    def to_repr(self):
        if self.point1 < self.point2:
            return self.point1 + self.point2 + (self.width,)
        else:
            return self.point2 + self.point1 + (self.width,)

    def to_svg(self):
        from svgpathtools import Line as SvgLine

        return SvgLine(complex(*self.point1), complex(*self.point2))


class BezierCurve(GraphicsPrimitive):
    """Cubic Bézier curve primitive.

    Represents a smooth curve defined by 4 control points with a given stroke width.
    Used for vectorizing curved edges in technical drawings and CAD files.

    Attributes:
        cpoint1 (tuple): (x, y) coordinates of first control point (start)
        cpoint2 (tuple): (x, y) coordinates of second control point
        cpoint3 (tuple): (x, y) coordinates of third control point
        cpoint4 (tuple): (x, y) coordinates of fourth control point (end)
        width (float): stroke width for rendering
    """

    def __init__(self, cpoint1, cpoint2, cpoint3, cpoint4, width):
        """Initialize a Bézier curve primitive.

        Args:
            cpoint1 (tuple): (x, y) coordinates of start point
            cpoint2 (tuple): (x, y) coordinates of first control point
            cpoint3 (tuple): (x, y) coordinates of second control point
            cpoint4 (tuple): (x, y) coordinates of end point
            width (float): stroke width
        """
        self.cpoint1 = cpoint1
        self.cpoint2 = cpoint2
        self.cpoint3 = cpoint3
        self.cpoint4 = cpoint4
        self.width = width
        super().__init__()

    def draw(self, ctx):
        """Draw the Bézier curve using Cairo context (currently commented out)."""
        """return cairo_bezier(ctx, self.to_repr())"""

    def clip_to_box(self, box_size):
        """Clip the Bézier curve to fit within a bounding box (not implemented)."""
        raise NotImplementedError

    @classmethod
    def from_repr(cls, bezier_repr):
        """Create a BezierCurve from neural network representation.

        Args:
            bezier_repr (list): [x1, y1, x2, y2, x3, y3, x4, y4, width] neural network output

        Returns:
            BezierCurve: reconstructed Bézier curve primitive
        """
        assert len(bezier_repr) == repr_len_by_type[PrimitiveType.PT_BEZIER]
        return cls(
            tuple(bezier_repr[0:2]),
            tuple(bezier_repr[2:4]),
            tuple(bezier_repr[4:6]),
            tuple(bezier_repr[6:8]),
            bezier_repr[8],
        )

    def to_repr(self):
        """Convert the Bézier curve to neural network representation."""
        cpoints_direct = (self.cpoint1, self.cpoint2, self.cpoint3, self.cpoint4)
        cpoints_inverse = tuple(coord for point in reversed(cpoints_direct) for coord in point)
        cpoints_direct = tuple(coord for point in cpoints_direct for coord in point)
        if cpoints_direct < cpoints_inverse:
            return cpoints_direct + (self.width,)
        else:
            return cpoints_inverse + (self.width,)


class Arc(GraphicsPrimitive):
    def __init__(self, center, radius, angle1, angle2, width):
        self.center = center
        self.radius = radius
        self.angle1 = angle1
        self.angle2 = angle2
        self.width = width
        super().__init__()

    def draw(self, ctx):
        """return cairo_arc(ctx, self.to_repr())"""

    def clip_to_box(self, box_size):
        raise NotImplementedError

    @classmethod
    def from_repr(cls, arc_repr):
        assert len(arc_repr) == repr_len_by_type[PrimitiveType.PT_ARC]
        return cls(tuple(arc_repr[0:2]), arc_repr[2], arc_repr[3], arc_repr[4], arc_repr[5])

    def to_repr(self):
        return self.center + (self.radius, self.angle1, self.angle2, self.width)


__all__ = ["PT_LINE", "PT_ARC", "PT_QBEZIER", "PT_CBEZIER", "PT_POINT", "Line"]
