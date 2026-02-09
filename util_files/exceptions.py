"""
Custom exceptions for DeepV refinement and merging operations.
"""


class DeepVError(Exception):
    """Base exception class for DeepV operations."""

    pass


class RefinementError(DeepVError):
    """Base class for refinement-related errors."""

    pass


class MergingError(DeepVError):
    """Base class for merging-related errors."""

    pass


class EmptyPixelError(RefinementError):
    """Raised when no empty pixels are found in expected regions during refinement."""

    def __init__(self, message: str = "No empty pixels found in required region"):
        super().__init__(message)


class OptimizationError(RefinementError):
    """Raised when optimization fails during refinement."""

    def __init__(self, message: str = "Optimization failed"):
        super().__init__(message)


class RenderingError(RefinementError):
    """Raised when rendering operations fail."""

    def __init__(self, message: str = "Rendering operation failed"):
        super().__init__(message)


class ClippingError(MergingError):
    """Raised when line clipping operations fail."""

    def __init__(self, message: str = "Line clipping failed"):
        super().__init__(message)


class MergeError(MergingError):
    """Raised when merging operations fail."""

    def __init__(self, message: str = "Merging operation failed"):
        super().__init__(message)


class ConfigurationError(DeepVError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(message)
