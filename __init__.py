"""DeepV package init.

Expose package metadata and provide a stable place for top-level exports.

This file intentionally contains minimal runtime logic — keep heavy imports
out of package-level initialization to avoid slow imports for CLI/tools.
"""

__all__: list[str] = []

__version__: str = "0.0.0"
