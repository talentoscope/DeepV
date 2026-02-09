from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Processor(Protocol):
    """Protocol for dataset processors."""

    def standardize(self, raw_dir: Path, output_base: Path, dry_run: bool = True) -> dict:
        """Standardize raw dataset files.

        Args:
            raw_dir: Path to the raw downloaded data (e.g., data/raw/{dataset})
            output_base: Base output directory (e.g., data/)
            dry_run: If True, do not modify filesystem, only report actions

        Returns:
            metadata dict describing actions performed
        """
        raise NotImplementedError()
