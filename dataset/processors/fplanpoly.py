from pathlib import Path
import shutil
from .base import Processor
from typing import Dict, List, Any
import os


class FPLANPOLYProcessor(Processor):
    """Processor for FPLAN-POLY dataset.

    Copies DXF vector files (floorplans and symbols) to the vector directory.
    The dataset contains 42 floorplan DXF files and 38 symbol model DXF files.
    """

    def standardize(self, input_dir: Path, output_base: Path, dry_run: bool = True) -> Dict:
        input_dir = Path(input_dir)
        output_base = Path(output_base)
        vec_dir = output_base / 'vector' / 'fplanpoly'

        # Find DXF files in floorplans and symbols directories
        floorplans_dir = input_dir / 'Floorplans'
        symbols_dir = input_dir / 'Model Symbols'

        dxf_files = []

        # Collect floorplan DXF files
        if floorplans_dir.exists():
            dxf_files.extend(list(floorplans_dir.glob('*.dxf')))

        # Collect symbol DXF files
        if symbols_dir.exists():
            dxf_files.extend(list(symbols_dir.glob('*.dxf')))

        if dry_run:
            return {
                'dataset': 'fplanpoly',
                'dxf_count': len(dxf_files),
                'vec_dir': str(vec_dir),
                'dry_run': True,
            }

        vec_dir.mkdir(parents=True, exist_ok=True)

        dxf_copied = 0

        for dxf_file in dxf_files:
            # Use the filename as-is (they're already well-named)
            dest_path = vec_dir / dxf_file.name

            # Copy DXF file
            if not dest_path.exists():
                shutil.copy2(dxf_file, dest_path)
                dxf_copied += 1

        return {
            'dataset': 'fplanpoly',
            'dxf_count': len(dxf_files),
            'dxf_copied': dxf_copied,
        }