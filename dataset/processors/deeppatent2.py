from pathlib import Path
import shutil
from .base import Processor
from typing import Dict


class DeepPatent2Processor(Processor):
    """Standardizer for DeepPatent2.

    Moves PNGs into `data/raster/deeppatent2`. If an 'Original_2020.tar.gz' archive
    is present in the raw dir, this processor will not attempt extraction; it
    assumes extraction is handled at download time.
    """

    def standardize(self, raw_dir: Path, output_base: Path, dry_run: bool = True) -> Dict:
        raw_dir = Path(raw_dir)
        output_base = Path(output_base)
        ras_dir = output_base / 'raster' / 'deeppatent2'

        pngs = list(raw_dir.rglob('*.png'))

        if dry_run:
            return {
                'dataset': 'deeppatent2',
                'png_count': len(pngs),
                'ras_dir': str(ras_dir),
                'dry_run': True,
            }

        ras_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for p in pngs:
            dest = ras_dir / p.name
            if dest.exists():
                continue
            shutil.copy2(p, dest)
            moved += 1

        return {
            'dataset': 'deeppatent2',
            'png_count': len(pngs),
            'moved': moved,
        }
