from pathlib import Path
import shutil
from .base import Processor
from typing import Dict


class FloorPlanCADProcessor(Processor):
    """Simple standardizer for FloorPlanCAD.

    Moves SVGs to `data/vector/floorplancad` and PNGs to `data/raster/floorplancad`.
    Writes a small metadata summary.
    """

    def standardize(self, raw_dir: Path, output_base: Path, dry_run: bool = True) -> Dict:
        raw_dir = Path(raw_dir)
        output_base = Path(output_base)
        vec_dir = output_base / 'vector' / 'floorplancad'
        ras_dir = output_base / 'raster' / 'floorplancad'

        actions = {'moved': [], 'skipped': []}

        # collect files
        svgs = list(raw_dir.rglob('*.svg'))
        pngs = list(raw_dir.rglob('*.png'))

        if dry_run:
            return {
                'dataset': 'floorplancad',
                'svg_count': len(svgs),
                'png_count': len(pngs),
                'vec_dir': str(vec_dir),
                'ras_dir': str(ras_dir),
                'dry_run': True,
            }

        vec_dir.mkdir(parents=True, exist_ok=True)
        ras_dir.mkdir(parents=True, exist_ok=True)

        for s in svgs:
            dest = vec_dir / s.name
            if dest.exists():
                actions['skipped'].append(str(s))
                continue
            shutil.copy2(s, dest)
            actions['moved'].append({'from': str(s), 'to': str(dest)})

        for p in pngs:
            dest = ras_dir / p.name
            if dest.exists():
                actions['skipped'].append(str(p))
                continue
            shutil.copy2(p, dest)
            actions['moved'].append({'from': str(p), 'to': str(dest)})

        # write small metadata
        meta = {
            'dataset': 'floorplancad',
            'svg_count': len(svgs),
            'png_count': len(pngs),
            'moved': len(actions['moved']),
            'skipped': len(actions['skipped']),
        }
        return meta
