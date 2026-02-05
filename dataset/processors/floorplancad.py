from pathlib import Path
import shutil
from .base import Processor
from typing import Dict
import base64


class FloorPlanCADProcessor(Processor):
    """Processor for FloorPlanCAD dataset.

    Loads Parquet files, extracts SVG and PNG data, and saves to vector/raster dirs.
    """

    def standardize(self, raw_dir: Path, output_base: Path, dry_run: bool = True) -> Dict:
        raw_dir = Path(raw_dir)
        output_base = Path(output_base)
        vec_dir = output_base / 'vector' / 'floorplancad'
        ras_dir = output_base / 'raster' / 'floorplancad'

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        # Load the dataset from Parquet files
        parquet_files = list(raw_dir.glob("*.parquet"))
        if not parquet_files:
            # Check for FiftyOne dataset
            samples_json = raw_dir / "samples.json"
            if samples_json.exists():
                if dry_run:
                    # Count PNGs as estimate
                    pngs = list(raw_dir.rglob('*.png'))
                    return {
                        'dataset': 'floorplancad',
                        'estimated_samples': len(pngs),
                        'vec_dir': str(vec_dir),
                        'ras_dir': str(ras_dir),
                        'dry_run': True,
                    }
                try:
                    import fiftyone as fo
                    dataset = fo.Dataset.from_json(str(samples_json))
                except ImportError:
                    raise ImportError("Install fiftyone: pip install fiftyone")

                vec_dir.mkdir(parents=True, exist_ok=True)
                ras_dir.mkdir(parents=True, exist_ok=True)

                actions = {'extracted': 0, 'skipped': 0}

                for sample in dataset:
                    sample_id = sample.id

                    # Extract SVG
                    if 'svg' in sample:
                        svg_content = sample['svg']
                        svg_path = vec_dir / f"{sample_id}.svg"
                        if not svg_path.exists():
                            with open(svg_path, 'w') as f:
                                f.write(svg_content)
                            actions['extracted'] += 1
                        else:
                            actions['skipped'] += 1

                    # Copy PNG
                    if 'png' in sample:
                        png_path_src = raw_dir / sample['png']
                        png_path_dst = ras_dir / f"{sample_id}.png"
                        if not png_path_dst.exists():
                            shutil.copy2(png_path_src, png_path_dst)
                            actions['extracted'] += 1
                        else:
                            actions['skipped'] += 1

                meta = {
                    'dataset': 'floorplancad',
                    'extracted': actions['extracted'],
                    'skipped': actions['skipped'],
                }
                return meta
            else:
                # Fallback to existing SVGs/PNGs if no Parquet or samples.json
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
                    shutil.copy2(s, vec_dir / s.name)
                for p in pngs:
                    shutil.copy2(p, ras_dir / p.name)
                return {'dataset': 'floorplancad', 'svg_count': len(svgs), 'png_count': len(pngs)}

        # Load dataset
        dataset = load_dataset("parquet", data_files=[str(p) for p in parquet_files])

        actions = {'extracted': 0, 'skipped': 0}

        if dry_run:
            # Estimate counts
            total_samples = sum(len(split) for split in dataset.values())
            return {
                'dataset': 'floorplancad',
                'estimated_samples': total_samples,
                'vec_dir': str(vec_dir),
                'ras_dir': str(ras_dir),
                'dry_run': True,
            }

        vec_dir.mkdir(parents=True, exist_ok=True)
        ras_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in dataset.items():
            for i, sample in enumerate(split_data):
                sample_id = f"{split_name}_{i}"

                # Extract SVG
                if 'svg' in sample:
                    svg_content = sample['svg']
                    if isinstance(svg_content, str):
                        svg_path = vec_dir / f"{sample_id}.svg"
                        if not svg_path.exists():
                            with open(svg_path, 'w') as f:
                                f.write(svg_content)
                            actions['extracted'] += 1
                        else:
                            actions['skipped'] += 1

                # Extract PNG
                if 'png' in sample:
                    png_data = sample['png']
                    if isinstance(png_data, str) and png_data.startswith('data:image/png;base64,'):
                        png_b64 = png_data.split(',')[1]
                        png_bytes = base64.b64decode(png_b64)
                        png_path = ras_dir / f"{sample_id}.png"
                        if not png_path.exists():
                            with open(png_path, 'wb') as f:
                                f.write(png_bytes)
                            actions['extracted'] += 1
                        else:
                            actions['skipped'] += 1

        meta = {
            'dataset': 'floorplancad',
            'extracted': actions['extracted'],
            'skipped': actions['skipped'],
        }
        return meta
