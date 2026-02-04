"""Download FloorPlanCAD up to max_items and verify results."""
from pathlib import Path
from dataset.downloaders import download_dataset as mod
import json

OUT = Path('data')
NAME = 'floorplancad'

print('Starting download for', NAME)
res = mod.download_dataset(NAME, OUT, test_mode=False, max_items=10000, prune=True)

ds_dir = OUT / 'raw' / NAME
RASTER_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
VECTOR_EXT = {'.svg', '.dxf', '.dwg', '.eps', '.ps'}

candidates = []
for p in ds_dir.rglob('*'):
    if p.is_file() and (p.suffix.lower() in RASTER_EXT or p.suffix.lower() in VECTOR_EXT):
        candidates.append(p)

overflow_dir = ds_dir / '_overflow'
overflow_count = 0
if overflow_dir.exists():
    overflow_count = sum(1 for p in overflow_dir.rglob('*') if p.is_file())

report = {
    'dataset': NAME,
    'metadata': res,
    'kept_count': len(candidates),
    'overflow_count': overflow_count,
}

print('Kept raster+vector files:', report['kept_count'])
print('Overflow files moved:', report['overflow_count'])

with open('floorplancad_download_report.json','w',encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print('Wrote floorplancad_download_report.json')
