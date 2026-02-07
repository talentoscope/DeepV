import sys
import numpy as np
from pathlib import Path
from PIL import Image

# ensure repo root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from util_files.rendering.cairo import render
from util_files.data.graphics_primitives import PT_LINE

# make a square primitive in coordinate space 0..100
lines = np.array([
    [0.0, 0.0, 100.0, 0.0, 1.0],
    [100.0, 0.0, 100.0, 100.0, 1.0],
    [100.0, 100.0, 0.0, 100.0, 1.0],
    [0.0, 100.0, 0.0, 0.0, 1.0],
])

primitive_sets = {PT_LINE: lines,}

# render at a larger target size
out_w, out_h = 600, 600
img = render(primitive_sets, (out_w, out_h), data_representation='vahe')

# normalize and save
if img.dtype != 'uint8':
    img = (img * 255).astype('uint8')

if img.ndim == 3:
    img = img[:, :, 0]

out_path = Path('scripts/tmp_square_render.png')
Image.fromarray(img).save(out_path)
print('Saved', out_path)
