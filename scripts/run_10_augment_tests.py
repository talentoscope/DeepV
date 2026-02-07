import sys
import random
from pathlib import Path
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import svgpathtools

# ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from util_files.rendering.cairo import render
from util_files.data.graphics_primitives import PT_LINE, PT_CBEZIER, PT_QBEZIER
from util_files.data.transforms.degradation_models import DegradationGenerator
from util_files.color_utils import img_8bit_to_float, gray_float_to_8bit, rgb_to_gray

OUT_DIM = (600, 600)
BASE_DIR = Path("data/raw/floorplancad/train1")

def parse_svg_to_primitives(svg_path: Path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    primitives = []

    for path_elem in root.iter('{http://www.w3.org/2000/svg}path'):
        if 'd' not in path_elem.attrib:
            continue
        d_string = path_elem.attrib['d']
        try:
            path = svgpathtools.parse_path(d_string)
            for segment in path:
                if isinstance(segment, svgpathtools.Line):
                    x1, y1 = segment.start.real, segment.start.imag
                    x2, y2 = segment.end.real, segment.end.imag
                    stroke_width = 1.0
                    if 'stroke-width' in path_elem.attrib:
                        try:
                            stroke_width = float(path_elem.attrib['stroke-width'])
                        except Exception:
                            pass
                    primitives.append((PT_LINE, np.array([x1, y1, x2, y2, stroke_width])))
                # other segments (Bezier/Arc) can be added later
        except Exception as e:
            # skip unparsable paths
            continue

    # convert to VAHE-like dict expected by render
    primitive_sets = {PT_LINE: [], PT_CBEZIER: [], PT_QBEZIER: []}
    for t, p in primitives:
        primitive_sets[t].append(p)

    for k in list(primitive_sets.keys()):
        if len(primitive_sets[k]) > 0:
            primitive_sets[k] = np.vstack(primitive_sets[k])
        else:
            primitive_sets[k] = np.array([])

    return primitive_sets


def save_uint8_gray(arr, path: Path):
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.mean(arr, axis=2).astype(np.uint8)
    Image.fromarray(arr).convert('L').save(path)


def run_tests():
    svg_files = list(BASE_DIR.glob('*.svg'))
    if len(svg_files) == 0:
        print('No SVGs found in', BASE_DIR)
        return

    selected = random.sample(svg_files, min(10, len(svg_files)))

    # prepare degradation configurations similar to earlier runs
    deg_lists = [
        (['gaussian_blur'], 'gaussian_blur'),
        (['random_blotches'], 'random_blotches'),
        (['binary_blur'], 'binary_blur'),
        (['gaussian_blur',], 'gaussian_blur'),
        (['noisy_binary_blur'], 'noisy_binary_blur'),
        (['distort'], 'distort'),
    ]

    for idx, svg in enumerate(selected, start=1):
        print(f"Processing ({idx}/10): {svg.name}")
        prim_sets = parse_svg_to_primitives(svg)
        if sum([0 if a.size==0 else 1 for a in prim_sets.values()]) == 0:
            print('  No supported primitives found, skipping')
            continue

        # render clean (renderer will auto-scale/center)
        rendered = render(prim_sets, OUT_DIM, data_representation='vahe')
        out_clean = Path(f"test_{idx:02d}_clean.png")
        save_uint8_gray(rendered, out_clean)
        print('  Saved', out_clean)

        # create degraded variants
        for j, (dlist, dname) in enumerate(deg_lists, start=1):
            dg = DegradationGenerator(degradations_list=dlist, max_num_degradations=len(dlist))
            # convert rendered to float [0,1]
            arr8 = np.array(Image.open(out_clean).convert('L'))
            arrf = img_8bit_to_float(arr8)
            degraded = dg.do_degrade(arrf)
            out_path = Path(f"test_{idx:02d}_deg_{j}_{dname}.png")
            save_uint8_gray(gray_float_to_8bit(degraded) if degraded.dtype != np.uint8 else degraded, out_path)
            print('  Saved', out_path)

if __name__ == '__main__':
    run_tests()
