import sys
import random
from pathlib import Path
import json
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import svgpathtools
from tqdm import tqdm

# ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from util_files.rendering.cairo import render
from util_files.data.graphics_primitives import PT_LINE, PT_CBEZIER, PT_QBEZIER
from util_files.color_utils import img_8bit_to_float, gray_float_to_8bit
import scipy.ndimage as ndi
import util_files.data.transforms.ocrodeg_degrade as ocrodeg
import io
from PIL import Image as PILImage
from skimage.morphology import remove_small_objects
from skimage.draw import line
from skimage.draw import line
from skimage.morphology import remove_small_objects
import shutil

OUT_DIM = (1000, 1000)
BASE_DIR = Path("data/raw/floorplancad/train1")
OUT_DIR = Path("aug_test")
# ensure clean output directory
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Augmentation functions (no geometric distortions) ---
def aug_gaussian_blur(img, intensity):
    intensity = min(intensity, 0.6)  # Cap max intensity
    sigma = float(intensity) * 4.0
    return ndi.gaussian_filter(img, sigma=sigma)

def aug_binary_blur(img, intensity):
    intensity = min(intensity, 0.3)  # Cap max intensity
    blur = float(intensity) * 3.0
    return ocrodeg.binary_blur(img, blur)

def aug_noisy_binary_blur(img, intensity):
    intensity = min(intensity, 0.4)  # Cap max intensity
    blur = float(intensity) * 2.0
    sigma = float(intensity) * 0.5
    return ocrodeg.binary_blur(img, blur, noise=sigma)

def aug_random_blotches(img, intensity):
    intensity = min(intensity, 0.4)  # Further cap max intensity
    fgblobs = max(1e-6, float(intensity) * 3e-4)  # Reduce blob density
    bgblobs = max(1e-6, float(intensity) * 2e-4)
    fgscale = int(3 + float(intensity) * 30)  # Vary scale 3-33
    bgscale = int(3 + float(intensity) * 30)
    return ocrodeg.random_blotches(img, fgblobs, bgblobs, fgscale=fgscale, bgscale=bgscale)

def aug_thresholding(img, intensity):
    intensity = min(intensity, 0.4)  # Further cap max intensity
    intensity = intensity * 0.8  # Reduce extremity
    thresh_range = 0.3 + float(intensity) * 0.4  # Vary threshold range 0.3-0.7
    t = 0.2 + float(intensity) * thresh_range
    return 1.0 * (img > t)

def aug_kanungo(img, intensity):
    from util_files.data.transforms.kanungo_degrade import kanungo_degrade

    runs = max(1, int(1 + intensity * 3))  # Vary runs 1-4
    out = img.copy()
    for _ in range(runs):
        out = kanungo_degrade(out)
    return out

def aug_gaussian_noise(img, intensity):
    sigma = float(intensity) * 0.2
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)

def aug_salt_pepper(img, intensity):
    amount = float(intensity) * 0.06
    out = img.copy()
    num = int(amount * img.size)
    if num <= 0:
        return out
    ys = np.random.randint(0, img.shape[0], num)
    xs = np.random.randint(0, img.shape[1], num)
    vals = np.random.choice([0.0, 1.0], num)
    out[ys, xs] = vals
    return out

def aug_contrast(img, intensity):
    intensity = min(intensity, 0.5)  # Cap max intensity
    intensity = intensity * 0.8  # Reduce extremity
    mean = img.mean()
    factor = 1.0 + (float(intensity) - 0.5) * 1.5
    out = (img - mean) * factor + mean
    return np.clip(out, 0.0, 1.0)

def aug_brightness(img, intensity):
    intensity = intensity * 0.8  # Reduce extremity
    shift = (float(intensity) - 0.5) * 0.6
    return np.clip(img + shift, 0.0, 1.0)

def aug_speckle(img, intensity):
    intensity = min(intensity, 0.5)  # Cap max intensity
    sigma = float(intensity) * 0.15
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + img * noise, 0.0, 1.0)

def aug_jpeg_artifact(img, intensity):
    intensity = min(intensity, 0.7)  # Cap max intensity
    q = int(100 - float(intensity) * 70)
    q = max(10, min(95, q))
    arr8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    pil = PILImage.fromarray(arr8).convert('L')
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    pil2 = PILImage.open(buf).convert('L')
    return np.asarray(pil2).astype(np.float32) / 255.0

def aug_morph_dilation(img, intensity):
    intensity = min(intensity, 0.6)  # Cap max intensity
    size = 1 + int(float(intensity) * 4)
    return ndi.grey_dilation(img, size=(size, size))

def aug_morph_erosion(img, intensity):
    size = 1 + int(float(intensity) * 4)
    return ndi.grey_erosion(img, size=(size, size))

def aug_unsharp(img, intensity):
    sigma = 1.0 + float(intensity) * 2.0
    blurred = ndi.gaussian_filter(img, sigma=sigma)
    amount = float(intensity) * 1.5
    return np.clip(img + (img - blurred) * amount, 0.0, 1.0)

# Additional augmentations
def aug_ink_bleed(img, intensity):
    intensity = min(intensity, 0.4)  # Cap max intensity
    # Small morphological dilation to simulate ink bleed
    size = 1 + int(float(intensity) * 2)
    return ndi.grey_dilation(img, size=(size, size))

def aug_stroke_erosion(img, intensity):
    # Morphological erosion to thin lines
    size = 1 + int(float(intensity) * 2)
    return ndi.grey_erosion(img, size=(size, size))

def aug_scanline_artifacts(img, intensity):
    # Add faint horizontal stripes
    h, w = img.shape
    stripe_freq = 10 + int(float(intensity) * 20)
    stripes = np.sin(np.arange(h) * 2 * np.pi / stripe_freq) * 0.05 * float(intensity)
    stripes = stripes[:, np.newaxis]
    return np.clip(img + stripes, 0.0, 1.0)

def aug_crease_shadows(img, intensity):
    # Add faint diagonal darker bands
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    crease = np.sin((X + Y) * 0.01) * 0.1 * float(intensity)
    return np.clip(img + crease, 0.0, 1.0)

def aug_paper_texture(img, intensity):
    # Add subtle noise pattern
    noise = np.random.normal(0, 0.02 * float(intensity), img.shape)
    return np.clip(img + noise, 0.0, 1.0)

def aug_yellowing(img, intensity):

    # Slight sepia tint to background (white areas)
    tint = np.array([0.95, 0.9, 0.8])  # sepia
    tinted = img * (1 - 0.1 * float(intensity)) + 0.1 * float(intensity) * tint[0]  # approximate
    return np.clip(tinted, 0.0, 1.0)

def aug_fax_stripes(img, intensity):
    # Horizontal stripes + impulsive noise
    h, w = img.shape
    stripes = np.zeros_like(img)
    for i in range(0, h, 20):
        stripes[i:i+2, :] = 0.1 * float(intensity)
    # Add impulsive noise
    num_noise = int(float(intensity) * img.size * 0.01)
    ys = np.random.randint(0, h, num_noise)
    xs = np.random.randint(0, w, num_noise)
    vals = np.random.choice([0.0, 1.0], num_noise)
    stripes[ys, xs] = vals
    return np.clip(img + stripes, 0.0, 1.0)

def aug_dithering(img, intensity):
    intensity = min(intensity, 0.5)  # Cap max intensity
    # Simple ordered dithering with variable pattern size
    pattern_size = 2 + int(float(intensity) * 2)  # Vary 2-4
    threshold_map = np.random.rand(pattern_size, pattern_size)
    threshold_map = (threshold_map - threshold_map.min()) / (threshold_map.max() - threshold_map.min())
    h, w = img.shape
    tiled = np.tile(threshold_map, (h//pattern_size + 1, w//pattern_size + 1))[:h, :w]
    dithered = (img > tiled * float(intensity)).astype(float)
    return dithered

def aug_webp_artifacts(img, intensity):
    intensity = min(intensity, 0.6)  # Cap max intensity
    # Simulate lossy compression by JPEG
    q = int(100 - float(intensity) * 80)
    q = max(5, min(95, q))
    arr8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    pil = PILImage.fromarray(arr8).convert('L')
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    pil2 = PILImage.open(buf).convert('L')
    return np.asarray(pil2).astype(np.float32) / 255.0

def aug_moire_interference(img, intensity):
    # Overlay faint grid
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    grid = (np.sin(X * 0.1) + np.sin(Y * 0.1)) * 0.02 * float(intensity)
    return np.clip(img + grid, 0.0, 1.0)

def aug_quantization_posterization(img, intensity):
    # Reduce to few levels
    levels = 2 + int(float(intensity) * 4)  # Reduced from 8
    quantized = np.round(img * (levels - 1)) / (levels - 1)
    return quantized

def aug_gap_filler(img, intensity):
    # Morphological closing to fill small gaps
    size = 1 + int(float(intensity) * 2)
    return ndi.grey_closing(img, size=(size, size))

def aug_dirt_blobs(img, intensity):
    # Add small filled circles with variable number
    h, w = img.shape
    num_blobs = int(float(intensity) * 8) + 1  # Vary 1-9
    out = img.copy()
    for _ in range(num_blobs):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        radius = np.random.randint(1, 5)
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - y)**2 + (xx - x)**2 <= radius**2
        out[mask] = np.random.choice([0.0, 1.0])
    return out

def aug_low_bit_depth(img, intensity):
    # Reduce to few anti-aliasing levels
    levels = 2 + int(float(intensity) * 6)
    return np.round(img * (levels - 1)) / (levels - 1)

def aug_inverted_patches(img, intensity):
    intensity = min(intensity, 0.6)  # Cap max intensity
    # Flip small regions
    h, w = img.shape
    num_patches = int(float(intensity) * 5)
    out = img.copy()
    for _ in range(num_patches):
        y1 = np.random.randint(0, h//2)
        x1 = np.random.randint(0, w//2)
        y2 = y1 + np.random.randint(10, 50)
        x2 = x1 + np.random.randint(10, 50)
        y2 = min(y2, h)
        x2 = min(x2, w)
        out[y1:y2, x1:x2] = 1.0 - out[y1:y2, x1:x2]
    return out

def aug_blueprint_negative(img, intensity):
    intensity = min(intensity, 0.3)  # Further cap max intensity
    # Invert and add cyan tint with variable strength
    inverted = 1.0 - img
    cyan_tint_strength = 0.05 + 0.1 * float(intensity)  # Vary tint 0.05-0.15
    cyan_tint = np.array([0.8, 0.9, 1.0])  # cyan
    tinted = inverted * (1 - cyan_tint_strength) + cyan_tint_strength * cyan_tint[2]  # approximate
    return np.clip(tinted, 0.0, 1.0)

def aug_adaptive_threshold_artifacts(img, intensity):
    intensity = min(intensity, 0.3)  # Further cap max intensity
    # Simulate patchy binarization with variable block size
    h, w = img.shape
    block_size = int(10 + float(intensity) * 20)  # Vary block size 10-30
    out = img.copy()
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            thresh = block.mean() + (np.random.random() - 0.5) * 0.1 * float(intensity)
            out[i:i+block_size, j:j+block_size] = (block > thresh).astype(float)
    return out

def aug_sauvola_halos(img, intensity):
    # Add halos around lines
    # Approximate by dilating and subtracting
    dilated = ndi.grey_dilation(img, size=3)
    halo = dilated - img
    return np.clip(img + halo * 0.3 * float(intensity), 0.0, 1.0)

def aug_bernsen_artifacts(img, intensity):
    intensity = min(intensity, 0.3)  # Further cap max intensity
    # Grid-like steps with variable block size
    h, w = img.shape
    block_size = int(5 + float(intensity) * 15)  # Vary block size 5-20
    out = img.copy()
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            min_val = block.min()
            max_val = block.max()
            thresh = (min_val + max_val) / 2
            out[i:i+block_size, j:j+block_size] = (block > thresh).astype(float)
    # Add grid effect with variable strength
    grid_strength = 0.02 * float(intensity)
    grid = np.zeros_like(img)
    grid[::block_size, :] = grid_strength
    grid[:, ::block_size] = grid_strength
    return np.clip(out + grid, 0.0, 1.0)

# New augmentations from user suggestions

# Document aging / physical degradation
def aug_foxing(img, intensity):
    # Brownish irregular spots near borders
    h, w = img.shape
    num_spots = int(float(intensity) * 20)
    out = img.copy()
    for _ in range(num_spots):
        # Prefer borders
        if random.random() < 0.7:
            x = random.randint(0, w//4) if random.random() < 0.5 else random.randint(3*w//4, w-1)
            y = random.randint(0, h//4) if random.random() < 0.5 else random.randint(3*h//4, h-1)
        else:
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
        radius = random.randint(2, 8)
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - y)**2 + (xx - x)**2) <= radius**2
        # Soft edges with gaussian
        soft_mask = ndi.gaussian_filter(mask.astype(float), sigma=radius/3)
        out -= soft_mask * 0.2 * float(intensity)  # Darken
    return np.clip(out, 0.0, 1.0)

def aug_browning(img, intensity):
    # Vignette-like yellow→brown gradient from edges
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h/2, w/2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = dist / max_dist
    # Darken towards edges
    darkening = vignette * 0.3 * float(intensity)
    return np.clip(img - darkening, 0.0, 1.0)

def aug_paper_fold_creases(img, intensity):
    # Thin diagonal bands
    h, w = img.shape
    out = img.copy()
    num_creases = random.randint(1, 3)
    for _ in range(num_creases):
        angle = random.uniform(-45, 45)
        thickness = random.randint(1, 3)
        # Simple diagonal line
        if abs(angle) < 22.5:
            # Horizontal-ish
            y_pos = random.randint(0, h-1)
            out[y_pos:y_pos+thickness, :] -= 0.1 * float(intensity)
        else:
            # Vertical-ish
            x_pos = random.randint(0, w-1)
            out[:, x_pos:x_pos+thickness] -= 0.1 * float(intensity)
    return np.clip(out, 0.0, 1.0)

# Scanner / copier / fax
def aug_roller_marks(img, intensity):
    # Faint vertical lines
    h, w = img.shape
    spacing = random.randint(30, 80)  # ~3-8 cm at 1000px
    out = img.copy()
    for x in range(0, w, spacing):
        width = random.randint(1, 3)
        out[:, x:x+width] -= 0.05 * float(intensity)
    return np.clip(out, 0.0, 1.0)

def aug_toner_starvation(img, intensity):
    # Faint lighter horizontal bands
    h, w = img.shape
    spacing = random.randint(50, 150)
    out = img.copy()
    for y in range(0, h, spacing):
        height = random.randint(5, 15)
        out[y:y+height, :] += 0.1 * float(intensity)
    return np.clip(out, 0.0, 1.0)

def aug_halftone_rosette(img, intensity):
    # Faint dot grid at 45°
    h, w = img.shape
    spacing = 10
    out = img.copy()
    for i in range(0, h+spacing, spacing):
        for j in range(0, w+spacing, spacing):
            # Offset for 45°
            offset = (i + j) % (2*spacing)
            if offset < spacing:
                y, x = i, j + offset
                if y < h and x < w:
                    radius = 1
                    yy, xx = np.ogrid[:h, :w]
                    mask = ((yy - y)**2 + (xx - x)**2) <= radius**2
                    out[mask] -= 0.02 * float(intensity)
    return np.clip(out, 0.0, 1.0)

# Modern digital degradation
def aug_subpixel_jitter(img, intensity):
    # Small perturbations along lines (approximate)
    h, w = img.shape
    noise = np.random.normal(0, 0.005 * float(intensity), (h, w))
    return np.clip(img + noise, 0.0, 1.0)

def aug_rolling_shutter(img, intensity):
    # Wavy horizontal displacement
    h, w = img.shape
    displacement = np.sin(np.arange(h) * 0.01) * 5 * float(intensity)
    out = np.zeros_like(img)
    for y in range(h):
        shift = int(displacement[y])
        if shift > 0:
            out[y, shift:] = img[y, :-shift]
        elif shift < 0:
            out[y, :shift] = img[y, -shift:]
        else:
            out[y] = img[y]
    return out

# Hand-drawn / tracing paper
def aug_wobbly_stroke(img, intensity):
    # Perpendicular noise along strokes (approximate with edge noise)
    edges = ndi.sobel(img)
    noise = np.random.normal(0, 0.01 * float(intensity), img.shape)
    return np.clip(img + edges * noise, 0.0, 1.0)

def aug_correction_fluid(img, intensity):
    # Irregular white blobs
    h, w = img.shape
    num_blobs = int(float(intensity) * 10)
    out = img.copy()
    for _ in range(num_blobs):
        y = random.randint(0, h-1)
        x = random.randint(0, w-1)
        radius = random.randint(5, 20)
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - y)**2 + (xx - x)**2) <= radius**2
        soft_mask = ndi.gaussian_filter(mask.astype(float), sigma=radius/4)
        out += soft_mask * 0.5 * float(intensity)  # Whiten
    return np.clip(out, 0.0, 1.0)

# Printer / plotter specific
def aug_banding(img, intensity):
    # Regular horizontal stripes
    h, w = img.shape
    period = random.randint(8, 30)
    amplitude = 0.05 * float(intensity)
    stripes = np.sin(np.arange(h) * 2 * np.pi / period) * amplitude
    stripes = stripes[:, np.newaxis]
    return np.clip(img + stripes, 0.0, 1.0)

def aug_pen_skipping(img, intensity):
    # Small gaps in lines (approximate by eroding thin parts)
    # Simple: add noise that removes pixels
    noise = np.random.random(img.shape) < 0.01 * float(intensity)
    return np.clip(img - noise, 0.0, 1.0)

# Additional new augmentations from latest suggestions
def aug_shadow_from_neighboring_page(img, intensity):
    h, w = img.shape
    out = img.copy()
    num_shadows = random.randint(2, 4)
    for _ in range(num_shadows):
        angle = random.uniform(10, 80)  # Diagonal or vertical
        thickness = random.randint(20, 60)
        darkness = 0.1 * float(intensity)
        # Create a gradient band
        if random.random() < 0.5:  # Diagonal
            # Simple diagonal gradient
            for i in range(thickness):
                offset = int(i * np.tan(np.radians(angle)))
                y_start = random.randint(0, h - thickness)
                x_start = random.randint(0, w - thickness)
                for dy in range(thickness):
                    dx = int(dy * np.tan(np.radians(angle)))
                    if 0 <= y_start + dy < h and 0 <= x_start + dx < w:
                        out[y_start + dy, x_start + dx] -= darkness * (1 - dy / thickness)
        else:  # Vertical
            x_pos = random.randint(0, w - thickness)
            gradient = np.linspace(darkness, 0, thickness)
            out[:, x_pos:x_pos+thickness] -= gradient[np.newaxis, :]
    return np.clip(out, 0.0, 1.0)

def aug_curved_page_distortion(img, intensity):
    from scipy.ndimage import map_coordinates
    h, w = img.shape
    # Barrel distortion
    y, x = np.mgrid[:h, :w]
    center_y, center_x = h/2, w/2
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_r = np.sqrt(center_x**2 + center_y**2)
    distortion = 1 + (r / max_r)**2 * 0.05 * float(intensity)  # Barrel
    x_new = center_x + (x - center_x) * distortion
    y_new = center_y + (y - center_y) * distortion
    coords = np.array([y_new, x_new])
    out = map_coordinates(img, coords, order=1, mode='constant', cval=1.0)
    return out

def aug_alignment_error(img, intensity):
    h, w = img.shape
    shift_x = random.randint(1, 12) * (1 if random.random() < 0.5 else -1)
    shift_y = random.randint(1, 12) * (1 if random.random() < 0.5 else -1)
    out = np.roll(img, shift=(shift_y, shift_x), axis=(0, 1))
    # Add faint shadow at edge
    if shift_x > 0:
        out[:, :shift_x] -= 0.05 * float(intensity)
    elif shift_x < 0:
        out[:, shift_x:] -= 0.05 * float(intensity)
    if shift_y > 0:
        out[:shift_y, :] -= 0.05 * float(intensity)
    elif shift_y < 0:
        out[shift_y:, :] -= 0.05 * float(intensity)
    return np.clip(out, 0.0, 1.0)

def aug_photocopy_ghosting(img, intensity):
    shift_y = random.randint(5, 30)
    shift_x = random.randint(-10, 10)
    ghost = np.roll(img, shift=(shift_y, shift_x), axis=(0, 1))
    ghost = 1.0 - ghost  # Invert
    ghost = ndi.gaussian_filter(ghost, sigma=3)
    return np.clip(img + ghost * 0.1 * float(intensity), 0.0, 1.0)

def aug_grid_bleed(img, intensity):
    h, w = img.shape
    spacing = 50  # mm grid
    out = img.copy()
    for y in range(0, h, spacing):
        out[y, :] += 0.02 * float(intensity)
    for x in range(0, w, spacing):
        out[:, x] += 0.02 * float(intensity)
    return np.clip(out, 0.0, 1.0)

def aug_eraser_smudge(img, intensity):
    h, w = img.shape
    num_patches = random.randint(3, 8)
    out = img.copy()
    for _ in range(num_patches):
        y = random.randint(0, h-20)
        x = random.randint(0, w-20)
        patch = out[y:y+20, x:x+20]
        smeared = ndi.gaussian_filter(patch, sigma=2)
        out[y:y+20, x:x+20] = smeared
    return out

def aug_sticky_tape_shadow(img, intensity):
    h, w = img.shape
    out = img.copy()
    # Rectangle
    y1 = random.randint(0, h//2)
    x1 = random.randint(0, w//2)
    height = random.randint(20, 100)
    width = random.randint(50, 200)
    y2 = min(y1 + height, h)
    x2 = min(x1 + width, w)
    tape = np.ones((y2-y1, x2-x1)) * 0.05 * float(intensity)  # Brighten
    # Add noise
    tape += np.random.normal(0, 0.01, tape.shape)
    out[y1:y2, x1:x2] += tape
    return np.clip(out, 0.0, 1.0)

def aug_hole_punch_shadows(img, intensity):
    h, w = img.shape
    out = img.copy()
    num_holes = random.randint(2, 4)
    for i in range(num_holes):
        y = random.randint(h//4, 3*h//4)
        x = random.randint(0, w//8)  # Near left edge
        radius = random.randint(10, 25)
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - y)**2 + (xx - x)**2) <= radius**2
        soft_mask = ndi.gaussian_filter(mask.astype(float), sigma=radius/4)
        out -= soft_mask * 0.2 * float(intensity)
    return np.clip(out, 0.0, 1.0)

def aug_thermal_paper_fading(img, intensity):
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h/2, w/2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = dist / max_dist
    # Fade middle grays
    faded = img.copy()
    mask = (img > 0.2) & (img < 0.8)
    faded[mask] += vignette[mask] * 0.3 * float(intensity)
    return np.clip(faded, 0.0, 1.0)

def aug_crease_highlight(img, intensity):
    h, w = img.shape
    out = img.copy()
    num_creases = random.randint(1, 3)
    for _ in range(num_creases):
        angle = random.uniform(-45, 45)
        thickness = random.randint(1, 3)
        # Bright band
        if abs(angle) < 22.5:
            y_pos = random.randint(0, h-1)
            out[y_pos:y_pos+thickness, :] += 0.1 * float(intensity)
        else:
            x_pos = random.randint(0, w-1)
            out[:, x_pos:x_pos+thickness] += 0.1 * float(intensity)
    return np.clip(out, 0.0, 1.0)

def aug_marker_bleed_through(img, intensity):
    # Faint mirrored blurred copy
    mirrored = np.flipud(img)  # Mirror vertically
    blurred = ndi.gaussian_filter(mirrored, sigma=5)
    return np.clip(img + blurred * 0.05 * float(intensity), 0.0, 1.0)

def aug_coffee_ring_stain(img, intensity):
    h, w = img.shape
    out = img.copy()
    num_rings = random.randint(1, 2)
    for _ in range(num_rings):
        y = random.randint(h//4, 3*h//4)
        x = random.randint(w//4, 3*w//4)
        # Two rings
        for r in [20, 40]:
            yy, xx = np.ogrid[:h, :w]
            mask = ((yy - y)**2 + (xx - x)**2) <= r**2
            inner_mask = ((yy - y)**2 + (xx - x)**2) <= (r-5)**2
            ring = mask & ~inner_mask
            soft_ring = ndi.gaussian_filter(ring.astype(float), sigma=2)
            out -= soft_ring * 0.15 * float(intensity)  # Brownish
    return np.clip(out, 0.0, 1.0)

def aug_fingerprint_oil(img, intensity):
    h, w = img.shape
    out = img.copy()
    num_prints = random.randint(3, 8)
    for _ in range(num_prints):
        y = random.randint(0, h-1)
        x = random.randint(0, w-1)
        radius_y = random.randint(8, 35)
        radius_x = random.randint(8, 35)
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - y)/radius_y)**2 + ((xx - x)/radius_x)**2 <= 1
        soft_mask = ndi.gaussian_filter(mask.astype(float), sigma=3)
        out -= soft_mask * 0.05 * float(intensity)
    return np.clip(out, 0.0, 1.0)

def aug_bad_autocrop_border(img, intensity):
    h, w = img.shape
    border_width = random.randint(5, 60)
    out = img.copy()
    out[:border_width, :] = 0.0
    out[-border_width:, :] = 0.0
    out[:, :border_width] = 0.0
    out[:, -border_width:] = 0.0
    return out

def aug_despeckle_overaggressive(img, intensity):
    intensity = min(intensity, 0.05)  # Further cap max intensity
    # Remove small black pixels with variable min_size
    from skimage.morphology import remove_small_objects
    min_size = int(1 + float(intensity) * 10)  # Vary min_size 1-11
    binary = img < 0.5
    cleaned = remove_small_objects(binary, min_size=min_size)
    return cleaned.astype(float)

def aug_tone_compression_clipping(img, intensity):
    intensity = min(intensity, 0.5)  # Cap max intensity
    # Strong contrast + clip
    out = (img - 0.5) * (1 + 2 * float(intensity)) + 0.5
    out = np.clip(out, 0.08, 0.92)
    return out

def aug_hand_drawn_hatching_noise(img, intensity):
    # Perturb hatches - add small random lines
    h, w = img.shape
    out = img.copy()
    num_lines = int(float(intensity) * 20)
    for _ in range(num_lines):
        y1 = random.randint(0, h-1)
        x1 = random.randint(0, w-1)
        length = random.randint(5, 20)
        angle = random.uniform(0, np.pi)
        y2 = y1 + int(length * np.sin(angle))
        x2 = x1 + int(length * np.cos(angle))
        # Draw line
        rr, cc = line(y1, x1, y2, x2)
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        out[rr[valid], cc[valid]] += 0.05 * float(intensity)
    return np.clip(out, 0.0, 1.0)

def aug_blueprint_fading_uneven(img, intensity):
    # Perlin noise field × radial gradient
    h, w = img.shape
    # Simple noise
    noise = np.random.normal(0, 1, (h, w))
    noise = ndi.gaussian_filter(noise, sigma=50)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h/2, w/2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    gradient = dist / max_dist
    fading = noise * gradient * 0.2 * float(intensity)
    return np.clip(img + fading, 0.0, 1.0)

def aug_microfiche_grid(img, intensity):
    # Fine rectangular grid + pincushion
    h, w = img.shape
    out = img.copy()
    spacing = 20
    for y in range(0, h, spacing):
        out[y, :] -= 0.02 * float(intensity)
    for x in range(0, w, spacing):
        out[:, x] -= 0.02 * float(intensity)
    # Pincushion distortion
    y, x = np.mgrid[:h, :w]
    center_y, center_x = h/2, w/2
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_r = np.sqrt(center_x**2 + center_y**2)
    distortion = 1 - (r / max_r)**2 * 0.02 * float(intensity)
    x_new = center_x + (x - center_x) * distortion
    y_new = center_y + (y - center_y) * distortion
    from scipy.ndimage import map_coordinates
    coords = np.array([y_new, x_new])
    out = map_coordinates(out, coords, order=1, mode='constant', cval=1.0)
    return out

def aug_halftone_moiré_2(img, intensity):
    # Second rotated dot grid
    h, w = img.shape
    spacing = 10
    out = img.copy()
    angle = 30  # degrees
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    for i in range(0, h+spacing, spacing):
        for j in range(0, w+spacing, spacing):
            x_rot = j * cos_a - i * sin_a
            y_rot = j * sin_a + i * cos_a
            offset = (x_rot + y_rot) % (2*spacing)
            if offset < spacing:
                y, x = i, j
                if y < h and x < w:
                    radius = 1
                    yy, xx = np.ogrid[:h, :w]
                    mask = ((yy - y)**2 + (xx - x)**2) <= radius**2
                    out[mask] -= 0.02 * float(intensity)
    return np.clip(out, 0.0, 1.0)

# map name -> function
AUG_FNS = {
    'gaussian_blur': aug_gaussian_blur,
    'binary_blur': aug_binary_blur,
    'noisy_binary_blur': aug_noisy_binary_blur,
    'random_blotches': aug_random_blotches,
    'thresholding': aug_thresholding,
    'kanungo': aug_kanungo,
    'gaussian_noise': aug_gaussian_noise,
    'salt_pepper': aug_salt_pepper,
    'contrast': aug_contrast,
    'brightness': aug_brightness,
    'speckle': aug_speckle,
    'jpeg_artifact': aug_jpeg_artifact,
    'dilate': aug_morph_dilation,
    'erode': aug_morph_erosion,
    'unsharp': aug_unsharp,
    'ink_bleed': aug_ink_bleed,
    'stroke_erosion': aug_stroke_erosion,
    'scanline_artifacts': aug_scanline_artifacts,
    'crease_shadows': aug_crease_shadows,
    'paper_texture': aug_paper_texture,
    'yellowing': aug_yellowing,
    'fax_stripes': aug_fax_stripes,
    'dithering': aug_dithering,
    'webp_artifacts': aug_webp_artifacts,
    'moire_interference': aug_moire_interference,
    'quantization_posterization': aug_quantization_posterization,
    'gap_filler': aug_gap_filler,
    'dirt_blobs': aug_dirt_blobs,
    'low_bit_depth': aug_low_bit_depth,
    'blueprint_negative': aug_blueprint_negative,
    'adaptive_threshold_artifacts': aug_adaptive_threshold_artifacts,
    'sauvola_halos': aug_sauvola_halos,
    'bernsen_artifacts': aug_bernsen_artifacts,
    'foxing': aug_foxing,
    'browning': aug_browning,
    'paper_fold_creases': aug_paper_fold_creases,
    'roller_marks': aug_roller_marks,
    'toner_starvation': aug_toner_starvation,
    'halftone_rosette': aug_halftone_rosette,
    'subpixel_jitter': aug_subpixel_jitter,
    'rolling_shutter': aug_rolling_shutter,
    'wobbly_stroke': aug_wobbly_stroke,
    'correction_fluid': aug_correction_fluid,
    'banding': aug_banding,
    'pen_skipping': aug_pen_skipping,
    'shadow_from_neighboring_page': aug_shadow_from_neighboring_page,
    'grid_bleed': aug_grid_bleed,
    'eraser_smudge': aug_eraser_smudge,
    'eraser_smudge': aug_eraser_smudge,
    'sticky_tape_shadow': aug_sticky_tape_shadow,
    'hole_punch_shadows': aug_hole_punch_shadows,
    'thermal_paper_fading': aug_thermal_paper_fading,
    'crease_highlight': aug_crease_highlight,
    'marker_bleed_through': aug_marker_bleed_through,
    'coffee_ring_stain': aug_coffee_ring_stain,
    'fingerprint_oil': aug_fingerprint_oil,
    'despeckle_overaggressive': aug_despeckle_overaggressive,
    'tone_compression_clipping': aug_tone_compression_clipping,
    'hand_drawn_hatching_noise': aug_hand_drawn_hatching_noise,
    'blueprint_fading_uneven': aug_blueprint_fading_uneven,
    'microfiche_grid': aug_microfiche_grid,
    'halftone_moiré_2': aug_halftone_moiré_2,
}

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
        except Exception:
            continue

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
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).convert('L').save(path)


def run(n=250):
    svg_files = list(BASE_DIR.glob('*.svg'))
    if len(svg_files) == 0:
        print('No SVGs found in', BASE_DIR)
        return

    chosen = random.sample(svg_files, min(n, len(svg_files)))

    # enumerate starting at 0 so files are numbered 000.. (n-1)
    for idx, svg in enumerate(tqdm(chosen, desc="Processing SVGs"), start=0):
        print(f'({idx:03d}/{len(chosen)-1:03d}) {svg.name}')
        prim_sets = parse_svg_to_primitives(svg)
        if sum([0 if a.size==0 else 1 for a in prim_sets.values()]) == 0:
            print('  No supported primitives, skipping')
            continue

        rendered = render(prim_sets, OUT_DIM, data_representation='vahe')
        if rendered.dtype == np.uint8:
            arrf = rendered.astype(np.float32) / 255.0
        else:
            arrf = rendered.astype(np.float32)
            if arrf.max() > 1.0:
                arrf = arrf / 255.0

        base_name = f'{idx:03d}'
        clean_path = OUT_DIR / f'{base_name}_clean.png'
        save_uint8_gray(arrf, clean_path)

        num_augs = random.randint(1, 4)
        chosen_augs = random.sample(list(AUG_FNS.keys()), num_augs)
        meta = {'svg': str(svg), 'base_name': base_name, 'augs': []}

        aug_img = arrf.copy()
        for a_name in chosen_augs:
            intensity = random.random()
            # Sample an additional numeric parameter ('n') to vary counts/scales
            # This is logged in metadata so analysis can test whether counts
            # (not just intensity) correlate with bad images.
            n_param = int(1 + intensity * 10)
            fn = AUG_FNS[a_name]
            try:
                aug_img = fn(aug_img, intensity)
            except Exception as e:
                print('  Aug failed', a_name, e)
                continue
            # record both intensity and the sampled numeric parameter
            meta['augs'].append({'name': a_name, 'intensity': float(intensity), 'params': {'n': int(n_param)}})

        out_path = OUT_DIR / f'{base_name}_aug.png'
        save_uint8_gray(aug_img, out_path)

        meta_path = OUT_DIR / f'{base_name}_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

if __name__ == '__main__':
    run(250)
