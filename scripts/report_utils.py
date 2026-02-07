import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def load_patch_image(trace_dir: Path, patch_id: str):
    pdir = trace_dir / "patches" / patch_id
    # prefer PNG patch thumbnail
    png = pdir / "patch.png"
    if png.exists():
        return Image.open(png).convert("L")
    npz = pdir / "patch.npz"
    if npz.exists():
        data = np.load(npz)
        if "patch" in data:
            arr = data["patch"]
            # handle single-channel or 3-channel arrays robustly
            if arr.ndim == 3:
                if arr.shape[2] == 3:
                    return Image.fromarray(arr.astype('uint8')).convert("L")
                if arr.shape[2] == 1:
                    arr2 = arr.squeeze(axis=2)
                    return Image.fromarray(arr2.astype('uint8'))
            if arr.ndim == 2:
                return Image.fromarray(arr.astype('uint8'))
    return None


def load_model_output(trace_dir: Path, patch_id: str):
    pdir = trace_dir / "patches" / patch_id
    npf = pdir / "model_output.npz"
    if not npf.exists():
        return {}
    data = np.load(npf, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    return out


def normalize_vectors(arr, patch_size=64):
    # arr: (N, D) where D >= 4 -> (x1,y1,x2,y2,...)
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] < 4:
        return np.zeros((0, 6))
    N = a.shape[0]
    out = np.zeros((N, 6), dtype=float)
    out[:, :4] = a[:, :4]
    # width default
    if a.shape[1] >= 5:
        out[:, 4] = a[:, 4]
    else:
        out[:, 4] = 1.0
    # opacity/default
    if a.shape[1] >= 6:
        out[:, 5] = a[:, 5]
    else:
        out[:, 5] = 1.0
    # clamp to patch bounds
    out[:, :4] = np.clip(out[:, :4], 0, patch_size - 1)
    return out


def render_primitives_pil(vectors, size=(64, 64)):
    """Simple PIL line renderer for patch-sized primitives.
    vectors: (N,6) array: x1,y1,x2,y2,width,opacity
    returns: PIL.Image L mode
    """
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    for v in vectors:
        x1, y1, x2, y2, w, op = v.tolist()
        # ensure integer coords
        coords = [(float(x1), float(y1)), (float(x2), float(y2))]
        width = max(1, int(round(w)))
        draw.line(coords, fill=255, width=width)
    return img


def binarize_image(img: Image.Image, threshold=1):
    a = np.array(img)
    return (a > threshold).astype(np.uint8)


def pixel_iou(img_true: Image.Image, img_pred: Image.Image):
    t = binarize_image(img_true)
    p = binarize_image(img_pred)
    inter = (t & p).sum()
    union = (t | p).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def make_patch_composite(trace_dir: Path, patch_id: str, out_dir: Path, history_iters=None, patch_size=64):
    """Create composite image for a single patch and compute metrics.
    history_iters: (I, N, D) or None
    """
    _ensure_dir(out_dir)
    gt = load_patch_image(trace_dir, patch_id)
    model = load_model_output(trace_dir, patch_id)
    vec = None
    if "vector" in model:
        vec = model["vector"]
    elif "pred" in model:
        vec = model["pred"]
    if vec is None:
        vec = np.zeros((0, 6))
    vecn = normalize_vectors(vec, patch_size=patch_size)
    rendered_model = render_primitives_pil(vecn, size=(patch_size, patch_size))
    # ensure GT is same size as rendered images
    if gt is None:
        gt = Image.new("L", (patch_size, patch_size), 0)
    else:
        if gt.size != (patch_size, patch_size):
            gt = gt.resize((patch_size, patch_size), resample=Image.NEAREST)

    # final refined primitives: try to use primitive_history last if provided
    if history_iters is not None:
        try:
            pidx = int(patch_id)
            final_all = history_iters[-1]  # shape (num_patches, num_prims, D)
            if final_all.ndim == 3:
                final_vec = final_all[pidx]
            else:
                final_vec = final_all
            finaln = normalize_vectors(final_vec, patch_size=patch_size)
            rendered_final = render_primitives_pil(finaln, size=(patch_size, patch_size))
        except Exception:
            rendered_final = rendered_model
    else:
        rendered_final = rendered_model

    # overlay: color composite
    gt_rgb = Image.new("RGB", (patch_size * 3, patch_size))
    # left: GT, mid: model, right: final
    if gt is None:
        gt = Image.new("L", (patch_size, patch_size), 0)
    gt_rgb.paste(gt.convert("RGB"), (0, 0))
    gt_rgb.paste(rendered_model.convert("RGB"), (patch_size, 0))
    gt_rgb.paste(rendered_final.convert("RGB"), (patch_size * 2, 0))

    # compute IoU metrics
    iou_model = pixel_iou(gt, rendered_model)
    iou_final = pixel_iou(gt, rendered_final)

    out_file = out_dir / f"patch_{patch_id}.png"
    gt_rgb.save(out_file)

    metrics = {
        "patch_id": patch_id,
        "iou_model": iou_model,
        "iou_final": iou_final,
        "n_model_primitives": int(vecn.shape[0]),
        "n_final_primitives": int(vecn.shape[0]),
    }

    # per-iteration IOU curve if history provided as (I, num_patches, num_prims, D) for this patch
    if history_iters is not None:
        try:
            pidx = int(patch_id)
            ious = []
            for it in history_iters:
                # it expected shape (num_patches, num_prims, D)
                if it.ndim == 3:
                    vec_this = it[pidx]
                else:
                    vec_this = it
                r = render_primitives_pil(normalize_vectors(vec_this, patch_size=patch_size), size=(patch_size, patch_size))
                ious.append(pixel_iou(gt, r))
            metrics["iou_curve"] = ious
        except Exception:
            metrics["iou_curve"] = []

    return out_file, metrics


def select_patch_ids(trace_dir: Path, count=20):
    patches_dir = trace_dir / "patches"
    if not patches_dir.exists():
        return []
    ids = [p.name for p in sorted(patches_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else x.name)][:count]
    return ids
