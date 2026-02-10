import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


class Tracer:
    """Simple tracer to record per-image, per-patch and per-stage artifacts.

    Usage:
        tracer = Tracer(enabled=True, base_dir="output/traces", image_id="img1")
        tracer.save_patch(patch_id, patch_array, offset=(x,y))
        tracer.save_model_output(patch_id, dict_of_arrays)
        tracer.save_pre_refinement(primitives_list)
    """

    def __init__(
        self,
        enabled: bool,
        base_dir: str = "output/traces",
        image_id: str = "unknown",
        seed: int = None,
        device: str = "cpu",
    ):
        self.enabled = enabled
        self.base_dir = Path(base_dir)
        self.image_id = str(image_id)
        self.seed = seed
        self.device = device
        self.timestamp = datetime.now().isoformat()
        if not self.enabled:
            return
        self.image_dir = self.base_dir / self.image_id
        self.patches_dir = self.image_dir / "patches"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        # Save determinism metadata
        self._save_determinism_meta()

    def _enabled(self):
        return self.enabled

    def _save_determinism_meta(self):
        """Save determinism metadata for reproducibility."""
        if not self._enabled():
            return
        meta = {"seed": self.seed, "device": self.device, "timestamp": self.timestamp}
        with open(self.image_dir / "determinism.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def save_patch(self, patch_id: str, patch_array: np.ndarray, offset=None) -> None:
        if not self._enabled():
            return
        pdir = self.patches_dir / str(patch_id)
        pdir.mkdir(parents=True, exist_ok=True)
        # save image
        try:
            img = Image.fromarray(patch_array)
            img.save(pdir / "patch.png")
        except (ValueError, TypeError, OSError):
            # fallback: save as npz if PIL fails
            try:
                np.savez_compressed(pdir / "patch.npz", patch=patch_array)
            except Exception:
                pass  # Silent failure for tracing
        # save metadata
        meta = {"patch_id": str(patch_id), "offset": offset}
        with open(pdir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def save_model_output(self, patch_id: str, model_output: dict) -> None:
        if not self._enabled():
            return
        pdir = self.patches_dir / str(patch_id)
        pdir.mkdir(parents=True, exist_ok=True)
        # store numeric arrays in compressed npz and small metadata
        arrays = {k: v for k, v in model_output.items() if isinstance(v, (np.ndarray, list))}
        # convert lists to arrays
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        if arrays:
            np.savez_compressed(pdir / "model_output.npz", **arrays)
        meta = {k: v for k, v in model_output.items() if not isinstance(v, (np.ndarray, list))}
        if meta:
            with open(pdir / "model_output_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

    def save_pre_refinement(self, primitives: list) -> None:
        if not self._enabled():
            return
        with open(self.image_dir / "pre_refinement.json", "w", encoding="utf-8") as f:
            json.dump(primitives, f)

    def save_post_refinement(self, primitives: list) -> None:
        if not self._enabled():
            return
        with open(self.image_dir / "post_refinement.json", "w", encoding="utf-8") as f:
            json.dump(primitives, f)

    def save_merge_trace(self, merge_trace: dict) -> None:
        if not self._enabled():
            return
        with open(self.image_dir / "merge_trace.json", "w", encoding="utf-8") as f:
            json.dump(merge_trace, f)

    def save_provenance(self, prov: dict) -> None:
        if not self._enabled():
            return
        with open(self.image_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(prov, f)

    def save_iteration(self, iteration: int, lines_batch: np.ndarray = None, renderings: np.ndarray = None):
        """Save per-iteration numeric and small image snapshots.

        Args:
            iteration: iteration index
            lines_batch: numpy array of shape (P, N, 5) with primitive params
            renderings: numpy array of rendered patches (P, H, W)
        """
        if not self._enabled():
            return
        it_dir = self.image_dir / "iterations"
        it_dir.mkdir(parents=True, exist_ok=True)
        meta = {"iteration": int(iteration)}
        if lines_batch is not None:
            try:
                np.savez_compressed(it_dir / f"lines_iter_{iteration}.npz", lines=lines_batch)
            except (OSError, ValueError):
                pass  # Silent failure for tracing
        if renderings is not None:
            # Save a small montage or individual PNGs
            try:
                for i in range(min(renderings.shape[0], 8)):
                    arr = renderings[i]
                    if arr.dtype != "uint8":
                        # normalize to 0-255
                        arrn = (255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)).astype(np.uint8)
                    else:
                        arrn = arr
                    Image.fromarray(arrn).save(it_dir / f"render_{iteration}_{i}.png")
            except (ValueError, TypeError, OSError, IndexError):
                try:
                    np.savez_compressed(it_dir / f"renderings_iter_{iteration}.npz", renderings=renderings)
                except (OSError, ValueError):
                    pass
        with open(it_dir / f"meta_{iteration}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def save_primitive_history(self, history_array: np.ndarray) -> None:
        """Save stacked history snapshots of primitives across iterations.

        history_array: numpy array with shape (num_iterations, P, N, 5)
        """
        if not self._enabled():
            return
        try:
            np.savez_compressed(self.image_dir / "primitive_history.npz", history=history_array)
        except Exception:
            # fallback to json (less efficient)
            try:
                with open(self.image_dir / "primitive_history.json", "w", encoding="utf-8") as f:
                    json.dump({"history_len": int(len(history_array))}, f)
            except Exception:
                pass

    def save_metrics(self, metrics: dict) -> None:
        """Save per-stage metrics (IoU, Chamfer, etc.).

        Args:
            metrics: dict with keys like 'pre_refinement_iou', 'post_refinement_iou', 'final_iou', etc.
        """
        if not self._enabled():
            return
        with open(self.image_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f)
