"""Adapter layer that calls into existing `dataset_downloaders`.

This keeps a stable API under `dataset.downloaders.download()` while reusing
the current implementations in `dataset_downloaders`.
"""
from pathlib import Path
from typing import Dict
from . import download_dataset as _dlmod


def download(dataset_name: str, output_dir: Path | str = "./data", test: bool = False, **kwargs) -> Dict:
    """Download `dataset_name` into `output_dir` using existing downloaders.

    Additional keyword args are forwarded to the underlying downloader when
    supported (e.g., `gdrive_urls` for `floorplancad`).
    """
    output_dir = Path(output_dir)
    mod = _dlmod

    fn_name = f"download_{dataset_name}"
    if hasattr(mod, fn_name):
        fn = getattr(mod, fn_name)
        # Try calling common signature
        kwargs2 = {}
        sig = None
        try:
            import inspect
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        if sig and 'output_dir' in sig.parameters:
            kwargs2['output_dir'] = output_dir
        if sig and 'test_mode' in sig.parameters:
            kwargs2['test_mode'] = test
        # forward any other args if accepted
        if sig:
            for k, v in kwargs.items():
                if k in sig.parameters:
                    kwargs2[k] = v

        return fn(**kwargs2)
    else:
        # Fallback to the local download_dataset CLI
        try:
            return _dlmod.download_dataset(dataset_name, output_dir, test_mode=test)
        except Exception as e:
            raise


__all__ = ["download"]
