import os
import os.path
import shutil


def require_empty(d, recreate=False):
    if os.path.exists(d):
        if recreate:
            shutil.rmtree(d)
        else:
            raise OSError("Path {} exists and no --overwrite set. Exiting".format(d))
    os.makedirs(d)


def ensure_dir(path):
    """Ensure directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
