#!/usr/bin/env python3
"""
DeepV File System Utilities

Common file and directory operations used throughout the DeepV codebase.
Provides safe file system operations with proper error handling.

Functions:
- require_empty: Ensure directory is empty or recreate it
- ensure_dir: Create directory if it doesn't exist
- Safe file operations with overwrite protection

Used by training scripts, data processing, and output management.
"""

import os
import os.path
import shutil


def require_empty(d, recreate=False):
    if os.path.exists(d):
        if recreate:
            shutil.rmtree(d)
        else:
            raise OSError(f"Path {d} exists and no --overwrite set. Exiting")
    os.makedirs(d)


def ensure_dir(path):
    """Ensure directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
