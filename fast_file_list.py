#!/usr/bin/env python3
"""
Fast Python File Lister for DeepV Audit

Quickly lists Python files while excluding common non-source directories.
Much faster than recursive os.walk() through data/, logs/, etc.
"""

import os
import sys
from pathlib import Path

def fast_list_python_files():
    """Fast listing of Python files, excluding common directories."""

    # Directories to exclude (most common first for performance)
    exclude_dirs = {
        'data', 'logs', 'models', '__pycache__', '.git', '.venv', 'venv',
        'node_modules', '.pytest_cache', '.mypy_cache', 'dist', 'build',
        'vectorization/models/specs', 'docs/_build'
    }

    # Patterns to exclude
    exclude_patterns = ['test_', '_test.py', 'conftest.py']

    python_files = []

    # Use os.scandir for better performance
    def scan_dir(path, prefix=""):
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        # Skip excluded directories
                        if entry.name in exclude_dirs:
                            continue
                        # Recurse into subdirectories
                        scan_dir(entry.path, f"{prefix}{entry.name}/")
                    elif entry.name.endswith('.py'):
                        # Skip test files
                        if not any(pattern in entry.name for pattern in exclude_patterns):
                            rel_path = f"{prefix}{entry.name}"
                            python_files.append(rel_path)
        except PermissionError:
            pass  # Skip directories we can't read

    # Start scanning from current directory
    scan_dir('.')

    return sorted(python_files)

if __name__ == '__main__':
    files = fast_list_python_files()
    print(f"Found {len(files)} Python files:")
    for file in files:
        print(file)