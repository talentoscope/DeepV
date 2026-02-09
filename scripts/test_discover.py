#!/usr/bin/env python3
"""
Dataset Downloader Discovery Script

Discovers and lists all available dataset downloader functions in the codebase.
Used for testing and validation of the download system.

Outputs:
- Total count of download functions found
- List of all downloader function names

Useful for:
- Verifying downloader availability
- Testing download system completeness
- Debugging import issues

Usage:
    python scripts/test_discover.py
"""

import importlib
import inspect

dd = importlib.import_module("dataset_downloaders.download_dataset")
funcs = [name for name, obj in inspect.getmembers(dd, inspect.isfunction) if name.startswith("download_")]
print("found", len(funcs), "downloaders")
for f in funcs:
    print("-", f)
