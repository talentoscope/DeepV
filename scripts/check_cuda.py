#!/usr/bin/env python3
"""
CUDA Availability and Configuration Check

Simple diagnostic script to verify PyTorch CUDA installation and GPU availability.
Reports CUDA status, device count, and GPU names for troubleshooting GPU training issues.

Outputs key-value pairs for easy parsing by other scripts:
- torch_version: PyTorch version string
- cuda_available: Boolean CUDA availability
- device_count: Number of CUDA devices
- device_name_0: Name of first GPU device

Usage:
    python scripts/check_cuda.py
"""

import sys

try:
    import torch
except Exception as e:
    print("torch_import_error", e)
    sys.exit(2)

print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
try:
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device_name_0", torch.cuda.get_device_name(0))
except Exception as e:
    print("cuda_probe_error", e)
