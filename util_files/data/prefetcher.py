#!/usr/bin/env python3
"""
CUDA Data Prefetching Utilities

GPU data prefetching for improved training performance.
Overlaps data transfer to GPU with computation using CUDA streams.

Features:
- Asynchronous GPU data transfer
- CUDA stream-based prefetching
- Memory-efficient data loading
- Reduced training time through pipelining

Based on NVIDIA Apex implementation, adapted for DeepV data loading.
"""

import torch


class CudaPrefetcher:
    """based on github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f
    examples/imagenet/main_amp.py#L256"""

    def __init__(self, loader, device):
        self._loader = loader
        self.size = len(loader)
        self.device = device
        self.stream = torch.cuda.Stream(torch.device(device))

    def _init_data(self):
        self.loader = iter(self._loader)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.device(self.device):
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
                self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        self._init_data()
        return self

    def __len__(self):
        return self.size

    def __next__(self):
        with torch.cuda.device(self.device):
            torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_input is None:
            raise StopIteration
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
