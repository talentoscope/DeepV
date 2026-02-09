#!/usr/bin/env python3
"""
Chamfer Distance Module

PyTorch implementation of Chamfer distance for point cloud comparison.
Provides GPU-accelerated Chamfer distance computation using CUDA.

Features:
- Chamfer distance calculation between point sets
- GPU acceleration with CUDA kernels
- Bidirectional distance computation
- PyTorch autograd integration

Used by evaluation pipelines for geometric distance metrics.
"""

# https://github.com/krrish94/chamferdist/blob/97051583f6fe72d5d4a855696dbfda0ea9b73a6a/chamferdist/ChamferDistance.py
import chamferdistcuda as chamfer
import torch
from torch import nn
from torch.autograd import Function


# Chamfer's distance module
# GPU tensors only
class chamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, idx1_, idx2_):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, input1, input2):
        return chamferFunction.apply(input1, input2)
