#!/usr/bin/env python3
"""
TensorBoard Utilities Module

TensorBoard integration utilities for experiment tracking.
Provides enhanced SummaryWriter with additional logging capabilities.

Features:
- Extended SummaryWriter class
- Image logging utilities
- Custom summary operations
- Experiment tracking integration

Used by training pipelines for visualization and monitoring.
"""

import tensorboardX.summary
from tensorboardX import SummaryWriter as OriginalSummaryWriter
from tensorboardX.proto.summary_pb2 import Summary


class SummaryWriter(OriginalSummaryWriter):
    def add_image(self, tag, img_np, global_step=None):
        """Add image data to summary.
        Note that this requires the ``pillow`` package.
        Args:
            tag (string): Data identifier
            img_np (np.ndarray): Image data, uint8
            global_step (int): Global step value to record
        Shape:
            img_tensor: :math:`(H, W, 3)`.
        """
        tag = tensorboardX.summary._clean_tag(tag)
        image = tensorboardX.summary.make_image(img_np)
        self.file_writer.add_summary(Summary(value=[Summary.Value(tag=tag, image=image)]), global_step)
