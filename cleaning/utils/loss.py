#!/usr/bin/env python3
"""
Cleaning Loss Functions Module

Loss functions for image cleaning and denoising models.
Provides evaluation metrics and loss functions for image-to-image tasks.

Features:
- IoU (Intersection over Union) calculation
- Dice coefficient computation
- PSNR (Peak Signal-to-Noise Ratio)
- Custom loss functions for image cleaning

Used by cleaning model training and evaluation.
"""

import torch
import torch.nn as nn

SMOOTH = 1e-6


def IOU(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate Intersection over Union (IoU) for binary segmentation.

    Args:
        outputs: Predicted binary masks, shape (B, H, W) or (B, 1, H, W)
        labels: Ground truth binary masks, shape (B, H, W)

    Returns:
        Thresholded IoU scores averaged over batch
    """
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our division to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresholds

    return thresholded.mean()


def PSNR(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate Peak Signal-to-Noise Ratio between outputs and labels.

    Args:
        outputs: Predicted images, shape (B, 1, H, W) or (B, H, W)
        labels: Ground truth images, shape (B, H, W)

    Returns:
        PSNR value in dB
    """
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    mse = torch.nn.functional.mse_loss(outputs, labels)
    return 10 * torch.log10(1 / mse)


class CleaningLoss(nn.Module):
    """Custom loss function for image cleaning tasks combining extraction and restoration losses."""

    def __init__(self, kind: str = "MSE", with_restore: bool = True, alpha: float = 1):
        """Initialize cleaning loss.

        Args:
            kind: Loss type, either 'MSE' or 'BCE'
            with_restore: Whether to include restoration loss component
            alpha: Weighting factor for losses (currently unused)
        """
        super().__init__()

        if kind == "MSE":
            self.loss_extraction = nn.MSELoss()
            self.loss_restoration = nn.MSELoss()
        else:
            self.loss_extraction = nn.BCELoss()
            self.loss_restoration = nn.BCELoss()

        self.alpha = alpha
        self.with_restor = with_restore

    def forward(self, y_pred_extract: torch.Tensor, y_pred_restore: torch.Tensor,
                y_true_extract: torch.Tensor, y_true_restore: torch.Tensor) -> torch.Tensor:
        """Compute the cleaning loss.

        Args:
            y_pred_extract: Predicted extraction output
            y_pred_restore: Predicted restoration output
            y_true_extract: Ground truth extraction target
            y_true_restore: Ground truth restoration target

        Returns:
            Combined loss value
        """
        loss = 0
        y_true_extract = y_true_extract.unsqueeze(1)
        y_true_restore = y_true_restore.unsqueeze(1)
        if self.with_restor:
            loss += self.loss_restoration(y_pred_restore, y_true_restore)

        loss += self.loss_extraction(y_pred_extract, y_true_extract)

        return loss


def MSE_loss(
    X_batch: torch.Tensor, y_batch: torch.Tensor, model: nn.Module = None, device: str = "cpu"
) -> torch.Tensor:
    """Calculate MSE loss for a batch.

    Args:
        X_batch: Input batch
        y_batch: Target batch
        model: Model to evaluate
        device: Device to run on

    Returns:
        MSE loss value
    """
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    y_batch = y_batch.unsqueeze(1)
    logits, _ = model(X_batch)
    loss_fun = nn.MSELoss()
    return loss_fun(logits, y_batch)


def MSE_synthetic_loss(
    X_batch: torch.Tensor, y_batch_er: torch.Tensor, y_batch_e: torch.Tensor,
    model: nn.Module = None, device: str = None
) -> torch.Tensor:
    """Calculate MSE loss for synthetic data with extraction and restoration.

    Args:
        X_batch: Input batch
        y_batch_er: Restoration targets
        y_batch_e: Extraction targets
        model: Model to evaluate
        device: Device to run on (optional)

    Returns:
        Combined MSE loss
    """
    if device is not None:
        X_batch = X_batch.to(device)
        y_batch_er = y_batch_er.to(device)
        y_batch_e = y_batch_e.to(device)

    y_batch_er = y_batch_er.unsqueeze(1)
    y_batch_e = y_batch_e.unsqueeze(1)

    logits_er, logits_e = model(X_batch)
    loss_first = nn.MSELoss()
    loss_second = nn.MSELoss()

    return loss_first(logits_er, y_batch_er) + loss_second(logits_e, y_batch_e)


def BCE_loss(
    X_batch: torch.Tensor, y_batch: torch.Tensor, model: nn.Module = None, device: str = "cpu"
) -> torch.Tensor:
    """Calculate BCE loss for a batch.

    Args:
        X_batch: Input batch
        y_batch: Target batch
        model: Model to evaluate
        device: Device to run on

    Returns:
        BCE loss value
    """
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    y_batch = y_batch.unsqueeze(1)
    logits = model(X_batch)
    loss_fun = nn.BCELoss()
    return loss_fun(logits, y_batch)
