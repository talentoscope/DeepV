"""
Mixed-precision training utilities for DeepV.

Provides automatic mixed-precision training support with gradient scaling
to improve training speed and reduce memory usage.
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager


class MixedPrecisionTrainer:
    """
    Mixed-precision training manager with automatic gradient scaling.

    Handles FP16/FP32 mixed precision training with gradient scaling to prevent
    gradient underflow, providing better numerical stability.
    """

    def __init__(self, enabled: bool = True, init_scale: float = 2.0**16,
                 growth_factor: float = 2.0, backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """
        Initialize mixed precision trainer.

        Args:
            enabled: Whether to enable mixed precision training
            init_scale: Initial gradient scaling factor
            growth_factor: Factor to increase scale when gradients are stable
            backoff_factor: Factor to decrease scale when gradients overflow
            growth_interval: Steps between scale growth attempts
        """
        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            self.scaler = GradScaler('cuda',
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
        else:
            self.scaler = None

    def is_enabled(self) -> bool:
        """Check if mixed precision is enabled and available."""
        return self.enabled

    @contextmanager
    def autocast_context(self):
        """
        Context manager for automatic mixed precision casting.

        Usage:
            with trainer.autocast_context():
                output = model(input)
                loss = criterion(output, target)
        """
        if self.enabled:
            with autocast('cuda'):
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale the loss for gradient computation.

        Args:
            loss: Loss tensor to scale

        Returns:
            Scaled loss tensor
        """
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer,
                      loss: Optional[torch.Tensor] = None) -> bool:
        """
        Perform optimizer step with gradient scaling.

        Args:
            optimizer: Optimizer to step
            loss: Loss tensor (optional, for scaler update)

        Returns:
            True if step was successful, False if gradients overflowed
        """
        if self.enabled and self.scaler is not None:
            # Try to step with scaler
            try:
                self.scaler.step(optimizer)
                self.scaler.update()
                return True
            except Exception:
                # If step fails, skip this step
                self.scaler.update()
                return False
        else:
            # Standard optimizer step
            if loss is not None:
                loss.backward()
            optimizer.step()
            return True

    def backward(self, loss: torch.Tensor):
        """
        Perform backward pass with gradient scaling.

        Args:
            loss: Loss tensor to backpropagate
        """
        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients before clipping.

        Args:
            optimizer: Optimizer whose gradients to unscale
        """
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def get_scale_info(self) -> Dict[str, Any]:
        """
        Get information about the current gradient scaling state.

        Returns:
            Dictionary with scale information
        """
        if self.enabled and self.scaler is not None:
            return {
                'scale': self.scaler.get_scale(),
                'growth_tracker': self.scaler._growth_tracker,
                'enabled': True
            }
        return {'enabled': False}

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dict for saving/resuming training.

        Returns:
            State dictionary containing scaler state
        """
        if self.enabled and self.scaler is not None:
            return {'scaler': self.scaler.state_dict()}
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state dict for resuming training.

        Args:
            state_dict: State dictionary containing scaler state
        """
        if self.enabled and self.scaler is not None and 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler'])


class MixedPrecisionWrapper(nn.Module):
    """
    Wrapper module that automatically applies mixed precision to forward passes.
    """

    def __init__(self, model: nn.Module, trainer: MixedPrecisionTrainer):
        """
        Initialize mixed precision wrapper.

        Args:
            model: Model to wrap
            trainer: Mixed precision trainer instance
        """
        super().__init__()
        self.model = model
        self.trainer = trainer

    def forward(self, *args, **kwargs):
        """Forward pass with automatic mixed precision."""
        with self.trainer.autocast_context():
            return self.model(*args, **kwargs)


def create_mixed_precision_trainer(enabled: bool = True,
                                  init_scale: float = 2.0**16) -> MixedPrecisionTrainer:
    """
    Convenience function to create a mixed precision trainer.

    Args:
        enabled: Whether to enable mixed precision
        init_scale: Initial gradient scaling factor

    Returns:
        Configured MixedPrecisionTrainer instance
    """
    return MixedPrecisionTrainer(enabled=enabled, init_scale=init_scale)


def enable_mixed_precision_for_model(model: nn.Module,
                                    enabled: bool = True) -> tuple:
    """
    Enable mixed precision for a model.

    Args:
        model: PyTorch model
        enabled: Whether to enable mixed precision

    Returns:
        Tuple of (wrapped_model, trainer)
    """
    trainer = create_mixed_precision_trainer(enabled=enabled)
    if enabled and trainer.is_enabled():
        wrapped_model = MixedPrecisionWrapper(model, trainer)
        return wrapped_model, trainer
    else:
        return model, trainer


# Example usage and training loop integration
def example_training_loop():
    """
    Example of how to integrate mixed precision into a training loop.
    """
    # Create model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Enable mixed precision
    model, mp_trainer = enable_mixed_precision_for_model(model, enabled=True)

    # Training loop
    for batch in range(100):
        # Generate dummy data
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)

        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        with mp_trainer.autocast_context():
            output = model(x)
            loss = criterion(output, y)

        # Backward pass with gradient scaling
        mp_trainer.backward(loss)

        # Unscale gradients before clipping (if needed)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with gradient scaling
        success = mp_trainer.step_optimizer(optimizer)
        if not success:
            print(f"Batch {batch}: Gradient overflow, skipping step")

        if batch % 10 == 0:
            scale_info = mp_trainer.get_scale_info()
            print(f"Batch {batch}: Loss={loss.item():.4f}, Scale={scale_info.get('scale', 'N/A')}")


if __name__ == "__main__":
    # Test mixed precision functionality
    if torch.cuda.is_available():
        print("Testing mixed precision training...")
        example_training_loop()
        print("Mixed precision test completed!")
    else:
        print("CUDA not available, skipping mixed precision test")