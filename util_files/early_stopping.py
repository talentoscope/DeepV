"""
Early stopping utilities for DeepV training.

Provides automatic training termination based on validation metrics to prevent
overfitting and optimize training time.
"""

from typing import Any, Dict, Optional


class EarlyStopping:
    """
    Early stopping monitor for training loops.

    Monitors a validation metric and stops training when the metric stops improving
    for a specified number of epochs (patience). Supports both minimization and
    maximization objectives.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize early stopping monitor.

        Args:
            monitor: Metric name to monitor (e.g., 'val_loss', 'val_iou')
            patience: Number of epochs to wait before stopping after no improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for minimization (loss), 'max' for maximization (accuracy, IoU)
            restore_best_weights: Whether to restore model weights from best epoch
            verbose: Whether to print early stopping messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # Internal state
        self.best_score: Optional[float] = None
        self.best_weights: Optional[Dict[str, Any]] = None
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_epoch = 0

        # Validate mode
        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        # Set comparison function based on mode
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(
        self,
        current_score: float,
        model: Optional[Any] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """
        Check if training should stop based on current validation score.

        Args:
            current_score: Current value of the monitored metric
            model: Model to save weights from (if restore_best_weights=True)
            epoch: Current epoch number (for logging)

        Returns:
            True if training should stop, False otherwise
        """
        epoch_str = f" (epoch {epoch})" if epoch is not None else ""

        if self.best_score is None:
            # First call - initialize best score
            self.best_score = current_score
            self.best_epoch = epoch or 0
            if self.restore_best_weights and model is not None:
                self.best_weights = self._get_model_weights(model)
            if self.verbose:
                print(f"EarlyStopping: Initial {self.monitor} = " f"{current_score:.6f}{epoch_str}")
            return False

        if self.is_better(current_score, self.best_score):
            # Improvement detected
            self.best_score = current_score
            self.best_epoch = epoch or 0
            self.wait_count = 0

            if self.restore_best_weights and model is not None:
                self.best_weights = self._get_model_weights(model)

            if self.verbose:
                improvement = "decreased" if self.mode == "min" else "increased"
                print(f"EarlyStopping: {self.monitor} {improvement} to " f"{current_score:.6f}{epoch_str}")
        else:
            # No improvement
            self.wait_count += 1
            if self.verbose:
                print(
                    f"EarlyStopping: {self.monitor} did not improve from "
                    f"{self.best_score:.6f} "
                    f"({self.wait_count}/{self.patience}){epoch_str}"
                )

            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch or 0
                if self.verbose:
                    print(
                        f"EarlyStopping: Stopping training at epoch "
                        f"{self.stopped_epoch}. "
                        f"Best {self.monitor} = {self.best_score:.6f} at epoch "
                        f"{self.best_epoch}"
                    )
                return True

        return False

    def restore_weights(self, model: Any):
        """
        Restore model weights from the best epoch.

        Args:
            model: Model to restore weights to
        """
        if self.restore_best_weights and self.best_weights is not None:
            self._set_model_weights(model, self.best_weights)
            if self.verbose:
                print(f"EarlyStopping: Restored model weights from epoch " f"{self.best_epoch}")
        elif self.verbose:
            print("EarlyStopping: No weights to restore")

    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved."""
        return self.best_score

    def get_best_epoch(self) -> int:
        """Get the epoch where the best score was achieved."""
        return self.best_epoch

    def get_stopped_epoch(self) -> int:
        """Get the epoch where training was stopped."""
        return self.stopped_epoch

    def reset(self):
        """Reset the early stopping state."""
        self.best_score = None
        self.best_weights = None
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving/resuming training."""
        return {
            "monitor": self.monitor,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "restore_best_weights": self.restore_best_weights,
            "best_score": self.best_score,
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_epoch": self.best_epoch,
            "best_weights": self.best_weights,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict for resuming training."""
        self.monitor = state_dict.get("monitor", self.monitor)
        self.patience = state_dict.get("patience", self.patience)
        self.min_delta = state_dict.get("min_delta", self.min_delta)
        self.mode = state_dict.get("mode", self.mode)
        self.restore_best_weights = state_dict.get("restore_best_weights", self.restore_best_weights)
        self.best_score = state_dict.get("best_score")
        self.wait_count = state_dict.get("wait_count", 0)
        self.stopped_epoch = state_dict.get("stopped_epoch", 0)
        self.best_epoch = state_dict.get("best_epoch", 0)
        self.best_weights = state_dict.get("best_weights")

    def _get_model_weights(self, model: Any) -> Dict[str, Any]:
        """Extract model weights for saving."""
        if hasattr(model, "state_dict"):
            return model.state_dict()
        elif hasattr(model, "get_weights"):
            return {"weights": model.get_weights()}
        else:
            raise ValueError("Model must have state_dict() or get_weights() method")

    def _set_model_weights(self, model: Any, weights: Dict[str, Any]):
        """Set model weights from saved state."""
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(weights)
        elif hasattr(model, "set_weights") and "weights" in weights:
            model.set_weights(weights["weights"])
        else:
            raise ValueError("Model must have load_state_dict() or set_weights() method")


class EarlyStoppingWithMetrics:
    """
    Enhanced early stopping that monitors multiple metrics with different criteria.
    """

    def __init__(
        self,
        metrics_config: Dict[str, Dict[str, Any]],
        patience: int = 10,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize multi-metric early stopping.

        Args:
            metrics_config: Dict mapping metric names to their config
                Example: {
                    'val_loss': {'mode': 'min', 'min_delta': 0.001},
                    'val_iou': {'mode': 'max', 'min_delta': 0.01}
                }
            patience: Base patience value
            restore_best_weights: Whether to restore best weights
            verbose: Whether to print messages
        """
        self.metrics_config = metrics_config
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # Create individual early stopping monitors
        self.monitors = {}
        for metric_name, config in metrics_config.items():
            self.monitors[metric_name] = EarlyStopping(
                monitor=metric_name,
                patience=patience,
                min_delta=config.get("min_delta", 0.0),
                mode=config.get("mode", "min"),
                restore_best_weights=False,  # Handle at this level
                verbose=verbose,
            )

        # Overall state
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0

    def __call__(
        self,
        metrics: Dict[str, float],
        model: Optional[Any] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """
        Check if training should stop based on multiple metrics.

        Args:
            metrics: Dict of metric names to values
            model: Model to save weights from
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        should_stop = False

        for metric_name, monitor in self.monitors.items():
            if metric_name in metrics:
                if monitor(metrics[metric_name], model, epoch):
                    should_stop = True
                    break

        # If any monitor triggered stopping, save the best weights
        if should_stop and self.restore_best_weights and model is not None:
            # Find the monitor with the best epoch
            best_monitor = max(self.monitors.values(), key=lambda m: m.get_best_epoch())
            if best_monitor.best_weights is not None:
                self.best_weights = best_monitor.best_weights
                self.best_epoch = best_monitor.best_epoch
                self.stopped_epoch = epoch or 0

        return should_stop

    def restore_weights(self, model: Any):
        """Restore the best weights from any of the monitors."""
        if self.restore_best_weights and self.best_weights is not None:
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(self.best_weights)
            elif hasattr(model, "set_weights"):
                model.set_weights(self.best_weights["weights"])
            if self.verbose:
                print(f"EarlyStopping: Restored model weights from epoch " f"{self.best_epoch}")


def create_early_stopping_for_vectorization(patience: int = 15, min_delta: float = 1e-4) -> EarlyStopping:
    """
    Create early stopping configured for vectorization training.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum improvement threshold

    Returns:
        Configured EarlyStopping instance
    """
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        mode="min",
        restore_best_weights=True,
        verbose=True,
    )


def create_early_stopping_for_cleaning(patience: int = 10, min_delta: float = 1e-3) -> EarlyStopping:
    """
    Create early stopping configured for cleaning training.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum improvement threshold

    Returns:
        Configured EarlyStopping instance
    """
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        mode="min",
        restore_best_weights=True,
        verbose=True,
    )


# Example usage and integration helper
def integrate_early_stopping_into_training():
    """
    Example of how to integrate early stopping into a training loop.
    """
    # Create early stopping monitor
    early_stopping = create_early_stopping_for_vectorization(patience=10)

    # Training loop example
    for epoch in range(100):
        # Training code here...

        # Validation
        val_loss = 0.5  # Your validation function here

        # Check early stopping
        if early_stopping(val_loss, model, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            early_stopping.restore_weights(model)
            break

    return early_stopping.get_best_score()


if __name__ == "__main__":
    # Test early stopping functionality
    import torch.nn as nn

    # Create a simple model for testing
    model = nn.Linear(10, 1)

    # Test early stopping
    early_stopping = EarlyStopping(patience=3, verbose=True)

    # Simulate training with decreasing then increasing loss
    losses = [1.0, 0.9, 0.8, 0.75, 0.74, 0.73, 0.74, 0.75, 0.76]

    for epoch, loss in enumerate(losses):
        should_stop = early_stopping(loss, model, epoch)
        if should_stop:
            print(f"Training stopped at epoch {epoch}")
            break

    print(f"Best loss: {early_stopping.get_best_score()} at epoch " f"{early_stopping.get_best_epoch()}")
