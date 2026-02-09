"""Configuration management utilities for DeepV.

This module provides centralized configuration management using Hydra.
It allows for hierarchical configuration with environment-specific overrides.
"""

import os
from pathlib import Path
from typing import Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """Centralized configuration manager for DeepV experiments."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to the config directory. If None, uses default.
        """
        if config_path is None:
            # Assume config is in the same directory as this file
            config_path = Path(__file__).parent.parent / "config"

        self.config_path = Path(config_path)
        self._hydra_initialized = False

    def _ensure_hydra_initialized(self):
        """Ensure Hydra is initialized."""
        if not self._hydra_initialized:
            GlobalHydra.instance().clear()
            initialize_config_dir(config_dir=str(self.config_path), version_base=None)
            self._hydra_initialized = True

    def get_config(self, config_name: str = "config", overrides: Optional[list] = None) -> DictConfig:
        """Get a configuration by name.

        Args:
            config_name: Name of the config file
                (without .yaml extension)
            overrides: List of configuration overrides

        Returns:
            OmegaConf DictConfig object
        """
        self._ensure_hydra_initialized()

        if overrides is None:
            overrides = []

        cfg = compose(config_name=config_name, overrides=overrides)
        return cfg

    def get_pipeline_config(self, overrides: Optional[list] = None) -> DictConfig:
        """Get pipeline configuration."""
        return self.get_config("config", overrides)

    def save_config(self, cfg: DictConfig, output_path: str):
        """Save configuration to a YAML file.

        Args:
            cfg: Configuration to save
            output_path: Path to save the configuration
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    def print_config(self, cfg: DictConfig):
        """Print configuration in a readable format."""
        print(OmegaConf.to_yaml(cfg))


# Global instance for easy access
config_manager = ConfigManager()


def get_config(config_name: str = "config", overrides: Optional[list] = None) -> DictConfig:
    """Convenience function to get configuration.

    Args:
        config_name: Name of the config
            file
        overrides: List of configuration overrides

    Returns:
        OmegaConf DictConfig object
    """
    return config_manager.get_config(config_name, overrides)


def get_pipeline_config(overrides: Optional[list] = None) -> DictConfig:
    """Convenience function to get pipeline configuration."""
    return config_manager.get_pipeline_config(overrides)
