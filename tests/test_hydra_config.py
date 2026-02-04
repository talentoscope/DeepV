import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running this test file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util_files.config_manager import get_config


def test_hydra_config_loading():
    """Test that Hydra configuration loads correctly."""
    cfg = get_config("config")

    # Verify structure
    assert hasattr(cfg, "pipeline")
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "data")
    assert hasattr(cfg, "training")
    assert hasattr(cfg, "paths")

    # Verify default values
    assert cfg.pipeline.primitive_type == "line"
    assert cfg.pipeline.curve_count == 10
    assert cfg.seed == 42


def test_hydra_config_overrides():
    """Test configuration overrides work."""
    overrides = ["pipeline.primitive_type=curve", "pipeline.curve_count=20", "seed=999"]

    cfg = get_config("config", overrides)

    assert cfg.pipeline.primitive_type == "curve"
    assert cfg.pipeline.curve_count == 20
    assert cfg.seed == 999


def test_local_config_loading():
    """Test that local configuration loads and overrides defaults."""
    cfg = get_config("local")

    # Should use local paths
    assert cfg.paths.code_dir == "."
    assert cfg.paths.data_dir == "./data"
    assert cfg.paths.logs_dir == "./logs"

    # Should have GPU setting
    assert cfg.gpu == 0
