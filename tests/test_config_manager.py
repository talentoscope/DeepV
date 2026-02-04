import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running this test file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util_files.config_manager import ConfigManager


def test_config_manager_initialization():
    """Test that ConfigManager initializes correctly."""
    manager = ConfigManager()
    assert manager.config_path.exists()
    assert manager.config_path.name == "config"


def test_get_default_config():
    """Test loading the default configuration."""
    manager = ConfigManager()
    cfg = manager.get_config("config")

    # Check that required fields are present
    assert hasattr(cfg, "seed")
    assert hasattr(cfg, "pipeline")
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "data")
    assert hasattr(cfg, "training")
    assert hasattr(cfg, "paths")

    # Check pipeline defaults
    assert cfg.pipeline.primitive_type == "line"
    assert cfg.pipeline.curve_count == 10


def test_config_overrides():
    """Test configuration overrides."""
    manager = ConfigManager()
    overrides = ["pipeline.primitive_type=curve", "seed=123"]
    cfg = manager.get_config("config", overrides)

    assert cfg.pipeline.primitive_type == "curve"
    assert cfg.seed == 123


def test_local_config():
    """Test loading local configuration."""
    manager = ConfigManager()
    cfg = manager.get_config("local")

    # Should have local paths - relative paths starting with .
    assert cfg.paths.code_dir == "."
    assert cfg.paths.data_dir == "./data"
    assert cfg.paths.logs_dir == "./logs"


def test_save_and_load_config():
    """Test saving and loading configuration."""
    import tempfile

    import yaml

    manager = ConfigManager()
    cfg = manager.get_config("config")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name

    try:
        manager.save_config(cfg, temp_path)

        # Load and verify
        with open(temp_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert "seed" in loaded
        assert loaded["pipeline"]["primitive_type"] == "line"

    finally:
        os.unlink(temp_path)
