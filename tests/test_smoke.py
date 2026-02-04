def test_run_pipeline_import():
    import importlib
    import os
    import sys

    import pytest

    # Ensure repo root is on sys.path when running this test file directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    pytest.importorskip("torchvision")
    rp = importlib.import_module("run_pipeline")
    assert hasattr(rp, "main")


def test_cleaning_entry_import():
    import importlib

    import pytest

    pytest.importorskip("torch")
    mod = importlib.import_module("cleaning.scripts.main_cleaning")
    assert hasattr(mod, "main") or hasattr(mod, "parse_args")
