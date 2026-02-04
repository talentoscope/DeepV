import os
import sys

import pytest

# Ensure repo root is on sys.path when running this test file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util_files.file_utils import require_empty


def test_require_empty_creates_dir(tmp_path):
    d = tmp_path / "newdir"
    # should create directory when missing
    require_empty(str(d), recreate=False)
    assert d.exists() and d.is_dir()


def test_require_empty_raises_when_exists(tmp_path):
    d = tmp_path / "exists"
    d.mkdir()
    with pytest.raises(OSError):
        require_empty(str(d), recreate=False)


def test_require_empty_recreate(tmp_path):
    d = tmp_path / "recreate"
    d.mkdir()
    (d / "file.txt").write_text("x")
    # should remove existing and create a fresh dir
    require_empty(str(d), recreate=True)
    assert d.exists() and not (d / "file.txt").exists()
