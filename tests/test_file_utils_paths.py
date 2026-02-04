import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util_files import file_utils as fu


def test_require_empty_on_tempdir():
    with tempfile.TemporaryDirectory() as td:
        # directory exists and is empty; allow recreate to avoid raising
        fu.require_empty(td, recreate=True)


def test_require_empty_fails_on_nonempty(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "f.txt").write_text("x")
    try:
        fu.require_empty(str(d))
        raised = False
    except Exception:
        raised = True
    assert raised
