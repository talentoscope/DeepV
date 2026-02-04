import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_refinement_register_sigint_present():
    """Quick smoke check: ensure `register_sigint_flag` appears in the lines refinement file.

    This avoids importing heavy ML modules while asserting the helper exists in source.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "refinement", "our_refinement", "refinement_for_lines.py")
    path = os.path.abspath(path)
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "def register_sigint_flag" in src
