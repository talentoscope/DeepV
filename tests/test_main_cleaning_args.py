import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cleaning.scripts import main_cleaning as mc


def test_parse_args_defaults_and_override():
    args = mc.parse_args(["--model", "UNET"])
    assert args.model == "UNET"
    assert args.batch_size == 8

    args2 = mc.parse_args(["--model", "SmallUNET", "--batch_size", "16"])
    assert args2.model == "SmallUNET"
    assert args2.batch_size == 16
