"""Environment validation script for developers and CI.

Checks Python version, presence of critical packages, and CUDA availability.
Run locally to get fast feedback before running heavier jobs.
"""

import importlib
import sys


def check_python():
    print(f"Python: {sys.version.splitlines()[0]}")


def check_packages(packages):
    missing = []
    for pkg in packages:
        try:
            m = importlib.import_module(pkg)
            print(f"Found {pkg}: {getattr(m, '__version__', 'unknown')}")
        except Exception:
            print(f"Missing {pkg}")
            missing.append(pkg)
    return missing


def check_torch():
    try:
        import torch

        print(f"torch: {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"CUDA available: {cuda}")
    except Exception as e:
        print(f"torch import failed: {e}")


def main():
    check_python()
    pkgs = [
        "numpy",
        "torch",
        "torchvision",
        "cairocffi",
        "tensorboardx",
    ]
    missing = check_packages([p for p in pkgs if p != "torch"])
    if "torch" in pkgs:
        check_torch()

    if missing:
        print("\nMissing packages:", ", ".join(missing))
        print("Install with: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
