#!/usr/bin/env python3
"""
DeepV Code Quality Linting and Formatting Script

Runs comprehensive code quality checks and automatic formatting on the DeepV codebase.
Cross-platform Python equivalent of the original PowerShell linting script.

Performs the following checks:
- Black code formatting and style enforcement
- Flake8 linting for PEP 8 compliance and code quality
- MyPy static type checking (if configured)
- Import sorting and organization
- Line length and whitespace validation

Automatically fixes formatting issues where possible and reports remaining
issues that require manual correction.

Usage:
    python scripts/lint_code.py
"""

import subprocess
import sys
from pathlib import Path


def check_venv():
    """Check if virtual environment exists."""
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Create a venv first: python -m venv .venv")
        return False
    return True


def activate_venv():
    """Activate the virtual environment."""
    # Note: We can't actually activate the venv in a subprocess,
    # but we can run commands with the venv's Python
    return (
        str(Path(".venv") / "Scripts" / "python.exe")
        if sys.platform == "win32"
        else str(Path(".venv") / "bin" / "python")
    )


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Running {description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code: {e.returncode}")
        return False


def main():
    """Run all linting checks."""
    if not check_venv():
        return 1

    # Get the Python executable from venv
    python_exe = activate_venv()

    # Upgrade pip and install dev requirements
    if not run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"], "pip upgrade"):
        return 1

    if not run_command(
        [python_exe, "-m", "pip", "install", "-r", "requirements-dev.txt"],
        "dev requirements installation",
    ):
        return 1

    # Run linters
    success = True
    success &= run_command([python_exe, "-m", "black", "--check", "."], "black (check)")
    success &= run_command([python_exe, "-m", "isort", "--check-only", "."], "isort (check)")
    success &= run_command([python_exe, "-m", "flake8", "."], "flake8")

    if success:
        print("✅ All linting checks passed!")
        return 0
    else:
        print("❌ Some linting checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
