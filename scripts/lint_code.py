#!/usr/bin/env python3
"""
Run Code Linters and Formatters

Cross-platform Python script to run code quality checks.
Equivalent to the original lint_code.ps1 PowerShell script.

Usage:
    python scripts/lint_code.py
"""

import os
import sys
import subprocess
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
    if sys.platform == "win32":
        activate_script = ".venv/Scripts/activate.bat"
    else:
        activate_script = ".venv/bin/activate"

    # Note: We can't actually activate the venv in a subprocess,
    # but we can run commands with the venv's Python
    return str(Path(".venv") / "Scripts" / "python.exe") if sys.platform == "win32" else str(Path(".venv") / "bin" / "python")

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
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
    if not run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                      "pip upgrade"):
        return 1

    if not run_command([python_exe, "-m", "pip", "install", "-r", "requirements-dev.txt"],
                      "dev requirements installation"):
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