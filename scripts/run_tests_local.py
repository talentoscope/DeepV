#!/usr/bin/env python3
"""
DeepV Local Test Runner

Sets up the development environment and runs the complete test suite locally.
Cross-platform Python equivalent of the original PowerShell test runner.

Features:
- Automatic virtual environment creation and activation
- Dependency installation from requirements-dev.txt
- Comprehensive test execution (unit, integration, smoke tests)
- Test coverage reporting
- Cross-platform compatibility (Windows/Linux/macOS)

Runs pytest with appropriate configuration for the DeepV codebase, including
GPU availability checks and optional test skipping.

Usage:
    python scripts/run_tests_local.py
"""

import subprocess
import sys
from pathlib import Path


def create_venv_if_needed():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("Virtual environment created successfully.")
    return True


def get_venv_python():
    """Get the Python executable from the virtual environment."""
    if sys.platform == "win32":
        python_exe = ".venv/Scripts/python.exe"
    else:
        python_exe = ".venv/bin/python"

    venv_python = Path(python_exe)
    if not venv_python.exists():
        print(f"Virtual environment Python not found at: {venv_python}")
        return None
    return str(venv_python)


def run_command(cmd, description, python_exe=None):
    """Run a command and return success status."""
    print(f"Running {description}...")
    try:
        if python_exe and cmd[0] == "python":
            cmd[0] = python_exe
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code: {e.returncode}")
        return False


def main():
    """Run the complete test setup and execution."""
    # Create venv if needed
    if not create_venv_if_needed():
        return 1

    # Get venv Python
    python_exe = get_venv_python()
    if not python_exe:
        return 1

    print("Using virtual environment Python:", python_exe)

    # Upgrade pip and install dev requirements
    if not run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"], "pip upgrade", python_exe):
        return 1

    if not run_command(
        [python_exe, "-m", "pip", "install", "-r", "requirements-dev.txt"], "dev requirements installation", python_exe
    ):
        return 1

    # Run environment validator
    if not run_command([python_exe, "scripts/validate_env.py"], "environment validator", python_exe):
        return 1

    # Run pytest
    if not run_command([python_exe, "-m", "pytest", "-q"], "pytest", python_exe):
        return 1

    print("✅ All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
