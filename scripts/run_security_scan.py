#!/usr/bin/env python3
"""
DeepV Security Scanner

Runs automated security scans on dependencies and provides reports.
This script helps maintain security hygiene for the DeepV project.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔍 Running {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        else:
            print(f"❌ {description} failed")
            print(f"Error output: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False, str(e)


def scan_with_safety():
    """Run safety check on dependencies."""
    try:
        import safety  # noqa: F401

        return run_command("safety check", "Safety vulnerability scan")
    except ImportError:
        print("⚠️  Safety not installed. Skipping safety scan.")
        return True, "Safety not available"


def scan_with_pip_audit():
    """Run pip-audit for comprehensive dependency analysis."""
    try:
        import pip_audit  # noqa: F401

        return run_command("pip-audit", "Pip-audit vulnerability scan")
    except ImportError:
        print("⚠️  pip-audit not installed. Skipping pip-audit scan.")
        return True, "pip-audit not available"


def check_outdated_packages():
    """Check for outdated packages."""
    return run_command("pip list --outdated", "Outdated package check")


def generate_report(results):
    """Generate a summary report."""
    print("\n" + "=" * 60)
    print("🔒 DEEPV SECURITY SCAN REPORT")
    print("=" * 60)

    all_passed = True
    for check_name, (success, output) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {check_name}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All security checks passed!")
    else:
        print("⚠️  Some security checks failed. Please review the output above.")
    print("=" * 60)

    return all_passed


def main():
    """Main security scanning function."""
    print("🔒 Starting DeepV Security Scan...")

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Run all security checks
    results = {}

    results["Safety Scan"] = scan_with_safety()
    results["Pip Audit"] = scan_with_pip_audit()
    results["Outdated Packages"] = check_outdated_packages()

    # Generate report
    all_passed = generate_report(results)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
