#!/usr/bin/env python3
"""
Batch Audit Helper for DeepV Codebase Audit

Helps audit multiple files at once with common patterns and quick commands.
"""

import os
import sys
from pathlib import Path
from audit_tracker import AuditTracker

def audit_file(filepath, status="completed", notes="", issues=None):
    """Audit a single file using the tracker."""
    tracker = AuditTracker()
    tracker.mark_file_audited(filepath, status, notes, issues)

def batch_audit_pattern(pattern, status="completed", notes="", issues=None):
    """Audit all files matching a pattern."""
    tracker = AuditTracker()
    progress = tracker.load_progress()

    if 'files_to_audit' not in progress:
        print("Audit not initialized. Run 'python audit_tracker.py --init' first.")
        return

    matching_files = [f for f in progress['files_to_audit'] if pattern in f]

    print(f"Found {len(matching_files)} files matching '{pattern}':")
    for file in matching_files[:10]:  # Show first 10
        print(f"  {file}")
    if len(matching_files) > 10:
        print(f"  ... and {len(matching_files) - 10} more")

    if not matching_files:
        print("No files found matching pattern.")
        return

    response = input(f"Audit all {len(matching_files)} files? (y/N): ").lower()
    if response == 'y':
        for file in matching_files:
            audit_file(file, status, notes, issues)
        print(f"✅ Audited {len(matching_files)} files")
    else:
        print("Audit cancelled.")

def quick_audit_menu():
    """Show quick audit options."""
    print("🚀 Quick Audit Helper")
    print("=" * 30)
    print("1. Audit all __init__.py files (usually simple)")
    print("2. Audit all scripts/ files")
    print("3. Audit all test files")
    print("4. Audit files by pattern")
    print("5. Show audit status")
    print("6. Generate report")
    print("0. Exit")

    choice = input("Choose option: ").strip()

    if choice == '1':
        batch_audit_pattern('__init__.py', notes='Simple initialization file')
    elif choice == '2':
        batch_audit_pattern('scripts/', notes='Command-line utility script')
    elif choice == '3':
        batch_audit_pattern('test', notes='Unit test file')
    elif choice == '4':
        pattern = input("Enter pattern to match: ")
        batch_audit_pattern(pattern)
    elif choice == '5':
        os.system('python audit_tracker.py --status')
    elif choice == '6':
        os.system('python audit_tracker.py --report')
    elif choice == '0':
        return False
    else:
        print("Invalid choice")

    return True

def main():
    if len(sys.argv) > 1:
        # Command line usage
        if sys.argv[1] == 'pattern':
            pattern = sys.argv[2] if len(sys.argv) > 2 else input("Pattern: ")
            batch_audit_pattern(pattern)
        elif sys.argv[1] == 'file':
            filepath = sys.argv[2] if len(sys.argv) > 2 else input("File: ")
            notes = sys.argv[3] if len(sys.argv) > 3 else ""
            audit_file(filepath, notes=notes)
    else:
        # Interactive menu
        while quick_audit_menu():
            pass

if __name__ == '__main__':
    main()