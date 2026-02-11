#!/usr/bin/env python3
"""
DeepV Codebase Audit Progress Tracker

Tracks progress of the comprehensive file-by-file audit for Phase 0 of the
codebase optimization project.

Usage:
    python audit_tracker.py --init          # Initialize audit tracking
    python audit_tracker.py --list          # List all Python files to audit
    python audit_tracker.py --status        # Show current audit progress
    python audit_tracker.py --audit FILE    # Mark file as audited
    python audit_tracker.py --report        # Generate audit report
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditTracker:
    """Tracks progress of the comprehensive codebase audit."""

    def __init__(self, audit_file: str = "audit_progress.json"):
        self.audit_file = Path(audit_file)
        self.exclude_dirs = {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "data",
            "logs",
            "models",
            "vectorization/models/specs",
            "notebooks",
            "docs/_build",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            "*.egg-info",
        }
        self.exclude_patterns = ["test_*", "*_test.py", "conftest.py"]

    def get_python_files(self) -> List[str]:
        """Get all Python files in the project, excluding common directories."""
        python_files = []

        for root, dirs, files in os.walk("."):
            # Skip excluded directories
            dirs[:] = [
                d
                for d in dirs
                if d not in self.exclude_dirs and not any(pattern in d for pattern in ["__pycache__", ".git"])
            ]

            for file in files:
                if file.endswith(".py"):
                    # Skip test files and other excluded patterns
                    if not any(
                        pattern.replace("*", "").replace("_", "") in file.lower() for pattern in self.exclude_patterns
                    ):
                        rel_path = os.path.relpath(os.path.join(root, file))
                        python_files.append(rel_path)

        return sorted(python_files)

    def load_progress(self) -> Dict[str, Any]:
        """Load audit progress from file."""
        if self.audit_file.exists():
            with open(self.audit_file, "r") as f:
                return json.load(f)  # type: ignore
        return {
            "total_files": 0,
            "audited_files": {},
            "start_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def save_progress(self, progress: Dict[str, Any]) -> None:
        """Save audit progress to file."""
        progress["last_updated"] = datetime.now().isoformat()
        with open(self.audit_file, "w") as f:
            json.dump(progress, f, indent=2)

    def initialize_audit(self) -> None:
        """Initialize the audit tracking system."""
        print("🔍 Initializing DeepV Codebase Audit Tracker...")
        # Get all Python files
        python_files = self.get_python_files()
        print(f"📁 Found {len(python_files)} Python files to audit")

        # Initialize progress
        progress = {
            "total_files": len(python_files),
            "files_to_audit": python_files,
            "audited_files": {},
            "start_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "audit_categories": {
                "core_pipeline": [],
                "vectorization": [],
                "refinement": [],
                "merging": [],
                "cleaning": [],
                "utilities": [],
                "scripts": [],
                "tests": [],
                "other": [],
            },
        }

        self.save_progress(progress)
        print(f"✅ Audit tracking initialized with {len(python_files)} files")
        print(f"📊 Progress file: {self.audit_file}")

    def categorize_file(self, filepath: str) -> str:
        """Categorize a file based on its path."""
        path_parts = Path(filepath).parts

        if "scripts" in path_parts:
            return "scripts"
        elif "vectorization" in path_parts:
            return "vectorization"
        elif "refinement" in path_parts:
            return "refinement"
        elif "merging" in path_parts:
            return "merging"
        elif "cleaning" in path_parts:
            return "cleaning"
        elif "tests" in path_parts:
            return "tests"
        elif any(part in ["pipeline", "run_pipeline"] for part in path_parts):
            return "core_pipeline"
        elif "util_files" in path_parts or "utils" in path_parts:
            return "utilities"
        else:
            return "other"

    def mark_file_audited(
        self, filepath: str, status: str = "completed", notes: str = "", issues: Optional[List[str]] = None
    ):
        """Mark a file as audited."""
        progress = self.load_progress()

        if filepath not in progress["files_to_audit"]:
            print(f"⚠️  File {filepath} not in audit list")
            return

        category = self.categorize_file(filepath)

        progress["audited_files"][filepath] = {
            "status": status,
            "category": category,
            "audited_date": datetime.now().isoformat(),
            "notes": notes,
            "issues": issues or [],
        }

        # Only add to category if not already present
        if filepath not in progress["audit_categories"][category]:
            progress["audit_categories"][category].append(filepath)

        self.save_progress(progress)
        print(f"✅ Marked {filepath} as audited ({status})")

    def show_status(self):
        """Show current audit progress."""
        progress = self.load_progress()

        total = progress["total_files"]
        audited = len(progress["audited_files"])
        remaining = total - audited

        if total == 0:
            print("❌ Audit not initialized. Run with --init first.")
            return

        percentage = (audited / total) * 100

        print("📊 DeepV Codebase Audit Progress")
        print("=" * 40)
        print(f"Total files: {total}")
        print(f"Audited: {audited}")
        print(f"Remaining: {remaining}")
        print(f"Progress: {percentage:.1f}%")
        print(f"Start date: {progress['start_date'][:10]}")
        print(f"Last updated: {progress['last_updated'][:10]}")
        print()

        # Show category breakdown
        print("📂 Category Breakdown:")
        for category, files in progress["audit_categories"].items():
            if files:
                audited_in_cat = len([f for f in files if f in progress["audited_files"]])
                print(f"  {category}: {audited_in_cat}/{len(files)}")

        print()
        print("🎯 Next Priority Files:")
        unaudited = [f for f in progress["files_to_audit"] if f not in progress["audited_files"]]
        for file in unaudited[:5]:  # Show next 5
            print(f"  • {file}")

    def generate_report(self):
        """Generate a comprehensive audit report."""
        progress = self.load_progress()

        print("📋 DeepV Codebase Audit Report")
        print("=" * 50)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print()

        # Overall progress
        total = progress["total_files"]
        audited = len(progress["audited_files"])
        percentage = (audited / total) * 100 if total > 0 else 0

        print("📊 Overall Progress:")
        print(f"  Files audited: {audited}/{total} ({percentage:.1f}%)")
        print()

        # Category breakdown
        print("📂 Category Breakdown:")
        for category, files in progress["audit_categories"].items():
            if files:
                audited_in_cat = len([f for f in files if f in progress["audited_files"]])
                cat_percentage = (audited_in_cat / len(files)) * 100 if files else 0
                print(f"  {category}: {audited_in_cat}/{len(files)} ({cat_percentage:.1f}%)")
        # Issues summary
        all_issues = []
        for file_data in progress["audited_files"].values():
            all_issues.extend(file_data.get("issues", []))

        if all_issues:
            print()
            print("🚨 Issues Found:")
            issue_counts: dict[str, int] = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {issue}: {count}")

        # Save detailed report
        report_file = Path("audit_report.md")
        with open(report_file, "w") as f:
            f.write("# DeepV Codebase Audit Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## Overall Progress\n\n")
            f.write(f"- Files audited: {audited}/{total} ({percentage:.1f}%)\n\n")
            f.write("## Detailed File Status\n\n")

            for filepath, data in progress["audited_files"].items():
                f.write(f"### {filepath}\n")
                f.write(f"- Status: {data['status']}\n")
                f.write(f"- Category: {data['category']}\n")
                f.write(f"- Audited: {data['audited_date'][:10]}\n")
                if data.get("notes"):
                    f.write(f"- Notes: {data['notes']}\n")
                if data.get("issues"):
                    f.write(f"- Issues: {', '.join(data['issues'])}\n")
                f.write("\n")

        print(f"\n📄 Detailed report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="DeepV Codebase Audit Tracker")
    parser.add_argument("--init", action="store_true", help="Initialize audit tracking")
    parser.add_argument("--list", action="store_true", help="List all Python files to audit")
    parser.add_argument("--status", action="store_true", help="Show current audit progress")
    parser.add_argument("--audit", metavar="FILE", help="Mark file as audited")
    parser.add_argument("--report", action="store_true", help="Generate audit report")
    parser.add_argument("--notes", default="", help="Notes for audited file")
    parser.add_argument("--issues", nargs="*", default=[], help="Issues found in file")

    args = parser.parse_args()

    tracker = AuditTracker()

    if args.init:
        tracker.initialize_audit()
    elif args.list:
        files = tracker.get_python_files()
        print(f"📁 Found {len(files)} Python files:")
        for file in files:
            print(f"  {file}")
    elif args.status:
        tracker.show_status()
    elif args.audit:
        tracker.mark_file_audited(args.audit, notes=args.notes, issues=args.issues)
    elif args.report:
        tracker.generate_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
