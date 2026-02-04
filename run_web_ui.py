#!/usr/bin/env python3
"""
Run script for DeepV Web UI

Usage:
    python run_web_ui.py

Or directly:
    python web_ui/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web_ui.app import create_interface

if __name__ == "__main__":
    print("Starting DeepV Web UI...")
    print("This is a demo interface showcasing rendering capabilities.")
    print("For full vectorization, use the main pipeline scripts.")

    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )