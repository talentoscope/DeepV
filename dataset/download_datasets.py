#!/usr/bin/env python3
"""
DeepV Dataset Download Script

Cross-platform Python script for downloading datasets used by DeepV.
Currently supports the Golden Set evaluation dataset.

Usage:
    python dataset/download_datasets.py
"""

import os
import sys
import zipfile
import shutil
import requests
from pathlib import Path
from typing import Optional
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Cross-platform dataset downloader for DeepV."""

    def __init__(self, base_dir: Optional[str] = None):
        # Get the repository root (parent of dataset directory)
        if base_dir is None:
            current_file = Path(__file__)
            # Go up one level from dataset/ to repo root
            self.base_dir = current_file.parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.data_dir = self.base_dir / "data"

    def download_golden_set(self) -> bool:
        """Download and extract the Golden Set evaluation dataset."""
        logger.info("✅ Downloading Golden Set (evaluation dataset from paper)...")

        file_id = "1dDs06LsLNQUg9HvUwNBIq-95bjmRAiMh"
        filename = "golden_set.zip"
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        try:
            # Download the file
            logger.info(f"Downloading {filename}...")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            zip_path = self.data_dir / filename
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded {filename}")

            # Extract the zip file
            logger.info("Extracting golden_set.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            # Clean up
            zip_path.unlink()

            # Remove __MACOSX directory if it exists
            macosx_dir = self.data_dir / "__MACOSX"
            if macosx_dir.exists():
                shutil.rmtree(macosx_dir)
                logger.info("Removed __MACOSX directory")

            logger.info("✅ SUCCESS: Golden Set downloaded and extracted")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download Golden Set: {e}")
            return False

    def show_removed_datasets(self) -> None:
        """Show information about removed/broken datasets."""
        print()
        logger.warning("❌ REMOVED: ABC dataset (links broken, use synthetic generation instead)")
        logger.warning("❌ REMOVED: Background images (Google Drive link broken)")
        logger.warning("❌ REMOVED: Dataset_of_cleaning (requires manual Yandex Disk download)")
        logger.warning("❌ REMOVED: Precision Floorplan (website down)")
        print()

    def show_synthetic_generation_info(self) -> None:
        """Show information about synthetic dataset generation."""
        logger.info("For missing datasets, use synthetic generation:")
        logger.info("python scripts/create_test_datasets.py")
        print()

    def run(self) -> None:
        """Run the complete download process."""
        print("DeepV Dataset Download Script")
        print("=" * 30)
        print()

        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)

        # Download available datasets
        success = self.download_golden_set()

        # Show information about removed datasets
        self.show_removed_datasets()

        if success:
            logger.info("✅ SUCCESS: All available datasets downloaded and extracted")
        else:
            logger.warning("⚠️  Some downloads failed")

        # Show synthetic generation info
        self.show_synthetic_generation_info()


def main():
    """Main entry point."""
    try:
        downloader = DatasetDownloader()
        downloader.run()
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()