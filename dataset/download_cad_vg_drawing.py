#!/usr/bin/env python3
"""
DeepV CAD-VGDrawing Dataset Download Script

Cross-platform Python script for downloading and setting up the CAD-VGDrawing dataset.
This dataset contains 161k SVG-to-CAD pairs for parametric CAD conversion tasks.

Usage:
    python dataset/download_cad_vg_drawing.py
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CADVGDrawingDownloader:
    """Downloader for CAD-VGDrawing dataset."""

    def __init__(self, base_dir: Optional[str] = None):
        # Get the repository root (parent of dataset directory)
        if base_dir is None:
            current_file = Path(__file__)
            # Go up one level from dataset/ to repo root
            self.base_dir = current_file.parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.dataset_dir = self.base_dir / "datasets" / "cad_vg_drawing"
        self.dataset_url = "https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU"

        # Expected files in the dataset
        self.expected_files = [
            "train_data.json",
            "val_data.json",
            "test_data.json",
            "README.md"
        ]

    def create_dataset_directory(self) -> None:
        """Create the dataset directory structure."""
        logger.info("Creating dataset directory...")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {self.dataset_dir}")

    def create_readme(self) -> None:
        """Create README file with dataset information."""
        readme_content = f"""# CAD-VGDrawing Dataset Integration

This directory contains the CAD-VGDrawing dataset downloaded from:
{self.dataset_url}

## Dataset Structure
- `train_data.json`: Training data with SVG-to-CAD pairs
- `val_data.json`: Validation data
- `test_data.json`: Test data
- SVG files: Vector drawings
- PNG files: Rasterized versions (if available)

## Dataset Statistics
- Total samples: ~161,000
- Format: SVG-to-parametric CAD conversion
- Use case: Parametric CAD generation from vector drawings

## Usage
After downloading, run the preprocessing script:
```bash
python dataset/preprocess_cad_vg_drawing.py
```

This will convert the dataset into the format expected by DeepV's training pipeline.

## Source
Paper: Drawing2CAD - "Drawing2CAD: Automated Conversion of Mechanical Drawings to CAD Models"
"""

        readme_path = self.dataset_dir / "README.md"
        readme_path.write_text(readme_content)
        logger.info(f"Created README: {readme_path}")

    def check_existing_files(self) -> Dict[str, bool]:
        """Check which expected files already exist."""
        existing = {}
        for filename in self.expected_files:
            file_path = self.dataset_dir / filename
            existing[filename] = file_path.exists()
        return existing

    def attempt_download(self) -> bool:
        """Attempt to download dataset files. Returns True if successful."""
        logger.info("Attempting to download CAD-VGDrawing dataset...")

        # Note: Google Drive folder downloads are complex and may require manual intervention
        # For now, we'll provide instructions for manual download

        logger.warning("⚠️  MANUAL DOWNLOAD REQUIRED:")
        logger.warning(f"   Please visit: {self.dataset_url}")
        logger.warning("   Download all files from the 'CAD-VGDrawing' folder")
        logger.warning("")
        logger.warning("Expected files:")
        for filename in self.expected_files:
            logger.warning(f"   - {filename}")
        logger.warning("   - And corresponding SVG/PNG files")
        logger.warning("")

        return False  # Manual download required

    def validate_dataset(self) -> bool:
        """Validate that the dataset is properly downloaded and structured."""
        logger.info("Validating dataset...")

        existing = self.check_existing_files()

        # Check for required JSON files
        json_files = ["train_data.json", "val_data.json", "test_data.json"]
        json_present = all(existing.get(f, False) for f in json_files)

        if not json_present:
            logger.error("❌ Required JSON files not found. Please download the dataset first.")
            return False

        # Try to validate JSON structure
        try:
            for json_file in json_files:
                file_path = self.dataset_dir / json_file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"✅ {json_file}: {len(data)} samples")
        except Exception as e:
            logger.error(f"❌ Error validating JSON files: {e}")
            return False

        logger.info("✅ Dataset validation successful!")
        return True

    def setup_preprocessing_script(self) -> None:
        """Create or update the preprocessing script."""
        preprocess_script = self.base_dir / "dataset" / "preprocess_cad_vg_drawing.py"

        if not preprocess_script.exists():
            logger.info("Creating preprocessing script template...")

            script_content = '''#!/usr/bin/env python3
"""
Preprocessing script for CAD-VGDrawing dataset.

This script converts the raw CAD-VGDrawing dataset into the format expected
by DeepV's training pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_cad_vg_drawing():
    """Preprocess the CAD-VGDrawing dataset."""
    dataset_dir = Path(__file__).parent / "datasets" / "cad_vg_drawing"

    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return

    logger.info("Preprocessing CAD-VGDrawing dataset...")

    # TODO: Implement preprocessing logic
    # - Convert SVG files to internal format
    # - Extract CAD parameters
    # - Create train/val/test splits
    # - Generate metadata

    logger.info("Preprocessing completed!")

if __name__ == "__main__":
    preprocess_cad_vg_drawing()
'''

            preprocess_script.write_text(script_content)
            logger.info(f"Created preprocessing script: {preprocess_script}")

    def run(self) -> None:
        """Run the complete download and setup process."""
        print("DeepV CAD-VGDrawing Dataset Download Script")
        print("=" * 50)
        print()

        logger.info("📥 Setting up CAD-VGDrawing Dataset (161k SVG-to-CAD pairs)...")
        logger.info(f"   Source: {self.dataset_url}")
        print()

        # Create directory structure
        self.create_dataset_directory()
        self.create_readme()

        # Check existing files
        existing = self.check_existing_files()
        logger.info("Existing files:")
        for filename, exists in existing.items():
            status = "✅" if exists else "❌"
            logger.info(f"   {status} {filename}")

        # Attempt download
        download_success = self.attempt_download()

        if download_success:
            logger.info("✅ Download completed successfully!")
        else:
            logger.info("📋 Manual download required (see instructions above)")

        # Setup preprocessing
        self.setup_preprocessing_script()

        print()
        logger.info("📋 Next steps:")
        logger.info("   1. Download files from Google Drive link above")
        logger.info("   2. Place files in datasets/cad_vg_drawing/")
        logger.info("   3. Run: python dataset/preprocess_cad_vg_drawing.py")
        print()


def main():
    """Main entry point."""
    try:
        downloader = CADVGDrawingDownloader()
        downloader.run()
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()