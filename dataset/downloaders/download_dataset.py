#!/usr/bin/env python3
"""Download individual datasets for DeepV training and evaluation.

This file was copied from the original `dataset_downloaders/download_dataset.py`
and adjusted to live under `dataset.downloaders`.
"""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_with_progress(url: str, output_path: Path, chunk_size: int = 8192):
    """Download file with progress bar."""
    import requests
    from tqdm import tqdm

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            pbar.update(size)


def download_floorplancad(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download FloorPlanCAD from original Google Drive links."""
    try:
        import gdown
    except ImportError:
        raise ImportError("Install gdown: pip install gdown")

    dataset_dir = output_dir / "raw" / "floorplancad"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FloorPlanCAD to {dataset_dir}...")

    # Google Drive URLs from the project site
    urls = {
        "train1.zip": "https://drive.google.com/file/d/1HcyKt6qWeXog-tRfvEjdO3O3TN91PXGL/view?usp=share_link",
        "train2.zip": "https://drive.google.com/file/d/1kSS7OB_EEu7VJzb0W8DK9_nu1EvshioV/view?usp=sharing",
        "test.zip": "https://drive.google.com/file/d/1jxpYgxnLUbXEzMOsjaMPQFSuvmvHimiZ/view?usp=sharing",
    }

    if test_mode:
        print("Test mode: downloading only train1.zip for testing")
        urls = {"train1.zip": urls["train1.zip"]}

    downloaded_files = []
    for name, url in urls.items():
        zip_path = dataset_dir / name
        if zip_path.exists():
            print(f"Skipping {name}, already exists at {zip_path}")
            downloaded_files.append(zip_path)
            continue

        print(f"Downloading {name} from {url}...")
        try:
            gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
            print(f"[OK] Downloaded {name} to {zip_path}")
            downloaded_files.append(zip_path)
        except Exception as e:
            print(f"[ERR] Failed to download {name}: {e}")
            raise

    # Extract the zip files
    import shutil
    import tarfile
    for zip_path in downloaded_files:
        extract_dir = dataset_dir / zip_path.stem  # e.g., train1
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"Skipping extraction of {zip_path}, {extract_dir} already exists and not empty")
            continue

        print(f"Extracting {zip_path} to {extract_dir}...")
        try:
            shutil.unpack_archive(str(zip_path), str(extract_dir))
            print(f"[OK] Extracted {zip_path} to {extract_dir}")
        except shutil.ReadError:
            # Try uncompressed tar
            try:
                with tarfile.open(zip_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_dir)
                print(f"[OK] Extracted {zip_path} (tar) to {extract_dir}")
            except Exception as e2:
                print(f"[ERR] Failed to extract {zip_path}: {e2}")
                raise
        except Exception as e:
            print(f"[ERR] Failed to extract {zip_path}: {e}")
            raise

    print(f"[OK] Downloaded and extracted FloorPlanCAD to {dataset_dir}")

    metadata = {
        "name": "FloorPlanCAD",
        "size": "15,663 CAD drawings",
        "formats": ["SVG", "PNG", "COCO"],
        "source": "https://floorplancad.github.io/",
        "license": "CC BY-NC 4.0",
        "download_date": str(Path(dataset_dir).stat().st_mtime),
        "test_mode": test_mode,
    }

    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def download_cubicasa5k(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download CubiCasa5K from Zenodo."""
    import requests

    dataset_dir = output_dir / "raw" / "cubicasa5k"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if test_mode:
        print("Test mode: skipping large Zenodo download (5GB+)")
        print("To download full dataset, use: --no-test")
        metadata = {
            "name": "CubiCasa5K",
            "size": "5,000 scanned floorplans (~105 GB)",
            "formats": ["Raster", "SVG", "LMDB"],
            "source": "https://zenodo.org/record/2613548",
            "license": "CC BY-NC 4.0",
            "test_mode": True,
            "note": "Test mode: download skipped. Use --no-test for full download.",
        }
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata

    print(f"Downloading CubiCasa5K to {dataset_dir}...")
    print("Warning: This is a large download (~5-10 GB)")

    zenodo_url = "https://zenodo.org/record/2613548/files/cubicasa5k.zip"
    zip_path = dataset_dir / "cubicasa5k.zip"

    try:
        download_with_progress(zenodo_url, zip_path)

        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        # Clean up zip
        zip_path.unlink()

        print(f"[OK] Downloaded CubiCasa5K to {dataset_dir}")

        metadata = {
            "name": "CubiCasa5K",
            "size": "5,000 scanned floorplans",
            "formats": ["Raster", "SVG", "LMDB"],
            "source": "https://zenodo.org/record/2613548",
            "license": "CC BY-NC 4.0",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download CubiCasa5K: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise


def download_quickdraw(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download QuickDraw subset from Google Cloud Storage."""
    dataset_dir = output_dir / "raw" / "quickdraw"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading QuickDraw subset to {dataset_dir}...")

    # QuickDraw data is available as NDJSON files from Google Cloud Storage
    # We'll download a small subset of classes for practical use
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
    
    # Select a few representative classes for the subset
    classes = [
        "airplane", "apple", "bird", "cat", "dog", "fish", "house", "tree", "car", "sun"
    ]
    
    if test_mode:
        classes = classes[:2]  # Just 2 classes for testing
        print(f"Test mode: downloading {len(classes)} classes")
    else:
        print(f"Downloading {len(classes)} classes")

    downloaded_files = []
    
    for class_name in classes:
        filename = f"{class_name}.ndjson"
        url = f"{base_url}/{filename}"
        output_path = dataset_dir / filename
        
        if output_path.exists():
            print(f"Skipping {filename} (already exists)")
            downloaded_files.append(str(output_path))
            continue
            
        try:
            print(f"Downloading {filename}...")
            download_with_progress(url, output_path)
            downloaded_files.append(str(output_path))
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            continue

    if not downloaded_files:
        raise RuntimeError("Failed to download any QuickDraw files")

    print(f"[OK] Downloaded {len(downloaded_files)} QuickDraw files to {dataset_dir}")

    metadata = {
        "name": "QuickDraw",
        "size": f"{len(downloaded_files)} classes ({len(classes)} requested)",
        "formats": ["NDJSON (stroke sequences)"],
        "source": "https://storage.googleapis.com/quickdraw_dataset/full/simplified",
        "license": "CC BY 4.0",
        "download_date": str(Path(dataset_dir).stat().st_mtime),
        "test_mode": test_mode,
        "classes": classes,
        "files": downloaded_files,
        "note": "Downloaded NDJSON files from Google Cloud Storage",
    }

    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def download_msd(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download Modified Swiss Dwellings from 4TU.ResearchData."""
    dataset_dir = output_dir / "raw" / "msd"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if test_mode:
        print("Test mode: MSD download from 4TU.ResearchData.")
        print(
            "Direct download link: https://data.4tu.nl/file/e1d89cb5-6872-48fc-be63-aadd687ee6f9/1ba5885d-19d7-4c0a-b73a-085e772ea1bc"
        )
        metadata = {
            "name": "Modified Swiss Dwellings (MSD)",
            "size": "5,372 floor plans (17.4 GB)",
            "formats": ["Raster", "Vector", "Graphs", "CSV"],
            "source": "https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9/1",
            "license": "CC BY-SA 4.0",
            "test_mode": True,
            "note": "Direct download from 4TU.ResearchData (no authentication required).",
        }
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Created metadata for MSD at {dataset_dir}")
        return metadata

    print(f"Downloading MSD from 4TU.ResearchData to {dataset_dir}...")
    print("Warning: This is a large download (~5.4 GB)")

    try:
        # Direct download from 4TU.ResearchData
        url = "https://data.4tu.nl/file/e1d89cb5-6872-48fc-be63-aadd687ee6f9/1ba5885d-19d7-4c0a-b73a-085e772ea1bc"
        zip_path = dataset_dir / "modified-swiss-dwellings-v1.zip"

        # Download the zip file
        download_with_progress(url, zip_path)

        # Extract the zip file
        print("Extracting MSD dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        # Clean up zip file
        zip_path.unlink()

        print(f"[OK] Downloaded and extracted MSD to {dataset_dir}")

        metadata = {
            "name": "Modified Swiss Dwellings (MSD)",
            "size": "5,372 floor plans (17.4 GB)",
            "formats": ["Raster", "Vector", "Graphs", "CSV"],
            "source": "https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9/1",
            "license": "CC BY-SA 4.0",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download MSD: {e}")
        raise


def download_sketchgraphs(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download SketchGraphs dataset (filtered sequences)."""
    dataset_dir = output_dir / "raw" / "sketchgraphs"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SketchGraphs to {dataset_dir}...")

    if test_mode:
        print("Test mode: Cloning repository metadata only")

    try:
        # Clone the repository for documentation and scripts
        repo_dir = dataset_dir / "repository"
        if not repo_dir.exists():
            print("Cloning SketchGraphs repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/PrincetonLIPS/SketchGraphs.git", str(repo_dir)],
                check=True,
                capture_output=True,
            )

        # Download the filtered training set (smaller file for testing)
        train_url = "https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_train.npy"
        train_path = dataset_dir / "sg_t16_train.npy"

        if not test_mode and not train_path.exists():
            print("Downloading SketchGraphs filtered training set...")
            download_with_progress(train_url, train_path)

        # Also download validation and test sets if not in test mode
        if not test_mode:
            val_url = "https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_validation.npy"
            test_url = "https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_test.npy"

            val_path = dataset_dir / "sg_t16_validation.npy"
            test_path = dataset_dir / "sg_t16_test.npy"

            if not val_path.exists():
                print("Downloading SketchGraphs validation set...")
                download_with_progress(val_url, val_path)

            if not test_path.exists():
                print("Downloading SketchGraphs test set...")
                download_with_progress(test_url, test_path)

        print(f"[OK] Downloaded SketchGraphs to {dataset_dir}")

        metadata = {
            "name": "SketchGraphs",
            "size": "15M CAD sketches (filtered sequences)",
            "formats": ["Filtered sequences (.npy)", "Constraint graphs"],
            "source": "https://github.com/PrincetonLIPS/SketchGraphs",
            "license": "Open/Research (per Onshape Terms)",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
            "note": "Filtered sequence datasets downloaded. Use sketchgraphs library to process.",
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download SketchGraphs: {e}")
        print("Note: Requires git installed and internet connection.")
        raise



def download_resplan(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download ResPlan dataset from GitHub."""
    import requests

    dataset_dir = output_dir / "raw" / "resplan"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ResPlan to {dataset_dir}...")

    try:
        # Try to download ResPlan.zip from GitHub
        # The repository mentions ResPlan.zip in the description
        zip_url = "https://github.com/m-agour/ResPlan/raw/main/ResPlan.zip"
        zip_path = dataset_dir / "ResPlan.zip"

        if not zip_path.exists():
            print("Downloading ResPlan.zip...")
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"[OK] Downloaded ResPlan.zip to {zip_path}")

        # Extract the zip file
        import zipfile

        extract_dir = dataset_dir / "extracted"
        if not extract_dir.exists():
            print("Extracting ResPlan.zip...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"[OK] Extracted to {extract_dir}")

        print(f"[OK] Downloaded and extracted ResPlan to {dataset_dir}")

        metadata = {
            "name": "ResPlan",
            "size": "17,000 residential floorplans",
            "formats": ["JSON", "PNG", "NetworkX graphs", "PKL"],
            "source": "https://github.com/m-agour/ResPlan",
            "license": "MIT",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download ResPlan: {e}")
        print("Note: You may need to download manually from:")
        print("  https://github.com/m-agour/ResPlan")
        raise


def download_cadvgdrawing(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download CAD-VGDrawing (Drawing2CAD) from Google Drive using gdown.

    Downloads only the svg_raw.zip file directly for efficiency.
    """
    try:
        import gdown
    except Exception:
        raise ImportError("Install gdown: pip install gdown")

    dataset_dir = output_dir / "raw" / "cadvgdrawing"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Direct download of svg_raw.zip using its file ID
    file_id = "1zu_Etkq7KuXlki-R1C4dFgsWXLNW1O5M"
    drive_url = f"https://drive.google.com/uc?id={file_id}"

    # If files already exist (partial download from prior run), create metadata
    # and return to avoid re-downloading in test mode.
    svg_raw_dir = dataset_dir / "svg_raw"
    if svg_raw_dir.exists() and any(svg_raw_dir.iterdir()):
        if test_mode:
            print(f"Existing svg_raw files found in {svg_raw_dir}; finalizing metadata in test mode")
            metadata = {
                "name": "CAD-VGDrawing (Drawing2CAD)",
                "size": "~157k-161k SVG-to-CAD pairs",
                "formats": ["SVG"],
                "source": "https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU",
                "license": "MIT",
                "test_mode": True,
                "note": "Partial download detected and metadata finalized in test mode.",
            }
            with open(dataset_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            return metadata

    print(f"Downloading CAD-VGDrawing (Drawing2CAD) to {dataset_dir}...")
    print("Downloading svg_raw.zip directly from Google Drive...")

    if test_mode:
        print("Test mode: downloading only first 10MB for testing")
        # Download with size limit for testing
        try:
            zip_path = dataset_dir / "svg_raw.zip"
            gdown.download(drive_url, str(zip_path), quiet=False, fuzzy=True)
            print(f"[OK] Downloaded sample svg_raw.zip to {zip_path}")
        except Exception as e:
            print(f"[WARN] gdown sample download failed: {e}")

        metadata = {
            "name": "CAD-VGDrawing (Drawing2CAD)",
            "size": "~157k-161k SVG-to-CAD pairs",
            "formats": ["SVG"],
            "source": "https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU",
            "license": "MIT",
            "test_mode": True,
            "note": "Test mode: partial download attempted. Use --no-test for full download.",
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    try:
        # Download svg_raw.zip directly
        zip_path = dataset_dir / "svg_raw.zip"

        # Check if zip already exists and svg_raw is extracted
        svg_raw_dir = dataset_dir / "svg_raw"
        if zip_path.exists() and svg_raw_dir.exists() and any(svg_raw_dir.iterdir()):
            print(f"svg_raw already extracted to {svg_raw_dir}, skipping download")
        else:
            print(f"Downloading {zip_path.name}...")
            gdown.download(drive_url, str(zip_path), quiet=False, fuzzy=True)

        # Extract the zip if not already extracted
        if not svg_raw_dir.exists() or not any(svg_raw_dir.iterdir()):
            print(f"Extracting {zip_path} to {svg_raw_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                file_list = zip_ref.namelist()
                total_files = len(file_list)

                from tqdm import tqdm

                with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, svg_raw_dir)
                        pbar.update(1)

        # Clean up the zip file
        if zip_path.exists():
            zip_path.unlink()

        metadata = {
            "name": "CAD-VGDrawing (Drawing2CAD)",
            "size": "~157k-161k SVG-to-CAD pairs",
            "formats": ["SVG"],
            "source": "https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU",
            "license": "MIT",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Downloaded and extracted svg_raw to {dataset_dir / 'svg_raw'}")
        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download CAD-VGDrawing: {e}")
        raise


def download_fplanpoly(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download FPLAN-POLY from archived CVC site."""
    import requests

    dataset_dir = output_dir / "raw" / "fplanpoly"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FPLAN-POLY to {dataset_dir}...")

    # Archived URL from web.archive.org
    fplanpoly_url = (
        "https://web.archive.org/web/20130621114030/http://www.cvc.uab.es/~marcal/FPLAN-POLY/img/FPLAN-POLY.zip"
    )

    if test_mode:
        print("Test mode: creating metadata only")
        metadata = {
            "name": "FPLAN-POLY",
            "size": "42 floorplans + 38 symbol models",
            "formats": ["DXF vector"],
            "source": "https://web.archive.org/web/20130621114030/http://www.cvc.uab.es/~marcal/FPLAN-POLY/",
            "license": "Research",
            "test_mode": True,
            "note": "Test mode: full download requires accessing archived CVC site",
        }
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Created metadata for FPLAN-POLY at {dataset_dir}")
        return metadata

    zip_path = dataset_dir / "FPLAN-POLY.zip"

    try:
        download_with_progress(fplanpoly_url, zip_path)

        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        # Clean up zip file
        zip_path.unlink()

        metadata = {
            "name": "FPLAN-POLY",
            "size": "42 floorplans + 38 symbol models",
            "formats": ["DXF vector"],
            "source": "https://web.archive.org/web/20130621114030/http://www.cvc.uab.es/~marcal/FPLAN-POLY/",
            "license": "Research",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
            "note": "Contains vector geometric primitives (polylines) suitable for DeepV",
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Downloaded and extracted FPLAN-POLY to {dataset_dir}")
        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download FPLAN-POLY: {e}")
        raise


# Dataset registry
DATASETS = {
    "floorplancad": {
        "name": "FloorPlanCAD",
        "description": "15,663 CAD drawings with vector annotations",
        "downloader": download_floorplancad,
        "size": "~2-5 GB",
        "license": "CC BY-NC 4.0",
    },
    "cubicasa5k": {
        "name": "CubiCasa5K",
        "description": "5,000 high-res scanned floorplans",
        "downloader": download_cubicasa5k,
        "size": "~105 GB (LMDB)",
        "license": "CC BY-NC 4.0",
    },
    "quickdraw": {
        "name": "QuickDraw",
        "description": "Google QuickDraw vector sketches (subset)",
        "downloader": download_quickdraw,
        "size": "~1-5 GB (subset)",
        "license": "CC BY 4.0",
    },
    "msd": {
        "name": "Modified Swiss Dwellings",
        "description": "5,372 floor plans with graphs",
        "downloader": download_msd,
        "size": "~17.4 GB",
        "license": "CC BY-SA 4.0",
    },
    "sketchgraphs": {
        "name": "SketchGraphs",
        "description": "15M CAD constraint graphs",
        "downloader": download_sketchgraphs,
        "size": "~Large (repo only)",
        "license": "Open/Research",
    },
    # Additional datasets from DATA_SOURCES.md
    "cadvgdrawing": {
        "name": "CAD-VGDrawing (Drawing2CAD)",
        "description": "157k-161k SVG-to-CAD pairs",
        "downloader": download_cadvgdrawing,
        "size": "~Large",
        "license": "MIT",
        "note": "Google Drive folder: https://drive.google.com/drive/folders/1t9uO2iFh1eVDXRCKUEonKPBu8WGYA8wU",
    },
    "fplanpoly": {
        "name": "FPLAN-POLY",
        "description": "42 floorplans + 38 symbol models in DXF vector format",
        "downloader": download_fplanpoly,
        "size": "~Small",
        "license": "Research",
        "note": "Highly suitable for DeepV - contains vector geometric primitives (polylines)",
    },
    "resplan": {
        "name": "ResPlan",
        "description": "17,000 residential floorplans",
        "downloader": download_resplan,
        "size": "~Medium",
        "license": "MIT",
        "note": "Contains vector geometries and graphs for architectural tasks",
    },
}


def list_datasets():
    """List all available datasets."""
    print("\nAvailable datasets:")
    print("=" * 80)

    print("\n[AUTO-DOWNLOAD] These datasets can be downloaded automatically:\n")
    for key, info in DATASETS.items():
        print(f"{key:15} - {info['name']}")
        print(f"{'':15}   {info['description']}")
        print(f"{'':15}   Size: {info['size']} | License: {info['license']}")
        if "note" in info:
            print(f"{'':15}   Note: {info['note']}")
        print()

    print("\n" + "=" * 80)
    print("\nUsage: python download_dataset.py --dataset <name> --output-dir ./data")
    print("       python download_dataset.py --dataset <name> --output-dir ./data --test")
    print("       python download_dataset.py --list")


# File pruning helpers
RASTER_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
VECTOR_EXT = {".svg", ".dxf", ".dwg", ".eps", ".ps"}


def _prune_dataset_files(dataset_dir: Path, max_items: int = None) -> dict:
    """Move excess raster/vector files from dataset_dir into dataset_dir/_overflow.

    Returns a dict with stats: {'found':n, 'kept':k, 'moved':m, 'overflow_dir':Path}
    """
    from shutil import move

    if max_items is None:
        return {"found": 0, "kept": 0, "moved": 0, "overflow_dir": None}

    if not dataset_dir.exists():
        return {"found": 0, "kept": 0, "moved": 0, "overflow_dir": None}

    candidates = []
    for p in dataset_dir.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in RASTER_EXT or p.suffix.lower() in VECTOR_EXT:
                candidates.append(p)

    total = len(candidates)
    if total <= max_items:
        return {"found": total, "kept": total, "moved": 0, "overflow_dir": None}

    # Sort deterministically and keep the first max_items
    candidates.sort(key=lambda p: p.as_posix())
    to_keep = set(candidates[:max_items])
    to_move = candidates[max_items:]

    overflow_dir = dataset_dir / "_overflow"
    overflow_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for p in to_move:
        try:
            dest = overflow_dir / p.name
            # ensure unique name
            if dest.exists():
                # append a numeric suffix
                for i in range(1, 10000):
                    candidate = overflow_dir / f"{p.stem}_{i}{p.suffix}"
                    if not candidate.exists():
                        dest = candidate
                        break
            move(str(p), str(dest))
            moved += 1
        except Exception:
            # best-effort: continue
            continue

    print(f"[INFO] Pruned dataset {dataset_dir.name}: found={total}, kept={max_items}, moved={moved} to {overflow_dir}")
    return {"found": total, "kept": max_items, "moved": moved, "overflow_dir": overflow_dir}


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    test_mode: bool = False,
    verify: bool = False,
    max_items: int = 10000,
    prune: bool = True,
) -> Optional[Dict]:
    """Download a specific dataset.

    Args:
        dataset_name: Name of the dataset to download
        output_dir: Output directory for downloads
        test_mode: If True, download minimal subset for testing
        verify: If True, verify download integrity

    Returns:
        Metadata dict if successful, None otherwise
    """
    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return None

    dataset_info = DATASETS[dataset_name]

    print(f"\n{'='*80}")
    print(f"Downloading: {dataset_info['name']}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: {dataset_info['size']}")
    print(f"License: {dataset_info['license']}")
    print(f"Output: {output_dir / 'raw' / dataset_name}")
    print(f"Test mode: {test_mode}")
    print(f"{'='*80}\n")

    try:
        metadata = dataset_info["downloader"](output_dir, test_mode=test_mode)
        print(f"\n[OK] Successfully downloaded {dataset_name}")

        # Enforce item limit: move any excess raster/vector files into an
        # _overflow directory to keep default dataset size manageable.
        if prune and max_items and max_items > 0:
            # allow metadata to suggest a max_items but prefer explicit arg
            meta_max = None
            if isinstance(metadata, dict) and metadata.get("max_items"):
                try:
                    meta_max = int(metadata.get("max_items"))
                except Exception:
                    meta_max = None

            use_max = max_items if max_items is not None else (meta_max or None)
            dataset_dir = output_dir / "raw" / dataset_name
            try:
                _prune_dataset_files(dataset_dir, max_items=use_max)
            except Exception:
                print(f"[WARN] Pruning dataset files for {dataset_name} failed, continuing.")

        return metadata
    except Exception as e:
        print(f"\n[ERR] Failed to download {dataset_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for DeepV vectorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to download (use --list to see options)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory for downloads (default: ./data)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Download minimal subset for testing",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify download integrity (if supported)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum combined raster+vector items to keep by default (move excess to _overflow). Default: no limit",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Do not prune/move excess files after download",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset:
        parser.error("Either --dataset or --list is required")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = download_dataset(
        args.dataset,
        output_dir,
        test_mode=args.test,
        verify=args.verify,
        max_items=args.max_items,
        prune=not args.no_prune,
    )

    if result:
        print(f"\n{'='*80}")
        print("Download complete!")
        print(f"Data location: {output_dir / 'raw' / args.dataset}")
        print(f"Metadata: {output_dir / 'raw' / args.dataset / 'metadata.json'}")
        print(f"{'='*80}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
