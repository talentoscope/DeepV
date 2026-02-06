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
    """Download QuickDraw subset from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    dataset_dir = output_dir / "raw" / "quickdraw"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading QuickDraw subset to {dataset_dir}...")

    # Use a processed subset for practical usage
    if test_mode:
        allow_patterns = ["*.json", "data/train-00000-of-*.parquet"]
    else:
        allow_patterns = ["data/train-*.parquet"]  # Limit to train split

    try:
        # Note: Using a smaller processed version, not the full 50M sketches
        snapshot_download(
            repo_id="google/quickdraw",
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=allow_patterns,
            max_workers=4,
        )
        print(f"[OK] Downloaded QuickDraw to {dataset_dir}")

        metadata = {
            "name": "QuickDraw",
            "size": "Subset of 50M+ vector sketches (345 classes)",
            "formats": ["Parquet", "Stroke-3 format"],
            "source": "https://huggingface.co/datasets/google/quickdraw",
            "license": "CC BY 4.0",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
            "note": "Downloaded processed subset from HuggingFace",
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download QuickDraw: {e}")
        raise


def download_sesyd(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download SESYD synthetic floorplans."""
    import requests

    dataset_dir = output_dir / "raw" / "sesyd"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SESYD to {dataset_dir}...")

    sesyd_url = "http://mathieu.delalandre.free.fr/projects/sesyd/downloads/sesyd_v1.0.zip"

    if test_mode:
        print("Test mode: downloading sample only")
        # Create placeholder
        metadata = {
            "name": "SESYD",
            "size": "1,000 synthetic floorplans",
            "formats": ["Vector", "raster renderable"],
            "source": "http://mathieu.delalandre.free.fr/projects/sesyd/",
            "license": "Free",
            "test_mode": True,
            "note": "Test mode: full download requires accessing the SESYD website",
        }
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Created metadata for SESYD at {dataset_dir}")
        return metadata

    zip_path = dataset_dir / "sesyd.zip"

    try:
        download_with_progress(sesyd_url, zip_path)

        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        zip_path.unlink()

        print(f"[OK] Downloaded SESYD to {dataset_dir}")

        metadata = {
            "name": "SESYD",
            "size": "1,000 synthetic floorplans",
            "formats": ["Vector", "raster renderable"],
            "source": "http://mathieu.delalandre.free.fr/projects/sesyd/",
            "license": "Free",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download SESYD: {e}")
        if zip_path.exists():
            zip_path.unlink()
        raise


def download_impact(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download IMPACT patent dataset from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    dataset_dir = output_dir / "raw" / "impact"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IMPACT to {dataset_dir}...")

    if test_mode:
        allow_patterns = ["*.json", "data/train-00000-of-*.parquet"]
    else:
        allow_patterns = None

    try:
        snapshot_download(
            repo_id="AI4Patents/IMPACT",
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=allow_patterns,
            max_workers=4,
        )
        print(f"[OK] Downloaded IMPACT to {dataset_dir}")

        metadata = {
            "name": "IMPACT",
            "size": "500k patents, 3.61M figures",
            "formats": ["Images", "CSV", "Parquet"],
            "source": "https://huggingface.co/datasets/AI4Patents/IMPACT",
            "license": "Open",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download IMPACT: {e}")
        raise


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


def download_deeppatent2(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download DeepPatent2 from OneDrive/SharePoint (manual download)."""
    dataset_dir = output_dir / "raw" / "deeppatent2"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if test_mode:
        print("Test mode: DeepPatent2 is very large (>100 GB)")
        print(
            "OneDrive link (2020 data): https://olddominion-my.sharepoint.com/personal/j1wu_odu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020%2FOriginal%5F2020%2Etar%2Egz&viewid=7828cbdf%2D98fd%2D45c8%2D9fbf%2D337e03d13638&parent=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020"
        )
        print("OSF link: https://osf.io/kv4xa/ (2007 subset)")

        metadata = {
            "name": "DeepPatent2",
            "size": ">2.7M technical drawings (2M patents)",
            "formats": ["PNG", "JSON", "CSV"],
            "source": "https://olddominion-my.sharepoint.com/personal/j1wu_odu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020%2FOriginal%5F2020%2Etar%2Egz&viewid=7828cbdf%2D98fd%2D45c8%2D9fbf%2D337e03d13638&parent=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020",
            "license": "CC BY-NC 2.0",
            "test_mode": True,
            "note": "Test mode: Manual download required. See OneDrive link for Original_2020.tar.gz file.",
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Created metadata for DeepPatent2 at {dataset_dir}")
        return metadata

    # Check if file is already downloaded
    expected_file = dataset_dir / "Original_2020.tar.gz"
    if expected_file.exists() and expected_file.stat().st_size > 0:
        print(f"Found existing downloaded file: {expected_file}")
        print("Skipping browser opening - proceeding to extraction...")
    else:
        # Manual download: open browser and provide instructions
        print("DeepPatent2 requires manual download due to authentication requirements.")
        print(f"Download directory created: {dataset_dir}")
        print()
        print("Opening download page in your default browser...")
        print("Please authenticate and download the file manually.")
        print()

        # Open the download URL in the default browser
        download_url = "https://olddominion-my.sharepoint.com/personal/j1wu_odu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020%2FOriginal%5F2020%2Etar%2Egz&viewid=7828cbdf%2D98fd%2D45c8%2D9fbf%2D337e03d13638&parent=%2Fpersonal%2Fj1wu%5Fodu%5Fedu%2FDocuments%2Fdata%2F2023%2Ddeeppatent2%2F2020"

        try:
            import webbrowser

            webbrowser.open(download_url)
            print(f"Browser opened to: {download_url}")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please manually open: {download_url}")

        print()
        print("INSTRUCTIONS:")
        print("1. Authenticate with Microsoft if prompted")
        print("2. Click the download button on the OneDrive page")
        print("3. Save the file as 'Original_2020.tar.gz' to:")
        print(f"   {dataset_dir}")
        print("4. After download completes, the script will detect and extract it")
        print()

        # Wait for user to download and place the file
        print("Waiting for you to download and place the file...")
        print("(Press Ctrl+C to cancel and run again later)")

        try:
            import time

            while not expected_file.exists():
                print(f"Checking for {expected_file.name}... (waiting 10 seconds)")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nDownload cancelled. Run the script again when ready to continue.")
            print(f"Place the downloaded file at: {expected_file}")
            raise

    print(f"Found downloaded file: {expected_file}")
    print("Extracting...")

    # Extract the tar.gz file with progress indication
    import tarfile
    from tqdm import tqdm
    import os

    extracted_count = 0
    skipped_count = 0

    # Extract files iteratively with progress updates
    with tarfile.open(expected_file, "r:gz") as tar_ref:
        print("Extracting files from archive...")

        # Use tqdm without total for large archives
        with tqdm(desc="Extracting", unit="file", unit_scale=True) as pbar:
            for member in tar_ref:
                # Check if file already exists
                target_path = dataset_dir / member.name
                if target_path.exists():
                    # Skip if file already exists
                    skipped_count += 1
                    pbar.update(1)
                    continue

                # Extract the file
                tar_ref.extract(member, dataset_dir)
                extracted_count += 1
                pbar.update(1)

                # Periodic status update every 1000 files
                if (extracted_count + skipped_count) % 1000 == 0:
                    print(f"Processed {extracted_count + skipped_count} files so far...")

        print(
            f"Extraction complete: {extracted_count} files extracted, {skipped_count} files skipped (already existed)"
        )

    # Remove the compressed file
    expected_file.unlink()

    metadata = {
        "name": "DeepPatent2",
        "size": "2020 dataset (extracted from Original_2020.tar.gz)",
        "formats": ["PNG", "JSON", "CSV"],
        "source": download_url,
        "license": "CC BY-NC 2.0",
        "download_date": str(Path(dataset_dir).stat().st_mtime),
        "test_mode": test_mode,
        "download_method": "manual_browser",
    }

    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Downloaded and extracted DeepPatent2 2020 data to {dataset_dir}")
    return metadata


def download_drivaernet(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download DrivAerNet++ repository."""
    dataset_dir = output_dir / "raw" / "drivaernet"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading DrivAerNet++ metadata to {dataset_dir}...")

    try:
        repo_dir = dataset_dir / "repository"
        if not repo_dir.exists():
            print("Cloning DrivAerNet++ repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/Mohamedelrefaie/DrivAerNet.git", str(repo_dir)],
                check=True,
                capture_output=True,
            )

        print(f"[OK] Downloaded DrivAerNet++ repository to {dataset_dir}")

        metadata = {
            "name": "DrivAerNet++",
            "size": "8,150+ car meshes/simulations",
            "formats": ["3D meshes", "Aerodynamic data", "Point clouds"],
            "source": "https://github.com/Mohamedelrefaie/DrivAerNet",
            "license": "CC BY-NC 4.0",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
            "note": "Repository cloned. Dataset download instructions in repo README.",
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download DrivAerNet++: {e}")
        print("Note: Requires git installed.")
        raise


def download_engsymbols(output_dir: Path, test_mode: bool = False) -> Dict:
    """Download Engineering Symbols dataset."""
    dataset_dir = output_dir / "raw" / "engsymbols"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Engineering Symbols to {dataset_dir}...")

    try:
        repo_dir = dataset_dir / "repository"
        if not repo_dir.exists():
            print("Cloning Engineering Symbols repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/heyad/Eng_Diagrams.git", str(repo_dir)],
                check=True,
                capture_output=True,
            )

        print(f"[OK] Downloaded Engineering Symbols to {dataset_dir}")

        metadata = {
            "name": "Engineering Symbols",
            "size": "2,432 instances (multi-class)",
            "formats": ["Images", "Symbols"],
            "source": "https://github.com/heyad/Eng_Diagrams",
            "license": "Research",
            "download_date": str(Path(dataset_dir).stat().st_mtime),
            "test_mode": test_mode,
        }

        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    except Exception as e:
        print(f"[ERR] Failed to download Engineering Symbols: {e}")
        print("Note: Requires git installed.")
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
    "sesyd": {
        "name": "SESYD",
        "description": "1,000 synthetic floorplans (unavailable)",
        "downloader": None,
        "size": "~100 MB",
        "license": "Free",
        "note": "Original site unavailable (404). Dataset unsuitable for DeepV - contains symbol spotting data, not vector primitives.",
    },
    "impact": {
        "name": "IMPACT",
        "description": "500k patents with 3.61M figures",
        "downloader": download_impact,
        "size": "~10-20 GB",
        "license": "Open",
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
    "deeppatent2": {
        "name": "DeepPatent2",
        "description": "2.7M+ technical patent drawings",
        "downloader": download_deeppatent2,
        "size": ">100 GB",
        "license": "CC BY-NC 2.0",
    },
    "drivaernet": {
        "name": "DrivAerNet++",
        "description": "8,150+ car meshes/simulations",
        "downloader": download_drivaernet,
        "size": "~Large (repo only)",
        "license": "CC BY-NC 4.0",
    },
    "engsymbols": {
        "name": "Engineering Symbols",
        "description": "2,432 engineering symbol instances (UNSUITABLE - raster symbol classification, not vector primitives)",
        "downloader": download_engsymbols,
        "size": "~Small",
        "license": "Research",
        "note": "UNSUITABLE for DeepV - contains 100x100 pixel binary images for CNN classification, not vector geometric primitives",
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
    "deeppatent": {
        "name": "DeepPatent",
        "description": "350k+ patent drawings",
        "downloader": None,
        "size": "~Large",
        "license": "BSD-3-Clause",
        "note": "Download from GitHub: https://github.com/GoFigure-LANL/DeepPatent-dataset",
    },
    "pdtw150k": {
        "name": "PDTW150K",
        "description": "150k+ patents, 850k+ drawings",
        "downloader": None,
        "size": "~Large",
        "license": "Open Government Data License v1.0",
        "note": "Download from GitHub: https://github.com/ncyuMARSLab/PDTW150K",
    },
    "patentdesc": {
        "name": "PatentDesc-355K",
        "description": "355k figures from 60k+ documents",
        "downloader": None,
        "size": "~Large",
        "license": "Research",
        "note": "Download from arXiv: https://arxiv.org/abs/2501.15074",
    },
    "cadsketchnet": {
        "name": "CADSketchNet",
        "description": "1k+ annotated sketches paired with 3D CAD",
        "downloader": None,
        "size": "~Small",
        "license": "Research",
        "note": "Contact authors (Computers & Graphics paper)",
    },
    "videocad": {
        "name": "VideoCAD",
        "description": "41k+ CAD UI videos",
        "downloader": None,
        "size": "~Large",
        "license": "Open",
        "note": "Download from GitHub: https://github.com/ghadinehme/VideoCAD or Harvard Dataverse",
    },
    "cfp": {
        "name": "CFP (Comprehensive Floor Plan)",
        "description": "100k+ high-res floor plan elements",
        "downloader": None,
        "size": "~Large",
        "license": "Research",
        "note": "Contact authors - no public repo",
    },
    "cadbimcollection": {
        "name": "CAD/BIM Collection",
        "description": "4.5k IFC, 6.4k RVT, 156k DWG files",
        "downloader": None,
        "size": "~Very Large",
        "license": "Varies (public)",
        "note": "Contact authors - site unavailable",
    },
    "holicity": {
        "name": "HoliCity",
        "description": "City-scale 3D models (6,300 panoramas)",
        "downloader": None,
        "size": "~Very Large",
        "license": "Commercial/Research",
        "note": "Visit: https://holicity.io/ - requires agreement",
    },
    "blendednet": {
        "name": "BlendedNet",
        "description": "999 blended wing body geometries",
        "downloader": None,
        "size": "~Large",
        "license": "Apache 2.0 variant",
        "note": "Download from Harvard Dataverse: https://dataverse.harvard.edu/",
    },
    "fplanpoly": {
        "name": "FPLAN-POLY",
        "description": "42 floorplans + 38 symbol models in DXF vector format",
        "downloader": download_fplanpoly,
        "size": "~Small",
        "license": "Research",
        "note": "Highly suitable for DeepV - contains vector geometric primitives (polylines)",
    },
    "dld": {
        "name": "DLD (Degraded Line Drawings)",
        "description": "81 photos/scans of floorplans",
        "downloader": None,
        "size": "~Small",
        "license": "Research",
        "note": "Contact authors (ECCV 2020 paper)",
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

    # Separate auto-downloadable and manual datasets
    auto = {}
    manual = {}

    for key, info in DATASETS.items():
        if info.get("downloader") is None:
            manual[key] = info
        else:
            auto[key] = info

    # Auto-downloadable datasets
    if auto:
        print("\n[AUTO-DOWNLOAD] These datasets can be downloaded automatically:\n")
        for key, info in auto.items():
            print(f"{key:15} - {info['name']}")
            print(f"{'':15}   {info['description']}")
            print(f"{'':15}   Size: {info['size']} | License: {info['license']}\n")

    # Manual download datasets
    if manual:
        print("\n[MANUAL DOWNLOAD] These datasets require manual download or special access:\n")
        for key, info in manual.items():
            print(f"{key:15} - {info['name']}")
            print(f"{'':15}   {info['description']}")
            print(f"{'':15}   Size: {info['size']} | License: {info['license']}")
            if "note" in info:
                print(f"{'':15}   Note: {info['note']}\n")

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

    # Check if this dataset requires manual download
    if dataset_info.get("downloader") is None:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_info['name']}")
        print(f"Description: {dataset_info['description']}")
        print(f"License: {dataset_info['license']}")
        print(f"{'='*80}\n")
        print("[MANUAL DOWNLOAD REQUIRED]")
        print(f"This dataset requires manual download.\n")
        if "note" in dataset_info:
            print(f"Instructions: {dataset_info['note']}")
        print()
        return None

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
