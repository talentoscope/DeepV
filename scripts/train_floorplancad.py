#!/usr/bin/env python3
"""
Train vectorization model on FloorPlanCAD dataset using custom train/val/test splits.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import svgpathtools
from tqdm import tqdm

# Add current directory to path for local imports
sys.path.append(".")

from vectorization import load_model
from util_files.data.vectordata.common import sample_primitive_representation


class FloorPlanCADDataset(Dataset):
    """Dataset for FloorPlanCAD with raster and vector data."""

    def __init__(self, split_file, raster_dir, vector_dir, transform=None):
        """
        Args:
            split_file: Path to text file containing filenames (one per line)
            raster_dir: Directory containing PNG files
            vector_dir: Directory containing SVG files
            transform: Optional transform to be applied to images
        """
        self.raster_dir = Path(raster_dir)
        self.vector_dir = Path(vector_dir)
        self.transform = transform

        # Read split file with proper encoding
        with open(split_file, 'r', encoding='utf-8') as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.filenames)} files from {split_file}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # Load raster image
        raster_path = self.raster_dir / filename
        try:
            image = Image.open(raster_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load raster image {raster_path}: {e}")
            # Fallback to placeholder
            image = Image.new('RGB', (256, 256), color='white')
        
        if self.transform:
            image = self.transform(image)

        # Load and parse SVG file
        vector_filename = filename.replace('.png', '.svg')
        vector_path = self.vector_dir / vector_filename
        try:
            # Parse SVG manually to extract path d attributes
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(vector_path))
            root = tree.getroot()
            
            paths = []
            attributes = []
            for elem in root.iter():
                if elem.tag.endswith('path') and 'd' in elem.attrib:
                    d = elem.attrib['d']
                    try:
                        path = svgpathtools.parse_path(d)
                        paths.append(path)
                        # Extract attributes
                        attr_dict = {}
                        for k, v in elem.attrib.items():
                            if k != 'd':
                                attr_dict[k] = v
                        attributes.append(attr_dict)
                    except Exception:
                        continue  # Skip invalid paths
            
            if not paths:
                raise ValueError("No valid paths found")
            
            # Convert attributes to the format expected by sample_primitive_representation
            attribute_dicts = []
            for attr in attributes:
                # Ensure stroke-width is present
                if 'stroke-width' not in attr:
                    attr['stroke-width'] = '1'  # Default stroke width
                attribute_dicts.append(attr)
            
            # Sample primitives from the SVG
            lines, arcs, beziers = sample_primitive_representation(
                paths, 
                attribute_dicts, 
                max_lines_n=10,  # Limit to 10 primitives for now
                max_beziers_n=10,
                sample_primitives_randomly=True
            )
            
            # Combine all primitives
            all_primitives = lines + beziers
            
            # Convert to tensor format (10 primitives × 6 features)
            target = torch.zeros(10, 6)
            for i, primitive in enumerate(all_primitives[:10]):
                repr_data = primitive.to_repr()
                # Take first 6 features, padding with zeros if shorter
                for j in range(min(len(repr_data), 6)):
                    target[i, j] = repr_data[j]
            
        except Exception as e:
            print(f"Warning: Failed to parse SVG {vector_filename}: {e}")
            # Fallback to placeholder
            target = torch.zeros(10, 6)

        return image, target


def create_data_loaders(split_dir, raster_dir, vector_dir, batch_size=8):
    """Create train/val/test data loaders."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = FloorPlanCADDataset(
        split_file=os.path.join(split_dir, 'train.txt'),
        raster_dir=raster_dir,
        vector_dir=vector_dir,
        transform=transform
    )

    val_dataset = FloorPlanCADDataset(
        split_file=os.path.join(split_dir, 'val.txt'),
        raster_dir=raster_dir,
        vector_dir=vector_dir,
        transform=transform
    )

    test_dataset = FloorPlanCADDataset(
        split_file=os.path.join(split_dir, 'test.txt'),
        raster_dir=raster_dir,
        vector_dir=vector_dir,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', model_output_count=10):
    """Train the model."""

    model = model.to(device)
    criterion = nn.MSELoss()  # Placeholder loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        
        print("Training phase...")
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", unit="batch")
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            # Check if model supports variable length output
            if hasattr(model.hidden, 'max_primitives'):
                # Variable length model
                outputs = model(images)
            else:
                # Fixed length model
                outputs = model(images, model_output_count)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar with current loss
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        print("Validation phase...")
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", unit="batch")
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_pbar):
                images, targets = images.to(device), targets.to(device)
                if hasattr(model.hidden, 'max_primitives'):
                    outputs = model(images)
                else:
                    outputs = model(images, model_output_count)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Update progress bar with current loss
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")


def main():
    print("Starting training script...")
    parser = argparse.ArgumentParser(description='Train vectorization model on FloorPlanCAD')
    print("Setting up argument parser...")
    parser.add_argument('--split-dir', type=str, default='data/splits/floorplancad',
                        help='Directory containing train.txt, val.txt, test.txt')
    parser.add_argument('--raster-dir', type=str, default='data/raster/floorplancad',
                        help='Directory containing raster PNG files')
    parser.add_argument('--vector-dir', type=str, default='data/vector/floorplancad',
                        help='Directory containing vector SVG files')
    parser.add_argument('--model-spec', type=str, required=True,
                        help='Path to model specification JSON file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint file (optional, for resuming training)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--model-output-count', type=int, default=10,
                        help='Number of primitives to output (for fixed-length models)')

    print("Parsing arguments...")
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")

    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Extra CUDA debug info to help diagnose why CPU is selected
    try:
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            try:
                cur = torch.cuda.current_device()
                print(f"Current CUDA device index: {cur}")
                print(f"CUDA device name: {torch.cuda.get_device_name(cur)}")
            except Exception as e:
                print(f"Error querying current CUDA device: {e}")
    except Exception as e:
        print(f"CUDA debug info error: {e}")

    print("Creating data loaders...")
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args.split_dir, args.raster_dir, args.vector_dir, args.batch_size
    )
    print("Data loaders created successfully!")

    # Load model
    print(f"Loading model from {args.model_spec}")
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = load_model(args.model_spec, checkpoint)
    else:
        print("Initializing model with random weights (no checkpoint provided)")
        model = load_model(args.model_spec)
    
    print(f"Model loaded successfully. Moving to {device}...")
    model = model.to(device)
    print("Model ready for training!")

    # Train model
    print("Starting training...")
    train_model(model, train_loader, val_loader, args.epochs, device, args.model_output_count)

    print("Training completed!")


if __name__ == "__main__":
    main()