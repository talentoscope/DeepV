#!/usr/bin/env python3
"""
Train vectorization model on FloorPlanCAD dataset using custom train/val/test splits.
"""

import argparse
import json
import os
import random
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
from datetime import datetime

# Add current directory to path for local imports
sys.path.append(".")

from vectorization import load_model
from util_files.data.vectordata.common import sample_primitive_representation


class FloorPlanCADDataset(Dataset):
    """Dataset for FloorPlanCAD with raster and vector data."""

    def __init__(self, split_file, raster_dir, vector_dir, transform=None, max_primitives=20, sample_primitives_randomly=True, cache_dir=None):
        """
        Args:
            split_file: Path to text file containing filenames (one per line)
            raster_dir: Directory containing PNG files
            vector_dir: Directory containing SVG files
            transform: Optional transform to be applied to images
            max_primitives: Maximum number of primitives per image
            sample_primitives_randomly: Whether to randomly permute primitives
            cache_dir: If provided, load pre-computed targets from cache instead of parsing SVG
        """
        self.raster_dir = Path(raster_dir)
        self.vector_dir = Path(vector_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform
        self.max_primitives = max_primitives
        self.sample_primitives_randomly = sample_primitives_randomly

        # Read split file with proper encoding
        with open(split_file, 'r', encoding='utf-8') as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.filenames)} files from {split_file}")
        if self.cache_dir:
            print(f"Using cached targets from {self.cache_dir}")

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
            image = Image.new('RGB', (256, 256), color='white')
        
        if self.transform:
            image = self.transform(image)

        # Load target from cache (fast!) or parse SVG (slow)
        try:
            if self.cache_dir:
                # Load from pre-computed cache (much faster!)
                cache_path = self.cache_dir / filename.replace('.png', '.pt')
                try:
                    cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
                except TypeError:
                    cache_data = torch.load(cache_path, map_location="cpu")
                target = cache_data['target'].clone()
                primitive_count = int(cache_data['count'])
                # Optionally randomize primitive order during training
                if self.sample_primitives_randomly:
                    perm = torch.randperm(self.max_primitives)
                    target = target[perm]
            else:
                # Parse SVG (slow, fallback for testing without cache)
                vector_filename = filename.replace('.png', '.svg')
                vector_path = self.vector_dir / vector_filename
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
                            attr_dict = {}
                            for k, v in elem.attrib.items():
                                if k != 'd':
                                    attr_dict[k] = v
                            attributes.append(attr_dict)
                        except Exception:
                            continue
                
                if not paths:
                    raise ValueError("No valid paths found")
                
                for attr in attributes:
                    if 'stroke-width' not in attr:
                        attr['stroke-width'] = '1'
                
                lines, arcs, beziers = sample_primitive_representation(
                    paths, attributes,
                    max_lines_n=self.max_primitives,
                    max_beziers_n=self.max_primitives,
                    sample_primitives_randomly=self.sample_primitives_randomly,
                )
                
                all_primitives = lines + beziers
                if self.sample_primitives_randomly:
                    random.shuffle(all_primitives)
                
                primitive_count = min(len(all_primitives), self.max_primitives)
                target = torch.zeros(self.max_primitives, 6)
                for i, primitive in enumerate(all_primitives[:self.max_primitives]):
                    repr_data = primitive.to_repr()
                    for j in range(min(len(repr_data), 6)):
                        target[i, j] = repr_data[j]
            
        except Exception as e:
            print(f"Warning: Failed to load targets for {filename}: {e}")
            target = torch.zeros(self.max_primitives, 6)
            primitive_count = 0

        return image, target, primitive_count


def create_data_loaders(
    split_dir,
    raster_dir,
    vector_dir,
    batch_size=8,
    max_primitives=20,
    num_workers=4,
    cache_dir=None,
    resize_to=None,
):
    """Create train/val/test data loaders."""

    transform_steps = []
    if resize_to is not None:
        transform_steps.append(transforms.Resize((resize_to, resize_to)))
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform_steps)

    # Create datasets
    train_dataset = FloorPlanCADDataset(
        split_file=os.path.join(split_dir, 'train.txt'),
        raster_dir=raster_dir,
        vector_dir=vector_dir,
        transform=transform,
        max_primitives=max_primitives,
        sample_primitives_randomly=True,
        cache_dir=os.path.join(cache_dir, 'train') if cache_dir else None,
    )

    val_dataset = FloorPlanCADDataset(
        split_file=os.path.join(split_dir, 'val.txt'),
        raster_dir=raster_dir,
        vector_dir=vector_dir,
        transform=transform,
        max_primitives=max_primitives,
        sample_primitives_randomly=False,
        cache_dir=os.path.join(cache_dir, 'val') if cache_dir else None,
    )

    test_dataset = FloorPlanCADDataset(
        split_file=os.path.join(split_dir, 'test.txt'),
        raster_dir=raster_dir,
        vector_dir=vector_dir,
        transform=transform,
        max_primitives=max_primitives,
        sample_primitives_randomly=False,
        cache_dir=os.path.join(cache_dir, 'test') if cache_dir else None,
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    device='cuda',
    model_output_count=10,
    max_primitives=20,
    output_dir='logs/training/default',
    use_amp=False,
    grad_accum_steps=1,
):
    """Train the model with checkpointing and logging."""

    model = model.to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up loss function based on model type
    if hasattr(model.hidden, 'count_predictor'):
        from util_files.loss_functions.supervised import non_autoregressive_vectran_loss
        criterion = non_autoregressive_vectran_loss
    elif hasattr(model.hidden, 'max_primitives'):
        # Variable length autoregressive model
        from util_files.loss_functions.supervised import variable_vectran_loss
        criterion = variable_vectran_loss
    else:
        # Fixed length model
        criterion = nn.MSELoss()  # Fallback
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Output directory: {output_dir}")
    
    # Tracking for checkpoints
    best_val_loss = float('inf')
    metrics_log = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        train_pred_count = 0.0
        train_true_count = 0.0
        
        print("Training phase...")
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", unit="batch")
        for batch_idx, (images, targets, primitive_counts) in enumerate(train_pbar):
            images, targets = images.to(device), targets.to(device)
            primitive_counts = primitive_counts.to(device).long().clamp(min=0, max=max_primitives)

            if batch_idx % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                # Check if model supports variable length output
                if hasattr(model.hidden, 'count_predictor'):
                    outputs = model(images)
                    loss = criterion(outputs[0], outputs[1], targets, primitive_counts)
                    train_pred_count += outputs[1].argmax(dim=1).float().sum().item()
                    train_true_count += primitive_counts.float().sum().item()
                elif hasattr(model.hidden, 'max_primitives'):
                    outputs = model(images)
                    mask = torch.arange(max_primitives, device=primitive_counts.device)[None, :] < primitive_counts[:, None]
                    loss = criterion(outputs, targets, mask=mask)
                else:
                    outputs = model(images, model_output_count)
                    loss = criterion(outputs, targets)

            loss = loss / float(grad_accum_steps)
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
            
            train_loss += loss.item()
            
            # Update progress bar with current loss
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        print("Validation phase...")
        model.eval()
        val_loss = 0.0
        val_pred_count = 0.0
        val_true_count = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", unit="batch")
        with torch.no_grad():
            for batch_idx, (images, targets, primitive_counts) in enumerate(val_pbar):
                images, targets = images.to(device), targets.to(device)
                primitive_counts = primitive_counts.to(device).long().clamp(min=0, max=max_primitives)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    if hasattr(model.hidden, 'count_predictor'):
                        outputs = model(images)
                        loss = criterion(outputs[0], outputs[1], targets, primitive_counts)
                        val_pred_count += outputs[1].argmax(dim=1).float().sum().item()
                        val_true_count += primitive_counts.float().sum().item()
                    elif hasattr(model.hidden, 'max_primitives'):
                        outputs = model(images)
                        mask = torch.arange(max_primitives, device=primitive_counts.device)[None, :] < primitive_counts[:, None]
                        loss = criterion(outputs, targets, mask=mask)
                    else:
                        outputs = model(images, model_output_count)
                        loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Update progress bar with current loss
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        # Log metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss_avg,
            'val_loss': val_loss_avg,
        }
        
        if hasattr(model.hidden, 'count_predictor'):
            train_count_avg = train_pred_count / max(len(train_loader.dataset), 1)
            train_true_avg = train_true_count / max(len(train_loader.dataset), 1)
            val_count_avg = val_pred_count / max(len(val_loader.dataset), 1)
            val_true_avg = val_true_count / max(len(val_loader.dataset), 1)
            epoch_metrics.update({
                'train_pred_count': train_count_avg,
                'train_true_count': train_true_avg,
                'val_pred_count': val_count_avg,
                'val_true_count': val_true_avg,
                'train_count_ratio': train_count_avg / max(train_true_avg, 0.001),
                'val_count_ratio': val_count_avg / max(val_true_avg, 0.001),
            })
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, "
                f"Train Count: {train_count_avg:.2f}/{train_true_avg:.2f}, Val Count: {val_count_avg:.2f}/{val_true_avg:.2f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}"
            )
        
        metrics_log.append(epoch_metrics)
        
        # Save best model checkpoint
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_avg,
                'train_loss': train_loss_avg,
                'epoch_metrics': epoch_metrics,
            }, checkpoint_path)
            print(f"✓ Saved best model checkpoint to {checkpoint_path}")
    
    # Save metrics log
    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"✓ Training metrics saved to {metrics_path}")
    
    print("Training completed!")


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
    parser.add_argument('--max-primitives', type=int, default=20,
                        help='Maximum number of primitives for variable-length models')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader worker count')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints and logs (default: logs/training/floorplancad_nonar_TIMESTAMP)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Pre-computed primitive cache directory (if not set, will parse SVG which is slow)')
    parser.add_argument('--resize', type=int, default=None,
                        help='Resize images to square size (e.g., 512) to reduce memory usage')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training (reduces memory usage)')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Accumulate gradients across N steps to reduce memory usage')

    print("Parsing arguments...")
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'logs/training/floorplancad_nonar_{timestamp}'
    
    print(f"Output directory: {args.output_dir}")

    # Set device
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        torch.cuda.set_device(args.gpu)
    else:
        device = 'cpu'
        if args.gpu is not None:
            print("Warning: CUDA unavailable. Falling back to CPU training.")
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
    # Normalize cache_dir: if user passed a parent cache folder, try to find the
    # dataset-specific subfolder (e.g. data/cache/floorplancad) that actually
    # contains train/val/test subfolders. This avoids warnings when cache is
    # organized under a dataset name.
    if args.cache_dir:
        cache_train_path = os.path.join(args.cache_dir, 'train')
        if not os.path.isdir(cache_train_path):
            # search for a child directory that contains train/val/test
            try:
                children = [d for d in os.listdir(args.cache_dir) if os.path.isdir(os.path.join(args.cache_dir, d))]
            except Exception:
                children = []
            found = None
            for c in children:
                if os.path.isdir(os.path.join(args.cache_dir, c, 'train')):
                    found = os.path.join(args.cache_dir, c)
                    break
            if found:
                print(f"Notice: remapping cache dir from {args.cache_dir} to {found} (contains train/val/test)")
                args.cache_dir = found
            else:
                print(f"Warning: cache dir {args.cache_dir} does not contain a 'train' subfolder; proceeding anyway")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args.split_dir,
        args.raster_dir,
        args.vector_dir,
        args.batch_size,
        max_primitives=args.max_primitives,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        resize_to=args.resize,
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
    train_model(
        model,
        train_loader,
        val_loader,
        args.epochs,
        device,
        args.model_output_count,
        max_primitives=args.max_primitives,
        output_dir=args.output_dir,
        use_amp=args.amp,
        grad_accum_steps=max(args.grad_accum_steps, 1),
    )

    print("Training completed!")


if __name__ == "__main__":
    main()