#!/usr/bin/env python3
"""Quick debug: load one batch, do one forward, save checkpoint."""
import os
import sys
from pathlib import Path
import torch

sys.path.append('.')

from scripts.train_floorplancad import create_data_loaders
from vectorization import load_model

def main():
    out_dir = Path('logs/training/debug_single_step')
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_data_loaders(
        'data/splits/floorplancad',
        'data/raster/floorplancad',
        'data/vector/floorplancad',
        batch_size=1,
        max_primitives=20,
        num_workers=0,
        cache_dir='data/cache',
        resize_to=128,
    )

    print(f"Loaded train loader with {len(train_loader.dataset)} samples")

    spec = 'vectorization/models/specs/non_autoregressive_resnet18_blocks1_bn_64__c2h__trans_heads4_feat256_blocks8_ffmaps512__h2o__out6.json'
    model = load_model(spec)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    batch = next(iter(train_loader))
    images, targets, counts = batch
    images = images.to(device)
    try:
        with torch.no_grad():
            outputs = model(images)
        ckpt = {
            'model_state_dict': model.state_dict(),
            'sample_output_shape': str(outputs[0].shape) if isinstance(outputs, (list, tuple)) else str(outputs.shape),
        }
        torch.save(ckpt, out_dir / 'single_step_checkpoint.pth')
        print('Saved checkpoint to', out_dir / 'single_step_checkpoint.pth')
    except Exception as e:
        print('Error during forward pass:', e)

if __name__ == '__main__':
    main()
