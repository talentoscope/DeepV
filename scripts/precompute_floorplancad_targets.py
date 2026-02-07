#!/usr/bin/env python3
"""
Pre-compute and cache FloorPlanCAD primitive targets from SVG files.
This eliminates expensive SVG parsing during training.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import svgpathtools

sys.path.append(".")

from util_files.data.vectordata.common import sample_primitive_representation


def precompute_targets(split_file, raster_dir, vector_dir, cache_dir, max_primitives=20):
    """Pre-compute primitive targets from SVG and cache them."""
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Read split file
    with open(split_file, 'r', encoding='utf-8') as f:
        filenames = [line.strip() for line in f if line.strip()]
    
    print(f"Pre-computing targets for {len(filenames)} files...")
    
    cache_info = {
        'success': 0,
        'errors': 0,
        'error_files': []
    }
    
    for filename in tqdm(filenames, desc="Pre-computing targets"):
        # Cache filename (same as raster but with .pt extension)
        cache_path = os.path.join(cache_dir, filename.replace('.png', '.pt'))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Skip if already cached
        if os.path.exists(cache_path):
            continue
        
        # Load SVG
        vector_filename = filename.replace('.png', '.svg')
        vector_path = os.path.join(vector_dir, vector_filename)
        
        try:
            # Parse SVG
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
                        continue
            
            if not paths:
                raise ValueError("No valid paths found")
            
            # Ensure stroke-width is present
            for attr in attributes:
                if 'stroke-width' not in attr:
                    attr['stroke-width'] = '1'
            
            # Sample primitives
            lines, arcs, beziers = sample_primitive_representation(
                paths,
                attributes,
                max_lines_n=max_primitives,
                max_beziers_n=max_primitives,
                sample_primitives_randomly=False,  # Don't randomize during precompute
            )
            
            all_primitives = lines + beziers
            primitive_count = min(len(all_primitives), max_primitives)
            
            # Create target tensor
            target = torch.zeros(max_primitives, 6)
            for i, primitive in enumerate(all_primitives[:max_primitives]):
                repr_data = primitive.to_repr()
                for j in range(min(len(repr_data), 6)):
                    target[i, j] = repr_data[j]
            
            # Cache as tuple (target, primitive_count)
            cache_data = {
                'target': target,
                'count': primitive_count,
            }
            torch.save(cache_data, cache_path)
            cache_info['success'] += 1
            
        except Exception as e:
            cache_info['errors'] += 1
            cache_info['error_files'].append((filename, str(e)))
    
    print(f"\nPre-computation complete!")
    print(f"  Success: {cache_info['success']}")
    print(f"  Errors: {cache_info['errors']}")
    if cache_info['error_files']:
        print(f"  First few errors:")
        for fname, err in cache_info['error_files'][:3]:
            print(f"    {fname}: {err}")
    
    return cache_dir


def main():
    parser = argparse.ArgumentParser(description='Pre-compute FloorPlanCAD primitive targets')
    parser.add_argument('--split-dir', type=str, default='data/splits/floorplancad',
                        help='Directory containing train.txt, val.txt, test.txt')
    parser.add_argument('--raster-dir', type=str, default='data/raster/floorplancad',
                        help='Directory containing raster PNG files')
    parser.add_argument('--vector-dir', type=str, default='data/vector/floorplancad',
                        help='Directory containing vector SVG files')
    parser.add_argument('--cache-dir', type=str, default='data/cache/floorplancad',
                        help='Directory to store cached primitives')
    parser.add_argument('--max-primitives', type=int, default=20,
                        help='Maximum number of primitives per image')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Splits to pre-compute (default: train val test)')
    
    args = parser.parse_args()
    
    for split in args.splits:
        split_file = os.path.join(args.split_dir, f'{split}.txt')
        split_cache_dir = os.path.join(args.cache_dir, split)
        
        print(f"\nProcessing {split} split...")
        precompute_targets(
            split_file,
            args.raster_dir,
            args.vector_dir,
            split_cache_dir,
            args.max_primitives
        )


if __name__ == '__main__':
    main()
