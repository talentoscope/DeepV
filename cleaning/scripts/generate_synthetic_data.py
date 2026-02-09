"""
Synthetic data generation script for cleaning models.

This script generates synthetic training data for the cleaning pipeline
using the Synthetic data generation utility.
"""

import argparse
import sys
from typing import NoReturn

sys.path.append("../../")
from tqdm import tqdm

from cleaning.utils.synthetic_data_generation import Synthetic


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic data for cleaning models")

    parser.add_argument(
        "--img_path", type=str, default="../../dataset/Cleaning/Synthetic/train",
        help="Path to save generated data"
    )
    parser.add_argument("--data_count", type=int, default=7,
                       help="Total number of data samples to generate")
    parser.add_argument("--data_count_start", type=int, default=0,
                       help="Starting index for data generation")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """Generate synthetic data samples."""
    syn = Synthetic()
    for it in tqdm(range(args.data_count_start, args.data_count)):
        syn.get_image(img_path=args.img_path, name=str(it))


if __name__ == "__main__":
    args = parse_args()
    main(args)
