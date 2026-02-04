#!/usr/bin/env python3
"""
Simple test script to demonstrate the benchmarking pipeline working.
Creates minimal test data and runs evaluation.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluation_suite import DatasetEvaluator

def create_test_data():
    """Create minimal test data for demonstration."""
    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp())

    # Create a simple test image (just zeros for now)
    image_path = test_dir / "test_image.png"
    # For simplicity, we'll just create a placeholder file
    with open(image_path, 'w') as f:
        f.write("placeholder image data")

    # Create ground truth in a simple format that might work
    # Let's try a simple dict format first
    ground_truth = {
        'primitives': [
            {'type': 'line', 'x1': 0, 'y1': 0, 'x2': 100, 'y2': 0},
            {'type': 'line', 'x1': 100, 'y1': 0, 'x2': 100, 'y2': 100},
        ]
    }

    # Create prediction (same as ground truth for perfect score)
    prediction = ground_truth.copy()

    return str(image_path), ground_truth, prediction

def test_evaluation():
    """Test the evaluation suite with minimal data."""
    print("Testing DeepV Evaluation Suite...")

    # Create test data
    image_path, ground_truth, prediction = create_test_data()

    # Create evaluator
    evaluator = DatasetEvaluator(
        dataset_name="test_dataset",
        model_name="test_model",
        output_dir="test_evaluation_output",
        metrics=['f1_score', 'iou_score']  # Start with just a couple metrics
    )

    print(f"Created evaluator for {evaluator.dataset_name}")

    # Try to evaluate sample
    try:
        results = evaluator.evaluate_sample(
            image_path=image_path,
            ground_truth=ground_truth,
            prediction=prediction,
            sample_id="test_001"
        )
        print("Sample evaluation completed!")
        print(f"Results: {results}")

        # Generate report
        evaluator._generate_report()
        print("Report generated successfully!")

        return True

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation()
    if success:
        print("\n✅ Evaluation test passed!")
    else:
        print("\n❌ Evaluation test failed!")
        sys.exit(1)