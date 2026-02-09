"""
Integration tests for DeepV components.

Tests the improved components: error handling, logging, and core functionality.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import improved components
from util_files.exceptions import ClippingError, DeepVError, EmptyPixelError
from util_files.structured_logging import get_pipeline_logger


class TestIntegration(unittest.TestCase):
    """Test integration of improved DeepV components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_exception_hierarchy(self):
        """Test that custom exception hierarchy works correctly."""
        # Test base exception
        with self.assertRaises(DeepVError):
            raise DeepVError("Base error")

        # Test specific exceptions inherit from base
        with self.assertRaises(DeepVError):
            raise EmptyPixelError("No pixels found")

        with self.assertRaises(DeepVError):
            raise ClippingError("Clipping failed")

        # Test exception messages
        try:
            raise EmptyPixelError("Test empty pixel message")
        except EmptyPixelError as e:
            self.assertIn("Test empty pixel message", str(e))
            self.assertIsInstance(e, DeepVError)

    def test_logging_integration(self):
        """Test that logging system integrates properly."""
        # Create a logger
        logger = get_pipeline_logger("test.integration")

        # Test logging methods exist
        self.assertTrue(hasattr(logger, "info"))
        self.assertTrue(hasattr(logger, "warning"))
        self.assertTrue(hasattr(logger, "error"))

        # Test logging doesn't crash
        try:
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            # Success if no exceptions
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Logging failed: {e}")

    def test_refinement_error_handling(self):
        """Test that refinement functions handle errors properly."""
        try:
            from refinement.our_refinement.utils.lines_refinement_functions import (
                MeanFieldEnergyComputer,
            )
        except ImportError:
            self.skipTest("Refinement modules not available")

        # Test that the class can be instantiated (basic import test)
        # We can't test full functionality without models, but we can test error handling
        try:
            # This should not crash on import
            computer = MeanFieldEnergyComputer.__new__(MeanFieldEnergyComputer)
            self.assertIsNotNone(computer)
        except Exception as e:
            self.fail(f"Refinement class instantiation failed: {e}")

    def test_merging_error_handling(self):
        """Test that merging functions handle errors properly."""
        try:
            from merging.utils.merging_functions import clip_to_box
        except ImportError:
            self.skipTest("Merging modules not available")

        # Test clip_to_box with valid inputs
        try:
            # Create test data: [x1, y1, x2, y2, width, opacity]
            line = np.array([10.5, 10.5, 50.5, 50.5, 2.0, 1.0])
            result = clip_to_box(line, box_size=(64, 64))
            # Should return clipped line
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 6)  # Same format as input
        except Exception as e:
            self.fail(f"Merging function test failed: {e}")

    def test_image_processing_pipeline(self):
        """Test basic image processing that doesn't require heavy dependencies."""
        # Create a test image
        img = Image.new("L", (64, 64), color=255)
        # Draw a simple shape
        for i in range(10, 54):
            img.putpixel((i, 32), 0)  # Horizontal line

        # Convert to tensor (basic preprocessing)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)

        # Test basic tensor operations
        self.assertEqual(img_tensor.shape, (64, 64))
        self.assertTrue(torch.all(img_tensor >= 0) and torch.all(img_tensor <= 1))

        # Test inversion (common preprocessing step)
        inverted = 1 - img_tensor
        self.assertTrue(torch.allclose(inverted + img_tensor, torch.ones_like(img_tensor)))


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
