"""
Basic integration tests for refactored refinement optimization.

Tests that the refactored classes can be imported and instantiated.
"""

import os
import sys
import unittest

import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestRefinementIntegration(unittest.TestCase):
    """Basic integration test for refactored refinement components."""

    def test_classes_import(self):
        """Test that all refactored classes can be imported."""
        try:
            from refinement.our_refinement.optimization_classes import (
                LineOptimizationState,
                BatchProcessor,
                OptimizationLoop,
            )
            from refinement.our_refinement.refinement_for_lines import render_optimization_hard
        except ImportError as e:
            self.fail(f"Import failed: {e}")

    def test_line_optimization_state_basic(self):
        """Test basic LineOptimizationState functionality."""
        from refinement.our_refinement.optimization_classes import LineOptimizationState

        # Create simple line batch
        lines_batch = torch.tensor([[[10, 10, 20, 20, 1.0]]], dtype=torch.float32)
        device = torch.device("cpu")

        opt_state = LineOptimizationState(lines_batch, device)

        # Check basic attributes exist
        self.assertTrue(hasattr(opt_state, 'cx'))
        self.assertTrue(hasattr(opt_state, 'cy'))
        self.assertTrue(hasattr(opt_state, 'theta'))
        self.assertTrue(hasattr(opt_state, 'length'))
        self.assertTrue(hasattr(opt_state, 'width'))

        # Test conversion back
        reconstructed = opt_state.get_lines_batch()
        self.assertEqual(reconstructed.shape, lines_batch.shape)

    def test_batch_processor_basic(self):
        """Test basic BatchProcessor functionality."""
        from refinement.our_refinement.optimization_classes import BatchProcessor

        patches_rgb = torch.rand(2, 1, 64, 64)
        patches_vector = torch.rand(2, 5, 5)

        processor = BatchProcessor(patches_rgb, patches_vector)

        # Check basic attributes
        self.assertEqual(processor.patches_rgb.shape, patches_rgb.shape)
        self.assertEqual(processor.patches_vector.shape, patches_vector.shape)

    def test_render_optimization_hard_exists(self):
        """Test that render_optimization_hard function exists."""
        from refinement.our_refinement.refinement_for_lines import render_optimization_hard

        self.assertTrue(callable(render_optimization_hard))

    def test_modules_compile(self):
        """Test that refactored modules compile without syntax errors."""
        import py_compile

        # Test compilation
        try:
            py_compile.compile('refinement/our_refinement/optimization_classes.py', doraise=True)
            py_compile.compile('refinement/our_refinement/refinement_for_lines.py', doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Compilation failed: {e}")


if __name__ == '__main__':
    unittest.main()