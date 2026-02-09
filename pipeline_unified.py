"""
Unified pipeline interface for DeepV.

Consolidates separate line/curve refinement and merging pipelines into unified interfaces.
"""

import sys
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, ".")


class UnifiedPipeline:
    """Unified interface for DeepV pipeline operations."""

    def __init__(self, primitive_type: str = "line"):
        """
        Initialize unified pipeline.

        Args:
            primitive_type: Type of primitives to process ("line" or "curve")
        """
        self.primitive_type = primitive_type.lower()
        if self.primitive_type not in ["line", "curve"]:
            raise ValueError(f"Unsupported primitive type: {primitive_type}")

    def run_refinement(
        self, patches_rgb: np.ndarray, patches_vector: torch.Tensor, device: torch.device, options: Any
    ) -> torch.Tensor:
        """
        Run refinement for the specified primitive type.

        Args:
            patches_rgb: RGB patches
            patches_vector: Initial vector predictions
            device: Computation device
            options: Configuration options

        Returns:
            Refined vector primitives
        """
        if self.primitive_type == "line":
            from refinement.our_refinement.refinement_for_lines import (
                render_optimization_hard,
            )

            return render_optimization_hard(patches_rgb, patches_vector, device, options, options.sample_name)
        elif self.primitive_type == "curve":
            # Curve refinement has a different interface - would need adaptation
            raise NotImplementedError("Curve refinement unification pending")
        else:
            raise ValueError(f"Unknown primitive type: {self.primitive_type}")

    def run_merging(
        self,
        vector_data: Union[torch.Tensor, Dict],
        patches_offsets: np.ndarray,
        input_rgb: np.ndarray,
        cleaned_image: torch.Tensor,
        options: Any,
    ) -> Tuple[Any, Any]:
        """
        Run merging for the specified primitive type.

        Args:
            vector_data: Vector data from refinement
            patches_offsets: Patch offset coordinates
            input_rgb: Original RGB image patches
            cleaned_image: Cleaned input image
            options: Configuration options

        Returns:
            Tuple of (merged_primitives, rendered_image)
        """
        if self.primitive_type == "line":
            from merging.merging_for_lines import postprocess

            return postprocess(vector_data, patches_offsets, input_rgb, cleaned_image, 0, options)
        elif self.primitive_type == "curve":
            # Curve merging has different interface - would need adaptation
            raise NotImplementedError("Curve merging unification pending")
        else:
            raise ValueError(f"Unknown primitive type: {self.primitive_type}")

    def run_full_pipeline(
        self, image_tensor: torch.Tensor, model: torch.nn.Module, device: torch.device, options: Any
    ) -> Any:
        """
        Run the complete pipeline: preprocessing -> vectorization -> refinement -> merging.

        Args:
            image_tensor: Input image tensor
            model: Trained vectorization model
            device: Computation device
            options: Configuration options

        Returns:
            Final merged vector primitives
        """
        from run_pipeline import split_to_patches, vector_estimation

        # Preprocessing and patching
        patches_rgb, patches_offsets, input_rgb = split_to_patches(
            image_tensor.cpu().numpy()[0] * 255, 64, options.overlap
        )

        # Vector estimation
        patches_vector = vector_estimation(patches_rgb, model, device, 0, options)

        # Refinement
        refined_vector = self.run_refinement(patches_rgb, patches_vector, device, options)

        # Merging
        merged_result, rendered_image = self.run_merging(
            refined_vector, patches_offsets, input_rgb, image_tensor, options
        )

        return merged_result


class PipelineFactory:
    """Factory for creating unified pipeline instances."""

    @staticmethod
    def create_pipeline(primitive_type: str) -> UnifiedPipeline:
        """
        Create a pipeline instance for the specified primitive type.

        Args:
            primitive_type: Type of primitives ("line" or "curve")

        Returns:
            Configured UnifiedPipeline instance
        """
        return UnifiedPipeline(primitive_type)

    @staticmethod
    def get_supported_types() -> list:
        """Get list of supported primitive types."""
        return ["line", "curve"]


# Convenience functions for backward compatibility
def create_line_pipeline() -> UnifiedPipeline:
    """Create a pipeline configured for line primitives."""
    return UnifiedPipeline("line")


def create_curve_pipeline() -> UnifiedPipeline:
    """Create a pipeline configured for curve primitives."""
    return UnifiedPipeline("curve")


# Example usage and testing
if __name__ == "__main__":
    # Test pipeline creation
    line_pipeline = create_line_pipeline()
    print(f"Created line pipeline: {line_pipeline.primitive_type}")

    curve_pipeline = create_curve_pipeline()
    print(f"Created curve pipeline: {curve_pipeline.primitive_type}")

    # Test factory
    factory = PipelineFactory()
    supported = factory.get_supported_types()
    print(f"Supported types: {supported}")

    pipeline_from_factory = factory.create_pipeline("line")
    print(f"Factory created pipeline: {pipeline_from_factory.primitive_type}")
