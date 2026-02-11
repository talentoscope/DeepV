"""
Unified pipeline interface for DeepV.

Consolidates separate line/curve refinement and merging pipelines into unified interfaces.
Provides a clean API for running the complete DeepV pipeline:
vectorization -> refinement -> merging.

This module serves as the main entry point for pipeline operations, abstracting away
the differences between line and curve processing while maintaining backward compatibility.
"""

import sys
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, ".")


class UnifiedPipeline:
    """Unified interface for DeepV pipeline operations.

    Provides a single entry point for running refinement and merging operations
    regardless of primitive type (line or curve). Handles the routing to appropriate
    specialized implementations while maintaining a consistent API.

    Currently supports line primitives with curve support planned for future releases.

    Attributes:
        primitive_type: The type of primitives this pipeline handles ("line" or "curve")
    """

    def __init__(self, primitive_type: str = "line"):
        """
        Initialize unified pipeline.

        Args:
            primitive_type: Type of primitives to process ("line" or "curve").
                          Currently only "line" is fully supported.

        Raises:
            ValueError: If primitive_type is not supported.
        """
        self.primitive_type = primitive_type.lower()
        if self.primitive_type not in ["line", "curve"]:
            raise ValueError(
                f"Unsupported primitive type: {primitive_type}. " "Supported types are 'line' and 'curve'."
            )

    def run_refinement(
        self, patches_rgb: np.ndarray, patches_vector: torch.Tensor, device: torch.device, options: Any
    ) -> torch.Tensor:
        """
        Run refinement for the specified primitive type.

        Applies differentiable optimization to improve the accuracy of predicted
        vector primitives by minimizing the difference between rendered primitives
        and the target raster image.

        Args:
            patches_rgb: RGB image patches as numpy array
            patches_vector: Initial vector predictions as torch tensor
            device: Computation device (CPU/GPU)
            options: Configuration options object with refinement parameters

        Returns:
            Refined vector primitives as torch tensor

        Raises:
            NotImplementedError: If curve refinement is requested (not yet implemented)
            ValueError: If primitive type is unknown
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

        Consolidates overlapping and redundant primitives from individual patches
        into a coherent final vector representation.

        Args:
            vector_data: Vector data from refinement stage
            patches_offsets: Offset coordinates for each patch
            input_rgb: Original RGB image patches
            cleaned_image: Preprocessed/cleaned input image
            options: Configuration options object

        Returns:
            Tuple of (merged_primitives, rendered_image) where:
            - merged_primitives: Final consolidated vector data
            - rendered_image: Raster rendering of merged primitives

        Raises:
            NotImplementedError: If curve merging is requested (not yet implemented)
            ValueError: If primitive type is unknown
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

        Executes the entire DeepV pipeline from input image to final vector output,
        including patch splitting, vector estimation, refinement optimization, and
        primitive merging.

        Args:
            image_tensor: Input image as torch tensor (shape: [1, C, H, W])
            model: Trained vectorization model for primitive prediction
            device: Computation device (CPU/GPU)
            options: Configuration options object with pipeline parameters

        Returns:
            Final merged vector primitives ready for export/CAD conversion

        Note:
            This method orchestrates the complete pipeline but currently only
            supports line primitives. Curve pipeline integration is planned.
        """
        from run_pipeline import split_to_patches, vector_estimation

        try:
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

        except Exception as e:
            raise RuntimeError(f"Pipeline execution failed: {e}") from e


class PipelineFactory:
    """Factory for creating unified pipeline instances.

    Provides static methods for creating and managing pipeline instances,
    ensuring consistent configuration and supporting future extension
    to additional primitive types.
    """

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
