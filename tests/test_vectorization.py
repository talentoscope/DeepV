import json
import os
import sys

import pytest

# Ensure repo root is on sys.path when running this test file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from vectorization import load_model


def test_load_model_from_spec():
    """Test loading a vectorization model from a spec file without checkpoint."""
    spec_path = "vectorization/models/specs/resnet18_blocks1_bn_64__c2h__trans_heads4_feat256_blocks8_ffmaps512__h2o__out512.json"

    # Load the model without checkpoint
    model = load_model(spec_path)

    # Verify it's a torch module
    assert isinstance(model, torch.nn.Module)

    # Verify it has expected components (GenericVectorizationNet has 'features', not 'conv')
    assert hasattr(model, "features")
    assert hasattr(model, "hidden")
    assert hasattr(model, "output")


def test_load_model_with_invalid_spec():
    """Test that loading with invalid spec raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent_spec.json")


def test_model_forward_pass():
    """Test that loaded model can do a forward pass with dummy input."""
    spec_path = "vectorization/models/specs/resnet18_blocks1_bn_64__c2h__trans_heads4_feat256_blocks8_ffmaps512__h2o__out512.json"

    model = load_model(spec_path)
    model.eval()  # Set to eval mode

    # Create dummy input (batch_size=1, channels=3, height=64, width=64)
    dummy_input = torch.randn(1, 3, 64, 64)
    n_primitives = 10  # Number of primitives to predict

    # Forward pass should not raise error
    with torch.no_grad():
        output = model(dummy_input, n_primitives)

    # Verify output shape (should be batch_size=1, n_primitives, features=8 based on spec)
    assert output.shape[0] == 1  # batch size
    assert output.shape[1] == n_primitives  # number of primitives
    assert output.shape[-1] == 8  # out_features from spec
