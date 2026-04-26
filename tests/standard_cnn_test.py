# To run these tests, use the command: pytest tests/standard_cnn_test.py

import pytest
from sys import path

import torch

path.append('..')

from models.standard_cnn import StandardCNN

@pytest.fixture
def model():
    return StandardCNN()

def test_forward_pass(model):
    # Create a dummy input tensor (batch_size=2, channels=1, height=28, width=28)
    dummy_input = torch.randn(2, 1, 28, 28)
    
    # Forward pass
    output = model(dummy_input)
    
    # Check output shape (should be [batch_size, num_classes])
    assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"

def test_output_values(model):
    # Create a dummy input tensor
    dummy_input = torch.randn(2, 1, 28, 28)
    
    # Forward pass
    output = model(dummy_input)
    
    # Check if output values are finite (not NaN or Inf)
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"


