import pytest
import torch
from src.my_model import CNN

# Fixtures
@pytest.fixture(scope="module")
def trained_model():
   model = CNN(num_classes=10)
   model.load_state_dict(torch.load("model_weights.pth"))
   model.eval()
   return model

@pytest.fixture
def valid_input_batch():
   return torch.randn(4, 3, 32, 32)  # Batch of  images
