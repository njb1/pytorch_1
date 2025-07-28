# test_model_serving.py
import pytest
import torch
import time
import numpy as np
from src.my_model import CNN  # The PyTorch model class


def test_invalid_input_type(trained_model):
   """
   Test that the model raises a TypeError when given an input of an invalid type.

   This test verifies that passing a NumPy array instead of a PyTorch tensor to the trained model
   results in a TypeError, ensuring that the model enforces correct input types.
   """
   with pytest.raises(TypeError):
       # Pass numpy array instead of tensor
       trained_model(np.random.rand(1, 3, 32, 32))


def test_incorrect_input_dimensions(trained_model):
   """
   Tests that the trained_model raises a RuntimeError when provided with input tensors of 
   incorrect dimensions.
   Specifically, checks for:
   - Missing batch dimension in the input tensor.
   - Incorrect channel size in the input tensor.
   """
   with pytest.raises(RuntimeError):
       # Missing batch dimension
       trained_model(torch.randn(3, 32, 32))

   with pytest.raises(RuntimeError):
       # Wrong channel size
       trained_model(torch.randn(1, 1, 32, 32))


def test_non_normalized_input(trained_model, valid_input_batch):
   """
   Tests the model's robustness to non-normalized input by providing input data with extreme values.
   Ensures that the model processes the input without crashing and that the output does not contain 
   NaN values.

   Args:
      trained_model: The trained PyTorch model to be tested.
      valid_input_batch: A batch of valid input data, expected to be normalized in the range [-1, 1].

   Raises:
      AssertionError: If the model's output contains any NaN values.
   """
   # Extreme value range [-1000, 1000] instead of [-1,1]
   extreme_input = valid_input_batch * 1000
   # Should process without crashing
   output = trained_model(extreme_input)
   assert not torch.isnan(output).any(), "Output contains NaNs"


def test_output_shape(trained_model, valid_input_batch):
   """
   Tests that the output shape of the trained model matches the expected dimensions.

   Args:
      trained_model: The model that has been trained and is to be evaluated.
      valid_input_batch: A batch of valid input data to be passed through the model.

   Asserts:
      The output tensor from the model has shape (4, 10), corresponding to [batch_size, num_classes].
   """
   output = trained_model(valid_input_batch)
   assert output.shape == (4, 10)  # [batch_size, num_classes]


def test_output_data_type(trained_model, valid_input_batch):
   """
   Tests that the output data type of the trained model, when given a valid input batch, is torch.float32.

   Args:
      trained_model: The model that has been trained and is to be tested.
      valid_input_batch: A batch of valid input data to be passed to the model.

   Asserts:
      The output tensor's dtype is torch.float32.
   """
   output = trained_model(valid_input_batch)
   assert output.dtype == torch.float32


def test_output_probability_range(trained_model, valid_input_batch):
   """
   Tests that the model's output probabilities are within the valid range [0, 1] and that they sum to 1 for each input in the batch.

   Args:
      trained_model: The trained PyTorch model to be evaluated.
      valid_input_batch: A batch of valid input data for the model.

   Asserts:
      - All softmax output values are between 0 and 1 (inclusive).
      - The sum of softmax probabilities for each input in the batch is 1.
   """
   output = trained_model(valid_input_batch)
   softmax = torch.nn.functional.softmax(output, dim=1)
   assert torch.all(softmax >= 0) and torch.all(softmax <= 1)
   assert torch.allclose(softmax.sum(dim=1), torch.tensor(1.0))

def test_model_determinism(trained_model, valid_input_batch):
   """
   Tests that the model produces deterministic outputs for the same input batch.

   Args:
      trained_model (torch.nn.Module): The trained PyTorch model to be tested.
      valid_input_batch (torch.Tensor): A batch of valid input data for the model.

   Asserts:
      The outputs of the model for two consecutive runs with the same input are close within a tolerance (atol=1e-6).
   """
   output1 = trained_model(valid_input_batch)
   output2 = trained_model(valid_input_batch)
   assert torch.allclose(output1, output2, atol=1e-6)


def test_batch_independence(trained_model):
   """
   Tests that the model produces the same outputs for individual samples as when those samples are part of a batch.
   Args:
      trained_model: The PyTorch model to be tested.
   Asserts:
      - The output for each sample in the batch is numerically close to the output when the sample is processed individually.
      - Uses a tolerance of 1e-6 for floating point comparison.
   """
   batch = torch.randn(2, 3, 32, 32)
   single_out0 = trained_model(batch[0].unsqueeze(0))
   single_out1 = trained_model(batch[1].unsqueeze(0))
   batch_out = trained_model(batch)

   assert torch.allclose(batch_out[0], single_out0[0], atol=1e-6)
   assert torch.allclose(batch_out[1], single_out1[0], atol=1e-6)


def test_all_zeros_input(trained_model):
   """
   Tests the trained_model's behavior when given an all-zeros input tensor.

   This test ensures that the model does not produce any NaN values in its output
   when provided with a tensor of zeros as input. The input tensor has the shape
   (1, 3, 32, 32), which typically corresponds to a single RGB image of size 32x32.

   Args:
      trained_model: The PyTorch model to be tested.

   Asserts:
      The output of the model does not contain any NaN values.
   """
   zeros = torch.zeros(1, 3, 32, 32)
   output = trained_model(zeros)
   assert not torch.isnan(output).any()


def test_all_ones_input(trained_model):
   """
   Tests the trained model's output when given an input tensor of all ones.

   This test creates a tensor filled with ones of shape (1, 3, 32, 32), passes it through the trained model,
   applies the softmax function to the output along dimension 1, and asserts that all values in the softmax
   output are less than or equal to 1.

   Args:
      trained_model (torch.nn.Module): The trained model to be tested.

   Asserts:
      All values in the softmax output are less than or equal to 1.
   """
   ones = torch.ones(1, 3, 32, 32)
   output = trained_model(ones)
   softmax = torch.softmax(output, dim=1)
   assert torch.all(softmax <= 1)


def test_inference_performance(trained_model, valid_input_batch):
   """
   Tests the inference performance of a trained model by measuring the time taken to process a batch of inputs.
   Args:
      trained_model: The trained model to be evaluated. Should be callable with the input batch.
      valid_input_batch: A batch of valid input data to be passed to the model.
   Asserts:
      The inference time for the input batch is less than 0.1 seconds.
   """
   
   start = time.time()
   _ = trained_model(valid_input_batch)
   elapsed = time.time() - start

   assert elapsed < 0.1  # Threshold for batch of 4
