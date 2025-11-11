import torch
import numpy as np
import math
import sys
sys.path.insert(0, '/Users/kausthubjadhav/sandbox/pytorch-mini')
import minitorch
from minitorch.tensor import Tensor
from minitorch.nn import Linear

def test_linear_shape():
    # Initialize torch.nn.Linear(784, 128) in pytorch
    pytorch_linear = torch.nn.Linear(784, 128)
    pytorch_linear_weights = pytorch_linear.weight.data.numpy()
    pytorch_linear_bias = pytorch_linear.bias.data.numpy()

    minitorch_linear = Linear(784, 128)
    minitorch_linear_weights = minitorch_linear.weights
    minitorch_linear_bias = minitorch_linear.bias

    print(f"pytorch weights shape: {pytorch_linear_weights.shape} vs minitorch weights shape: {minitorch_linear_weights.data.shape}")
    print(f"pytorch bias shape: {pytorch_linear_bias.shape} vs minitorch bias shape: {minitorch_linear_bias.data.shape}")


def test_kaiming_uniform_bounds():
    # Note: Exact values won't match due to different random seeds,
    # but statistical properties should be similar

    torch.manual_seed(0)
    np.random.seed(0)

    num_inputs = 784
    num_outputs = 128

    pytorch_linear = torch.nn.Linear(num_inputs, num_outputs)
    pytorch_linear_weights = pytorch_linear.weight.data.numpy()
    pytorch_linear_bias = pytorch_linear.bias.data.numpy()

    minitorch_linear = Linear(num_inputs, num_outputs)
    minitorch_linear_weights = minitorch_linear.weights
    minitorch_linear_bias = minitorch_linear.bias

    expected_bound = 1.0 / math.sqrt(num_inputs)
    minitorch_weights = minitorch_linear_weights.data
    minitorch_bias = minitorch_linear_bias.data

    def check(label, passed, detail):
        emoji = "✅" if passed else "❌"
        print(f"{emoji} {label}: {detail}")

    def within_bounds(arr):
        return arr.min() >= -expected_bound - 1e-6 and arr.max() <= expected_bound + 1e-6

    shapes_match = pytorch_linear_weights.shape == minitorch_weights.shape
    bias_shapes_match = pytorch_linear_bias.shape == minitorch_bias.shape
    torch_in_bounds = within_bounds(pytorch_linear_weights)
    mini_in_bounds = within_bounds(minitorch_weights)

    mean_diff = abs(pytorch_linear_weights.mean() - minitorch_weights.mean())
    std_diff = abs(pytorch_linear_weights.std() - minitorch_weights.std())

    check("Weight shape matches", shapes_match, f"torch {pytorch_linear_weights.shape} vs minitorch {minitorch_weights.shape}")
    check("Bias shape matches", bias_shapes_match, f"torch {pytorch_linear_bias.shape} vs minitorch {minitorch_bias.shape}")
    check("torch weights within ±{expected_bound:.4f}", torch_in_bounds,
          f"min {pytorch_linear_weights.min():.4f}, max {pytorch_linear_weights.max():.4f}")
    check("minitorch weights within ±{expected_bound:.4f}", mini_in_bounds,
          f"min {minitorch_weights.min():.4f}, max {minitorch_weights.max():.4f}")
    check("Mean close", mean_diff < 0.15, f"diff {mean_diff:.4f}")
    check("Std close", std_diff < 0.10, f"diff {std_diff:.4f}")

if __name__ == "__main__":
    test_kaiming_uniform_bounds()
    # test_linear_shape()
