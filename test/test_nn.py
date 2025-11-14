import torch
import numpy as np
import math
import sys
sys.path.insert(0, '/Users/kausthubjadhav/sandbox/pytorch-mini')
import minitorch
from minitorch.tensor import Tensor
from minitorch.nn import Linear, Sequential, ReLU

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


def test_sequential_composition():
    class AddOne:
        def parameters(self):
            return []

        def __call__(self, x):
            return x + 1

    class Double:
        def parameters(self):
            return []

        def __call__(self, x):
            return x * 2

    seq = Sequential(AddOne(), Double(), AddOne())
    minitorch_output = seq(3)

    class TorchAddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class TorchDouble(torch.nn.Module):
        def forward(self, x):
            return x * 2

    torch_seq = torch.nn.Sequential(TorchAddOne(), TorchDouble(), TorchAddOne())
    input_value = torch.tensor(3.0)
    torch_output = torch_seq(input_value).item()

    print(f"MiniTorch seq(3) = {minitorch_output}, PyTorch seq(3) = {torch_output}")


def test_linear_relu_small():
    torch.manual_seed(1)
    np.random.seed(1)

    in_features = 2
    hidden_features = 3
    output_features = 1

    minitorch_model = Sequential(
        Linear(in_features, hidden_features),
        ReLU(),
        Linear(hidden_features, output_features)
    )

    input_data = Tensor(np.array([[0.5, -0.3]], dtype=np.float32))
    minitorch_output = minitorch_model(input_data)

    torch_linear1 = torch.nn.Linear(in_features, hidden_features)

    # Using identical weights to check direct comparison
    with torch.no_grad():
        torch_linear1.weight.copy_(torch.tensor(minitorch_model[0].weights.data, dtype=torch.float32))
        torch_linear1.bias.copy_(torch.tensor(minitorch_model[0].bias.data, dtype=torch.float32))

    torch_linear2 = torch.nn.Linear(hidden_features, output_features)
    with torch.no_grad():
        torch_linear2.weight.copy_(torch.tensor(minitorch_model[2].weights.data, dtype=torch.float32))
        torch_linear2.bias.copy_(torch.tensor(minitorch_model[2].bias.data, dtype=torch.float32))

    torch_seq = torch.nn.Sequential(
        torch_linear1,
        torch.nn.ReLU(),
        torch_linear2)

    torch_input = torch.tensor([[0.5, -0.3]], dtype=torch.float32)
    torch_output = torch_seq(torch_input)

    print(
        f"MiniTorch Linear+ReLU output: {minitorch_output.data}, PyTorch Linear+ReLU output: {torch_output.detach().numpy()}"
    )


def test_sequential_parameters():
    seq = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    params = seq.parameters()

    # Compute expected scalar parameter counts using the general formula
    # for each Linear layer: out_features * (in_features + 1).
    total_formula_params = 0
    print("Per-layer parameter counts (by formula):")
    for idx, layer in enumerate(seq.layers):
        if isinstance(layer, Linear):
            # weights shape is (out_features, in_features)
            out_features, in_features = layer.weights.data.shape
            layer_params = out_features * (in_features + 1)
            total_formula_params += layer_params
            print(f"  Layer {idx}: Linear({in_features}, {out_features}) -> {layer_params} params")

    # Actual scalar parameters based on tensors returned by seq.parameters()
    actual_scalar_params = sum(p.data.size for p in params)

    print("\nSequential collected parameters (tensors):", params)
    print(f"Total scalar params (by formula): {total_formula_params}")
    print(f"Total scalar params (from tensors): {actual_scalar_params}")


if __name__ == "__main__":
    # test_kaiming_uniform_bounds()
    # test_linear_shape()
    # test_sequential_composition()
    # test_linear_relu_small()
    test_sequential_parameters()    