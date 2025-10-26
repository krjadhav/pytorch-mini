# PyTorch Mini

A work-in-progress minimalist PyTorch library, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Overview

This project aims to replicate the core concepts of micrograd while building towards a minimalist PyTorch-like library. It implements automatic differentiation (autograd) for scalar values with a simple, educational codebase.

## Features

- **Automatic Differentiation**: Implements backpropagation through a dynamically built computation graph
- **Mathematical Operations**: Supports addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), power (`**`) and reverse arithmetic operations (`__radd__`, `__rmul__`, etc.), exponential (`exp`), logarithm (`log`)
- **Activation Functions**: ReLU and Tanh with automatic differentiation support
- **Neural Network Components**: Neuron, Layer, and Multi-Layer Perceptron (MLP) classes
- **Gradient Accumulation**: Handles multiple uses of the same variable in computation graphs
- **Topological Sorting**: Uses depth-first search for efficient gradient computation

## Project Structure

```
pytorch-mini/
├── minitorch/
│   ├── __init__.py
│   ├── engine.py          # Core Tensor class with autograd
│   └── nn.py              # Neural network components (Neuron, Layer, MLP)
├── test/
│   ├── test_engine.py     # Tests comparing with PyTorch
│   └── test_nn.py         # Tests for neural network components
├── pyproject.toml
└── README.md
```

## Current Implementation

### Tensor Class

The `Tensor` class (currently working with scalar values) includes:

- **Forward pass**: Computes outputs and builds computation graph
- **Backward pass**: Computes gradients using reverse-mode automatic differentiation via topological sort
- **Arithmetic Operations**: `+`, `-`, `*`, `/`, `**` with proper gradient functions
- **Mathematical Functions**: `exp()`, `log()` with automatic differentiation support
- **Activation Functions**: `relu()`, `tanh()` with automatic differentiation support
- **Operator Overloading**: Full support for Python operators including reverse operations

### Neural Network Components

The `nn.py` module provides building blocks for creating neural networks:

- **Neuron**: Single artificial neuron with learnable weights and bias, applies tanh activation
- **Layer**: Collection of neurons forming a fully-connected layer
- **MLP (Multi-Layer Perceptron)**: Stack of layers forming a complete neural network

### Example Usage

#### Basic Operations
```python
from minitorch.engine import Tensor

# Create tensors
a = Tensor(4)
b = Tensor(3)
c = Tensor(2)

# Build computation graph
d = a + b  # d = 7
e = d * c  # e = 14

# Compute gradients
e.backward()

print(f"a.grad: {a.grad}")  # 2 (de/da = c)
print(f"b.grad: {b.grad}")  # 2 (de/db = c)
print(f"c.grad: {c.grad}")  # 7 (de/dc = d)
```

#### Advanced Operations
```python
from minitorch.engine import Tensor

# More complex computation graph
x = Tensor(2.0)
y = Tensor(3.0)

# Using multiple operations
z = (x ** 2 + y).relu()  # z = relu(4 + 3) = 7
w = z.exp() / (z + 1)    # w = exp(7) / 8

# Backpropagation
w.backward()

print(f"x.grad: {x.grad}")  # Gradient of w with respect to x
print(f"y.grad: {y.grad}")  # Gradient of w with respect to y
```

#### Neural Network Example
```python
from minitorch.nn import MLP

# Create a 3-layer neural network
# Input: 3 features, Hidden layers: 4 neurons each, Output: 1 neuron
mlp = MLP(3, [4, 4, 1])

# Forward pass
inputs = [2.0, 3.0, 4.0]
output = mlp(inputs)

# Backpropagation
output.backward()

print(f"Output: {output}")
```

## Testing

Tests compare the custom implementation against PyTorch to verify correctness:

```bash
# Test autograd engine
python test/test_engine.py

# Test neural network components
python test/test_nn.py
```

## Roadmap

- [x] Add more operations (subtraction, division, power, exp, log)
- [x] Add activation functions (ReLU, Tanh)
- [x] Implement basic neural network components (Neuron, Layer, MLP)
- [ ] Add more activation functions (Sigmoid, Softmax)
- [ ] Implement parameter access and optimization methods
- [ ] Add loss functions (MSE, Cross-Entropy)
- [ ] Implement optimizers (SGD, Adam)
- [ ] Implement true tensor support (multi-dimensional arrays)
- [ ] Add neural network layers (Linear, Conv2d, etc.)
- [ ] Implement optimizers (SGD, Adam)
- [ ] Support for GPU acceleration
- [ ] Add broadcasting support

## References

- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [PyTorch](https://pytorch.org/) - The full-featured deep learning framework

## License

MIT
