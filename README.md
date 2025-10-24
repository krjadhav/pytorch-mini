# PyTorch Mini

A work-in-progress minimalist PyTorch library, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Overview

This project aims to replicate the core concepts of micrograd while building towards a minimalist PyTorch-like library. It implements automatic differentiation (autograd) for scalar values with a simple, educational codebase.

## Features

- **Automatic Differentiation**: Implements backpropagation through a dynamically built computation graph
- **Basic Operations**: Supports addition (`+`) and multiplication (`*`) operations
- **Gradient Accumulation**: Handles multiple uses of the same variable in computation graphs
- **Topological Sorting**: Uses depth-first search for efficient gradient computation

## Project Structure

```
pytorch-mini/
├── minitorch/
│   ├── __init__.py
│   └── engine.py          # Core Tensor class with autograd
├── test/
│   └── test_engine.py     # Tests comparing with PyTorch
├── pyproject.toml
└── README.md
```

## Current Implementation

### Tensor Class

The `Tensor` class (currently working with scalar values) includes:

- **Forward pass**: Computes outputs and builds computation graph
- **Backward pass**: Computes gradients using reverse-mode automatic differentiation
- **Operations**: `__add__` and `__mul__` with proper gradient functions

### Example Usage

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

## Testing

Tests compare the custom implementation against PyTorch to verify correctness:

```bash
python test/test_engine.py
```

## Roadmap

- [ ] Add more operations (subtraction, division, power, exp, log)
- [ ] Implement true tensor support (multi-dimensional arrays)
- [ ] Add neural network layers (Linear, Conv2d, etc.)
- [ ] Implement optimizers (SGD, Adam)
- [ ] Add activation functions (ReLU, Sigmoid, Tanh)
- [ ] Support for GPU acceleration

## References

- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [PyTorch](https://pytorch.org/) - The full-featured deep learning framework

## License

MIT
