# PyTorch Mini

A minimalist PyTorch-like autograd engine and neural network library, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Overview

This project implements automatic differentiation (autograd) for scalar values and provides neural network building blocks. It's designed to be educational, showing how modern deep learning frameworks work under the hood with a simple, well-documented codebase.

**Key Learning Goals:**
- Understand how automatic differentiation works through computation graphs
- Learn the mechanics of backpropagation and gradient descent
- See how neural networks are built from simple components
- Compare custom implementations with PyTorch to verify correctness

## Quick Start

```python
from minitorch.engine import Tensor
from minitorch.nn import MLP

# Basic autograd
x = Tensor(2.0)
y = (x ** 2).relu()
y.backward()
print(f"dy/dx = {x.grad}")  # 4.0

# Train a neural network
model = MLP(3, [4, 4, 1])
# ... (see examples below for full training loop)
```

## Features

- **Automatic Differentiation**: Implements backpropagation through a dynamically built computation graph using topological sorting
- **Mathematical Operations**: 
  - Arithmetic: `+`, `-`, `*`, `/`, `**` (with reverse operations)
  - Functions: `exp()`, `log()`
  - Activations: `relu()`, `tanh()`
- **Neural Network Components**: 
  - `Neuron`: Single neuron with learnable weights and bias
  - `Layer`: Fully-connected layer of neurons
  - `MLP`: Multi-layer perceptron with configurable architecture
- **Training Support**: 
  - Manual gradient descent implementation
  - Gradient accumulation for reused variables
  - Zero-gradient functionality
- **Validation**: Comprehensive tests comparing against PyTorch to ensure correctness

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

### Tensor Class (`engine.py`)

The `Tensor` class works with scalar values and includes:

- **Forward pass**: Computes outputs and builds a dynamic computation graph
- **Backward pass**: Computes gradients using reverse-mode automatic differentiation
  - Uses topological sorting (DFS) for efficient gradient computation
  - Supports gradient accumulation for variables used multiple times
- **Operations**: All operations automatically track gradients
  - Arithmetic: `+`, `-`, `*`, `/`, `**`
  - Reverse operations: `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`
  - Math functions: `exp()`, `log()`
  - Activations: `relu()`, `tanh()`

### Neural Network Components (`nn.py`)

The `nn.py` module provides building blocks for creating neural networks:

- **Neuron**: Single artificial neuron with:
  - Randomly initialized weights and bias (uniform distribution [-1, 1])
  - Tanh activation function
  - `parameters()` method to access learnable parameters
- **Layer**: Fully-connected layer with:
  - Configurable number of inputs and outputs
  - Collection of neurons
  - Returns single value for 1 output, list for multiple outputs
- **MLP (Multi-Layer Perceptron)**: Complete neural network with:
  - Configurable architecture (input size + list of layer sizes)
  - Sequential layer composition
  - `parameters()` method to access all learnable parameters

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

#### Neural Network Training Example
```python
from minitorch.nn import MLP

# Create a 3-layer neural network
# Input: 3 features, Hidden layers: 4 neurons each, Output: 1 neuron
model = MLP(3, [4, 4, 1])

# Training data
input_data = [
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 1.0],
    [1.0, 0.5, 0.5],
    [1.0, 1.0, 1.0],
]
desired_targets = [-1.0, 1.0, -1.0, 1.0]

# Training loop
lr = 0.01  # learning rate
for iteration in range(100):
    # Zero gradients
    for p in model.parameters():
        p.grad = 0
    
    # Forward pass
    predictions = [model(x) for x in input_data]
    
    # Calculate loss (MSE)
    loss = sum((pred - target) ** 2 for pred, target in zip(predictions, desired_targets))
    
    # Backward pass
    loss.backward()
    
    # Update weights (gradient descent)
    for p in model.parameters():
        p.data -= lr * p.grad
    
    print(f"Iteration {iteration}: Loss = {loss.data:.6f}")
```

## Testing

The project includes comprehensive tests that validate the implementation against PyTorch:

### Engine Tests (`test/test_engine.py`)
- Basic tensor operations (addition, multiplication)
- Gradient computation for complex expressions
- Tanh activation function
- Gradient accumulation for reused variables

### Neural Network Tests (`test/test_nn.py`)
- Individual neuron forward pass
- Layer forward pass
- MLP forward pass
- End-to-end training loop with loss visualization
- Comparison with PyTorch's equivalent implementation

Run tests:
```bash
# Install test dependencies
pip install torch matplotlib

# Test autograd engine
python test/test_engine.py

# Test neural network components (includes training visualization)
python test/test_nn.py
```

## What's Implemented

- [x] Scalar autograd engine with reverse-mode differentiation
- [x] Arithmetic operations: `+`, `-`, `*`, `/`, `**`
- [x] Math functions: `exp()`, `log()`
- [x] Activation functions: `relu()`, `tanh()`
- [x] Neural network components: `Neuron`, `Layer`, `MLP`
- [x] Parameter access via `.parameters()` method
- [x] Manual gradient descent training loop
- [x] Comprehensive tests against PyTorch

## Future Enhancements

- [ ] Additional activation functions (Sigmoid, Softmax, LeakyReLU)
- [ ] Built-in loss functions (MSE, Cross-Entropy)
- [ ] Optimizer classes (SGD, Adam, RMSprop)
- [ ] True tensor support (multi-dimensional arrays)
- [ ] Broadcasting support
- [ ] Batch processing
- [ ] Additional layers (Linear, Conv2d, BatchNorm)
- [ ] Model serialization (save/load weights)
- [ ] GPU acceleration support

## References

- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [PyTorch](https://pytorch.org/) - The full-featured deep learning framework

## License

MIT
