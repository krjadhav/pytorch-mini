# PyTorch Mini

A minimalistic [PyTorch](https://github.com/pytorch/pytorch)-like library for learning purposes, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

This project implements autograd from scratch with two implementations:
1. **Scalar-based engine** (`engine.py`) - Educational implementation working with individual scalars
2. **NumPy-based tensors** (`tensor.py`) - Actual tensor support for n-dimensional arrays

More on [roadmap](https://github.com/users/krjadhav/projects/3/views/1).

## Features

### Scalar Engine (`minitorch.engine`)
- **Arithmetic operations**: `+`, `-`, `*`, `/`, `**`
- **Math functions**: `exp()`, `log()`
- **Activations**: `relu()`, `tanh()`
- **Autograd**: Forward pass with dynamic computation graph and reverse-mode automatic differentiation

### Tensor Engine (`minitorch.tensor`)
- **NumPy-backed tensors**: n-dimensional array support
- **Operations**
  - Binary Operators (takes 2 tensors of the same size and returns a tensor of the same size)
    - `+`, `-`, `*`, `/`, `**`
  - Unary Operators (takes a tensor and returns a tensor of the same size)
    - `exp()`, `log()`, `relu()`, `tanh()`
  - Reduction Operators (takes a tensor and returns a scalar/ 3d tensor to 2d tensor)
    - `sum()`
  - Movement Operators (not implemented yet, but plan to add.)
    
- **Autograd**: Gradient computation for tensor operations

### Neural Network API (`minitorch.nn`)
- **`Neuron`**: Single neuron with learnable weights and bias
- **`Layer`**: Fully-connected layer of neurons
- **`MLP`**: Multi-layer perceptron with configurable architecture
- **`parameters()`**: Access learnable weights and biases

### Testing
- Comprehensive tests comparing outputs with PyTorch
- `test_engine.py`: Tests for scalar-based engine
- `test_tensor.py`: Tests for NumPy-based tensors

## Installation

```bash
# Clone the repository
git clone https://github.com/krjadhav/pytorch-mini.git
cd pytorch-mini

# Install dependencies for tensor support
pip install numpy

# Install test dependencies (optional)
pip install torch matplotlib
```

## Project Structure

```
pytorch-mini/
├── minitorch/
│   ├── engine.py          # Scalar-based Tensor with autograd
│   ├── tensor.py          # NumPy-based Tensor with autograd
│   └── nn.py              # Neural network components (Neuron, Layer, MLP)
├── test/
│   ├── test_engine.py     # Tests for scalar engine
│   ├── test_tensor.py     # Tests for NumPy tensors
│   └── test_nn.py         # Tests for neural network components
├── pyproject.toml
└── README.md
```

## Example Usage

### Scalar Operations (engine.py)
```python
from minitorch.engine import Tensor

# Create scalar tensors
a = Tensor(4)
b = Tensor(3)
c = Tensor(2)

# Build computation graph
d = a + b  # d = 7
l = d * c  # l = 14

# Compute gradients
l.backward()

print(f"a.grad: {a.grad}")  # 2 (dl/da = c)
print(f"b.grad: {b.grad}")  # 2 (dl/db = c)
print(f"c.grad: {c.grad}")  # 7 (dl/dc = d)
```

### NumPy Tensor Operations (tensor.py)
```python
from minitorch.tensor import Tensor

# Create n-dimensional tensors
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Element-wise operations
c = a + b  # [5, 7, 9]
d = a * b  # [4, 10, 18]

# Compute gradients
loss = d.sum()  # Sum all elements
loss.backward()

print(f"a.grad: {a.grad}")  # Gradient with respect to a
print(f"b.grad: {b.grad}")  # Gradient with respect to b
```


### Neural Network Training Example
Still need to add tensor implementation
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