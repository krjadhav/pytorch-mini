# PyTorch Mini

This is WIP attempt to build a minimalistic [PyTorch](https://github.com/pytorch/pytorch) library for learning purposes inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

Currently, it's a tensor class wrapped around scalar, and I plan to work towards actual tensor support. More on [roadmap](https://github.com/users/krjadhav/projects/3/views/1).


At it's current stage it supports:
- Scalar operations:
  - Arithmetic: `+`, `-`, `*`, `/`, `**`
  - Math functions: `exp()`, `log()`
  - Activations: `relu()`, `tanh()`
- Autograd Engine:
  - Forward pass computing outputs and building a dynamic computation graph
  - Backpropagation computing gradients using reverse-mode automatic differentiation
- Neural Network API:
  - `Neuron`: Single neuron with learnable weights and bias
  - `Layer`: Fully-connected layer of neurons
  - `MLP`: Multi-layer perceptron with configurable architecture
  - `parameters()` method to access learnable weights and biases
- Testing
  - In `/test` directory, there are tests comparing with PyTorch

## Project Structure

```
pytorch-mini/
├── minitorch/
│   ├── engine.py          # Core Tensor class with autograd
│   └── nn.py              # Neural network components (Neuron, Layer, MLP)
├── test/
│   ├── test_engine.py     # Tests comparing with PyTorch
│   └── test_nn.py         # Tests for neural network components
├── pyproject.toml
└── README.md
```

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
l = d * c  # e = 14

# Compute gradients
l.backward()

print(f"a.grad: {a.grad}")  # 2 (dl/da = c)
print(f"b.grad: {b.grad}")  # 2 (dl/db = c)
print(f"c.grad: {c.grad}")  # 7 (dl/dc = d)
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

## License

MIT
