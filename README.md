# Pytorch-Mini

This is an attempt to create a minimalist deep learning library from scratch to demystify how [PyTorch](https://github.com/pytorch/pytorch) works internally. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), I started with a scalar engine and then extended it to a tensor engine (More on [roadmap](https://github.com/users/krjadhav/projects/3/views/1)). My goal is to write most of the code without relying on external libraries and abstractions. So far it's been a rewarding learning experience and I'm heading into PyTorch codebase's rabbit hole. Along the way, I've also come across several jargons that I've found helpful to understand [jargons](#jargons). Oh well, I guess I'm in for a long journey :) 

A deep learning library like PyTorch has 4 main components:
1. Tensors
2. Autograd
3. Neural Network API
4. Dataloaders

**Tensors**
Tensors are the smallest unit of data structure in the library. I've extended micrograd's scalar engine to a tensor engine using numpy arrays. This can be seen in the [tensor.py](minitorch/tensor.py) file. It essentially encapsulates **data** as well as the **gradient**. For each of these operations, you also need to define the [forward pass](#forward-pass) and [backward pass](#backward-pass).

There are 4 main categories of Tensor operations:
1. **Binary Operations** (takes 2 tensors of the same size and returns a tensor of the same size)
    - Eg. `+`, `-`, `*`, `/`, `**`
    Unlike scalars, tensors are N-dimensional arrays which introduces edge cases in the implementation. For eg. when adding two tensors of different shapes, we need to broadcast the smaller tensor to match the shape of the larger tensor. For now, numpy's broadcasting feature is used. Also I've learned that it helps to explicilty initialize precisions to avoid errors like `numpy._core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'`
2. **Unary Operations** (takes a tensor and returns a tensor of the same size)
    - Eg. `exp()`, `log()`, `relu()`, `tanh()`
3. **Reduction Operations** (takes a tensor and returns a scalar/ 3d tensor to 2d tensor)
    - Eg. `sum()`
    These operations are especially helpful when calculating gradients. Otherwise you run into the error `RuntimeError: grad can be implicitly created only for scalar outputs`
4. Movement Operations (not implemented yet, but plan to add.)
    I'm yet to go through the codebase to understand how to implement this but they essentially transform tensor data without needing to copy the data.
    - Eg. `reshape()`, `to()`, `cpu()`, `cuda()`


## Autograd
Autograd tracks operations performed on tensors and builds a computational graph dynamically to be able to compute [gradients](#gradient). A topological graph is created in the function `Tensor.backward()` and the graph is traversed in reverse order to compute the [gradient](#gradient).

## Neural Network API
(WIP) The scalar version is available in [nn_scalar.py](minitorch/deprecated/nn_scalar.py). I'm hoping to implement the `nn` and `nn.optim` modules such as: `nn.Sequential`, `nn.Linear`, `nn.ReLU`, `nn.CrossEntropyLoss` and `nn.SGD` here.

## Dataloaders
(WIP) Dataloaders are used to load data in the models and provide features like shuffling, batch size, etc. I'm hoping to implement the `utils.data.TensorDataset` and `utils.data.DataLoader` modules here.

Currently, I'm working on getting the NN module, Optimizers and Dataloaders to work so that I can get a MNIST training example working in [test_mnist.py](test/test_mnist.py). Later, I plan to get it to work with convolutions and also go deeper into low level optimizations.

# Jargons

### forward pass
Calculation of the output of a model given the input. This is the normal flow of the model. Eg. f(x)

### loss
A value that measures the difference between the predicted output and the actual output. It is used to evaluate the performance of the model.

### chain rule
If a loss is 2x increased when model output is changed and the model output is 4x increased when a parameter is changed then loss is influenced 2 x 4 = 8x when a parameter is changed. Mathematically, if l = f(y) and y = g(x) then dl/dx = dl/dy * dy/dx.

### gradient
A derivative extended to multi-variable functions is a gradient. A derivative is the slope of a function at a point. Mathematically, derivatives of a y = f(x) is given by dy/dx = lim h->0 (f(x+h) - f(x))/h

### parameters
All the weights and biases in a model are called parameters.

### backward pass
Calculation of the [gradient](#gradient) of the [loss](#loss) with respect to the [parameters](#parameters) of the model using the [chain rule](#chain-rule).

### automatic differentiation
For a simple function, we can calculate the [gradient](#gradient) using the [chain rule](#chain-rule). For a more complex function, we can create a computational graph and calculate the [gradient](#gradient) using the [autograd](#autograd) engine.

### DAG
A directed acyclic graph is a graph with directed edges and no cycles.

### topological sort
In a DAG, this is a sorted list of nodes where for every edge u -> v, u comes before v in the ordering.

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

## Example Tensor Operations
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