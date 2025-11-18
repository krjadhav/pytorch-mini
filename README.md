# Pytorch-Mini

This is an attempt to create a minimalist deep learning library from scratch to demystify how [PyTorch](https://github.com/pytorch/pytorch) works internally. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), I started with a scalar engine and then extended it to a tensor engine (More on [roadmap](https://github.com/users/krjadhav/projects/3/views/1)).

This library supports a full MNIST training example on CPU. I ran it against PyTorch on a similar architecture and it seems to perform equivalently and slightly outperorms PyTorch on accuracy and time taken.

```
========================================================================
                         MNIST Training Summary                         
========================================================================
Framework     Final Acc (%)     Final Loss   Total Time (s)
------------------------------------------------------------------------
PyTorch               96.04         0.1419             4.81
Minitorch             96.41         0.1319             4.42
========================================================================
```

My goal is to write most of the code without relying on external libraries and abstractions. So far it's been a rewarding learning experience and I'm heading into PyTorch codebase's rabbit hole. Along the way, I've also come across several jargons that I've found helpful to understand [jargons](#jargons). It's also been very helpful to debug issues by comparing outputs to Pytorch so I've done so in the [test/](test). Oh well, I guess I'm in for a long journey :)

A deep learning library like PyTorch has 4 main components:
1. Tensors
2. Autograd
3. Neural Network API
4. Dataloaders

## Tensors
Tensors are the smallest unit of data structure in the library. I've extended micrograd's scalar engine to a tensor engine using numpy arrays. This can be seen in the [tensor.py](minitorch/tensor.py) file. It essentially encapsulates **data** as well as the **gradient**. For each of these operations, you also need to define the [forward pass](#forward-pass) and [backward pass](#backward-pass).

There are 4 main categories of Tensor operations:
1. **Binary Operations** (takes 2 tensors of the same size and returns a tensor of the same size)
    - Eg. `+`, `-`, `*`, `/`, `**`
    Unlike scalars, tensors are N-dimensional arrays which introduces edge cases in the implementation. For eg. when adding two tensors of different shapes, we need to broadcast the smaller tensor to match the shape of the larger tensor. For now, numpy's broadcasting feature is used. Also I've learned that it helps to explicilty initialize precisions to avoid errors like `numpy._core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'`. I ran into several `AttributeError: 'int' object has no attribute errors` especially with power and division operations during the [backward pass](#backward-pass). 
2. **Unary Operations** (takes a tensor and returns a tensor of the same size)
    - Eg. `exp()`, `log()`, `relu()`, `tanh()`, `flatten()`
3. **Reduction Operations** (takes a tensor and returns a scalar/ 3d tensor to 2d tensor)
    - Eg. `sum()`, `argmax()`
    These operations are especially helpful when calculating gradients. Otherwise you run into the error `RuntimeError: grad can be implicitly created only for scalar outputs`
4. Movement Operations (not implemented yet, but plan to add.)
    I'm yet to go through the codebase to understand how to implement this but they essentially transform tensor data without needing to copy the data.
    - Eg. `reshape()`, `to()`, `cpu()`, `cuda()` 


## Autograd
Autograd tracks operations performed on tensors and builds a computational graph dynamically to be able to compute [gradients](#gradient). A topological graph is created in the function `Tensor.backward()` and the graph is traversed in reverse order to compute the [gradient](#gradient).

**Optimizers**
There are several optimization techniques that researchers have come up with. [SGD](#sgd) is the most basic one and there are others variations like Adam, Adagrad, RMSprop, etc. Broadly, they use different heuristics to adjust [parameters](#parameters) quickly and avoid getting stuck in a poor local minima. I've implemented a simple [SGD](#sgd) in [minitorch/optim.py](minitorch/optim.py). It would be nice to add some way to auto adjust the learning rate.

**Loss Functions**
Similar to optimizers, there are several loss functions. I didn't quite understand the difference between loss, error and accuracy. They seemed like different ways of telling how close the prediction is to the ground truth. The numbers humans use to measure model performance aren’t helpful enough for the model itself. If at every step, the model is still 90% accurate, it’s stuck because it has no way of knowing which direction to adjust [parameters](#parameters). So the feedback it receives needs to be continuous and differentiable that it can be minimized. This can't be done on discrete values like accuracy or error.

I've implemented [Cross Entropy Loss](#cross-entropy-loss) in [minitorch/loss.py](minitorch/loss.py) which compares a model’s predictions against the actual class using logarithmic scaling. It assigns a high penalty for confident and wrong predictions, a moderate penalty for unconfident wrong predictions, and a low penalty for confident and correct predictions.

## Neural Network API
A tensor-based neural network API in [nn.py](minitorch/nn.py) that mirrors a tiny subset of `torch.nn`, and the scalar version is still available in [nn_scalar.py](minitorch/deprecated/nn_scalar.py). Currently, it implements [Linear](#linear), and the [Sequential](#sequential) container. I don't quite like the `nn.ReLU()` syntax and feel it's an unnecessary boilerplate so I plan to remove this.



### Initialization
Unlike micrograd, you can't rely on random initialization to initialize weights. But why randomly initialize weights to begin with? Why not initialize them to 0? Or 1 (or some constant)? It's because all neurons will have identical outputs and the same gradient update so it never learns anything.

The range of initialization is also important. If the range is too small, updates are tiny and training becomes slow ([vanishing gradient](#vanishing-gradient)). If the range is too large, updates are large and doesn't learn anything ([exploding gradient](#exploding-gradient)).

Many researchers have come up with some heuristics by experimentation to initialize weights. The intuition is to depend on the architecture of the network to adjust the range of initialization. The most popular ones are the [Xavier initialization](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) and the [He initialization](https://arxiv.org/pdf/1502.01852). I'm going with the **He initialization** for now because it seems to be simple and works well for the ReLU MLP used in the MNIST example.

### Layers
Unlike Pytorch, this doesn't use the `nn.Module` class to define layers. I'm using the `__call__` method to define the [forward pass](#forward-pass).

## Dataloaders
Dataloaders are used to feed data into the model. In most cases, you need to feed data in [batches](#batch) where the it's shuffled each [epoch](#epoch). When testing the MNIST model on Pytorch, batch size does seem to impact the accuracy of the model but I'm not sure what the optimal batch size is but from experience, 128 seems to be a good starting point.

I've implemented a very simple data loader in [dataloaders.py](minitorch/dataloaders.py) that shuffles the data and feeds it in batches. I'm not using the `TensorDataset` because it seems to create tuples under the hood and figured I'd just combine this with the `DataLoader` class. It shuffles data using numpy's random shuffle and Python's generator to `yield` (compute and return shuffled pairs of (data, labels) one at a time using index-based slicing). I ran into a couple of `TypeErrors` here because I forgot to add `__len__`, `__getitem__` and `item()` methods to the `Tensor` class.

# Jargons

#### forward pass
Calculation of the output of a model given the input. This is the normal flow of the model. Eg. f(x)

#### loss
A value that measures the difference between the predicted output and the actual output. It is used to evaluate the performance of the model.

#### chain rule
If a loss is 2x increased when model output is changed and the model output is 4x increased when a parameter is changed then loss is influenced 2 x 4 = 8x when a parameter is changed. Mathematically, if l = f(y) and y = g(x) then dl/dx = dl/dy * dy/dx.

#### gradient
A derivative extended to multi-variable functions is a gradient. A derivative is the slope of a function at a point. Mathematically, derivatives of a y = f(x) is given by dy/dx = lim h->0 (f(x+h) - f(x))/h

#### parameters
All the weights and biases in a model are called parameters.

#### backward pass
Calculation of the [gradient](#gradient) of the [loss](#loss) with respect to the [parameters](#parameters) of the model using the [chain rule](#chain-rule).

#### automatic differentiation
For a simple function, we can calculate the [gradient](#gradient) using the [chain rule](#chain-rule). For a more complex function, we can create a computational graph and calculate the [gradient](#gradient) using the [autograd](#autograd) engine.

#### DAG
A directed acyclic graph is a graph with directed edges and no cycles.

#### topological sort
In a DAG, this is a sorted list of nodes where for every edge u -> v, u comes before v in the ordering.

#### vanishing gradient
Layer updates are so tiny that early layers stop learning.

#### exploding gradient
Layer updates are so large that the model doesn't learn anything.

#### SGD
Stochastic gradient descent. basically ```w -= grad(w)*lr```

#### Cross Entropy Loss
A type of [loss](#loss) function.

#### batch
chunks of data

#### epoch
Iteration of the training loop

## Installation

```bash
# Clone the repository
git clone https://github.com/krjadhav/pytorch-mini.git
cd pytorch-mini
pip install numpy

# Install test dependencies (optional)
pip install torch matplotlib
```

## Project Structure

```
pytorch-mini/
├── minitorch/
│   ├── nn.py                   # Neural network APIs (eg. Linear, ReLU, CrossEntropyLoss)
│   ├── tensor.py               # NumPy-based Tensor with autograd
│   ├── dataloaders.py          # DataLoader for MNIST
│   ├── loss.py                 # Loss functions (eg. CrossEntropyLoss)
│   ├── optimizer.py            # Optimizers (eg. SGD)
│   ├── datasets.py             # utils to download and use external datasets like MNIST
│   └── deprecated/             # Micrograd implementation
│       ├── nn_scalar.py        # Neural network APIs (Neuron, Layer, MLP)
│       └── scalar.py           # Scalar with autograd
│   
├── test/
│   ├── test_nn.py              # Tests for neural network components
│   ├── test_tensor.py          # Tests for NumPy tensors
│   └── test_mnist.py           # Tests for MNIST
│   └── deprecated/             # Micrograd Tests
│       └── test_engine.py      # Tests for scalar engine
│       └── test_nn_scalar.py   # Tests for scalar neural network APIs
├── pyproject.toml
└── README.md
```

## Example
See [examples/mnist.py](examples/mnist.py)
```python
import minitorch
from minitorch.tensor import Tensor
from minitorch.dataloaders import DataLoader

def train(model, train_dataloader, optimizer, epoch, loss_function):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

def test(model, test_dataloader, loss_function):
    test_loss, correct, total = 0.0, 0, 0
    for data, target in test_dataloader:
        output = model(data)
        test_loss += loss_function(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()        
        total += len(target)
    
    print(f"Average loss: {test_loss/total:.4f}, Accuracy: {100.0 * correct/total:.2f}%")

if __name__ == "__main__":
    # Load data
    X_train, Y_train, X_test, Y_test = minitorch.datasets.mnist()
    
    # create tensors
    x_train = Tensor(X_train/255).flatten(start_dim=1)
    y_train = Tensor(Y_train)
    x_test = Tensor(X_test/255).flatten(start_dim=1)
    y_test = Tensor(Y_test)
    
    # prepare dataloaders
    train_dataloader = DataLoader(x_train, y_train, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(x_test, y_test, batch_size=128, shuffle=False)

    # define model
    model = minitorch.nn.Sequential(
        minitorch.nn.Linear(784, 128),
        minitorch.nn.ReLU(),
        minitorch.nn.Linear(128, 64),
    )

    # define optimizer
    optimizer = minitorch.optim.SGD(model.parameters(), lr=0.01)

    # define loss function
    loss_function = minitorch.loss.CrossEntropyLoss(reduction="sum")
    
    # train
    for epoch in range(1, 11):
        train(model, train_dataloader, optimizer, epoch, loss_function)
        test(model, test_dataloader, loss_function)
```