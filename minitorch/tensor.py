"""Tensor class built on top of numpy arrays"""
import numpy as np

class Tensor:
    """Tensor is a n-dimensional array of scalars."""
    def __init__(self, data, _children=(), _op=None):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data # numpy array
        self.grad = np.zeros_like(self.data) # numpy array
        self._backward = lambda: None # Store function to compute the backward pass
        self._prev = set(_children) # Store the previous tensors
        self._op = _op # Store the name of the operation that created this tensor

    def backward(self):
        # pytorch has a ctx to store the graph. Consider refactoring this
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child) # Recursively visit all the children
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    # Binary Operators
    def __add__(x, y):
        out = Tensor(x.data + y.data, (x, y), '+')

        def _backward():
            # d/dx (x + y) = 1
            # d/dy (x + y) = 1
            x.grad += out.grad
            y.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(x, y):
        out = Tensor(x.data * y.data, (x, y), '*')

        def _backward():
            # d/dx (x * y) = y
            # d/dy (x * y) = x
            x.grad += y.data * out.grad
            y.grad += x.data * out.grad
        out._backward = _backward
        return out

    # Unary Operators
    def __pow__(x, y):
        # Edge cases to deal with type of y. Also accept numpy arrays
        # If y is an int or float then y.data would throw an error
        # If y is a numpy array this should still work I think, but would I need to update y grad?
        y = y.data if isinstance(y, Tensor) else y
        
        out = Tensor(x.data ** y, (x, y), f'**{y}')

        def _backward():
            # d/dx x^y = y * x^(y-1)
            x.grad += y * x.data**(y - 1) * out.grad
        out._backward = _backward
        return out

    def exp(x):
        out = Tensor(np.exp(x.data), (x, ), 'e^x')

        def _backward():
            # d/dx e^x = e^x
            x.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(x):
        out = Tensor(np.log(x.data), (x, ), 'log')

        def _backward():
            # d/dx log(x) = 1/x
            x.grad += 1 / x.data * out.grad
        out._backward = _backward
        return out

    def relu(x):
        out = Tensor(np.maximum(0, x.data), (x, ), 'ReLU')

        def _backward():
            # d/dx ReLU(x) = 1 if x > 0 else 0
            x.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(x):
        out = Tensor(np.tanh(x.data), (x, ), 'tanh')

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            x.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    # Reduction Operators
    def sum(self):
        out = Tensor(np.sum(self.data), (self, ), 'sum')

        def _backward():
            # d/dx sum(x) = 1
            self.grad += out.grad
        out._backward = _backward

        return out

    # TODO: Add movement operators

    def __neg__(x):
        return x * -1
    
    def __radd__(x, y):
        return x + y
    
    def __rmul__(x, y):
        return x * y

    def __sub__(x, y):
        return x + (-y)

    def __rsub__(x, y):
        return y + (-x)

    def __truediv__(x, y):
        return x * y**-1
    
    def __rtruediv__(x, y):
        return y * x**-1

    def __repr__(self):
        return f"Tensor({self.data})"