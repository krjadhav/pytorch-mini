"""Tensor class built on top of numpy arrays"""

import math
import numpy as np

class Tensor:
    """Tensor is a n-dimensional array of scalars."""
    def __init__(self, data, _children=(), _op=None):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data # numpy array
        self.grad = np.zeros_like(self.data) # numpy array
        self._backward = lambda: None # Store function to compute the backward pass
        self._prev = set(_children) # Store the previous tensors
        self._op = _op # Store the name of the operation that created this tensor

    def __add__(x, y):
        out = Tensor(x.data + y.data, (x, y), '+')

        def _backward():
            x.grad += out.grad
            y.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(x, y):
        out = Tensor(x.data * y.data, (x, y), '*')

        def _backward():
            x.grad += y.data * out.grad
            y.grad += x.data * out.grad
        out._backward = _backward
        return out

    # Add a method that sums all the elements in the tensor
    def sum(self):
        out = Tensor(np.sum(self.data), (self, ), 'sum')

        def _backward():
            self.grad += out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child) # Recursively visit all the children
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()