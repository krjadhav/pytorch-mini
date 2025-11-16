"""Tensor class built on top of numpy arrays"""
import math
import numpy as np

class Tensor:
    """Tensor is a n-dimensional array of scalars."""
    def __init__(self, data, _children=(), _op=None):
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data # numpy array
        self.grad = np.zeros_like(self.data, dtype=np.float32) # numpy array
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
        # Support Tensor * Tensor and Tensor * scalar
        if isinstance(y, Tensor):
            out = Tensor(x.data * y.data, (x, y), '*')

            def _backward():
                # d/dx (x * y) = y
                # d/dy (x * y) = x
                x.grad += y.data * out.grad
                y.grad += x.data * out.grad
        else:
            # y is a scalar or numpy array treated as constant (no grad)
            out = Tensor(x.data * y, (x,), '*')

            def _backward():
                # d/dx (x * c) = c
                x.grad += y * out.grad

        out._backward = _backward
        return out

    # Unary Operators
    def __pow__(x, y):
        # Edge cases to deal with type of y. Also accept numpy arrays
        # If y is an int or float then y.data would throw an error
        # If y is a numpy array this should still work I think, but would I need to update y grad?
        # For now, I'll treat y as a constant and not include it in the computational graph
        y = y.data if isinstance(y, Tensor) else y
        
        out = Tensor(x.data ** y, (x, ), f'**{y}')

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

    def linear(x, weights, bias):
        output_data = np.matmul(x.data, weights.data.T) + bias.data
        out = Tensor(output_data, (x, weights, bias), 'linear')

        def _backward():
            x.grad += np.matmul(out.grad, weights.data)
            weights.grad += np.matmul(out.grad.T, x.data)
            bias.grad += out.grad.sum(axis=0)

        out._backward = _backward
        return out

    def __matmul__(x, y):
        data = np.matmul(x.data, y.data)
        out = Tensor(data, (x, y), '@')

        def _backward():
            x.grad += np.matmul(out.grad, y.data.T)
            y.grad += np.matmul(x.data.T, out.grad)

        out._backward = _backward
        return out

    def __rmatmul__(x, y):
        return y.__matmul__(x)

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
        # Format array with PyTorch-like style with trailing zeros
        if self.data.shape == ():
            # Scalar
            data_str = f"{self.data.item():.4f}"
        else:
            # Array - format each element with 4 decimals
            formatted = [f"{x:.4f}" for x in self.data.flatten()]
            data_str = "[" + ", ".join(formatted) + "]"
        return f"Tensor({data_str})"

    # Weight Initialization Methods
    @staticmethod
    def kaiming_uniform(num_inputs, num_outputs):
        # num_inputs is fan_in, num_outputs is fan_out. This is the convention used in PyTorch
        # Check this thread for more details on choices made: https://github.com/pytorch/pytorch/issues/57109
        # Standard formula:
        #   gain = sqrt(2 / (1 + a^2)) where `a` is the negative slope of the
        #        subsequent leaky ReLU (He et al., 2015)
        #   bound = gain * sqrt(3 / fan_in)
        #   U(-bound, bound)
        # Implementation detail:
        #   PyTorch's kaiming_uniform_ defaults to a = sqrt(5), which makes
        #   gain = sqrt(1/3) and the bound collapse to 1/sqrt(fan_in).
        #   We mirror that behaviour so this helper matches nn.Linear's
        #   initialization convention exactly.
        gain = math.sqrt(1/3.0)
        bound = gain * math.sqrt(3.0 / num_inputs)
        return Tensor(np.random.uniform(-bound, bound, (num_outputs, num_inputs)))
    
    @staticmethod
    # Simple uniform initialization
    def uniform(num_inputs, num_outputs):
        bound = 1.0 / math.sqrt(num_inputs)
        return Tensor(np.random.uniform(-bound, bound, (num_outputs)))

    @staticmethod
    def normal(num_inputs, num_outputs):
        return Tensor(np.random.normal(0, 1, (num_outputs, num_inputs)))