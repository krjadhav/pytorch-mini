import math

class Scalar:
    """ Scalar is a single value. Same as Value in Karapathy's micrograd"""
    def __init__(self, data, _children=(), _op=None):
        self.data = data # This can be int or float for now but I think it would be better to use some form of optimized float
        self.grad = 0 # Store the gradient of the Scalar, by default we assume changing this variable has no effect on the loss
        self._backward = lambda: None # Store function to compute the backward pass
        self._prev = set(_children) # Store the previous Scalars
        self._op = _op # Store the name of the operation that created this Scalar

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')
        
        def _backward():
            """
            Let's say we have an expression o = a + b
            We want to find the derivative of o with respect to a (do/da)
            Using the definition of the derivative: d/dx f(x) as lim h->0 = (f(x+h) - f(x)) / h 
            
            do/da = (((a + h) + b) - (a + b)) / h
            => (a + h + b - a - b) / h
            => h / h => 1

            Similarly, do/db = ((a + (b + h)) - (a + b)) / h
            => (a + b + h - a - b) / h
            => h / h => 1

            

            Suppose L is the loss and o is one of it's intermediate nodes in the computation graph.
            We can find the gradient of L with respect to a and b, using the chain rule.
            dL/da = (dL/do) * (do/da)
            dL/db = (dL/do) * (do/db)

            We know from calculating the local derivatives that,
            do/da = 1
            do/db = 1

            dL/da = (dL/do) * 1
            dL/db = (dL/do) * 1

            The forward pass, calculates outputs o and the backward pass calculates gradients.

            When we trace back the computation graph, each node's .grad stores the computed gradient of the loss with respect to the node's value.
            In this case, out.grad stores the computed gradient of the loss L with respect to the output of the function o.
            Substituting, dL/do = out.grad

            self.grad += out.grad
            other.grad += out.grad

            Why += instead of = ?
            Anytime we have a node that is used multiple times in the computation graph, we need to accumulate the gradients.
            If we use = instead of +=, it will override the gradient of the previous computation.

            Let's say o = a + a
            Then do/da = ((a + h) + (a + h)) - (a + a) /h
            => (2a + 2h - 2a)/h => 2

            o.grad = 1
            a.grad = o.grad * 1 = 1
            a.grad = o.grad * 1 = 1

            Then a.grad = 1 which is wrong

            However, if we use += instead of =,
            a.grad = 0
            a.grad += o.grad * 1 = 1
            a.grad += o.grad * 1 = 2

            a.grad = 2 which is correct

            """
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')
        
        def _backward():
            """
            Let's say we have an express o = a * b
            We want to find the derivative of o with respect to a (do/da)
            Using the definition of the derivative: d/dx f(x) as lim h->0 = (f(x+h) - f(x)) / h 
            
            do/da = (((a + h) * b) - (ab)) / h
            => (ab + bh - ab) / h
            => bh / h
            => b

            Similarly, do/db = ((a * (b + h)) - (ab)) / h
            => (ab + ah - ab) / h
            => ah / h
            => a

            So using chain rule, we get:
            do/da = b
            do/db = a

            Suppose L is the loss and o is one of it's intermediate nodes in the computation graph.
            We can find the gradient of L with respect to a and b, using the chain rule.
            Using the chain rule, we get:
            dL/da = (dL/do) * (do/da)
            dL/da = (dL/do) * b

            and,

            dL/db = (dL/do) * (do/db)
            dL/db = (dL/do) * a

            substituting, L = out.grad, a = self.data, b = other.data,
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data            

            """
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def backward(self):
        """
        Performs topological sort on DAG and then performs backpropagation.

        Depth first search is used to perform topological sort.

        If node is visited, do nothing
        Otherwise, mark node as visited and for each child node, recursively visit the child node
        Add from leaf to root

        Time Complexity O(V + E), where V is the number of nodes and E is the number of edges
        Space Complexity O(V)

        Then it calculates the gradients of each node using the _backward function in reverse order.

        """
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

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int or float for now"
        out = Scalar(self.data ** other, (self, ), f'^{other}')

        def _backward():
            # d/dx x^y = y * x^(y-1)
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Scalar(math.exp(self.data), (self, ), 'exp')

        def _backward():
            # d/dx e ^ x = e ^ x but we already have e ^ x stored in out.data
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Scalar(math.log(self.data), (self, ), 'log')

        def _backward():
            # d/dx log(x) = 1/x
            self.grad += 1 / self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """
        ReLU(x) = max(0, x) = {
            x   if x > 0
            0   if x ≤ 0
        }

        d/dx ReLU(x) = {
            1   if x > 0 # since d/dx x = 1
            0   if x ≤ 0 # since d/dx 0 = 0
        }
        """
        out = Scalar(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """
        This is also used as a activation function. It is a smooth, non-linear function that maps real-valued numbers to the range (-1, 1).
        An activation function is a function that is applied to the output of a neuron in a neural network. It is used to introduce non-linearity into the network, which allows it to learn complex patterns in the data.
        Intuition behind using non-linear functions is that straght lines may not be enough to learn complex patterns in real-world data. By allowing the possibility to have curves, we can learn more complex patterns.
        See: https://en.wikipedia.org/wiki/Hyperbolic_functions
        Hyperbolic tangent is defined as:
        tanh(x) = sinh(x) / cosh(x) = (e^x - e^-x) / (e^x + e^-x) = (e^2x - 1) / (e^2x + 1)
        """
        tanh_val = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Scalar(tanh_val, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - tanh_val**2) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        # if a + b fails, then it tries b + a
        return self + other
    
    def __rmul__(self, other):
        # if a * b fails, then it tries b * a
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        # if a - b fails, then it tries b - a
        return (-self) + other

    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        # if a / b fails, then it tries b / a
        return (other ** -1) * self

    def __repr__(self):
        return f"scalar({self.data:.4f})"