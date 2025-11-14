from minitorch.tensor import Tensor

class Linear:
    """
    Implements y = mx + b but with tensors. This is called an affline linear transformation.
    """
    def __init__(self, num_inputs, num_outputs):
        # Initialize weights with Kaiming He uniform initialization
        self.weights = Tensor.kaiming_uniform(num_inputs, num_outputs)
        self.bias = Tensor.uniform(num_inputs, num_outputs)

    def parameters(self):
        return [self.weights, self.bias]
    
    def __call__(self, x):
        return Tensor.linear(x, self.weights, self.bias)


class ReLU:
    """Stateless ReLU activation usable within Sequential containers."""

    def parameters(self):
        return []

    def __call__(self, x):
        return Tensor.relu(x)

class Sequential:
    """Chains layers so that output of each feeds the next."""

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]