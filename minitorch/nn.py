from minitorch.tensor import Tensor

class Linear:
    """
    Implements y = mx + b but with tensors. This is called an affline linear transformation.
    """
    def __init__(self, num_inputs, num_outputs):
        # Initialize weights with Kaiming He uniform initialization
        self.weights = Tensor.kaiming_uniform(num_inputs, num_outputs)
        self.bias = Tensor.uniform(num_inputs, num_outputs)