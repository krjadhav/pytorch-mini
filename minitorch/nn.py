import random
from minitorch.engine import Tensor

class Neuron:
    def __init__(self, num_inputs):
        """
        Artificial Neuron Structure: https://en.wikipedia.org/wiki/Artificial_neuron#/media/File:Artificial_neuron_structure.svg
        Similar to neuron:
        - takes in multiple inputs (x1, x2, x3, ... xn from dendrites)
        - processes the signal (by placing importance on certain inputs using "weights" and adjusts the threshold using a "bias" or just think about it as a linear function y = mx + b)
        - has an activation function that fires if the combined signal exceeds the threshold and changes the shape of the curve to learn more complex decisions.
        """
        self.weights = [Tensor(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Tensor(random.uniform(-1, 1))
    
    def __call__(self, x):
        """
        This means we can call the neuron as a function. Eg.
        n = Neuron(2)
        n(x)
        Here x is the input to the neuron.
        """
        activation = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return activation.tanh() # using tanh for now but this can be changed to any activation function
    
class Layer:
    """
    This sets how many neurons are in a layer.
    A neuron in a layer is connected to all the neurons in the previous layer
    but not to the neurons in the same layer.
    So it's essentially an array of neurons.
    """
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        """
        This means we can call the layer as a function. Eg.
        l = Layer(2, 3)
        l(x)
        Here x is the input to the layer.
        """
        outs = [n(x) for n in self.neurons]
        return outs if len(outs) > 1 else outs[0] # return the list if it has more than one element else return the first element

class MLP:
    def __init__(self, num_inputs, num_outputs):
        """
        num_inputs: number of inputs to the first layer
        num_outputs: list of number of outputs for each layer

        Let's say the MLP was this: https://share.google/cm87QPLhuy3fOFjZX
        - The input layer has 3 neurons
        - The first hidden layer has 4 neurons
        - The second hidden layer has 4 neurons
        - The output layer has 1 neuron
        
        Then sizes = [3, 4, 4, 1]. So we create layers as follows:
        - We don't create the input layer because it's just the input to the first layer
        - Layer(3, 4) # 3 inputs, 4 outputs
        - Layer(4, 4) # 4 inputs, 4 outputs
        - Layer(4, 1) # 4 inputs, 1 output
        """
        sizes = [num_inputs] + num_outputs # 3 + [4, 4, 1] = [3, 4, 4, 1]
        self.layers = [Layer(size, sizes[i + 1]) for i, size in enumerate(sizes[:-1])]

    def __call__(self, x):
        """
        This means we can call the MLP as a function. Eg.
        mlp = MLP(3, [4, 4, 1])
        mlp(x)
        Here x is the input to the MLP.
        Let's say the MLP was this: https://share.google/cm87QPLhuy3fOFjZX
        Then, it starts a sequential transformation pipeline where
        Raw input (size 3) -> Layer 1 -> tensor list (size 4) -> Layer 2 -> tensor list (size 4) -> Layer 3 (size 1) -> final output (size 1)
        """
        for layer in self.layers:
            x = layer(x)
        return x
