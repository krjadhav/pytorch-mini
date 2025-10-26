from minitorch.nn import Neuron, Layer, MLP


def test_neuron():
    print("Running test_neuron")
    n = Neuron(2)
    x = [2.0, 3.0]
    print(n(x))
    
def test_layer():
    print("Running Test Layer")
    inputs = [2.0, 3.0]
    layer = Layer(2, 3)
    for i, n in enumerate(layer.neurons):
        print(f"Neuron {i} | weights={n.weights} | biases={n.bias}")

    outputs = layer(inputs)
    print("=====")
    for i, n in enumerate(outputs):
        print(f"Neuron {i} | output={n}")

def test_mlp():
    print("Running test_mlp")
    inputs = [2.0, 3.0, 4.0]
    mlp = MLP(3, [4, 4, 1])
    print(mlp(inputs))            

if __name__ == "__main__":
    # test_neuron()
    # test_layer()
    # test_mlp()