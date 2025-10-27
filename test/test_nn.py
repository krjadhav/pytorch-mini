from minitorch.nn import Neuron, Layer, MLP
import matplotlib.pyplot as plt


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

def test_nn():
    print("Running test_nn")
    input_data = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 1.0],
        [1.0, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ]
    desired_targets = [1.0, -1.0, -1.0, 1.0]
    
    # Predict with MLP
    nn = MLP(3, [4, 4, 1])
    pred = [nn(x) for x in input_data]
    print(pred)

    # Calculate loss
    loss = sum((p - t) ** 2 for p, t in zip(pred, desired_targets))
    print(f"loss: {loss}, loss._prev: {loss._prev}")

    # Backpropagate
    loss.backward()
    # print(nn.layers[0].neurons[0].weights[0].grad) # gradients on the input data doesn't matter because they don't change. It's probably useless at this point.
    
    # Show parameters
    params = nn.parameters()
    print(f"Total params = {len(params)} | parameters = {params}")

    # Perform Gradient Descent
    lr = 0.01 # learning rate 
    for p in params:
        # Gradients always point in the direction of steepest ascent (ie. maximize loss)
        # So we need to subtract to move in the direction of steepest descent
        p.data -= lr * p.grad
    
    # Predict with MLP
    print("====")
    print("Updated Loss")
    pred = [nn(x) for x in input_data]
    loss = sum((p - t) ** 2 for p, t in zip(pred, desired_targets))
    print(f"loss: {loss}, loss._prev: {loss._prev}")
    print(f"Total params = {len(params)} | parameters = {params}")

def test_mintorch_e2e():
    print("Running test_mintorch_e2e")
    desired_targets = [-1.0, 1.0, -1.0, 1.0]
    print(f"Desired targets: {desired_targets}")
    model = MLP(3, [4, 4, 1])
    input_data = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 1.0],
        [1.0, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ]
    
    lr = 0.01 # learning rate
    i = 0
    loss_history = []  # Track loss over time
    while True:
        # Zero gradients. This is important otherwise gradients from previous iterations will accumulate
        # and you will see jagged loss curves.
        # It does seem like this is not required for this example because it's a simple dataset and the accumulation of gradients does help in this case
        # However on larger datasets it leads to incorrect updates and poor learning
        for p in model.parameters():
            p.grad = 0

        # Forward pass
        pred = [model(x) for x in input_data]
        loss = sum((p - t) ** 2 for p, t in zip(pred, desired_targets))

        # Backpropagate
        loss.backward()
        
        # Update weights
        for p in model.parameters():
            p.data -= lr * p.grad
        
        print(f"iter {i}: | Loss: {loss} | pred: {pred}")
        loss_history.append(loss.data)  # Store loss value
        i += 1

        if loss.data <= 0.01:
            break
    
    # Plot the loss over time
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    plt.tight_layout()
    plt.show()

def test_pytorch_e2e():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    print("Running test_pytorch_e2e")
    
    # Convert data to PyTorch tensors
    input_data = torch.tensor([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 1.0],
        [1.0, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    
    # This converts the list of desired targets to a tensor
    desired_targets = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float32).unsqueeze(1)
    print(f"Desired targets: {desired_targets.squeeze().tolist()}")
    
    # Define MLP in PyTorch (3 inputs -> 4 hidden -> 4 hidden -> 1 output)
    """
    - nn.Sequential is a container that allows you to stack multiple layers of a neural network. (https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
    - nn.Linear is let's you do y = mx + b (https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.linear.Linear.html)    
    - nn.Tanh is let's you do y = tanh(x) (https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.Tanh.html)    
    """
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
        nn.Tanh()
    )
    
    # Define optimizer (SGD with learning rate 0.01)
    lr = 0.01
    # Implements Stochastic Gradient Descent on params (https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Training loop
    i = 0
    loss_history = []
    
    while True:
        # Zero gradients (https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)
        optimizer.zero_grad()
        
        # Forward pass (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward)
        pred = model(input_data)
        
        # Calculate MSE loss (https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
        loss = ((pred - desired_targets) ** 2).sum()
        
        # Backpropagate
        loss.backward()
        
        # Update weights (https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html)
        optimizer.step()
        
        print(f"iter {i}: | Loss: {loss.item():.6f} | pred: {pred.squeeze().tolist()}")
        loss_history.append(loss.item())
        i += 1
        
        if loss.item() <= 0.01:
            break
    
    # Plot the loss over time
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('PyTorch Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # test_neuron()
    # test_layer()
    # test_mlp()
    # test_nn()
    # test_nn_e2e()
    test_pytorch_e2e()