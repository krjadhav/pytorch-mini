from mnist_util import load_mnist_data

import torch
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameters
HIDDEN_LAYERS = 128
LR = 0.01
BATCH_SIZE = 128
EPOCHS = 10

def pytorch_train(model, train_dataloader, optimizer, epoch, loss_function):
    model.train()
    epoch_losses = []
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to("cpu"), target.to("cpu")
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        print(f"Train Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
 
def pytorch_test(model, test_dataloader, loss_function):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to("cpu"), target.to("cpu")
            output = model(data)
            test_loss += loss_function(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({100. * correct / len(test_dataloader.dataset):.0f}%)\n")

def run_pytorch_training(HIDDEN_LAYERS, LR, BATCH_SIZE, EPOCHS):
    # ========================
    # Prepare data
    # ========================
    # Convert to torch tensor
    # Here we're also changing the shape from (60000, 28, 28) to (60000, 784) and normalizing it
    # This is because a linear layer only takes in a 1D vector.
    x_train = torch.tensor(X_train, dtype=torch.float32).flatten(start_dim=1) / 255
    y_train = torch.tensor(Y_train, dtype=torch.long)
    x_test = torch.tensor(X_test, dtype=torch.float32).flatten(start_dim=1) / 255
    y_test = torch.tensor(Y_test, dtype=torch.long)

    # Dataset 
    # They manage the samples and labels
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # Dataloaders
    # This provides an iterable over the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

    # ========================
    # Define Model
    # ========================
    model = torch.nn.Sequential(
        torch.nn.Linear(784, HIDDEN_LAYERS),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_LAYERS, 10)
    ).to("cpu")

    # Set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Set loss function
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")

    # ========================
    # Training
    # ========================
    for epoch in range(1, EPOCHS + 1):
        pytorch_train(model, train_dataloader, optimizer, epoch, loss_function)
        pytorch_test(model, test_dataloader, loss_function)


if __name__ == "__main__":
    # Load MNIST data
    # _train is the training data
    # _test is the test/validation data
    # X is the input data
    # Y is the label data
    X_train, Y_train, X_test, Y_test = load_mnist_data()
    run_pytorch_training(HIDDEN_LAYERS=HIDDEN_LAYERS, LR=LR, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS)
    