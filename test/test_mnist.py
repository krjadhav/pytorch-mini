import time
from mnist_util import load_mnist_data
# torch imports
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

# minitorch imports
import minitorch
from minitorch.tensor import Tensor
from minitorch.dataloaders import DataLoader as MinitorchDataLoader


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
        # print(f"Train Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

def minitorch_train(model, train_dataloader, optimizer, epoch, loss_function):
    epoch_losses = []
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        # print(f"Train Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

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

    num_samples = len(test_dataloader.dataset)
    if num_samples > 0:
        test_loss /= num_samples
        accuracy = 100.0 * correct / num_samples
    else:
        accuracy = 0.0

    # print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_samples} ({accuracy:.0f}%)\n")

    return test_loss, accuracy, correct, num_samples

def minitorch_test(model, test_dataloader, loss_function):
    test_loss, correct, total = 0.0, 0, 0
    for data, target in test_dataloader:
        output = model(data)
        test_loss += loss_function(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()        
        total += len(target)

    avg_loss = test_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    # print(f"\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n")

    return avg_loss, accuracy, correct, total

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
    train_dataloader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = TorchDataLoader(test_dataset, batch_size=10000, shuffle=False)

    # ========================
    # Define Model
    # ========================
    # nn.Sequential (https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
    # is a container that let you build nueral networks by stacking layers
    # in a specific order such that the output of one layer is the input of the next layer
    # 
    # A module (https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
    # It's like a layer with many neurons or activation function.
    model = torch.nn.Sequential(
        torch.nn.Linear(784, HIDDEN_LAYERS),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_LAYERS, 10)
    ).to("cpu")

    # Set optimizer
    # Stochastic Gradient Descent (SGD) https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Set loss function
    # Cross Entropy Loss (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
    # Cross-entropy loss compares a model’s predictions against the actual class using logarithmic scaling.
    # It assigns a high penalty for confident and wrong predictions, a moderate penalty for unconfident wrong predictions, and a low penalty for confident and correct predictions.    # 
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    # loss_function = torch.nn.MultiMarginLoss(reduction="sum") # This gives ~98% accuracy

    # ========================
    # Training
    # ========================
    start_time = time.time()
    epoch_metrics = []
    for epoch in range(1, EPOCHS + 1):
        pytorch_train(model, train_dataloader, optimizer, epoch, loss_function)
        avg_loss, accuracy, correct, total = pytorch_test(model, test_dataloader, loss_function)
        epoch_metrics.append(
            {
                "epoch": epoch,
                "test_loss": avg_loss,
                "test_accuracy": accuracy,
            }
        )

    total_time = time.time() - start_time
    if epoch_metrics:
        final_loss = epoch_metrics[-1]["test_loss"]
        final_accuracy = epoch_metrics[-1]["test_accuracy"]
    else:
        final_loss = 0.0
        final_accuracy = 0.0

    return {
        "framework": "PyTorch",
        "epochs": EPOCHS,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "total_time": total_time,
        "epoch_metrics": epoch_metrics,
    }


def run_minitorch_training(HIDDEN_LAYERS, LR, BATCH_SIZE, EPOCHS):
    # create tensors
    x_train = Tensor(X_train/255).flatten(start_dim=1)
    y_train = Tensor(Y_train)
    x_test = Tensor(X_test/255).flatten(start_dim=1)
    y_test = Tensor(Y_test)

    # load dataset
    train_dataloader = MinitorchDataLoader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = MinitorchDataLoader(x_test, y_test, batch_size=10000, shuffle=False)

    # define model
    model = minitorch.nn.Sequential(
        minitorch.nn.Linear(784, HIDDEN_LAYERS),
        minitorch.nn.ReLU(),
        minitorch.nn.Linear(HIDDEN_LAYERS, 10)
    )

    # define optimizer
    optimizer = minitorch.optim.SGD(model.parameters(), lr=LR)

    # define loss function
    loss_function = minitorch.loss.CrossEntropyLoss(reduction="sum")

    # train
    start_time = time.time()
    epoch_metrics = []
    for epoch in range(1, EPOCHS + 1):
        minitorch_train(model, train_dataloader, optimizer, epoch, loss_function)
        avg_loss, accuracy, correct, total = minitorch_test(model, test_dataloader, loss_function)
        epoch_metrics.append(
            {
                "epoch": epoch,
                "test_loss": avg_loss,
                "test_accuracy": accuracy,
            }
        )

    total_time = time.time() - start_time
    if epoch_metrics:
        final_loss = epoch_metrics[-1]["test_loss"]
        final_accuracy = epoch_metrics[-1]["test_accuracy"]
    else:
        final_loss = 0.0
        final_accuracy = 0.0

    return {
        "framework": "Minitorch",
        "epochs": EPOCHS,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "total_time": total_time,
        "epoch_metrics": epoch_metrics,
    }
        

if __name__ == "__main__":
    # Load MNIST data
    # _train is the training data
    # _test is the test/validation data
    # X is the input data
    # Y is the label data
    X_train, Y_train, X_test, Y_test = load_mnist_data()
    pytorch_summary = run_pytorch_training(HIDDEN_LAYERS=HIDDEN_LAYERS, LR=LR, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS)
    minitorch_summary = run_minitorch_training(HIDDEN_LAYERS=HIDDEN_LAYERS, LR=LR, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS)

    print("=" * 72)
    print("MNIST Training Summary".center(72))
    print("=" * 72)
    header = f"{'Framework':<12} {'Final Acc (%)':>14} {'Final Loss':>14} {'Total Time (s)':>16}"
    print(header)
    print("-" * len(header))

    def print_summary_row(summary):
        framework = summary.get("framework", "")
        final_accuracy = summary.get("final_accuracy", 0.0)
        final_loss = summary.get("final_loss", 0.0)
        total_time = summary.get("total_time", 0.0)
        print(f"{framework:<12} {final_accuracy:>14.2f} {final_loss:>14.4f} {total_time:>16.2f}")

    print_summary_row(pytorch_summary)
    print_summary_row(minitorch_summary)
    print("=" * 72)