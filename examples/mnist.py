import minitorch
from minitorch.tensor import Tensor
from minitorch.dataloaders import DataLoader

def train(model, train_dataloader, optimizer, epoch, loss_function):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

def test(model, test_dataloader, loss_function):
    test_loss, correct, total = 0.0, 0, 0
    for data, target in test_dataloader:
        output = model(data)
        test_loss += loss_function(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()        
        total += len(target)
    
    print(f"Average loss: {test_loss/total:.4f}, Accuracy: {100.0 * correct/total:.2f}%")

if __name__ == "__main__":
    # Load data
    X_train, Y_train, X_test, Y_test = minitorch.datasets.mnist()
    
    # create tensors
    x_train = Tensor(X_train/255).flatten(start_dim=1)
    y_train = Tensor(Y_train)
    x_test = Tensor(X_test/255).flatten(start_dim=1)
    y_test = Tensor(Y_test)
    
    # prepare dataloaders
    train_dataloader = DataLoader(x_train, y_train, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(x_test, y_test, batch_size=128, shuffle=False)

    # define model
    model = minitorch.nn.Sequential(
        minitorch.nn.Linear(784, 128),
        minitorch.nn.ReLU(),
        minitorch.nn.Linear(128, 64),
    )

    # define optimizer
    optimizer = minitorch.optim.SGD(model.parameters(), lr=0.01)

    # define loss function
    loss_function = minitorch.loss.CrossEntropyLoss(reduction="sum")
    
    # train
    for epoch in range(1, 11):
        train(model, train_dataloader, optimizer, epoch, loss_function)
        test(model, test_dataloader, loss_function)