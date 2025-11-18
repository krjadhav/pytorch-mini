import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

from mnist_util import load_mnist_data
from minitorch.dataloaders import DataLoader as MiniDataLoader


def test_minitorch_dataloader_vs_torch_mnist():
    X_train, Y_train, X_test, Y_test = load_mnist_data()

    # Use a subset for speed
    n_samples = 1024
    batch_size = 128
    shuffle = False

    X_subset = X_train[:n_samples]
    Y_subset = Y_train[:n_samples]

    # Normalize and flatten data in the same way for both loaders
    X_subset_flat = X_subset.reshape(n_samples, -1) / 255.0

    # MiniTorch DataLoader works directly with NumPy arrays
    mini_loader = MiniDataLoader(X_subset_flat, Y_subset, batch_size=batch_size, shuffle=shuffle)

    # Torch DataLoader uses TensorDataset
    x_tensor = torch.tensor(X_subset_flat, dtype=torch.float32)
    y_tensor = torch.tensor(Y_subset, dtype=torch.long)
    torch_dataset = TensorDataset(x_tensor, y_tensor)
    torch_loader = TorchDataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)

    print("\n=== MiniTorch DataLoader vs PyTorch DataLoader on MNIST (subset) ===")
    print(f"Total samples: {n_samples}, batch_size: {batch_size}, shuffle: {shuffle}")
    print(f"MiniTorch DataLoader num_batches: {len(mini_loader)}")
    print(f"PyTorch DataLoader   num_batches: {len(torch_loader)}")

    mini_iter = iter(mini_loader)
    torch_iter = iter(torch_loader)

    num_batches_to_show = 3
    for b in range(num_batches_to_show):
        try:
            mini_x, mini_y = next(mini_iter)
            torch_x, torch_y = next(torch_iter)
        except StopIteration:
            break

        print(f"\nBatch {b}:")
        print("  MiniTorch - data shape:", mini_x.shape, "labels[:10]:", mini_y[:10].tolist())
        print("  PyTorch   - data shape:", tuple(torch_x.shape), "labels[:10]:", torch_y[:10].tolist())


if __name__ == "__main__":
    test_minitorch_dataloader_vs_torch_mnist()