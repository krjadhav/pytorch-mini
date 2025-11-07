import os
import gzip
import urllib.request
import numpy as np


def download_mnist():
    """Download MNIST dataset and save it in the test folder."""
    
    # Base URL for MNIST dataset
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    # Files to download
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mnist_dir = os.path.join(script_dir, "mnist_data")
    
    # Create directory if it doesn't exist
    os.makedirs(mnist_dir, exist_ok=True)
    
    # Check if all files already exist
    all_files_exist = all(os.path.exists(os.path.join(mnist_dir, f)) for f in files)
    
    if all_files_exist:
        print(f"All MNIST dataset files already exist in: {mnist_dir}")
        print("Skipping download.")
        return True
    
    print(f"Downloading MNIST dataset to: {mnist_dir}")
    
    for filename in files:
        filepath = os.path.join(mnist_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"File {filename} already exists, skipping download.")
            continue
        
        print(f"Downloading {filename}...")
        url = base_url + filename
        
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    
    print("\nAll files downloaded successfully!")
    print(f"MNIST data saved in: {mnist_dir}")
    return True


def load_mnist_images(filename):
    """Load MNIST images from a gzip file."""
    with gzip.open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        
    return data


def load_mnist_labels(filename):
    """Load MNIST labels from a gzip file."""
    with gzip.open(filename, 'rb') as f:
        # Read magic number and number of labels
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels


def load_mnist_data():
    """Load all MNIST data (train and test sets)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mnist_dir = os.path.join(script_dir, "mnist_data")
    
    # Download data if it doesn't exist
    if not os.path.exists(mnist_dir) or not all(
        os.path.exists(os.path.join(mnist_dir, f)) 
        for f in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                  "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    ):
        print("MNIST data not found. Downloading...")
        download_mnist()
    
    # Load training data
    train_images = load_mnist_images(
        os.path.join(mnist_dir, "train-images-idx3-ubyte.gz")
    )
    train_labels = load_mnist_labels(
        os.path.join(mnist_dir, "train-labels-idx1-ubyte.gz")
    )
    
    # Load test data
    test_images = load_mnist_images(
        os.path.join(mnist_dir, "t10k-images-idx3-ubyte.gz")
    )
    test_labels = load_mnist_labels(
        os.path.join(mnist_dir, "t10k-labels-idx1-ubyte.gz")
    )
    
    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    # Download the dataset
    success = download_mnist()
    
    if success:
        print("\n" + "="*50)
        print("Loading and verifying MNIST data...")
        print("="*50 + "\n")
        
        # Load and display info about the dataset
        (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
        
        print(f"\nDataset statistics:")
        print(f"  Training images shape: {train_images.shape}")
        print(f"  Training labels shape: {train_labels.shape}")
        print(f"  Test images shape: {test_images.shape}")
        print(f"  Test labels shape: {test_labels.shape}")
        print(f"  Pixel value range: [{train_images.min()}, {train_images.max()}]")
        print(f"  Unique labels: {np.unique(train_labels)}")
