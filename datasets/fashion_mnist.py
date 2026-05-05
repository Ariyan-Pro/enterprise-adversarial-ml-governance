"""
Fashion-MNIST Dataset Loader - No torchvision dependency
"""
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
import pickle
import hashlib
import urllib.request
import gzip
import struct
import numpy as np

FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def download_fashion_mnist(data_dir='data/raw/fashion_mnist'):
    """Download Fashion-MNIST dataset manually without torchvision"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist"
    files = {
        'train-images': ('train-images-idx3-ubyte.gz',),
        'train-labels': ('train-labels-idx1-ubyte.gz',),
        'test-images': ('t10k-images-idx3-ubyte.gz',),
        'test-labels': ('t10k-labels-idx1-ubyte.gz',)
    }
    
    downloaded_files = {}
    for key, (filename,) in files.items():
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            url = f"{base_url}/{filename}"
            urllib.request.urlretrieve(url, filepath)
        
        extracted_path = Path(data_dir) / filename.replace('.gz', '')
        if not extracted_path.exists():
            with gzip.open(filepath, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        
        downloaded_files[key] = extracted_path
    
    return downloaded_files

def parse_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def parse_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def create_mnist_dataset(images, labels):
    images = images.astype(np.float32) / 255.0
    images = (images - 0.1307) / 0.3081
    images = torch.from_numpy(images).unsqueeze(1)
    labels = torch.from_numpy(labels).long()
    return TensorDataset(images, labels)

def load_fashion_mnist(root="data/raw/fashion_mnist", cache=True):
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_file = processed_dir / "fashion_mnist.pkl"

    if cache and cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Downloading/processing Fashion-MNIST dataset...")
    files = download_fashion_mnist(root)
    
    train_images = parse_mnist_images(files['train-images'])
    train_labels = parse_mnist_labels(files['train-labels'])
    test_images = parse_mnist_images(files['test-images'])
    test_labels = parse_mnist_labels(files['test-labels'])
    
    train_set = create_mnist_dataset(train_images, train_labels)
    test_set = create_mnist_dataset(test_images, test_labels)

    if cache:
        with open(cache_file, 'wb') as f:
            pickle.dump((train_set, test_set), f)

    return train_set, test_set

def get_fashion_mnist_stats(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, targets = next(iter(loader))
    return {
        'size': len(dataset),
        'mean': data.mean().item(),
        'std': data.std().item(),
        'class_distribution': torch.bincount(targets).tolist()
    }
