"""
MNIST Dataset Loader
"""
import torch
from torchvision import datasets, transforms
from pathlib import Path
import pickle
import hashlib

def load_mnist(root="data/raw/mnist", cache=True, augment=False):
    """
    Load MNIST dataset
    
    Args:
        root: Root directory for data
        cache: Whether to cache processed data
        augment: Whether to apply data augmentation
    
    Returns:
        train_set, test_set: PyTorch datasets
    """
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    cache_key = f"mnist_augment_{augment}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_file = processed_dir / f"mnist_{cache_hash}.pkl"
    
    if cache and cache_file.exists():
        print(f"Loading cached MNIST dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Downloading/processing MNIST dataset...")
    
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ] + base_transforms)
    else:
        train_transform = transforms.Compose(base_transforms)
    
    test_transform = transforms.Compose(base_transforms)
    
    train_set = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_set = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    if cache:
        with open(cache_file, 'wb') as f:
            pickle.dump((train_set, test_set), f)
        print(f"Cached MNIST dataset to {cache_file}")
    
    return train_set, test_set

def get_mnist_stats(dataset):
    """Get MNIST dataset statistics"""
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, targets = next(iter(loader))
    
    return {
        'size': len(dataset),
        'mean': data.mean().item(),
        'std': data.std().item(),
        'class_distribution': torch.bincount(targets).tolist()
    }
