"""
Fashion-MNIST Dataset Loader
10-class fashion items dataset - More challenging than MNIST
"""
import torch
from torchvision import datasets, transforms
from pathlib import Path
import pickle
import hashlib

def load_fashion_mnist(root="data/raw/fashion_mnist", cache=True, augment=False):
    """
    Load Fashion-MNIST dataset
    
    Args:
        root: Root directory for data
        cache: Whether to cache processed data
        augment: Whether to apply data augmentation
    
    Returns:
        train_set, test_set: PyTorch datasets
    """
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    cache_key = f"fashion_mnist_augment_{augment}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_file = processed_dir / f"fashion_mnist_{cache_hash}.pkl"
    
    if cache and cache_file.exists():
        print(f"Loading cached Fashion-MNIST dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Downloading/processing Fashion-MNIST dataset...")
    
    # Fashion-MNIST normalization (different from MNIST)
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean/std
    ]
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ] + base_transforms)
    else:
        train_transform = transforms.Compose(base_transforms)
    
    test_transform = transforms.Compose(base_transforms)
    
    train_set = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_set = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    if cache:
        with open(cache_file, 'wb') as f:
            pickle.dump((train_set, test_set), f)
        print(f"Cached Fashion-MNIST dataset to {cache_file}")
    
    return train_set, test_set

def get_fashion_mnist_stats(dataset):
    """Get Fashion-MNIST dataset statistics"""
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, targets = next(iter(loader))
    
    return {
        'size': len(dataset),
        'mean': data.mean().item(),
        'std': data.std().item(),
        'class_distribution': torch.bincount(targets).tolist(),
        'classes': [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    }

# Class names for Fashion-MNIST
FASHION_CLASSES = [
    'T-shirt/top',
    'Trouser', 
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]
