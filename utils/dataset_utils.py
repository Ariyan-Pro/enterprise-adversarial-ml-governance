"""
Dataset Utilities - Bridge to new datasets module
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def load_mnist(data_dir="data/raw/mnist", cache=True, augment=False):
    """Load MNIST dataset (redirects to datasets module)"""
    from datasets.mnist import load_mnist as load_mnist_impl
    return load_mnist_impl(root=data_dir, cache=cache, augment=augment)

def get_dataset_stats(dataset):
    """Get dataset statistics (redirects to appropriate dataset module)"""
    from datasets.mnist import get_mnist_stats
    return get_mnist_stats(dataset)

def create_dataloaders(train_set, test_set, batch_size=64, val_split=0.1):
    """
    Create train/validation/test dataloaders
    
    Args:
        train_set: Training dataset
        test_set: Test dataset
        batch_size: Batch size
        val_split: Fraction of training data for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    val_size = int(len(train_set) * val_split)
    train_size = len(train_set) - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader
