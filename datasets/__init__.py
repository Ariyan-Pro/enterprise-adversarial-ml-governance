"""
Datasets module for adversarial ML security suite
"""

# Import dataset loaders
from .mnist import load_mnist, get_mnist_stats
from .fashion_mnist import load_fashion_mnist, get_fashion_mnist_stats, FASHION_CLASSES

# Import registry functions
try:
    from .dataset_registry import (
        get_dataset, 
        get_dataset_info, 
        list_datasets, 
        get_dataset_stats, 
        load_dataset, 
        get_num_classes, 
        get_input_shape
    )
except ImportError as e:
    print(f"Warning: Could not import dataset registry: {e}")
    
    # Define fallback functions
    def get_dataset(name, **kwargs):
        if name == "mnist":
            return load_mnist(**kwargs)
        elif name == "fashion_mnist":
            return load_fashion_mnist(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    def list_datasets():
        return ["mnist", "fashion_mnist"]
    
    def get_dataset_info(name):
        if name == "mnist":
            return {"num_classes": 10, "input_shape": (1, 28, 28), "description": "MNIST digits"}
        elif name == "fashion_mnist":
            return {"num_classes": 10, "input_shape": (1, 28, 28), "description": "Fashion-MNIST"}
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    def load_dataset(*args, **kwargs):
        return get_dataset(*args, **kwargs)
    
    def get_num_classes(name):
        info = get_dataset_info(name)
        return info.get("num_classes", 10)
    
    def get_input_shape(name):
        info = get_dataset_info(name)
        return info.get("input_shape", (1, 28, 28))
    
    def get_dataset_stats(name, dataset):
        if name == "mnist":
            return get_mnist_stats(dataset)
        elif name == "fashion_mnist":
            return get_fashion_mnist_stats(dataset)
        else:
            return {"size": len(dataset)}

__all__ = [
    'load_mnist',
    'get_mnist_stats',
    'load_fashion_mnist',
    'get_fashion_mnist_stats',
    'FASHION_CLASSES',
    'get_dataset',
    'get_dataset_info',
    'list_datasets',
    'get_dataset_stats',
    'load_dataset',
    'get_num_classes',
    'get_input_shape'
]
