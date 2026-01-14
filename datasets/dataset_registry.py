"""
Simple Dataset Registry
"""
from .mnist import load_mnist, get_mnist_stats
from .fashion_mnist import load_fashion_mnist, get_fashion_mnist_stats, FASHION_CLASSES

DATASET_REGISTRY = {
    "mnist": {
        "loader": load_mnist,
        "stats": get_mnist_stats,
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "mean": 0.1307,
        "std": 0.3081,
        "classes": [str(i) for i in range(10)],
        "description": "Handwritten digits dataset"
    },
    "fashion_mnist": {
        "loader": load_fashion_mnist,
        "stats": get_fashion_mnist_stats,
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "mean": 0.2860,
        "std": 0.3530,
        "classes": FASHION_CLASSES,
        "description": "Fashion items dataset"
    }
}

def get_dataset(name, **kwargs):
    """Get dataset by name"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    loader = DATASET_REGISTRY[name]["loader"]
    return loader(**kwargs)

def get_dataset_info(name):
    """Get dataset metadata"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name].copy()

def list_datasets():
    """List all available datasets"""
    return list(DATASET_REGISTRY.keys())

def get_dataset_stats(name, dataset):
    """Get statistics for a dataset"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    
    stats_func = DATASET_REGISTRY[name]["stats"]
    return stats_func(dataset)

# Aliases for backward compatibility
load_dataset = get_dataset
get_num_classes = lambda name: DATASET_REGISTRY[name]["num_classes"]
get_input_shape = lambda name: DATASET_REGISTRY[name]["input_shape"]
