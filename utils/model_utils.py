"""
Model utilities: loading, saving, evaluation, etc.
"""

import torch
import torch.nn as nn
import json
import yaml
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

def save_model(model: nn.Module, path: str, metadata: Optional[Dict] = None):
    """
    Save model with metadata
    
    Args:
        model: PyTorch model
        path: Path to save model
        metadata: Additional metadata to save
    """
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    torch.save({
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'metadata': metadata or {}
    }, path)
    
    # Save model card
    model_card = {
        'path': path,
        'model_class': model.__class__.__name__,
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'save_timestamp': str(datetime.now()),
        **metadata
    }
    
    model_card_path = Path(path).with_suffix('.json')
    with open(model_card_path, 'w') as f:
        json.dump(model_card, f, indent=2)

def load_model(path: str, model_class: Optional[nn.Module] = None, device: str = 'cpu'):
    """
    Load model with error handling
    
    Args:
        path: Path to saved model
        model_class: Model class (if None, tries to import from saved metadata)
        device: Device to load model on
    
    Returns:
        Loaded model and metadata
    """
    
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # FIX: Remove weights_only=True to handle numpy objects
    checkpoint = torch.load(path, map_location=device)  # No weights_only
    
    if model_class is None:
        # Try to import model class from base directory
        import sys
        sys.path.insert(0, 'models/base')
        try:
            module = __import__('mnist_cnn')
            model_class = getattr(module, checkpoint['model_class'])
        except ImportError:
            raise ValueError(f"Could not import model class: {checkpoint['model_class']}")
    
    model = model_class()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('metadata', {})

def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                   device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate model accuracy
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device for computation
    
    Returns:
        Dictionary of metrics
    """
    
    model.eval()
    correct = 0
    total = 0
    losses = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            losses.append(loss.item())
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = np.mean(losses)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total
    }

def get_model_summary(model: nn.Module) -> str:
    """Generate a summary of model architecture"""
    summary_lines = []
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if isinstance(module, nn.Conv2d):
                summary_lines.append(
                    f"{name}: Conv2d(in={module.in_channels}, out={module.out_channels}, "
                    f"kernel={module.kernel_size}, stride={module.stride})"
                )
            elif isinstance(module, nn.Linear):
                summary_lines.append(
                    f"{name}: Linear(in={module.in_features}, out={module.out_features})"
                )
    
    summary = "\n".join(summary_lines)
    summary += f"\n\nTotal parameters: {total_params:,}"
    summary += f"\nTrainable parameters: {trainable_params:,}"
    summary += f"\nNon-trainable parameters: {total_params - trainable_params:,}"
    
    return summary

def update_registry(model_name: str, path: str, metadata: Dict[str, Any]):
    """Update model registry"""
    registry_path = Path("models/registry.json")
    
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            try:
                registry = json.load(f)
            except json.JSONDecodeError:
                registry = {}
    else:
        registry = {}
    
    registry[model_name] = {
        'path': path,
        'input_size': '1x28x28',
        'num_classes': 10,
        'metadata': metadata,
        'timestamp': str(datetime.now())
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

# Keep the datetime import at the end
