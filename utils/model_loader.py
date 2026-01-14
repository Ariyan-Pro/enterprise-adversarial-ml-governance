"""
Model loading utilities with compatibility fixes
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

def load_model_weights(model: nn.Module, model_path: str) -> bool:
    """
    Load model weights with compatibility handling
    
    Args:
        model: Model instance
        model_path: Path to model file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # New format with metadata
                state_dict = checkpoint['state_dict']
                # Remove 'module.' prefix if present (for DataParallel)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print(f"Loaded model from checkpoint with metadata")
                return True
            elif 'model_state_dict' in checkpoint:
                # Alternative format
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint with model_state_dict")
                return True
            else:
                # Assume it's a state dict
                try:
                    model.load_state_dict(checkpoint)
                    print(f"Loaded model from state dict")
                    return True
                except:
                    # Try with strict=False
                    model.load_state_dict(checkpoint, strict=False)
                    print(f"Loaded model with strict=False (some keys missing)")
                    return True
        else:
            # Assume it's a state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded model directly")
            return True
            
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return False

def load_model_with_flexibility(model: nn.Module, model_path: str) -> bool:
    """
    Load model weights with flexibility for size mismatches
    
    Args:
        model: Model instance
        model_path: Path to model file
    
    Returns:
        True if successful (with warnings), False if failed
    """
    try:
        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Get current model state dict
        model_dict = model.state_dict()
        
        # Filter out incompatible keys
        filtered_state_dict = {}
        missing_keys = []
        unexpected_keys = []
        size_mismatches = []
        
        for k, v in state_dict.items():
            if k in model_dict:
                if v.size() == model_dict[k].size():
                    filtered_state_dict[k] = v
                else:
                    size_mismatches.append((k, v.size(), model_dict[k].size()))
            else:
                unexpected_keys.append(k)
        
        # Check for missing keys in state_dict
        for k in model_dict.keys():
            if k not in state_dict:
                missing_keys.append(k)
        
        # Load filtered state dict
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict, strict=False)
        
        # Print warnings
        if size_mismatches:
            print(f"⚠️  Size mismatches ({len(size_mismatches)}):")
            for k, saved_size, current_size in size_mismatches[:3]:  # Show first 3
                print(f"    {k}: saved {saved_size} != current {current_size}")
            if len(size_mismatches) > 3:
                print(f"    ... and {len(size_mismatches) - 3} more")
        
        if missing_keys:
            print(f"⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:5]}")
            if len(missing_keys) > 5:
                print(f"    ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}")
            if len(unexpected_keys) > 5:
                print(f"    ... and {len(unexpected_keys) - 5} more")
        
        if filtered_state_dict:
            print(f"✅ Loaded {len(filtered_state_dict)}/{len(model_dict)} parameters")
            return True
        else:
            print("❌ No parameters loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def create_and_load_model(model_class, model_path: str, **kwargs) -> Optional[nn.Module]:
    """
    Create model and load weights
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to model weights
        **kwargs: Arguments for model constructor
    
    Returns:
        Loaded model or None
    """
    try:
        model = model_class(**kwargs)
        if load_model_with_flexibility(model, model_path):
            model.eval()
            return model
        return None
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def save_model_with_metadata(model: nn.Module, model_path: str, metadata: Dict[str, Any] = None):
    """
    Save model with metadata
    
    Args:
        model: Model to save
        model_path: Path to save to
        metadata: Additional metadata
    """
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'metadata': metadata or {}
    }
    
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path} with metadata")
