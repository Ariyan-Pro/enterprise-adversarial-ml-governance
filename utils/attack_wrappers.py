"""
Attack wrapper utilities for compatibility
"""
import torch
import torch.nn as nn
from typing import Dict, Any

def create_fgsm_attack(model: nn.Module, epsilon: float = 0.3, **kwargs):
    """Create FGSM attack with flexible parameters"""
    from attacks.fgsm import FGSMAttack
    
    try:
        # Try with config dict
        config = {'epsilon': epsilon, **kwargs}
        return FGSMAttack(model, config)
    except TypeError:
        try:
            # Try without config
            return FGSMAttack(model)
        except Exception as e:
            print(f"Failed to create FGSM attack: {e}")
            return None

def create_fast_cw_attack(model: nn.Module, const: float = 1.0, iterations: int = 50, **kwargs):
    """Create fast C&W attack with flexible parameters"""
    from attacks.cw import FastCarliniWagnerL2
    
    try:
        # Try with config dict
        config = {'const': const, 'iterations': iterations, **kwargs}
        return FastCarliniWagnerL2(model, config)
    except TypeError:
        try:
            # Try without parameters
            return FastCarliniWagnerL2(model)
        except Exception as e:
            print(f"Failed to create C&W attack: {e}")
            return None

def create_pgd_attack(model: nn.Module, epsilon: float = 0.3, alpha: float = 0.01, steps: int = 40, **kwargs):
    """Create PGD attack with flexible parameters"""
    from attacks.pgd import PGDAttack
    
    try:
        # Try with config dict
        config = {'epsilon': epsilon, 'alpha': alpha, 'steps': steps, **kwargs}
        return PGDAttack(model, config)
    except TypeError:
        try:
            # Try without parameters
            return PGDAttack(model)
        except Exception as e:
            print(f"Failed to create PGD attack: {e}")
            return None
