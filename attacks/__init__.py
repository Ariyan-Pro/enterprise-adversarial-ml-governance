"""
Attacks module for adversarial ML security suite
"""
from .fgsm import FGSMAttack
from .pgd import PGDAttack
from .deepfool import DeepFoolAttack
from .cw import CarliniWagnerL2, FastCarliniWagnerL2, create_cw_attack, create_fast_cw_attack

__all__ = [
    'FGSMAttack', 
    'PGDAttack', 
    'DeepFoolAttack',
    'CarliniWagnerL2',
    'FastCarliniWagnerL2',
    'create_cw_attack',
    'create_fast_cw_attack'
]
