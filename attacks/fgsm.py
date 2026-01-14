"""
Fast Gradient Sign Method (FGSM) Attack
Fixed device validation issue
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import numpy as np

class FGSMAttack:
    """FGSM attack with targeted/non-targeted variants"""
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FGSM attack
        
        Args:
            model: PyTorch model to attack
            config: Attack configuration dictionary
        """
        self.model = model
        self.config = config or {}
        
        # Default parameters
        self.epsilon = self.config.get('epsilon', 0.15)
        self.targeted = self.config.get('targeted', False)
        self.clip_min = self.config.get('clip_min', 0.0)
        self.clip_max = self.config.get('clip_max', 1.0)
        self.device = self.config.get('device', 'cpu')
        
        self.criterion = nn.CrossEntropyLoss()
        self.model.eval()
        self.model.to(self.device)
        
    def _validate_inputs(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Validate input tensors - FIXED: Remove strict device check"""
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be torch.Tensor, got {type(images)}")
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"labels must be torch.Tensor, got {type(labels)}")
        # FIX: Move to device instead of strict check
        if images.device != torch.device(self.device):
            images = images.to(self.device)
        if labels.device != torch.device(self.device):
            labels = labels.to(self.device)
        
    def generate(self, 
                 images: torch.Tensor, 
                 labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate adversarial examples
        
        Args:
            images: Clean images [batch, channels, height, width]
            labels: True labels for non-targeted attack
            target_labels: Target labels for targeted attack (optional)
            
        Returns:
            Adversarial images
        """
        # Move inputs to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        # Input validation
        self._validate_inputs(images, labels)
        
        # Setup targeted attack if specified
        if self.targeted and target_labels is None:
            raise ValueError("target_labels required for targeted attack")
        
        # Clone and detach for safety
        images = images.clone().detach()
        labels = labels.clone().detach()
        
        if target_labels is not None:
            target_labels = target_labels.clone().detach()
        
        # Enable gradient computation
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        
        # Loss calculation
        if self.targeted:
            # Targeted: maximize loss for target class
            loss = -self.criterion(outputs, target_labels)
        else:
            # Non-targeted: maximize loss for true class
            loss = self.criterion(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # FGSM update: x' = x + e * sign(?x J(?, x, y))
        perturbation = self.epsilon * images.grad.sign()
        
        # Generate adversarial examples
        if self.targeted:
            adversarial_images = images - perturbation  # Move away from true class
        else:
            adversarial_images = images + perturbation  # Move away from true class
        
        # Clip to valid range
        adversarial_images = torch.clamp(adversarial_images, self.clip_min, self.clip_max)
        
        return adversarial_images.detach()
    
    def attack_success_rate(self,
                           images: torch.Tensor,
                           labels: torch.Tensor,
                           adversarial_images: torch.Tensor) -> Dict[str, float]:
        """
        Calculate attack success metrics
        
        Args:
            images: Original images
            labels: True labels
            adversarial_images: Generated adversarial images
            
        Returns:
            Dictionary of metrics
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        adversarial_images = adversarial_images.to(self.device)
        
        with torch.no_grad():
            # Original predictions
            orig_outputs = self.model(images)
            orig_preds = orig_outputs.argmax(dim=1)
            orig_accuracy = (orig_preds == labels).float().mean().item()
            
            # Adversarial predictions
            adv_outputs = self.model(adversarial_images)
            adv_preds = adv_outputs.argmax(dim=1)
            
            # Attack success rate
            if self.targeted:
                success = (adv_preds == labels).float().mean().item()
            else:
                success = (adv_preds != labels).float().mean().item()
            
            # Confidence metrics
            orig_confidence = torch.softmax(orig_outputs, dim=1).max(dim=1)[0].mean().item()
            adv_confidence = torch.softmax(adv_outputs, dim=1).max(dim=1)[0].mean().item()
            
            # Perturbation metrics
            perturbation = adversarial_images - images
            l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1).mean().item()
            linf_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=float('inf'), dim=1).mean().item()
            
        return {
            'original_accuracy': orig_accuracy * 100,
            'attack_success_rate': success * 100,
            'original_confidence': orig_confidence,
            'adversarial_confidence': adv_confidence,
            'perturbation_l2': l2_norm,
            'perturbation_linf': linf_norm,
            'epsilon': self.epsilon
        }
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Callable interface"""
        return self.generate(images, labels, **kwargs)

def create_fgsm_attack(model: nn.Module, epsilon: float = 0.15, **kwargs) -> FGSMAttack:
    """Factory function for creating FGSM attack"""
    config = {'epsilon': epsilon, **kwargs}
    return FGSMAttack(model, config)
