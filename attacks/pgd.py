"""
Projected Gradient Descent (PGD) Attack
Enterprise implementation with multiple restarts and adaptive step size
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from attacks.fgsm import FGSMAttack

class PGDAttack:
    """PGD attack with random restarts and adaptive step size"""
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PGD attack
        
        Args:
            model: PyTorch model to attack
            config: Attack configuration dictionary
        """
        self.model = model
        self.config = config or {}
        
        # Default parameters
        self.epsilon = self.config.get('epsilon', 0.3)
        self.alpha = self.config.get('alpha', 0.01)
        self.steps = self.config.get('steps', 10)
        self.random_start = self.config.get('random_start', True)
        self.targeted = self.config.get('targeted', False)
        self.clip_min = self.config.get('clip_min', 0.0)
        self.clip_max = self.config.get('clip_max', 1.0)
        self.device = self.config.get('device', 'cpu')
        self.restarts = self.config.get('restarts', 1)
        
        self.criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
    def _project_onto_l_inf_ball(self, 
                                x: torch.Tensor, 
                                perturbation: torch.Tensor) -> torch.Tensor:
        """Project perturbation onto Linf epsilon-ball"""
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)
    
    def _random_initialization(self, x: torch.Tensor) -> torch.Tensor:
        """Random initialization within epsilon-ball"""
        delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x + delta, self.clip_min, self.clip_max)
        return x_adv - x  # Return delta
        
    def _single_restart(self,
                       images: torch.Tensor,
                       labels: torch.Tensor,
                       target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single PGD restart"""
        batch_size = images.shape[0]
        
        # Initialize adversarial examples
        if self.random_start:
            delta = self._random_initialization(images)
        else:
            delta = torch.zeros_like(images)
        
        x_adv = images + delta
        
        # PGD iterations
        for step in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
            # Forward pass
            outputs = self.model(x_adv)
            
            # Loss calculation
            if self.targeted:
                loss = -self.criterion(outputs, target_labels)
            else:
                loss = self.criterion(outputs, labels)
            
            # Gradient calculation
            grad = torch.autograd.grad(loss, [x_adv])[0]
            
            # PGD update: x' = x + a * sign(?x)
            if self.targeted:
                delta = delta - self.alpha * grad.sign()
            else:
                delta = delta + self.alpha * grad.sign()
            
            # Project onto epsilon-ball
            delta = self._project_onto_l_inf_ball(images, delta)
            
            # Update adversarial examples
            x_adv = torch.clamp(images + delta, self.clip_min, self.clip_max)
        
        return x_adv
    
    def generate(self,
                images: torch.Tensor,
                labels: torch.Tensor,
                target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate adversarial examples with multiple restarts
        
        Args:
            images: Clean images
            labels: True labels
            target_labels: Target labels for targeted attack
            
        Returns:
            Best adversarial examples across restarts
        """
        if self.targeted and target_labels is None:
            raise ValueError("target_labels required for targeted attack")
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)
        
        # Initialize best adversarial examples
        best_adv = None
        best_loss = -float('inf') if self.targeted else float('inf')
        
        # Multiple restarts
        for restart in range(self.restarts):
            # Generate adversarial examples for this restart
            x_adv = self._single_restart(images, labels, target_labels)
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.model(x_adv)
                if self.targeted:
                    loss = -self.criterion(outputs, target_labels)
                else:
                    loss = self.criterion(outputs, labels)
            
            # Update best adversarial examples
            if self.targeted:
                if loss > best_loss:
                    best_loss = loss
                    best_adv = x_adv
            else:
                if loss < best_loss:
                    best_loss = loss
                    best_adv = x_adv
        
        return best_adv
    
    def adaptive_attack(self,
                       images: torch.Tensor,
                       labels: torch.Tensor,
                       initial_epsilon: float = 0.1,
                       max_iterations: int = 20) -> Tuple[torch.Tensor, float]:
        """
        Adaptive PGD that finds minimal epsilon for successful attack
        
        Args:
            images: Clean images
            labels: True labels
            initial_epsilon: Starting epsilon
            max_iterations: Maximum binary search iterations
            
        Returns:
            Tuple of (adversarial examples, optimal epsilon)
        """
        eps_low = 0.0
        eps_high = initial_epsilon * 2
        
        # Find upper bound
        for _ in range(10):
            self.epsilon = eps_high
            adv_images = self.generate(images, labels)
            
            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
                success_rate = (preds != labels).float().mean().item()
            
            if success_rate > 0.9:  # 90% success rate
                break
            eps_high *= 2
        
        # Binary search for optimal epsilon
        best_epsilon = eps_high
        best_adv = adv_images
        
        for _ in range(max_iterations):
            epsilon = (eps_low + eps_high) / 2
            self.epsilon = epsilon
            
            adv_images = self.generate(images, labels)
            
            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
                success_rate = (preds != labels).float().mean().item()
            
            if success_rate > 0.9:  # 90% success threshold
                eps_high = epsilon
                best_epsilon = epsilon
                best_adv = adv_images
            else:
                eps_low = epsilon
        
        return best_adv, best_epsilon
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Callable interface"""
        return self.generate(images, labels, **kwargs)

def create_pgd_attack(model: nn.Module, epsilon: float = 0.3, **kwargs) -> PGDAttack:
    """Factory function for creating PGD attack"""
    config = {'epsilon': epsilon, **kwargs}
    return PGDAttack(model, config)
