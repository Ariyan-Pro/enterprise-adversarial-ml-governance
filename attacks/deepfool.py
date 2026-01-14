"""
DeepFool Attack Implementation
Enterprise-grade with support for multi-class and binary classification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import warnings

class DeepFoolAttack:
    """DeepFool attack for minimal perturbation"""
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DeepFool attack
        
        Args:
            model: PyTorch model to attack
            config: Attack configuration dictionary
        """
        self.model = model
        self.config = config or {}
        
        # Default parameters
        self.max_iter = self.config.get('max_iter', 50)
        self.overshoot = self.config.get('overshoot', 0.02)
        self.num_classes = self.config.get('num_classes', 10)
        self.clip_min = self.config.get('clip_min', 0.0)
        self.clip_max = self.config.get('clip_max', 1.0)
        self.device = self.config.get('device', 'cpu')
        
        self.model.eval()
        
    def _compute_gradients(self, 
                          x: torch.Tensor, 
                          target_class: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients for all classes
        
        Args:
            x: Input tensor
            target_class: Optional target class for binary search
            
        Returns:
            Tuple of (gradients, outputs)
        """
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x)
        
        # Get gradients for all classes
        gradients = []
        for k in range(self.num_classes):
            if k == target_class and target_class is not None:
                continue
                
            # Zero gradients
            if x.grad is not None:
                x.grad.zero_()
            
            # Backward for class k
            outputs[0, k].backward(retain_graph=True)
            gradients.append(x.grad.clone())
        
        # Clean up
        if x.grad is not None:
            x.grad.zero_()
        
        return torch.stack(gradients, dim=0), outputs.detach()
    
    def _binary_search(self,
                      x: torch.Tensor,
                      perturbation: torch.Tensor,
                      original_class: int,
                      target_class: int,
                      max_search_iter: int = 10) -> torch.Tensor:
        """
        Binary search for minimal perturbation
        
        Args:
            x: Original image
            perturbation: Initial perturbation
            original_class: Original predicted class
            target_class: Target class for misclassification
            max_search_iter: Maximum binary search iterations
        
        Returns:
            Minimal perturbation that causes misclassification
        """
        eps_low = 0.0
        eps_high = 1.0
        best_perturbation = perturbation
        
        for _ in range(max_search_iter):
            eps = (eps_low + eps_high) / 2
            x_adv = torch.clamp(x + eps * perturbation, self.clip_min, self.clip_max)
            
            with torch.no_grad():
                outputs = self.model(x_adv)
                pred_class = outputs.argmax(dim=1).item()
            
            if pred_class == target_class:
                eps_high = eps
                best_perturbation = eps * perturbation
            else:
                eps_low = eps
        
        return best_perturbation
    
    def _deepfool_single(self, x: torch.Tensor, original_class: int) -> Tuple[torch.Tensor, int, int]:
        """
        DeepFool for a single sample
        
        Args:
            x: Input tensor [1, C, H, W]
            original_class: Original predicted class
        
        Returns:
            Tuple of (perturbation, target_class, iterations)
        """
        x = x.to(self.device)
        x_adv = x.clone().detach()
        
        # Initialize
        r_total = torch.zeros_like(x)
        iterations = 0
        
        with torch.no_grad():
            outputs = self.model(x_adv)
            current_class = outputs.argmax(dim=1).item()
        
        while current_class == original_class and iterations < self.max_iter:
            # Compute gradients for all classes
            gradients, outputs = self._compute_gradients(x_adv)
            
            # Get current class score
            f_k = outputs[0, original_class]
            
            # Compute distances to decision boundaries
            distances = []
            for k in range(self.num_classes):
                if k == original_class:
                    continue
                
                w_k = gradients[k - (1 if k > original_class else 0)] - gradients[-1]
                f_k_prime = outputs[0, k]
                
                distance = torch.abs(f_k - f_k_prime) / (torch.norm(w_k.flatten()) + 1e-8)
                distances.append((distance.item(), k, w_k))
            
            # Find closest decision boundary
            distances.sort(key=lambda x: x[0])
            min_distance, target_class, w = distances[0]
            
            # Compute perturbation
            perturbation = (torch.abs(f_k - outputs[0, target_class]) + 1e-8) / \
                          (torch.norm(w.flatten()) ** 2 + 1e-8) * w
            
            # Update adversarial example
            x_adv = torch.clamp(x_adv + perturbation, self.clip_min, self.clip_max)
            r_total = r_total + perturbation
            
            # Check new prediction
            with torch.no_grad():
                outputs = self.model(x_adv)
                current_class = outputs.argmax(dim=1).item()
            
            iterations += 1
        
        # Apply overshoot
        if iterations < self.max_iter:
            r_total = (1 + self.overshoot) * r_total
        
        # Binary search for minimal perturbation
        if iterations > 0:
            r_total = self._binary_search(x, r_total, original_class, target_class)
        
        return r_total, target_class, iterations
    
    def generate(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate adversarial examples
        
        Args:
            images: Clean images [batch, C, H, W]
            labels: Optional labels for validation
        
        Returns:
            Adversarial images
        """
        batch_size = images.shape[0]
        images = images.clone().detach().to(self.device)
        
        # Get original predictions
        with torch.no_grad():
            outputs = self.model(images)
            original_classes = outputs.argmax(dim=1)
        
        adversarial_images = []
        success_count = 0
        total_iterations = 0
        
        # Process each image separately
        for i in range(batch_size):
            x = images[i:i+1]
            original_class = original_classes[i].item()
            
            # Generate perturbation
            perturbation, target_class, iterations = self._deepfool_single(x, original_class)
            
            # Create adversarial example
            x_adv = torch.clamp(x + perturbation, self.clip_min, self.clip_max)
            adversarial_images.append(x_adv)
            
            # Update statistics
            total_iterations += iterations
            if target_class != original_class:
                success_count += 1
        
        adversarial_images = torch.cat(adversarial_images, dim=0)
        
        # Calculate metrics
        with torch.no_grad():
            adv_outputs = self.model(adversarial_images)
            adv_classes = adv_outputs.argmax(dim=1)
            
            success_rate = success_count / batch_size * 100
            avg_iterations = total_iterations / batch_size
            
            # Perturbation metrics
            perturbation_norm = torch.norm(
                (adversarial_images - images).view(batch_size, -1), 
                p=2, dim=1
            ).mean().item()
        
        # Store metrics
        self.metrics = {
            'success_rate': success_rate,
            'avg_iterations': avg_iterations,
            'avg_perturbation': perturbation_norm,
            'original_accuracy': (original_classes == labels).float().mean().item() * 100 if labels is not None else None
        }
        
        return adversarial_images
    
    def get_minimal_perturbation(self, 
                                images: torch.Tensor, 
                                target_accuracy: float = 10.0) -> Tuple[torch.Tensor, float]:
        """
        Find minimal epsilon for target attack success rate
        
        Args:
            images: Clean images
            target_accuracy: Target accuracy after attack
        
        Returns:
            Tuple of (adversarial images, epsilon)
        """
        warnings.warn("DeepFool doesn't use epsilon parameter like FGSM/PGD")
        
        # Generate adversarial examples
        adv_images = self.generate(images)
        
        # Calculate effective epsilon (Linf norm)
        perturbation = adv_images - images
        epsilon = torch.norm(perturbation.view(perturbation.shape[0], -1), 
                           p=float('inf'), dim=1).mean().item()
        
        return adv_images, epsilon
    
    def __call__(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """Callable interface"""
        return self.generate(images, **kwargs)

def create_deepfool_attack(model: nn.Module, max_iter: int = 50, **kwargs) -> DeepFoolAttack:
    """Factory function for creating DeepFool attack"""
    config = {'max_iter': max_iter, **kwargs}
    return DeepFoolAttack(model, config)
