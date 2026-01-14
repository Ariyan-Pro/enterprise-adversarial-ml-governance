"""
Carlini & Wagner (C&W) L2 Attack
Enterprise implementation with full error handling and optimization
Reference: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import time


class CarliniWagnerL2:
    """
    Carlini & Wagner L2 Attack - Enterprise Implementation
    
    Features:
    - CPU-optimized with early stopping
    - Multiple search methods for optimal c parameter
    - Confidence thresholding
    - Comprehensive logging and metrics
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        Initialize C&W attack
        
        Args:
            model: PyTorch model to attack
            config: Attack configuration dictionary
        """
        self.model = model
        self.config = config or {}
        
        # Attack parameters with defaults
        self.confidence = self.config.get('confidence', 0.0)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.binary_search_steps = self.config.get('binary_search_steps', 9)
        self.initial_const = self.config.get('initial_const', 1e-3)
        self.abort_early = self.config.get('abort_early', True)
        self.device = self.config.get('device', 'cpu')
        
        # Optimization parameters
        self.box_min = self.config.get('box_min', 0.0)
        self.box_max = self.config.get('box_max', 1.0)
        
        self.model.eval()
        self.model.to(self.device)
        
    def _tanh_space(self, x: torch.Tensor, boxmin: float, boxmax: float) -> torch.Tensor:
        """Transform to tanh space to handle box constraints"""
        return torch.tanh(x) * (boxmax - boxmin) / 2 + (boxmax + boxmin) / 2
    
    def _inverse_tanh_space(self, x: torch.Tensor, boxmin: float, boxmax: float) -> torch.Tensor:
        """Inverse transform from tanh space"""
        return torch.atanh((2 * (x - boxmin) / (boxmax - boxmin) - 1).clamp(-1 + 1e-7, 1 - 1e-7))
    
    def _compute_loss(self, 
                     adv_images: torch.Tensor,
                     images: torch.Tensor,
                     labels: torch.Tensor,
                     const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute C&W loss components
        
        Returns:
            total_loss, distance_loss, classification_loss
        """
        # L2 distance
        l2_dist = torch.norm((adv_images - images).view(images.size(0), -1), p=2, dim=1)
        distance_loss = l2_dist.sum()
        
        # Classification loss (C&W formulation)
        logits = self.model(adv_images)
        
        # Get correct class logits
        correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze()
        
        # Get maximum logit of incorrect classes
        mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0)
        other_logits = torch.max(logits * mask, dim=1)[0]
        
        # C&W loss: max(other_logits - correct_logits, -confidence)
        classification_loss = torch.clamp(other_logits - correct_logits + self.confidence, min=0.0)
        classification_loss = (const * classification_loss).sum()
        
        total_loss = distance_loss + classification_loss
        
        return total_loss, distance_loss, classification_loss
    
    def _optimize_single(self,
                        images: torch.Tensor,
                        labels: torch.Tensor,
                        const: float,
                        early_stop_threshold: float = 1e-4) -> Tuple[torch.Tensor, float, bool]:
        """
        Single optimization run for given constant
        
        Returns:
            adversarial_images, best_l2, attack_successful
        """
        batch_size = images.size(0)
        
        # Initialize in tanh space
        w = self._inverse_tanh_space(images, self.box_min, self.box_max).detach()
        w.requires_grad = True
        
        # Optimizer
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        
        # For early stopping
        prev_loss = float('inf')
        best_l2 = float('inf')
        best_adv = images.clone()
        const_tensor = torch.full((batch_size,), const, device=self.device)
        
        attack_successful = False
        
        for iteration in range(self.max_iterations):
            # Forward pass
            adv_images = self._tanh_space(w, self.box_min, self.box_max)
            
            # Compute loss
            total_loss, distance_loss, classification_loss = self._compute_loss(
                adv_images, images, labels, const_tensor
            )
            
            # Check attack success
            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
                success_mask = (preds != labels)
                current_l2 = torch.norm((adv_images - images).view(batch_size, -1), p=2, dim=1)
                
                # Update best adversarial examples
                for i in range(batch_size):
                    if success_mask[i] and current_l2[i] < best_l2:
                        best_l2 = current_l2[i].item()
                        best_adv[i] = adv_images[i]
                        attack_successful = True
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Early stopping check
            if self.abort_early and iteration % 10 == 0:
                if total_loss.item() > prev_loss * 0.9999:
                    break
                prev_loss = total_loss.item()
        
        return best_adv, best_l2, attack_successful
    
    def generate(self,
                images: torch.Tensor,
                labels: torch.Tensor,
                targeted: bool = False,
                target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate adversarial examples using C&W attack
        
        Args:
            images: Clean images [batch, channels, height, width]
            labels: True labels for non-targeted attack
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack
            
        Returns:
            Adversarial images
        """
        if targeted and target_labels is None:
            raise ValueError("target_labels required for targeted attack")
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if targeted:
            labels = target_labels.clone().detach().to(self.device)
        
        batch_size = images.size(0)
        
        # Binary search for optimal const
        const_lower_bound = torch.zeros(batch_size, device=self.device)
        const_upper_bound = torch.ones(batch_size, device=self.device) * 1e10
        const = torch.ones(batch_size, device=self.device) * self.initial_const
        
        # Best results tracking
        best_l2 = torch.ones(batch_size, device=self.device) * float('inf')
        best_adv = images.clone()
        
        for binary_step in range(self.binary_search_steps):
            print(f"  Binary search step {binary_step + 1}/{self.binary_search_steps}")
            
            # Optimize for current const values
            for i in range(batch_size):
                const_i = const[i].item()
                adv_i, l2_i, success_i = self._optimize_single(
                    images[i:i+1], labels[i:i+1], const_i
                )
                
                if success_i:
                    # Success: try smaller const
                    const_upper_bound[i] = min(const_upper_bound[i], const_i)
                    if const_upper_bound[i] < 1e9:
                        const[i] = (const_lower_bound[i] + const_upper_bound[i]) / 2
                    
                    # Update best result
                    if l2_i < best_l2[i]:
                        best_l2[i] = l2_i
                        best_adv[i] = adv_i
                else:
                    # Failure: try larger const
                    const_lower_bound[i] = max(const_lower_bound[i], const_i)
                    if const_upper_bound[i] < 1e9:
                        const[i] = (const_lower_bound[i] + const_upper_bound[i]) / 2
                    else:
                        const[i] = const[i] * 10
        
        return best_adv
    
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
            success_rate = (adv_preds != labels).float().mean().item()
            
            # Perturbation metrics
            perturbation = adversarial_images - images
            l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1)
            linf_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=float('inf'), dim=1)
            
            # Confidence metrics
            orig_probs = F.softmax(orig_outputs, dim=1)
            adv_probs = F.softmax(adv_outputs, dim=1)
            orig_confidence = orig_probs.max(dim=1)[0].mean().item()
            adv_confidence = adv_probs.max(dim=1)[0].mean().item()
            
            # Successful attack statistics
            success_mask = (adv_preds != labels)
            if success_mask.any():
                successful_l2 = l2_norm[success_mask].mean().item()
                successful_linf = linf_norm[success_mask].mean().item()
            else:
                successful_l2 = 0.0
                successful_linf = 0.0
        
        return {
            'original_accuracy': orig_accuracy * 100,
            'attack_success_rate': success_rate * 100,
            'avg_l2_perturbation': l2_norm.mean().item(),
            'avg_linf_perturbation': linf_norm.mean().item(),
            'successful_l2_perturbation': successful_l2,
            'successful_linf_perturbation': successful_linf,
            'original_confidence': orig_confidence,
            'adversarial_confidence': adv_confidence,
            'confidence_threshold': self.confidence
        }
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Callable interface"""
        return self.generate(images, labels, **kwargs)


class FastCarliniWagnerL2:
    """
    Faster C&W implementation for CPU - Uses fixed const and fewer iterations
    Suitable for larger batches and quicker evaluations
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        
        self.const = self.config.get('const', 1.0)
        self.iterations = self.config.get('iterations', 50)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.device = self.config.get('device', 'cpu')
        
        self.model.eval()
        self.model.to(self.device)
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Fast C&W generation with fixed const"""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        batch_size = images.size(0)
        
        # Initialize in tanh space
        w = torch.zeros_like(images, requires_grad=True)
        w.data = torch.atanh((2 * (images - 0.5) / 1).clamp(-1 + 1e-7, 1 - 1e-7))
        
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        
        for iteration in range(self.iterations):
            adv_images = torch.tanh(w) * 0.5 + 0.5
            
            # L2 distance
            l2_dist = torch.norm((adv_images - images).view(batch_size, -1), p=2, dim=1)
            
            # C&W classification loss
            logits = self.model(adv_images)
            correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze()
            mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0)
            other_logits = torch.max(logits * mask, dim=1)[0]
            
            classification_loss = torch.clamp(other_logits - correct_logits, min=0.0)
            
            # Total loss
            loss = torch.mean(self.const * classification_loss + l2_dist)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return torch.tanh(w) * 0.5 + 0.5


# Factory functions
def create_cw_attack(model: nn.Module, const: float = 1e-3, **kwargs) -> CarliniWagnerL2:
    """Factory function for creating C&W attack"""
    config = {'initial_const': const, **kwargs}
    return CarliniWagnerL2(model, config)

def create_fast_cw_attack(model: nn.Module, const: float = 1.0, **kwargs) -> FastCarliniWagnerL2:
    """Factory function for creating fast C&W attack"""
    config = {'const': const, **kwargs}
    return FastCarliniWagnerL2(model, config)
