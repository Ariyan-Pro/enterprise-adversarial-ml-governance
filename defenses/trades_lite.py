"""
TRADES-Lite Defense - CPU-optimized variant of TRADES
Reference: Zhang et al., "Theoretically Principled Trade-off between Robustness and Accuracy" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np

from attacks.fgsm import create_fgsm_attack
from attacks.pgd import create_pgd_attack


def trades_loss(model: nn.Module,
                x_natural: torch.Tensor,
                y: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                step_size: float = 0.003,
                epsilon: float = 0.031,
                perturb_steps: int = 10,
                beta: float = 1.0,
                distance: str = 'l_inf') -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    TRADES loss function - CPU optimized version
    
    Args:
        model: PyTorch model
        x_natural: Natural (clean) examples
        y: True labels
        optimizer: Model optimizer (for gradient updates in PGD)
        step_size: Attack step size
        epsilon: Maximum perturbation
        perturb_steps: Number of PGD steps
        beta: Trade-off parameter (β in paper)
        distance: Distance metric ('l_inf' or 'l_2')
    
    Returns:
        loss: Total TRADES loss
        metrics: Dictionary of loss components
    """
    # Generate adversarial examples
    if distance == 'l_inf':
        # PGD for L∞ perturbation
        pgd_config = {
            'epsilon': epsilon,
            'alpha': step_size,
            'steps': perturb_steps,
            'random_start': True,
            'device': x_natural.device
        }
        attack = create_pgd_attack(model, **pgd_config)
        x_adv = attack.generate(x_natural, y)
    else:
        # L2 perturbation (slower but more stable)
        x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        for _ in range(perturb_steps):
            x_adv.requires_grad = True
            with torch.enable_grad():
                loss_kl = F.kl_div(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                    reduction='batchmean'
                )
            
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            
            # Projection
            if distance == 'l_2':
                delta = x_adv - x_natural
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                factor = torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-8))
                delta = delta * factor.view(-1, 1, 1, 1)
            else:  # l_inf
                delta = torch.clamp(x_adv - x_natural, -epsilon, epsilon)
            
            x_adv = torch.clamp(x_natural + delta, 0.0, 1.0).detach()
    
    # Calculate losses
    model.train()
    
    # Natural loss
    logits_natural = model(x_natural)
    loss_natural = F.cross_entropy(logits_natural, y)
    
    # Robustness loss (KL divergence)
    logits_adv = model(x_adv)
    loss_robust = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_natural, dim=1),
        reduction='batchmean'
    )
    
    # Total loss: L = L_natural + β * L_robust
    loss = loss_natural + beta * loss_robust
    
    # Calculate metrics
    with torch.no_grad():
        natural_acc = (logits_natural.argmax(dim=1) == y).float().mean().item()
        adv_acc = (logits_adv.argmax(dim=1) == y).float().mean().item()
        
        # Perturbation magnitude
        perturbation = x_adv - x_natural
        if distance == 'l_inf':
            perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), 
                                          p=float('inf'), dim=1).mean().item()
        else:
            perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), 
                                          p=2, dim=1).mean().item()
    
    metrics = {
        'total_loss': loss.item(),
        'natural_loss': loss_natural.item(),
        'robust_loss': loss_robust.item(),
        'natural_accuracy': natural_acc * 100,
        'adversarial_accuracy': adv_acc * 100,
        'perturbation_norm': perturbation_norm,
        'beta': beta
    }
    
    return loss, metrics


class TRADESTrainer:
    """
    TRADES Training Wrapper - Manages TRADES training process
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        
        # TRADES parameters
        self.beta = self.config.get('beta', 6.0)  # Default from paper
        self.epsilon = self.config.get('epsilon', 0.031)
        self.step_size = self.config.get('step_size', 0.003)
        self.perturb_steps = self.config.get('perturb_steps', 10)
        self.distance = self.config.get('distance', 'l_inf')
        
        # Training parameters
        self.lr = self.config.get('lr', 0.01)
        self.momentum = self.config.get('momentum', 0.9)
        self.weight_decay = self.config.get('weight_decay', 2e-4)
        
        # Setup optimizer
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[75, 90],
            gamma=0.1
        )
        
        # Training history
        self.history = []
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with TRADES loss
        
        Args:
            x: Batch of images
            y: Batch of labels
        
        Returns:
            Step metrics
        """
        self.model.train()
        
        # Move to device
        x = x.to(next(self.model.parameters()).device)
        y = y.to(next(self.model.parameters()).device)
        
        # Calculate TRADES loss
        loss, metrics = trades_loss(
            model=self.model,
            x_natural=x,
            y=y,
            optimizer=self.optimizer,
            step_size=self.step_size,
            epsilon=self.epsilon,
            perturb_steps=self.perturb_steps,
            beta=self.beta,
            distance=self.distance
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return metrics
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate model on clean and adversarial data
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_natural_correct = 0
        total_adv_correct = 0
        total_samples = 0
        
        # Create PGD attack for validation
        pgd_config = {
            'epsilon': self.epsilon,
            'alpha': self.step_size,
            'steps': self.perturb_steps,
            'device': next(self.model.parameters()).device
        }
        attack = create_pgd_attack(self.model, **pgd_config)
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(next(self.model.parameters()).device)
                y = y.to(next(self.model.parameters()).device)
                
                batch_size = x.size(0)
                
                # Natural accuracy
                natural_outputs = self.model(x)
                natural_preds = natural_outputs.argmax(dim=1)
                natural_correct = (natural_preds == y).sum().item()
                
                # Adversarial accuracy
                x_adv = attack.generate(x, y)
                adv_outputs = self.model(x_adv)
                adv_preds = adv_outputs.argmax(dim=1)
                adv_correct = (adv_preds == y).sum().item()
                
                total_natural_correct += natural_correct
                total_adv_correct += adv_correct
                total_samples += batch_size
        
        natural_accuracy = total_natural_correct / total_samples * 100
        adv_accuracy = total_adv_correct / total_samples * 100
        
        return {
            'natural_accuracy': natural_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'robustness_gap': natural_accuracy - adv_accuracy
        }
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Epoch metrics
        """
        epoch_metrics = {
            'loss': 0.0,
            'natural_accuracy': 0.0,
            'adversarial_accuracy': 0.0,
            'perturbation_norm': 0.0
        }
        
        batch_count = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Training step
            step_metrics = self.train_step(x, y)
            
            # Accumulate metrics
            for key in epoch_metrics.keys():
                if key in step_metrics:
                    epoch_metrics[key] += step_metrics[key]
            
            batch_count += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {step_metrics['total_loss']:.4f} | "
                      f"Nat Acc: {step_metrics['natural_accuracy']:.1f}% | "
                      f"Adv Acc: {step_metrics['adversarial_accuracy']:.1f}%")
        
        # Average metrics
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= batch_count
        
        # Update learning rate
        self.scheduler.step()
        
        # Store in history
        epoch_record = {
            'epoch': epoch,
            **epoch_metrics,
            'lr': self.scheduler.get_last_lr()[0]
        }
        self.history.append(epoch_record)
        
        return epoch_metrics
    
    def get_training_history(self) -> list:
        """Get training history"""
        return self.history.copy()


# Factory function
def create_trades_trainer(model: nn.Module, **kwargs) -> TRADESTrainer:
    """Factory function for creating TRADES trainer"""
    return TRADESTrainer(model, kwargs)
