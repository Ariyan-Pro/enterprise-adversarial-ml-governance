"""
Adversarial Training Defense
Enterprise implementation with mixed batch training and curriculum learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List, Union
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
import numpy as np

class AdversarialTraining:
    """Adversarial training defense with multiple attack types"""
    
    def __init__(self, 
                 model: nn.Module,
                 attack_type: str = 'fgsm',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize adversarial training
        
        Args:
            model: PyTorch model to defend
            attack_type: Type of attack to use ('fgsm', 'pgd', 'mixed')
            config: Training configuration
        """
        self.model = model
        self.attack_type = attack_type.lower()
        self.config = config or {}
        
        # Training parameters
        self.epsilon = self.config.get('epsilon', 0.15)
        self.alpha = self.config.get('alpha', 0.8)  # Mix ratio: clean vs adversarial
        self.epochs = self.config.get('epochs', 8)
        self.attack_steps = self.config.get('attack_steps', 10)
        self.curriculum = self.config.get('curriculum', True)
        
        # Attack configuration
        self.attack_config = {
            'epsilon': self.epsilon,
            'device': self.config.get('device', 'cpu'),
            'clip_min': 0.0,
            'clip_max': 1.0
        }
        
        # Initialize attacks
        self._init_attacks()
        
        # Statistics
        self.training_history = []
        
    def _init_attacks(self):
        """Initialize attack objects"""
        if self.attack_type == 'fgsm':
            from attacks.fgsm import create_fgsm_attack
            self.attack = create_fgsm_attack(self.model, **self.attack_config)
        elif self.attack_type == 'pgd':
            from attacks.pgd import create_pgd_attack
            self.attack_config['steps'] = self.attack_steps
            self.attack_config['alpha'] = self.attack_config.get('alpha', 0.01)
            self.attack = create_pgd_attack(self.model, **self.attack_config)
        elif self.attack_type == 'mixed':
            # Initialize both attacks for mixed training
            from attacks.fgsm import create_fgsm_attack
            from attacks.pgd import create_pgd_attack
            
            self.fgsm_attack = create_fgsm_attack(self.model, **self.attack_config)
            
            pgd_config = self.attack_config.copy()
            pgd_config['steps'] = self.attack_steps
            pgd_config['alpha'] = pgd_config.get('alpha', 0.01)
            self.pgd_attack = create_pgd_attack(self.model, **pgd_config)
        else:
            raise ValueError(f"Unsupported attack type: {self.attack_type}")
    
    def _generate_adversarial_batch(self,
                                   images: torch.Tensor,
                                   labels: torch.Tensor,
                                   epoch: int) -> torch.Tensor:
        """
        Generate adversarial batch based on curriculum
        
        Args:
            images: Clean images
            labels: True labels
            epoch: Current epoch for curriculum scheduling
            
        Returns:
            Adversarial images
        """
        # Curriculum learning: increase difficulty over time
        if self.curriculum:
            effective_epsilon = min(self.epsilon, self.epsilon * (epoch + 1) / self.epochs)
            effective_steps = min(self.attack_steps, int(self.attack_steps * (epoch + 1) / self.epochs))
        else:
            effective_epsilon = self.epsilon
            effective_steps = self.attack_steps
        
        # Generate adversarial examples
        if self.attack_type == 'mixed':
            # Mix FGSM and PGD attacks
            if epoch % 2 == 0:
                adversarial_images = self.fgsm_attack.generate(images, labels)
            else:
                pgd_config = self.attack_config.copy()
                pgd_config['epsilon'] = effective_epsilon
                pgd_config['steps'] = effective_steps
                adversarial_images = self.pgd_attack.generate(images, labels)
        else:
            # Single attack type
            if self.attack_type == 'pgd':
                self.attack.config['epsilon'] = effective_epsilon
                self.attack.config['steps'] = effective_steps
            
            adversarial_images = self.attack.generate(images, labels)
        
        return adversarial_images
    
    def train_step(self,
                  images: torch.Tensor,
                  labels: torch.Tensor,
                  optimizer: optim.Optimizer,
                  criterion: nn.Module,
                  epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Single training step with adversarial examples
        
        Args:
            images: Batch of images
            labels: Batch of labels
            optimizer: Model optimizer
            criterion: Loss function
            epoch: Current epoch
            
        Returns:
            Tuple of (loss, metrics)
        """
        self.model.train()
        
        # Generate adversarial examples
        with torch.no_grad():
            adversarial_images = self._generate_adversarial_batch(images, labels, epoch)
        
        # Create mixed batch
        batch_size = images.size(0)
        num_clean = int(batch_size * (1 - self.alpha))
        num_adv = batch_size - num_clean
        
        # Select indices for clean and adversarial examples
        if num_clean > 0 and num_adv > 0:
            indices = torch.randperm(batch_size)
            clean_indices = indices[:num_clean]
            adv_indices = indices[num_clean:]
            
            # Combine clean and adversarial examples
            mixed_images = torch.cat([
                images[clean_indices],
                adversarial_images[adv_indices]
            ], dim=0)
            
            mixed_labels = torch.cat([
                labels[clean_indices],
                labels[adv_indices]
            ], dim=0)
        elif num_adv == 0:
            # All clean examples
            mixed_images = images
            mixed_labels = labels
        else:
            # All adversarial examples
            mixed_images = adversarial_images
            mixed_labels = labels
        
        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(mixed_images)
        loss = criterion(outputs, mixed_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Clean accuracy
            clean_outputs = self.model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_acc = (clean_preds == labels).float().mean().item()
            
            # Adversarial accuracy
            adv_outputs = self.model(adversarial_images)
            adv_preds = adv_outputs.argmax(dim=1)
            adv_acc = (adv_preds == labels).float().mean().item()
            
            # Loss breakdown
            clean_loss = criterion(clean_outputs, labels).item()
            adv_loss = criterion(adv_outputs, labels).item()
        
        metrics = {
            'loss': loss.item(),
            'clean_accuracy': clean_acc * 100,
            'adversarial_accuracy': adv_acc * 100,
            'clean_loss': clean_loss,
            'adversarial_loss': adv_loss,
            'mixed_ratio': self.alpha
        }
        
        return loss.item(), metrics
    
    def train_epoch(self,
                   train_loader: torch.utils.data.DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module,
                   epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Model optimizer
            criterion: Loss function
            epoch: Current epoch
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_clean_acc = 0.0
        epoch_adv_acc = 0.0
        batch_count = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.config.get('device', 'cpu'))
            labels = labels.to(self.config.get('device', 'cpu'))
            
            # Training step
            loss, metrics = self.train_step(images, labels, optimizer, criterion, epoch)
            
            # Accumulate metrics
            epoch_loss += loss
            epoch_clean_acc += metrics['clean_accuracy']
            epoch_adv_acc += metrics['adversarial_accuracy']
            batch_count += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss:.4f} | "
                      f"Clean Acc: {metrics['clean_accuracy']:.2f}% | "
                      f"Adv Acc: {metrics['adversarial_accuracy']:.2f}%")
        
        # Calculate epoch averages
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': epoch_loss / batch_count,
            'clean_accuracy': epoch_clean_acc / batch_count,
            'adversarial_accuracy': epoch_adv_acc / batch_count,
            'attack_type': self.attack_type,
            'epsilon': self.epsilon,
            'alpha': self.alpha
        }
        
        self.training_history.append(epoch_metrics)
        
        return epoch_metrics
    
    def validate(self,
                val_loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                attack: Optional[Any] = None) -> Dict[str, float]:
        """
        Validate model on clean and adversarial data
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            attack: Optional attack for adversarial validation
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        if attack is None:
            # Use the training attack
            attack = self.attack if self.attack_type != 'mixed' else self.pgd_attack
        
        total_loss = 0.0
        total_clean_correct = 0
        total_adv_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.config.get('device', 'cpu'))
                labels = labels.to(self.config.get('device', 'cpu'))
                
                batch_size = images.size(0)
                
                # Clean predictions
                clean_outputs = self.model(images)
                clean_loss = criterion(clean_outputs, labels)
                clean_preds = clean_outputs.argmax(dim=1)
                
                # Generate adversarial examples
                adversarial_images = attack.generate(images, labels)
                
                # Adversarial predictions
                adv_outputs = self.model(adversarial_images)
                adv_loss = criterion(adv_outputs, labels)
                adv_preds = adv_outputs.argmax(dim=1)
                
                # Accumulate metrics
                total_loss += (clean_loss.item() + adv_loss.item()) / 2
                total_clean_correct += (clean_preds == labels).sum().item()
                total_adv_correct += (adv_preds == labels).sum().item()
                total_samples += batch_size
        
        metrics = {
            'validation_loss': total_loss / len(val_loader),
            'clean_accuracy': total_clean_correct / total_samples * 100,
            'adversarial_accuracy': total_adv_correct / total_samples * 100,
            'robustness_gap': (total_clean_correct - total_adv_correct) / total_samples * 100
        }
        
        return metrics
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """Get training history"""
        return self.training_history
    
    def save_checkpoint(self, path: str, optimizer: Optional[optim.Optimizer] = None):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'attack_type': self.attack_type
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, optimizer: Optional[optim.Optimizer] = None):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.config.get('device', 'cpu'))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
