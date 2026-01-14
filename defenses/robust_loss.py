"""
Robust Loss Utilities - Metrics and loss functions for robustness evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List


def robustness_gap(clean_accuracy: float, adversarial_accuracy: float) -> float:
    """
    Calculate robustness gap (difference between clean and adversarial accuracy)
    
    Args:
        clean_accuracy: Accuracy on clean data (%)
        adversarial_accuracy: Accuracy on adversarial data (%)
    
    Returns:
        Robustness gap (%)
    """
    return clean_accuracy - adversarial_accuracy


def adversarial_success_rate(clean_accuracy: float, adversarial_accuracy: float) -> float:
    """
    Calculate adversarial success rate
    
    Args:
        clean_accuracy: Accuracy on clean data (%)
        adversarial_accuracy: Accuracy on adversarial data (%)
    
    Returns:
        Adversarial success rate (%)
    """
    return 100.0 - adversarial_accuracy


def perturbation_norm(clean_images: torch.Tensor, 
                     adversarial_images: torch.Tensor, 
                     norm_type: str = 'l2') -> float:
    """
    Calculate average perturbation norm
    
    Args:
        clean_images: Clean images
        adversarial_images: Adversarial images
        norm_type: Norm type ('l2', 'linf', 'l1')
    
    Returns:
        Average perturbation norm
    """
    perturbation = adversarial_images - clean_images
    batch_size = perturbation.size(0)
    
    if norm_type == 'l2':
        norms = torch.norm(perturbation.view(batch_size, -1), p=2, dim=1)
    elif norm_type == 'linf':
        norms = torch.norm(perturbation.view(batch_size, -1), p=float('inf'), dim=1)
    elif norm_type == 'l1':
        norms = torch.norm(perturbation.view(batch_size, -1), p=1, dim=1)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    
    return norms.mean().item()


def confidence_drop(clean_outputs: torch.Tensor, 
                   adversarial_outputs: torch.Tensor) -> float:
    """
    Calculate average confidence drop
    
    Args:
        clean_outputs: Model outputs on clean data
        adversarial_outputs: Model outputs on adversarial data
    
    Returns:
        Average confidence drop (0-1)
    """
    clean_probs = F.softmax(clean_outputs, dim=1)
    adv_probs = F.softmax(adversarial_outputs, dim=1)
    
    clean_confidence = clean_probs.max(dim=1)[0].mean().item()
    adv_confidence = adv_probs.max(dim=1)[0].mean().item()
    
    return clean_confidence - adv_confidence


def calculate_robustness_metrics(model: nn.Module,
                                clean_images: torch.Tensor,
                                adversarial_images: torch.Tensor,
                                labels: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive robustness metrics
    
    Args:
        model: PyTorch model
        clean_images: Clean images
        adversarial_images: Adversarial images
        labels: True labels
    
    Returns:
        Dictionary of robustness metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Move to same device as model
        device = next(model.parameters()).device
        clean_images = clean_images.to(device)
        adversarial_images = adversarial_images.to(device)
        labels = labels.to(device)
        
        # Get predictions
        clean_outputs = model(clean_images)
        adv_outputs = model(adversarial_images)
        
        clean_preds = clean_outputs.argmax(dim=1)
        adv_preds = adv_outputs.argmax(dim=1)
        
        # Calculate accuracies
        clean_acc = (clean_preds == labels).float().mean().item() * 100
        adv_acc = (adv_preds == labels).float().mean().item() * 100
        
        # Calculate metrics
        metrics = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'robustness_gap': robustness_gap(clean_acc, adv_acc),
            'adversarial_success_rate': adversarial_success_rate(clean_acc, adv_acc),
            'perturbation_l2': perturbation_norm(clean_images, adversarial_images, 'l2'),
            'perturbation_linf': perturbation_norm(clean_images, adversarial_images, 'linf'),
            'confidence_drop': confidence_drop(clean_outputs, adv_outputs),
            'clean_confidence': F.softmax(clean_outputs, dim=1).max(dim=1)[0].mean().item(),
            'adversarial_confidence': F.softmax(adv_outputs, dim=1).max(dim=1)[0].mean().item()
        }
    
    return metrics


class RobustnessScorer:
    """
    Robustness Scoring System - Calculates enterprise robustness KPIs
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def add_evaluation(self, 
                      clean_accuracy: float,
                      adversarial_accuracy: float,
                      perturbation_l2: float,
                      perturbation_linf: float,
                      confidence_drop: float,
                      metadata: Dict[str, Any] = None):
        """
        Add evaluation results
        
        Args:
            clean_accuracy: Clean accuracy (%)
            adversarial_accuracy: Adversarial accuracy (%)
            perturbation_l2: L2 perturbation norm
            perturbation_linf: L∞ perturbation norm
            confidence_drop: Confidence drop (0-1)
            metadata: Additional metadata
        """
        evaluation = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'robustness_gap': robustness_gap(clean_accuracy, adversarial_accuracy),
            'adversarial_success_rate': adversarial_success_rate(clean_accuracy, adversarial_accuracy),
            'perturbation_l2': perturbation_l2,
            'perturbation_linf': perturbation_linf,
            'confidence_drop': confidence_drop,
            'robustness_score': self._calculate_robustness_score(
                clean_accuracy, adversarial_accuracy, perturbation_l2
            ),
            'metadata': metadata or {}
        }
        
        self.metrics_history.append(evaluation)
        return evaluation
    
    def _calculate_robustness_score(self,
                                   clean_accuracy: float,
                                   adversarial_accuracy: float,
                                   perturbation_l2: float) -> float:
        """
        Calculate composite robustness score (0-100)
        
        Higher score = better robustness
        """
        # Normalize to 0-100 scale
        accuracy_score = adversarial_accuracy  # 0-100
        
        # Invert perturbation (lower perturbation = higher score)
        # Assuming typical L2 perturbation range 0-50 for MNIST
        perturbation_score = max(0, 100 - (perturbation_l2 * 2))
        
        # Weighted combination
        score = 0.7 * accuracy_score + 0.3 * perturbation_score
        
        return min(100, max(0, score))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations"""
        if not self.metrics_history:
            return {}
        
        summary = {
            'num_evaluations': len(self.metrics_history),
            'avg_clean_accuracy': np.mean([m['clean_accuracy'] for m in self.metrics_history]),
            'avg_adversarial_accuracy': np.mean([m['adversarial_accuracy'] for m in self.metrics_history]),
            'avg_robustness_gap': np.mean([m['robustness_gap'] for m in self.metrics_history]),
            'avg_robustness_score': np.mean([m['robustness_score'] for m in self.metrics_history]),
            'best_robustness_score': max([m['robustness_score'] for m in self.metrics_history]),
            'worst_robustness_score': min([m['robustness_score'] for m in self.metrics_history])
        }
        
        return summary
    
    def clear_history(self):
        """Clear evaluation history"""
        self.metrics_history = []
    
    def save_to_json(self, filepath: str):
        """Save metrics history to JSON file"""
        import json
        from utils.json_utils import safe_json_dump
        
        data = {
            'metrics_history': self.metrics_history,
            'summary': self.get_summary()
        }
        
        safe_json_dump(data, filepath)
    
    def load_from_json(self, filepath: str):
        """Load metrics history from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics_history = data.get('metrics_history', [])


# Factory function
def create_robustness_scorer() -> RobustnessScorer:
    """Factory function for creating robustness scorer"""
    return RobustnessScorer()
