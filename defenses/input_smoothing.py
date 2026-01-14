"""
Input Smoothing Defense
Enterprise implementation with multiple smoothing techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import cv2
from scipy.ndimage import gaussian_filter

class InputSmoothing:
    """Input smoothing defense with multiple filter types"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize input smoothing defense
        
        Args:
            config: Smoothing configuration
        """
        self.config = config or {}
        
        # Smoothing parameters
        self.smoothing_type = self.config.get('smoothing_type', 'gaussian')
        self.kernel_size = self.config.get('kernel_size', 3)
        self.sigma = self.config.get('sigma', 1.0)
        self.median_kernel = self.config.get('median_kernel', 3)
        self.bilateral_d = self.config.get('bilateral_d', 9)
        self.bilateral_sigma_color = self.config.get('bilateral_sigma_color', 75)
        self.bilateral_sigma_space = self.config.get('bilateral_sigma_space', 75)
        
        # Adaptive parameters
        self.adaptive = self.config.get('adaptive', False)
        self.detection_threshold = self.config.get('detection_threshold', 0.8)
        
        # Statistics
        self.defense_stats = {
            'smoothing_applied': 0,
            'adaptive_triggered': 0,
            'total_samples': 0
        }
    
    def _detect_anomaly(self, images: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Detect potential adversarial examples
        
        Args:
            images: Input images
            model: Model for confidence scoring
            
        Returns:
            Boolean tensor indicating potential adversarial examples
        """
        with torch.no_grad():
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            max_probs, _ = probabilities.max(dim=1)
            
            # Low confidence indicates potential adversarial example
            is_suspicious = max_probs < self.detection_threshold
        
        return is_suspicious
    
    def _gaussian_smooth(self, images: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing"""
        smoothed = []
        
        for img in images:
            # Convert to numpy for OpenCV processing
            img_np = img.squeeze().cpu().numpy()
            
            # Apply Gaussian filter
            smoothed_np = cv2.GaussianBlur(
                img_np, 
                (self.kernel_size, self.kernel_size), 
                self.sigma
            )
            
            # Convert back to tensor
            smoothed_tensor = torch.from_numpy(smoothed_np).unsqueeze(0).unsqueeze(0)
            smoothed.append(smoothed_tensor)
        
        return torch.cat(smoothed, dim=0).to(images.device)
    
    def _median_smooth(self, images: torch.Tensor) -> torch.Tensor:
        """Apply median filtering"""
        smoothed = []
        
        for img in images:
            img_np = img.squeeze().cpu().numpy()
            smoothed_np = cv2.medianBlur(img_np, self.median_kernel)
            smoothed_tensor = torch.from_numpy(smoothed_np).unsqueeze(0).unsqueeze(0)
            smoothed.append(smoothed_tensor)
        
        return torch.cat(smoothed, dim=0).to(images.device)
    
    def _bilateral_smooth(self, images: torch.Tensor) -> torch.Tensor:
        """Apply bilateral filtering"""
        smoothed = []
        
        for img in images:
            img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
            smoothed_np = cv2.bilateralFilter(
                img_np,
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space
            )
            smoothed_np = smoothed_np.astype(np.float32) / 255.0
            smoothed_tensor = torch.from_numpy(smoothed_np).unsqueeze(0).unsqueeze(0)
            smoothed.append(smoothed_tensor)
        
        return torch.cat(smoothed, dim=0).to(images.device)
    
    def _adaptive_smooth(self, images: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Adaptive smoothing based on confidence
        
        Args:
            images: Input images
            model: Model for confidence scoring
            
        Returns:
            Smoothed images
        """
        # Detect suspicious samples
        is_suspicious = self._detect_anomaly(images, model)
        
        # Apply smoothing only to suspicious samples
        smoothed_images = images.clone()
        
        if is_suspicious.any():
            suspicious_indices = torch.where(is_suspicious)[0]
            suspicious_images = images[suspicious_indices]
            
            # Apply smoothing to suspicious images
            if self.smoothing_type == 'gaussian':
                smoothed_suspicious = self._gaussian_smooth(suspicious_images)
            elif self.smoothing_type == 'median':
                smoothed_suspicious = self._median_smooth(suspicious_images)
            elif self.smoothing_type == 'bilateral':
                smoothed_suspicious = self._bilateral_smooth(suspicious_images)
            else:
                smoothed_suspicious = suspicious_images
            
            # Replace suspicious images with smoothed versions
            smoothed_images[suspicious_indices] = smoothed_suspicious
            
            # Update statistics
            self.defense_stats['adaptive_triggered'] += len(suspicious_indices)
        
        return smoothed_images
    
    def apply(self, 
              images: torch.Tensor, 
              model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Apply input smoothing defense
        
        Args:
            images: Input images [batch, channels, height, width]
            model: Optional model for adaptive smoothing
            
        Returns:
            Smoothed images
        """
        self.defense_stats['total_samples'] += images.size(0)
        
        # Adaptive smoothing
        if self.adaptive and model is not None:
            smoothed_images = self._adaptive_smooth(images, model)
            self.defense_stats['smoothing_applied'] += images.size(0)
        else:
            # Standard smoothing
            if self.smoothing_type == 'gaussian':
                smoothed_images = self._gaussian_smooth(images)
            elif self.smoothing_type == 'median':
                smoothed_images = self._median_smooth(images)
            elif self.smoothing_type == 'bilateral':
                smoothed_images = self._bilateral_smooth(images)
            elif self.smoothing_type == 'none':
                smoothed_images = images
            else:
                raise ValueError(f"Unknown smoothing type: {self.smoothing_type}")
            
            self.defense_stats['smoothing_applied'] += images.size(0)
        
        return smoothed_images
    
    def evaluate_defense(self,
                        images: torch.Tensor,
                        adversarial_images: torch.Tensor,
                        model: nn.Module,
                        labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate defense effectiveness
        
        Args:
            images: Clean images
            adversarial_images: Adversarial images
            model: Target model
            labels: True labels
            
        Returns:
            Dictionary of defense metrics
        """
        model.eval()
        
        with torch.no_grad():
            # Clean accuracy (baseline)
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_acc = (clean_preds == labels).float().mean().item()
            
            # Adversarial accuracy (without defense)
            adv_outputs = model(adversarial_images)
            adv_preds = adv_outputs.argmax(dim=1)
            adv_acc = (adv_preds == labels).float().mean().item()
            
            # Apply defense to adversarial images
            defended_images = self.apply(adversarial_images, model)
            
            # Defended accuracy
            defended_outputs = model(defended_images)
            defended_preds = defended_outputs.argmax(dim=1)
            defended_acc = (defended_preds == labels).float().mean().item()
            
            # Calculate defense improvement
            improvement = defended_acc - adv_acc
            
            # Confidence metrics
            clean_confidence = F.softmax(clean_outputs, dim=1).max(dim=1)[0].mean().item()
            adv_confidence = F.softmax(adv_outputs, dim=1).max(dim=1)[0].mean().item()
            defended_confidence = F.softmax(defended_outputs, dim=1).max(dim=1)[0].mean().item()
        
        metrics = {
            'clean_accuracy': clean_acc * 100,
            'adversarial_accuracy': adv_acc * 100,
            'defended_accuracy': defended_acc * 100,
            'defense_improvement': improvement * 100,
            'clean_confidence': clean_confidence,
            'adversarial_confidence': adv_confidence,
            'defended_confidence': defended_confidence,
            'smoothing_type': self.smoothing_type,
            'adaptive': self.adaptive
        }
        
        return metrics
    
    def get_defense_stats(self) -> Dict[str, Any]:
        """Get defense statistics"""
        return self.defense_stats.copy()
    
    def __call__(self, images: torch.Tensor, model: Optional[nn.Module] = None) -> torch.Tensor:
        """Callable interface"""
        return self.apply(images, model)

def create_input_smoothing(smoothing_type: str = 'gaussian', **kwargs) -> InputSmoothing:
    """Factory function for creating input smoothing defense"""
    config = {'smoothing_type': smoothing_type, **kwargs}
    return InputSmoothing(config)
