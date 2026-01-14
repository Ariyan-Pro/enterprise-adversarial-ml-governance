"""
Randomized Transformation Defense
Applies random transformations to inputs to increase robustness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

class RandomizedTransformDefense(nn.Module):
    """
    Defense using randomized transformations
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        super(RandomizedTransformDefense, self).__init__()
        self.model = model
        self.config = config or {}
        
        # Transformation parameters
        self.rotation_range = self.config.get('rotation_range', 15)
        self.translation_range = self.config.get('translation_range', 0.1)
        self.scale_range = self.config.get('scale_range', (0.9, 1.1))
        self.num_samples = self.config.get('num_samples', 10)
        
        self.model.eval()
    
    def apply_random_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random affine transformation
        """
        batch_size = x.size(0)
        
        # Generate random transformation parameters
        angle = torch.rand(batch_size) * self.rotation_range * 2 - self.rotation_range
        translate_x = torch.rand(batch_size) * self.translation_range * 2 - self.translation_range
        translate_y = torch.rand(batch_size) * self.translation_range * 2 - self.translation_range
        scale = torch.rand(batch_size) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        
        # Create transformation matrix
        theta = torch.zeros(batch_size, 2, 3)
        for i in range(batch_size):
            # Rotation matrix
            rot_rad = angle[i] * np.pi / 180.0
            cos_a, sin_a = np.cos(rot_rad), np.sin(rot_rad)
            
            # Scale matrix
            s = scale[i]
            
            # Combined transformation
            theta[i, 0, 0] = cos_a * s
            theta[i, 0, 1] = -sin_a * s
            theta[i, 0, 2] = translate_x[i]
            theta[i, 1, 0] = sin_a * s
            theta[i, 1, 1] = cos_a * s
            theta[i, 1, 2] = translate_y[i]
        
        # Apply grid sampling
        grid = F.affine_grid(theta, x.size(), align_corners=False).to(x.device)
        transformed = F.grid_sample(x, grid, align_corners=False)
        
        return transformed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with randomized smoothing
        """
        if self.training:
            # During training, apply random transform
            x = self.apply_random_transform(x)
            return self.model(x)
        else:
            # During evaluation, use multiple samples
            outputs = []
            for _ in range(self.num_samples):
                x_transformed = self.apply_random_transform(x)
                output = self.model(x_transformed)
                outputs.append(output.unsqueeze(0))
            
            # Average predictions
            outputs = torch.cat(outputs, dim=0)
            return outputs.mean(dim=0)
    
    def predict_with_confidence(self, x: torch.Tensor, num_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with confidence estimation
        """
        predictions = []
        for _ in range(num_samples):
            x_transformed = self.apply_random_transform(x)
            output = self.model(x_transformed)
            predictions.append(output.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        confidence = predictions.std(dim=0).mean(dim=1)  # Average std across classes
        
        return mean_prediction, confidence

# For backward compatibility
RandomizedTransform = RandomizedTransformDefense

def create_randomized_transform(model: nn.Module, **kwargs) -> RandomizedTransformDefense:
    """Factory function for randomized transform defense"""
    return RandomizedTransformDefense(model, kwargs)
