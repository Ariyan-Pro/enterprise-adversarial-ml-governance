"""
Model Wrappers for Enhanced Defenses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple

class ModelWrapper(nn.Module):
    """
    Base wrapper class for model defenses
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.config = config or {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward pass"""
        return self.model(x)
    
    def predict_with_defense(self, x: torch.Tensor) -> torch.Tensor:
        """Predict with defense mechanism"""
        return self.forward(x)

class EnsembleModelWrapper(ModelWrapper):
    """
    Ensemble of models for improved robustness
    """
    
    def __init__(self, models: List[nn.Module], config: Optional[Dict[str, Any]] = None):
        super().__init__(models[0], config)  # Use first model as base
        self.models = nn.ModuleList(models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models"""
        outputs = []
        for model in self.models:
            outputs.append(model(x).unsqueeze(0))
        
        outputs = torch.cat(outputs, dim=0)
        return outputs.mean(dim=0)

class DistillationWrapper(ModelWrapper):
    """
    Knowledge distillation wrapper
    """
    
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, 
                 temperature: float = 3.0, config: Optional[Dict[str, Any]] = None):
        super().__init__(student_model, config)
        self.teacher = teacher_model
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with distillation"""
        student_logits = self.model(x)
        
        if self.training:
            with torch.no_grad():
                teacher_logits = self.teacher(x)
            return student_logits, teacher_logits
        else:
            return student_logits

class AdversarialDetectorWrapper(ModelWrapper):
    """
    Wrapper that detects adversarial examples
    """
    
    def __init__(self, model: nn.Module, detection_threshold: float = 0.7,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(model, config)
        self.detection_threshold = detection_threshold
        self.detector = self._create_detector()
        
    def _create_detector(self) -> nn.Module:
        """Create simple detector based on prediction consistency"""
        return nn.Sequential(
            nn.Linear(10, 32),  # Assuming 10 classes
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predictions and detection scores"""
        predictions = self.model(x)
        
        # Simple detection based on prediction confidence
        probs = F.softmax(predictions, dim=1)
        max_probs, _ = probs.max(dim=1)
        detection_scores = 1.0 - max_probs  # Lower confidence = higher detection score
        
        return predictions, detection_scores
    
    def is_adversarial(self, x: torch.Tensor) -> torch.Tensor:
        """Check if input is adversarial"""
        _, detection_scores = self.forward(x)
        return detection_scores > self.detection_threshold

# Factory functions
def create_ensemble_wrapper(models: List[nn.Module], **kwargs) -> EnsembleModelWrapper:
    """Create ensemble wrapper"""
    return EnsembleModelWrapper(models, kwargs)

def create_distillation_wrapper(student: nn.Module, teacher: nn.Module, **kwargs) -> DistillationWrapper:
    """Create distillation wrapper"""
    return DistillationWrapper(student, teacher, **kwargs)

def create_detector_wrapper(model: nn.Module, **kwargs) -> AdversarialDetectorWrapper:
    """Create adversarial detector wrapper"""
    return AdversarialDetectorWrapper(model, **kwargs)
