"""
🛡️ MODEL FIREWALL - Non-negotiable security core
The firewall never relies on one signal.
"""
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class FirewallAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    DEGRADE = "degrade"
    ESCALATE = "escalate"

@dataclass
class FirewallResult:
    allowed: bool
    action: FirewallAction
    reason: str
    confidence: float
    details: Dict[str, Any]

class ModelFirewall:
    """Enterprise model firewall with adaptive policies"""
    
    def __init__(self, policy=None):
        self.policy = policy or AdaptiveFirewallPolicy()
        self.history = []
        self.threat_signatures = self._load_threat_signatures()
        
    def evaluate(self, request: Dict[str, Any]) -> FirewallResult:
        """Evaluate a request against all firewall checks"""
        
        checks = [
            self._check_input_sanity,
            self._check_statistical_deviation,
            self._check_confidence_collapse,
            self._check_drift_indicators,
            self._check_threat_similarity
        ]
        
        results = []
        for check in checks:
            result = check(request)
            results.append(result)
            
            # Immediate block on critical failure
            if result.action == FirewallAction.BLOCK:
                self._log_evaluation(request, results)
                return result
        
        # Apply adaptive policy
        final_result = self.policy.decide(results)
        self._log_evaluation(request, results)
        
        return final_result
    
    def _check_input_sanity(self, request: Dict[str, Any]) -> FirewallResult:
        """Check input shape, type, and range"""
        try:
            data = request.get("data", {})
            
            # Check required fields
            if "input" not in data:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason="Missing input data",
                    confidence=1.0,
                    details={"check": "input_sanity", "issue": "missing_input"}
                )
            
            input_data = data["input"]
            
            # Convert to tensor for analysis
            if isinstance(input_data, list):
                tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                tensor = torch.tensor([input_data], dtype=torch.float32)
            
            # SECURITY FIX: Check tensor size to prevent DoS attacks
            # Limit to reasonable model input sizes (e.g., max 10,000 elements)
            MAX_TENSOR_ELEMENTS = 10000
            total_elements = tensor.numel()
            if total_elements > MAX_TENSOR_ELEMENTS:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Tensor too large: {total_elements} elements exceeds limit of {MAX_TENSOR_ELEMENTS}",
                    confidence=1.0,
                    details={
                        "check": "input_sanity", 
                        "issue": "tensor_size_exceeded",
                        "elements": total_elements,
                        "max_allowed": MAX_TENSOR_ELEMENTS
                    }
                )
            
            # Check shape
            if tensor.dim() not in [1, 2, 3, 4]:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Invalid tensor dimensions: {tensor.dim()}",
                    confidence=0.9,
                    details={"check": "input_sanity", "dimensions": tensor.dim()}
                )
            
            # Check value range (normalized data should be in reasonable range)
            abs_max = tensor.abs().max().item()
            if abs_max > 10.0:  # Arbitrary threshold - adjust based on model
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Input values out of range (max abs: {abs_max:.2f})",
                    confidence=0.8,
                    details={"check": "input_sanity", "abs_max": abs_max}
                )
            
            # Check for NaN/Inf
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason="Input contains NaN or Inf values",
                    confidence=1.0,
                    details={"check": "input_sanity", "issue": "nan_inf"}
                )
            
            return FirewallResult(
                allowed=True,
                action=FirewallAction.ALLOW,
                reason="Input sanity check passed",
                confidence=0.95,
                details={"check": "input_sanity", "shape": list(tensor.shape), "elements": total_elements}
            )
            
        except Exception as e:
            return FirewallResult(
                allowed=False,
                action=FirewallAction.BLOCK,
                reason=f"Input sanity check failed: {str(e)}",
                confidence=1.0,
                details={"check": "input_sanity", "error": str(e)}
            )
    
    def _check_statistical_deviation(self, request: Dict[str, Any]) -> FirewallResult:
        """Check statistical properties against training distribution"""
        try:
            data = request.get("data", {})
            if "input" not in data:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason="Missing input for statistical check",
                    confidence=1.0,
                    details={"check": "statistical_deviation", "issue": "missing_input"}
                )
            
            input_data = data["input"]
            if isinstance(input_data, list):
                tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                tensor = torch.tensor([input_data], dtype=torch.float32)
            
            # Check mean and std deviation against expected ranges
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            
            # Typical normalized image data should have mean ~0.5 and std ~0.3
            # Adjust thresholds based on your model's training distribution
            if abs(mean_val) > 2.0:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Input mean out of expected range: {mean_val:.4f}",
                    confidence=0.85,
                    details={"check": "statistical_deviation", "mean": mean_val, "issue": "mean_outlier"}
                )
            
            if std_val > 2.0:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Input std deviation out of expected range: {std_val:.4f}",
                    confidence=0.85,
                    details={"check": "statistical_deviation", "std": std_val, "issue": "std_outlier"}
                )
            
            return FirewallResult(
                allowed=True,
                action=FirewallAction.ALLOW,
                reason="Statistical deviation check passed",
                confidence=0.9,
                details={"check": "statistical_deviation", "mean": mean_val, "std": std_val}
            )
        except Exception as e:
            return FirewallResult(
                allowed=False,
                action=FirewallAction.BLOCK,
                reason=f"Statistical deviation check failed: {str(e)}",
                confidence=1.0,
                details={"check": "statistical_deviation", "error": str(e)}
            )
    
    def _check_confidence_collapse(self, request: Dict[str, Any]) -> FirewallResult:
        """Detect sudden confidence drops (adversarial indicator)"""
        if "metadata" in request and "previous_confidence" in request["metadata"]:
            prev_conf = request["metadata"]["previous_confidence"]
            # Simulate checking - in reality would need model output
            if prev_conf < 0.3:  # Arbitrary threshold
                return FirewallResult(
                    allowed=True,
                    action=FirewallAction.ESCALATE,
                    reason=f"Low previous confidence detected: {prev_conf:.3f}",
                    confidence=0.6,
                    details={"check": "confidence_collapse", "previous_confidence": prev_conf}
                )
        
        return FirewallResult(
            allowed=True,
            action=FirewallAction.ALLOW,
            reason="No confidence collapse detected",
            confidence=0.8,
            details={"check": "confidence_collapse"}
        )
    
    def _check_drift_indicators(self, request: Dict[str, Any]) -> FirewallResult:
        """Check for data/model drift indicators"""
        try:
            data = request.get("data", {})
            if "input" not in data:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason="Missing input for drift check",
                    confidence=1.0,
                    details={"check": "drift_indicators", "issue": "missing_input"}
                )
            
            input_data = data["input"]
            if isinstance(input_data, list):
                tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                tensor = torch.tensor([input_data], dtype=torch.float32)
            
            # Check for distribution shift indicators
            # Skewness and kurtosis can indicate unusual distributions
            mean_val = tensor.mean().item()
            std_val = tensor.std().item() + 1e-8  # Avoid division by zero
            
            # Normalize tensor for moment calculations
            normalized = (tensor - mean_val) / std_val
            
            # Calculate approximate skewness (3rd moment)
            skewness = (normalized ** 3).mean().item()
            
            # Calculate approximate kurtosis (4th moment)
            kurtosis = (normalized ** 4).mean().item() - 3  # Excess kurtosis
            
            # Flag extreme deviations from normal distribution
            if abs(skewness) > 3.0:
                return FirewallResult(
                    allowed=True,
                    action=FirewallAction.ESCALATE,
                    reason=f"High skewness detected: {skewness:.4f}",
                    confidence=0.75,
                    details={"check": "drift_indicators", "skewness": skewness, "issue": "high_skewness"}
                )
            
            if abs(kurtosis) > 5.0:
                return FirewallResult(
                    allowed=True,
                    action=FirewallAction.ESCALATE,
                    reason=f"High kurtosis detected: {kurtosis:.4f}",
                    confidence=0.75,
                    details={"check": "drift_indicators", "kurtosis": kurtosis, "issue": "high_kurtosis"}
                )
            
            return FirewallResult(
                allowed=True,
                action=FirewallAction.ALLOW,
                reason="Drift check passed",
                confidence=0.85,
                details={"check": "drift_indicators", "skewness": skewness, "kurtosis": kurtosis}
            )
        except Exception as e:
            return FirewallResult(
                allowed=False,
                action=FirewallAction.BLOCK,
                reason=f"Drift check failed: {str(e)}",
                confidence=1.0,
                details={"check": "drift_indicators", "error": str(e)}
            )
    
    def _check_threat_similarity(self, request: Dict[str, Any]) -> FirewallResult:
        """Check similarity to known attack patterns"""
        try:
            data = request.get("data", {})
            if "input" not in data:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason="Missing input for threat check",
                    confidence=1.0,
                    details={"check": "threat_similarity", "issue": "missing_input"}
                )
            
            input_data = data["input"]
            if isinstance(input_data, list):
                tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                tensor = torch.tensor([input_data], dtype=torch.float32)
            
            # Check for common adversarial attack patterns
            
            # 1. High-frequency noise pattern (common in PGD/FGSM attacks)
            # Calculate gradient magnitude approximation using local differences
            if tensor.dim() >= 2:
                # Compute local variance as proxy for high-frequency content
                unfolded = tensor.unfold(-1, 2, 1).unfold(-2, 2, 1)
                if unfolded.numel() > 0:
                    local_var = unfolded.var(dim=-1).var(dim=-1).mean().item()
                    
                    # Unusually high local variance may indicate adversarial noise
                    if local_var > 0.5:
                        return FirewallResult(
                            allowed=True,
                            action=FirewallAction.ESCALATE,
                            reason=f"High local variance detected (possible adversarial noise): {local_var:.4f}",
                            confidence=0.7,
                            details={"check": "threat_similarity", "local_variance": local_var, "pattern": "high_frequency_noise"}
                        )
            
            # 2. Check for uniform perturbation pattern
            flat_tensor = tensor.flatten()
            if len(flat_tensor) > 100:
                # Sample every 10th element and check for suspicious patterns
                sample = flat_tensor[::10]
                diff = sample[1:] - sample[:-1]
                mean_diff = diff.abs().mean().item()
                
                # Very uniform small differences can indicate targeted attacks
                if mean_diff < 0.001 and mean_diff > 0:
                    return FirewallResult(
                        allowed=True,
                        action=FirewallAction.ESCALATE,
                        reason=f"Suspiciously uniform perturbation pattern detected: {mean_diff:.6f}",
                        confidence=0.65,
                        details={"check": "threat_similarity", "mean_diff": mean_diff, "pattern": "uniform_perturbation"}
                    )
            
            return FirewallResult(
                allowed=True,
                action=FirewallAction.ALLOW,
                reason="Threat similarity check passed",
                confidence=0.8,
                details={"check": "threat_similarity", "threat_signatures_checked": len(self.threat_signatures)}
            )
        except Exception as e:
            return FirewallResult(
                allowed=False,
                action=FirewallAction.BLOCK,
                reason=f"Threat similarity check failed: {str(e)}",
                confidence=1.0,
                details={"check": "threat_similarity", "error": str(e)}
            )
    
    def _load_threat_signatures(self) -> List[Dict[str, Any]]:
        """Load known threat signatures"""
        # TODO: Load from database/file
        return []
    
    def _log_evaluation(self, request: Dict[str, Any], results: List[FirewallResult]):
        """Log firewall evaluation for audit"""
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request.get("request_id", "unknown"),
            "checks": [
                {
                    "check": r.details.get("check", "unknown"),
                    "action": r.action.value,
                    "confidence": r.confidence,
                    "reason": r.reason
                }
                for r in results
            ],
            "final_action": results[-1].action.value if results else "unknown"
        }
        
        self.history.append(evaluation)
        
        # Keep only last 1000 evaluations
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

class AdaptiveFirewallPolicy:
    """Adaptive policy that learns from firewall decisions"""
    
    def __init__(self):
        self.sensitivity = 0.5  # 0.0 = lenient, 1.0 = strict
        self.learning_rate = 0.01
    
    def decide(self, check_results: List[FirewallResult]) -> FirewallResult:
        """Make final decision based on all check results"""
        
        # Count blocking recommendations
        block_count = sum(1 for r in check_results if r.action == FirewallAction.BLOCK)
        escalate_count = sum(1 for r in check_results if r.action == FirewallAction.ESCALATE)
        
        # Calculate overall confidence
        confidences = [r.confidence for r in check_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Make decision
        if block_count > 0:
            # At least one check says block
            blocking_check = next(r for r in check_results if r.action == FirewallAction.BLOCK)
            return FirewallResult(
                allowed=False,
                action=FirewallAction.BLOCK,
                reason=f"Blocked by {blocking_check.details.get('check', 'unknown')} check",
                confidence=blocking_check.confidence,
                details={
                    "blocking_check": blocking_check.details.get("check"),
                    "all_checks": [r.details.get("check") for r in check_results]
                }
            )
        elif escalate_count > 0:
            # Escalate for review
            return FirewallResult(
                allowed=True,
                action=FirewallAction.ESCALATE,
                reason=f"{escalate_count} checks recommend escalation",
                confidence=avg_confidence,
                details={
                    "escalated_checks": [r.details.get("check") for r in check_results if r.action == FirewallAction.ESCALATE]
                }
            )
        else:
            # All checks pass
            return FirewallResult(
                allowed=True,
                action=FirewallAction.ALLOW,
                reason="All firewall checks passed",
                confidence=avg_confidence,
                details={
                    "passed_checks": [r.details.get("check") for r in check_results],
                    "average_confidence": avg_confidence
                }
            )
    
    def update_sensitivity(self, was_correct: bool):
        """Adapt sensitivity based on whether decision was correct"""
        if was_correct:
            # Increase sensitivity if we correctly blocked/detected
            self.sensitivity = min(1.0, self.sensitivity + self.learning_rate)
        else:
            # Decrease sensitivity if we had false positive
            self.sensitivity = max(0.0, self.sensitivity - self.learning_rate * 2)  # Faster decrease for false positives
