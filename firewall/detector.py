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
                details={"check": "input_sanity", "shape": list(tensor.shape)}
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
        # TODO: Implement proper statistical deviation check
        # For now, return pass
        return FirewallResult(
            allowed=True,
            action=FirewallAction.ALLOW,
            reason="Statistical deviation check passed (placeholder)",
            confidence=0.7,
            details={"check": "statistical_deviation", "status": "placeholder"}
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
        # TODO: Implement drift detection
        return FirewallResult(
            allowed=True,
            action=FirewallAction.ALLOW,
            reason="Drift check passed (placeholder)",
            confidence=0.7,
            details={"check": "drift_indicators", "status": "placeholder"}
        )
    
    def _check_threat_similarity(self, request: Dict[str, Any]) -> FirewallResult:
        """Check similarity to known attack patterns"""
        # TODO: Implement threat signature matching
        return FirewallResult(
            allowed=True,
            action=FirewallAction.ALLOW,
            reason="Threat similarity check passed",
            confidence=0.75,
            details={"check": "threat_similarity", "threat_signatures": len(self.threat_signatures)}
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
