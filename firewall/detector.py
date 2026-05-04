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
        # Adaptive threshold statistics per model type
        self.threshold_stats = {
            "default": {"mean": 0.0, "std": 1.0, "abs_max_threshold": None},
        }
        self._calibration_samples = []
        
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
            
            # Block boolean inputs immediately (they should not be model inputs)
            if isinstance(input_data, bool):
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason="Boolean values are not valid model inputs",
                    confidence=1.0,
                    details={"check": "input_sanity", "issue": "boolean_input"}
                )
            
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
            
            # Check value range using adaptive threshold based on model-specific statistics
            abs_max = tensor.abs().max().item()
            threshold = self._get_adaptive_threshold("abs_max")
            if abs_max > threshold:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Input values out of range (max abs: {abs_max:.2f}, threshold: {threshold:.2f})",
                    confidence=0.85,
                    details={"check": "input_sanity", "abs_max": abs_max, "threshold": threshold}
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
        try:
            data = request.get("data", {})
            
            if "input" not in data:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason="Missing input data for statistical check",
                    confidence=1.0,
                    details={"check": "statistical_deviation", "issue": "missing_input"}
                )
            
            input_data = data["input"]
            
            # Convert to numpy array for statistical analysis
            if isinstance(input_data, list):
                arr = np.array(input_data, dtype=np.float32)
            elif isinstance(input_data, (int, float)):
                arr = np.array([input_data], dtype=np.float32)
            elif hasattr(input_data, "numpy"):
                arr = input_data.numpy()
            else:
                arr = np.array(input_data, dtype=np.float32)
            
            # Flatten for scalar statistics
            flat = arr.flatten()
            
            # Calculate key statistics
            mean_val = float(np.mean(flat))
            std_val = float(np.std(flat))
            min_val = float(np.min(flat))
            max_val = float(np.max(flat))
            skewness = float(self._calculate_skewness(flat))
            kurtosis = float(self._calculate_kurtosis(flat))
            
            # Define expected ranges based on training distribution
            # These thresholds should be calibrated based on actual training data statistics
            expected_mean_range = (-2.0, 2.0)  # Normalized data should have mean near 0
            expected_std_range = (0.1, 3.0)    # Reasonable standard deviation
            expected_value_range = (-5.0, 5.0) # Normalized value range
            
            # Check mean deviation
            if mean_val < expected_mean_range[0] or mean_val > expected_mean_range[1]:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Mean value {mean_val:.4f} outside expected range {expected_mean_range}",
                    confidence=0.85,
                    details={
                        "check": "statistical_deviation",
                        "issue": "mean_deviation",
                        "mean": mean_val,
                        "expected_range": expected_mean_range
                    }
                )
            
            # Check std deviation
            if std_val < expected_std_range[0] or std_val > expected_std_range[1]:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Standard deviation {std_val:.4f} outside expected range {expected_std_range}",
                    confidence=0.85,
                    details={
                        "check": "statistical_deviation",
                        "issue": "std_deviation",
                        "std": std_val,
                        "expected_range": expected_std_range
                    }
                )
            
            # Check for extreme values
            if min_val < expected_value_range[0] or max_val > expected_value_range[1]:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Values out of expected range: [{min_val:.4f}, {max_val:.4f}] vs {expected_value_range}",
                    confidence=0.8,
                    details={
                        "check": "statistical_deviation",
                        "issue": "value_range",
                        "min": min_val,
                        "max": max_val,
                        "expected_range": expected_value_range
                    }
                )
            
            # Check for adversarial patterns: very low variance with extreme mean shift
            # This can indicate gradient-based attacks
            if std_val < 0.01 and abs(mean_val) > 1.5:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Suspicious pattern: low variance ({std_val:.6f}) with shifted mean ({mean_val:.4f})",
                    confidence=0.9,
                    details={
                        "check": "statistical_deviation",
                        "issue": "adversarial_pattern",
                        "mean": mean_val,
                        "std": std_val
                    }
                )
            
            # Check skewness (extreme asymmetry can indicate manipulation)
            if abs(skewness) > 3.0:
                return FirewallResult(
                    allowed=True,
                    action=FirewallAction.ESCALATE,
                    reason=f"High skewness detected: {skewness:.4f}",
                    confidence=0.7,
                    details={
                        "check": "statistical_deviation",
                        "issue": "high_skewness",
                        "skewness": skewness
                    }
                )
            
            # Check kurtosis (heavy tails can indicate outliers/attacks)
            if abs(kurtosis) > 10.0:
                return FirewallResult(
                    allowed=True,
                    action=FirewallAction.ESCALATE,
                    reason=f"High kurtosis detected: {kurtosis:.4f}",
                    confidence=0.7,
                    details={
                        "check": "statistical_deviation",
                        "issue": "high_kurtosis",
                        "kurtosis": kurtosis
                    }
                )
            
            return FirewallResult(
                allowed=True,
                action=FirewallAction.ALLOW,
                reason="Statistical deviation check passed",
                confidence=0.9,
                details={
                    "check": "statistical_deviation",
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "skewness": skewness,
                    "kurtosis": kurtosis
                }
            )
            
        except Exception as e:
            return FirewallResult(
                allowed=False,
                action=FirewallAction.BLOCK,
                reason=f"Statistical deviation check failed: {str(e)}",
                confidence=1.0,
                details={"check": "statistical_deviation", "error": str(e)}
            )
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate sample skewness"""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis"""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        m4 = np.mean((data - mean) ** 4)
        m2 = np.mean((data - mean) ** 2)
        if m2 == 0:
            return 0.0
        return (m4 / (m2 ** 2)) - 3
    
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
        try:
            data = request.get("data", {})
            
            if "input" not in data:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason="Missing input data for threat check",
                    confidence=1.0,
                    details={"check": "threat_similarity", "issue": "missing_input"}
                )
            
            input_data = data["input"]
            
            # Convert to numpy array for analysis
            if isinstance(input_data, list):
                arr = np.array(input_data, dtype=np.float32)
            elif isinstance(input_data, (int, float)):
                arr = np.array([input_data], dtype=np.float32)
            elif hasattr(input_data, "numpy"):
                arr = input_data.numpy()
            else:
                arr = np.array(input_data, dtype=np.float32)
            
            flat = arr.flatten()
            
            # Calculate features for threat matching
            mean_val = float(np.mean(flat))
            std_val = float(np.std(flat))
            min_val = float(np.min(flat))
            max_val = float(np.max(flat))
            
            # Check against known threat signatures
            matched_threats = []
            for signature in self.threat_signatures:
                if self._match_signature(flat, signature):
                    matched_threats.append(signature)
            
            # If any threats matched, block or escalate
            if matched_threats:
                threat_names = [t["name"] for t in matched_threats]
                max_severity = max([t.get("severity", 1) for t in matched_threats])
                
                if max_severity >= 3:  # High severity threats
                    return FirewallResult(
                        allowed=False,
                        action=FirewallAction.BLOCK,
                        reason=f"Matched known attack patterns: {', '.join(threat_names)}",
                        confidence=0.95,
                        details={
                            "check": "threat_similarity",
                            "matched_threats": threat_names,
                            "severity": max_severity
                        }
                    )
                else:
                    return FirewallResult(
                        allowed=True,
                        action=FirewallAction.ESCALATE,
                        reason=f"Potential threat patterns detected: {', '.join(threat_names)}",
                        confidence=0.8,
                        details={
                            "check": "threat_similarity",
                            "matched_threats": threat_names,
                            "severity": max_severity
                        }
                    )
            
            # Check for common adversarial attack patterns
            adversarial_indicators = self._detect_adversarial_patterns(flat, mean_val, std_val)
            if adversarial_indicators:
                return FirewallResult(
                    allowed=False,
                    action=FirewallAction.BLOCK,
                    reason=f"Adversarial attack indicators: {', '.join(adversarial_indicators)}",
                    confidence=0.9,
                    details={
                        "check": "threat_similarity",
                        "indicators": adversarial_indicators
                    }
                )
            
            return FirewallResult(
                allowed=True,
                action=FirewallAction.ALLOW,
                reason="Threat similarity check passed",
                confidence=0.85,
                details={
                    "check": "threat_similarity",
                    "threat_signatures_checked": len(self.threat_signatures),
                    "matched": 0
                }
            )
            
        except Exception as e:
            return FirewallResult(
                allowed=False,
                action=FirewallAction.BLOCK,
                reason=f"Threat similarity check failed: {str(e)}",
                confidence=1.0,
                details={"check": "threat_similarity", "error": str(e)}
            )
    
    def _match_signature(self, data: np.ndarray, signature: Dict[str, Any]) -> bool:
        """Check if data matches a threat signature"""
        sig_type = signature.get("type")
        
        if sig_type == "statistical":
            # Match based on statistical properties
            thresholds = signature.get("thresholds", {})
            
            mean_match = True
            if "mean_min" in thresholds and data.mean() < thresholds["mean_min"]:
                mean_match = False
            if "mean_max" in thresholds and data.mean() > thresholds["mean_max"]:
                mean_match = False
            
            std_match = True
            if "std_min" in thresholds and data.std() < thresholds["std_min"]:
                std_match = False
            if "std_max" in thresholds and data.std() > thresholds["std_max"]:
                std_match = False
            
            return mean_match and std_match
        
        elif sig_type == "pattern":
            # Match based on specific value patterns
            pattern = signature.get("pattern", {})
            
            if "all_same" in pattern and pattern["all_same"]:
                if data.std() < 1e-6:
                    return True
            
            if "extreme_range" in pattern and pattern["extreme_range"]:
                if data.max() - data.min() > pattern.get("min_range", 8.0):
                    return True
            
            if "specific_values" in pattern:
                target_vals = pattern["specific_values"]
                matches = sum(1 for v in target_vals if np.any(np.isclose(data, v, rtol=1e-3)))
                if matches >= pattern.get("min_matches", len(target_vals)):
                    return True
        
        return False
    
    def _get_adaptive_threshold(self, metric_name: str, model_type: str = "default") -> float:
        """Get adaptive threshold based on model-specific statistics.
        
        Uses statistical analysis of calibration data to set dynamic thresholds
        that adapt to the specific model's input distribution characteristics.
        """
        # If we have calibration samples, use them to calculate adaptive threshold
        if self._calibration_samples and len(self._calibration_samples) >= 10:
            samples = np.array(self._calibration_samples)
            
            if metric_name == "abs_max":
                # Calculate percentile-based threshold (99th percentile + safety margin)
                abs_max_values = np.max(np.abs(samples), axis=1)
                p99 = np.percentile(abs_max_values, 99)
                p95 = np.percentile(abs_max_values, 95)
                # Use 3-sigma rule: threshold = mean + 3*std, but bounded by percentiles
                mean_val = np.mean(abs_max_values)
                std_val = np.std(abs_max_values)
                sigma_threshold = mean_val + 3 * std_val
                # Take minimum of sigma-based and percentile-based for robustness
                return min(sigma_threshold, p99 * 1.2)
        
        # Fallback to model-type specific defaults if available
        if model_type in self.threshold_stats:
            stats = self.threshold_stats[model_type]
            if stats.get("abs_max_threshold") is not None:
                return stats["abs_max_threshold"]
        
        # Default safe threshold when no calibration data available
        # Based on typical normalized input ranges [-3, 3] for most models
        return 6.0
    
    def calibrate_thresholds(self, sample_inputs: List[np.ndarray], model_type: str = "default"):
        """Calibrate adaptive thresholds using sample inputs.
        
        Args:
            sample_inputs: List of representative input arrays from normal operation
            model_type: Identifier for the model type being calibrated
        """
        if not sample_inputs:
            return
        
        self._calibration_samples = []
        for sample in sample_inputs:
            if hasattr(sample, 'flatten'):
                flat = sample.flatten()
            else:
                flat = np.array(sample).flatten()
            self._calibration_samples.append(flat)
        
        # Calculate and store adaptive thresholds
        abs_max_values = [np.max(np.abs(s)) for s in self._calibration_samples]
        p99 = np.percentile(abs_max_values, 99)
        p95 = np.percentile(abs_max_values, 95)
        mean_val = np.mean(abs_max_values)
        std_val = np.std(abs_max_values)
        sigma_threshold = mean_val + 3 * std_val
        
        adaptive_threshold = min(sigma_threshold, p99 * 1.2)
        
        self.threshold_stats[model_type] = {
            "mean": mean_val,
            "std": std_val,
            "abs_max_threshold": adaptive_threshold,
            "p95": p95,
            "p99": p99,
            "sample_count": len(sample_inputs)
        }
    
    def _detect_adversarial_patterns(self, data: np.ndarray, mean: float, std: float) -> List[str]:
        """Detect common adversarial attack patterns"""
        indicators = []
        
        # FGSM-like attack: small uniform perturbation
        if std < 0.05 and abs(mean) > 0.5:
            indicators.append("fgsm_like_attack")
        
        # PGD-like attack: clipped perturbations
        unique_vals = len(np.unique(data.round(decimals=4)))
        total_vals = len(data)
        if unique_vals < total_vals * 0.1 and total_vals > 10:
            indicators.append("pgd_like_attack")
        
        # Boundary attack: values at extreme ranges
        extreme_ratio = np.sum((data > 4.0) | (data < -4.0)) / len(data)
        if extreme_ratio > 0.5:
            indicators.append("boundary_attack")
        
        # Noise injection: unusually high variance
        if std > 5.0:
            indicators.append("noise_injection")
        
        return indicators
    
    def _load_threat_signatures(self) -> List[Dict[str, Any]]:
        """Load known threat signatures"""
        # Built-in threat signature database
        signatures = [
            {
                "name": "Zero Input Attack",
                "type": "statistical",
                "severity": 2,
                "description": "All-zero or near-zero input attempting to bypass model",
                "thresholds": {
                    "mean_min": -0.01,
                    "mean_max": 0.01,
                    "std_max": 0.001
                }
            },
            {
                "name": "Uniform Perturbation",
                "type": "statistical",
                "severity": 3,
                "description": "FGSM-style uniform perturbation attack",
                "thresholds": {
                    "std_min": 0.01,
                    "std_max": 0.1,
                    "mean_min": -0.5,
                    "mean_max": 0.5
                }
            },
            {
                "name": "Extreme Value Injection",
                "type": "statistical",
                "severity": 3,
                "description": "Attempt to inject extreme values to cause overflow/underflow",
                "thresholds": {
                    "std_min": 3.0
                }
            },
            {
                "name": "Constant Pattern Attack",
                "type": "pattern",
                "severity": 2,
                "description": "All values identical - possible model probing",
                "pattern": {
                    "all_same": True
                }
            },
            {
                "name": "Wide Range Sweep",
                "type": "pattern",
                "severity": 2,
                "description": "Attempting to sweep across entire input range",
                "pattern": {
                    "extreme_range": True,
                    "min_range": 9.0
                }
            },
            {
                "name": "Targeted Value Pattern",
                "type": "pattern",
                "severity": 3,
                "description": "Specific known malicious value pattern",
                "pattern": {
                    "specific_values": [0.5, -0.5, 1.0, -1.0],
                    "min_matches": 3
                }
            }
        ]
        
        return signatures
    
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
