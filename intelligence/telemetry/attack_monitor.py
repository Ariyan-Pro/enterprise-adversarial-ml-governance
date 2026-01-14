"""
🔍 ADVERSARIAL INTELLIGENCE CORE - REAL IMPLEMENTATION
Attacks are signals, not failures.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
import hashlib

class AttackTelemetry:
    """Real attack pattern analysis and threat scoring"""
    
    def __init__(self, telemetry_dir: str = "intelligence/telemetry"):
        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory telemetry storage
        self.inference_log: List[Dict] = []
        self.attack_patterns: List[Dict] = []
        self.threat_signatures: Dict[str, Dict] = {}
        
        # Load existing telemetry
        self._load_telemetry()
    
    def record_inference(self, request_id: str, request: Dict, prediction: Dict):
        """Record inference with adversarial indicators"""
        telemetry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "input_hash": self._hash_input(request.get("data", {})),
            "prediction_confidence": prediction.get("confidence", 0.0),
            "prediction_class": prediction.get("class", -1),
            "inference_time_ms": prediction.get("inference_time_ms", 0),
            "adversarial_indicators": self._extract_indicators(request, prediction)
        }
        
        self.inference_log.append(telemetry)
        
        # Check for attack patterns
        self._analyze_for_attacks(telemetry)
        
        # Save to disk
        self._save_telemetry()
    
    def record_attack(self, attack_type: str, success: bool, 
                     request: Dict, original_pred: Dict, adversarial_pred: Dict):
        """Record a confirmed attack attempt"""
        attack_record = {
            "timestamp": datetime.now().isoformat(),
            "attack_type": attack_type,
            "success": success,
            "original_confidence": original_pred.get("confidence", 0.0),
            "adversarial_confidence": adversarial_pred.get("confidence", 0.0),
            "confidence_drop": original_pred.get("confidence", 0.0) - adversarial_pred.get("confidence", 0.0),
            "input_signature": self._extract_attack_signature(request),
            "metadata": {
                "model": request.get("model", "unknown"),
                "domain": request.get("domain", "unknown")
            }
        }
        
        self.attack_patterns.append(attack_record)
        
        # Update threat signatures
        self._update_threat_signatures(attack_record)
        
        # Generate immediate alert for high-confidence attacks
        if attack_record["confidence_drop"] > 0.5:  # 50% confidence drop
            self._generate_attack_alert(attack_record)
    
    def generate_threat_report(self, timeframe_hours: int = 24) -> Dict:
        """Generate real threat intelligence report"""
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        # Filter recent data
        recent_attacks = [
            a for a in self.attack_patterns
            if datetime.fromisoformat(a["timestamp"]) > cutoff_time
        ]
        
        recent_inferences = [
            i for i in self.inference_log
            if datetime.fromisoformat(i["timestamp"]) > cutoff_time
        ]
        
        # Calculate threat metrics
        attack_success_rate = self._calculate_attack_success_rate(recent_attacks)
        threat_score = self._calculate_threat_score(recent_attacks, recent_inferences)
        top_attack_types = self._get_top_attack_types(recent_attacks)
        
        # Cluster similar attacks
        attack_clusters = self._cluster_attacks(recent_attacks)
        
        return {
            "report_time": datetime.now().isoformat(),
            "timeframe_hours": timeframe_hours,
            "summary": {
                "total_inferences": len(recent_inferences),
                "total_attacks_detected": len(recent_attacks),
                "attack_success_rate": attack_success_rate,
                "overall_threat_score": threat_score,
                "top_attack_types": top_attack_types
            },
            "detailed_analysis": {
                "attack_clusters": attack_clusters,
                "confidence_trends": self._analyze_confidence_trends(recent_inferences),
                "temporal_patterns": self._analyze_temporal_patterns(recent_attacks)
            },
            "recommendations": self._generate_recommendations(recent_attacks)
        }
    
    def get_attack_statistics(self) -> Dict:
        """Get real-time attack statistics"""
        total_attacks = len(self.attack_patterns)
        successful_attacks = sum(1 for a in self.attack_patterns if a["success"])
        
        attacks_by_type = defaultdict(int)
        for attack in self.attack_patterns:
            attacks_by_type[attack["attack_type"]] += 1
        
        # Calculate average confidence drop
        if self.attack_patterns:
            avg_drop = sum(a["confidence_drop"] for a in self.attack_patterns) / len(self.attack_patterns)
        else:
            avg_drop = 0.0
        
        return {
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "success_rate": successful_attacks / total_attacks if total_attacks > 0 else 0,
            "attacks_by_type": dict(attacks_by_type),
            "average_confidence_drop": avg_drop,
            "threat_signatures_count": len(self.threat_signatures)
        }
    
    def _extract_indicators(self, request: Dict, prediction: Dict) -> Dict:
        """Extract adversarial indicators from request and prediction"""
        indicators = {}
        
        # Confidence anomaly
        confidence = prediction.get("confidence", 0.0)
        if confidence < 0.3:
            indicators["low_confidence"] = confidence
        
        # Input statistics
        data = request.get("data", {}).get("input", [])
        if isinstance(data, list) and len(data) > 0:
            data_array = np.array(data)
            indicators["input_stats"] = {
                "mean": float(np.mean(data_array)),
                "std": float(np.std(data_array)),
                "max": float(np.max(data_array)),
                "min": float(np.min(data_array))
            }
        
        # Check for suspicious patterns (placeholder for real pattern detection)
        if "metadata" in request and "suspicious" in request["metadata"]:
            indicators["suspicious_metadata"] = True
        
        return indicators
    
    def _analyze_for_attacks(self, telemetry: Dict):
        """Analyze telemetry for attack patterns"""
        indicators = telemetry["adversarial_indicators"]
        
        # Simple heuristic: very low confidence + unusual input stats
        if indicators.get("low_confidence", 1.0) < 0.2:
            stats = indicators.get("input_stats", {})
            if stats.get("std", 0) > 0.5:  # High variance
                self._flag_potential_attack(telemetry)
    
    def _flag_potential_attack(self, telemetry: Dict):
        """Flag a potential attack for further investigation"""
        flag_record = {
            **telemetry,
            "flagged_as": "potential_attack",
            "review_status": "pending"
        }
        
        # Save flagged record
        flag_file = self.telemetry_dir / "flagged_attacks.jsonl"
        with open(flag_file, "a") as f:
            f.write(json.dumps(flag_record) + "\n")
    
    def _hash_input(self, data: Any) -> str:
        """Create hash of input data for deduplication"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _extract_attack_signature(self, request: Dict) -> Dict:
        """Extract signature features from attack request"""
        data = request.get("data", {})
        signature = {
            "input_shape": self._get_shape(data.get("input", [])),
            "feature_stats": self._calculate_feature_stats(data.get("input", [])),
            "request_pattern": {
                "has_metadata": "metadata" in request,
                "has_multiple_inputs": isinstance(data.get("input"), list) and len(data.get("input", [])) > 1
            }
        }
        return signature
    
    def _get_shape(self, data: Any) -> List[int]:
        """Get shape of input data"""
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                return [len(data), len(data[0])]
            return [len(data)]
        return []
    
    def _calculate_feature_stats(self, data: Any) -> Dict:
        """Calculate statistical features"""
        if not data or not isinstance(data, list):
            return {}
        
        try:
            flat_data = np.array(data).flatten()
            return {
                "mean": float(np.mean(flat_data)),
                "std": float(np.std(flat_data)),
                "skew": float(self._safe_skew(flat_data)),
                "kurtosis": float(self._safe_kurtosis(flat_data))
            }
        except:
            return {}
    
    def _safe_skew(self, data):
        """Calculate skewness safely"""
        from scipy.stats import skew
        try:
            return skew(data) if len(data) > 0 else 0.0
        except:
            return 0.0
    
    def _safe_kurtosis(self, data):
        """Calculate kurtosis safely"""
        from scipy.stats import kurtosis
        try:
            return kurtosis(data) if len(data) > 0 else 0.0
        except:
            return 0.0
    
    def _update_threat_signatures(self, attack_record: Dict):
        """Update threat signatures database"""
        signature_hash = hashlib.sha256(
            json.dumps(attack_record["input_signature"], sort_keys=True).encode()
        ).hexdigest()[:12]
        
        if signature_hash not in self.threat_signatures:
            self.threat_signatures[signature_hash] = {
                "first_seen": attack_record["timestamp"],
                "last_seen": attack_record["timestamp"],
                "attack_type": attack_record["attack_type"],
                "occurrences": 1,
                "signature": attack_record["input_signature"]
            }
        else:
            self.threat_signatures[signature_hash]["last_seen"] = attack_record["timestamp"]
            self.threat_signatures[signature_hash]["occurrences"] += 1
    
    def _generate_attack_alert(self, attack_record: Dict):
        """Generate alert for serious attack"""
        alert = {
            "alert_type": "HIGH_CONFIDENCE_ATTACK",
            "timestamp": datetime.now().isoformat(),
            "severity": "HIGH",
            "attack_details": attack_record,
            "recommended_action": "Review firewall thresholds and consider model retraining"
        }
        
        # Save alert
        alert_file = self.telemetry_dir / "alerts.jsonl"
        with open(alert_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
        
        # Log to console
        print(f"🚨 HIGH SEVERITY ATTACK ALERT: {attack_record['attack_type']} "
              f"caused {attack_record['confidence_drop']:.1%} confidence drop")
    
    def _calculate_attack_success_rate(self, attacks: List[Dict]) -> float:
        """Calculate attack success rate"""
        if not attacks:
            return 0.0
        successful = sum(1 for a in attacks if a["success"])
        return successful / len(attacks)
    
    def _calculate_threat_score(self, attacks: List[Dict], inferences: List[Dict]) -> float:
        """Calculate composite threat score (0-100)"""
        if not inferences:
            return 0.0
        
        # Attack frequency component
        attack_rate = len(attacks) / len(inferences) if inferences else 0
        
        # Attack success component
        success_rate = self._calculate_attack_success_rate(attacks)
        
        # Confidence drop component
        avg_drop = sum(a["confidence_drop"] for a in attacks) / len(attacks) if attacks else 0
        
        # Composite score
        threat_score = (attack_rate * 40 + success_rate * 30 + avg_drop * 30)
        return min(100.0, threat_score * 100)
    
    def _get_top_attack_types(self, attacks: List[Dict]) -> List[Dict]:
        """Get top attack types by frequency"""
        from collections import Counter
        attack_types = Counter(a["attack_type"] for a in attacks)
        return [
            {"type": atype, "count": count}
            for atype, count in attack_types.most_common(5)
        ]
    
    def _cluster_attacks(self, attacks: List[Dict]) -> List[Dict]:
        """Cluster similar attacks using feature vectors"""
        if len(attacks) < 2:
            return []
        
        # Extract feature vectors
        features = []
        for attack in attacks:
            sig = attack["input_signature"]
            stats = sig.get("feature_stats", {})
            feat = [
                stats.get("mean", 0),
                stats.get("std", 0),
                stats.get("skew", 0),
                stats.get("kurtosis", 0)
            ]
            features.append(feat)
        
        # Cluster using DBSCAN
        try:
            features_array = np.array(features)
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(features_array)
            
            clusters = defaultdict(list)
            for idx, label in enumerate(clustering.labels_):
                if label != -1:  # -1 means noise in DBSCAN
                    clusters[label].append(attacks[idx]["attack_type"])
            
            return [
                {"cluster_id": cid, "attack_types": list(set(types)), "size": len(types)}
                for cid, types in clusters.items()
            ]
        except:
            return []
    
    def _analyze_confidence_trends(self, inferences: List[Dict]) -> Dict:
        """Analyze confidence trends over time"""
        if not inferences:
            return {}
        
        # Group by hour
        hourly_confidences = defaultdict(list)
        for inf in inferences:
            dt = datetime.fromisoformat(inf["timestamp"])
            hour_key = dt.strftime("%Y-%m-%d %H:00")
            hourly_confidences[hour_key].append(inf["prediction_confidence"])
        
        # Calculate hourly averages
        hourly_avg = {
            hour: sum(confs) / len(confs)
            for hour, confs in hourly_confidences.items()
        }
        
        return {
            "hourly_averages": hourly_avg,
            "overall_average": sum(inf["prediction_confidence"] for inf in inferences) / len(inferences),
            "confidence_volatility": np.std([inf["prediction_confidence"] for inf in inferences]) if inferences else 0
        }
    
    def _analyze_temporal_patterns(self, attacks: List[Dict]) -> Dict:
        """Analyze temporal patterns in attacks"""
        if not attacks:
            return {}
        
        # Attacks by hour of day
        hourly_counts = defaultdict(int)
        for attack in attacks:
            dt = datetime.fromisoformat(attack["timestamp"])
            hour = dt.hour
            hourly_counts[hour] += 1
        
        return {
            "attacks_by_hour": dict(hourly_counts),
            "peak_attack_hour": max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None,
            "total_attack_period_hours": len(set(
                datetime.fromisoformat(a["timestamp"]).strftime("%Y-%m-%d %H")
                for a in attacks
            ))
        }
    
    def _generate_recommendations(self, attacks: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not attacks:
            recommendations.append("No attacks detected in timeframe. Maintain current security posture.")
            return recommendations
        
        # Analyze attack patterns
        fgsm_count = sum(1 for a in attacks if a["attack_type"] == "FGSM")
        pgd_count = sum(1 for a in attacks if a["attack_type"] == "PGD")
        cw_count = sum(1 for a in attacks if a["attack_type"] == "C&W")
        
        if fgsm_count > 0:
            recommendations.append(
                f"FGSM attacks detected ({fgsm_count} occurrences). "
                "Consider implementing gradient masking or input preprocessing."
            )
        
        if pgd_count > 0:
            recommendations.append(
                f"PGD attacks detected ({pgd_count} occurrences). "
                "Consider adversarial training or certified defenses."
            )
        
        if cw_count > 0:
            recommendations.append(
                f"C&W attacks detected ({cw_count} occurrences). "
                "High sophistication attack. Consider ensemble defenses or detection-based approaches."
            )
        
        # Check success rate
        success_rate = self._calculate_attack_success_rate(attacks)
        if success_rate > 0.3:
            recommendations.append(
                f"High attack success rate ({success_rate:.1%}). "
                "Immediate model retraining with adversarial examples recommended."
            )
        
        # Check confidence drops
        avg_drop = sum(a["confidence_drop"] for a in attacks) / len(attacks)
        if avg_drop > 0.4:
            recommendations.append(
                f"Large confidence drops detected (average {avg_drop:.1%}). "
                "Review model calibration and consider confidence threshold adjustments."
            )
        
        return recommendations
    
    def _load_telemetry(self):
        """Load telemetry from disk"""
        telemetry_file = self.telemetry_dir / "telemetry.json"
        if telemetry_file.exists():
            try:
                with open(telemetry_file, "r") as f:
                    data = json.load(f)
                    self.inference_log = data.get("inference_log", [])
                    self.attack_patterns = data.get("attack_patterns", [])
                    self.threat_signatures = data.get("threat_signatures", {})
            except:
                pass
    
    def _save_telemetry(self):
        """Save telemetry to disk"""
        telemetry_file = self.telemetry_dir / "telemetry.json"
        data = {
            "inference_log": self.inference_log[-10000:],  # Keep last 10k
            "attack_patterns": self.attack_patterns,
            "threat_signatures": self.threat_signatures,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(telemetry_file, "w") as f:
            json.dump(data, f, indent=2)
