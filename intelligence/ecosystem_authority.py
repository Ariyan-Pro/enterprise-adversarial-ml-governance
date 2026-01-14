from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json

# ============================================================================
# ECOSYSTEM DATA STRUCTURES
# ============================================================================

class ModelDomain(Enum):
    VISION = "vision"
    TABULAR = "tabular"
    TEXT = "text"
    TIME_SERIES = "time_series"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"

class RiskProfile(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXPERIMENTAL = "experimental"

class SecurityState(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"

@dataclass
class ModelRegistryEntry:
    model_id: str
    domain: ModelDomain
    risk_profile: RiskProfile
    version: str
    deployment_time: str
    owner: str
    confidence_baseline: float
    telemetry_enabled: bool = True
    governance_applied: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityMemoryRecord:
    memory_id: str
    pattern_type: str
    domain: ModelDomain
    first_seen: str
    last_seen: str
    recurrence_count: int
    compressed_signature: str
    severity_score: float
    mitigation_suggested: str
    learned_from: List[str] = field(default_factory=list)

# ============================================================================
# ECOSYSTEM GOVERNANCE ENGINE
# ============================================================================

class EcosystemGovernance:
    def __init__(self):
        self.model_registry: Dict[str, ModelRegistryEntry] = {}
        self.security_memory: Dict[str, SecurityMemoryRecord] = {}
        self.cross_model_signals: Dict[str, List[Dict]] = {}
        self.security_state = SecurityState.NORMAL
        
        self.ecosystem_health = {
            "total_models": 0,
            "protected_models": 0,
            "active_policies": 2,
            "memory_patterns": 0,
            "cross_model_alerts": 0,
            "last_audit": datetime.now().isoformat()
        }
        
        self._register_existing_model()
        
        print("\n" + "="*80)
        print("🏛️  ECOSYSTEM GOVERNANCE INITIALIZED")
        print("="*80)
    
    def _register_existing_model(self):
        mnist_model = ModelRegistryEntry(
            model_id="mnist_cnn_v1",
            domain=ModelDomain.VISION,
            risk_profile=RiskProfile.MEDIUM,
            version="1.0.0",
            deployment_time=datetime.now().isoformat(),
            owner="adversarial-ml-suite",
            confidence_baseline=0.85,
            telemetry_enabled=True,
            governance_applied=True,
            metadata={
                "parameters": 207018,
                "accuracy": 0.99,
                "robustness_score": 0.88,
                "architecture": "CNN",
                "phase": 4
            }
        )
        self.model_registry[mnist_model.model_id] = mnist_model
        self.ecosystem_health["total_models"] += 1
        self.ecosystem_health["protected_models"] += 1
        print(f"✅ Registered: {mnist_model.model_id}")
    
    def register_model(self, model_entry: ModelRegistryEntry) -> Dict[str, Any]:
        if model_entry.model_id in self.model_registry:
            return {"status": "already_registered", "model_id": model_entry.model_id}
        
        self.model_registry[model_entry.model_id] = model_entry
        self.ecosystem_health["total_models"] += 1
        
        if model_entry.governance_applied:
            self.ecosystem_health["protected_models"] += 1
        
        return {
            "status": "registered",
            "model_id": model_entry.model_id,
            "domain": model_entry.domain.value,
            "risk_profile": model_entry.risk_profile.value,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_cross_model_signal(self, source_model: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        if source_model not in self.model_registry:
            return {"status": "rejected", "reason": "Model not registered"}
        
        signal_str = json.dumps(signal, sort_keys=True)
        signal_id = f"SIG_{hashlib.sha256(signal_str.encode()).hexdigest()[:12]}"
        
        signal_with_metadata = {
            **signal,
            "signal_id": signal_id,
            "source_model": source_model,
            "timestamp": datetime.now().isoformat(),
            "processed": True
        }
        
        if source_model not in self.cross_model_signals:
            self.cross_model_signals[source_model] = []
        
        self.cross_model_signals[source_model].append(signal_with_metadata)
        self.ecosystem_health["cross_model_alerts"] += 1
        
        # Update security state based on threat level
        threat_level = signal.get("threat_level", "low")
        if threat_level == "critical":
            self.security_state = SecurityState.EMERGENCY
        elif threat_level == "high":
            self.security_state = SecurityState.ELEVATED
        
        return {
            "status": "processed",
            "signal_id": signal_id,
            "security_state": self.security_state.value,
            "ecosystem_alerts": self.ecosystem_health["cross_model_alerts"]
        }
    
    def get_model_recommendations(self, model_id: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        if model_id not in self.model_registry:
            return {"status": "model_not_registered"}
        
        model = self.model_registry[model_id]
        recommendations = []
        
        # Basic recommendations based on security state
        if self.security_state == SecurityState.EMERGENCY:
            recommendations.append({
                "type": "security_emergency",
                "action": "increase_confidence_threshold",
                "value": "+0.15",
                "reason": "Ecosystem in emergency state"
            })
        
        # Context-based recommendations
        if current_context.get("confidence", 1.0) < 0.7:
            recommendations.append({
                "type": "confidence_low",
                "action": "review_model_inputs",
                "value": "immediate",
                "reason": f"Confidence below threshold: {current_context.get('confidence')}"
            })
        
        return {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "domain": model.domain.value,
            "risk_profile": model.risk_profile.value,
            "security_state": self.security_state.value,
            "recommendations": recommendations,
            "ecosystem_context": {
                "total_models": self.ecosystem_health["total_models"],
                "protected_models": self.ecosystem_health["protected_models"],
                "active_alerts": self.ecosystem_health["cross_model_alerts"]
            }
        }
    
    def get_ecosystem_status(self) -> Dict[str, Any]:
        domain_dist = {}
        risk_dist = {}
        
        for model in self.model_registry.values():
            domain = model.domain.value
            risk = model.risk_profile.value
            domain_dist[domain] = domain_dist.get(domain, 0) + 1
            risk_dist[risk] = risk_dist.get(risk, 0) + 1
        
        protection_coverage = 0
        if self.ecosystem_health["total_models"] > 0:
            protection_coverage = (self.ecosystem_health["protected_models"] / self.ecosystem_health["total_models"]) * 100
        
        return {
            "timestamp": datetime.now().isoformat(),
            "security_state": self.security_state.value,
            "ecosystem_health": self.ecosystem_health.copy(),
            "domain_distribution": domain_dist,
            "risk_distribution": risk_dist,
            "protection_coverage": round(protection_coverage, 2),
            "model_count": self.ecosystem_health["total_models"]
        }
    
    def add_test_model(self, domain: ModelDomain, risk: RiskProfile) -> Dict[str, Any]:
        model_id = f"test_{domain.value}_{datetime.now().strftime('%H%M%S')}"
        model = ModelRegistryEntry(
            model_id=model_id,
            domain=domain,
            risk_profile=risk,
            version="1.0.0",
            deployment_time=datetime.now().isoformat(),
            owner="test_suite",
            confidence_baseline=0.8,
            telemetry_enabled=True,
            governance_applied=True
        )
        return self.register_model(model)

# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 ECOSYSTEM AUTHORITY - DIRECT TEST")
    print("="*80)
    
    ecosystem = EcosystemGovernance()
    
    # Test 1: Ecosystem Status
    status = ecosystem.get_ecosystem_status()
    print(f"\n📊 STATUS: {status['security_state']}")
    print(f"   Models: {status['model_count']}")
    print(f"   Protection: {status['protection_coverage']}%")
    
    # Test 2: Add test models
    print("\n📝 ADDING TEST MODELS:")
    result1 = ecosystem.add_test_model(ModelDomain.TABULAR, RiskProfile.HIGH)
    result2 = ecosystem.add_test_model(ModelDomain.TEXT, RiskProfile.MEDIUM)
    print(f"   ✅ {result1['model_id']} - {result1['risk_profile']}")
    print(f"   ✅ {result2['model_id']} - {result2['risk_profile']}")
    
    # Test 3: Cross-model signal
    print("\n📡 TESTING CROSS-MODEL SIGNAL:")
    test_signal = {
        "threat_level": "high",
        "attack_type": "adversarial",
        "confidence_drop": 0.4,
        "source": "detection_engine"
    }
    signal_result = ecosystem.process_cross_model_signal("mnist_cnn_v1", test_signal)
    print(f"   Signal ID: {signal_result['signal_id']}")
    print(f"   New State: {signal_result['security_state']}")
    
    # Test 4: Model recommendations
    print("\n🎯 GETTING RECOMMENDATIONS:")
    context = {"confidence": 0.65, "request_rate": 25}
    recs = ecosystem.get_model_recommendations("mnist_cnn_v1", context)
    print(f"   Model: {recs['model_id']}")
    print(f"   Recommendations: {len(recs['recommendations'])}")
    for rec in recs['recommendations']:
        print(f"   • {rec['action']}: {rec['reason']}")
    
    # Final status
    final_status = ecosystem.get_ecosystem_status()
    print("\n" + "="*80)
    print("🏁 FINAL ECOSYSTEM STATE")
    print("="*80)
    print(f"Total Models: {final_status['model_count']}")
    print(f"Protected: {final_status['protection_coverage']}%")
    print(f"Alerts: {final_status['ecosystem_health']['cross_model_alerts']}")
    print(f"State: {final_status['security_state']}")
    
    print("\n✅ ECOSYSTEM AUTHORITY OPERATIONAL - PHASE 5 READY")
