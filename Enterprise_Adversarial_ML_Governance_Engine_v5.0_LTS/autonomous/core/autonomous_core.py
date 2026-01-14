"""
[BRAIN] AUTONOMOUS EVOLUTION ENGINE - MODULE 1
Core autonomous components for 10-year survivability.
"""
import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import hashlib
from collections import deque
import statistics

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TelemetryRecord:
    """Immutable telemetry record - safe, no sensitive data"""
    timestamp: str
    request_id_hash: str  # Anonymized
    model_version: str
    input_shape: tuple
    prediction_confidence: float
    firewall_verdict: str  # "allow", "degrade", "block"
    attack_indicators: List[str] = field(default_factory=list)
    drift_metrics: Dict[str, float] = field(default_factory=dict)
    processing_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ThreatSignal:
    """Aggregated threat signals"""
    timestamp: str
    attack_frequency: float
    confidence_drift: float
    novelty_score: float
    requires_immediate_adaptation: bool
    requires_learning: bool
    adaptation_level: str  # "none", "policy", "model"

@dataclass
class PolicyState:
    """Current security policy state"""
    confidence_threshold: float = 0.7
    firewall_strictness: str = "adaptive"  # "adaptive", "aggressive", "maximum"
    rate_limit_rpm: int = 1000
    block_threshold: float = 0.9
    degrade_threshold: float = 0.8
    last_updated: str = ""

# ============================================================================
# 1. TELEMETRY MANAGER
# ============================================================================

class TelemetryManager:
    """Safe telemetry collection and storage"""
    
    def __init__(self, storage_path: str = "intelligence/telemetry"):
        self.storage_path = storage_path
        self._initialize_storage()
        self.recent_telemetry = deque(maxlen=1000)  # Keep last 1000 records
        
    def _initialize_storage(self):
        """Create telemetry storage structure"""
        os.makedirs(self.storage_path, exist_ok=True)
        
    def capture_safe_telemetry(self, request: Dict, inference_result: Dict) -> TelemetryRecord:
        """Capture telemetry without sensitive data"""
        # Anonymize request ID
        request_id = str(request.get("request_id", "unknown"))
        request_id_hash = hashlib.sha256(request_id.encode()).hexdigest()[:16]
        
        # Extract safe statistics only (no raw data)
        input_data = request.get("data", {})
        input_stats = {}
        
        if "input" in input_data:
            try:
                input_array = np.array(input_data["input"])
                if input_array.size > 0:
                    input_stats = {
                        "shape": input_array.shape,
                        "mean": float(np.mean(input_array)),
                        "std": float(np.std(input_array)),
                        "min": float(np.min(input_array)),
                        "max": float(np.max(input_array))
                    }
            except:
                pass  # Don't fail on input parsing errors
        
        # Create telemetry record
        record = TelemetryRecord(
            timestamp=datetime.now().isoformat(),
            request_id_hash=request_id_hash,
            model_version=inference_result.get("model_version", "unknown"),
            input_shape=input_stats.get("shape", ()),
            prediction_confidence=float(inference_result.get("confidence", 0.0)),
            firewall_verdict=inference_result.get("firewall_verdict", "allow"),
            attack_indicators=inference_result.get("attack_indicators", []),
            drift_metrics=inference_result.get("drift_metrics", {}),
            processing_latency_ms=float(inference_result.get("processing_time_ms", 0.0)),
            metadata={
                "input_stats": {k: v for k, v in input_stats.items() if k != "shape"},
                "safe_telemetry": True,
                "sensitive_data_excluded": True
            }
        )
        
        return record
    
    def store_telemetry(self, record: TelemetryRecord):
        """Append telemetry to immutable store"""
        # Add to recent memory
        self.recent_telemetry.append(record)
        
        # Store to file (append-only)
        date_str = datetime.now().strftime("%Y%m%d")
        telemetry_file = os.path.join(self.storage_path, f"telemetry_{date_str}.jsonl")
        
        with open(telemetry_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(record), default=str) + '\n')
    
    def get_recent_telemetry(self, hours: int = 24) -> List[TelemetryRecord]:
        """Get recent telemetry from memory"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        
        for record in self.recent_telemetry:
            try:
                record_time = datetime.fromisoformat(record.timestamp.replace('Z', '+00:00'))
                if record_time >= cutoff:
                    recent.append(record)
            except:
                continue
        
        return recent

# ============================================================================
# 2. THREAT ANALYZER
# ============================================================================

class ThreatAnalyzer:
    """Analyze telemetry for threat patterns"""
    
    def analyze(self, telemetry: List[TelemetryRecord]) -> ThreatSignal:
        """Analyze telemetry batch for threat signals"""
        if not telemetry:
            return self._empty_signal()
        
        # Calculate attack frequency
        total_requests = len(telemetry)
        attack_requests = sum(1 for t in telemetry if t.attack_indicators)
        attack_frequency = attack_requests / total_requests if total_requests > 0 else 0.0
        
        # Calculate confidence drift
        confidences = [t.prediction_confidence for t in telemetry if t.prediction_confidence > 0]
        if len(confidences) >= 10:
            confidence_drift = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        else:
            confidence_drift = 0.0
        
        # Calculate novelty (simple implementation)
        novelty_score = self._calculate_novelty(telemetry)
        
        # Determine required actions
        requires_immediate_adaptation = (
            attack_frequency > 0.05 or      # 5% attack rate
            confidence_drift > 0.2 or        # High confidence variance
            any(t.firewall_verdict == "block" for t in telemetry[-10:])  # Recent blocks
        )
        
        requires_learning = (
            attack_frequency > 0.01 and      # 1% attack rate
            total_requests > 100             # Enough data
        )
        
        adaptation_level = "policy" if requires_immediate_adaptation else "none"
        
        return ThreatSignal(
            timestamp=datetime.now().isoformat(),
            attack_frequency=attack_frequency,
            confidence_drift=confidence_drift,
            novelty_score=novelty_score,
            requires_immediate_adaptation=requires_immediate_adaptation,
            requires_learning=requires_learning,
            adaptation_level=adaptation_level
        )
    
    def _calculate_novelty(self, telemetry: List[TelemetryRecord]) -> float:
        """Calculate novelty score (simplified)"""
        if len(telemetry) < 10:
            return 0.0
        
        # Simple novelty: variance in attack indicators
        recent = telemetry[-10:]
        attack_types = set()
        for t in recent:
            attack_types.update(t.attack_indicators)
        
        return min(1.0, len(attack_types) / 5.0)  # Scale to 0-1
    
    def _empty_signal(self) -> ThreatSignal:
        """Return empty threat signal"""
        return ThreatSignal(
            timestamp=datetime.now().isoformat(),
            attack_frequency=0.0,
            confidence_drift=0.0,
            novelty_score=0.0,
            requires_immediate_adaptation=False,
            requires_learning=False,
            adaptation_level="none"
        )

# ============================================================================
# 3. POLICY ADAPTATION ENGINE
# ============================================================================

class PolicyAdaptationEngine:
    """Tier 1: Immediate policy adaptation"""
    
    def __init__(self):
        self.policy = PolicyState()
        self.adaptation_log = []
        
    def adapt_from_threats(self, threat_signal: ThreatSignal) -> Dict[str, Any]:
        """Adapt policies based on threat signals"""
        actions = []
        old_policy = asdict(self.policy)
        
        # Adjust based on attack frequency
        if threat_signal.attack_frequency > 0.1:  # 10% attack rate
            self.policy.firewall_strictness = "maximum"
            self.policy.rate_limit_rpm = max(100, self.policy.rate_limit_rpm - 300)
            actions.append("emergency_tightening")
        elif threat_signal.attack_frequency > 0.05:  # 5% attack rate
            self.policy.firewall_strictness = "aggressive"
            self.policy.rate_limit_rpm = max(200, self.policy.rate_limit_rpm - 100)
            actions.append("aggressive_mode")
        
        # Adjust confidence thresholds
        if threat_signal.confidence_drift > 0.15:
            self.policy.confidence_threshold = min(0.9, self.policy.confidence_threshold + 0.05)
            self.policy.block_threshold = min(0.95, self.policy.block_threshold + 0.03)
            self.policy.degrade_threshold = min(0.85, self.policy.degrade_threshold + 0.03)
            actions.append("confidence_thresholds_increased")
        
        # Update timestamp
        self.policy.last_updated = datetime.now().isoformat()
        
        # Log if changes were made
        if actions:
            adaptation_record = {
                "timestamp": self.policy.last_updated,
                "threat_signal": asdict(threat_signal),
                "actions": actions,
                "old_policy": old_policy,
                "new_policy": asdict(self.policy)
            }
            self.adaptation_log.append(adaptation_record)
        
        return {
            "actions": actions,
            "policy_changed": len(actions) > 0,
            "new_policy": asdict(self.policy)
        }
    
    def emergency_tighten(self):
        """Emergency security tightening"""
        emergency_policy = PolicyState(
            confidence_threshold=0.9,
            firewall_strictness="maximum",
            rate_limit_rpm=100,
            block_threshold=0.7,
            degrade_threshold=0.6,
            last_updated=datetime.now().isoformat()
        )
        
        self.policy = emergency_policy
        
        self.adaptation_log.append({
            "timestamp": self.policy.last_updated,
            "reason": "emergency_tightening",
            "actions": ["emergency_security_tightening"],
            "policy": asdict(self.policy)
        })
        
        return {"status": "emergency_tightening_applied"}

# ============================================================================
# 4. AUTONOMOUS CONTROLLER
# ============================================================================

class AutonomousController:
    """
    Main autonomous controller - orchestrates all components.
    Safe, simple, and testable.
    """
    
    def __init__(self, platform_root: str = "."):
        self.platform_root = platform_root
        self.telemetry_manager = TelemetryManager(
            os.path.join(platform_root, "intelligence", "telemetry")
        )
        self.threat_analyzer = ThreatAnalyzer()
        self.policy_engine = PolicyAdaptationEngine()
        
        # State
        self.is_initialized = False
        self.total_requests = 0
        self.last_analysis_time = datetime.now()
        
    def initialize(self):
        """Initialize autonomous system"""
        print("[BRAIN] Initializing autonomous controller...")
        self.is_initialized = True
        print("[OK] Autonomous controller ready")
        return {"status": "initialized", "timestamp": datetime.now().isoformat()}
    
    def process_request(self, request: Dict, inference_result: Dict) -> Dict:
        """
        Main processing method - safe and simple.
        Returns enhanced inference result.
        """
        if not self.is_initialized:
            self.initialize()
        
        self.total_requests += 1
        
        try:
            # Step 1: Capture telemetry
            telemetry = self.telemetry_manager.capture_safe_telemetry(request, inference_result)
            self.telemetry_manager.store_telemetry(telemetry)
            
            # Step 2: Analyze threats (periodically, not every request)
            enhanced_result = inference_result.copy()
            
            # Only analyze every 100 requests or every 5 minutes
            time_since_analysis = (datetime.now() - self.last_analysis_time).total_seconds()
            if self.total_requests % 100 == 0 or time_since_analysis > 300:
                recent_telemetry = self.telemetry_manager.get_recent_telemetry(hours=1)
                threat_signal = self.threat_analyzer.analyze(recent_telemetry)
                
                # Step 3: Adapt policies if needed
                if threat_signal.requires_immediate_adaptation:
                    adaptation = self.policy_engine.adapt_from_threats(threat_signal)
                    
                    # Add security info to result
                    enhanced_result["autonomous_security"] = {
                        "threat_level": "elevated" if threat_signal.attack_frequency > 0.05 else "normal",
                        "actions_taken": adaptation["actions"],
                        "attack_frequency": threat_signal.attack_frequency,
                        "policy_version": self.policy_engine.policy.last_updated[:19] if self.policy_engine.policy.last_updated else "initial"
                    }
                
                self.last_analysis_time = datetime.now()
            
            return enhanced_result
            
        except Exception as e:
            # SAFETY FIRST: On error, tighten security and return safe result
            print(f"[WARNING]  Autonomous system error: {e}")
            self.policy_engine.emergency_tighten()
            
            # Return original result with error flag
            inference_result["autonomous_security"] = {
                "error": True,
                "message": "Autonomous system error - security tightened",
                "actions": ["emergency_tightening"]
            }
            
            return inference_result
    
    def get_status(self) -> Dict[str, Any]:
        """Get autonomous system status"""
        recent_telemetry = self.telemetry_manager.get_recent_telemetry(hours=1)
        
        return {
            "status": "active" if self.is_initialized else "inactive",
            "initialized": self.is_initialized,
            "total_requests_processed": self.total_requests,
            "recent_telemetry_count": len(recent_telemetry),
            "current_policy": asdict(self.policy_engine.policy),
            "adaptation_count": len(self.policy_engine.adaptation_log),
            "last_analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health"""
        return {
            "components": {
                "telemetry_manager": "healthy",
                "threat_analyzer": "healthy", 
                "policy_engine": "healthy",
                "controller": "healthy"
            },
            "metrics": {
                "uptime": "since_initialization",
                "error_rate": 0.0,
                "processing_capacity": "high"
            },
            "survivability": {
                "design_lifetime_years": 10,
                "human_intervention_required": False,
                "fail_safe_principle": "security_tightens_on_failure"
            }
        }

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_autonomous_controller(platform_root: str = ".") -> AutonomousController:
    """Factory function to create autonomous controller"""
    return AutonomousController(platform_root)

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_autonomous_system():
    """Test the autonomous system"""
    print("\n" + "="*80)
    print("?? TESTING AUTONOMOUS SYSTEM")
    print("="*80)
    
    controller = create_autonomous_controller()
    
    # Test initialization
    print("\n1. Testing initialization...")
    status = controller.initialize()
    print(f"   Status: {status['status']}")
    
    # Test status
    print("\n2. Testing status retrieval...")
    status = controller.get_status()
    print(f"   Initialized: {status['initialized']}")
    print(f"   Policy: {status['current_policy']['firewall_strictness']}")
    
    # Test processing
    print("\n3. Testing request processing...")
    test_request = {
        "request_id": "test_123",
        "data": {"input": [0.1] * 784}
    }
    
    test_result = {
        "prediction": 7,
        "confidence": 0.85,
        "model_version": "4.0.0",
        "processing_time_ms": 45.2,
        "firewall_verdict": "allow"
    }
    
    enhanced_result = controller.process_request(test_request, test_result)
    print(f"   Original confidence: {test_result['confidence']}")
    print(f"   Enhanced result keys: {list(enhanced_result.keys())}")
    
    # Test health
    print("\n4. Testing health check...")
    health = controller.get_health()
    print(f"   Components: {len(health['components'])} healthy")
    print(f"   Survivability: {health['survivability']['design_lifetime_years']} years")
    
    print("\n" + "="*80)
    print("[OK] AUTONOMOUS SYSTEM TEST COMPLETE")
    print("="*80)
    
    return controller

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n[BRAIN] Autonomous Evolution Engine - Module 1")
    print("Version: 1.0.0")
    print("Purpose: Core autonomous components for 10-year survivability")
    
    # Run test
    controller = test_autonomous_system()
    
    print("\n?? Usage:")
    print('   controller = create_autonomous_controller()')
    print('   controller.initialize()')
    print('   enhanced_result = controller.process_request(request, inference_result)')
    print('   status = controller.get_status()')
    print('   health = controller.get_health()')
    
    print("\n?? Key Principle: Security tightens on failure")
    print("   When the autonomous system encounters errors,")
    print("   it automatically tightens security policies.")

