"""
🚀 PHASE 5.2: ECOSYSTEM AUTHORITY ENGINE
Purpose: Makes the security nervous system authoritative across all ML domains.
Scope: Vision, Tabular, Text, Time-series models.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

from autonomous.core.database_engine import DatabaseAwareEngine
from database.config import DATABASE_CONFIG

class DomainType(Enum):
    """ML Domain Types"""
    VISION = "vision"
    TABULAR = "tabular" 
    TEXT = "text"
    TIME_SERIES = "time_series"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"

class RiskTier(Enum):
    """Risk Tiers for models"""
    TIER_0 = "tier_0"  # Critical: Financial fraud, medical diagnosis
    TIER_1 = "tier_1"  # High: Authentication, security systems
    TIER_2 = "tier_2"  # Medium: Content recommendation, marketing
    TIER_3 = "tier_3"  # Low: Research, non-critical analytics
    
class ThreatSeverity(Enum):
    """Threat Severity Levels"""
    CRITICAL = "critical"    # Immediate system-wide action required
    HIGH = "high"          # Domain-wide alert and escalation
    MEDIUM = "medium"      # Model-specific action required
    LOW = "low"           # Monitor and log
    INFO = "info"         # Information only

@dataclass
class ThreatSignature:
    """Compressed threat signature for cross-domain correlation"""
    signature_hash: str
    domain: DomainType
    model_id: str
    confidence_delta: float  # Δ confidence from baseline
    feature_sensitivity: np.ndarray  # Which features most sensitive
    attack_type: str  # FGSM, PGD, DeepFool, CW, etc.
    epsilon_range: Tuple[float, float]  # Perturbation range
    timestamp: datetime
    cross_domain_correlations: List[str] = None  # Other signatures this correlates with
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            "signature_hash": self.signature_hash,
            "domain": self.domain.value,
            "model_id": self.model_id,
            "confidence_delta": float(self.confidence_delta),
            "feature_sensitivity": self.feature_sensitivity.tolist() if hasattr(self.feature_sensitivity, "tolist") else list(self.feature_sensitivity),
            "attack_type": self.attack_type,
            "epsilon_range": list(self.epsilon_range),
            "timestamp": self.timestamp.isoformat(),
            "cross_domain_correlations": self.cross_domain_correlations or []
        }

class EcosystemAuthorityEngine(DatabaseAwareEngine):

    def __init__(self):
        # MUST call super().__init__() FIRST
        super().__init__()
        
        # Now initialize Phase 5.2 specific attributes
        self.authority_level = "ecosystem"
        self.domains_governed = []
        self.cross_domain_memory = {}
        self.threat_propagation_rules = {}
        self.policy_cascade_enabled = True
        self.ecosystem_risk_score = 0.0
        
        # Initialize domain governance
        self._initialize_domain_governance()
    """
    🧠 ECOSYSTEM AUTHORITY ENGINE - PHASE 5.2
    Makes security decisions across all ML domains in the ecosystem.
    """
    
    def __init__(self):
        super().__init__()
        self.authority_level = "ecosystem"
        self.domains_governed = []
        self.cross_domain_memory = {}
        self.threat_propagation_rules = {}
        self.policy_cascade_enabled = True
        self.ecosystem_risk_score = 0.0
        
        # Initialize domain governance
        self._initialize_domain_governance()
    
    def _initialize_domain_governance(self):
        """Initialize governance for all ML domains"""
        self.domain_policies = {
            DomainType.VISION: {
                "risk_tier": RiskTier.TIER_1,
                "confidence_threshold": 0.85,
                "max_adversarial_epsilon": 0.3,
                "requires_explainability": True,
                "cross_domain_alerting": True
            },
            DomainType.TABULAR: {
                "risk_tier": RiskTier.TIER_0,
                "confidence_threshold": 0.90,
                "max_adversarial_epsilon": 0.2,
                "requires_explainability": True,
                "cross_domain_alerting": True
            },
            DomainType.TEXT: {
                "risk_tier": RiskTier.TIER_2,
                "confidence_threshold": 0.80,
                "max_adversarial_epsilon": 0.4,
                "requires_explainability": False,
                "cross_domain_alerting": True
            },
            DomainType.TIME_SERIES: {
                "risk_tier": RiskTier.TIER_1,
                "confidence_threshold": 0.88,
                "max_adversarial_epsilon": 0.25,
                "requires_explainability": True,
                "cross_domain_alerting": True
            }
        }
        
        # Track which domains are active
        self.domains_governed = list(self.domain_policies.keys())
        
        print(f"✅ Ecosystem Authority initialized: Governing {len(self.domains_governed)} domains")
    
    def register_model(self, model_id: str, domain: DomainType, 
                      risk_tier: Optional[RiskTier] = None,
                      metadata: Dict = None) -> bool:
        """
        Register a model into ecosystem governance
        
        Args:
            model_id: Unique model identifier
            domain: ML domain type
            risk_tier: Override default risk tier
            metadata: Additional model metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Get or create risk tier
            if risk_tier is None:
                risk_tier = self.domain_policies.get(domain, {}).get("risk_tier", RiskTier.TIER_2)
            
            # Create model registration
            model_data = {
                "model_id": model_id,
                "domain": domain.value,
                "risk_tier": risk_tier.value,
                "registered_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
                "threat_history": [],
                "compliance_score": 1.0  # Start fully compliant
            }
            
            # Store in database
            if hasattr(self, "database_session") and self.database_session:
                from database.models.model_registry import ModelRegistry
                
                # Check if already exists
                existing = self.database_session.query(ModelRegistry).filter(
                    ModelRegistry.model_id == model_id
                ).first()
                
                if not existing:
                    model = ModelRegistry(
                        model_id=model_id,
                        model_type=domain.value,
                        risk_tier=risk_tier.value,
                        deployment_phase="production" if risk_tier in [RiskTier.TIER_0, RiskTier.TIER_1] else "development",
                        confidence_threshold=self.domain_policies[domain]["confidence_threshold"],
                        parameters_count=metadata.get("parameters", 0) if metadata else 0,
                        last_updated=datetime.utcnow()
                    )
                    self.database_session.add(model)
                    self.database_session.commit()
                    print(f"✅ Registered model {model_id} in {domain.value} domain (Tier: {risk_tier.value})")
                else:
                    print(f"⚠️  Model {model_id} already registered")
            
            # Also store in memory
            if model_id not in self.cross_domain_memory:
                self.cross_domain_memory[model_id] = model_data
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to register model {model_id}: {e}")
            return False
    
    def analyze_threat_cross_domain(self, threat_signature: ThreatSignature) -> Dict:
        """
        Analyze threat across all domains for correlation
        
        Args:
            threat_signature: Threat signature from one domain
            
        Returns:
            Dict: Cross-domain analysis results
        """
        analysis = {
            "original_signature": threat_signature.signature_hash,
            "domain": threat_signature.domain.value,
            "model_id": threat_signature.model_id,
            "cross_domain_correlations": [],
            "propagation_recommendations": [],
            "ecosystem_risk_impact": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check for similar threats in other domains
        for model_id, model_data in self.cross_domain_memory.items():
            if model_id == threat_signature.model_id:
                continue  # Skip same model
                
            model_domain = DomainType(model_data["domain"])
            
            # Check if threat patterns correlate
            correlation_score = self._calculate_threat_correlation(
                threat_signature, 
                model_data.get("threat_history", [])
            )
            
            if correlation_score > 0.6:  # Strong correlation threshold
                correlation_entry = {
                    "correlated_model": model_id,
                    "correlated_domain": model_domain.value,
                    "correlation_score": correlation_score,
                    "risk_tier": model_data.get("risk_tier", "tier_2")
                }
                
                analysis["cross_domain_correlations"].append(correlation_entry)
                
                # Generate propagation recommendation
                recommendation = self._generate_propagation_recommendation(
                    threat_signature, 
                    model_domain,
                    correlation_score
                )
                
                if recommendation:
                    analysis["propagation_recommendations"].append(recommendation)
        
        # Calculate ecosystem risk impact
        if analysis["cross_domain_correlations"]:
            # Higher impact if correlated with high-risk models
            risk_scores = []
            for corr in analysis["cross_domain_correlations"]:
                risk_tier = corr["risk_tier"]
                tier_multiplier = {
                    "tier_0": 2.0,
                    "tier_1": 1.5,
                    "tier_2": 1.0,
                    "tier_3": 0.5
                }.get(risk_tier, 1.0)
                
                risk_scores.append(corr["correlation_score"] * tier_multiplier)
            
            analysis["ecosystem_risk_impact"] = max(risk_scores) if risk_scores else 0.0
            
            # Update ecosystem risk score
            self.ecosystem_risk_score = max(self.ecosystem_risk_score, analysis["ecosystem_risk_impact"])
        
        # Store analysis in database
        if hasattr(self, "database_session") and self.database_session and analysis["cross_domain_correlations"]:
            try:
                from database.models.security_memory import SecurityMemory
                
                memory = SecurityMemory(
                    threat_pattern_hash=threat_signature.signature_hash,
                    model_id=threat_signature.model_id,
                    threat_type=threat_signature.attack_type,
                    confidence_delta=threat_signature.confidence_delta,
                    epsilon_range_min=threat_signature.epsilon_range[0],
                    epsilon_range_max=threat_signature.epsilon_range[1],
                    cross_model_correlation=json.dumps(analysis["cross_domain_correlations"]),
                    timestamp=datetime.utcnow()
                )
                self.database_session.add(memory)
                self.database_session.commit()
            except Exception as e:
                print(f"⚠️  Failed to store cross-domain analysis: {e}")
        
        return analysis
    
    def _calculate_threat_correlation(self, new_threat: ThreatSignature, 
                                    threat_history: List[Dict]) -> float:
        """
        Calculate correlation between new threat and historical threats
        
        Args:
            new_threat: New threat signature
            threat_history: List of historical threats
            
        Returns:
            float: Correlation score 0-1
        """
        if not threat_history:
            return 0.0
        
        best_correlation = 0.0
        
        for historical in threat_history:
            # Compare attack types
            if historical.get("attack_type") != new_threat.attack_type:
                continue
            
            # Compare epsilon ranges (similar perturbation magnitude)
            hist_epsilon = historical.get("epsilon_range", [0, 0])
            new_epsilon = new_threat.epsilon_range
            
            epsilon_overlap = self._calculate_range_overlap(hist_epsilon, new_epsilon)
            
            # Compare confidence deltas (similar impact)
            hist_delta = abs(historical.get("confidence_delta", 0))
            new_delta = abs(new_threat.confidence_delta)
            delta_similarity = 1.0 - min(abs(hist_delta - new_delta), 1.0)
            
            # Combined correlation score
            correlation = (epsilon_overlap * 0.6) + (delta_similarity * 0.4)
            best_correlation = max(best_correlation, correlation)
        
        return best_correlation
    
    def _calculate_range_overlap(self, range1: List[float], range2: Tuple[float, float]) -> float:
        """Calculate overlap between two ranges"""
        if not range1 or not range2:
            return 0.0
        
        start1, end1 = range1[0], range1[1]
        start2, end2 = range2[0], range2[1]
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start > overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        range1_length = end1 - start1
        range2_length = end2 - start2
        
        # Normalized overlap
        return overlap_length / max(range1_length, range2_length)
    
    def _generate_propagation_recommendation(self, threat: ThreatSignature,
                                           target_domain: DomainType,
                                           correlation_score: float) -> Optional[Dict]:
        """
        Generate propagation recommendation to other domains
        
        Args:
            threat: Threat signature
            target_domain: Domain to propagate to
            correlation_score: Correlation strength
            
        Returns:
            Optional[Dict]: Propagation recommendation
        """
        if correlation_score < 0.7:
            return None
        
        # Get policy for target domain
        target_policy = self.domain_policies.get(target_domain, {})
        
        recommendation = {
            "action": "propagate_threat_alert",
            "source_domain": threat.domain.value,
            "target_domain": target_domain.value,
            "threat_type": threat.attack_type,
            "correlation_score": correlation_score,
            "recommended_actions": [],
            "urgency": "high" if correlation_score > 0.8 else "medium"
        }
        
        # Generate specific actions based on threat type
        if threat.attack_type in ["FGSM", "PGD"]:
            recommendation["recommended_actions"].extend([
                f"Increase {target_domain.value} confidence threshold by {correlation_score * 10:.1f}%",
                f"Activate adversarial training for {target_domain.value} models",
                f"Enable {target_domain.value} model monitoring for epsilon {threat.epsilon_range[1]:.2f} attacks"
            ])
        elif threat.attack_type == "DeepFool":
            recommendation["recommended_actions"].extend([
                f"Review {target_domain.value} model decision boundaries",
                f"Add robustness regularization to {target_domain.value} training",
                f"Test {target_domain.value} models with decision boundary attacks"
            ])
        
        return recommendation
    
    def propagate_intelligence(self, source_domain: DomainType, 
                             intelligence: Dict, 
                             target_domains: List[DomainType] = None) -> Dict:
        """
        Propagate intelligence from one domain to others
        
        Args:
            source_domain: Source domain
            intelligence: Intelligence data
            target_domains: Specific domains to propagate to (None = all)
            
        Returns:
            Dict: Propagation results
        """
        if target_domains is None:
            target_domains = self.domains_governed
        
        results = {
            "source_domain": source_domain.value,
            "propagation_time": datetime.utcnow().isoformat(),
            "target_domains": [],
            "success_count": 0,
            "fail_count": 0
        }
        
        for target_domain in target_domains:
            if target_domain == source_domain:
                continue
            
            try:
                # Apply domain-specific propagation rules
                propagation_success = self._apply_propagation_rules(
                    source_domain, target_domain, intelligence
                )
                
                if propagation_success:
                    results["target_domains"].append({
                        "domain": target_domain.value,
                        "status": "success",
                        "applied_rules": len(self.threat_propagation_rules.get(f"{source_domain.value}_{target_domain.value}", []))
                    })
                    results["success_count"] += 1
                else:
                    results["target_domains"].append({
                        "domain": target_domain.value,
                        "status": "failed",
                        "reason": "No applicable propagation rules"
                    })
                    results["fail_count"] += 1
                    
            except Exception as e:
                results["target_domains"].append({
                    "domain": target_domain.value,
                    "status": "error",
                    "reason": str(e)
                })
                results["fail_count"] += 1
        
        # Store propagation results
        if hasattr(self, "database_session") and self.database_session and results["success_count"] > 0:
            try:
                from database.models.autonomous_decisions import AutonomousDecision
                
                decision = AutonomousDecision(
                    trigger_type="ecosystem_signal",
                    system_state=self.system_state,
                    security_posture=self.security_posture,
                    decision_type="propagate_alert",
                    decision_scope="ecosystem",
                    affected_domains=[d.value for d in target_domains],
                    decision_rationale={
                        "intelligence_type": intelligence.get("type", "unknown"),
                        "propagation_results": results,
                        "ecosystem_risk_score": self.ecosystem_risk_score
                    },
                    confidence_in_decision=min(results["success_count"] / len(target_domains), 1.0)
                )
                self.database_session.add(decision)
                self.database_session.commit()
            except Exception as e:
                print(f"⚠️  Failed to log propagation decision: {e}")
        
        return results
    
    def _apply_propagation_rules(self, source_domain: DomainType,
                               target_domain: DomainType,
                               intelligence: Dict) -> bool:
        """
        Apply domain-specific propagation rules
        
        Args:
            source_domain: Source domain
            target_domain: Target domain
            intelligence: Intelligence to propagate
            
        Returns:
            bool: Success status
        """
        rule_key = f"{source_domain.value}_{target_domain.value}"
        
        if rule_key not in self.threat_propagation_rules:
            # Create default propagation rules
            self.threat_propagation_rules[rule_key] = self._create_propagation_rules(
                source_domain, target_domain
            )
        
        rules = self.threat_propagation_rules[rule_key]
        
        # Apply rules
        applied_count = 0
        for rule in rules:
            if self._evaluate_rule(rule, intelligence):
                applied_count += 1
                # Execute rule action
                self._execute_rule_action(rule, target_domain, intelligence)
        
        return applied_count > 0
    
    def _create_propagation_rules(self, source: DomainType, target: DomainType) -> List[Dict]:
        """Create propagation rules between domains"""
        rules = []
        
        # Generic cross-domain rules
        rules.append({
            "name": f"{source.value}_to_{target.value}_confidence_anomaly",
            "condition": "intelligence.get('type') == 'confidence_anomaly' and intelligence.get('severity') in ['high', 'critical']",
            "action": "adjust_confidence_threshold",
            "action_params": {"adjustment_percent": 10.0},
            "priority": "high"
        })
        
        rules.append({
            "name": f"{source.value}_to_{target.value}_adversarial_pattern",
            "condition": "intelligence.get('type') == 'adversarial_pattern' and intelligence.get('attack_type') in ['FGSM', 'PGD', 'DeepFool']",
            "action": "enable_adversarial_monitoring",
            "action_params": {"attack_types": ["FGSM", "PGD", "DeepFool"]},
            "priority": "medium"
        })
        
        # Domain-specific rules
        if source == DomainType.VISION and target == DomainType.TABULAR:
            rules.append({
                "name": "vision_to_tabular_feature_attack",
                "condition": "intelligence.get('attack_type') == 'feature_perturbation'",
                "action": "enable_feature_sensitivity_analysis",
                "action_params": {"analysis_depth": "deep"},
                "priority": "high"
            })
        
        return rules
    
    def _evaluate_rule(self, rule: Dict, intelligence: Dict) -> bool:
        """Evaluate if a rule condition is met"""
        try:
            # Simple condition evaluation (in production, use a proper rule engine)
            condition = rule.get("condition", "")
            
            # Very basic evaluation - in production, use a proper expression evaluator
            if "confidence_anomaly" in condition and intelligence.get("type") == "confidence_anomaly":
                return True
            elif "adversarial_pattern" in condition and intelligence.get("type") == "adversarial_pattern":
                return True
            elif "feature_attack" in condition and intelligence.get("attack_type") == "feature_perturbation":
                return True
            
            return False
        except:
            return False
    
    def _execute_rule_action(self, rule: Dict, target_domain: DomainType, intelligence: Dict):
        """Execute rule action"""
        action = rule.get("action", "")
        
        if action == "adjust_confidence_threshold":
            adjustment = rule.get("action_params", {}).get("adjustment_percent", 5.0)
            print(f"   ⚡ Adjusting {target_domain.value} confidence threshold by {adjustment}%")
            
        elif action == "enable_adversarial_monitoring":
            attack_types = rule.get("action_params", {}).get("attack_types", [])
            print(f"   ⚡ Enabling adversarial monitoring for {target_domain.value}: {attack_types}")
            
        elif action == "enable_feature_sensitivity_analysis":
            analysis_depth = rule.get("action_params", {}).get("analysis_depth", "standard")
            print(f"   ⚡ Enabling {analysis_depth} feature sensitivity analysis for {target_domain.value}")
    
    def get_ecosystem_health(self) -> Dict:
        """
        Get comprehensive ecosystem health report
        
        Returns:
            Dict: Ecosystem health data
        """
        health = super().get_ecosystem_health()
        
        # Add Phase 5.2 specific metrics
        health.update({
            "phase": "5.2_ecosystem_authority",
            "authority_level": self.authority_level,
            "domains_governed": [d.value for d in self.domains_governed],
            "cross_domain_memory_size": len(self.cross_domain_memory),
            "threat_propagation_rules_count": sum(len(rules) for rules in self.threat_propagation_rules.values()),
            "ecosystem_risk_score": self.ecosystem_risk_score,
            "policy_cascade_enabled": self.policy_cascade_enabled,
            "domain_policies": {
                domain.value: policy 
                for domain, policy in self.domain_policies.items()
            }
        })
        
        return health
    
    def make_ecosystem_decision(self, trigger: str, context: Dict) -> Dict:
        """
        Make ecosystem-wide autonomous decision
        
        Args:
            trigger: Decision trigger
            context: Decision context
            
        Returns:
            Dict: Decision with rationale
        """
        decision = {
            "decision_id": hashlib.sha256(f"{trigger}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16],
            "timestamp": datetime.utcnow().isoformat(),
            "trigger": trigger,
            "authority_level": self.authority_level,
            "affected_domains": [],
            "actions": [],
            "rationale": {},
            "confidence": 0.0
        }
        
        # Analyze context
        affected_domains = self._analyze_context_for_domains(context)
        decision["affected_domains"] = [d.value for d in affected_domains]
        
        # Generate actions based on trigger and domains
        if trigger == "cross_domain_threat_correlation":
            actions = self._generate_cross_domain_threat_actions(context, affected_domains)
            decision["actions"] = actions
            decision["confidence"] = min(context.get("correlation_score", 0.0), 0.9)
            
        elif trigger == "ecosystem_risk_elevation":
            actions = self._generate_risk_mitigation_actions(context, affected_domains)
            decision["actions"] = actions
            decision["confidence"] = 0.85
            
        elif trigger == "policy_cascade_required":
            actions = self._generate_policy_cascade_actions(context, affected_domains)
            decision["actions"] = actions
            decision["confidence"] = 0.95
        
        # Store decision in database
        if hasattr(self, "database_session") and self.database_session:
            try:
                from database.models.autonomous_decisions import AutonomousDecision
                
                db_decision = AutonomousDecision(
                    trigger_type=trigger,
                    system_state=self.system_state,
                    security_posture=self.security_posture,
                    policy_version=1,
                    decision_type="ecosystem_action",
                    decision_scope="ecosystem",
                    affected_domains=decision["affected_domains"],
                    decision_rationale=decision,
                    confidence_in_decision=decision["confidence"]
                )
                self.database_session.add(db_decision)
                self.database_session.commit()
                
                decision["database_id"] = str(db_decision.decision_id)
            except Exception as e:
                print(f"⚠️  Failed to store ecosystem decision: {e}")
        
        return decision
    
    def _analyze_context_for_domains(self, context: Dict) -> List[DomainType]:
        """Analyze context to determine affected domains"""
        domains = set()
        
        # Check for explicit domain mentions
        if "domain" in context:
            try:
                domains.add(DomainType(context["domain"]))
            except:
                pass
        
        # Check for model references
        if "model_id" in context:
            model_id = context["model_id"]
            for domain in self.domains_governed:
                # Simple heuristic: check if model_id contains domain hint
                domain_hints = {
                    DomainType.VISION: ["vision", "image", "cnn", "resnet", "vgg"],
                    DomainType.TABULAR: ["tabular", "xgb", "lgbm", "randomforest", "logistic"],
                    DomainType.TEXT: ["text", "bert", "gpt", "transformer", "nlp"],
                    DomainType.TIME_SERIES: ["time", "series", "lstm", "arima", "prophet"]
                }
                
                for hint in domain_hints.get(domain, []):
                    if hint.lower() in model_id.lower():
                        domains.add(domain)
                        break
        
        # Default to all domains if none identified
        if not domains:
            domains = set(self.domains_governed)
        
        return list(domains)
    
    def _generate_cross_domain_threat_actions(self, context: Dict, domains: List[DomainType]) -> List[Dict]:
        """Generate actions for cross-domain threat correlation"""
        actions = []
        
        correlation_score = context.get("correlation_score", 0.0)
        threat_type = context.get("threat_type", "unknown")
        
        for domain in domains:
            domain_policy = self.domain_policies.get(domain, {})
            
            if correlation_score > 0.8:
                # High correlation - aggressive actions
                actions.append({
                    "domain": domain.value,
                    "action": "increase_confidence_threshold",
                    "parameters": {"increase_percent": 15.0},
                    "rationale": f"High cross-domain threat correlation ({correlation_score:.2f}) with {threat_type}"
                })
                
                actions.append({
                    "domain": domain.value,
                    "action": "enable_enhanced_monitoring",
                    "parameters": {"duration_hours": 24, "sampling_rate": 1.0},
                    "rationale": "Enhanced monitoring due to cross-domain threat"
                })
                
            elif correlation_score > 0.6:
                # Medium correlation - moderate actions
                actions.append({
                    "domain": domain.value,
                    "action": "increase_confidence_threshold",
                    "parameters": {"increase_percent": 8.0},
                    "rationale": f"Medium cross-domain threat correlation ({correlation_score:.2f})"
                })
                
                if domain_policy.get("requires_explainability", False):
                    actions.append({
                        "domain": domain.value,
                        "action": "require_explainability_review",
                        "parameters": {"review_depth": "targeted"},
                        "rationale": "Explainability review for threat correlation"
                    })
        
        return actions
    
    def _generate_risk_mitigation_actions(self, context: Dict, domains: List[DomainType]) -> List[Dict]:
        """Generate risk mitigation actions"""
        actions = []
        
        risk_level = context.get("risk_level", "medium")
        
        for domain in domains:
            if risk_level in ["high", "critical"]:
                actions.append({
                    "domain": domain.value,
                    "action": "activate_defensive_measures",
                    "parameters": {"level": "maximum"},
                    "rationale": f"Ecosystem risk level: {risk_level}"
                })
                
                if self.domain_policies.get(domain, {}).get("cross_domain_alerting", False):
                    actions.append({
                        "domain": domain.value,
                        "action": "broadcast_ecosystem_alert",
                        "parameters": {"alert_level": risk_level},
                        "rationale": "Cross-domain alert broadcast"
                    })
        
        return actions
    
    def _generate_policy_cascade_actions(self, context: Dict, domains: List[DomainType]) -> List[Dict]:
        """Generate policy cascade actions"""
        actions = []
        
        policy_type = context.get("policy_type", "confidence_threshold")
        new_value = context.get("new_value")
        
        for domain in domains:
            actions.append({
                "domain": domain.value,
                "action": "apply_policy_cascade",
                "parameters": {
                    "policy_type": policy_type,
                    "new_value": new_value,
                    "cascade_source": context.get("source_domain", "system")
                },
                "rationale": f"Policy cascade: {policy_type} = {new_value}"
            })
        
        return actions

# Factory function for ecosystem authority engine
def create_ecosystem_authority_engine():
    """Create and initialize ecosystem authority engine"""
    engine = EcosystemAuthorityEngine()
    
    # Register some example models (in production, these would come from actual model registry)
    example_models = [
        {"id": "mnist_cnn_fixed", "domain": DomainType.VISION, "risk_tier": RiskTier.TIER_2},
        {"id": "credit_fraud_xgboost", "domain": DomainType.TABULAR, "risk_tier": RiskTier.TIER_0},
        {"id": "sentiment_bert", "domain": DomainType.TEXT, "risk_tier": RiskTier.TIER_2},
        {"id": "stock_lstm", "domain": DomainType.TIME_SERIES, "risk_tier": RiskTier.TIER_1},
    ]
    
    for model in example_models:
        engine.register_model(
            model_id=model["id"],
            domain=model["domain"],
            risk_tier=model["risk_tier"],
            metadata={"parameters": 1000000, "framework": "pytorch"}
        )
    
    return engine



