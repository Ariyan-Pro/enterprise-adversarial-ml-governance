"""
🧠 ECOSYSTEM AUTHORITY ENGINE - Phase 5.2
Purpose: Authoritative control across multiple ML domains with threat correlation.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

from autonomous.core.database_engine import create_phase5_engine, DatabaseAwareEngine

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CrossDomainThreat:
    """Threat pattern that spans multiple domains"""
    threat_id: str
    pattern_signature: str
    affected_domains: List[str]
    domain_severity_scores: Dict[str, float]  # severity per domain
    first_seen: datetime
    last_seen: datetime
    recurrence_count: int
    correlation_score: float  # How strongly domains are correlated
    propagation_path: List[str]  # How threat moved between domains
    
    def is_multi_domain(self) -> bool:
        """Check if threat affects multiple domains"""
        return len(self.affected_domains) > 1
    
    def get_overall_severity(self) -> float:
        """Calculate overall severity across domains"""
        if not self.domain_severity_scores:
            return 0.0
        
        # Weight by domain criticality
        domain_weights = {
            "vision": 1.0,
            "tabular": 1.2,    # Higher weight for financial/risk domains
            "text": 0.9,
            "time_series": 1.1,
            "hybrid": 1.3
        }
        
        weighted_scores = []
        for domain, score in self.domain_severity_scores.items():
            weight = domain_weights.get(domain, 1.0)
            weighted_scores.append(score * weight)
        
        return max(weighted_scores)  # Use max severity across domains

@dataclass
class EcosystemPolicy:
    """Policy that applies across multiple domains"""
    policy_id: str
    policy_type: str  # "cross_domain_alert", "propagation_block", "confidence_synchronization"
    affected_domains: List[str]
    trigger_conditions: Dict[str, Any]
    actions: List[str]
    effectiveness_score: float = 0.0
    last_applied: Optional[datetime] = None
    application_count: int = 0

@dataclass
class DomainIntelligence:
    """Intelligence profile for a specific domain"""
    domain: str
    threat_frequency: float  # threats per day
    avg_severity: float
    model_count: int
    risk_distribution: Dict[str, int]  # count by risk tier
    last_major_incident: Optional[datetime] = None
    intelligence_maturity: float = 0.0  # 0-1 scale

# ============================================================================
# ECOSYSTEM AUTHORITY ENGINE
# ============================================================================

class EcosystemAuthorityEngine(DatabaseAwareEngine):
    """
    Phase 5.2: Ecosystem authority with cross-domain threat correlation
    and unified policy enforcement.
    """
    
    def __init__(self):
        super().__init__()
        self.cross_domain_threats: Dict[str, CrossDomainThreat] = {}
        self.ecosystem_policies: Dict[str, EcosystemPolicy] = {}
        self.domain_intelligence: Dict[str, DomainIntelligence] = {}
        self._initialize_ecosystem()
    
    def _initialize_ecosystem(self):
        """Initialize ecosystem with domain intelligence"""
        # Initialize domain intelligence from database
        try:
            domains = ["vision", "tabular", "text", "time_series", "hybrid"]
            
            for domain in domains:
                models = self.get_models_by_domain(domain)
                
                if models:
                    # Calculate domain intelligence
                    threat_count = self._get_threat_count_for_domain(domain)
                    severity_scores = [m.get("robustness_baseline", 0.0) for m in models]
                    avg_severity = 1.0 - (sum(severity_scores) / len(severity_scores)) if severity_scores else 0.5
                    
                    # Count by risk tier
                    risk_distribution = defaultdict(int)
                    for model in models:
                        risk_tier = model.get("risk_tier", "unknown")
                        risk_distribution[risk_tier] += 1
                    
                    self.domain_intelligence[domain] = DomainIntelligence(
                        domain=domain,
                        threat_frequency=threat_count / 30 if threat_count > 0 else 0.0,  # per day estimate
                        avg_severity=avg_severity,
                        model_count=len(models),
                        risk_distribution=dict(risk_distribution),
                        intelligence_maturity=min(len(models) * 0.1, 1.0)  # Maturity based on model count
                    )
        
        except Exception as e:
            print(f"⚠️  Failed to initialize ecosystem intelligence: {e}")
    
    def _get_threat_count_for_domain(self, domain: str, days: int = 30) -> int:
        """Get threat count for a domain (simplified - would query database)"""
        # This would query SecurityMemory table for domain-specific threats
        return 0  # Placeholder
    
    # ============================================================================
    # CROSS-DOMAIN THREAT CORRELATION
    # ============================================================================
    
    def detect_cross_domain_threats(self, time_window_hours: int = 24) -> List[CrossDomainThreat]:
        """
        Detect threats that appear across multiple domains.
        """
        try:
            # Get recent threats from all domains
            recent_threats = self._get_recent_threats(time_window_hours)
            
            # Group by threat signature pattern
            threat_groups = defaultdict(list)
            for threat in recent_threats:
                signature = threat.get("pattern_signature", "")
                if signature:
                    threat_groups[signature].append(threat)
            
            # Identify cross-domain patterns
            cross_domain_threats = []
            
            for signature, threats in threat_groups.items():
                if len(threats) < 2:
                    continue  # Need at least 2 threats for correlation
                
                # Get unique domains
                domains = set()
                domain_severity = defaultdict(list)
                timestamps = []
                
                for threat in threats:
                    domain = threat.get("source_domain", "unknown")
                    domains.add(domain)
                    domain_severity[domain].append(threat.get("severity_score", 0.0))
                    timestamps.append(datetime.fromisoformat(threat.get("first_observed", datetime.now().isoformat())))
                
                if len(domains) > 1:
                    # Calculate domain severity averages
                    severity_scores = {}
                    for domain, scores in domain_severity.items():
                        severity_scores[domain] = statistics.mean(scores) if scores else 0.0
                    
                    # Calculate correlation score based on timing
                    correlation_score = self._calculate_temporal_correlation(timestamps)
                    
                    # Determine propagation path
                    propagation_path = self._determine_propagation_path(threats)
                    
                    cross_threat = CrossDomainThreat(
                        threat_id=f"cdt_{hashlib.md5(signature.encode()).hexdigest()[:16]}",
                        pattern_signature=signature,
                        affected_domains=list(domains),
                        domain_severity_scores=severity_scores,
                        first_seen=min(timestamps) if timestamps else datetime.now(),
                        last_seen=max(timestamps) if timestamps else datetime.now(),
                        recurrence_count=len(threats),
                        correlation_score=correlation_score,
                        propagation_path=propagation_path
                    )
                    
                    cross_domain_threats.append(cross_threat)
                    self.cross_domain_threats[cross_threat.threat_id] = cross_threat
            
            return cross_domain_threats
            
        except Exception as e:
            print(f"❌ Cross-domain threat detection failed: {e}")
            return []
    
    def _get_recent_threats(self, hours: int) -> List[Dict]:
        """Get recent threats (simplified - would query database)"""
        # This would query SecurityMemory table
        return []  # Placeholder - returns mock data for now
    
    def _calculate_temporal_correlation(self, timestamps: List[datetime]) -> float:
        """Calculate temporal correlation between threats"""
        if len(timestamps) < 2:
            return 0.0
        
        # Sort timestamps
        sorted_times = sorted(timestamps)
        
        # Calculate time differences
        time_diffs = []
        for i in range(1, len(sorted_times)):
            diff = (sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600  # hours
            time_diffs.append(diff)
        
        # If threats are within 2 hours of each other, high correlation
        avg_diff = statistics.mean(time_diffs) if time_diffs else 24.0
        correlation = max(0.0, 1.0 - (avg_diff / 6.0))  # 0-1 scale, 6 hours threshold
        
        return min(correlation, 1.0)
    
    def _determine_propagation_path(self, threats: List[Dict]) -> List[str]:
        """Determine likely propagation path between domains"""
        if not threats:
            return []
        
        # Sort by time
        sorted_threats = sorted(
            threats, 
            key=lambda x: datetime.fromisoformat(x.get("first_observed", datetime.now().isoformat()))
        )
        
        # Extract domains in order
        path = []
        for threat in sorted_threats:
            domain = threat.get("source_domain", "unknown")
            if domain not in path:
                path.append(domain)
        
        return path
    
    # ============================================================================
    # ECOSYSTEM-WIDE POLICY ENFORCEMENT
    # ============================================================================
    
    def create_ecosystem_policy(self, 
                                policy_type: str, 
                                affected_domains: List[str],
                                trigger_conditions: Dict[str, Any],
                                actions: List[str]) -> str:
        """
        Create a policy that applies across multiple domains.
        """
        policy_id = f"ep_{hashlib.md5((policy_type + ''.join(affected_domains)).encode()).hexdigest()[:16]}"
        
        policy = EcosystemPolicy(
            policy_id=policy_id,
            policy_type=policy_type,
            affected_domains=affected_domains,
            trigger_conditions=trigger_conditions,
            actions=actions
        )
        
        self.ecosystem_policies[policy_id] = policy
        
        # Record policy creation in database
        self._record_ecosystem_policy(policy)
        
        return policy_id
    
    def _record_ecosystem_policy(self, policy: EcosystemPolicy):
        """Record ecosystem policy in database"""
        try:
            # This would create an AutonomousDecision record
            decision_data = {
                "type": "ecosystem_policy_creation",
                "trigger": "cross_domain_threat",
                "scope": "ecosystem",
                "reversible": True,
                "safety": "high"
            }
            
            # Add policy context
            decision_data["policy_context"] = {
                "policy_id": policy.policy_id,
                "policy_type": policy.policy_type,
                "affected_domains": policy.affected_domains,
                "actions": policy.actions
            }
            
            # Make autonomous decision with context
            self.make_autonomous_decision_with_context(decision_data)
            
        except Exception as e:
            print(f"⚠️  Failed to record ecosystem policy: {e}")
    
    def apply_ecosystem_policy(self, policy_id: str, threat_context: Dict[str, Any]) -> bool:
        """
        Apply an ecosystem policy to a specific threat context.
        """
        if policy_id not in self.ecosystem_policies:
            return False
        
        policy = self.ecosystem_policies[policy_id]
        
        # Check if trigger conditions are met
        if not self._check_policy_conditions(policy, threat_context):
            return False
        
        # Execute actions
        success = self._execute_policy_actions(policy, threat_context)
        
        if success:
            # Update policy statistics
            policy.last_applied = datetime.now()
            policy.application_count += 1
            
            # Record policy application
            self._record_policy_application(policy, threat_context, success)
        
        return success
    
    def _check_policy_conditions(self, policy: EcosystemPolicy, context: Dict[str, Any]) -> bool:
        """Check if policy conditions are met"""
        try:
            # Check domain match
            threat_domain = context.get("domain", "")
            if threat_domain and threat_domain not in policy.affected_domains:
                return False
            
            # Check severity threshold
            min_severity = policy.trigger_conditions.get("min_severity", 0.0)
            threat_severity = context.get("severity", 0.0)
            if threat_severity < min_severity:
                return False
            
            # Check if cross-domain
            is_cross_domain = context.get("is_cross_domain", False)
            if policy.trigger_conditions.get("require_cross_domain", False) and not is_cross_domain:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _execute_policy_actions(self, policy: EcosystemPolicy, context: Dict[str, Any]) -> bool:
        """Execute policy actions"""
        try:
            actions_executed = 0
            
            for action in policy.actions:
                if action == "increase_security_posture":
                    # Increase security posture for affected domains
                    for domain in policy.affected_domains:
                        self._increase_domain_security(domain, context)
                    actions_executed += 1
                
                elif action == "propagate_alert":
                    # Propagate alert to other domains
                    self._propagate_threat_alert(context, policy.affected_domains)
                    actions_executed += 1
                
                elif action == "synchronize_confidence":
                    # Synchronize confidence thresholds across domains
                    self._synchronize_confidence_thresholds(policy.affected_domains)
                    actions_executed += 1
            
            return actions_executed > 0
            
        except Exception as e:
            print(f"❌ Failed to execute policy actions: {e}")
            return False
    
    def _increase_domain_security(self, domain: str, context: Dict[str, Any]):
        """Increase security posture for a domain"""
        print(f"🛡️  Increasing security posture for domain: {domain}")
        # This would update domain-specific security policies
    
    def _propagate_threat_alert(self, context: Dict[str, Any], target_domains: List[str]):
        """Propagate threat alert to other domains"""
        print(f"📢 Propagating threat alert to domains: {target_domains}")
        # This would send alerts to other domain controllers
    
    def _synchronize_confidence_thresholds(self, domains: List[str]):
        """Synchronize confidence thresholds across domains"""
        print(f"🔄 Synchronizing confidence thresholds for domains: {domains}")
        # This would update confidence thresholds
    
    def _record_policy_application(self, policy: EcosystemPolicy, context: Dict[str, Any], success: bool):
        """Record policy application in database"""
        try:
            decision_data = {
                "type": "ecosystem_policy_application",
                "trigger": "policy_trigger",
                "scope": "ecosystem",
                "reversible": True,
                "safety": "medium"
            }
            
            # Add policy and context
            decision_data["policy_application"] = {
                "policy_id": policy.policy_id,
                "policy_type": policy.policy_type,
                "affected_domains": policy.affected_domains,
                "context": context,
                "success": success
            }
            
            self.make_autonomous_decision_with_context(decision_data)
            
        except Exception as e:
            print(f"⚠️  Failed to record policy application: {e}")
    
    # ============================================================================
    # INTELLIGENCE PROPAGATION
    # ============================================================================
    
    def propagate_intelligence_across_domains(self, 
                                             source_domain: str, 
                                             intelligence_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Propagate intelligence from one domain to others.
        Returns success status for each target domain.
        """
        results = {}
        
        try:
            # Get all other domains
            all_domains = list(self.domain_intelligence.keys())
            target_domains = [d for d in all_domains if d != source_domain]
            
            for target_domain in target_domains:
                success = self._propagate_to_domain(source_domain, target_domain, intelligence_data)
                results[target_domain] = success
            
            # Update source domain intelligence maturity
            if source_domain in self.domain_intelligence:
                self.domain_intelligence[source_domain].intelligence_maturity = min(
                    self.domain_intelligence[source_domain].intelligence_maturity + 0.05,
                    1.0
                )
            
            return results
            
        except Exception as e:
            print(f"❌ Intelligence propagation failed: {e}")
            return {domain: False for domain in target_domains}
    
    def _propagate_to_domain(self, source: str, target: str, intelligence: Dict[str, Any]) -> bool:
        """Propagate intelligence to specific domain"""
        try:
            # Calculate propagation effectiveness based on domain similarity
            similarity = self._calculate_domain_similarity(source, target)
            
            # Apply decay based on similarity
            decay_factor = 0.3 + (similarity * 0.7)  # 30-100% effectiveness
            
            # Get intelligence score
            intelligence_score = intelligence.get("score", 0.0)
            propagated_score = intelligence_score * decay_factor
            
            # Find models in target domain to update
            target_models = self.get_models_by_domain(target)
            
            if target_models:
                # Update intelligence for all models in target domain
                for model in target_models:
                    model_id = model.get("model_id")
                    if model_id:
                        self.propagate_intelligence(model_id, {"score": propagated_score})
                
                print(f"📤 Propagated intelligence {source} → {target}: {propagated_score:.3f} (similarity: {similarity:.3f})")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Failed to propagate to domain {target}: {e}")
            return False
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between two domains"""
        # Domain similarity matrix (could be learned over time)
        similarity_matrix = {
            "vision": {"tabular": 0.3, "text": 0.2, "time_series": 0.4, "hybrid": 0.5},
            "tabular": {"vision": 0.3, "text": 0.4, "time_series": 0.7, "hybrid": 0.6},
            "text": {"vision": 0.2, "tabular": 0.4, "time_series": 0.3, "hybrid": 0.5},
            "time_series": {"vision": 0.4, "tabular": 0.7, "text": 0.3, "hybrid": 0.6},
            "hybrid": {"vision": 0.5, "tabular": 0.6, "text": 0.5, "time_series": 0.6}
        }
        
        if domain1 == domain2:
            return 1.0
        
        matrix = similarity_matrix.get(domain1, {})
        return matrix.get(domain2, 0.2)  # Default low similarity
    
    # ============================================================================
    # ECOSYSTEM HEALTH & ANALYTICS
    # ============================================================================
    
    def get_ecosystem_health_report(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem health report"""
        try:
            # Domain health scores
            domain_health = {}
            for domain, intelligence in self.domain_intelligence.items():
                health_score = self._calculate_domain_health(intelligence)
                domain_health[domain] = {
                    "health_score": health_score,
                    "model_count": intelligence.model_count,
                    "threat_frequency": intelligence.threat_frequency,
                    "intelligence_maturity": intelligence.intelligence_maturity
                }
            
            # Cross-domain threat analysis
            cross_domain_threats = list(self.cross_domain_threats.values())
            multi_domain_threats = [t for t in cross_domain_threats if t.is_multi_domain()]
            
            # Policy effectiveness
            policy_effectiveness = {}
            for policy_id, policy in self.ecosystem_policies.items():
                effectiveness = policy.effectiveness_score if policy.application_count > 0 else 0.0
                policy_effectiveness[policy_id] = {
                    "type": policy.policy_type,
                    "effectiveness": effectiveness,
                    "application_count": policy.application_count
                }
            
            # Overall ecosystem health
            overall_health = self._calculate_overall_ecosystem_health(domain_health)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_health": overall_health,
                "domain_health": domain_health,
                "cross_domain_threats": {
                    "total": len(cross_domain_threats),
                    "multi_domain": len(multi_domain_threats),
                    "recent_multi_domain": [t.threat_id for t in multi_domain_threats[:5]]
                },
                "ecosystem_policies": policy_effectiveness,
                "intelligence_propagation": self._get_propagation_metrics(),
                "recommendations": self._generate_ecosystem_recommendations(domain_health)
            }
            
        except Exception as e:
            print(f"❌ Failed to generate ecosystem health report: {e}")
            return {"error": str(e)}
    
    def _calculate_domain_health(self, intelligence: DomainIntelligence) -> float:
        """Calculate health score for a domain"""
        # Start with intelligence maturity
        health = intelligence.intelligence_maturity * 0.4
        
        # Adjust for threat frequency (higher threats = lower health)
        threat_penalty = min(intelligence.threat_frequency * 0.2, 0.3)
        health -= threat_penalty
        
        # Adjust for model count (more models = better coverage)
        model_bonus = min(intelligence.model_count * 0.05, 0.3)
        health += model_bonus
        
        # Adjust for risk distribution (more high-risk = lower health)
        high_risk_count = intelligence.risk_distribution.get("critical", 0) + intelligence.risk_distribution.get("high", 0)
        risk_penalty = min(high_risk_count * 0.05, 0.2)
        health -= risk_penalty
        
        return max(0.0, min(1.0, health))
    
    def _calculate_overall_ecosystem_health(self, domain_health: Dict[str, Dict]) -> float:
        """Calculate overall ecosystem health"""
        if not domain_health:
            return 0.7  # Default
        
        # Weight domains by criticality
        domain_weights = {
            "tabular": 1.3,    # Financial/risk critical
            "time_series": 1.2,
            "vision": 1.0,
            "text": 0.9,
            "hybrid": 1.1
        }
        
        weighted_scores = []
        total_weight = 0
        
        for domain, health_data in domain_health.items():
            weight = domain_weights.get(domain, 1.0)
            weighted_scores.append(health_data["health_score"] * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.7
        
        return sum(weighted_scores) / total_weight
    
    def _get_propagation_metrics(self) -> Dict[str, Any]:
        """Get intelligence propagation metrics"""
        # This would query propagation history from database
        return {
            "total_propagations": 0,
            "success_rate": 0.0,
            "recent_propagations": []
        }
    
    def _generate_ecosystem_recommendations(self, domain_health: Dict[str, Dict]) -> List[str]:
        """Generate ecosystem improvement recommendations"""
        recommendations = []
        
        # Check for low health domains
        for domain, health_data in domain_health.items():
            if health_data["health_score"] < 0.6:
                recommendations.append(
                    f"Improve security coverage for {domain} domain "
                    f"(health: {health_data['health_score']:.2f})"
                )
        
        # Check for intelligence maturity
        for domain, health_data in domain_health.items():
            if health_data["intelligence_maturity"] < 0.5:
                recommendations.append(
                    f"Increase intelligence gathering for {domain} domain "
                    f"(maturity: {health_data['intelligence_maturity']:.2f})"
                )
        
        # Check for cross-domain threat readiness
        if not self.ecosystem_policies:
            recommendations.append(
                "Create ecosystem-wide policies for cross-domain threat response"
            )
        
        # Ensure at least one recommendation
        if not recommendations:
            recommendations.append(
                "Ecosystem is healthy. Consider proactive threat hunting exercises."
            )
        
        return recommendations[:5]  # Return top 5 recommendations

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_ecosystem_authority_engine():
    """Factory function to create Phase 5.2 ecosystem authority engine"""
    return EcosystemAuthorityEngine()
