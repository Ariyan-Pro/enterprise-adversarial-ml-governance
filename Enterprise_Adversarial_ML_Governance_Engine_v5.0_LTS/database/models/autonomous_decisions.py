"""
4️⃣ AUTONOMOUS DECISIONS - Explainability & accountability
Purpose: Every autonomous decision logged for 10-year auditability.
"""

from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean, CheckConstraint, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from database.models.base import Base

class AutonomousDecision(Base):
    __tablename__ = "autonomous_decisions"
    
    # Core Identification
    decision_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    decision_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Decision Context
    trigger_type = Column(String(30), nullable=False)
    
    # System State at Decision
    system_state = Column(String(20), nullable=False)
    security_posture = Column(String(20), nullable=False)
    
    # Policy Application
    policy_envelope_hash = Column(String(64), nullable=False)
    policy_version = Column(Integer, nullable=False)
    
    # Decision Details
    decision_type = Column(String(30), nullable=False)
    decision_scope = Column(String(20), nullable=False)
    
    # Reversibility & Safety
    is_reversible = Column(Boolean, nullable=False, default=True)
    safety_level = Column(String(20), nullable=False, default="medium")
    
    # Affected Entities
    affected_model_id = Column(String(100), ForeignKey("model_registry.model_id"))
    affected_model = relationship("ModelRegistry", back_populates="decisions")
    affected_domains = Column(JSON, nullable=False, default=list, server_default="[]")
    
    # Decision Rationale (compressed)
    decision_rationale = Column(JSON, nullable=False)
    confidence_in_decision = Column(Float, nullable=False, default=0.5, server_default="0.5")
    
    # Outcome Tracking
    outcome_recorded = Column(Boolean, nullable=False, default=False, server_default="false")
    outcome_score = Column(Float)
    outcome_observed_at = Column(DateTime(timezone=True))
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "trigger_type IN ('threat_detected', 'confidence_anomaly', 'rate_limit_breach', 'model_uncertainty', 'ecosystem_signal', 'scheduled_policy', 'human_override')",
            name="ck_decision_trigger_type"
        ),
        CheckConstraint(
            "system_state IN ('normal', 'elevated', 'emergency', 'degraded')",
            name="ck_decision_system_state"
        ),
        CheckConstraint(
            "security_posture IN ('relaxed', 'balanced', 'strict', 'maximal')",
            name="ck_decision_security_posture"
        ),
        CheckConstraint(
            "decision_type IN ('block_request', 'increase_threshold', 'reduce_confidence', 'escalate_security', 'propagate_alert', 'pause_learning', 'model_freeze')",
            name="ck_decision_decision_type"
        ),
        CheckConstraint(
            "decision_scope IN ('local', 'model', 'domain', 'ecosystem')",
            name="ck_decision_scope"
        ),
        CheckConstraint(
            "safety_level IN ('low', 'medium', 'high', 'critical')",
            name="ck_decision_safety_level"
        ),
        CheckConstraint(
            "confidence_in_decision >= 0.0 AND confidence_in_decision <= 1.0",
            name="ck_decision_confidence"
        ),
        CheckConstraint(
            "outcome_score IS NULL OR (outcome_score >= 0.0 AND outcome_score <= 1.0)",
            name="ck_decision_outcome_score"
        ),
        Index("idx_decisions_time", "decision_time"),
        Index("idx_decisions_trigger", "trigger_type"),
        Index("idx_decisions_model", "affected_model_id"),
        Index("idx_decisions_outcome", "outcome_score"),
        Index("idx_decisions_reversible", "is_reversible"),
    )
    
    def __repr__(self):
        return f"<AutonomousDecision {self.decision_id[:8]}: {self.decision_type}>"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "decision_id": str(self.decision_id),
            "decision_time": self.decision_time.isoformat() if self.decision_time else None,
            "trigger_type": self.trigger_type,
            "system_state": self.system_state,
            "security_posture": self.security_posture,
            "decision_type": self.decision_type,
            "decision_scope": self.decision_scope,
            "is_reversible": self.is_reversible,
            "safety_level": self.safety_level,
            "affected_model_id": self.affected_model_id,
            "confidence_in_decision": self.confidence_in_decision,
            "outcome_recorded": self.outcome_recorded,
            "outcome_score": self.outcome_score,
            "outcome_observed_at": self.outcome_observed_at.isoformat() if self.outcome_observed_at else None
        }
    
    @classmethod
    def get_recent_decisions(cls, session, limit: int = 100):
        """Get recent autonomous decisions"""
        return (
            session.query(cls)
            .order_by(cls.decision_time.desc())
            .limit(limit)
            .all()
        )
    
    @classmethod
    def get_decisions_by_trigger(cls, session, trigger_type: str, limit: int = 50):
        """Get decisions by trigger type"""
        return (
            session.query(cls)
            .filter(cls.trigger_type == trigger_type)
            .order_by(cls.decision_time.desc())
            .limit(limit)
            .all()
        )
    
    def record_outcome(self, score: float, notes: str = ""):
        """Record outcome of this decision"""
        from datetime import datetime
        
        self.outcome_recorded = True
        self.outcome_score = score
        self.outcome_observed_at = datetime.utcnow()
        
        # Update rationale with outcome
        if "outcomes" not in self.decision_rationale:
            self.decision_rationale["outcomes"] = []
        
        self.decision_rationale["outcomes"].append({
            "timestamp": self.outcome_observed_at.isoformat(),
            "score": score,
            "notes": notes
        })
    
    def get_effectiveness(self) -> float:
        """Calculate decision effectiveness score"""
        if not self.outcome_recorded or self.outcome_score is None:
            return self.confidence_in_decision  # Fallback to initial confidence
        
        # Weighted average: 70% outcome, 30% initial confidence
        return 0.7 * self.outcome_score + 0.3 * self.confidence_in_decision
