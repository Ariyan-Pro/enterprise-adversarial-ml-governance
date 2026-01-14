"""
6️⃣ OPERATOR INTERACTIONS - Human-aware security
Purpose: Learns human behavior patterns for better cohabitation.
"""

from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean, Text, CheckConstraint, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from database.models.base import Base

class OperatorInteraction(Base):
    __tablename__ = "operator_interactions"
    
    # Core Identification
    interaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interaction_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Interaction Context
    interaction_type = Column(String(30), nullable=False)
    
    # Operator Identity (hashed for privacy)
    operator_hash = Column(String(64), nullable=False)
    operator_role = Column(String(20), nullable=False)
    
    # Target of Interaction
    target_type = Column(String(30), nullable=False)
    target_id = Column(String(100), nullable=False)
    
    # Interaction Details
    action_taken = Column(String(50), nullable=False)
    action_parameters = Column(JSON, nullable=False, default=dict, server_default="{}")
    
    # Decision Context
    autonomous_decision_id = Column(UUID(as_uuid=True), ForeignKey("autonomous_decisions.decision_id"))
    autonomous_decision = relationship("AutonomousDecision")
    system_state_at_interaction = Column(String(20), nullable=False)
    
    # Timing & Hesitation Patterns
    decision_latency_ms = Column(Integer)  # Time from suggestion to action
    review_duration_ms = Column(Integer)   # Time spent reviewing before action
    
    # Override Information
    was_override = Column(Boolean, nullable=False, default=False, server_default="false")
    override_reason = Column(Text)
    override_confidence = Column(Float)
    
    # Outcome
    outcome_recorded = Column(Boolean, nullable=False, default=False, server_default="false")
    outcome_notes = Column(Text)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "interaction_type IN ('policy_override', 'model_governance_change', 'security_state_adjustment', 'decision_review', 'system_configuration', 'audit_review')",
            name="ck_interaction_type"
        ),
        CheckConstraint(
            "operator_role IN ('executive', 'observer', 'analyst', 'engineer', 'admin')",
            name="ck_operator_role"
        ),
        CheckConstraint(
            "system_state_at_interaction IN ('normal', 'elevated', 'emergency', 'degraded')",
            name="ck_interaction_system_state"
        ),
        CheckConstraint(
            "override_confidence IS NULL OR (override_confidence >= 0.0 AND override_confidence <= 1.0)",
            name="ck_override_confidence"
        ),
        Index("idx_interactions_time", "interaction_time"),
        Index("idx_interactions_operator", "operator_hash"),
        Index("idx_interactions_type", "interaction_type"),
        Index("idx_interactions_override", "was_override"),
        Index("idx_interactions_decision", "autonomous_decision_id"),
    )
    
    def __repr__(self):
        return f"<OperatorInteraction {self.interaction_type} by {self.operator_role}>"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "interaction_id": str(self.interaction_id),
            "interaction_time": self.interaction_time.isoformat() if self.interaction_time else None,
            "interaction_type": self.interaction_type,
            "operator_role": self.operator_role,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "action_taken": self.action_taken,
            "was_override": self.was_override,
            "decision_latency_ms": self.decision_latency_ms,
            "review_duration_ms": self.review_duration_ms,
            "outcome_recorded": self.outcome_recorded
        }
    
    @classmethod
    def get_operator_interactions(cls, session, operator_hash: str, limit: int = 50):
        """Get interactions by specific operator"""
        return (
            session.query(cls)
            .filter(cls.operator_hash == operator_hash)
            .order_by(cls.interaction_time.desc())
            .limit(limit)
            .all()
        )
    
    @classmethod
    def get_recent_overrides(cls, session, limit: int = 100):
        """Get recent override interactions"""
        return (
            session.query(cls)
            .filter(cls.was_override == True)
            .order_by(cls.interaction_time.desc())
            .limit(limit)
            .all()
        )
    
    @classmethod
    def get_operator_statistics(cls, session, operator_hash: str):
        """Get statistics for an operator"""
        from sqlalchemy import func as sql_func
        
        stats = session.query(
            sql_func.count(cls.interaction_id).label("total_interactions"),
            sql_func.avg(cls.decision_latency_ms).label("avg_decision_latency"),
            sql_func.avg(cls.review_duration_ms).label("avg_review_duration"),
            sql_func.sum(sql_func.cast(cls.was_override, Integer)).label("total_overrides")
        ).filter(cls.operator_hash == operator_hash).first()
        
        return {
            "total_interactions": stats.total_interactions or 0,
            "avg_decision_latency": float(stats.avg_decision_latency or 0),
            "avg_review_duration": float(stats.avg_review_duration or 0),
            "total_overrides": stats.total_overrides or 0
        }
    
    def record_override(self, reason: str, confidence: float = None):
        """Record that this was an override"""
        self.was_override = True
        self.override_reason = reason
        if confidence is not None:
            self.override_confidence = confidence
    
    def record_outcome(self, notes: str):
        """Record outcome of this interaction"""
        self.outcome_recorded = True
        self.outcome_notes = notes
    
    def get_hesitation_score(self) -> float:
        """Calculate hesitation score (0-1, higher = more hesitant)"""
        if not self.review_duration_ms:
            return 0.0
        
        # Normalize review duration (assuming > 5 minutes is high hesitation)
        normalized = min(self.review_duration_ms / (5 * 60 * 1000), 1.0)
        
        # If decision latency is high, increase hesitation score
        if self.decision_latency_ms and self.decision_latency_ms > 30000:  # 30 seconds
            normalized = min(normalized + 0.3, 1.0)
        
        return normalized
