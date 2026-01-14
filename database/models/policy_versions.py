"""
5️⃣ POLICY VERSIONS - Governance over time
Purpose: All policy changes versioned, tracked, and auditable for rollback.
"""

from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean, Text, CheckConstraint, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from database.models.base import Base

class PolicyVersion(Base):
    __tablename__ = "policy_versions"
    
    # Core Identification
    policy_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    effective_from = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Policy Identity
    policy_type = Column(String(30), nullable=False)
    policy_scope = Column(String(20), nullable=False)
    
    # Version Chain
    previous_version = Column(UUID(as_uuid=True), ForeignKey("policy_versions.policy_id"))
    previous = relationship("PolicyVersion", remote_side=[policy_id])
    version_hash = Column(String(64), unique=True, nullable=False)
    version_number = Column(Integer, nullable=False)
    
    # Policy Content
    policy_parameters = Column(JSON, nullable=False)
    policy_constraints = Column(JSON, nullable=False)
    
    # Change Management
    change_reason = Column(String(200), nullable=False)
    change_trigger = Column(String(30), nullable=False)
    
    # Effectiveness Tracking
    threat_correlation = Column(JSON, nullable=False, default=dict, server_default="{}")
    effectiveness_score = Column(Float)
    effectiveness_measured_at = Column(DateTime(timezone=True))
    
    # Rollback Information
    can_rollback_to = Column(Boolean, nullable=False, default=True, server_default="true")
    rollback_instructions = Column(Text)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "policy_type IN ('confidence_threshold', 'rate_limiting', 'security_escalation', 'learning_parameters', 'model_promotion', 'cross_model_alerting')",
            name="ck_policy_type"
        ),
        CheckConstraint(
            "policy_scope IN ('global', 'domain', 'risk_tier', 'model')",
            name="ck_policy_scope"
        ),
        CheckConstraint(
            "change_trigger IN ('threat_response', 'false_positive_adjustment', 'performance_optimization', 'ecosystem_evolution', 'human_intervention', 'scheduled_review')",
            name="ck_policy_change_trigger"
        ),
        CheckConstraint(
            "effectiveness_score IS NULL OR (effectiveness_score >= 0.0 AND effectiveness_score <= 1.0)",
            name="ck_policy_effectiveness_score"
        ),
        Index("idx_policies_type", "policy_type"),
        Index("idx_policies_version", "version_number"),
        Index("idx_policies_effective", "effective_from"),
        Index("idx_policies_effectiveness", "effectiveness_score"),
        Index("idx_policies_type_scope", "policy_type", "policy_scope", "version_number", unique=True),
    )
    
    def __repr__(self):
        return f"<PolicyVersion {self.policy_type}/{self.policy_scope}: v{self.version_number}>"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "policy_id": str(self.policy_id),
            "policy_type": self.policy_type,
            "policy_scope": self.policy_scope,
            "version_number": self.version_number,
            "version_hash": self.version_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "change_reason": self.change_reason,
            "change_trigger": self.change_trigger,
            "effectiveness_score": self.effectiveness_score,
            "can_rollback_to": self.can_rollback_to
        }
    
    @classmethod
    def get_current_version(cls, session, policy_type: str, policy_scope: str):
        """Get current version of a policy"""
        return (
            session.query(cls)
            .filter(cls.policy_type == policy_type)
            .filter(cls.policy_scope == policy_scope)
            .order_by(cls.version_number.desc())
            .first()
        )
    
    @classmethod
    def get_version_history(cls, session, policy_type: str, policy_scope: str, limit: int = 20):
        """Get version history of a policy"""
        return (
            session.query(cls)
            .filter(cls.policy_type == policy_type)
            .filter(cls.policy_scope == policy_scope)
            .order_by(cls.version_number.desc())
            .limit(limit)
            .all()
        )
    
    @classmethod
    def create_new_version(cls, session, policy_type: str, policy_scope: str, 
                          parameters: dict, constraints: dict, change_reason: str, 
                          change_trigger: str, previous_version=None):
        """Create new policy version"""
        import hashlib
        import json
        
        # Get current version number
        current = cls.get_current_version(session, policy_type, policy_scope)
        version_number = current.version_number + 1 if current else 1
        
        # Create version hash
        content = {
            "policy_type": policy_type,
            "policy_scope": policy_scope,
            "version": version_number,
            "parameters": parameters,
            "constraints": constraints
        }
        content_json = json.dumps(content, sort_keys=True)
        version_hash = hashlib.sha256(content_json.encode()).hexdigest()
        
        # Create new version
        new_version = cls(
            policy_type=policy_type,
            policy_scope=policy_scope,
            version_number=version_number,
            version_hash=version_hash,
            policy_parameters=parameters,
            policy_constraints=constraints,
            change_reason=change_reason,
            change_trigger=change_trigger,
            previous_version=previous_version.policy_id if previous_version else None
        )
        
        session.add(new_version)
        return new_version
    
    def record_effectiveness(self, score: float, threat_data: dict = None):
        """Record effectiveness measurement"""
        from datetime import datetime
        
        self.effectiveness_score = score
        self.effectiveness_measured_at = datetime.utcnow()
        
        if threat_data:
            if "threat_correlations" not in self.threat_correlation:
                self.threat_correlation["threat_correlations"] = []
            
            self.threat_correlation["threat_correlations"].append({
                "timestamp": self.effectiveness_measured_at.isoformat(),
                "score": score,
                "threat_data": threat_data
            })
    
    def get_rollback_path(self, session):
        """Get path to rollback to this version"""
        path = []
        current = self
        
        while current:
            path.append({
                "policy_id": str(current.policy_id),
                "version_number": current.version_number,
                "created_at": current.created_at.isoformat() if current.created_at else None,
                "change_reason": current.change_reason
            })
            
            if current.previous_version:
                current = session.query(cls).filter(cls.policy_id == current.previous_version).first()
            else:
                break
        
        return list(reversed(path))
