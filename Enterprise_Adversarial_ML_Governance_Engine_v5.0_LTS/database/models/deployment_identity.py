"""
1️⃣ DEPLOYMENT IDENTITY - Personalize intelligence per installation
Purpose: Ensures every instance evolves differently.
"""

from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from database.config import DATABASE_CONFIG
from database.models.base import Base

class DeploymentIdentity(Base):
    __tablename__ = "deployment_identity"
    
    # Core Identity
    deployment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Environment Fingerprint (hashed, not raw)
    environment_hash = Column(String(64), unique=True, nullable=False)
    environment_summary = Column(JSON, nullable=False)
    
    # Risk Posture Configuration
    default_risk_posture = Column(
        String(20), 
        nullable=False, 
        default="balanced",
        server_default="balanced"
    )
    
    # System Maturity (evolves over time)
    system_maturity_score = Column(
        Float, 
        nullable=False, 
        default=0.0,
        server_default="0.0"
    )
    
    # Policy Envelopes (bounds for autonomous operation)
    policy_envelopes = Column(JSON, nullable=False, default=dict, server_default="{}")
    
    # Operational Metadata
    last_heartbeat = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    heartbeat_count = Column(Integer, nullable=False, default=0, server_default="0")
    
    # Survivability Tracking
    consecutive_days_operational = Column(Integer, nullable=False, default=0, server_default="0")
    longest_uptime_days = Column(Integer, nullable=False, default=0, server_default="0")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "default_risk_posture IN ('conservative', 'balanced', 'aggressive')",
            name="ck_deployment_risk_posture"
        ),
        CheckConstraint(
            "system_maturity_score >= 0.0 AND system_maturity_score <= 1.0",
            name="ck_deployment_maturity_score"
        ),
        Index("idx_deployment_heartbeat", "last_heartbeat"),
        Index("idx_deployment_maturity", "system_maturity_score"),
    )
    
    def __repr__(self):
        return f"<DeploymentIdentity {self.deployment_id}: {self.default_risk_posture}>"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "deployment_id": str(self.deployment_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "environment_hash": self.environment_hash,
            "default_risk_posture": self.default_risk_posture,
            "system_maturity_score": self.system_maturity_score,
            "policy_envelopes": self.policy_envelopes,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "heartbeat_count": self.heartbeat_count,
            "consecutive_days_operational": self.consecutive_days_operational,
            "longest_uptime_days": self.longest_uptime_days
        }
    
    @classmethod
    def get_current_deployment(cls, session):
        """Get the current deployment (latest)"""
        return session.query(cls).order_by(cls.created_at.desc()).first()
    
    def update_heartbeat(self):
        """Update heartbeat and count"""
        from datetime import datetime
        self.last_heartbeat = datetime.utcnow()
        self.heartbeat_count += 1
        
        # Update consecutive days
        # (Simplified - real implementation would track actual uptime)
        self.consecutive_days_operational = min(
            self.consecutive_days_operational + 1,
            365 * 10  # Cap at 10 years for display
        )
        
        # Update longest uptime
        if self.consecutive_days_operational > self.longest_uptime_days:
            self.longest_uptime_days = self.consecutive_days_operational
