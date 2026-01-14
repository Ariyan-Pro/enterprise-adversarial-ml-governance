"""
2️⃣ MODEL REGISTRY - Cross-domain model governance
Purpose: Central registry for all ML models with risk-tier classification.
"""

from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from database.models.base import Base

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    # Core Identification
    model_id = Column(String(100), primary_key=True)
    model_type = Column(String(30), nullable=False)  # vision, tabular, text, time_series
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Model Characteristics
    model_family = Column(String(50), nullable=False)
    parameters_count = Column(Integer, nullable=False, default=0)
    model_size_mb = Column(Float, nullable=False, default=0.0)
    
    # Risk & Compliance
    risk_tier = Column(String(10), nullable=False)
    deployment_phase = Column(String(20), nullable=False)
    confidence_threshold = Column(Float, nullable=False, default=0.85)
    
    # Performance Metrics
    clean_accuracy = Column(Float)
    robust_accuracy = Column(Float)
    inference_latency_ms = Column(Float)
    
    # Operational Status
    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    health_score = Column(Float, nullable=False, default=1.0, server_default="1.0")
    
    # Metadata
    metadata = Column(JSON, nullable=False, default=dict, server_default="{}")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "model_type IN ('vision', 'tabular', 'text', 'time_series', 'multimodal', 'unknown')",
            name="ck_model_type"
        ),
        CheckConstraint(
            "risk_tier IN ('tier_0', 'tier_1', 'tier_2', 'tier_3')",
            name="ck_risk_tier"
        ),
        CheckConstraint(
            "deployment_phase IN ('development', 'staging', 'production', 'deprecated', 'archived')",
            name="ck_deployment_phase"
        ),
        CheckConstraint(
            "confidence_threshold >= 0.0 AND confidence_threshold <= 1.0",
            name="ck_confidence_threshold"
        ),
        CheckConstraint(
            "clean_accuracy IS NULL OR (clean_accuracy >= 0.0 AND clean_accuracy <= 1.0)",
            name="ck_clean_accuracy"
        ),
        CheckConstraint(
            "robust_accuracy IS NULL OR (robust_accuracy >= 0.0 AND robust_accuracy <= 1.0)",
            name="ck_robust_accuracy"
        ),
        CheckConstraint(
            "health_score >= 0.0 AND health_score <= 1.0",
            name="ck_health_score"
        ),
        Index("idx_models_type", "model_type"),
        Index("idx_models_risk", "risk_tier"),
        Index("idx_models_phase", "deployment_phase"),
        Index("idx_models_health", "health_score"),
        Index("idx_models_updated", "last_updated"),
    )
    
    def __repr__(self):
        return f"<ModelRegistry {self.model_id}: {self.model_type} ({self.risk_tier})>"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_family": self.model_family,
            "risk_tier": self.risk_tier,
            "deployment_phase": self.deployment_phase,
            "confidence_threshold": self.confidence_threshold,
            "parameters_count": self.parameters_count,
            "clean_accuracy": self.clean_accuracy,
            "robust_accuracy": self.robust_accuracy,
            "is_active": self.is_active,
            "health_score": self.health_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }
    
    @classmethod
    def get_active_models(cls, session, limit: int = 100):
        """Get active models"""
        return (
            session.query(cls)
            .filter(cls.is_active == True)
            .order_by(cls.last_updated.desc())
            .limit(limit)
            .all()
        )
    
    @classmethod
    def get_models_by_type(cls, session, model_type: str, limit: int = 50):
        """Get models by type"""
        return (
            session.query(cls)
            .filter(cls.model_type == model_type)
            .filter(cls.is_active == True)
            .order_by(cls.last_updated.desc())
            .limit(limit)
            .all()
        )
    
    @classmethod
    def get_models_by_risk_tier(cls, session, risk_tier: str, limit: int = 50):
        """Get models by risk tier"""
        return (
            session.query(cls)
            .filter(cls.risk_tier == risk_tier)
            .filter(cls.is_active == True)
            .order_by(cls.last_updated.desc())
            .limit(limit)
            .all()
        )
    
    def update_health_score(self, new_score: float):
        """Update health score"""
        from datetime import datetime
        
        self.health_score = max(0.0, min(1.0, new_score))
        self.last_updated = datetime.utcnow()
    
    def deactivate(self, reason: str = ""):
        """Deactivate model"""
        from datetime import datetime
        
        self.is_active = False
        self.last_updated = datetime.utcnow()
        
        # Add deactivation reason to metadata
        if "deactivation" not in self.metadata:
            self.metadata["deactivation"] = []
        
        self.metadata["deactivation"].append({
            "timestamp": self.last_updated.isoformat(),
            "reason": reason
        })
