"""
3️⃣ SECURITY MEMORY - Compressed threat experience
Purpose: Stores signals only, never raw data. Enables learning without liability.
"""

from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, CheckConstraint, Index, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from database.models.base import Base

class SecurityMemory(Base):
    __tablename__ = "security_memory"
    
    # Core Identification
    memory_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Threat Pattern Signature (hashed, not raw)
    pattern_signature = Column(String(64), unique=True, nullable=False)
    pattern_type = Column(String(30), nullable=False)
    
    # Domain Context
    source_domain = Column(String(20), nullable=False)
    affected_domains = Column(ARRAY(String(20)), nullable=False, default=[])
    
    # Signal Compression (NO RAW DATA)
    confidence_delta_vector = Column(JSON, nullable=False)  # Array of deltas, not raw confidences
    perturbation_statistics = Column(JSON, nullable=False)   # Stats only, not perturbations
    anomaly_signature_hash = Column(String(64), nullable=False)
    
    # Recurrence Tracking
    first_observed = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_observed = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    recurrence_count = Column(Integer, nullable=False, default=1, server_default="1")
    
    # Severity & Impact
    severity_score = Column(
        Float, 
        nullable=False, 
        default=0.5,
        server_default="0.5"
    )
    
    confidence_impact = Column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0.0"
    )
    
    # Cross-Model Correlations
    correlated_patterns = Column(ARRAY(UUID(as_uuid=True)), nullable=False, default=[])
    correlation_strength = Column(Float, nullable=False, default=0.0, server_default="0.0")
    
    # Mitigation Intelligence
    effective_mitigations = Column(ARRAY(String(100)), nullable=False, default=[])
    mitigation_effectiveness = Column(Float, nullable=False, default=0.0, server_default="0.0")
    
    # Learning Source
    learned_from_models = Column(ARRAY(String(100)), nullable=False, default=[])
    compressed_experience = Column(JSON, nullable=False, default=dict, server_default="{}")
    
    # Relationships
    model_id = Column(String(100), ForeignKey("model_registry.model_id"))
    model = relationship("ModelRegistry", back_populates="security_memories")
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "pattern_type IN ('confidence_erosion', 'adversarial_pattern', 'anomaly_signature', 'distribution_shift', 'temporal_attack', 'cross_model_correlation')",
            name="ck_security_memory_pattern_type"
        ),
        CheckConstraint(
            "severity_score >= 0.0 AND severity_score <= 1.0",
            name="ck_security_memory_severity"
        ),
        CheckConstraint(
            "confidence_impact >= -1.0 AND confidence_impact <= 1.0",
            name="ck_security_memory_confidence_impact"
        ),
        CheckConstraint(
            "correlation_strength >= 0.0 AND correlation_strength <= 1.0",
            name="ck_security_memory_correlation"
        ),
        CheckConstraint(
            "mitigation_effectiveness >= 0.0 AND mitigation_effectiveness <= 1.0",
            name="ck_security_memory_mitigation"
        ),
        Index("idx_security_memory_pattern_type", "pattern_type"),
        Index("idx_security_memory_severity", "severity_score"),
        Index("idx_security_memory_recurrence", "recurrence_count"),
        Index("idx_security_memory_domain", "source_domain"),
        Index("idx_security_memory_recency", "last_observed"),
    )
    
    def __repr__(self):
        return f"<SecurityMemory {self.pattern_signature[:16]}...: {self.pattern_type}>"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "memory_id": str(self.memory_id),
            "pattern_signature": self.pattern_signature,
            "pattern_type": self.pattern_type,
            "source_domain": self.source_domain,
            "affected_domains": self.affected_domains,
            "severity_score": self.severity_score,
            "confidence_impact": self.confidence_impact,
            "recurrence_count": self.recurrence_count,
            "first_observed": self.first_observed.isoformat() if self.first_observed else None,
            "last_observed": self.last_observed.isoformat() if self.last_observed else None,
            "correlation_strength": self.correlation_strength,
            "effective_mitigations": self.effective_mitigations,
            "mitigation_effectiveness": self.mitigation_effectiveness,
            "learned_from_models": self.learned_from_models
        }
    
    @classmethod
    def get_by_pattern_type(cls, session, pattern_type, limit: int = 100):
        """Get security memories by pattern type"""
        return (
            session.query(cls)
            .filter(cls.pattern_type == pattern_type)
            .order_by(cls.last_observed.desc())
            .limit(limit)
            .all()
        )
    
    @classmethod
    def get_recent_threats(cls, session, hours: int = 24, limit: int = 50):
        """Get recent threats within specified hours"""
        from datetime import datetime, timedelta
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        return (
            session.query(cls)
            .filter(cls.last_observed >= time_threshold)
            .order_by(cls.severity_score.desc(), cls.last_observed.desc())
            .limit(limit)
            .all()
        )
    
    def record_recurrence(self, new_severity: float = None, new_confidence_impact: float = None):
        """Record another occurrence of this pattern"""
        from datetime import datetime
        
        self.last_observed = datetime.utcnow()
        self.recurrence_count += 1
        
        # Update severity with decayed average
        if new_severity is not None:
            decay = 0.8  # 80% weight to history
            self.severity_score = (
                decay * self.severity_score + 
                (1 - decay) * new_severity
            )
        
        # Update confidence impact
        if new_confidence_impact is not None:
            self.confidence_impact = (
                decay * self.confidence_impact + 
                (1 - decay) * new_confidence_impact
            )
    
    def add_mitigation(self, mitigation: str, effectiveness: float):
        """Add a mitigation strategy for this pattern"""
        if mitigation not in self.effective_mitigations:
            self.effective_mitigations.append(mitigation)
        
        # Update effectiveness score
        if self.mitigation_effectiveness == 0.0:
            self.mitigation_effectiveness = effectiveness
        else:
            # Weighted average
            self.mitigation_effectiveness = 0.7 * self.mitigation_effectiveness + 0.3 * effectiveness
    
    def add_correlation(self, other_memory_id: uuid.UUID, strength: float):
        """Add correlation with another security memory pattern"""
        if other_memory_id not in self.correlated_patterns:
            self.correlated_patterns.append(other_memory_id)
        
        # Update correlation strength
        if strength > self.correlation_strength:
            self.correlation_strength = strength
