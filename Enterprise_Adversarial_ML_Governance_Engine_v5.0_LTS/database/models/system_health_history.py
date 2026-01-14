"""
7️⃣ SYSTEM HEALTH HISTORY - Self-healing diagnostics
Purpose: Long-term health tracking for predictive maintenance and failure analysis.
"""

from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from database.models.base import Base

class SystemHealthHistory(Base):
    __tablename__ = "system_health_history"
    
    # Core Identification
    health_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Health Metrics
    system_state = Column(String(20), nullable=False)
    security_posture = Column(String(20), nullable=False)
    
    # Performance Metrics
    avg_response_time_ms = Column(Integer, nullable=False)
    p95_response_time_ms = Column(Integer, nullable=False)
    request_rate_per_minute = Column(Integer, nullable=False)
    
    # Resource Utilization
    memory_usage_mb = Column(Integer, nullable=False)
    cpu_utilization_percent = Column(Integer, nullable=False)
    
    # Component Health
    database_latency_ms = Column(Integer)
    telemetry_gap_seconds = Column(Integer)
    firewall_latency_ms = Column(Integer)
    
    # Anomaly Indicators
    anomaly_score = Column(Float, nullable=False, default=0.0, server_default="0.0")
    has_degradation = Column(Boolean, nullable=False, default=False, server_default="false")
    
    # Watchdog Status
    watchdog_actions_taken = Column(Integer, nullable=False, default=0, server_default="0")
    degradation_level = Column(String(20))
    
    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "system_state IN ('normal', 'elevated', 'emergency', 'degraded')",
            name="ck_health_system_state"
        ),
        CheckConstraint(
            "security_posture IN ('relaxed', 'balanced', 'strict', 'maximal')",
            name="ck_health_security_posture"
        ),
        CheckConstraint(
            "cpu_utilization_percent >= 0 AND cpu_utilization_percent <= 100",
            name="ck_health_cpu_utilization"
        ),
        CheckConstraint(
            "anomaly_score >= 0.0 AND anomaly_score <= 1.0",
            name="ck_health_anomaly_score"
        ),
        CheckConstraint(
            "degradation_level IS NULL OR degradation_level IN ('minor', 'moderate', 'severe')",
            name="ck_health_degradation_level"
        ),
        Index("idx_health_time", "recorded_at"),
        Index("idx_health_state", "system_state"),
        Index("idx_health_anomaly", "anomaly_score"),
        Index("idx_health_degradation", "has_degradation"),
    )
    
    def __repr__(self):
        return f"<SystemHealthHistory {self.recorded_at}: {self.system_state}>"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "health_id": str(self.health_id),
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "system_state": self.system_state,
            "security_posture": self.security_posture,
            "avg_response_time_ms": self.avg_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "request_rate_per_minute": self.request_rate_per_minute,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "database_latency_ms": self.database_latency_ms,
            "anomaly_score": self.anomaly_score,
            "has_degradation": self.has_degradation,
            "watchdog_actions_taken": self.watchdog_actions_taken,
            "degradation_level": self.degradation_level
        }
    
    @classmethod
    def get_recent_health(cls, session, hours: int = 24, limit: int = 100):
        """Get recent health records"""
        from datetime import datetime, timedelta
        
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        return (
            session.query(cls)
            .filter(cls.recorded_at >= time_threshold)
            .order_by(cls.recorded_at.desc())
            .limit(limit)
            .all()
        )
    
    @classmethod
    def get_health_trends(cls, session, metric: str, hours: int = 24):
        """Get trend data for a specific metric"""
        from datetime import datetime, timedelta
        from sqlalchemy import func as sql_func
        
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        # Group by hour to see trends
        if metric == "cpu":
            metric_column = cls.cpu_utilization_percent
        elif metric == "memory":
            metric_column = cls.memory_usage_mb
        elif metric == "response_time":
            metric_column = cls.avg_response_time_ms
        elif metric == "anomaly":
            metric_column = cls.anomaly_score
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        trends = session.query(
            sql_func.date_trunc('hour', cls.recorded_at).label('hour'),
            sql_func.avg(metric_column).label('avg_value'),
            sql_func.min(metric_column).label('min_value'),
            sql_func.max(metric_column).label('max_value')
        ).filter(
            cls.recorded_at >= time_threshold
        ).group_by(
            sql_func.date_trunc('hour', cls.recorded_at)
        ).order_by('hour').all()
        
        return [
            {
                "hour": trend.hour.isoformat(),
                "avg": float(trend.avg_value),
                "min": float(trend.min_value),
                "max": float(trend.max_value)
            }
            for trend in trends
        ]
    
    @classmethod
    def get_degradation_events(cls, session, hours: int = 24):
        """Get all degradation events in timeframe"""
        from datetime import datetime, timedelta
        
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        return (
            session.query(cls)
            .filter(cls.recorded_at >= time_threshold)
            .filter(cls.has_degradation == True)
            .order_by(cls.recorded_at.desc())
            .all()
        )
    
    def calculate_overall_score(self) -> float:
        """Calculate overall health score (0-1, higher is better)"""
        # Base score starts at 1.0
        score = 1.0
        
        # Deduct for system state
        state_deductions = {
            "normal": 0.0,
            "elevated": 0.1,
            "degraded": 0.3,
            "emergency": 0.5
        }
        score -= state_deductions.get(self.system_state, 0.0)
        
        # Deduct for high CPU (>80%)
        if self.cpu_utilization_percent > 80:
            cpu_excess = (self.cpu_utilization_percent - 80) / 20  # 0-1 scale for 80-100%
            score -= cpu_excess * 0.2
        
        # Deduct for high response time (>1000ms)
        if self.avg_response_time_ms > 1000:
            response_excess = min((self.avg_response_time_ms - 1000) / 5000, 1.0)
            score -= response_excess * 0.2
        
        # Deduct for anomaly score
        score -= self.anomaly_score * 0.2
        
        # Deduct for degradation
        if self.has_degradation:
            score -= 0.1
        
        # Ensure score stays in bounds
        return max(0.0, min(1.0, score))
