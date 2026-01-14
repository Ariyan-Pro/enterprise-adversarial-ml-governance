"""
📦 DATABASE CONFIGURATION - PostgreSQL for 10-year survivability
Core principle: Database enhances, never gates execution.
"""

import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime, timedelta

# ============================================================================
# DATABASE CONNECTION MANAGEMENT
# ============================================================================

@dataclass
class DatabaseConfig:
    def get(self, key, default=None):
        """Dictionary-like get method for compatibility"""
        return getattr(self, key, default)
    """Database configuration with fail-safe defaults"""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "security_nervous_system")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    
    # Connection pooling
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Timeouts (seconds)
    connect_timeout: int = 10
    statement_timeout: int = 30  # Fail fast if DB is slow
    
    # Reliability
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def test_connection_string(self) -> str:
        """Connection string for testing (no database)"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/postgres"

class DatabaseStatus(Enum):
    """Database connectivity status"""
    CONNECTED = "connected"
    DEGRADED = "degraded"    # High latency but working
    FAILOVER = "failover"    # Using memory fallback
    OFFLINE = "offline"      # Complete failure
    
    def can_write(self) -> bool:
        """Can we write to database?"""
        return self in [DatabaseStatus.CONNECTED, DatabaseStatus.DEGRADED]
    
    def can_read(self) -> bool:
        """Can we read from database?"""
        return self != DatabaseStatus.OFFLINE

# ============================================================================
# DATABASE FAILURE MODES
# ============================================================================

class DatabaseFailureMode:
    """
    Failure response strategies based on database status.
    Principle: Security tightens on failure.
    """
    
    @staticmethod
    def get_security_multiplier(status: DatabaseStatus) -> float:
        """
        How much to tighten security when database has issues.
        Higher multiplier = stricter security.
        """
        multipliers = {
            DatabaseStatus.CONNECTED: 1.0,   # Normal operation
            DatabaseStatus.DEGRADED: 1.3,    # Slightly stricter
            DatabaseStatus.FAILOVER: 1.7,    # Much stricter
            DatabaseStatus.OFFLINE: 2.0      # Maximum security
        }
        return multipliers.get(status, 2.0)
    
    @staticmethod
    def get_operation_mode(status: DatabaseStatus) -> str:
        """What mode should system operate in?"""
        modes = {
            DatabaseStatus.CONNECTED: "normal",
            DatabaseStatus.DEGRADED: "conservative",
            DatabaseStatus.FAILOVER: "memory_only",
            DatabaseStatus.OFFLINE: "emergency"
        }
        return modes.get(status, "emergency")

# ============================================================================
# DATABASE HEALTH MONITOR
# ============================================================================

class DatabaseHealthMonitor:
    """
    Monitors database health and triggers failover when needed.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.status = DatabaseStatus.CONNECTED
        self.last_check = datetime.now()
        self.latency_history = []
        self.error_count = 0
        
    def check_health(self) -> DatabaseStatus:
        """Check database health and update status"""
        try:
            import psycopg2
            start_time = datetime.now()
            
            # Try to connect and execute a simple query
            conn = psycopg2.connect(
                self.config.connection_string,
                connect_timeout=self.config.connect_timeout
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            
            # Calculate latency
            latency = (datetime.now() - start_time).total_seconds() * 1000  # ms
            self.latency_history.append(latency)
            
            # Keep only last 10 readings
            if len(self.latency_history) > 10:
                self.latency_history = self.latency_history[-10:]
            
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            
            # Determine status based on latency
            if avg_latency > 1000:  # 1 second
                self.status = DatabaseStatus.DEGRADED
            elif avg_latency > 5000:  # 5 seconds
                self.status = DatabaseStatus.FAILOVER
            else:
                self.status = DatabaseStatus.CONNECTED
                self.error_count = 0
                
        except Exception as e:
            print(f"Database health check failed: {e}")
            self.error_count += 1
            
            if self.error_count >= 3:
                self.status = DatabaseStatus.OFFLINE
            else:
                self.status = DatabaseStatus.FAILOVER
        
        self.last_check = datetime.now()
        return self.status
    
    def get_metrics(self) -> dict:
        """Get database health metrics"""
        return {
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "avg_latency_ms": sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0,
            "error_count": self.error_count,
            "security_multiplier": DatabaseFailureMode.get_security_multiplier(self.status)
        }

# ============================================================================
# DATABASE SESSION MANAGEMENT
# ============================================================================

class DatabaseSessionManager:
    """
    Manages database connections with fail-safe behavior.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.health_monitor = DatabaseHealthMonitor(config)
        self._engine = None
        self._session_factory = None
        
    def initialize(self):
        """Initialize database connection pool"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            # Create engine with connection pooling
            self._engine = create_engine(
                self.config.connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False  # Set to True for debugging
            )
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False
            )
            
            print(f"Database connection pool initialized: {self.config.database}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize database: {e}")
            self._engine = None
            self._session_factory = None
            return False
    
    def get_session(self):
        """Get a database session with health check"""
        if not self._session_factory:
            raise RuntimeError("Database not initialized")
        
        # Check health before providing session
        status = self.health_monitor.check_health()
        
        if not status.can_write():
            raise DatabaseUnavailableError(
                f"Database unavailable for writes: {status.value}"
            )
        
        return self._session_factory()
    
    def execute_with_retry(self, operation, max_retries: int = None):
        """
        Execute database operation with retry logic.
        """
        if max_retries is None:
            max_retries = self.config.retry_attempts
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    import time
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise DatabaseOperationError(
                        f"Operation failed after {max_retries} attempts"
                    ) from last_exception
    
    def close(self):
        """Close all database connections"""
        if self._engine:
            self._engine.dispose()
            print("Database connections closed")

# ============================================================================
# DATABASE ERRORS
# ============================================================================

class DatabaseError(Exception):
    """Base database error"""
    pass

class DatabaseUnavailableError(DatabaseError):
    """Database is unavailable"""
    pass

class DatabaseOperationError(DatabaseError):
    """Database operation failed"""
    pass

class DatabaseConstraintError(DatabaseError):
    """Database constraint violation"""
    pass

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

# Global database configuration
DATABASE_CONFIG = DatabaseConfig()

# Initialize session manager
SESSION_MANAGER = DatabaseSessionManager(DATABASE_CONFIG)

def init_database() -> bool:
    """Initialize database connection"""
    return SESSION_MANAGER.initialize()

def get_db_session():
    """Get database session (use in FastAPI dependency)"""
    return SESSION_MANAGER.get_session()

def get_database_health() -> dict:
    """Get database health status"""
    return SESSION_MANAGER.health_monitor.get_metrics()

def shutdown_database():
    """Shutdown database connections"""
    SESSION_MANAGER.close()




# SQLite Configuration for Development
# Add this to database/config.py as an alternative

import os
from pathlib import Path

# SQLite configuration
SQLITE_CONFIG = {
    "dialect": "sqlite",
    "database": str(Path(__file__).parent.parent / "security_nervous_system.db"),
    "echo": False,
    "pool_size": 1,
    "max_overflow": 0,
    "connect_args": {"check_same_thread": False}
}

# Use SQLite if PostgreSQL not available
USE_SQLITE = True  # Set to False for production PostgreSQL

