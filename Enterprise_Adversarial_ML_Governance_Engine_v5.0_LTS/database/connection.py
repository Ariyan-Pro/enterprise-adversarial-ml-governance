"""
🔌 DATABASE CONNECTION MODULE
Provides database session management for SQLite/PostgreSQL with mock fallback.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.config import DATABASE_CONFIG

class MockSession:
    """
    🧪 MOCK DATABASE SESSION
    Provides mock database functionality when real database isn't available.
    """
    
    def __init__(self):
        self._data = {
            'deployments': [],
            'models': [],
            'security_memory': [],
            'autonomous_decisions': [],
            'policy_versions': [],
            'operator_interactions': [],
            'system_health': []
        }
        self.committed = False
        
    def query(self, model_class):
        """Mock query method"""
        class MockQuery:
            def __init__(self, data):
                self.data = data
                
            def all(self):
                return []
                
            def filter(self, *args, **kwargs):
                return self
                
            def order_by(self, *args):
                return self
                
            def limit(self, limit):
                return self
                
            def first(self):
                return None
                
            def count(self):
                return 0
                
            def delete(self):
                return self
                
        return MockQuery([])
        
    def add(self, item):
        """Mock add method"""
        pass
        
    def commit(self):
        """Mock commit method"""
        self.committed = True
        
    def close(self):
        """Mock close method"""
        pass
        
    def rollback(self):
        """Mock rollback method"""
        pass

def create_sqlite_engine():
    """Create SQLite engine for development"""
    try:
        db_path = Path(__file__).parent.parent / "security_nervous_system.db"
        db_path.parent.mkdir(exist_ok=True)
        
        sqlite_url = f"sqlite:///{db_path}"
        engine = create_engine(
            sqlite_url,
            echo=False,
            connect_args={"check_same_thread": False}
        )
        
        print(f"✅ SQLite engine created at {db_path}")
        return engine
        
    except Exception as e:
        print(f"❌ Failed to create SQLite engine: {e}")
        return None

def create_postgresql_engine():
    """Create PostgreSQL engine for production"""
    try:
        # Check if we have PostgreSQL config
        if not hasattr(DATABASE_CONFIG, 'host'):
            print("⚠️  PostgreSQL not configured, using SQLite")
            return create_sqlite_engine()
            
        # Build PostgreSQL connection URL
        db_url = (
            f"postgresql://{DATABASE_CONFIG.user}:{DATABASE_CONFIG.password}"
            f"@{DATABASE_CONFIG.host}:{DATABASE_CONFIG.port}/{DATABASE_CONFIG.database}"
        )
        
        engine = create_engine(
            db_url,
            pool_size=DATABASE_CONFIG.pool_size,
            max_overflow=DATABASE_CONFIG.max_overflow,
            pool_recycle=3600,
            echo=DATABASE_CONFIG.get('echo', False)
        )
        
        print(f"✅ PostgreSQL engine created for {DATABASE_CONFIG.database}")
        return engine
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("💡 Falling back to SQLite")
        return create_sqlite_engine()

def get_engine():
    """Get database engine (PostgreSQL -> SQLite -> Mock)"""
    # Try PostgreSQL first
    engine = create_postgresql_engine()
    
    # Fallback to SQLite if PostgreSQL fails
    if engine is None:
        engine = create_sqlite_engine()
    
    # Final fallback: Mock engine
    if engine is None:
        print("⚠️  All database engines failed, using mock mode")
        return None
        
    return engine

def get_session():
    """
    Get database session with automatic fallback.
    
    Returns:
        SQLAlchemy session or MockSession
    """
    try:
        engine = get_engine()
        
        if engine is None:
            print("📊 Using MOCK database session (development)")
            return MockSession()
        
        # Create SQLAlchemy session
        Session = scoped_session(sessionmaker(bind=engine))
        session = Session()
        
        # Test connection
        session.execute("SELECT 1")
        
        print("✅ Real database session created")
        return session
        
    except OperationalError as e:
        print(f"⚠️  Database connection failed: {e}")
        print("📊 Using MOCK database session (fallback)")
        return MockSession()
        
    except Exception as e:
        print(f"❌ Unexpected database error: {e}")
        print("📊 Using MOCK database session (error fallback)")
        return MockSession()

def get_session_factory():
    """Get session factory for creating multiple sessions"""
    engine = get_engine()
    
    if engine is None:
        # Return mock session factory
        def mock_session_factory():
            return MockSession()
        return mock_session_factory
    
    Session = sessionmaker(bind=engine)
    return Session

# Global session for convenience (thread-local)
_session = None

def get_global_session():
    """Get or create global session (thread-local)"""
    global _session
    
    if _session is None:
        _session = get_session()
    
    return _session

def close_global_session():
    """Close global session"""
    global _session
    
    if _session is not None:
        _session.close()
        _session = None
        print("✅ Global database session closed")

