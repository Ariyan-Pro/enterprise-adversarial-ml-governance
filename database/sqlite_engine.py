
"""
🧪 SQLITE DATABASE ENGINE FOR DEVELOPMENT
Provides SQLite support when PostgreSQL isn't available.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

def create_sqlite_engine():
    """Create SQLite engine for development"""
    db_path = Path(__file__).parent.parent.parent / "security_nervous_system.db"
    db_path.parent.mkdir(exist_ok=True)
    
    sqlite_url = f"sqlite:///{db_path}"
    engine = create_engine(
        sqlite_url,
        echo=False,
        connect_args={"check_same_thread": False}
    )
    
    return engine

def create_sqlite_session():
    """Create SQLite session"""
    engine = create_sqlite_engine()
    Session = sessionmaker(bind=engine)
    return Session()
