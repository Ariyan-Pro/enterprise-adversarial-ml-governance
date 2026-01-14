"""
BASE MODEL - Common functionality for all database models
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import inspect

Base = declarative_base()

class ModelMixin:
    """Mixin with common model methods"""
    
    def to_dict(self, exclude: list = None):
        """Convert model to dictionary, excluding specified columns"""
        if exclude is None:
            exclude = []
        
        result = {}
        for column in inspect(self.__class__).columns:
            column_name = column.name
            if column_name in exclude:
                continue
            
            value = getattr(self, column_name)
            
            # Handle special types
            if hasattr(value, 'isoformat'):
                value = value.isoformat()
            elif isinstance(value, list):
                # Convert lists of UUIDs to strings
                value = [str(v) if hasattr(v, 'hex') else v for v in value]
            elif hasattr(value, 'hex'):  # UUID
                value = str(value)
            
            result[column_name] = value
        
        return result
    
    @classmethod
    def from_dict(cls, session, data: dict):
        """Create model instance from dictionary"""
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance
    
    def update_from_dict(self, data: dict):
        """Update model instance from dictionary"""
        for key, value in data.items():
            if hasattr(self, key) and key != 'id':  # Don't update primary key
                setattr(self, key, value)
