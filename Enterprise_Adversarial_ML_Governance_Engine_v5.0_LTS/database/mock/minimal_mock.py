
"""
🧪 MINIMAL MOCK DATABASE SESSION
For testing when real database isn't available.
"""

class MockDatabaseSession:
    def __init__(self):
        self.deployments = []
        self.models = []
        
    def query(self, model_class):
        class MockQuery:
            def __init__(self, data):
                self.data = data
            
            def all(self):
                return []
            
            def count(self):
                return 0
            
            def filter(self, *args, **kwargs):
                return self
            
            def order_by(self, *args):
                return self
            
            def limit(self, limit):
                return self
            
            def first(self):
                return None
        
        return MockQuery([])
    
    def add(self, item):
        pass
    
    def commit(self):
        pass
    
    def close(self):
        pass

MOCK_SESSION = MockDatabaseSession()
def get_mock_session():
    return MOCK_SESSION
