import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def safe_json_dump(data, file_path, indent=2):
    """Safely dump data to JSON file, handling numpy types"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
