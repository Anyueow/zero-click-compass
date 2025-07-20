"""
Data storage implementations for the Zero-Click Compass system.
"""
import os
import json
import pickle
from typing import Any, List, Dict, Optional
from datetime import datetime

from ..core.interfaces import DataRepository


class JSONLRepository(DataRepository):
    """Repository using JSONL files for storage."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save(self, data: Any, identifier: str) -> None:
        """Save data as JSONL file."""
        filepath = os.path.join(self.data_dir, f"{identifier}.jsonl")
        
        # Handle different data types
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            items = [{"data": data, "timestamp": datetime.now().isoformat()}]
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def load(self, identifier: str) -> Optional[Any]:
        """Load data from JSONL file."""
        filepath = os.path.join(self.data_dir, f"{identifier}.jsonl")
        
        if not os.path.exists(filepath):
            return None
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return data
    
    def delete(self, identifier: str) -> bool:
        """Delete data file."""
        filepath = os.path.join(self.data_dir, f"{identifier}.jsonl")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except OSError:
                return False
        return False
    
    def list_identifiers(self) -> List[str]:
        """List all available identifiers."""
        if not os.path.exists(self.data_dir):
            return []
        
        identifiers = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.jsonl'):
                identifiers.append(filename[:-6])  # Remove .jsonl extension
        
        return identifiers


class CSVRepository(DataRepository):
    """Repository using CSV files for storage."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save(self, data: Any, identifier: str) -> None:
        """Save data as CSV file."""
        import pandas as pd
        
        filepath = os.path.join(self.data_dir, f"{identifier}.csv")
        
        # Convert to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            # Fallback for other data types
            df = pd.DataFrame({"data": [str(data)], "timestamp": [datetime.now().isoformat()]})
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
    
    def load(self, identifier: str) -> Optional[Any]:
        """Load data from CSV file."""
        import pandas as pd
        
        filepath = os.path.join(self.data_dir, f"{identifier}.csv")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            return pd.read_csv(filepath)
        except Exception:
            return None
    
    def delete(self, identifier: str) -> bool:
        """Delete CSV file."""
        filepath = os.path.join(self.data_dir, f"{identifier}.csv")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except OSError:
                return False
        return False
    
    def list_identifiers(self) -> List[str]:
        """List all available identifiers."""
        if not os.path.exists(self.data_dir):
            return []
        
        identifiers = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                identifiers.append(filename[:-4])  # Remove .csv extension
        
        return identifiers


class PickleRepository(DataRepository):
    """Repository using pickle files for storage."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save(self, data: Any, identifier: str) -> None:
        """Save data as pickle file."""
        filepath = os.path.join(self.data_dir, f"{identifier}.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, identifier: str) -> Optional[Any]:
        """Load data from pickle file."""
        filepath = os.path.join(self.data_dir, f"{identifier}.pkl")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def delete(self, identifier: str) -> bool:
        """Delete pickle file."""
        filepath = os.path.join(self.data_dir, f"{identifier}.pkl")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except OSError:
                return False
        return False
    
    def list_identifiers(self) -> List[str]:
        """List all available identifiers."""
        if not os.path.exists(self.data_dir):
            return []
        
        identifiers = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pkl'):
                identifiers.append(filename[:-4])  # Remove .pkl extension
        
        return identifiers 