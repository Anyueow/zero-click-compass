"""
Configuration providers implementing the ConfigurationProvider interface.
"""
import os
import json
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from ..core.interfaces import ConfigurationProvider


class EnvironmentConfigProvider(ConfigurationProvider):
    """Configuration provider using environment variables."""
    
    def __init__(self):
        load_dotenv()
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from environment."""
        if key in self._cache:
            return self._cache[key]
        
        value = os.getenv(key, default)
        
        # Type conversion for common cases
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif self._is_float(value):
                value = float(value)
        
        self._cache[key] = value
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value (for this session only)."""
        self._cache[key] = value
        os.environ[key] = str(value)
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False


class JSONConfigProvider(ConfigurationProvider):
    """Configuration provider using JSON files."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self._config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from JSON config."""
        # Support nested keys with dot notation
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value and save to file."""
        # Support nested keys with dot notation
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration to JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except IOError:
            pass  # Fail silently


class LayeredConfigProvider(ConfigurationProvider):
    """Configuration provider that layers multiple sources."""
    
    def __init__(self, providers: list[ConfigurationProvider]):
        self.providers = providers
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from first provider that has it."""
        for provider in self.providers:
            value = provider.get(key, None)
            if value is not None:
                return value
        return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value in the first provider."""
        if self.providers:
            self.providers[0].set(key, value)


def create_default_config_provider() -> ConfigurationProvider:
    """Create a default layered configuration provider."""
    return LayeredConfigProvider([
        EnvironmentConfigProvider(),
        JSONConfigProvider()
    ])


def get_default_configuration() -> Dict[str, Any]:
    """Get default system configuration."""
    return {
        'embedding': {
            'provider': 'gemini',
            'model': 'gemini-1.5-flash',
            'batch_size': 10
        },
        'index': {
            'type': 'faiss',
            'dimension': 768
        },
        'query_expansion': {
            'provider': 'gemini',
            'max_expansions': 15
        },
        'scoring': {
            'method': 'composite',
            'semantic_weight': 0.7,
            'token_overlap_weight': 0.3
        },
        'crawler': {
            'type': 'selenium',
            'max_pages': 50,
            'delay': 1.0,
            'timeout': 30
        },
        'chunking': {
            'target_tokens': 150,
            'overlap_tokens': 20
        },
        'social_media': {
            'reddit': {
                'enabled': True,
                'subreddits': ['marketing', 'SEO', 'entrepreneur', 'business']
            },
            'twitter': {
                'enabled': True,
                'search_limit': 100
            }
        },
        'data': {
            'directory': 'data',
            'formats': ['jsonl', 'csv']
        }
    } 