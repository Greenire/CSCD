import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

class Config:
    """Configuration manager for loading and accessing YAML config"""
    
    def __init__(self, config: Union[str, Dict[str, Any], None] = None):
        """
        Initialize configuration manager
        
        Args:
            config: Configuration file path or dictionary (defaults to config.yaml in current directory)
        """
        self.logger = logging.getLogger(__name__)
        
        if isinstance(config, dict):
            self.config = config
            self.config_path = None
        else:
            self.config_path = Path(config) if config else Path('config.yaml')
            self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            self.logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key using dot notation (e.g. 'data_processing.raw_data_path')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            self.logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)


_config_instance = None

def get_config(config: Union[str, Dict[str, Any], None] = None) -> Config:

    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config)
    return _config_instance

def load_config(config: Union[str, Dict[str, Any], None] = None) -> Dict[str, Any]:

    return get_config(config).config
