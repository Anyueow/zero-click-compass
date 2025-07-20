"""
Logging implementations for the Zero-Click Compass system.
"""
import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime

from ..core.interfaces import Logger as LoggerInterface


class StandardLogger(LoggerInterface):
    """Standard Python logging implementation."""
    
    def __init__(self, name: str = "zero-click-compass", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Only add handler if none exist
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            self.logger.addHandler(handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


class StructuredLogger(LoggerInterface):
    """Structured logging implementation with JSON output."""
    
    def __init__(self, name: str = "zero-click-compass"):
        self.name = name
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self._log("ERROR", message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        self._log("DEBUG", message, **kwargs)
    
    def _log(self, level: str, message: str, **kwargs) -> None:
        """Log structured message."""
        import json
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs
        }
        
        print(json.dumps(log_entry))


class FileLogger(LoggerInterface):
    """File-based logging implementation."""
    
    def __init__(self, log_file: str = "zero-click-compass.log", 
                 level: int = logging.INFO):
        self.logger = logging.getLogger(f"file-{log_file}")
        self.logger.setLevel(level)
        
        # Only add handler if none exist
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            handler.setLevel(level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            self.logger.addHandler(handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


class CompositeLogger(LoggerInterface):
    """Composite logger that delegates to multiple loggers."""
    
    def __init__(self, loggers: list[LoggerInterface]):
        self.loggers = loggers
    
    def info(self, message: str) -> None:
        """Log info message to all loggers."""
        for logger in self.loggers:
            logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message to all loggers."""
        for logger in self.loggers:
            logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message to all loggers."""
        for logger in self.loggers:
            logger.error(message)


class NullLogger(LoggerInterface):
    """Null logger that does nothing (for testing)."""
    
    def info(self, message: str) -> None:
        """Do nothing."""
        pass
    
    def warning(self, message: str) -> None:
        """Do nothing."""
        pass
    
    def error(self, message: str) -> None:
        """Do nothing."""
        pass 