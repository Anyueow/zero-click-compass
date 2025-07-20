"""
Core interfaces for the Zero-Click Compass system.
Following Interface Segregation Principle (ISP) - focused, role-specific interfaces.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol


# ============================================================================
# CONTENT INTERFACES
# ============================================================================

class ContentSource(Protocol):
    """Interface for content sources (web pages, documents, etc.)"""
    
    def get_content(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieve content by identifier."""
        ...
    
    def list_content(self) -> List[str]:
        """List available content identifiers."""
        ...


class ContentProcessor(ABC):
    """Abstract base for content processing operations."""
    
    @abstractmethod
    def process(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process content and return transformed result."""
        pass


class ContentChunker(ABC):
    """Abstract base for content chunking strategies."""
    
    @abstractmethod
    def chunk(self, content: str) -> List[Dict[str, Any]]:
        """Split content into semantic chunks."""
        pass


# ============================================================================
# EMBEDDING INTERFACES
# ============================================================================

class EmbeddingProvider(ABC):
    """Abstract base for embedding generation services."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text."""
        pass
    
    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        pass


class VectorIndex(ABC):
    """Abstract base for vector search indexes."""
    
    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Add vectors with metadata to the index."""
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass


# ============================================================================
# QUERY INTERFACES
# ============================================================================

class QueryExpander(ABC):
    """Abstract base for query expansion strategies."""
    
    @abstractmethod
    def expand(self, query: str, max_expansions: int = 10) -> List[str]:
        """Expand query into related queries."""
        pass


class RelevanceScorer(ABC):
    """Abstract base for relevance scoring strategies."""
    
    @abstractmethod
    def score(self, query: str, content: Dict[str, Any]) -> float:
        """Calculate relevance score between query and content."""
        pass


# ============================================================================
# STORAGE INTERFACES
# ============================================================================

class DataRepository(ABC):
    """Abstract base for data storage operations."""
    
    @abstractmethod
    def save(self, data: Any, identifier: str) -> None:
        """Save data with identifier."""
        pass
    
    @abstractmethod
    def load(self, identifier: str) -> Optional[Any]:
        """Load data by identifier."""
        pass
    
    @abstractmethod
    def delete(self, identifier: str) -> bool:
        """Delete data by identifier."""
        pass
    
    @abstractmethod
    def list_identifiers(self) -> List[str]:
        """List all available identifiers."""
        pass


# ============================================================================
# SOCIAL MEDIA INTERFACES
# ============================================================================

class SocialMediaProvider(ABC):
    """Abstract base for social media data providers."""
    
    @abstractmethod
    def search(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search for social media content."""
        pass
    
    @abstractmethod
    def get_platform_name(self) -> str:
        """Get the name of the social media platform."""
        pass


# ============================================================================
# PIPELINE INTERFACES
# ============================================================================

class PipelineStep(ABC):
    """Abstract base for pipeline step operations."""
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute the pipeline step."""
        pass
    
    @abstractmethod
    def get_step_name(self) -> str:
        """Get the name of the pipeline step."""
        pass


class Pipeline(ABC):
    """Abstract base for data processing pipelines."""
    
    @abstractmethod
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        pass
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute the entire pipeline."""
        pass


# ============================================================================
# CONFIGURATION INTERFACES
# ============================================================================

class ConfigurationProvider(ABC):
    """Abstract base for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass


# ============================================================================
# LOGGING INTERFACES
# ============================================================================

class Logger(Protocol):
    """Interface for logging operations."""
    
    def info(self, message: str) -> None:
        """Log info message."""
        ...
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        ...
    
    def error(self, message: str) -> None:
        """Log error message."""
        ...


# ============================================================================
# SERVICE INTERFACES
# ============================================================================

class WebCrawlerService(ABC):
    """Abstract base for web crawling services."""
    
    @abstractmethod
    def crawl_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Crawl a single URL."""
        pass
    
    @abstractmethod
    def crawl_website(self, start_url: str, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Crawl an entire website."""
        pass


class AnalyticsService(ABC):
    """Abstract base for analytics and reporting services."""
    
    @abstractmethod
    def analyze_performance(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content performance."""
        pass
    
    @abstractmethod
    def generate_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis report."""
        pass 