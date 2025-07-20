"""
Dependency injection container for the Zero-Click Compass system.
Following Dependency Inversion Principle (DIP) for loose coupling.
"""
from typing import Dict, Any, Type, TypeVar, Callable, Optional
from abc import ABC, abstractmethod
import inspect

T = TypeVar('T')


class DIContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        self._services[interface] = implementation
        self._singletons[interface] = None
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service (new instance each time)."""
        self._services[interface] = implementation
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for creating instances."""
        self._factories[interface] = factory
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a specific instance."""
        self._singletons[interface] = instance
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service dependency."""
        # Check for registered instance first
        if interface in self._singletons and self._singletons[interface] is not None:
            return self._singletons[interface]
        
        # Check for factory
        if interface in self._factories:
            instance = self._factories[interface]()
            return instance
        
        # Check for registered service
        if interface in self._services:
            implementation = self._services[interface]
            
            # Create instance with dependency injection
            instance = self._create_instance(implementation)
            
            # Cache singleton
            if interface in self._singletons:
                self._singletons[interface] = instance
            
            return instance
        
        raise ValueError(f"Service {interface} not registered")
    
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create instance with automatic dependency injection."""
        # Get constructor signature
        signature = inspect.signature(implementation.__init__)
        parameters = signature.parameters
        
        # Skip 'self' parameter
        param_names = [name for name in parameters.keys() if name != 'self']
        
        if not param_names:
            # No dependencies
            return implementation()
        
        # Resolve dependencies
        kwargs = {}
        for param_name in param_names:
            param = parameters[param_name]
            param_type = param.annotation
            
            if param_type != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.resolve(param_type)
                except ValueError:
                    # Use default value if available
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        raise ValueError(f"Cannot resolve dependency {param_type} for {implementation}")
        
        return implementation(**kwargs)


class ServiceRegistry:
    """Registry for managing service configurations."""
    
    def __init__(self):
        self.container = DIContainer()
        self._configured = False
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the service registry with implementations."""
        if self._configured:
            return
        
        # Import interfaces
        from .interfaces import (
            EmbeddingProvider, VectorIndex, QueryExpander, RelevanceScorer,
            DataRepository, SocialMediaProvider, WebCrawlerService,
            ConfigurationProvider, Logger
        )
        
        # Register default implementations
        self._register_default_services()
        
        # Override with configuration
        self._register_from_config(config)
        
        self._configured = True
    
    def _register_default_services(self) -> None:
        """Register default service implementations."""
        from ..infrastructure.config import EnvironmentConfigProvider
        from ..infrastructure.logging import StandardLogger
        from ..infrastructure.storage import JSONLRepository
        
        # Core services
        self.container.register_singleton(ConfigurationProvider, EnvironmentConfigProvider)
        self.container.register_singleton(Logger, StandardLogger)
        self.container.register_singleton(DataRepository, JSONLRepository)
    
    def _register_from_config(self, config: Dict[str, Any]) -> None:
        """Register services from configuration."""
        # Embedding provider
        embedding_config = config.get('embedding', {})
        if embedding_config.get('provider') == 'gemini':
            from ..infrastructure.embeddings import GeminiEmbeddingProvider
            self.container.register_singleton(EmbeddingProvider, GeminiEmbeddingProvider)
        
        # Vector index
        index_config = config.get('index', {})
        if index_config.get('type') == 'faiss':
            from ..infrastructure.indexes import FAISSVectorIndex
            self.container.register_singleton(VectorIndex, FAISSVectorIndex)
        
        # Query expander
        expander_config = config.get('query_expansion', {})
        if expander_config.get('provider') == 'gemini':
            from ..infrastructure.query import GeminiQueryExpander
            self.container.register_singleton(QueryExpander, GeminiQueryExpander)
        
        # Relevance scorer
        scorer_config = config.get('scoring', {})
        if scorer_config.get('method') == 'composite':
            from ..infrastructure.scoring import CompositeRelevanceScorer
            self.container.register_singleton(RelevanceScorer, CompositeRelevanceScorer)
        
        # Web crawler
        crawler_config = config.get('crawler', {})
        if crawler_config.get('type') == 'selenium':
            from ..infrastructure.crawling import SeleniumWebCrawler
            self.container.register_singleton(WebCrawlerService, SeleniumWebCrawler)
        
        # Social media providers
        social_config = config.get('social_media', {})
        if 'reddit' in social_config:
            from ..infrastructure.social import RedditProvider
            self.container.register_transient(SocialMediaProvider, RedditProvider)
        if 'twitter' in social_config:
            from ..infrastructure.social import TwitterProvider
            self.container.register_transient(SocialMediaProvider, TwitterProvider)
    
    def get_service(self, interface: Type[T]) -> T:
        """Get a service instance."""
        return self.container.resolve(interface)


# Global service registry instance
service_registry = ServiceRegistry()


def configure_services(config: Dict[str, Any]) -> None:
    """Configure the global service registry."""
    service_registry.configure(config)


def get_service(interface: Type[T]) -> T:
    """Get a service from the global registry."""
    return service_registry.get_service(interface) 