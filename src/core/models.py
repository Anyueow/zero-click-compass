"""
Domain models and value objects for the Zero-Click Compass system.
Following Domain-Driven Design (DDD) principles.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class ContentType(Enum):
    """Types of content chunks."""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TEXT = "text"
    CODE = "code"
    LIST_ITEM = "list_item"


class PlatformType(Enum):
    """Social media platform types."""
    REDDIT = "reddit"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"


class ScoringMethod(Enum):
    """Available scoring methods."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    TOKEN_OVERLAP = "token_overlap"
    LENGTH = "length"
    POSITION = "position"


# ============================================================================
# VALUE OBJECTS
# ============================================================================

@dataclass(frozen=True)
class URL:
    """Value object for URLs with validation."""
    value: str
    
    def __post_init__(self):
        if not self.value or not self.value.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {self.value}")
    
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        return urlparse(self.value).netloc
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Score:
    """Value object for relevance scores."""
    value: float
    method: ScoringMethod
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.value}")


@dataclass(frozen=True)
class TokenCount:
    """Value object for token counts."""
    count: int
    
    def __post_init__(self):
        if self.count < 0:
            raise ValueError(f"Token count cannot be negative: {self.count}")


@dataclass(frozen=True)
class EmbeddingVector:
    """Value object for embedding vectors."""
    vector: List[float]
    dimension: int = field(init=False)
    
    def __post_init__(self):
        object.__setattr__(self, 'dimension', len(self.vector))
        if self.dimension == 0:
            raise ValueError("Embedding vector cannot be empty")


# ============================================================================
# ENTITIES
# ============================================================================

@dataclass
class ContentChunk:
    """Entity representing a semantic chunk of content."""
    id: str
    content: str
    url: URL
    content_type: ContentType
    token_count: TokenCount
    position: int = 0
    embedding: Optional[EmbeddingVector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_embedding(self, vector: List[float]) -> None:
        """Add embedding vector to the chunk."""
        self.embedding = EmbeddingVector(vector)
    
    def get_text_preview(self, max_chars: int = 100) -> str:
        """Get a preview of the content."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."


@dataclass
class WebPage:
    """Entity representing a crawled web page."""
    url: URL
    title: str
    content: str
    description: str = ""
    headings: List[str] = field(default_factory=list)
    links: List[URL] = field(default_factory=list)
    crawled_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_link(self, url_str: str) -> None:
        """Add a link to the page."""
        try:
            url = URL(url_str)
            if url not in self.links:
                self.links.append(url)
        except ValueError:
            pass  # Ignore invalid URLs


@dataclass
class Query:
    """Entity representing a search query."""
    text: str
    expansions: List[str] = field(default_factory=list)
    embedding: Optional[EmbeddingVector] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_expansion(self, expansion: str) -> None:
        """Add a query expansion."""
        if expansion and expansion not in self.expansions:
            self.expansions.append(expansion)
    
    def add_embedding(self, vector: List[float]) -> None:
        """Add embedding vector to the query."""
        self.embedding = EmbeddingVector(vector)


@dataclass
class SocialMediaPost:
    """Entity representing a social media post."""
    id: str
    platform: PlatformType
    author: str
    content: str
    url: Optional[URL] = None
    engagement_score: float = 0.0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_engagement_score(self) -> float:
        """Calculate engagement score based on interactions."""
        # Simple engagement formula
        total_interactions = self.likes + (self.shares * 2) + (self.comments * 3)
        self.engagement_score = min(total_interactions / 1000.0, 1.0)
        return self.engagement_score


# ============================================================================
# AGGREGATES
# ============================================================================

@dataclass
class SearchResult:
    """Aggregate representing search results."""
    query: Query
    chunks: List[ContentChunk]
    scores: Dict[str, Score]
    total_results: int
    execution_time: float = 0.0
    
    def get_top_chunks(self, k: int = 10) -> List[ContentChunk]:
        """Get top k chunks by composite score."""
        if 'composite' in self.scores:
            # Sort by composite score
            scored_chunks = [(chunk, self.scores['composite'].value) for chunk in self.chunks]
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in scored_chunks[:k]]
        return self.chunks[:k]


@dataclass
class ContentAnalysis:
    """Aggregate representing content performance analysis."""
    chunks: List[ContentChunk]
    queries: List[Query]
    results: List[SearchResult]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def calculate_coverage(self) -> float:
        """Calculate query coverage percentage."""
        if not self.queries:
            return 0.0
        
        covered_queries = sum(1 for result in self.results if result.chunks)
        return covered_queries / len(self.queries)
    
    def get_best_performing_chunks(self, k: int = 10) -> List[ContentChunk]:
        """Get chunks that perform best across all queries."""
        chunk_scores = {}
        
        for result in self.results:
            for chunk in result.chunks:
                if chunk.id not in chunk_scores:
                    chunk_scores[chunk.id] = []
                if 'composite' in result.scores:
                    chunk_scores[chunk.id].append(result.scores['composite'].value)
        
        # Calculate average score for each chunk
        avg_scores = {
            chunk_id: sum(scores) / len(scores) 
            for chunk_id, scores in chunk_scores.items()
        }
        
        # Sort by average score
        sorted_chunks = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k chunks
        chunk_dict = {chunk.id: chunk for chunk in self.chunks}
        return [chunk_dict[chunk_id] for chunk_id, _ in sorted_chunks[:k] if chunk_id in chunk_dict]


@dataclass
class SocialMediaAnalysis:
    """Aggregate representing social media analysis."""
    query: str
    posts: List[SocialMediaPost]
    platforms: Set[PlatformType] = field(default_factory=set)
    top_influencers: List[str] = field(default_factory=list)
    total_engagement: float = 0.0
    
    def __post_init__(self):
        """Calculate derived values after initialization."""
        self.platforms = {post.platform for post in self.posts}
        self.total_engagement = sum(post.engagement_score for post in self.posts)
        self._calculate_top_influencers()
    
    def _calculate_top_influencers(self) -> None:
        """Calculate top influencers based on engagement."""
        author_engagement = {}
        for post in self.posts:
            if post.author not in author_engagement:
                author_engagement[post.author] = 0.0
            author_engagement[post.author] += post.engagement_score
        
        # Sort by total engagement
        sorted_authors = sorted(
            author_engagement.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        self.top_influencers = [author for author, _ in sorted_authors[:10]]
    
    def get_platform_breakdown(self) -> Dict[PlatformType, int]:
        """Get count of posts per platform."""
        breakdown = {}
        for platform in PlatformType:
            breakdown[platform] = sum(1 for post in self.posts if post.platform == platform)
        return breakdown


# ============================================================================
# PIPELINE CONTEXT
# ============================================================================

@dataclass
class PipelineContext:
    """Context object for pipeline execution."""
    start_url: URL
    query: Query
    config: Dict[str, Any] = field(default_factory=dict)
    web_pages: List[WebPage] = field(default_factory=list)
    chunks: List[ContentChunk] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    social_analysis: Optional[SocialMediaAnalysis] = None
    
    def add_web_page(self, page: WebPage) -> None:
        """Add a web page to the context."""
        self.web_pages.append(page)
    
    def add_chunk(self, chunk: ContentChunk) -> None:
        """Add a content chunk to the context."""
        self.chunks.append(chunk)
    
    def add_search_result(self, result: SearchResult) -> None:
        """Add a search result to the context."""
        self.search_results.append(result)
    
    def set_social_analysis(self, analysis: SocialMediaAnalysis) -> None:
        """Set the social media analysis."""
        self.social_analysis = analysis 