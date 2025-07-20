"""
Pipeline service for orchestrating the Zero-Click Compass workflow.
Follows Single Responsibility and Open/Closed principles.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.interfaces import (
    WebCrawlerService, ContentChunker, EmbeddingProvider, VectorIndex,
    QueryExpander, RelevanceScorer, SocialMediaProvider, DataRepository,
    Logger, ConfigurationProvider, PipelineStep, Pipeline
)
from ..core.models import (
    URL, WebPage, ContentChunk, Query, SearchResult, ContentAnalysis,
    SocialMediaAnalysis, PipelineContext, ContentType, TokenCount
)
from ..core.container import get_service


class CrawlStep(PipelineStep):
    """Pipeline step for web crawling."""
    
    def __init__(self, crawler: WebCrawlerService, logger: Logger):
        self.crawler = crawler
        self.logger = logger
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute web crawling step."""
        self.logger.info(f"Starting crawl of {context.start_url}")
        
        max_pages = context.config.get('max_pages', 50)
        pages_data = self.crawler.crawl_website(str(context.start_url), max_pages)
        
        for page_data in pages_data:
            url = URL(page_data['url'])
            page = WebPage(
                url=url,
                title=page_data.get('title', ''),
                content=page_data.get('text', ''),
                description=page_data.get('description', ''),
                headings=page_data.get('headings', []),
                metadata=page_data.get('metadata', {})
            )
            context.add_web_page(page)
        
        self.logger.info(f"Crawled {len(context.web_pages)} pages")
        return context
    
    def get_step_name(self) -> str:
        return "crawl"


class ChunkStep(PipelineStep):
    """Pipeline step for content chunking."""
    
    def __init__(self, chunker: ContentChunker, logger: Logger):
        self.chunker = chunker
        self.logger = logger
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute content chunking step."""
        self.logger.info("Starting content chunking")
        
        total_chunks = 0
        for page in context.web_pages:
            chunks_data = self.chunker.chunk(page.content)
            
            for i, chunk_data in enumerate(chunks_data):
                chunk = ContentChunk(
                    id=f"{page.url.value}_{i}",
                    content=chunk_data['content'],
                    url=page.url,
                    content_type=ContentType(chunk_data.get('content_type', 'text')),
                    token_count=TokenCount(chunk_data.get('tokens', 0)),
                    position=i,
                    metadata=chunk_data.get('metadata', {})
                )
                context.add_chunk(chunk)
                total_chunks += 1
        
        self.logger.info(f"Created {total_chunks} chunks")
        return context
    
    def get_step_name(self) -> str:
        return "chunk"


class EmbedStep(PipelineStep):
    """Pipeline step for generating embeddings."""
    
    def __init__(self, embedding_provider: EmbeddingProvider, 
                 vector_index: VectorIndex, logger: Logger):
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.logger = logger
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute embedding generation step."""
        self.logger.info("Generating embeddings and building index")
        
        # Extract texts for batch processing
        texts = [chunk.content for chunk in context.chunks]
        
        # Generate embeddings in batches
        embeddings = self.embedding_provider.generate_embeddings_batch(texts)
        
        # Add embeddings to chunks and prepare for indexing
        vectors = []
        metadata = []
        
        for chunk, embedding in zip(context.chunks, embeddings):
            if embedding:
                chunk.add_embedding(embedding)
                vectors.append(embedding)
                metadata.append({
                    'id': chunk.id,
                    'url': str(chunk.url),
                    'content': chunk.content,
                    'content_type': chunk.content_type.value,
                    'tokens': chunk.token_count.count
                })
        
        # Build vector index
        if vectors:
            self.vector_index.add_vectors(vectors, metadata)
        
        self.logger.info(f"Generated embeddings for {len(vectors)} chunks")
        return context
    
    def get_step_name(self) -> str:
        return "embed"


class QueryExpansionStep(PipelineStep):
    """Pipeline step for query expansion."""
    
    def __init__(self, query_expander: QueryExpander, logger: Logger):
        self.query_expander = query_expander
        self.logger = logger
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute query expansion step."""
        self.logger.info(f"Expanding query: {context.query.text}")
        
        max_expansions = context.config.get('max_expansions', 15)
        expansions = self.query_expander.expand(context.query.text, max_expansions)
        
        for expansion in expansions:
            context.query.add_expansion(expansion)
        
        self.logger.info(f"Generated {len(expansions)} query expansions")
        return context
    
    def get_step_name(self) -> str:
        return "expand"


class SearchStep(PipelineStep):
    """Pipeline step for searching and scoring."""
    
    def __init__(self, vector_index: VectorIndex, embedding_provider: EmbeddingProvider,
                 relevance_scorer: RelevanceScorer, logger: Logger):
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.relevance_scorer = relevance_scorer
        self.logger = logger
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute search and scoring step."""
        self.logger.info("Searching and scoring content")
        
        top_k = context.config.get('top_k', 10)
        
        # Search for original query and expansions
        all_queries = [context.query.text] + context.query.expansions
        
        for query_text in all_queries:
            # Generate query embedding
            query_embedding = self.embedding_provider.generate_embedding(query_text)
            if not query_embedding:
                continue
            
            # Search vector index
            search_results = self.vector_index.search(query_embedding, top_k)
            
            # Convert to domain objects
            chunks = []
            for result in search_results:
                # Find corresponding chunk
                chunk = next(
                    (c for c in context.chunks if c.id == result['id']), 
                    None
                )
                if chunk:
                    chunks.append(chunk)
            
            # Calculate relevance scores
            scores = {}
            for chunk in chunks:
                chunk_scores = self.relevance_scorer.score(query_text, {
                    'content': chunk.content,
                    'metadata': chunk.metadata
                })
                scores[chunk.id] = chunk_scores
            
            # Create search result
            search_result = SearchResult(
                query=Query(query_text),
                chunks=chunks,
                scores=scores,
                total_results=len(chunks)
            )
            
            context.add_search_result(search_result)
        
        self.logger.info(f"Completed search for {len(all_queries)} queries")
        return context
    
    def get_step_name(self) -> str:
        return "search"


class SocialAnalysisStep(PipelineStep):
    """Pipeline step for social media analysis."""
    
    def __init__(self, social_providers: List[SocialMediaProvider], logger: Logger):
        self.social_providers = social_providers
        self.logger = logger
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute social media analysis step."""
        if not context.config.get('include_social', False):
            return context
        
        self.logger.info("Analyzing social media")
        
        all_posts = []
        
        for provider in self.social_providers:
            try:
                posts_data = provider.search(context.query.text)
                
                # Convert to domain objects
                for post_data in posts_data:
                    # Create SocialMediaPost (implementation would need proper conversion)
                    # This is simplified for brevity
                    all_posts.append(post_data)
                
                self.logger.info(f"Found {len(posts_data)} posts on {provider.get_platform_name()}")
            except Exception as e:
                self.logger.error(f"Error with {provider.get_platform_name()}: {e}")
        
        # Create social media analysis
        if all_posts:
            # This would need proper conversion to SocialMediaAnalysis
            # Simplified for brevity
            self.logger.info(f"Total social media posts: {len(all_posts)}")
        
        return context
    
    def get_step_name(self) -> str:
        return "social"


class ZeroClickCompassPipeline(Pipeline):
    """Main pipeline for Zero-Click Compass analysis."""
    
    def __init__(self, config: ConfigurationProvider, logger: Logger):
        self.config = config
        self.logger = logger
        self.steps: List[PipelineStep] = []
        self._build_pipeline()
    
    def _build_pipeline(self) -> None:
        """Build the pipeline with all required steps."""
        # Get services from DI container
        crawler = get_service(WebCrawlerService)
        chunker = get_service(ContentChunker)
        embedding_provider = get_service(EmbeddingProvider)
        vector_index = get_service(VectorIndex)
        query_expander = get_service(QueryExpander)
        relevance_scorer = get_service(RelevanceScorer)
        
        # Add steps in order
        self.add_step(CrawlStep(crawler, self.logger))
        self.add_step(ChunkStep(chunker, self.logger))
        self.add_step(EmbedStep(embedding_provider, vector_index, self.logger))
        self.add_step(QueryExpansionStep(query_expander, self.logger))
        self.add_step(SearchStep(vector_index, embedding_provider, relevance_scorer, self.logger))
        
        # Optional social media step
        if self.config.get('social_media.enabled', False):
            social_providers = []  # Would get from container
            self.add_step(SocialAnalysisStep(social_providers, self.logger))
    
    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the pipeline."""
        self.steps.append(step)
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the entire pipeline."""
        self.logger.info("Starting Zero-Click Compass pipeline")
        start_time = datetime.now()
        
        try:
            for step in self.steps:
                self.logger.info(f"Executing step: {step.get_step_name()}")
                context = step.execute(context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


class PipelineService:
    """Service for managing pipeline execution."""
    
    def __init__(self, config: ConfigurationProvider, logger: Logger,
                 repository: DataRepository):
        self.config = config
        self.logger = logger
        self.repository = repository
    
    def run_analysis(self, start_url: str, query_text: str, 
                    config_overrides: Dict[str, Any] = None) -> ContentAnalysis:
        """Run complete Zero-Click Compass analysis."""
        # Create pipeline context
        url = URL(start_url)
        query = Query(query_text)
        
        pipeline_config = {}
        pipeline_config.update(config_overrides or {})
        
        context = PipelineContext(
            start_url=url,
            query=query,
            config=pipeline_config
        )
        
        # Create and execute pipeline
        pipeline = ZeroClickCompassPipeline(self.config, self.logger)
        result_context = pipeline.execute(context)
        
        # Create analysis result
        analysis = ContentAnalysis(
            chunks=result_context.chunks,
            queries=[result_context.query],
            results=result_context.search_results
        )
        
        # Save results
        self._save_results(analysis, result_context)
        
        return analysis
    
    def _save_results(self, analysis: ContentAnalysis, context: PipelineContext) -> None:
        """Save analysis results to repository."""
        try:
            # Save chunks
            chunks_data = [self._chunk_to_dict(chunk) for chunk in context.chunks]
            self.repository.save(chunks_data, 'chunks')
            
            # Save search results
            results_data = [self._result_to_dict(result) for result in context.search_results]
            self.repository.save(results_data, 'search_results')
            
            # Save analysis
            analysis_data = {
                'coverage': analysis.calculate_coverage(),
                'best_chunks': [chunk.id for chunk in analysis.get_best_performing_chunks()],
                'total_chunks': len(analysis.chunks),
                'total_queries': len(analysis.queries),
                'timestamp': datetime.now().isoformat()
            }
            self.repository.save(analysis_data, 'analysis')
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _chunk_to_dict(self, chunk: ContentChunk) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            'id': chunk.id,
            'content': chunk.content,
            'url': str(chunk.url),
            'content_type': chunk.content_type.value,
            'tokens': chunk.token_count.count,
            'position': chunk.position,
            'metadata': chunk.metadata
        }
    
    def _result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert search result to dictionary for storage."""
        return {
            'query': result.query.text,
            'chunks': [chunk.id for chunk in result.chunks],
            'total_results': result.total_results,
            'execution_time': result.execution_time
        } 