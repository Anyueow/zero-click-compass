"""
Scoring module for ranking query-chunk pairs and calculating relevance scores.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from .utils import logger, Tokenizer

class RelevanceScorer:
    """Score relevance between queries and content chunks."""
    
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def score_query_chunk_pair(self, query: str, chunk: Dict, 
                              scoring_methods: List[str] = None) -> Dict:
        """Score a single query-chunk pair using 0.7 cosine + 0.3 token_overlap."""
        if scoring_methods is None:
            scoring_methods = ['semantic', 'keyword']  # Only the two methods we need
        
        scores = {}
        
        for method in scoring_methods:
            try:
                if method == 'semantic':
                    scores['semantic'] = self._semantic_score(query, chunk)
                elif method == 'keyword':
                    scores['keyword'] = self._keyword_score(query, chunk)
                elif method == 'length':
                    scores['length'] = self._length_score(chunk)
                elif method == 'position':
                    scores['position'] = self._position_score(chunk)
                elif method == 'content_type':
                    scores['content_type'] = self._content_type_score(chunk)
            except Exception as e:
                logger.error(f"Error in {method} scoring: {e}")
                scores[method] = 0.0
        
        # Calculate composite score
        scores['composite'] = self._calculate_composite_score(scores)
        
        return {
            'query': query,
            'chunk_id': chunk.get('id', ''),
            'url': chunk.get('url', ''),
            'scores': scores,
            'chunk_content': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', '')
        }
    
    def _semantic_score(self, query: str, chunk: Dict) -> float:
        """Calculate semantic similarity score using embeddings."""
        if 'embedding' not in chunk or 'similarity_score' not in chunk:
            return 0.0
        
        # Use the pre-computed similarity score from FAISS
        return float(chunk['similarity_score'])
    
    def _keyword_score(self, query: str, chunk: Dict) -> float:
        """Calculate keyword overlap score."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', chunk.get('content', '').lower()))
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _length_score(self, chunk: Dict) -> float:
        """Score based on chunk length (prefer medium-length chunks)."""
        content = chunk.get('content', '')
        tokens = self.tokenizer.count_tokens(content)
        
        # Prefer chunks between 50-200 tokens
        if 50 <= tokens <= 200:
            return 1.0
        elif tokens < 20:
            return 0.3  # Too short
        elif tokens > 300:
            return 0.7  # Too long
        else:
            # Gradual falloff
            if tokens < 50:
                return 0.3 + 0.7 * (tokens - 20) / 30
            else:
                return 1.0 - 0.3 * (tokens - 200) / 100
    
    def _position_score(self, chunk: Dict) -> float:
        """Score based on chunk position in the page."""
        chunk_id = chunk.get('id', '')
        
        # Extract chunk number from ID
        match = re.search(r'chunk_(\d+)', chunk_id)
        if match:
            chunk_num = int(match.group(1))
            # Prefer earlier chunks (assume they're more important)
            return max(0.5, 1.0 - (chunk_num * 0.05))
        
        return 0.8  # Default score
    
    def _content_type_score(self, chunk: Dict) -> float:
        """Score based on content type."""
        content_type = chunk.get('content_type', 'text')
        
        # Prefer titles and headings
        if content_type == 'title':
            return 1.0
        elif 'heading' in content_type:
            return 0.9
        else:
            return 0.7
    
    def _calculate_composite_score(self, scores: Dict) -> float:
        """Calculate weighted composite score using 0.7 cosine + 0.3 token_overlap."""
        # Use the exact formula: 0.7 * cosine_similarity + 0.3 * token_overlap
        cosine_score = scores.get('semantic', 0.0)
        token_overlap = scores.get('keyword', 0.0)  # Using keyword as token overlap
        
        composite = 0.7 * cosine_score + 0.3 * token_overlap
        return composite

class QueryChunkRanker:
    """Rank chunks for a given query or set of queries."""
    
    def __init__(self):
        self.scorer = RelevanceScorer()
    
    def rank_chunks_for_query(self, query: str, chunks: List[Dict], 
                             top_k: int = 10) -> List[Dict]:
        """Rank chunks for a single query."""
        scored_pairs = []
        
        for chunk in chunks:
            score_result = self.scorer.score_query_chunk_pair(query, chunk)
            scored_pairs.append(score_result)
        
        # Sort by composite score
        scored_pairs.sort(key=lambda x: x['scores']['composite'], reverse=True)
        
        return scored_pairs[:top_k]
    
    def rank_chunks_for_queries(self, queries: List[str], chunks: List[Dict],
                               top_k: int = 10) -> Dict[str, List[Dict]]:
        """Rank chunks for multiple queries."""
        results = {}
        
        for query in queries:
            results[query] = self.rank_chunks_for_query(query, chunks, top_k)
        
        return results
    
    def get_best_chunks_across_queries(self, query_results: Dict[str, List[Dict]],
                                     top_k: int = 10) -> List[Dict]:
        """Get the best chunks across all queries."""
        all_chunks = []
        
        for query, results in query_results.items():
            for result in results:
                result['source_query'] = query
                all_chunks.append(result)
        
        # Remove duplicates based on chunk_id
        seen_chunks = set()
        unique_chunks = []
        
        for chunk in all_chunks:
            chunk_id = chunk['chunk_id']
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_chunks.append(chunk)
        
        # Sort by composite score
        unique_chunks.sort(key=lambda x: x['scores']['composite'], reverse=True)
        
        return unique_chunks[:top_k]

class PerformanceAnalyzer:
    """Analyze how well content performs across different queries."""
    
    def __init__(self):
        self.ranker = QueryChunkRanker()
    
    def analyze_content_performance(self, queries: List[str], chunks: List[Dict],
                                   top_k: int = 10) -> Dict:
        """Analyze how content performs across different queries."""
        # Rank chunks for all queries
        query_results = self.ranker.rank_chunks_for_queries(queries, chunks, top_k)
        
        # Get best chunks across all queries
        best_chunks = self.ranker.get_best_chunks_across_queries(query_results, top_k)
        
        # Analyze performance metrics
        performance_metrics = self._calculate_performance_metrics(query_results, best_chunks)
        
        return {
            'query_results': query_results,
            'best_chunks': best_chunks,
            'performance_metrics': performance_metrics
        }
    
    def _calculate_performance_metrics(self, query_results: Dict[str, List[Dict]],
                                     best_chunks: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        metrics = {
            'total_queries': len(query_results),
            'total_chunks': len(best_chunks),
            'average_scores': {},
            'top_performers': [],
            'coverage_analysis': {}
        }
        
        # Calculate average scores by scoring method
        scoring_methods = ['semantic', 'keyword', 'length', 'position', 'content_type', 'composite']
        
        for method in scoring_methods:
            scores = []
            for chunk in best_chunks:
                if method in chunk['scores']:
                    scores.append(chunk['scores'][method])
            
            if scores:
                metrics['average_scores'][method] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
        
        # Identify top performers
        top_performers = sorted(best_chunks, 
                              key=lambda x: x['scores']['composite'], 
                              reverse=True)[:5]
        
        metrics['top_performers'] = [
            {
                'chunk_id': chunk['chunk_id'],
                'url': chunk['url'],
                'composite_score': chunk['scores']['composite'],
                'content_preview': chunk['chunk_content'][:100] + '...'
            }
            for chunk in top_performers
        ]
        
        # Coverage analysis
        url_coverage = {}
        for chunk in best_chunks:
            url = chunk['url']
            if url not in url_coverage:
                url_coverage[url] = 0
            url_coverage[url] += 1
        
        metrics['coverage_analysis'] = {
            'urls_covered': len(url_coverage),
            'url_distribution': url_coverage
        }
        
        return metrics
    
    def generate_performance_report(self, analysis_results: Dict) -> str:
        """Generate a human-readable performance report."""
        metrics = analysis_results['performance_metrics']
        
        report = f"""
# Content Performance Analysis Report

## Overview
- Total Queries Analyzed: {metrics['total_queries']}
- Total Chunks Evaluated: {metrics['total_chunks']}
- URLs Covered: {metrics['coverage_analysis']['urls_covered']}

## Average Scores
"""
        
        for method, stats in metrics['average_scores'].items():
            report += f"- **{method.title()}**: {stats['mean']:.3f} (±{stats['std']:.3f})\n"
        
        report += "\n## Top Performing Content\n"
        
        for i, performer in enumerate(metrics['top_performers'], 1):
            report += f"""
{i}. **Score**: {performer['composite_score']:.3f}
   **URL**: {performer['url']}
   **Content**: {performer['content_preview']}
"""
        
        return report

def score_query_chunks(query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    """Convenience function to score chunks for a single query."""
    ranker = QueryChunkRanker()
    return ranker.rank_chunks_for_query(query, chunks, top_k)

def analyze_content_performance_simple(queries: List[str], chunks: List[Dict]) -> Dict:
    """Convenience function for content performance analysis."""
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze_content_performance(queries, chunks) 