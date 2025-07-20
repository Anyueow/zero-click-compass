#!/usr/bin/env python3
"""
Comprehensive Content Scoring and Channel Analysis System
Analyzes content chunks against sub-queries and provides platform-specific strategies.
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveScorer:
    """Comprehensive content scoring and channel analysis system."""
    
    def __init__(self):
        self.supported_platforms = {
            'REDDIT': {
                'keywords': ['reddit', 'subreddit', 'community', 'discussion', 'thread', 'post'],
                'query_patterns': ['how to', 'what is', 'best', 'recommend', 'experience', 'review'],
                'content_type': 'community_participation',
                'engagement_style': 'detailed_posts_and_comments'
            },
            'TWITTER': {
                'keywords': ['twitter', 'tweet', 'thread', 'conversation', 'trending', 'hashtag'],
                'query_patterns': ['trending', 'opinion', 'quick tip', 'news', 'update', 'thread'],
                'content_type': 'conversation_participation',
                'engagement_style': 'threads_and_replies'
            },
            'GOOGLE': {
                'keywords': ['google', 'search', 'seo', 'organic', 'ranking', 'keyword'],
                'query_patterns': ['what is', 'how to', 'guide', 'tutorial', 'definition', 'explain'],
                'content_type': 'seo_optimization',
                'engagement_style': 'comprehensive_articles'
            },
            'YELP': {
                'keywords': ['yelp', 'review', 'business', 'service', 'rating', 'feedback'],
                'query_patterns': ['review', 'rating', 'service', 'business', 'experience', 'recommend'],
                'content_type': 'review_responses',
                'engagement_style': 'professional_engagement'
            },
            'QUORA': {
                'keywords': ['quora', 'question', 'answer', 'expert', 'knowledge', 'q&a'],
                'query_patterns': ['what is', 'how does', 'why', 'explain', 'difference', 'compare'],
                'content_type': 'expert_answers',
                'engagement_style': 'detailed_responses'
            },
            'LINKEDIN': {
                'keywords': ['linkedin', 'professional', 'industry', 'networking', 'career', 'business'],
                'query_patterns': ['industry', 'career', 'professional', 'business', 'strategy', 'trends'],
                'content_type': 'professional_networking',
                'engagement_style': 'industry_insights'
            }
        }
        
        self.content_gap_patterns = {
            'missing': ['missing', 'lack', 'not found', 'unavailable', 'need', 'want'],
            'expand': ['expand', 'more', 'additional', 'extend', 'comprehensive', 'detailed'],
            'improve': ['improve', 'better', 'enhance', 'optimize', 'upgrade', 'refine'],
            'clarify': ['clarify', 'explain', 'define', 'understand', 'confusing', 'unclear']
        }
    
    def score_chunks_against_queries(self, chunks: List[Dict], fanout_queries: List[Dict]) -> Dict[str, Any]:
        """
        Score content chunks against fan-out queries and provide comprehensive analysis.
        
        Args:
            chunks: List of content chunks
            fanout_queries: List of fan-out queries with metadata
            
        Returns:
            Comprehensive scoring results with optimizations and channel strategies
        """
        logger.info(f"🎯 Starting comprehensive scoring of {len(chunks)} chunks against {len(fanout_queries)} queries")
        
        results = {
            'chunk_scores': [],
            'query_analysis': {},
            'content_gaps': {},
            'channel_strategy': {},
            'optimization_recommendations': [],
            'summary': {}
        }
        
        # Debug: Log chunk structure
        if chunks:
            logger.info(f"First chunk keys: {list(chunks[0].keys())}")
            logger.info(f"First chunk text preview: {chunks[0].get('text', '')[:100]}...")
        
        # Get top 10 fan-out queries
        top_queries = sorted(fanout_queries, key=lambda x: x.get('best_score', 0), reverse=True)[:10]
        query_texts = [q.get('query', '') for q in top_queries]
        
        logger.info(f"Top queries for analysis: {query_texts[:3]}...")
        
        # Score each chunk against top queries
        for i, chunk in enumerate(chunks):
            try:
                chunk_score = self._score_single_chunk(chunk, query_texts)
                results['chunk_scores'].append(chunk_score)
                logger.info(f"Scored chunk {i+1}/{len(chunks)}: {chunk_score['avg_relevance_score']:.3f}")
            except Exception as e:
                logger.error(f"Error scoring chunk {i}: {e}")
                # Add a default score for failed chunks
                results['chunk_scores'].append({
                    'chunk_id': chunk.get('id', f'chunk_{i}'),
                    'url': chunk.get('url', ''),
                    'text_preview': chunk.get('text', '')[:200] + "..." if chunk.get('text') else "No text",
                    'query_matches': [],
                    'avg_relevance_score': 0.0,
                    'max_relevance_score': 0.0,
                    'overall_grade': 'F',
                    'priority_level': 'high'
                })
        
        # Analyze content gaps
        results['content_gaps'] = self._analyze_content_gaps(results['chunk_scores'])
        
        # Generate channel strategy
        results['channel_strategy'] = self._generate_channel_strategy(query_texts, results['chunk_scores'])
        
        # Generate optimization recommendations
        results['optimization_recommendations'] = self._generate_optimization_recommendations(
            results['chunk_scores'], results['content_gaps'], results['channel_strategy']
        )
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        logger.info(f"✅ Comprehensive scoring completed: {len(results['chunk_scores'])} chunks scored")
        return results
    
    def _score_single_chunk(self, chunk: Dict, query_texts: List[str]) -> Dict[str, Any]:
        """Score a single chunk against the top queries."""
        chunk_text = chunk.get('text', '').lower()
        chunk_id = chunk.get('id', '')
        url = chunk.get('url', '')
        
        scores = []
        query_matches = []
        
        for query in query_texts:
            query_lower = query.lower()
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(chunk_text, query_lower)
            
            # Check for content gaps
            gaps = self._identify_content_gaps(chunk_text, query_lower)
            
            # Determine best channels
            channels = self._identify_best_channels(query_lower)
            
            query_match = {
                'query': query,
                'relevance_score': relevance_score,
                'content_gaps': gaps,
                'recommended_channels': channels,
                'optimization_suggestions': self._generate_chunk_optimizations(chunk_text, query_lower, gaps)
            }
            
            query_matches.append(query_match)
            scores.append(relevance_score)
        
        # Calculate overall chunk score
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        return {
            'chunk_id': chunk_id,
            'url': url,
            'text_preview': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
            'query_matches': query_matches,
            'avg_relevance_score': avg_score,
            'max_relevance_score': max_score,
            'overall_grade': self._calculate_grade(avg_score),
            'priority_level': self._calculate_priority_level(avg_score)
        }
    
    def _calculate_relevance_score(self, chunk_text: str, query: str) -> float:
        """Calculate relevance score between chunk and query."""
        if not chunk_text or not query:
            return 0.0
            
        # Simple keyword matching for now (can be enhanced with semantic similarity)
        query_words = set(query.lower().split())
        chunk_words = set(chunk_text.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(chunk_words))
        total_query_words = len(query_words)
        
        if total_query_words == 0:
            return 0.0
        
        # Base score from word overlap (weighted by importance)
        base_score = (overlap / total_query_words) * 0.6
        
        # Bonus for exact phrase matches
        phrase_bonus = 0.3 if query.lower() in chunk_text.lower() else 0.0
        
        # Bonus for partial phrase matches
        partial_bonus = 0.0
        query_parts = query.lower().split()
        if len(query_parts) > 1:
            # Check for consecutive word matches
            for i in range(len(query_parts) - 1):
                phrase = f"{query_parts[i]} {query_parts[i+1]}"
                if phrase in chunk_text.lower():
                    partial_bonus += 0.1
        
        # Bonus for semantic similarity (simplified)
        semantic_bonus = 0.1 if any(word in chunk_text.lower() for word in query_words if len(word) > 3) else 0.0
        
        final_score = min(1.0, base_score + phrase_bonus + partial_bonus + semantic_bonus)
        return round(final_score, 3)
    
    def _identify_content_gaps(self, chunk_text: str, query: str) -> List[str]:
        """Identify content gaps in the chunk for the given query."""
        gaps = []
        
        for gap_type, patterns in self.content_gap_patterns.items():
            for pattern in patterns:
                if pattern in query.lower() and pattern not in chunk_text.lower():
                    gaps.append(f"Missing {gap_type} content")
                    break
        
        return gaps
    
    def _identify_best_channels(self, query: str) -> List[str]:
        """Identify best channels for the given query."""
        channel_scores = {}
        
        for platform, config in self.supported_platforms.items():
            score = 0
            
            # Check for platform-specific keywords
            for keyword in config['keywords']:
                if keyword in query.lower():
                    score += 1
            
            # Check for platform-specific query patterns
            for pattern in config['query_patterns']:
                if pattern in query.lower():
                    score += 2
            
            if score > 0:
                channel_scores[platform] = score
        
        # Return top 3 channels
        sorted_channels = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)
        return [channel for channel, score in sorted_channels[:3]]
    
    def _generate_chunk_optimizations(self, chunk_text: str, query: str, gaps: List[str]) -> List[str]:
        """Generate optimization suggestions for a chunk."""
        optimizations = []
        
        if gaps:
            for gap in gaps:
                if "Missing" in gap:
                    optimizations.append(f"Add {gap.lower().replace('missing ', '')} to address query: '{query}'")
        
        # Check for content length
        if len(chunk_text.split()) < 50:
            optimizations.append("Expand content with more detailed information")
        
        # Check for keyword density
        query_words = query.lower().split()
        chunk_lower = chunk_text.lower()
        for word in query_words:
            if word not in chunk_lower and len(word) > 3:
                optimizations.append(f"Include keyword '{word}' naturally in content")
        
        return optimizations
    
    def _analyze_content_gaps(self, chunk_scores: List[Dict]) -> Dict[str, Any]:
        """Analyze common content gaps across all chunks."""
        all_gaps = []
        for chunk_score in chunk_scores:
            for query_match in chunk_score['query_matches']:
                all_gaps.extend(query_match['content_gaps'])
        
        gap_counts = Counter(all_gaps)
        
        return {
            'total_gaps': len(all_gaps),
            'gap_distribution': dict(gap_counts),
            'most_common_gaps': gap_counts.most_common(5)
        }
    
    def _generate_channel_strategy(self, query_texts: List[str], chunk_scores: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive channel strategy."""
        channel_queries = defaultdict(list)
        channel_scores = defaultdict(int)
        
        # Analyze each query for channel fit
        for query in query_texts:
            for platform, config in self.supported_platforms.items():
                score = 0
                
                # Check keywords and patterns
                for keyword in config['keywords']:
                    if keyword in query.lower():
                        score += 1
                
                for pattern in config['query_patterns']:
                    if pattern in query.lower():
                        score += 2
                
                if score > 0:
                    channel_queries[platform].append(query)
                    channel_scores[platform] += score
        
        # Calculate channel priorities
        channel_priorities = []
        for platform, score in channel_scores.items():
            query_count = len(channel_queries[platform])
            priority = {
                'platform': platform,
                'score': score,
                'query_count': query_count,
                'focus_level': 'high' if score > 10 else 'medium' if score > 5 else 'low',
                'top_queries': channel_queries[platform][:5]
            }
            channel_priorities.append(priority)
        
        # Sort by score
        channel_priorities.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'total_queries_analyzed': len(query_texts),
            'channel_distribution': {p['platform']: p['query_count'] for p in channel_priorities},
            'channel_priorities': channel_priorities,
            'top_platforms': [p['platform'] for p in channel_priorities[:3]]
        }
    
    def _generate_optimization_recommendations(self, chunk_scores: List[Dict], 
                                            content_gaps: Dict, 
                                            channel_strategy: Dict) -> List[Dict]:
        """Generate comprehensive optimization recommendations."""
        recommendations = []
        
        # Content optimization recommendations
        low_scoring_chunks = [c for c in chunk_scores if c['avg_relevance_score'] < 0.3]
        if low_scoring_chunks:
            recommendations.append({
                'type': 'content_optimization',
                'priority': 'high',
                'title': 'Optimize Low-Scoring Content',
                'description': f"Improve {len(low_scoring_chunks)} chunks with scores below 0.3",
                'action_items': [
                    "Add missing keywords naturally",
                    "Expand content with detailed information",
                    "Address identified content gaps"
                ]
            })
        
        # Channel-specific recommendations
        for platform_info in channel_strategy['channel_priorities'][:3]:
            if platform_info['focus_level'] in ['high', 'medium']:
                recommendations.append({
                    'type': 'channel_strategy',
                    'priority': platform_info['focus_level'],
                    'title': f"Focus on {platform_info['platform']}",
                    'description': f"Optimize for {platform_info['query_count']} queries on {platform_info['platform']}",
                    'action_items': [
                        f"Create {platform_info['platform'].lower()}-specific content",
                        f"Engage with {platform_info['platform'].lower()} community",
                        f"Monitor {platform_info['platform'].lower()} performance"
                    ]
                })
        
        return recommendations
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade based on score."""
        if score >= 0.8:
            return 'A'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.4:
            return 'C'
        elif score >= 0.2:
            return 'D'
        else:
            return 'F'
    
    def _calculate_priority_level(self, score: float) -> str:
        """Calculate priority level based on score."""
        if score >= 0.7:
            return 'low'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'high'
    
    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary statistics."""
        chunk_scores = results['chunk_scores']
        
        return {
            'chunks_optimized': len(chunk_scores),
            'queries_analyzed': results['channel_strategy'].get('total_queries_analyzed', 0),
            'avg_chunk_score': sum(c['avg_relevance_score'] for c in chunk_scores) / len(chunk_scores) if chunk_scores else 0,
            'top_content_gaps': results['content_gaps'].get('most_common_gaps', []),
            'top_channels': results['channel_strategy'].get('top_platforms', []),
            'optimization_count': len(results['optimization_recommendations'])
        }
    
    def format_xai_output(self, results: Dict) -> str:
        """Format results in XAI output style."""
        summary = results['summary']
        
        output = "=== XAI Optimization Results ===\n"
        output += f"Chunks optimized: {summary['chunks_optimized']}\n"
        output += f"Queries analyzed: {summary['queries_analyzed']}\n\n"
        
        output += "Top Content Gaps:\n"
        for gap, count in summary['top_content_gaps']:
            output += f"  • {gap} (mentioned in {count} chunks)\n"
        
        output += "\n=== Channel Strategy ===\n"
        output += f"Total queries analyzed: {summary['queries_analyzed']}\n\n"
        
        channel_strategy = results['channel_strategy']
        output += "Channel Distribution:\n"
        for platform, count in channel_strategy['channel_distribution'].items():
            output += f"  {platform}: {count} queries\n"
        
        output += "\nTop Implementation Priorities:\n"
        for i, platform_info in enumerate(channel_strategy['channel_priorities'][:3], 1):
            output += f"  {i}. {platform_info['platform']} (Score: {platform_info['score']}, Focus: {platform_info['focus_level']})\n"
        
        return output 