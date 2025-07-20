"""
Query Generator - Reverse engineering queries from content chunks.
Works backwards to answer "what queries does this content answer?"
"""
import os
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from datetime import datetime

from .utils import (
    get_env_var, retry_on_failure, create_data_dir, save_jsonl, 
    load_jsonl, logger, sanitize_text
)


class ReverseQueryGenerator:
    """Generate queries that content chunks answer by working backwards."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        api_key = get_env_var("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def generate_queries_for_chunk(self, chunk: Dict[str, Any], 
                                 query_types: List[str] = None) -> Dict[str, Any]:
        """Generate queries that a content chunk answers."""
        if query_types is None:
            query_types = ['direct', 'related', 'long_tail', 'questions', 'intent_based']
        
        content = chunk.get('content', '')
        url = chunk.get('url', '')
        title = chunk.get('title', '')
        
        if not content:
            logger.warning(f"No content found in chunk from {url}")
            return {
                'chunk_id': chunk.get('id'),
                'url': url,
                'queries': [],
                'query_count': 0,
                'generated_at': datetime.now().isoformat()
            }
        
        # Prepare context for AI
        context = f"""
        Content Title: {title}
        Content URL: {url}
        Content Text: {content[:2000]}  # Limit for API
        
        Task: Generate queries that this content answers. Work backwards from the content to identify what questions/problems this content solves.
        
        Generate queries in these categories:
        1. DIRECT: Exact queries this content directly answers
        2. RELATED: Related queries that would find this content relevant
        3. LONG_TAIL: Specific, detailed queries this content addresses
        4. QUESTIONS: Question-form queries this content answers
        5. INTENT_BASED: Queries based on user intent (informational, navigational, transactional)
        
        For each query, provide:
        - query_text: The actual search query
        - category: Which type of query
        - relevance_score: How directly this content answers (1-10)
        - intent: What the user is trying to accomplish
        
        Return as JSON array of query objects.
        """
        
        try:
            response = self.model_instance.generate_content(context)
            queries_data = self._parse_ai_response(response.text)
            
            # Add metadata
            result = {
                'chunk_id': chunk.get('id'),
                'url': url,
                'title': title,
                'queries': queries_data,
                'query_count': len(queries_data),
                'generated_at': datetime.now().isoformat(),
                'model_used': self.model
            }
            
            logger.info(f"Generated {len(queries_data)} queries for chunk from {url}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating queries for chunk: {e}")
            return {
                'chunk_id': chunk.get('id'),
                'url': url,
                'queries': [],
                'query_count': 0,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def _parse_ai_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured query data."""
        try:
            # Try to extract JSON from response
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_str = response_text[json_start:json_end].strip()
            elif '[' in response_text and ']' in response_text:
                # Try to find JSON array in response
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                json_str = response_text[start:end]
            else:
                # Fallback: try to parse the whole response
                json_str = response_text
            
            queries = json.loads(json_str)
            
            # Validate and clean queries
            cleaned_queries = []
            for query in queries:
                if isinstance(query, dict) and 'query_text' in query:
                    cleaned_query = {
                        'query_text': query.get('query_text', '').strip(),
                        'category': query.get('category', 'general'),
                        'relevance_score': min(10, max(1, query.get('relevance_score', 5))),
                        'intent': query.get('intent', 'informational'),
                        'confidence': query.get('confidence', 0.8)
                    }
                    if cleaned_query['query_text']:
                        cleaned_queries.append(cleaned_query)
            
            return cleaned_queries
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            # Fallback: extract queries manually
            return self._extract_queries_manually(response_text)
    
    def _extract_queries_manually(self, response_text: str) -> List[Dict[str, Any]]:
        """Manual extraction of queries from AI response."""
        queries = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # Simple heuristic: lines that look like queries
                if any(word in line.lower() for word in ['how', 'what', 'why', 'when', 'where', 'best', 'guide', 'tips']):
                    queries.append({
                        'query_text': line,
                        'category': 'extracted',
                        'relevance_score': 7,
                        'intent': 'informational',
                        'confidence': 0.6
                    })
        
        return queries
    
    def generate_queries_for_all_chunks(self, chunks: List[Dict[str, Any]], 
                                      output_file: str = "generated_queries.jsonl") -> Dict[str, Any]:
        """Generate queries for all content chunks."""
        create_data_dir()
        
        all_results = []
        total_queries = 0
        
        logger.info(f"Generating queries for {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} from {chunk.get('url', 'unknown')}")
            
            result = self.generate_queries_for_chunk(chunk)
            all_results.append(result)
            total_queries += result['query_count']
            
            # Save progress incrementally
            if (i + 1) % 10 == 0:
                save_jsonl(all_results, output_file)
                logger.info(f"Saved progress: {i+1} chunks processed, {total_queries} queries generated")
        
        # Final save
        save_jsonl(all_results, output_file)
        
        summary = {
            'total_chunks_processed': len(chunks),
            'total_queries_generated': total_queries,
            'average_queries_per_chunk': total_queries / len(chunks) if chunks else 0,
            'output_file': output_file,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Query generation complete: {summary}")
        return summary
    
    def analyze_query_coverage(self, generated_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the coverage and distribution of generated queries."""
        if not generated_queries:
            return {'error': 'No generated queries to analyze'}
        
        # Flatten all queries
        all_queries = []
        for chunk_result in generated_queries:
            all_queries.extend(chunk_result.get('queries', []))
        
        if not all_queries:
            return {'error': 'No queries found in results'}
        
        # Analyze categories
        categories = {}
        intents = {}
        relevance_scores = []
        
        for query in all_queries:
            category = query.get('category', 'unknown')
            intent = query.get('intent', 'unknown')
            score = query.get('relevance_score', 5)
            
            categories[category] = categories.get(category, 0) + 1
            intents[intent] = intents.get(intent, 0) + 1
            relevance_scores.append(score)
        
        # Calculate statistics
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        analysis = {
            'total_queries': len(all_queries),
            'unique_chunks': len(generated_queries),
            'category_distribution': categories,
            'intent_distribution': intents,
            'average_relevance_score': avg_relevance,
            'top_categories': sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_intents': sorted(intents.items(), key=lambda x: x[1], reverse=True)[:5],
            'relevance_score_range': {
                'min': min(relevance_scores),
                'max': max(relevance_scores),
                'avg': avg_relevance
            }
        }
        
        return analysis


class QueryAnalyzer:
    """Analyze and optimize generated queries."""
    
    def __init__(self):
        self.query_generator = ReverseQueryGenerator()
    
    def find_content_gaps(self, generated_queries: List[Dict[str, Any]], 
                         target_queries: List[str] = None) -> Dict[str, Any]:
        """Find content gaps by comparing generated queries with target queries."""
        # Extract all generated query texts
        generated_texts = set()
        for chunk_result in generated_queries:
            for query in chunk_result.get('queries', []):
                generated_texts.add(query.get('query_text', '').lower())
        
        gaps = []
        if target_queries:
            for target in target_queries:
                target_lower = target.lower()
                if not any(target_lower in gen for gen in generated_texts):
                    gaps.append({
                        'target_query': target,
                        'gap_type': 'missing_coverage',
                        'priority': 'high'
                    })
        
        return {
            'total_generated_queries': len(generated_texts),
            'target_queries_analyzed': len(target_queries) if target_queries else 0,
            'content_gaps': gaps,
            'gap_count': len(gaps)
        }
    
    def optimize_content_strategy(self, generated_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provide content optimization recommendations based on query analysis."""
        analysis = self.query_generator.analyze_query_coverage(generated_queries)
        
        recommendations = []
        
        # Analyze category distribution
        categories = analysis.get('category_distribution', {})
        if categories:
            top_category = max(categories.items(), key=lambda x: x[1])
            if top_category[1] > len(generated_queries) * 0.4:  # If one category dominates
                recommendations.append({
                    'type': 'category_balance',
                    'message': f"Content is heavily focused on '{top_category[0]}' queries. Consider diversifying into other categories.",
                    'priority': 'medium'
                })
        
        # Analyze relevance scores
        avg_relevance = analysis.get('average_relevance_score', 5)
        if avg_relevance < 6:
            recommendations.append({
                'type': 'relevance_improvement',
                'message': f"Average relevance score is {avg_relevance:.1f}/10. Consider making content more directly answer-focused.",
                'priority': 'high'
            })
        
        # Analyze intent distribution
        intents = analysis.get('intent_distribution', {})
        if 'transactional' not in intents or intents.get('transactional', 0) < 5:
            recommendations.append({
                'type': 'intent_coverage',
                'message': "Limited transactional intent coverage. Consider adding more action-oriented content.",
                'priority': 'medium'
            })
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'recommendation_count': len(recommendations)
        } 