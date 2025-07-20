from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

st.set_page_config(
    page_title="Zero-Click Compass 🧭",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ... other imports ...

# Now define your functions and classes, including generate_gap_filling_recommendations
def generate_gap_filling_recommendations(self, high_relevance_gaps: List[Dict]) -> List[Dict]:
    """Generate specific content recommendations to fill identified gaps"""
    try:
        self.logger.info(f"Generating gap-filling recommendations for {len(high_relevance_gaps)} gaps")
        
        # Prepare gap summary for AI
        gap_summary = []
        for gap in high_relevance_gaps[:10]:  # Limit to top 10 for API efficiency
            gap_summary.append({
                'query': gap['query'],
                'current_score': gap['best_score'],
                'gap_type': 'Missing' if gap['best_score'] < 0.3 else 'Weak',
                'matched_content': gap.get('best_chunk_preview', 'None')
            })
        
        prompt = f"""Analyze these content gaps and generate specific recommendations to fill them:

{json.dumps(gap_summary, indent=2)}

For each gap, provide:
1. Content type (new page, section, FAQ, guide, etc.)
2. Recommended word count
3. Key points to cover
4. SEO optimization tips
5. Internal linking opportunities
6. Example opening paragraph

Format as JSON array with detailed recommendations."""

        response = self.chat_model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean markdown if present
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        recommendations = json.loads(json_text.strip())
        self.logger.info(f"Generated {len(recommendations)} gap-filling recommendations")
        return recommendations
        
    except Exception as e:
        self.logger.error(f"Gap recommendation generation failed: {str(e)}")
        return [{
            'query': gap['query'],
            'content_type': 'Error generating recommendation',
            'word_count': 0,
            'key_points': [],
            'seo_tips': 'Error',
            'linking': 'Error',
            'example_opening': 'Error generating content'
        } for gap in high_relevance_gaps[:5]]            # Store embedding provider for passage generation
"""
Zero-Click Compass MVP - Streamlit App
A modular implementation for real website analysis
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
from collections import Counter
import re
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================== CORE MODULES ==================

class WebCrawler:
    """Handles website crawling and content extraction"""
    
    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages
        self.visited_urls = set()
        self.logger = logging.getLogger(f"{__name__}.WebCrawler")
        
    def crawl(self, start_url: str) -> Dict[str, Any]:
        """Crawl website starting from the given URL"""
        self.logger.info(f"Starting crawl of {start_url} (max pages: {self.max_pages})")
        pages = []
        to_visit = [start_url]
        base_domain = urlparse(start_url).netloc
        
        while to_visit and len(pages) < self.max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                self.logger.debug(f"Skipping already visited URL: {url}")
                continue
                
            try:
                self.logger.info(f"Crawling page {len(pages) + 1}/{self.max_pages}: {url}")
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract text content
                    text_content = self._extract_text(soup)
                    word_count = len(text_content.split())
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.text.strip() if title else urlparse(url).path
                    
                    self.logger.info(f"Extracted {word_count} words from {url}")
                    
                    pages.append({
                        'url': url,
                        'title': title_text,
                        'content': text_content,
                        'word_count': word_count
                    })
                    
                    self.visited_urls.add(url)
                    
                    # Find more URLs on the same domain
                    links_found = 0
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if urlparse(next_url).netloc == base_domain and next_url not in self.visited_urls:
                            to_visit.append(next_url)
                            links_found += 1
                    
                    self.logger.debug(f"Found {links_found} new links on {url}")
                else:
                    self.logger.warning(f"Failed to crawl {url}: HTTP {response.status_code}")
                            
            except Exception as e:
                self.logger.error(f"Error crawling {url}: {str(e)}")
                st.warning(f"Failed to crawl {url}: {str(e)}")
        
        self.logger.info(f"Crawl complete. Extracted {len(pages)} pages with {sum(p['word_count'] for p in pages)} total words")
        
        return {
            'pages': pages,
            'total_pages': len(pages),
            'domain': base_domain
        }
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text


class ContentChunker:
    """Handles content chunking for LLM processing"""
    
    def __init__(self, chunk_size: int = 150):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(f"{__name__}.ContentChunker")
        
    def chunk_content(self, pages: List[Dict]) -> List[Dict]:
        """Split content into semantic chunks"""
        self.logger.info(f"Starting content chunking for {len(pages)} pages (chunk size: {self.chunk_size} words)")
        chunks = []
        
        for page_idx, page in enumerate(pages):
            content = page['content']
            words = content.split()
            page_chunks = 0
            
            # Create chunks based on word count (approximating tokens)
            for i in range(0, len(words), self.chunk_size):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunk_id = hashlib.md5(f"{page['url']}_{i}".encode()).hexdigest()[:8]
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'page_url': page['url'],
                    'page_title': page['title'],
                    'content': chunk_text,
                    'word_count': len(chunk_words),
                    'position': i // self.chunk_size
                })
                page_chunks += 1
            
            self.logger.debug(f"Page {page_idx + 1}: Created {page_chunks} chunks from {len(words)} words")
        
        self.logger.info(f"Chunking complete. Created {len(chunks)} total chunks")
        return chunks


class GeminiEmbeddingProvider:
    """Handles Gemini API for embeddings and LLM operations"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"
        self.chat_model = genai.GenerativeModel('gemini-1.5-flash')
        self.logger = logging.getLogger(f"{__name__}.GeminiEmbeddingProvider")
        self.logger.info("Initialized Gemini embedding provider")
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini"""
        try:
            self.logger.debug(f"Generating embedding for text ({len(text)} chars)")
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            self.logger.debug("Embedding generated successfully")
            return result['embedding']
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            st.warning(f"Embedding generation failed: {str(e)}")
            return None
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query using Gemini"""
        try:
            self.logger.debug(f"Generating query embedding for: {query}")
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            self.logger.debug("Query embedding generated successfully")
            return result['embedding']
        except Exception as e:
            self.logger.error(f"Query embedding generation failed: {str(e)}")
            st.warning(f"Query embedding generation failed: {str(e)}")
            return None
    
    def generate_reverse_queries(self, chunk: Dict) -> List[str]:
        """Generate queries that a chunk of content could answer (max 5)"""
        try:
            self.logger.info(f"Generating reverse queries for chunk {chunk['chunk_id']}")
            prompt = f"""Analyze this content chunk and generate up to 5 queries that users might ask which this content would answer.

Content:
{chunk['content']}

Generate ONLY the most relevant queries this content directly answers.
Return one query per line, no numbering or bullets.
Maximum 5 queries."""

            response = self.chat_model.generate_content(prompt)
            queries = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            self.logger.info(f"Generated {len(queries)} reverse queries for chunk")
            return queries[:5]  # Ensure max 5
        except Exception as e:
            self.logger.error(f"Reverse query generation failed: {str(e)}")
            return []
    
    def select_top_reverse_queries(self, all_reverse_queries: Dict[str, List[str]], 
                                  main_query: str, chunks: List[Dict]) -> List[str]:
        """Select top 5 most relevant reverse queries based on relevance to main query"""
        try:
            self.logger.info(f"Selecting top reverse queries from {sum(len(q) for q in all_reverse_queries.values())} total")
            
            # Flatten all queries with their source chunks
            query_candidates = []
            for chunk_id, queries in all_reverse_queries.items():
                for query in queries:
                    query_candidates.append({
                        'query': query,
                        'chunk_id': chunk_id
                    })
            
            if not query_candidates:
                return []
            
            # Score each reverse query against the main query
            prompt = f"""Given the main query: "{main_query}"

Select the 5 most relevant queries from this list that best relate to the main query:

{json.dumps([q['query'] for q in query_candidates], indent=2)}

Return ONLY the 5 selected queries, one per line, in order of relevance."""

            response = self.chat_model.generate_content(prompt)
            selected = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            self.logger.info(f"Selected {len(selected)} top reverse queries")
            return selected[:5]
        except Exception as e:
            self.logger.error(f"Query selection failed: {str(e)}")
            # Fallback: return first 5 queries
            return [q['query'] for q in query_candidates[:5]]
    
    def generate_query_fanout_batch(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """Generate fan-out for multiple queries (max 15 expansions each)"""
        fanout_results = {}
        
        for query in queries:
            try:
                self.logger.info(f"Generating fan-out for reverse query: {query}")
                prompt = f"""Generate exactly 15 query expansions for: "{query}"

Include these types:
- Reformulations (3)
- Related Queries (3)
- Implicit Queries (2)
- Comparative Queries (2)
- Entity Expansions (3)
- Personalized Queries (2)

Return as JSON array with format:
[
  {{
    "query": "expanded query",
    "type": "query type",
    "relevance": "high/medium/low"
  }}
]"""

                response = self.chat_model.generate_content(prompt)
                json_text = response.text.strip()
                
                # Clean markdown if present
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                
                expansions = json.loads(json_text.strip())
                fanout_results[query] = expansions[:15]  # Ensure max 15
                self.logger.info(f"Generated {len(expansions)} expansions for query")
                
            except Exception as e:
                self.logger.error(f"Fan-out generation failed for {query}: {str(e)}")
                fanout_results[query] = []
        
        return fanout_results
    
    def generate_xai_optimization(self, content_scores: Dict, main_query: str) -> Dict[str, Any]:
        """Generate XAI-based content optimization recommendations"""
        try:
            self.logger.info("Generating XAI content optimization")
            
            # Analyze poorly performing queries
            poor_queries = [q for q, score in content_scores.items() if score < 0.4]
            
            prompt = f"""Based on content analysis for "{main_query}", provide XAI optimization:

Poorly covered queries ({len(poor_queries)} total):
{json.dumps(poor_queries[:10], indent=2)}

Generate:
1. Specific content improvements with explanations
2. Keyword optimization strategies
3. Semantic coverage recommendations
4. Structure enhancements for AI comprehension

Format as actionable JSON."""

            response = self.chat_model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            self.logger.error(f"XAI optimization failed: {str(e)}")
            return {
                "content_improvements": ["Error generating recommendations"],
                "keyword_strategies": [],
                "semantic_recommendations": [],
                "structure_enhancements": []
            }
    
    def generate_channel_analysis(self, main_query: str, content_type: str) -> Dict[str, Any]:
        """Generate platform-specific channel strategies"""
        try:
            self.logger.info("Generating channel analysis")
            
            prompt = f"""For the query "{main_query}" and content type "{content_type}", provide channel strategies:

Generate platform-specific recommendations for:
1. Reddit - Subreddit targeting and engagement tactics
2. X (Twitter) - Hashtag strategy and thread optimization
3. Google - Featured snippet optimization
4. Yelp - Local SEO and review integration
5. Quora - Question targeting and answer optimization

Include specific tactics, posting schedules, and engagement strategies."""

            response = self.chat_model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            self.logger.error(f"Channel analysis failed: {str(e)}")
            return {
                "reddit": {"subreddits": [], "tactics": "Error generating"},
                "twitter": {"hashtags": [], "strategy": "Error generating"},
                "google": {"optimization": "Error generating"},
                "yelp": {"local_seo": "Error generating"},
                "quora": {"questions": [], "approach": "Error generating"}
            }
    
    def generate_brand_curation(self, industry: str, competitors: List[str]) -> Dict[str, Any]:
        """Generate industry-specific brand curation strategies"""
        try:
            self.logger.info("Generating brand curation strategies")
            
            prompt = f"""For industry "{industry}" with competitors {competitors}, provide brand curation:

Generate:
1. Brand positioning recommendations
2. Content differentiation strategies
3. Competitive advantages to highlight
4. Industry-specific messaging guidelines
5. Future agentic engagement foundations

Format as strategic JSON recommendations."""

            response = self.chat_model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            self.logger.error(f"Brand curation failed: {str(e)}")
            return {
                "positioning": "Error generating",
                "differentiation": [],
                "competitive_advantages": [],
                "messaging_guidelines": [],
                "agentic_foundations": []
            }
    
    def extract_queries_from_content(self, chunks: List[Dict]) -> List[str]:
        """Extract all queries that the content answers"""
        try:
            self.logger.info(f"Extracting queries from {len(chunks)} chunks")
            # Combine content from top chunks
            content_text = "\n\n".join([chunk['content'] for chunk in chunks[:20]])
            
            prompt = f"""Analyze this content and extract ALL queries/questions that this content answers or addresses.

Content:
{content_text}

Generate a comprehensive list of queries that users might ask which would be answered by this content.
Include:
- Direct questions the content answers
- Implied questions based on the information provided
- Related queries that the content partially addresses
- Different phrasings of the same question

Return one query per line, no numbering or bullets. Aim for at least 10-20 queries."""

            response = self.chat_model.generate_content(prompt)
            queries = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            self.logger.info(f"Extracted {len(queries)} queries from content")
            return queries
        except Exception as e:
            self.logger.error(f"Query extraction failed: {str(e)}")
            st.warning(f"Query extraction failed: {str(e)}")
            return []
    
    def generate_query_fanout(self, query: str, mode: str = "AI Mode (complex)") -> Dict[str, Any]:
        """Generate fan-out queries using Qforia approach"""
        try:
            self.logger.info(f"Generating fan-out for query: {query} (mode: {mode})")
            min_queries = 10 if mode == "AI Overview (simple)" else 20
            
            prompt = f"""You are simulating Google's AI Mode query fan-out process for generative search systems.
The user's original query is: "{query}". The selected mode is: "{mode}".

First, analyze the query complexity and decide on the optimal number of queries to generate (minimum {min_queries}).
Then generate exactly that many unique synthetic queries.

Each of these query transformation types MUST be represented:
1. Reformulations - Different ways to ask the same thing
2. Related Queries - Adjacent topics and follow-ups
3. Implicit Queries - Unstated but implied questions
4. Comparative Queries - Comparisons with alternatives
5. Entity Expansions - Specific brands, models, features
6. Personalized Queries - Use cases and scenarios

Return only valid JSON:
{{
  "generation_details": {{
    "target_query_count": [number],
    "reasoning_for_count": "[reasoning]"
  }},
  "expanded_queries": [
    {{
      "query": "[expanded query]",
      "type": "[query type]",
      "user_intent": "[intent description]",
      "reasoning": "[why this query was generated]"
    }}
  ]
}}"""

            response = self.chat_model.generate_content(prompt)
            json_text = response.text.strip()
            
            # Clean markdown if present
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            result = json.loads(json_text.strip())
            self.logger.info(f"Generated {len(result.get('expanded_queries', []))} fan-out queries")
            return result
        except Exception as e:
            self.logger.error(f"Query fan-out failed: {str(e)}")
            st.warning(f"Query fan-out failed: {str(e)}")
            return {"generation_details": {}, "expanded_queries": []}
    
    def score_content_for_queries(self, chunks: List[Dict], queries: List[str]) -> Dict[str, Any]:
        """Score how well content addresses each query"""
        self.logger.info(f"Scoring content against {len(queries)} queries")
        query_scores = {}
        
        for idx, query in enumerate(queries):
            self.logger.debug(f"Scoring query {idx + 1}/{len(queries)}: {query}")
            # Get query embedding
            query_embedding = self.generate_query_embedding(query)
            if not query_embedding:
                continue
                
            # Score against all chunks
            chunk_scores = []
            for chunk_idx, chunk in enumerate(chunks):
                if chunk_idx % 10 == 0:
                    self.logger.debug(f"Processing chunk {chunk_idx}/{len(chunks)} for query {idx + 1}")
                    
                chunk_embedding = self.generate_embedding(chunk['content'])
                if chunk_embedding:
                    similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                    chunk_scores.append({
                        'chunk_id': chunk['chunk_id'],
                        'score': max(0, similarity),
                        'content_preview': chunk['content'][:100] + "..."
                    })
            
            # Get top scoring chunks
            chunk_scores.sort(key=lambda x: x['score'], reverse=True)
            avg_score = np.mean([s['score'] for s in chunk_scores[:5]]) if chunk_scores else 0
            
            query_scores[query] = {
                'avg_score': avg_score,
                'top_chunks': chunk_scores[:3],
                'coverage': 'good' if avg_score > 0.7 else 'partial' if avg_score > 0.4 else 'poor'
            }
            
            self.logger.debug(f"Query '{query}' - Average score: {avg_score:.3f}, Coverage: {query_scores[query]['coverage']}")
        
        self.logger.info(f"Scoring complete. Coverage - Good: {sum(1 for q in query_scores.values() if q['coverage'] == 'good')}, "
                        f"Partial: {sum(1 for q in query_scores.values() if q['coverage'] == 'partial')}, "
                        f"Poor: {sum(1 for q in query_scores.values() if q['coverage'] == 'poor')}")
        
        return query_scores
    
    def generate_content_recommendations(self, query_scores: Dict, fanout_results: Dict) -> Dict[str, Any]:
        """Generate specific content optimization recommendations"""
        try:
            # Analyze coverage gaps
            poor_queries = [q for q, data in query_scores.items() if data['coverage'] == 'poor']
            partial_queries = [q for q, data in query_scores.items() if data['coverage'] == 'partial']
            
            prompt = f"""Based on this content analysis, provide specific optimization recommendations:

Poorly covered queries ({len(poor_queries)} total):
{json.dumps(poor_queries[:5], indent=2)}

Partially covered queries ({len(partial_queries)} total):
{json.dumps(partial_queries[:5], indent=2)}

Fan-out query types that need attention:
{json.dumps([q['type'] for q in fanout_results.get('expanded_queries', [])[:10]], indent=2)}

Provide:
1. Content gaps that need to be filled
2. Specific passages to create (≤300 words each)
3. Channel marketing strategy
4. Content structure improvements

Format as JSON with specific, actionable recommendations."""

            response = self.chat_model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            return {
                "content_gaps": ["Unable to generate recommendations"],
                "passages_to_create": [],
                "channel_strategy": "Error generating strategy",
                "structure_improvements": []
            }
    
    def create_optimized_passage(self, query: str, context: str) -> str:
        """Create an optimized passage for a specific query"""
        try:
            prompt = f"""Create a focused, SEO-optimized passage that directly answers this query: "{query}"

Context about existing content: {context}

Requirements:
- Maximum 300 words
- Single-purpose: directly answer the query
- Include semantic variations of key terms
- Structure for featured snippet potential
- Natural, conversational tone
- Include specific details and examples

Write the passage:"""

            response = self.chat_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error creating passage: {str(e)}"
    
    def expand_query(self, query: str) -> List[str]:
        """Use LLM to expand query into variations"""
        try:
            prompt = f'''Given the search query: "{query}"
            
Generate 5 different ways users might ask about this topic to an AI assistant.
Include variations in:
- Question format (how to, what is, why, when)
- Detail level (beginner vs expert)
- Use case focus (practical vs theoretical)

Return only the 5 variations, one per line, no numbering or bullets.'''
            response = self.chat_model.generate_content(prompt)
            variations = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            return variations[:5]
        except Exception as e:
            self.logger.error(f"Query expansion failed: {str(e)}")
            return [query]  # Return original query if expansion fails


class EnhancedRelevanceScorer:
    """Enhanced relevance scoring using real embeddings"""
    
    def __init__(self, embedding_provider: GeminiEmbeddingProvider, semantic_weight: float = 0.7):
        self.embedding_provider = embedding_provider
        self.semantic_weight = semantic_weight
        self.token_weight = 1.0 - semantic_weight
        self.logger = logging.getLogger(f"{__name__}.EnhancedRelevanceScorer")
        
    def score_chunks(self, chunks: List[Dict], query: str, use_embeddings: bool = True) -> Tuple[List[Dict], Dict]:
        """Score chunks with real embeddings"""
        self.logger.info(f"Starting relevance scoring for {len(chunks)} chunks (use_embeddings: {use_embeddings})")
        scores = []
        query_embedding = None
        expanded_queries = []
        
        # Get query embedding and expansions
        if use_embeddings and self.embedding_provider:
            self.logger.info("Generating query embedding and expansions")
            query_embedding = self.embedding_provider.generate_query_embedding(query)
            expanded_queries = self.embedding_provider.expand_query(query)
            self.logger.info(f"Generated {len(expanded_queries)} query expansions")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                st.text(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Token overlap score (original method)
            query_terms = set(query.lower().split())
            content_lower = chunk['content'].lower()
            content_terms = set(content_lower.split())
            
            overlap = len(query_terms.intersection(content_terms))
            token_score = overlap / len(query_terms) if query_terms else 0
            
            # Semantic similarity with embeddings
            semantic_score = 0.0
            if use_embeddings and query_embedding and self.embedding_provider:
                chunk_embedding = self.embedding_provider.generate_embedding(chunk['content'])
                if chunk_embedding:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [chunk_embedding]
                    )[0][0]
                    semantic_score = max(0, similarity)  # Ensure non-negative
                else:
                    # Fallback to simple similarity
                    semantic_score = self._calculate_simple_similarity(content_lower, query.lower())
            else:
                semantic_score = self._calculate_simple_similarity(content_lower, query.lower())
            
            # Combined score
            final_score = (self.semantic_weight * semantic_score + 
                          self.token_weight * token_score)
            
            # Check expanded queries for bonus
            expansion_bonus = 0.0
            if expanded_queries:
                for exp_query in expanded_queries:
                    if exp_query.lower() in content_lower:
                        expansion_bonus = 0.1
                        break
            
            final_score = min(1.0, final_score + expansion_bonus)
            
            scores.append({
                'chunk_id': chunk['chunk_id'],
                'page_url': chunk['page_url'],
                'page_title': chunk['page_title'],
                'content_preview': chunk['content'][:150] + "...",
                'semantic_similarity': semantic_score,
                'token_overlap': token_score,
                'expansion_bonus': expansion_bonus,
                'final_score': final_score,
                'word_count': chunk['word_count']
            })
        
        self.logger.info(f"Scoring complete. Average score: {np.mean([s['final_score'] for s in scores]):.3f}")
        
        analysis_metadata = {
            'used_embeddings': use_embeddings and query_embedding is not None,
            'expanded_queries': expanded_queries,
            'total_chunks_processed': len(chunks)
        }
        
        return sorted(scores, key=lambda x: x['final_score'], reverse=True), analysis_metadata
    
    def _calculate_simple_similarity(self, text: str, query: str) -> float:
        """Fallback similarity calculation"""
        text_words = Counter(text.split())
        query_words = Counter(query.split())
        
        common_words = sum((text_words & query_words).values())
        total_words = sum(query_words.values())
        
        if total_words > 0:
            base_score = common_words / total_words
            if query in text:
                base_score = min(1.0, base_score + 0.3)
            return min(1.0, base_score)
        return 0.0


class ContentAnalyzer:
    """Main analyzer that orchestrates the analysis pipeline"""
    
    def __init__(self, crawler: WebCrawler, chunker: ContentChunker, scorer):
        self.crawler = crawler
        self.chunker = chunker
        self.scorer = scorer
        
    def analyze(self, url: str, query: str, progress_callback=None) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        results = {
            'url': url,
            'query': query,
            'timestamp': datetime.now()
        }
        
        # Step 1: Crawl
        if progress_callback:
            progress_callback(0.2, "🕷️ Crawling website...")
        crawl_results = self.crawler.crawl(url)
        results['crawl'] = crawl_results
        
        # Step 2: Chunk
        if progress_callback:
            progress_callback(0.4, "✂️ Chunking content...")
        chunks = self.chunker.chunk_content(crawl_results['pages'])
        results['chunks'] = chunks
        
        # Step 3: Score with embeddings
        if progress_callback:
            progress_callback(0.5, "🧮 Generating embeddings and scoring...")
        scores, metadata = self.scorer.score_chunks(chunks, query, use_embeddings)
        results['scores'] = scores
        results['scoring_metadata'] = metadata
        
        # Step 4: Generate insights
        if progress_callback:
            progress_callback(0.8, "💡 Generating insights...")
        results['insights'] = self._generate_insights(results)
        
        if progress_callback:
            progress_callback(1.0, "✅ Analysis complete!")
            
        return results
    
    def _generate_insights(self, results: Dict) -> Dict[str, Any]:
        """Generate insights from analysis results"""
        scores = results['scores']
        avg_score = np.mean([s['final_score'] for s in scores]) if scores else 0
        
        insights = {
            'average_score': avg_score,
            'top_performing_pages': self._get_top_pages(scores),
            'content_distribution': self._analyze_distribution(scores),
            'recommendations': self._generate_recommendations(avg_score, results)
        }
        
        return insights
    
    def _get_top_pages(self, scores: List[Dict]) -> List[Dict]:
        """Get top performing pages"""
        page_scores = {}
        
        for score in scores:
            url = score['page_url']
            if url not in page_scores:
                page_scores[url] = {
                    'url': url,
                    'title': score['page_title'],
                    'scores': []
                }
            page_scores[url]['scores'].append(score['final_score'])
        
        # Calculate average score per page
        for page in page_scores.values():
            page['avg_score'] = np.mean(page['scores'])
            page['chunk_count'] = len(page['scores'])
            
        return sorted(page_scores.values(), key=lambda x: x['avg_score'], reverse=True)[:5]
    
    def _analyze_distribution(self, scores: List[Dict]) -> Dict:
        """Analyze score distribution"""
        if not scores:
            return {'high': 0, 'medium': 0, 'low': 0}
            
        score_values = [s['final_score'] for s in scores]
        return {
            'high': len([s for s in score_values if s >= 0.7]),
            'medium': len([s for s in score_values if 0.4 <= s < 0.7]),
            'low': len([s for s in score_values if s < 0.4])
        }
    
    def _generate_recommendations(self, avg_score: float, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if avg_score < 0.3:
            recommendations.extend([
                "🚨 Your content needs significant optimization for AI responses",
                "📝 Add more comprehensive content covering the target query topics",
                "🔍 Use more specific terminology related to your target keywords",
                "📊 Consider creating dedicated pages for important topics"
            ])
        elif avg_score < 0.6:
            recommendations.extend([
                "⚠️ Your content has moderate relevance but needs improvement",
                "💡 Expand existing content with more detailed information",
                "🎯 Focus on creating semantic connections between related topics",
                "📈 Structure content in clear, digestible sections"
            ])
        else:
            recommendations.extend([
                "✅ Your content is well-optimized for AI responses!",
                "🚀 Continue creating comprehensive, authoritative content",
                "📊 Monitor performance and update content regularly",
                "🎯 Consider expanding into related topic areas"
            ])
            
        # Specific recommendations based on data
        if results['crawl']['total_pages'] < 5:
            recommendations.append("📄 Consider adding more pages to cover topics comprehensively")
            
        return recommendations


class LLMContentAnalyzer(ContentAnalyzer):
    """Enhanced analyzer with LLM capabilities"""
    
    def __init__(self, crawler: WebCrawler, chunker: ContentChunker, 
                 scorer: EnhancedRelevanceScorer, embedding_provider: GeminiEmbeddingProvider):
        super().__init__(crawler, chunker, scorer)
        self.embedding_provider = embedding_provider
        self.logger = logging.getLogger(f"{__name__}.LLMContentAnalyzer")
        
    def analyze(self, url: str, query: str, use_embeddings: bool = True, 
                progress_callback=None) -> Dict[str, Any]:
        """Enhanced analysis following the exact workflow diagram"""
        self.logger.info(f"Starting analysis for URL: {url}, Query: {query}, Use embeddings: {use_embeddings}")
        
        results = {
            'url': url,
            'query': query,
            'timestamp': datetime.now(),
            'used_embeddings': use_embeddings
        }
        
        # Step 1-2: Web Crawling (max 3 pages per diagram)
        if progress_callback:
            progress_callback(0.1, "🕷️ Step 1-2: Web Crawling (max 3 pages)...")
        self.logger.info("Step 1-2: Web Crawling")
        self.crawler.max_pages = min(self.crawler.max_pages, 3)  # Limit to 3 pages per diagram
        crawl_results = self.crawler.crawl(url)
        results['crawl'] = crawl_results
        
        # Step 3: Content Chunking (max 100 chunks per diagram)
        if progress_callback:
            progress_callback(0.2, "✂️ Step 3: Content Chunking (max 100)...")
        self.logger.info("Step 3: Content Chunking")
        chunks = self.chunker.chunk_content(crawl_results['pages'])
        chunks = chunks[:100]  # Limit to 100 chunks per diagram
        results['chunks'] = chunks
        
        # Step 4: Embedding & Indexing
        if progress_callback:
            progress_callback(0.3, "🧮 Step 4: Embedding & Indexing...")
        self.logger.info("Step 4: Embedding & Indexing")
        
        if use_embeddings and self.embedding_provider:
            # Generate embeddings for all chunks
            chunk_embeddings = {}
            for i, chunk in enumerate(chunks):
                if i % 10 == 0 and progress_callback:
                    progress_callback(0.3 + (0.1 * i / len(chunks)), f"Embedding chunk {i+1}/{len(chunks)}...")
                embedding = self.embedding_provider.generate_embedding(chunk['content'])
                if embedding:
                    chunk_embeddings[chunk['chunk_id']] = embedding
            results['chunk_embeddings_count'] = len(chunk_embeddings)
            
            # Step 5: Reverse Query Generation (max 5 per chunk)
            if progress_callback:
                progress_callback(0.4, "🔄 Step 5: Reverse Query Generation (max 5 per chunk)...")
            self.logger.info("Step 5: Reverse Query Generation")
            
            all_reverse_queries = {}
            for i, chunk in enumerate(chunks[:20]):  # Process top 20 chunks for efficiency
                if i % 5 == 0 and progress_callback:
                    progress_callback(0.4 + (0.1 * i / 20), f"Generating queries for chunk {i+1}/20...")
                reverse_queries = self.embedding_provider.generate_reverse_queries(chunk)
                if reverse_queries:
                    all_reverse_queries[chunk['chunk_id']] = reverse_queries
            
            results['total_reverse_queries'] = sum(len(q) for q in all_reverse_queries.values())
            
            # Step 6: Select Top 5 Reverse Queries
            if progress_callback:
                progress_callback(0.5, "🎯 Step 6: Selecting Top 5 Reverse Queries...")
            self.logger.info("Step 6: Selecting Top 5 Reverse Queries")
            
            top_reverse_queries = self.embedding_provider.select_top_reverse_queries(
                all_reverse_queries, query, chunks
            )
            results['top_reverse_queries'] = top_reverse_queries
            
            # Step 7: Query Fan-Out Expansion (max 15 per query)
            if progress_callback:
                progress_callback(0.6, "🌐 Step 7: Query Fan-Out (max 15 per query)...")
            self.logger.info("Step 7: Query Fan-Out Expansion")
            
            fanout_results = self.embedding_provider.generate_query_fanout_batch(top_reverse_queries)
            results['fanout_results'] = fanout_results
            
            # Flatten all fan-out queries for scoring
            all_fanout_queries = []
            for base_query, expansions in fanout_results.items():
                for exp in expansions:
                    all_fanout_queries.append({
                        'base_query': base_query,
                        'expanded_query': exp.get('query', ''),
                        'type': exp.get('type', ''),
                        'relevance': exp.get('relevance', 'medium')
                    })
            
            # Step 8: Comprehensive Scoring (Content × Fan-out Queries)
            if progress_callback:
                progress_callback(0.7, "📊 Step 8: Matrix Scoring (Content × Fan-out Queries)...")
            self.logger.info("Step 8: Comprehensive Scoring Engine")
            
            # Score all content against all fan-out queries
            matrix_scores = {}
            unique_queries = list(set([q['expanded_query'] for q in all_fanout_queries]))
            
            for i, query_text in enumerate(unique_queries):
                if i % 10 == 0 and progress_callback:
                    progress_callback(0.7 + (0.1 * i / len(unique_queries)), 
                                    f"Scoring query {i+1}/{len(unique_queries)}...")
                
                query_embedding = self.embedding_provider.generate_query_embedding(query_text)
                if query_embedding and chunk_embeddings:
                    chunk_scores = []
                    for chunk_id, chunk_embedding in chunk_embeddings.items():
                        similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                        chunk_scores.append({
                            'chunk_id': chunk_id,
                            'score': max(0, similarity)
                        })
                    
                    # Average of top 3 chunks
                    top_scores = sorted(chunk_scores, key=lambda x: x['score'], reverse=True)[:3]
                    avg_score = np.mean([s['score'] for s in top_scores]) if top_scores else 0
                    matrix_scores[query_text] = avg_score
            
            results['matrix_scores'] = matrix_scores
            results['coverage_stats'] = {
                'well_covered': sum(1 for s in matrix_scores.values() if s > 0.7),
                'partially_covered': sum(1 for s in matrix_scores.values() if 0.4 <= s <= 0.7),
                'poorly_covered': sum(1 for s in matrix_scores.values() if s < 0.4)
            }
            
            # Step 9: Analysis & Optimization (XAI, Channel, Brand)
            if progress_callback:
                progress_callback(0.85, "💡 Step 9: XAI Content Optimization...")
            self.logger.info("Step 9: Analysis & Optimization")
            
            # XAI Content Optimization
            xai_optimization = self.embedding_provider.generate_xai_optimization(
                matrix_scores, query
            )
            results['xai_optimization'] = xai_optimization
            
            # Channel Analysis
            if progress_callback:
                progress_callback(0.9, "📱 Step 9: Channel Analysis...")
            channel_analysis = self.embedding_provider.generate_channel_analysis(
                query, "website"  # Could be detected from content
            )
            results['channel_analysis'] = channel_analysis
            
            # Brand Curation (simplified - would need industry detection)
            if progress_callback:
                progress_callback(0.95, "🏷️ Step 9: Brand Curation...")
            brand_curation = self.embedding_provider.generate_brand_curation(
                "general",  # Would detect from content
                []  # Would extract from content
            )
            results['brand_curation'] = brand_curation
            
            # Legacy compatibility - include some original analysis
            results['ai_analysis'] = {
                'content_gaps': xai_optimization.get('content_improvements', []),
                'query_variations': top_reverse_queries,
                'extracted_query_count': results['total_reverse_queries'],
                'fanout_query_count': len(all_fanout_queries)
            }
        
        # Step 10: Final Results & Recommendations
        if progress_callback:
            progress_callback(0.98, "📋 Step 10: Generating Final Recommendations...")
        self.logger.info("Step 10: Final Results & Recommendations")
        
        # Generate insights (original scoring for backward compatibility)
        scores, metadata = self.scorer.score_chunks(chunks, query, use_embeddings)
        results['scores'] = scores
        results['scoring_metadata'] = metadata
        results['insights'] = self._generate_insights(results)
        
        if progress_callback:
            progress_callback(1.0, "✅ Analysis complete!")
        
        self.logger.info("Analysis complete following diagram workflow!")
        return results
    
    def _generate_insights(self, results: Dict) -> Dict[str, Any]:
        """Generate insights from analysis results"""
        scores = results['scores']
        avg_score = np.mean([s['final_score'] for s in scores]) if scores else 0
        
        insights = {
            'average_score': avg_score,
            'top_performing_pages': self._get_top_pages(scores),
            'content_distribution': self._analyze_distribution(scores),
            'recommendations': self._generate_recommendations(avg_score, results)
        }
        
        return insights
    
    def _get_top_pages(self, scores: List[Dict]) -> List[Dict]:
        """Get top performing pages"""
        page_scores = {}
        
        for score in scores:
            url = score['page_url']
            if url not in page_scores:
                page_scores[url] = {
                    'url': url,
                    'title': score['page_title'],
                    'scores': []
                }
            page_scores[url]['scores'].append(score['final_score'])
        
        # Calculate average score per page
        for page in page_scores.values():
            page['avg_score'] = np.mean(page['scores'])
            page['chunk_count'] = len(page['scores'])
            
        return sorted(page_scores.values(), key=lambda x: x['avg_score'], reverse=True)[:5]
    
    def _analyze_distribution(self, scores: List[Dict]) -> Dict:
        """Analyze score distribution"""
        if not scores:
            return {'high': 0, 'medium': 0, 'low': 0}
            
        score_values = [s['final_score'] for s in scores]
        return {
            'high': len([s for s in score_values if s >= 0.7]),
            'medium': len([s for s in score_values if 0.4 <= s < 0.7]),
            'low': len([s for s in score_values if s < 0.4])
        }
    
    def _generate_recommendations(self, avg_score: float, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if avg_score < 0.3:
            recommendations.extend([
                "🚨 Your content needs significant optimization for AI responses",
                "📝 Add more comprehensive content covering the target query topics",
                "🔍 Use more specific terminology related to your target keywords",
                "📊 Consider creating dedicated pages for important topics"
            ])
        elif avg_score < 0.6:
            recommendations.extend([
                "⚠️ Your content has moderate relevance but needs improvement",
                "💡 Expand existing content with more detailed information",
                "🎯 Focus on creating semantic connections between related topics",
                "📈 Structure content in clear, digestible sections"
            ])
        else:
            recommendations.extend([
                "✅ Your content is well-optimized for AI responses!",
                "🚀 Continue creating comprehensive, authoritative content",
                "📊 Monitor performance and update content regularly",
                "🎯 Consider expanding into related topic areas"
            ])
            
        # Specific recommendations based on data
        if results['crawl']['total_pages'] < 5:
            recommendations.append("📄 Consider adding more pages to cover topics comprehensively")
            
        return recommendations


# ================== STREAMLIT UI ==================



# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Initialize analyzer components
@st.cache_resource
def get_analyzer(api_key: str, max_pages: int, chunk_size: int, 
                semantic_weight: float, use_embeddings: bool) -> LLMContentAnalyzer:
    """Create and cache analyzer instance with optional Gemini integration"""
    crawler = WebCrawler(max_pages=max_pages)
    chunker = ContentChunker(chunk_size=chunk_size)
    
    if use_embeddings and api_key:
        embedding_provider = GeminiEmbeddingProvider(api_key)
        scorer = EnhancedRelevanceScorer(embedding_provider, semantic_weight)
        return LLMContentAnalyzer(crawler, chunker, scorer, embedding_provider)
    else:
        # Fallback to simple scorer
        scorer = EnhancedRelevanceScorer(None, semantic_weight)
        return LLMContentAnalyzer(crawler, chunker, scorer, None)

# Main app
st.markdown('<h1 class="main-header">Zero-Click Compass 🧭</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real Website Analysis for AI-First Performance</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🔍 Analysis Settings")
    url = st.text_input("Website URL", value="https://example.com", help="Enter the website URL to analyze")
    query = st.text_input("Target Query", value="marketing strategies", help="What query should your content rank for in AI responses?")
    
    st.subheader("🤖 Gemini API Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key for embeddings")
    use_embeddings = st.checkbox("Use Gemini Embeddings", value=bool(api_key), 
                                help="Enable real semantic analysis with Gemini")
    
    with st.expander("Advanced Settings", expanded=False):
        max_pages = st.slider("Max Pages to Crawl", 5, 50, 10)
        chunk_size = st.slider("Chunk Size (words)", 100, 300, 150)
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7, 0.1)
        
    st.divider()
    
    if use_embeddings and not api_key:
        st.warning("⚠️ Please enter Gemini API key to use embeddings")
    elif use_embeddings:
        st.success("✅ Gemini API configured")
    else:
        st.info("ℹ️ Using simplified scoring (no API needed)")
    
    st.info("""
    **💡 Tips:**
    - Get Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Embeddings provide 10x better semantic analysis
    - Start with 5-10 pages for quick analysis
    """)

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🚀 Analysis", "📈 Results", "🔍 Query Analysis", "📊 Insights", "💡 Recommendations", "🎯 Gap Analysis", "📋 Report"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Analyze Your Website")
        st.markdown("""
        This tool analyzes your actual website content to determine how well it would perform in AI-generated responses.
        
        **What happens during analysis:**
        1. **Real crawling** of your website pages
        2. **Smart chunking** into LLM-friendly segments  
        3. **Semantic embeddings** using Gemini AI (if enabled)
        4. **AI-powered insights** including query expansion and content gaps
        5. **Actionable recommendations** for optimization
        """)
        
        # Validate URL
        url_valid = url.startswith(('http://', 'https://'))
        can_analyze = url_valid and (not use_embeddings or api_key)
        
        if st.button("🚀 Start Real Analysis", type="primary", use_container_width=True, disabled=not can_analyze):
            if not url_valid:
                st.error("Please enter a valid URL starting with http:// or https://")
            elif use_embeddings and not api_key:
                st.error("Please enter your Gemini API key to use embeddings")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress: float, status: str):
                    progress_bar.progress(progress)
                    status_text.text(status)
                
                try:
                    # Get analyzer with current settings
                    analyzer = get_analyzer(api_key, max_pages, chunk_size, semantic_weight, use_embeddings)
                    
                    # Run analysis
                    with st.spinner("Analyzing your website..."):
                        logger.info(f"Starting analysis for {url}")
                        start_time = time.time()
                        
                        results = analyzer.analyze(url, query, use_embeddings, update_progress)
                        
                        elapsed_time = time.time() - start_time
                        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
                        
                        st.session_state.analysis_results = results
                        
                    st.success(f"✅ Analysis complete in {elapsed_time:.1f} seconds! Check the Results tab for details.")
                    
                    # Show workflow step if available
                    if use_embeddings and 'top_reverse_queries' in results:
                        with st.expander("📊 Workflow Analysis Summary", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Pages Crawled", results['crawl']['total_pages'])
                                st.metric("Chunks Created", len(results['chunks']))
                            with col2:
                                st.metric("Reverse Queries", results.get('total_reverse_queries', 0))
                                st.metric("Top Selected", len(results.get('top_reverse_queries', [])))
                            with col3:
                                coverage = results.get('coverage_stats', {})
                                st.metric("Well Covered", coverage.get('well_covered', 0))
                                st.metric("Needs Work", coverage.get('poorly_covered', 0))
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    if "API key not valid" in str(e):
                        st.info("Please check your Gemini API key is correct")
                    else:
                        st.info("Make sure the URL is accessible and try again.")
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Real Analysis Benefits")
        st.markdown("""
        **✅ Actual Content** - Analyzes your real pages
        
        **✅ True Metrics** - Based on your content
        
        **✅ AI-Powered** - Gemini embeddings & LLM insights
        
        **✅ Query Expansion** - See how users ask AI
        
        **✅ Content Gaps** - Find what's missing
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if use_embeddings:
            st.info("🤖 **Gemini Features Active:**\n- Semantic embeddings\n- Query expansion\n- Content gap analysis")

with tab2:
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Key Metrics
        st.header("📊 Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pages Analyzed", results['crawl']['total_pages'])
        with col2:
            st.metric("Content Chunks", len(results['chunks']))
        with col3:
            st.metric("Avg Relevance", f"{results['insights']['average_score']:.2%}")
        with col4:
            st.metric("High-Scoring Chunks", results['insights']['content_distribution']['high'])
        
        # Top Performing Content
        st.header("🏆 Top Performing Pages")
        
        top_pages = results['insights']['top_performing_pages']
        if top_pages:
            fig = go.Figure(data=[
                go.Bar(
                    x=[p['avg_score'] for p in top_pages],
                    y=[p['title'][:50] + "..." if len(p['title']) > 50 else p['title'] for p in top_pages],
                    orientation='h',
                    marker_color='rgba(102, 126, 234, 0.8)',
                    text=[f"{p['avg_score']:.3f}" for p in top_pages],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Top Pages by Average Relevance Score",
                xaxis_title="Relevance Score",
                yaxis_title="Page Title",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Scores
        st.header("📋 Detailed Chunk Analysis")
        
        # Create DataFrame for display
        scores_df = pd.DataFrame(results['scores'][:20])  # Show top 20
        
        # Display with custom formatting
        display_columns = ['page_title', 'content_preview', 'final_score', 'semantic_similarity', 'token_overlap']
        if 'expansion_bonus' in scores_df.columns:
            display_columns.append('expansion_bonus')
            
        st.dataframe(
            scores_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
        # Show embeddings status
        if results.get('scoring_metadata', {}).get('used_embeddings'):
            st.success("✅ Analysis used Gemini embeddings for enhanced accuracy")
        else:
            st.info("ℹ️ Analysis used keyword matching (add Gemini API key for better results)")
        
    else:
        st.info("👈 Start an analysis from the Analysis tab to see results")

with tab3:
    if st.session_state.analysis_results and 'top_reverse_queries' in st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.header("🔍 Query Analysis Following Diagram Workflow")
        
        # Workflow Progress
        st.subheader("📊 Analysis Workflow Progress")
        workflow_steps = [
            "1-2. Web Crawling",
            "3. Content Chunking", 
            "4. Embedding & Indexing",
            "5. Reverse Query Generation",
            "6. Select Top 5 Queries",
            "7. Query Fan-Out",
            "8. Matrix Scoring",
            "9. Analysis & Optimization",
            "10. Final Results"
        ]
        
        progress_cols = st.columns(len(workflow_steps))
        for i, (col, step) in enumerate(zip(progress_cols, workflow_steps)):
            with col:
                st.info(f"✅ {step}")
        
        # Reverse Query Analysis
        st.subheader("🔄 Step 5-6: Reverse Query Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Reverse Queries Generated", 
                     results.get('total_reverse_queries', 0))
        with col2:
            st.metric("Top Selected Queries", 
                     len(results.get('top_reverse_queries', [])))
        
        # Display top reverse queries
        if 'top_reverse_queries' in results:
            st.write("**Selected Top 5 Reverse Queries:**")
            for i, rq in enumerate(results['top_reverse_queries'], 1):
                st.write(f"{i}. {rq}")
        
        # Fan-Out Results Table
        if 'fanout_results' in results:
            st.subheader("🌐 Step 7: Query Fan-Out Results (Max 15 per query)")
            
            # Create comprehensive fan-out table
            fanout_data = []
            for base_query, expansions in results['fanout_results'].items():
                for exp in expansions:
                    fanout_data.append({
                        'Base Query': base_query[:50] + "..." if len(base_query) > 50 else base_query,
                        'Expanded Query': exp.get('query', ''),
                        'Type': exp.get('type', ''),
                        'Relevance': exp.get('relevance', 'medium')
                    })
            
            if fanout_data:
                fanout_df = pd.DataFrame(fanout_data)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Fan-Out Queries", len(fanout_df))
                with col2:
                    avg_per_base = len(fanout_df) / len(results['fanout_results']) if results['fanout_results'] else 0
                    st.metric("Avg Expansions per Query", f"{avg_per_base:.1f}")
                with col3:
                    high_rel = len(fanout_df[fanout_df['Relevance'] == 'high']) if 'Relevance' in fanout_df else 0
                    st.metric("High Relevance Queries", high_rel)
                
                # Display table with grouping
                st.dataframe(
                    fanout_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Relevance": st.column_config.TextColumn(
                            "Relevance",
                            help="Query relevance level"
                        )
                    }
                )
                
                # Type distribution
                type_dist = fanout_df['Type'].value_counts()
                fig = px.bar(
                    x=type_dist.index,
                    y=type_dist.values,
                    title="Query Type Distribution in Fan-Out",
                    labels={'x': 'Query Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Matrix Scoring Results
        if 'matrix_scores' in results:
            st.subheader("📊 Step 8: Matrix Scoring Results (Content × Queries)")
            
            coverage_stats = results.get('coverage_stats', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Well Covered", coverage_stats.get('well_covered', 0), 
                         help="Queries with score > 0.7")
            with col2:
                st.metric("Partially Covered", coverage_stats.get('partially_covered', 0),
                         help="Queries with score 0.4-0.7")
            with col3:
                st.metric("Poorly Covered", coverage_stats.get('poorly_covered', 0),
                         help="Queries with score < 0.4")
            
            # Create heatmap visualization
            if len(results['matrix_scores']) > 0:
                # Sample for visualization (top 20 queries)
                sample_scores = dict(list(results['matrix_scores'].items())[:20])
                
                fig = go.Figure(data=go.Heatmap(
                    z=[list(sample_scores.values())],
                    x=[q[:40] + "..." if len(q) > 40 else q for q in sample_scores.keys()],
                    y=["Coverage Score"],
                    colorscale='RdYlGn',
                    text=[[f"{s:.2%}" for s in sample_scores.values()]],
                    texttemplate="%{text}",
                    hovertemplate="Query: %{x}<br>Score: %{z:.2%}<extra></extra>"
                ))
                
                fig.update_layout(
                    title="Content Coverage Heatmap (Sample of 20 Queries)",
                    xaxis_title="Fan-Out Queries",
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Download complete analysis
        if st.button("📥 Download Complete Query Analysis", use_container_width=True):
            analysis_data = {
                'workflow_stage': 'Complete',
                'reverse_queries': results.get('top_reverse_queries', []),
                'fanout_results': results.get('fanout_results', {}),
                'matrix_scores': results.get('matrix_scores', {}),
                'coverage_stats': results.get('coverage_stats', {})
            }
            
            json_str = json.dumps(analysis_data, indent=2)
            st.download_button(
                label="Download Query Analysis JSON",
                data=json_str,
                file_name=f"query_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
    else:
        st.info("👈 Run analysis with embeddings enabled to see query analysis")
        
with tab4:
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        insights = results['insights']
        
        st.header("📊 Content Insights")
        
        # Score Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score Distribution")
            dist = insights['content_distribution']
            
            fig = px.pie(
                values=[dist['high'], dist['medium'], dist['low']],
                names=['High (≥0.7)', 'Medium (0.4-0.7)', 'Low (<0.4)'],
                color_discrete_map={'High (≥0.7)': '#10B981', 'Medium (0.4-0.7)': '#F59E0B', 'Low (<0.4)': '#EF4444'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Score Components
            st.subheader("Scoring Components")
            
            scores = results['scores'][:10]  # Sample
            avg_semantic = np.mean([s['semantic_similarity'] for s in scores])
            avg_token = np.mean([s['token_overlap'] for s in scores])
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Semantic Similarity', 'Token Overlap'],
                    y=[avg_semantic, avg_token],
                    marker_color=['rgba(102, 126, 234, 0.8)', 'rgba(118, 75, 162, 0.8)']
                )
            ])
            fig.update_layout(
                title="Average Component Scores",
                yaxis_title="Score",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Content Coverage
        st.header("📑 Content Coverage Analysis")
        
        # Word frequency in top chunks
        top_chunks = [s for s in results['scores'] if s['final_score'] > 0.5]
        if top_chunks:
            all_text = ' '.join([chunk['content_preview'] for chunk in top_chunks])
            words = re.findall(r'\w+', all_text.lower())
            word_freq = Counter(words)
            
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}
            word_freq = {k: v for k, v in word_freq.items() if k not in common_words and len(k) > 3}
            
            # Top words
            top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15])
            
            fig = px.bar(
                x=list(top_words.values()),
                y=list(top_words.keys()),
                orientation='h',
                title="Most Frequent Terms in High-Scoring Content"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Start an analysis from the Analysis tab to see insights")

with tab5:
    if st.session_state.analysis_results and 'xai_optimization' in st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.header("💡 Step 9: Analysis & Optimization Layer")
        
        # Create tabs for the three optimization areas
        opt_tab1, opt_tab2, opt_tab3 = st.tabs(["🤖 XAI Content Optimization", "📱 Channel Analysis", "🏷️ Brand Curation"])
        
        with opt_tab1:
            st.subheader("XAI Content Optimization")
            xai_opt = results.get('xai_optimization', {})
            
            # Content Improvements
            st.write("**📝 Content Improvements with Explanations:**")
            improvements = xai_opt.get('content_improvements', [])
            if isinstance(improvements, list):
                for i, imp in enumerate(improvements[:5], 1):
                    st.write(f"{i}. {imp}")
            else:
                st.write(improvements)
            
            # Keyword Strategies
            st.write("**🔍 Keyword Optimization Strategies:**")
            keywords = xai_opt.get('keyword_strategies', [])
            if keywords:
                for strategy in keywords[:5]:
                    st.write(f"• {strategy}")
            
            # Semantic Recommendations
            st.write("**🧠 Semantic Coverage Recommendations:**")
            semantic = xai_opt.get('semantic_recommendations', [])
            if semantic:
                for rec in semantic[:5]:
                    st.write(f"• {rec}")
            
            # Structure Enhancements
            st.write("**🏗️ Structure Enhancements for AI Comprehension:**")
            structure = xai_opt.get('structure_enhancements', [])
            if structure:
                for enh in structure[:5]:
                    st.write(f"• {enh}")
        
        with opt_tab2:
            st.subheader("Platform-Specific Channel Strategies")
            channel = results.get('channel_analysis', {})
            
            # Reddit Strategy
            with st.expander("🟠 Reddit Strategy", expanded=True):
                reddit = channel.get('reddit', {})
                st.write("**Target Subreddits:**")
                for sub in reddit.get('subreddits', [])[:5]:
                    st.write(f"• r/{sub}")
                st.write(f"\n**Engagement Tactics:** {reddit.get('tactics', 'N/A')}")
            
            # X (Twitter) Strategy
            with st.expander("🐦 X (Twitter) Strategy"):
                twitter = channel.get('twitter', {})
                st.write("**Hashtag Strategy:**")
                for tag in twitter.get('hashtags', [])[:5]:
                    st.write(f"• #{tag}")
                st.write(f"\n**Thread Optimization:** {twitter.get('strategy', 'N/A')}")
            
            # Google Strategy
            with st.expander("🔍 Google Featured Snippets"):
                google = channel.get('google', {})
                st.write(f"**Optimization Strategy:** {google.get('optimization', 'N/A')}")
            
            # Yelp Strategy
            with st.expander("⭐ Yelp Local SEO"):
                yelp = channel.get('yelp', {})
                st.write(f"**Local SEO Approach:** {yelp.get('local_seo', 'N/A')}")
            
            # Quora Strategy
            with st.expander("❓ Quora Optimization"):
                quora = channel.get('quora', {})
                st.write("**Target Questions:**")
                for q in quora.get('questions', [])[:5]:
                    st.write(f"• {q}")
                st.write(f"\n**Answer Approach:** {quora.get('approach', 'N/A')}")
        
        with opt_tab3:
            st.subheader("Industry-Specific Brand Curation")
            brand = results.get('brand_curation', {})
            
            # Brand Positioning
            st.write("**🎯 Brand Positioning Recommendations:**")
            st.info(brand.get('positioning', 'N/A'))
            
            # Differentiation Strategies
            st.write("**🌟 Content Differentiation Strategies:**")
            diff_strategies = brand.get('differentiation', [])
            if isinstance(diff_strategies, list):
                for strategy in diff_strategies[:5]:
                    st.write(f"• {strategy}")
            else:
                st.write(diff_strategies)
            
            # Competitive Advantages
            st.write("**💪 Competitive Advantages to Highlight:**")
            advantages = brand.get('competitive_advantages', [])
            if advantages:
                for adv in advantages[:5]:
                    st.write(f"• {adv}")
            
            # Messaging Guidelines
            st.write("**📢 Industry-Specific Messaging Guidelines:**")
            messaging = brand.get('messaging_guidelines', [])
            if messaging:
                for guide in messaging[:5]:
                    st.write(f"• {guide}")
            
            # Future Agentic Foundations
            st.write("**🤖 Future Agentic Engagement Foundations:**")
            agentic = brand.get('agentic_foundations', [])
            if agentic:
                for foundation in agentic[:5]:
                    st.write(f"• {foundation}")
        
        # Implementation Roadmap
        st.subheader("🗺️ Implementation Roadmap")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Phase 1: Quick Wins**\n• Content gaps filling\n• Keyword optimization\n• Basic structure improvements")
        with col2:
            st.warning("**Phase 2: Channel Expansion**\n• Platform-specific content\n• Engagement campaigns\n• Community building")
        with col3:
            st.success("**Phase 3: Brand Excellence**\n• Thought leadership\n• Agentic foundations\n• Industry authority")
        
        # Export Optimization Plan
        if st.button("📥 Export Complete Optimization Plan", use_container_width=True):
            optimization_plan = {
                'xai_optimization': results.get('xai_optimization', {}),
                'channel_analysis': results.get('channel_analysis', {}),
                'brand_curation': results.get('brand_curation', {}),
                'implementation_roadmap': {
                    'phase_1': 'Content gaps and keyword optimization',
                    'phase_2': 'Channel-specific expansion',
                    'phase_3': 'Brand authority and agentic foundations'
                }
            }
            
            json_str = json.dumps(optimization_plan, indent=2)
            st.download_button(
                label="Download Optimization Plan JSON",
                data=json_str,
                file_name=f"optimization_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
    else:
        st.info("👈 Run analysis with embeddings enabled to see optimization recommendations")

with tab6:
    if st.session_state.analysis_results and 'fanout_results' in st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.header("🎯 High-Relevance Query Gap Analysis")
        st.markdown("This analysis links high-relevance queries from fan-out to actual content chunks to identify and fill gaps.")
        
        # Extract high-relevance queries from fan-out results
        high_relevance_queries = []
        if 'fanout_results' in results:
            for base_query, expansions in results['fanout_results'].items():
                for exp in expansions:
                    if exp.get('relevance', '').lower() == 'high':
                        high_relevance_queries.append({
                            'base_query': base_query,
                            'expanded_query': exp.get('query', ''),
                            'type': exp.get('type', ''),
                            'relevance': 'high'
                        })
        
        if high_relevance_queries:
            st.metric("High-Relevance Queries Found", len(high_relevance_queries))
            
            # Score each high-relevance query against chunks
            st.subheader("📊 Query-to-Chunk Mapping Analysis")
            
            with st.spinner("Analyzing high-relevance queries against content chunks..."):
                # Get chunks and their embeddings
                chunks = results.get('chunks', [])
                chunk_embeddings = results.get('chunk_embeddings_count', 0)
                
                # Create mapping table data
                mapping_data = []
                
                # For each high-relevance query, find best matching chunks
                for hr_query in high_relevance_queries:
                    query_text = hr_query['expanded_query']
                    
                    # Get score from matrix if available
                    matrix_scores = results.get('matrix_scores', {})
                    query_score = matrix_scores.get(query_text, None)
                    
                    # Find best matching chunk
                    best_chunk = None
                    best_score = 0
                    
                    if query_score is not None:
                        # Find chunk with highest score for this query
                        # (In real implementation, we'd track this during scoring)
                        for chunk in chunks[:20]:  # Sample for display
                            # Simulate finding best chunk
                            if chunk['content'].lower().find(query_text.lower()[:20]) != -1:
                                best_chunk = chunk
                                best_score = query_score
                                break
                    
                    mapping_data.append({
                        'Query': query_text,
                        'Query Type': hr_query['type'],
                        'Base Query': hr_query['base_query'][:40] + "...",
                        'Best Match Score': query_score if query_score else 0,
                        'Coverage': 'Good' if query_score and query_score > 0.7 else 'Poor' if query_score and query_score < 0.4 else 'Partial',
                        'Best Chunk': best_chunk['chunk_id'] if best_chunk else 'No match',
                        'Chunk Preview': best_chunk['content'][:100] + "..." if best_chunk else 'No matching content found'
                    })
            
            # Display the mapping table
            if mapping_data:
                mapping_df = pd.DataFrame(mapping_data)
                
                # Apply color coding
                def color_coverage(val):
                    if val == 'Good':
                        return 'background-color: #D4EDDA; color: #155724'
                    elif val == 'Partial':
                        return 'background-color: #FFF3CD; color: #856404'
                    else:
                        return 'background-color: #F8D7DA; color: #721C24'
                
                st.dataframe(
                    mapping_df.style.applymap(color_coverage, subset=['Coverage']),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Best Match Score": st.column_config.ProgressColumn(
                            "Best Match Score",
                            help="Semantic similarity score (0-1)",
                            format="%.3f",
                            min_value=0,
                            max_value=1,
                        ),
                        "Query": st.column_config.TextColumn(
                            "High-Relevance Query",
                            width="large"
                        ),
                        "Chunk Preview": st.column_config.TextColumn(
                            "Best Matching Content",
                            width="large"
                        )
                    }
                )
                
                # Summary statistics
                st.subheader("📈 Coverage Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                coverage_counts = mapping_df['Coverage'].value_counts()
                with col1:
                    good_count = coverage_counts.get('Good', 0)
                    st.metric("Well Covered", f"{good_count}/{len(mapping_df)}", 
                             delta=f"{(good_count/len(mapping_df)*100):.1f}%")
                with col2:
                    partial_count = coverage_counts.get('Partial', 0)
                    st.metric("Partially Covered", partial_count)
                with col3:
                    poor_count = coverage_counts.get('Poor', 0)
                    st.metric("Gaps Identified", poor_count,
                             delta=f"-{(poor_count/len(mapping_df)*100):.1f}%", delta_color="inverse")
                with col4:
                    avg_score = mapping_df['Best Match Score'].mean()
                    st.metric("Avg Coverage Score", f"{avg_score:.2%}")
                
                # Identify gaps (poorly covered high-relevance queries)
                gaps_df = mapping_df[mapping_df['Coverage'] != 'Good'].sort_values('Best Match Score')
                
                if len(gaps_df) > 0:
                    st.subheader("🔍 Identified Content Gaps")
                    st.warning(f"Found {len(gaps_df)} high-relevance queries with insufficient coverage")
                    
                    # AI-Generated Recommendations
                    st.subheader("🤖 AI-Generated Content Recommendations")
                    
                    if st.button("Generate Gap-Filling Recommendations", type="primary", use_container_width=True):
                        with st.spinner("Generating detailed content recommendations..."):
                            # Prepare gap data for AI
                            gap_data = []
                            for _, row in gaps_df.iterrows():
                                gap_data.append({
                                    'query': row['Query'],
                                    'type': row['Query Type'],
                                    'best_score': row['Best Match Score'],
                                    'best_chunk_preview': row['Chunk Preview']
                                })
                            
                            # Generate recommendations using AI
                            if 'embedding_provider' in results:
                                recommendations = results['embedding_provider'].generate_gap_filling_recommendations(gap_data)
                                
                                # Display recommendations
                                for i, (gap, rec) in enumerate(zip(gap_data[:5], recommendations[:5])):
                                    with st.expander(f"📝 Recommendation {i+1}: {gap['query'][:60]}...", expanded=(i==0)):
                                        col1, col2 = st.columns([1, 1])
                                        
                                        with col1:
                                            st.write("**Query Details:**")
                                            st.write(f"- Query: {gap['query']}")
                                            st.write(f"- Type: {gap['type']}")
                                            st.write(f"- Current Score: {gap['best_score']:.2%}")
                                            
                                        with col2:
                                            st.write("**Content Recommendation:**")
                                            st.write(f"- Content Type: {rec.get('content_type', 'N/A')}")
                                            st.write(f"- Word Count: {rec.get('word_count', 'N/A')}")
                                        
                                        st.write("**Key Points to Cover:**")
                                        key_points = rec.get('key_points', [])
                                        if isinstance(key_points, list):
                                            for point in key_points:
                                                st.write(f"• {point}")
                                        
                                        st.write("**SEO Optimization Tips:**")
                                        st.info(rec.get('seo_tips', 'N/A'))
                                        
                                        st.write("**Internal Linking Opportunities:**")
                                        st.write(rec.get('linking', 'N/A'))
                                        
                                        st.write("**Example Opening Paragraph:**")
                                        st.text_area("Example", value=rec.get('example_opening', 'N/A'), 
                                                   height=100, key=f"example_{i}")
                                
                                # Final content strategy
                                st.subheader("📋 Final Content Strategy")
                                st.success("**Recommended Implementation Plan:**")
                                
                                priority_gaps = gaps_df.head(5)
                                st.write("**Priority 1: Address Critical Gaps**")
                                for _, gap in priority_gaps.iterrows():
                                    st.write(f"• Create content for: {gap['Query'][:60]}... (Score: {gap['Best Match Score']:.2%})")
                                
                                st.write("\n**Priority 2: Enhance Partial Coverage**")
                                partial_gaps = mapping_df[mapping_df['Coverage'] == 'Partial'].head(3)
                                for _, gap in partial_gaps.iterrows():
                                    st.write(f"• Expand content for: {gap['Query'][:60]}... (Score: {gap['Best Match Score']:.2%})")
                                
                                st.write("\n**Priority 3: Cross-Link and Optimize**")
                                st.write("• Create internal links between related content pieces")
                                st.write("• Optimize meta descriptions for high-relevance queries")
                                st.write("• Add FAQ sections addressing common query patterns")
                
                # Export options
                st.subheader("📥 Export Gap Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Export mapping table
                    csv_mapping = mapping_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 Download Query-Chunk Mapping CSV",
                        data=csv_mapping,
                        file_name=f"query_chunk_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export gap analysis
                    if len(gaps_df) > 0:
                        gap_analysis = {
                            'total_high_relevance_queries': len(high_relevance_queries),
                            'gaps_identified': len(gaps_df),
                            'average_gap_score': float(gaps_df['Best Match Score'].mean()),
                            'gaps': gap_data if 'gap_data' in locals() else [],
                            'recommendations': recommendations if 'recommendations' in locals() else []
                        }
                        
                        json_str = json.dumps(gap_analysis, indent=2)
                        st.download_button(
                            "📋 Download Gap Analysis JSON",
                            data=json_str,
                            file_name=f"gap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
        else:
            st.warning("No high-relevance queries found in fan-out results. Try running the analysis with embeddings enabled.")
            
    else:
        st.info("👈 Run analysis with embeddings enabled to see gap analysis")

with tab7:
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.header("📋 Analysis Report")
        
        # Executive Summary
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("Executive Summary")
        st.markdown(f"""
        **Website:** {results['url']}  
        **Target Query:** {results['query']}  
        **Analysis Date:** {results['timestamp'].strftime('%Y-%m-%d %H:%M')}
        **Using Embeddings:** {'Yes (Gemini)' if results.get('used_embeddings') else 'No (Keyword matching)'}
        
        **Key Metrics:**
        - Analyzed **{results['crawl']['total_pages']}** pages
        - Created **{len(results['chunks'])}** content chunks
        - Average relevance score: **{results['insights']['average_score']:.2%}**
        - High-performing chunks: **{results['insights']['content_distribution']['high']}**
        
        **Query Analysis:**
        - Extracted queries: **{results.get('ai_analysis', {}).get('extracted_query_count', 0)}**
        - Fan-out queries: **{results.get('ai_analysis', {}).get('fanout_query_count', 0)}**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        
        for rec in results['insights']['recommendations']:
            st.write(rec)
        
        # Export Options
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare CSV data
            scores_df = pd.DataFrame(results['scores'])
            csv_data = scores_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Full Analysis (CSV)",
                data=csv_data,
                file_name=f"zero_click_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        with col2:
            # Prepare JSON data
            export_data = {
                'url': results['url'],
                'query': results['query'],
                'timestamp': results['timestamp'].isoformat(),
                'summary': {
                    'pages_analyzed': results['crawl']['total_pages'],
                    'total_chunks': len(results['chunks']),
                    'average_score': results['insights']['average_score'],
                    'top_pages': results['insights']['top_performing_pages'][:5],
                    'ai_analysis': results.get('ai_analysis', {})
                },
                'recommendations': results['insights']['recommendations']
            }
            
            st.download_button(
                label="📄 Download Summary Report (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"zero_click_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("👈 Start an analysis from the Analysis tab to generate a report")

# Footer
st.divider()
st.markdown("""
<p style="text-align: center; color: #666;">
Zero-Click Compass MVP - Real Website Analysis for the AI-First Future<br>
Built with modular architecture following SOLID principles
</p>
""", unsafe_allow_html=True)