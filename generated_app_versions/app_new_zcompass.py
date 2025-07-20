"""
Zero-Click Compass MVP - Streamlit App
A modular implementation for real website analysis
"""

import streamlit as st
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

# ================== CORE MODULES ==================

class WebCrawler:
    """Handles website crawling and content extraction"""
    
    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages
        self.visited_urls = set()
        
    def crawl(self, start_url: str) -> Dict[str, Any]:
        """Crawl website starting from the given URL"""
        pages = []
        to_visit = [start_url]
        base_domain = urlparse(start_url).netloc
        
        while to_visit and len(pages) < self.max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
                
            try:
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract text content
                    text_content = self._extract_text(soup)
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.text.strip() if title else urlparse(url).path
                    
                    pages.append({
                        'url': url,
                        'title': title_text,
                        'content': text_content,
                        'word_count': len(text_content.split())
                    })
                    
                    self.visited_urls.add(url)
                    
                    # Find more URLs on the same domain
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if urlparse(next_url).netloc == base_domain and next_url not in self.visited_urls:
                            to_visit.append(next_url)
                            
            except Exception as e:
                st.warning(f"Failed to crawl {url}: {str(e)}")
                
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
        
    def chunk_content(self, pages: List[Dict]) -> List[Dict]:
        """Split content into semantic chunks"""
        chunks = []
        
        for page in pages:
            content = page['content']
            words = content.split()
            
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
                
        return chunks


class GeminiEmbeddingProvider:
    """Handles Gemini API for embeddings and LLM operations"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"
        self.chat_model = genai.GenerativeModel('gemini-1.5-flash')
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            st.warning(f"Embedding generation failed: {str(e)}")
            return None
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query using Gemini"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            st.warning(f"Query embedding generation failed: {str(e)}")
            return None
    
    def extract_queries_from_content(self, chunks: List[Dict]) -> List[str]:
        """Extract all queries that the content answers"""
        try:
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
            return queries
        except Exception as e:
            st.warning(f"Query extraction failed: {str(e)}")
            return []
    
    def generate_query_fanout(self, query: str, mode: str = "AI Mode (complex)") -> Dict[str, Any]:
        """Generate fan-out queries using Qforia approach"""
        try:
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
            
            return json.loads(json_text.strip())
        except Exception as e:
            st.warning(f"Query fan-out failed: {str(e)}")
            return {"generation_details": {}, "expanded_queries": []}
    
    def score_content_for_queries(self, chunks: List[Dict], queries: List[str]) -> Dict[str, Any]:
        """Score how well content addresses each query"""
        query_scores = {}
        
        for query in queries:
            # Get query embedding
            query_embedding = self.generate_query_embedding(query)
            if not query_embedding:
                continue
                
            # Score against all chunks
            chunk_scores = []
            for chunk in chunks:
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
            st.warning(f"Query expansion failed: {str(e)}")
            return [query]  # Return original query if expansion fails

    def analyze_content_gaps(self, top_chunks: List[Dict], query: str) -> str:
        """Use LLM to analyze content gaps"""
        try:
            # Prepare content summary
            content_summary = "\n".join([
                f"- {chunk['content'][:150]}..." 
                for chunk in top_chunks[:5]
            ])
            
            prompt = f'''Analyze this content for the query "{query}":\n\nTop performing content chunks:\n{content_summary}\n\nIdentify:\n1. What aspects of "{query}" are well-covered\n2. What important information is missing\n3. Specific recommendations to improve AI visibility\n\nBe concise and actionable.'''
            response = self.chat_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Content gap analysis failed: {str(e)}"


class EnhancedRelevanceScorer:
    """Enhanced relevance scoring using real embeddings"""
    
    def __init__(self, embedding_provider: GeminiEmbeddingProvider, semantic_weight: float = 0.7):
        self.embedding_provider = embedding_provider
        self.semantic_weight = semantic_weight
        self.token_weight = 1.0 - semantic_weight
        
    def score_chunks(self, chunks: List[Dict], query: str, use_embeddings: bool = True) -> Tuple[List[Dict], Dict]:
        """Score chunks with real embeddings"""
        scores = []
        query_embedding = None
        expanded_queries = []
        
        # Get query embedding and expansions
        if use_embeddings and self.embedding_provider:
            query_embedding = self.embedding_provider.generate_query_embedding(query)
            expanded_queries = self.embedding_provider.expand_query(query)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
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
            
            # Progress indicator
            if i % 10 == 0:
                st.text(f"Processing chunk {i+1}/{len(chunks)}...")
        
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
        
    def analyze(self, url: str, query: str, use_embeddings: bool = True, progress_callback=None) -> Dict[str, Any]:
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
        
    def analyze(self, url: str, query: str, use_embeddings: bool = True, 
                progress_callback=None) -> Dict[str, Any]:
        """Enhanced analysis with embeddings and query extraction"""
        results = {
            'url': url,
            'query': query,
            'timestamp': datetime.now(),
            'used_embeddings': use_embeddings
        }
        
        # Step 1: Crawl
        if progress_callback:
            progress_callback(0.1, "🕷️ Crawling website...")
        crawl_results = self.crawler.crawl(url)
        results['crawl'] = crawl_results
        
        # Step 2: Chunk
        if progress_callback:
            progress_callback(0.2, "✂️ Chunking content...")
        chunks = self.chunker.chunk_content(crawl_results['pages'])
        results['chunks'] = chunks
        
        # Step 3: Score with embeddings
        if progress_callback:
            progress_callback(0.3, "🧮 Generating embeddings and scoring...")
        scores, metadata = self.scorer.score_chunks(chunks, query, use_embeddings)
        results['scores'] = scores
        results['scoring_metadata'] = metadata
        
        if use_embeddings and self.embedding_provider:
            # Step 4: Extract queries from content
            if progress_callback:
                progress_callback(0.4, "🔍 Extracting queries from content...")
            extracted_queries = self.embedding_provider.extract_queries_from_content(chunks)
            results['extracted_queries'] = extracted_queries
            
            # Step 5: Generate fan-out for main query
            if progress_callback:
                progress_callback(0.5, "🌐 Generating query fan-out...")
            fanout_results = self.embedding_provider.generate_query_fanout(query)
            results['fanout_queries'] = fanout_results
            
            # Step 6: Score content against all queries
            if progress_callback:
                progress_callback(0.6, "📊 Scoring content coverage...")
            all_queries = [query] + extracted_queries[:10]  # Limit for performance
            query_scores = self.embedding_provider.score_content_for_queries(chunks, all_queries)
            results['query_scores'] = query_scores
            
            # Step 7: Generate recommendations
            if progress_callback:
                progress_callback(0.8, "💡 Generating optimization recommendations...")
            recommendations = self.embedding_provider.generate_content_recommendations(
                query_scores, fanout_results
            )
            results['recommendations'] = recommendations
            
            # Step 8: AI Analysis
            if progress_callback:
                progress_callback(0.9, "🤖 Finalizing AI analysis...")
            top_chunks = scores[:10]
            content_gaps = self.embedding_provider.analyze_content_gaps(top_chunks, query)
            results['ai_analysis'] = {
                'content_gaps': content_gaps,
                'query_variations': metadata.get('expanded_queries', []),
                'extracted_query_count': len(extracted_queries),
                'fanout_query_count': len(fanout_results.get('expanded_queries', []))
            }
        
        # Step 9: Generate insights
        if progress_callback:
            progress_callback(0.95, "💡 Generating insights...")
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


# ================== STREAMLIT UI ==================

# Page configuration
st.set_page_config(
    page_title="Zero-Click Compass 🧭",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 Analysis", "📈 Results", "🔍 Query Analysis", "📊 Insights", "💡 Recommendations", "📋 Report"])

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
                        results = analyzer.analyze(url, query, use_embeddings, update_progress)
                        st.session_state.analysis_results = results
                        
                    st.success("✅ Analysis complete! Check the Results tab for details.")
                    
                    # Show AI insights immediately if available
                    if use_embeddings and 'ai_analysis' in results:
                        with st.expander("🤖 AI Quick Insights", expanded=True):
                            ai_analysis = results['ai_analysis']
                            st.metric("Extracted Queries", ai_analysis.get('extracted_query_count', 0))
                            st.metric("Fan-out Queries", ai_analysis.get('fanout_query_count', 0))
                            
                            if 'recommendations' in results:
                                st.write("**Top Recommendation:**")
                                gaps = results['recommendations'].get('content_gaps', [])
                                if gaps and isinstance(gaps, list) and len(gaps) > 0:
                                    st.write(f"• {gaps[0]}")
                    
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
    if st.session_state.analysis_results and 'extracted_queries' in st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.header("🔍 Query Analysis & Fan-Out")
        
        # Extracted Queries Section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Queries Your Content Answers")
            extracted = results.get('extracted_queries', [])
            
            if extracted:
                st.info(f"Found {len(extracted)} queries that your content addresses")
                
                # Display queries with coverage scores
                query_scores = results.get('query_scores', {})
                for i, eq in enumerate(extracted[:10], 1):
                    score_data = query_scores.get(eq, {})
                    coverage = score_data.get('coverage', 'unknown')
                    avg_score = score_data.get('avg_score', 0)
                    
                    # Coverage indicator
                    if coverage == 'good':
                        icon = "🟢"
                    elif coverage == 'partial':
                        icon = "🟡"
                    else:
                        icon = "🔴"
                    
                    st.write(f"{icon} **{i}.** {eq} (Score: {avg_score:.2f})")
                
                if len(extracted) > 10:
                    st.info(f"... and {len(extracted) - 10} more queries")
            else:
                st.warning("No queries extracted. Check if embeddings are enabled.")
        
        with col2:
            st.subheader("🌐 Query Fan-Out Analysis")
            fanout = results.get('fanout_queries', {})
            
            if fanout and 'expanded_queries' in fanout:
                expanded = fanout['expanded_queries']
                details = fanout.get('generation_details', {})
                
                st.metric("Target Queries", details.get('target_query_count', 'N/A'))
                st.write(f"**Reasoning:** {details.get('reasoning_for_count', 'Not provided')}")
                
                # Group by type
                query_types = {}
                for q in expanded:
                    qtype = q.get('type', 'unknown')
                    if qtype not in query_types:
                        query_types[qtype] = []
                    query_types[qtype].append(q)
                
                # Display by type
                for qtype, queries in query_types.items():
                    with st.expander(f"{qtype.title()} ({len(queries)} queries)", expanded=False):
                        for q in queries[:3]:
                            st.write(f"**Q:** {q['query']}")
                            st.write(f"*Intent:* {q.get('user_intent', 'N/A')}")
                            st.write("---")
            else:
                st.info("Enable embeddings to see query fan-out analysis")
        
        # Coverage Matrix
        if 'query_scores' in results:
            st.subheader("📊 Content Coverage Matrix")
            
            # Create coverage visualization
            coverage_data = []
            for query, data in results['query_scores'].items():
                coverage_data.append({
                    'Query': query[:50] + "..." if len(query) > 50 else query,
                    'Score': data['avg_score'],
                    'Coverage': data['coverage']
                })
            
            if coverage_data:
                coverage_df = pd.DataFrame(coverage_data)
                
                # Color-coded bar chart
                fig = px.bar(
                    coverage_df,
                    x='Score',
                    y='Query',
                    orientation='h',
                    color='Coverage',
                    color_discrete_map={
                        'good': '#10B981',
                        'partial': '#F59E0B',
                        'poor': '#EF4444'
                    },
                    title="Query Coverage Scores"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
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
    if st.session_state.analysis_results and 'recommendations' in st.session_state.analysis_results:
        results = st.session_state.analysis_results
        recommendations = results.get('recommendations', {})
        
        st.header("💡 Content Optimization Recommendations")
        
        # Content Gaps
        st.subheader("🔍 Content Gaps to Fill")
        gaps = recommendations.get('content_gaps', [])
        if isinstance(gaps, list):
            for i, gap in enumerate(gaps, 1):
                st.write(f"{i}. {gap}")
        else:
            st.write(gaps)
        
        # Passages to Create
        st.subheader("✍️ Recommended Passages to Create")
        passages = recommendations.get('passages_to_create', [])
        
        if passages:
            for i, passage_info in enumerate(passages[:5], 1):
                with st.expander(f"Passage {i}: {passage_info.get('query', 'N/A')}", expanded=False):
                    st.write(f"**Target Query:** {passage_info.get('query', 'N/A')}")
                    st.write(f"**Purpose:** {passage_info.get('purpose', 'N/A')}")
                    st.write(f"**Key Points:** {passage_info.get('key_points', 'N/A')}")
                    
                    if st.button(f"Generate Passage {i}", key=f"gen_passage_{i}"):
                        with st.spinner("Generating optimized passage..."):
                            context = f"Current content scores {results['insights']['average_score']:.2f} for main query"
                            passage = st.session_state.analysis_results['embedding_provider'].create_optimized_passage(
                                passage_info.get('query', ''), context
                            )
                            st.text_area(f"Generated Passage {i}", value=passage, height=200)
        
        # Channel Strategy
        st.subheader("📱 Channel Marketing Strategy")
        channel_strategy = recommendations.get('channel_strategy', 'No strategy generated')
        st.write(channel_strategy)
        
        # Structure Improvements
        st.subheader("🏗️ Content Structure Improvements")
        structure = recommendations.get('structure_improvements', [])
        if isinstance(structure, list):
            for improvement in structure:
                st.write(f"• {improvement}")
        else:
            st.write(structure)
            
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Recommendations", use_container_width=True):
                reco_json = json.dumps(recommendations, indent=2)
                st.download_button(
                    label="Download Recommendations JSON",
                    data=reco_json,
                    file_name=f"content_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Generate Content Plan", use_container_width=True):
                st.info("Content planning feature coming soon!")
        
        with col3:
            if st.button("🔄 Re-analyze", use_container_width=True):
                st.info("Re-run analysis from the Analysis tab")
    else:
        st.info("👈 Run analysis with embeddings enabled to see recommendations")

with tab6:
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