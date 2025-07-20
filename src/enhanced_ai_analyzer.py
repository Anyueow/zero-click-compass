#!/usr/bin/env python3
"""
Enhanced AI Analyzer for Zero-Click Compass
Uses XAI for comprehensive recommendations and Together API (Mistral 7B) for keyword/content suggestions.
"""

import os
import re
import json
import requests
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAIAnalyzer:
    """Enhanced AI analyzer using XAI and Mistral 7B for comprehensive recommendations."""
    
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.together_api_key = os.getenv('TOGETHER_API_KEY')
        self.together_api_url = "https://api.together.xyz/v1/chat/completions"
        
    def generate_comprehensive_recommendations(self, chunks: List[Dict], fanout_queries: List[Dict], 
                                             chunk_scores: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive XAI recommendations with chunk previews and Mistral 7B suggestions.
        """
        logger.info("🎯 Generating comprehensive AI recommendations...")
        
        results = {
            'xai_recommendations': {},
            'mistral_keywords': {},
            'chunk_improvements': [],
            'channel_recommendations': [],
            'content_gaps': {},
            'summary': {}
        }
        
        # Get top queries for analysis
        top_queries = sorted(fanout_queries, key=lambda x: x.get('best_score', 0), reverse=True)[:10]
        query_texts = [q.get('query', '') for q in top_queries]
        
        # Generate XAI recommendations
        results['xai_recommendations'] = self._generate_xai_recommendations(chunks, query_texts, chunk_scores)
        
        # Generate Mistral 7B keyword and content recommendations
        results['mistral_keywords'] = self._generate_mistral_recommendations(chunks, query_texts)
        
        # Generate chunk-specific improvements
        results['chunk_improvements'] = self._generate_chunk_improvements(chunks, chunk_scores, query_texts)
        
        # Generate channel recommendations
        results['channel_recommendations'] = self._generate_channel_recommendations(query_texts, chunk_scores)
        
        # Analyze content gaps
        results['content_gaps'] = self._analyze_content_gaps(chunks, query_texts)
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        logger.info("✅ Comprehensive AI recommendations generated")
        return results
    
    def _generate_xai_recommendations(self, chunks: List[Dict], query_texts: List[str], 
                                    chunk_scores: List[Dict]) -> Dict[str, Any]:
        """Generate XAI-based comprehensive recommendations."""
        
        # Prepare chunk data with previews
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            chunk_preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            
            # Find corresponding score
            chunk_score = next((score for score in chunk_scores if score.get('chunk_id') == chunk.get('id')), None)
            avg_score = chunk_score.get('avg_relevance_score', 0) if chunk_score else 0
            
            chunk_data.append({
                'id': chunk.get('id', f'chunk_{i}'),
                'url': chunk.get('url', ''),
                'preview': chunk_preview,
                'avg_score': avg_score,
                'grade': chunk_score.get('overall_grade', 'F') if chunk_score else 'F'
            })
        
        # Create XAI prompt
        xai_prompt = f"""
        Analyze these content chunks and provide comprehensive recommendations for improving their AI Overview (AIO) scores and context window performance.

        CONTENT CHUNKS:
        {json.dumps(chunk_data, indent=2)}

        TARGET QUERIES:
        {query_texts}

        Provide recommendations in this format:
        1. OVERALL ASSESSMENT: Brief summary of content performance
        2. CONTENT GAPS: Specific missing content patterns
        3. IMPROVEMENT STRATEGIES: How to improve AIO scores
        4. CONTEXT WINDOW OPTIMIZATION: How to perform better in AI overviews
        5. QUERY-SPECIFIC RECOMMENDATIONS: For queries with low scores

        Keep recommendations actionable and specific. Do not use markdown formatting.
        """
        
        # Call Google Gemini for XAI analysis
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.google_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(xai_prompt)
            xai_analysis = response.text
            
            # Clean markdown formatting
            xai_analysis = self._clean_markdown(xai_analysis)
            
            return {
                'analysis': xai_analysis,
                'chunks_analyzed': len(chunks),
                'queries_analyzed': len(query_texts)
            }
            
        except Exception as e:
            logger.error(f"Error generating XAI recommendations: {e}")
            return {
                'analysis': "XAI analysis unavailable. Please check API configuration.",
                'chunks_analyzed': len(chunks),
                'queries_analyzed': len(query_texts)
            }
    
    def _generate_mistral_recommendations(self, chunks: List[Dict], query_texts: List[str]) -> Dict[str, Any]:
        """Generate keyword and content recommendations using Mistral 7B via Together API."""
        
        if not self.together_api_key:
            return {
                'keywords': [],
                'content_suggestions': [],
                'error': 'Together API key not configured'
            }
        
        # Prepare prompt for Mistral
        mistral_prompt = f"""
        Analyze these content chunks and provide user-friendly recommendations for better AI Overview performance.

        CONTENT CHUNKS:
        {json.dumps([{'preview': chunk.get('text', '')[:300] + "..."} for chunk in chunks[:5]], indent=2)}

        TARGET QUERIES:
        {query_texts[:5]}

        Provide recommendations in this format:

        KEYWORDS TO ADD:
        List 5-8 specific keywords that would improve your content's relevance

        SAMPLE CONTENT PARAGRAPHS:
        Provide 2-3 short, engaging paragraphs (50-100 words each) that would help your content rank better

        QUICK IMPROVEMENT TIPS:
        List 3-4 specific, actionable tips for better AI Overview performance

        Keep everything simple, practical, and easy to understand. Do not use markdown formatting or technical jargon.
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [
                    {"role": "user", "content": mistral_prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(self.together_api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                mistral_analysis = result['choices'][0]['message']['content']
                
                # Clean markdown formatting
                mistral_analysis = self._clean_markdown(mistral_analysis)
                
                return {
                    'analysis': mistral_analysis,
                    'keywords': self._extract_keywords(mistral_analysis),
                    'content_suggestions': self._extract_content_suggestions(mistral_analysis)
                }
            else:
                logger.error(f"Together API error: {response.status_code}")
                return {
                    'analysis': "Mistral analysis unavailable. Please check API configuration.",
                    'keywords': [],
                    'content_suggestions': []
                }
                
        except Exception as e:
            logger.error(f"Error generating Mistral recommendations: {e}")
            return {
                'analysis': "Mistral analysis unavailable. Please check API configuration.",
                'keywords': [],
                'content_suggestions': []
            }
    
    def _generate_chunk_improvements(self, chunks: List[Dict], chunk_scores: List[Dict], 
                                   query_texts: List[str]) -> List[Dict]:
        """Generate specific improvements for each chunk."""
        
        improvements = []
        
        for chunk in chunks:
            chunk_id = chunk.get('id', '')
            chunk_text = chunk.get('text', '')
            chunk_preview = chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text
            
            # Find corresponding score
            chunk_score = next((score for score in chunk_scores if score.get('chunk_id') == chunk_id), None)
            avg_score = chunk_score.get('avg_relevance_score', 0) if chunk_score else 0
            
            # Generate improvement suggestions
            if avg_score < 0.3:
                priority = "high"
                suggestion = "Significant improvement needed. Add more relevant keywords and expand content."
            elif avg_score < 0.6:
                priority = "medium"
                suggestion = "Moderate improvement needed. Enhance keyword density and content relevance."
            else:
                priority = "low"
                suggestion = "Good performance. Minor optimizations recommended."
            
            improvements.append({
                'chunk_id': chunk_id,
                'url': chunk.get('url', ''),
                'preview': chunk_preview,
                'current_score': avg_score,
                'priority': priority,
                'suggestion': suggestion,
                'target_queries': query_texts[:3]  # Top 3 queries to target
            })
        
        return improvements
    
    def _generate_channel_recommendations(self, query_texts: List[str], chunk_scores: List[Dict]) -> List[Dict]:
        """Generate channel-specific recommendations."""
        
        # Analyze queries for channel fit
        channel_analysis = {
            'GOOGLE': {'queries': [], 'score': 0},
            'REDDIT': {'queries': [], 'score': 0},
            'TWITTER': {'queries': [], 'score': 0},
            'QUORA': {'queries': [], 'score': 0},
            'LINKEDIN': {'queries': [], 'score': 0},
            'YELP': {'queries': [], 'score': 0}
        }
        
        for query in query_texts:
            query_lower = query.lower()
            
            # Google/SEO - Most queries fit here as it's the primary search engine
            channel_analysis['GOOGLE']['queries'].append(query)
            channel_analysis['GOOGLE']['score'] += 1
            
            # Add bonus points for specific Google-friendly queries
            if any(word in query_lower for word in ['how to', 'what is', 'guide', 'tutorial', 'best', 'compare']):
                channel_analysis['GOOGLE']['score'] += 2
            
            # Reddit
            if any(word in query_lower for word in ['experience', 'review', 'recommend', 'community', 'opinion']):
                channel_analysis['REDDIT']['queries'].append(query)
                channel_analysis['REDDIT']['score'] += 2
            
            # Twitter
            if any(word in query_lower for word in ['trending', 'opinion', 'quick tip', 'news', 'latest']):
                channel_analysis['TWITTER']['queries'].append(query)
                channel_analysis['TWITTER']['score'] += 2
            
            # Quora
            if any(word in query_lower for word in ['what is', 'how does', 'why', 'explain', 'difference']):
                channel_analysis['QUORA']['queries'].append(query)
                channel_analysis['QUORA']['score'] += 2
            
            # LinkedIn
            if any(word in query_lower for word in ['industry', 'career', 'professional', 'business', 'strategy']):
                channel_analysis['LINKEDIN']['queries'].append(query)
                channel_analysis['LINKEDIN']['score'] += 2
            
            # Yelp
            if any(word in query_lower for word in ['review', 'rating', 'service', 'business', 'customer']):
                channel_analysis['YELP']['queries'].append(query)
                channel_analysis['YELP']['score'] += 2
        
        # Generate recommendations - include all platforms with at least some relevance
        recommendations = []
        for platform, data in channel_analysis.items():
            # Include Google even with low score as it's primary
            if platform == 'GOOGLE' or data['score'] > 0:
                focus_level = 'high' if data['score'] > 5 else 'medium' if data['score'] > 2 else 'low'
                
                # Ensure we have queries for each platform
                if not data['queries'] and platform != 'GOOGLE':
                    data['queries'] = query_texts[:3]  # Use top queries as fallback
                
                recommendations.append({
                    'platform': platform,
                    'score': data['score'],
                    'focus_level': focus_level,
                    'queries': data['queries'][:3],  # Top 3 queries
                    'strategy': self._get_channel_strategy(platform, data['queries'])
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations
    
    def _get_channel_strategy(self, platform: str, queries: List[str]) -> str:
        """Get platform-specific strategy."""
        strategies = {
            'GOOGLE': 'Create comprehensive, SEO-optimized content targeting long-tail keywords',
            'REDDIT': 'Engage in community discussions with detailed, helpful responses',
            'TWITTER': 'Participate in trending conversations with valuable insights',
            'QUORA': 'Provide expert, detailed answers to specific questions',
            'LINKEDIN': 'Share professional insights and industry expertise',
            'YELP': 'Respond professionally to reviews and engage with customers'
        }
        return strategies.get(platform, 'Focus on platform-specific content optimization')
    
    def _analyze_content_gaps(self, chunks: List[Dict], query_texts: List[str]) -> Dict[str, Any]:
        """Analyze content gaps across all chunks."""
        
        all_gaps = []
        gap_explanations = {
            'Missing how-to content': 'Your content lacks step-by-step guides and tutorials that users search for',
            'Missing definition content': 'Your content doesn\'t explain key terms and concepts clearly',
            'Missing best-of content': 'Your content doesn\'t include "best" lists, recommendations, or top picks that users love',
            'Missing comparison content': 'Your content doesn\'t compare options or help users choose between alternatives',
            'Missing review content': 'Your content lacks detailed reviews and evaluations',
            'Missing benefits content': 'Your content doesn\'t clearly explain the benefits and advantages',
            'Missing problem-solving content': 'Your content doesn\'t address common problems and solutions'
        }
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            
            for query in query_texts:
                query_lower = query.lower()
                
                # Check for missing content patterns
                if 'how to' in query_lower and 'how to' not in chunk_text:
                    all_gaps.append('Missing how-to content')
                if 'what is' in query_lower and 'what is' not in chunk_text:
                    all_gaps.append('Missing definition content')
                if 'best' in query_lower and 'best' not in chunk_text:
                    all_gaps.append('Missing best-of content')
                if 'compare' in query_lower and 'compare' not in chunk_text:
                    all_gaps.append('Missing comparison content')
                if 'review' in query_lower and 'review' not in chunk_text:
                    all_gaps.append('Missing review content')
                if 'benefit' in query_lower and 'benefit' not in chunk_text:
                    all_gaps.append('Missing benefits content')
                if 'problem' in query_lower and 'problem' not in chunk_text:
                    all_gaps.append('Missing problem-solving content')
        
        # Count gaps
        gap_counts = {}
        for gap in all_gaps:
            gap_counts[gap] = gap_counts.get(gap, 0) + 1
        
        # Create user-friendly gap analysis
        user_friendly_gaps = []
        for gap, count in sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            user_friendly_gaps.append({
                'gap_type': gap,
                'frequency': count,
                'percentage': f"{(count / len(all_gaps) * 100):.1f}%",
                'explanation': gap_explanations.get(gap, 'Content gap identified'),
                'suggestion': self._get_gap_suggestion(gap)
            })
        
        return {
            'total_gaps': len(all_gaps),
            'gap_distribution': gap_counts,
            'most_common_gaps': user_friendly_gaps
        }
    
    def _get_gap_suggestion(self, gap_type: str) -> str:
        """Get user-friendly suggestions for each gap type."""
        suggestions = {
            'Missing how-to content': 'Create step-by-step guides, tutorials, and instructional content',
            'Missing definition content': 'Add clear explanations of key terms and concepts',
            'Missing best-of content': 'Create "best" lists, top recommendations, and curated selections',
            'Missing comparison content': 'Compare different options and help users make informed choices',
            'Missing review content': 'Provide detailed reviews and evaluations of products/services',
            'Missing benefits content': 'Clearly explain the advantages and benefits of your offerings',
            'Missing problem-solving content': 'Address common problems and provide solutions'
        }
        return suggestions.get(gap_type, 'Add relevant content to address this gap')
    
    def _extract_keywords(self, analysis: str) -> List[str]:
        """Extract keywords from Mistral analysis."""
        keywords = []
        lines = analysis.split('\n')
        for line in lines:
            if 'keyword' in line.lower() or ':' in line:
                # Extract potential keywords
                parts = line.split(':')
                if len(parts) > 1:
                    keywords.extend([kw.strip() for kw in parts[1].split(',') if kw.strip()])
        return keywords[:10]  # Limit to 10 keywords
    
    def _extract_content_suggestions(self, analysis: str) -> List[str]:
        """Extract content suggestions from Mistral analysis."""
        suggestions = []
        lines = analysis.split('\n')
        for line in lines:
            if len(line.strip()) > 50 and not line.startswith(('1.', '2.', '3.')):
                suggestions.append(line.strip())
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        # Remove bold formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # Remove italic formatting
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        # Remove code formatting
        text = re.sub(r'`(.*?)`', r'\1', text)
        # Remove headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Remove list markers
        text = re.sub(r'^[\s]*[-*+]\s*', '', text, flags=re.MULTILINE)
        # Remove numbered lists
        text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary of all recommendations."""
        return {
            'total_chunks_analyzed': len(results.get('chunk_improvements', [])),
            'total_queries_analyzed': results.get('xai_recommendations', {}).get('queries_analyzed', 0),
            'channel_recommendations_count': len(results.get('channel_recommendations', [])),
            'content_gaps_count': results.get('content_gaps', {}).get('total_gaps', 0),
            'keywords_generated': len(results.get('mistral_keywords', {}).get('keywords', [])),
            'improvements_generated': len(results.get('chunk_improvements', []))
        } 