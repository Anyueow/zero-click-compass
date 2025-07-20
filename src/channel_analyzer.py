"""
Channel Analyzer - Analyzes where users ask specific queries and recommends engagement strategies.
Follows SOLID principles and integrates with the existing architecture.
"""
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from datetime import datetime

from .utils import (
    get_env_var, retry_on_failure, create_data_dir, save_jsonl, 
    load_jsonl, logger, sanitize_text
)


class ChannelAnalyzer:
    """Analyze where users ask specific queries and recommend engagement channels."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        api_key = get_env_var("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
    
    def _build_channel_analysis_prompt(self, query: str, user_intent: str = None) -> str:
        """Build prompt for channel analysis."""
        
        intent_context = f"User Intent: {user_intent}" if user_intent else "User Intent: Not specified"
        
        return f"""
        You are an expert in digital marketing and user behavior analysis. Your task is to analyze where users are most likely to ask specific queries and recommend engagement strategies.

        QUERY TO ANALYZE: "{query}"
        {intent_context}

        TASK: Provide a comprehensive channel analysis with the following sections:

        1. CHANNEL DISTRIBUTION ANALYSIS:
           - Where are users most likely to ask this query?
           - Rank platforms by likelihood (Reddit, X/Twitter, Google, Yelp, Quora, etc.)
           - Explain the reasoning for each platform's relevance

        2. USER BEHAVIOR PATTERNS:
           - What type of user asks this query on each platform?
           - What are their expectations and pain points?
           - How do they typically phrase this query on different platforms?

        3. ENGAGEMENT STRATEGY RECOMMENDATIONS:
           - For each relevant platform, provide specific engagement tactics
           - What type of content should be created for each platform?
           - How to position responses and content for maximum impact?

        4. CONTENT ADAPTATION STRATEGIES:
           - How should content be adapted for each platform?
           - What format works best for each channel?
           - Tone and style recommendations for each platform

        5. FUTURE AGENTIC ENGAGEMENT OPPORTUNITIES:
           - How could an AI agent engage with users on each platform?
           - What automated responses or content could be created?
           - Engagement timing and frequency recommendations

        Return a structured JSON response with these sections. Be specific and actionable.
        Focus on practical strategies that can be implemented immediately.
        Consider the personalized nature of future internet interactions.
        """

    @retry_on_failure(max_retries=3, delay=2.0)
    def analyze_query_channels(self, query: str, user_intent: str = None) -> Dict[str, Any]:
        """Analyze which channels are most relevant for a specific query."""
        
        prompt = self._build_channel_analysis_prompt(query, user_intent)
        
        try:
            response = self.model_instance.generate_content(prompt)
            analysis_data = self._parse_channel_response(response.text)
            
            result = {
                'query': query,
                'user_intent': user_intent,
                'analysis': analysis_data,
                'generated_at': datetime.now().isoformat(),
                'model_used': self.model
            }
            
            logger.info(f"Channel analysis completed for query: {query[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error in channel analysis: {e}")
            return {
                'query': query,
                'user_intent': user_intent,
                'error': f"Analysis failed: {e}",
                'generated_at': datetime.now().isoformat()
            }
    
    def _parse_channel_response(self, response_text: str) -> Dict[str, Any]:
        """Parse channel analysis response into structured data."""
        try:
            # Try to extract JSON from response
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_str = response_text[json_start:json_end].strip()
            elif '{' in response_text and '}' in response_text:
                # Try to find JSON object in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
            else:
                # Fallback: create structured response from text
                return self._extract_channel_analysis_manually(response_text)
            
            analysis = json.loads(json_str)
            
            # Ensure all required sections exist
            required_sections = [
                'channel_distribution_analysis', 'user_behavior_patterns',
                'engagement_strategy_recommendations', 'content_adaptation_strategies',
                'future_agentic_engagement_opportunities'
            ]
            
            for section in required_sections:
                if section not in analysis:
                    analysis[section] = f"Section {section} not found in AI response"
            
            return analysis
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse channel response as JSON: {e}")
            return self._extract_channel_analysis_manually(response_text)
    
    def _extract_channel_analysis_manually(self, response_text: str) -> Dict[str, Any]:
        """Manual extraction of channel analysis from response."""
        sections = {
            'channel_distribution_analysis': '',
            'user_behavior_patterns': '',
            'engagement_strategy_recommendations': '',
            'content_adaptation_strategies': '',
            'future_agentic_engagement_opportunities': ''
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if 'channel distribution' in line.lower():
                current_section = 'channel_distribution_analysis'
            elif 'user behavior' in line.lower():
                current_section = 'user_behavior_patterns'
            elif 'engagement strategy' in line.lower():
                current_section = 'engagement_strategy_recommendations'
            elif 'content adaptation' in line.lower():
                current_section = 'content_adaptation_strategies'
            elif 'future agentic' in line.lower() or 'agentic engagement' in line.lower():
                current_section = 'future_agentic_engagement_opportunities'
            elif current_section and line:
                sections[current_section] += line + ' '
        
        return sections
    
    def analyze_queries_for_channels(self, queries: List[Dict[str, Any]], 
                                   output_file: str = "data/channel_analysis.jsonl") -> Dict[str, Any]:
        """Analyze multiple queries for channel distribution and engagement strategies."""
        create_data_dir()
        
        all_analyses = []
        total_queries = len(queries)
        
        logger.info(f"Analyzing {total_queries} queries for channel distribution...")
        
        for i, query_data in enumerate(queries, 1):
            query_text = query_data.get('query_text', '')
            user_intent = query_data.get('user_intent', '')
            
            if not query_text:
                continue
                
            logger.info(f"Analyzing query {i}/{total_queries}: {query_text[:50]}...")
            
            analysis = self.analyze_query_channels(query_text, user_intent)
            
            # Add query metadata
            analysis['query_type'] = query_data.get('query_type', 'unknown')
            analysis['original_query'] = query_data.get('original_query', '')
            
            all_analyses.append(analysis)
            
            # Save progress incrementally
            if i % 10 == 0:
                save_jsonl(all_analyses, output_file)
                logger.info(f"Saved progress: {i} queries analyzed")
        
        # Final save
        save_jsonl(all_analyses, output_file)
        
        summary = {
            'total_queries_analyzed': len(all_analyses),
            'output_file': output_file,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Channel analysis complete: {summary}")
        return summary


class ChannelStrategyGenerator:
    """Generate comprehensive channel engagement strategies."""
    
    def __init__(self):
        self.channel_analyzer = ChannelAnalyzer()
    
    def generate_channel_strategy(self, channel_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive channel engagement strategy from analyses."""
        if not channel_analyses:
            return {'error': 'No channel analyses to process'}
        
        # Aggregate channel insights
        channel_insights = {}
        engagement_strategies = []
        content_adaptations = []
        agentic_opportunities = []
        
        for analysis in channel_analyses:
            analysis_data = analysis.get('analysis', {})
            
            if isinstance(analysis_data, dict):
                # Extract channel distribution
                channel_dist = analysis_data.get('channel_distribution_analysis', '')
                if channel_dist:
                    # Simple channel extraction (could be enhanced with NLP)
                    for platform in ['reddit', 'twitter', 'x', 'google', 'yelp', 'quora', 'facebook', 'linkedin']:
                        if platform in channel_dist.lower():
                            if platform not in channel_insights:
                                channel_insights[platform] = []
                            channel_insights[platform].append(analysis['query'])
                
                # Collect strategies
                engagement_strategies.append(analysis_data.get('engagement_strategy_recommendations', ''))
                content_adaptations.append(analysis_data.get('content_adaptation_strategies', ''))
                agentic_opportunities.append(analysis_data.get('future_agentic_engagement_opportunities', ''))
        
        # Generate platform-specific strategies
        platform_strategies = self._generate_platform_strategies(channel_insights)
        
        # Compile comprehensive strategy
        strategy = {
            'total_queries_analyzed': len(channel_analyses),
            'channel_distribution': channel_insights,
            'platform_strategies': platform_strategies,
            'common_engagement_patterns': self._find_common_patterns(engagement_strategies),
            'content_adaptation_insights': self._find_common_patterns(content_adaptations),
            'agentic_engagement_roadmap': self._prioritize_agentic_opportunities(agentic_opportunities),
            'implementation_priority': self._prioritize_implementation(channel_insights),
            'generated_at': datetime.now().isoformat()
        }
        
        return strategy
    
    def _generate_platform_strategies(self, channel_insights: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """Generate specific strategies for each platform."""
        platform_strategies = {}
        
        platform_configs = {
            'reddit': {
                'engagement_type': 'community_participation',
                'content_format': 'detailed_posts_and_comments',
                'tone': 'helpful_and_informative',
                'frequency': 'daily_engagement',
                'key_subreddits': 'relevant_communities'
            },
            'twitter': {
                'engagement_type': 'conversation_participation',
                'content_format': 'threads_and_replies',
                'tone': 'conversational_and_insightful',
                'frequency': 'multiple_times_daily',
                'hashtag_strategy': 'relevant_hashtags'
            },
            'x': {
                'engagement_type': 'conversation_participation',
                'content_format': 'threads_and_replies',
                'tone': 'conversational_and_insightful',
                'frequency': 'multiple_times_daily',
                'hashtag_strategy': 'relevant_hashtags'
            },
            'google': {
                'engagement_type': 'seo_optimization',
                'content_format': 'comprehensive_articles',
                'tone': 'authoritative_and_helpful',
                'frequency': 'regular_publishing',
                'keyword_strategy': 'long_tail_keywords'
            },
            'yelp': {
                'engagement_type': 'review_responses',
                'content_format': 'professional_responses',
                'tone': 'professional_and_helpful',
                'frequency': 'as_needed',
                'response_strategy': 'addressing_concerns'
            },
            'quora': {
                'engagement_type': 'expert_answers',
                'content_format': 'detailed_answers',
                'tone': 'expert_and_thorough',
                'frequency': 'weekly_engagement',
                'topic_focus': 'expertise_areas'
            }
        }
        
        for platform, queries in channel_insights.items():
            if platform in platform_configs:
                platform_strategies[platform] = {
                    'relevant_queries': queries[:10],  # Top 10 queries
                    'query_count': len(queries),
                    'strategy': platform_configs[platform],
                    'priority': 'high' if len(queries) > 5 else 'medium' if len(queries) > 2 else 'low'
                }
        
        return platform_strategies
    
    def _find_common_patterns(self, strategies: List[str]) -> List[str]:
        """Find common patterns across engagement strategies."""
        common_patterns = []
        
        # Look for common strategy keywords
        strategy_keywords = [
            'engage', 'respond', 'comment', 'post', 'share', 'create',
            'participate', 'contribute', 'help', 'answer', 'provide'
        ]
        
        for keyword in strategy_keywords:
            count = sum(1 for strategy in strategies if keyword.lower() in strategy.lower())
            if count > len(strategies) * 0.2:  # If 20%+ mention this pattern
                common_patterns.append(f"Common strategy: {keyword} (mentioned in {count}/{len(strategies)} analyses)")
        
        return common_patterns
    
    def _prioritize_agentic_opportunities(self, opportunities: List[str]) -> Dict[str, List[str]]:
        """Prioritize agentic engagement opportunities."""
        priorities = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        for opportunity in opportunities:
            if any(word in opportunity.lower() for word in ['automated', 'bot', 'ai agent', 'immediate']):
                priorities['immediate'].append(opportunity[:200] + '...' if len(opportunity) > 200 else opportunity)
            elif any(word in opportunity.lower() for word in ['future', 'advanced', 'complex']):
                priorities['long_term'].append(opportunity[:200] + '...' if len(opportunity) > 200 else opportunity)
            else:
                priorities['short_term'].append(opportunity[:200] + '...' if len(opportunity) > 200 else opportunity)
        
        return priorities
    
    def _prioritize_implementation(self, channel_insights: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Prioritize implementation order for channel strategies."""
        priorities = []
        
        for platform, queries in channel_insights.items():
            priority_score = len(queries)  # More queries = higher priority
            
            priorities.append({
                'platform': platform,
                'priority_score': priority_score,
                'query_count': len(queries),
                'implementation_order': len(priorities) + 1,
                'recommended_focus': 'high' if priority_score > 5 else 'medium' if priority_score > 2 else 'low'
            })
        
        # Sort by priority score
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Update implementation order
        for i, priority in enumerate(priorities):
            priority['implementation_order'] = i + 1
        
        return priorities 