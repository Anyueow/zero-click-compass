"""
XAI Content Optimizer - Uses XAI to optimize content for maximum query coverage.
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


class XAIContentOptimizer:
    """XAI-powered content optimization and gap analysis."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        api_key = get_env_var("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
    
    def _build_optimization_prompt(self, content: str, queries: List[Dict[str, Any]], 
                                 target_queries: List[str] = None) -> str:
        """Build XAI prompt for content optimization analysis."""
        
        # Prepare query context
        query_context = []
        for i, query_data in enumerate(queries[:20], 1):  # Limit to top 20 for context
            query_context.append(f"{i}. {query_data.get('query_text', '')} (Type: {query_data.get('query_type', 'unknown')})")
        
        query_context_str = "\n".join(query_context)
        
        # Prepare target queries context
        target_context = ""
        if target_queries:
            target_context = f"\nTarget Queries to Check:\n" + "\n".join([f"- {q}" for q in target_queries])
        
        return f"""
        You are an expert content optimization analyst using XAI (Explainable AI) to help content creators maximize their AI visibility and query coverage.

        CONTENT TO ANALYZE:
        {content[:3000]}  # Limit content length for API

        QUERIES THIS CONTENT ANSWERS:
        {query_context_str}
        {target_context}

        TASK: Provide a comprehensive XAI analysis with the following sections:

        1. CONTENT COVERAGE ANALYSIS:
           - What aspects of the topic are well-covered?
           - What specific queries does this content answer effectively?
           - Identify the content's strengths in addressing user intent

        2. CONTENT GAP ANALYSIS:
           - What aspects are missing or underdeveloped?
           - What queries are NOT being answered?
           - Identify opportunities for expansion

        3. AI VISIBILITY OPTIMIZATION:
           - How can this content be optimized to answer more sub-queries?
           - What additional sections or content should be added?
           - How to structure content for better AI comprehension?

        4. QUERY EXPANSION OPPORTUNITIES:
           - What related queries should this content address?
           - How to broaden the content's scope without losing focus?
           - Specific sub-topics to include

        5. CONTENT STRUCTURE RECOMMENDATIONS:
           - How to reorganize content for better query coverage?
           - What headings, sections, or formats would improve AI visibility?
           - Specific content additions needed

        Return a structured JSON response with these sections. Be specific and actionable.
        Focus on practical recommendations that can be implemented immediately.
        """

    @retry_on_failure(max_retries=3, delay=2.0)
    def analyze_content_optimization(self, content: str, queries: List[Dict[str, Any]], 
                                   target_queries: List[str] = None) -> Dict[str, Any]:
        """Analyze content optimization opportunities using XAI."""
        
        prompt = self._build_optimization_prompt(content, queries, target_queries)
        
        try:
            response = self.model_instance.generate_content(prompt)
            analysis_data = self._parse_optimization_response(response.text)
            
            result = {
                'content_preview': content[:500] + '...' if len(content) > 500 else content,
                'total_queries_analyzed': len(queries),
                'target_queries_checked': len(target_queries) if target_queries else 0,
                'analysis': analysis_data,
                'generated_at': datetime.now().isoformat(),
                'model_used': self.model
            }
            
            logger.info(f"XAI optimization analysis completed for content")
            return result
            
        except Exception as e:
            logger.error(f"Error in XAI optimization analysis: {e}")
            return {
                'error': f"Analysis failed: {e}",
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                'generated_at': datetime.now().isoformat()
            }
    
    def _parse_optimization_response(self, response_text: str) -> Dict[str, Any]:
        """Parse XAI response into structured analysis data."""
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
                return self._extract_analysis_manually(response_text)
            
            analysis = json.loads(json_str)
            
            # Ensure all required sections exist
            required_sections = [
                'content_coverage_analysis', 'content_gap_analysis', 
                'ai_visibility_optimization', 'query_expansion_opportunities',
                'content_structure_recommendations'
            ]
            
            for section in required_sections:
                if section not in analysis:
                    analysis[section] = f"Section {section} not found in AI response"
            
            return analysis
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse XAI response as JSON: {e}")
            return self._extract_analysis_manually(response_text)
    
    def _extract_analysis_manually(self, response_text: str) -> Dict[str, Any]:
        """Manual extraction of analysis from XAI response."""
        sections = {
            'content_coverage_analysis': '',
            'content_gap_analysis': '',
            'ai_visibility_optimization': '',
            'query_expansion_opportunities': '',
            'content_structure_recommendations': ''
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if 'content coverage' in line.lower():
                current_section = 'content_coverage_analysis'
            elif 'content gap' in line.lower():
                current_section = 'content_gap_analysis'
            elif 'ai visibility' in line.lower():
                current_section = 'ai_visibility_optimization'
            elif 'query expansion' in line.lower():
                current_section = 'query_expansion_opportunities'
            elif 'content structure' in line.lower():
                current_section = 'content_structure_recommendations'
            elif current_section and line:
                sections[current_section] += line + ' '
        
        return sections
    
    def optimize_content_for_queries(self, chunks: List[Dict[str, Any]], 
                                   queries: List[Dict[str, Any]],
                                   target_queries: List[str] = None,
                                   output_file: str = "data/xai_optimization.jsonl") -> Dict[str, Any]:
        """Optimize all content chunks for maximum query coverage."""
        create_data_dir()
        
        all_optimizations = []
        total_queries = len(queries)
        
        logger.info(f"Running XAI optimization for {len(chunks)} chunks against {total_queries} queries...")
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('content', '')
            if not content:
                continue
                
            logger.info(f"Optimizing chunk {i}/{len(chunks)} from {chunk.get('url', 'unknown')}")
            
            # Get queries relevant to this chunk (simplified - could be enhanced with semantic matching)
            relevant_queries = queries[:20]  # Top 20 for now
            
            optimization = self.analyze_content_optimization(content, relevant_queries, target_queries)
            
            # Add chunk metadata
            optimization['chunk_id'] = chunk.get('id', f'chunk_{i}')
            optimization['url'] = chunk.get('url', '')
            optimization['title'] = chunk.get('title', '')
            
            all_optimizations.append(optimization)
            
            # Save progress incrementally
            if i % 5 == 0:
                save_jsonl(all_optimizations, output_file)
                logger.info(f"Saved progress: {i} chunks optimized")
        
        # Final save
        save_jsonl(all_optimizations, output_file)
        
        summary = {
            'total_chunks_optimized': len(all_optimizations),
            'total_queries_analyzed': total_queries,
            'target_queries_checked': len(target_queries) if target_queries else 0,
            'output_file': output_file,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"XAI optimization complete: {summary}")
        return summary


class XAIOptimizationAnalyzer:
    """Analyze and summarize XAI optimization results."""
    
    def __init__(self):
        self.optimizer = XAIContentOptimizer()
    
    def analyze_optimization_results(self, optimization_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze optimization results across all content."""
        if not optimization_results:
            return {'error': 'No optimization results to analyze'}
        
        # Aggregate analysis across all chunks
        coverage_insights = []
        gap_insights = []
        optimization_insights = []
        expansion_insights = []
        structure_insights = []
        
        for result in optimization_results:
            analysis = result.get('analysis', {})
            
            if isinstance(analysis, dict):
                coverage_insights.append(analysis.get('content_coverage_analysis', ''))
                gap_insights.append(analysis.get('content_gap_analysis', ''))
                optimization_insights.append(analysis.get('ai_visibility_optimization', ''))
                expansion_insights.append(analysis.get('query_expansion_opportunities', ''))
                structure_insights.append(analysis.get('content_structure_recommendations', ''))
        
        # Find common patterns
        common_gaps = self._find_common_patterns(gap_insights)
        common_optimizations = self._find_common_patterns(optimization_insights)
        common_expansions = self._find_common_patterns(expansion_insights)
        
        analysis = {
            'total_chunks_analyzed': len(optimization_results),
            'common_content_gaps': common_gaps,
            'common_optimizations': common_optimizations,
            'common_expansions': common_expansions,
            'optimization_priority': self._prioritize_optimizations(optimization_insights),
            'coverage_strengths': self._identify_strengths(coverage_insights)
        }
        
        return analysis
    
    def _find_common_patterns(self, insights: List[str]) -> List[str]:
        """Find common patterns across optimization insights."""
        # Simple keyword-based pattern detection
        # In a real implementation, this could use more sophisticated NLP
        common_patterns = []
        
        # Look for common optimization keywords
        optimization_keywords = [
            'missing', 'gap', 'add', 'include', 'expand', 'improve',
            'enhance', 'optimize', 'structure', 'organize', 'clarify'
        ]
        
        for keyword in optimization_keywords:
            count = sum(1 for insight in insights if keyword.lower() in insight.lower())
            if count > len(insights) * 0.3:  # If 30%+ mention this pattern
                common_patterns.append(f"Common pattern: {keyword} (mentioned in {count}/{len(insights)} chunks)")
        
        return common_patterns
    
    def _prioritize_optimizations(self, optimizations: List[str]) -> List[str]:
        """Prioritize optimization recommendations."""
        # Simple priority scoring based on urgency keywords
        priority_keywords = {
            'high': ['critical', 'essential', 'must', 'urgent', 'important'],
            'medium': ['should', 'recommend', 'consider', 'improve'],
            'low': ['nice to have', 'optional', 'enhancement']
        }
        
        priorities = {'high': [], 'medium': [], 'low': []}
        
        for optimization in optimizations:
            priority = 'medium'  # default
            for level, keywords in priority_keywords.items():
                if any(keyword in optimization.lower() for keyword in keywords):
                    priority = level
                    break
            
            priorities[priority].append(optimization[:200] + '...' if len(optimization) > 200 else optimization)
        
        return priorities
    
    def _identify_strengths(self, coverage_insights: List[str]) -> List[str]:
        """Identify content strengths from coverage analysis."""
        strength_keywords = ['well', 'good', 'strong', 'effective', 'comprehensive', 'thorough']
        strengths = []
        
        for insight in coverage_insights:
            if any(keyword in insight.lower() for keyword in strength_keywords):
                strengths.append(insight[:200] + '...' if len(insight) > 200 else insight)
        
        return strengths[:5]  # Top 5 strengths 