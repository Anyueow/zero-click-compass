"""
Query Fan-out Generator - Expands queries into comprehensive fan-out sets.
Follows SOLID principles and integrates with the existing architecture.
"""
import json
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from datetime import datetime

from .utils import (
    get_env_var, retry_on_failure, create_data_dir, save_jsonl, 
    load_jsonl, logger, sanitize_text
)


class QueryFanoutGenerator:
    """Generate comprehensive query fan-outs from base queries."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        api_key = get_env_var("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
    
    def _build_fanout_prompt(self, query: str, mode: str = "AI Overview (simple)") -> str:
        """Build the fan-out prompt with detailed Chain-of-Thought logic."""
        min_queries_simple = 10
        min_queries_complex = 20

        if mode == "AI Overview (simple)":
            num_queries_instruction = (
                f"First, analyze the user's query: \"{query}\". Based on its complexity and the '{mode}' mode, "
                f"**you must decide on an optimal number of queries to generate.** "
                f"This number must be **at least {min_queries_simple}**. "
                f"For a straightforward query, generating around {min_queries_simple}-{min_queries_simple + 2} queries might be sufficient. "
                f"If the query has a few distinct aspects or common follow-up questions, aim for a slightly higher number, perhaps {min_queries_simple + 3}-{min_queries_simple + 5} queries. "
                f"Provide a brief reasoning for why you chose this specific number of queries. The queries themselves should be tightly scoped and highly relevant."
            )
        else:  # AI Mode (complex)
            num_queries_instruction = (
                f"First, analyze the user's query: \"{query}\". Based on its complexity and the '{mode}' mode, "
                f"**you must decide on an optimal number of queries to generate.** "
                f"This number must be **at least {min_queries_complex}**. "
                f"For multifaceted queries requiring exploration of various angles, sub-topics, comparisons, or deeper implications, "
                f"you should generate a more comprehensive set, potentially {min_queries_complex + 5}-{min_queries_complex + 10} queries, or even more if the query is exceptionally broad or deep. "
                f"Provide a brief reasoning for why you chose this specific number of queries. The queries should be diverse and in-depth."
            )

        return (
            f"You are simulating Google's AI Mode query fan-out process for generative search systems.\n"
            f"The user's original query is: \"{query}\". The selected mode is: \"{mode}\".\n\n"
            f"**Your first task is to determine the total number of queries to generate and the reasoning for this number, based on the instructions below:**\n"
            f"{num_queries_instruction}\n\n"
            f"**Once you have decided on the number and the reasoning, generate exactly that many unique synthetic queries.**\n"
            "Each of the following query transformation types MUST be represented at least once in the generated set, if the total number of queries you decide to generate allows for it (e.g., if you generate 12 queries, try to include all 6 types at least once, and then add more of the relevant types):\n"
            "1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n5. Entity Expansions\n6. Personalized Queries\n\n"
            "The 'reasoning' field for each *individual query* should explain why that specific query was generated in relation to the original query, its type, and the overall user intent.\n"
            "Do NOT include queries dependent on real-time user history or geolocation.\n\n"
            "Return only a valid JSON object. The JSON object should strictly follow this format:\n"
            "{\n"
            "  \"generation_details\": {\n"
            "    \"target_query_count\": 12, // This is an EXAMPLE number; you will DETERMINE the actual number based on your analysis.\n"
            "    \"reasoning_for_count\": \"The user query was moderately complex, so I chose to generate slightly more than the minimum for a simple overview to cover key aspects like X, Y, and Z.\" // This is an EXAMPLE reasoning; provide your own.\n"
            "  },\n"
            "  \"expanded_queries\": [\n"
            "    // Array of query objects. The length of this array MUST match your 'target_query_count'.\n"
            "    {\n"
            "      \"query\": \"Example query 1...\",\n"
            "      \"type\": \"reformulation\",\n"
            "      \"user_intent\": \"Example intent...\",\n"
            "      \"reasoning\": \"Example reasoning for this specific query...\"\n"
            "    },\n"
            "    // ... more query objects ...\n"
            "  ]\n"
            "}"
        )
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def generate_fanout(self, query: str, mode: str = "AI Overview (simple)") -> Dict[str, Any]:
        """Generate a comprehensive fan-out for a single query."""
        prompt = self._build_fanout_prompt(query, mode)
        
        try:
            response = self.model_instance.generate_content(prompt)
            json_text = response.text.strip()
            
            # Clean potential markdown code block fences
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            data = json.loads(json_text)
            generation_details = data.get("generation_details", {})
            expanded_queries = data.get("expanded_queries", [])

            result = {
                'original_query': query,
                'mode': mode,
                'generation_details': generation_details,
                'expanded_queries': expanded_queries,
                'query_count': len(expanded_queries),
                'generated_at': datetime.now().isoformat(),
                'model_used': self.model
            }
            
            logger.info(f"Generated {len(expanded_queries)} fan-out queries for: {query}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Raw response: {json_text if 'json_text' in locals() else 'N/A'}")
            return {
                'original_query': query,
                'mode': mode,
                'error': f"JSON parsing error: {e}",
                'expanded_queries': [],
                'query_count': 0,
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Unexpected error during fan-out generation: {e}")
            return {
                'original_query': query,
                'mode': mode,
                'error': f"Generation error: {e}",
                'expanded_queries': [],
                'query_count': 0,
                'generated_at': datetime.now().isoformat()
            }
    
    def generate_fanout_for_queries(self, queries: List[str], 
                                  mode: str = "AI Overview (simple)",
                                  output_file: str = "data/query_fanout.jsonl") -> Dict[str, Any]:
        """Generate fan-outs for multiple queries."""
        create_data_dir()
        
        all_results = []
        total_expanded_queries = 0
        
        logger.info(f"Generating fan-outs for {len(queries)} queries in {mode} mode...")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            
            result = self.generate_fanout(query, mode)
            all_results.append(result)
            total_expanded_queries += result['query_count']
            
            # Save progress incrementally
            if i % 5 == 0:
                save_jsonl(all_results, output_file)
                logger.info(f"Saved progress: {i} queries processed, {total_expanded_queries} expanded queries generated")
        
        # Final save
        save_jsonl(all_results, output_file)
        
        summary = {
            'total_original_queries': len(queries),
            'total_expanded_queries': total_expanded_queries,
            'average_expansion_ratio': total_expanded_queries / len(queries) if queries else 0,
            'mode_used': mode,
            'output_file': output_file,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Fan-out generation complete: {summary}")
        return summary
    
    def flatten_fanout_queries(self, fanout_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten all expanded queries into a single list for scoring."""
        flattened_queries = []
        
        for result in fanout_results:
            original_query = result.get('original_query', '')
            for expanded in result.get('expanded_queries', []):
                flattened_query = {
                    'query_text': expanded.get('query', ''),
                    'original_query': original_query,
                    'query_type': expanded.get('type', 'unknown'),
                    'user_intent': expanded.get('user_intent', ''),
                    'reasoning': expanded.get('reasoning', ''),
                    'fanout_id': f"{original_query}_{expanded.get('type', 'unknown')}",
                    'generated_at': result.get('generated_at', '')
                }
                if flattened_query['query_text']:
                    flattened_queries.append(flattened_query)
        
        return flattened_queries


class FanoutAnalyzer:
    """Analyze and optimize query fan-outs."""
    
    def __init__(self):
        self.fanout_generator = QueryFanoutGenerator()
    
    def analyze_fanout_coverage(self, fanout_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the coverage and distribution of fan-out queries."""
        if not fanout_results:
            return {'error': 'No fan-out results to analyze'}
        
        # Flatten all queries
        all_queries = []
        query_types = {}
        user_intents = {}
        
        for result in fanout_results:
            original_query = result.get('original_query', '')
            expanded_queries = result.get('expanded_queries', [])
            
            for query in expanded_queries:
                query_type = query.get('type', 'unknown')
                user_intent = query.get('user_intent', 'unknown')
                
                query_types[query_type] = query_types.get(query_type, 0) + 1
                user_intents[user_intent] = user_intents.get(user_intent, 0) + 1
                all_queries.append(query)
        
        analysis = {
            'total_original_queries': len(fanout_results),
            'total_expanded_queries': len(all_queries),
            'average_expansion_per_query': len(all_queries) / len(fanout_results) if fanout_results else 0,
            'query_type_distribution': query_types,
            'user_intent_distribution': user_intents,
            'top_query_types': sorted(query_types.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_user_intents': sorted(user_intents.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return analysis
    
    def optimize_fanout_strategy(self, fanout_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provide optimization recommendations for fan-out strategy."""
        analysis = self.analyze_fanout_coverage(fanout_results)
        
        recommendations = []
        
        # Analyze query type distribution
        query_types = analysis.get('query_type_distribution', {})
        if query_types:
            # Check for balanced distribution
            total_queries = sum(query_types.values())
            avg_per_type = total_queries / len(query_types)
            
            for query_type, count in query_types.items():
                if count < avg_per_type * 0.5:  # Significantly underrepresented
                    recommendations.append({
                        'type': 'query_type_balance',
                        'message': f"Query type '{query_type}' is underrepresented. Consider generating more of this type.",
                        'priority': 'medium'
                    })
        
        # Analyze expansion ratio
        expansion_ratio = analysis.get('average_expansion_per_query', 0)
        if expansion_ratio < 8:
            recommendations.append({
                'type': 'expansion_depth',
                'message': f"Average expansion ratio is {expansion_ratio:.1f}. Consider using 'AI Mode (complex)' for deeper exploration.",
                'priority': 'low'
            })
        elif expansion_ratio > 25:
            recommendations.append({
                'type': 'expansion_breadth',
                'message': f"Very high expansion ratio ({expansion_ratio:.1f}). Consider using 'AI Overview (simple)' for more focused queries.",
                'priority': 'medium'
            })
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'recommendation_count': len(recommendations)
        } 