"""
Comprehensive Content Scorer - Scores content against fan-out queries.
Follows SOLID principles and integrates with the existing architecture.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from .utils import (
    get_env_var, retry_on_failure, create_data_dir, save_jsonl, 
    load_jsonl, logger, sanitize_text
)
from .score import RelevanceScorer
from .query_fanout import QueryFanoutGenerator, FanoutAnalyzer


class ComprehensiveContentScorer:
    """Score content comprehensively against fan-out queries."""
    
    def __init__(self):
        self.relevance_scorer = RelevanceScorer()
        self.fanout_generator = QueryFanoutGenerator()
        self.fanout_analyzer = FanoutAnalyzer()
    
    def score_content_against_fanout(self, chunks: List[Dict[str, Any]], 
                                   fanout_queries: List[Dict[str, Any]],
                                   top_k: int = 10) -> Dict[str, Any]:
        """Score all content chunks against all fan-out queries."""
        logger.info(f"Scoring {len(chunks)} chunks against {len(fanout_queries)} fan-out queries...")
        
        all_scores = []
        chunk_scores = {}
        
        # Initialize chunk scores
        for chunk in chunks:
            chunk_id = chunk.get('id', 'unknown')
            chunk_scores[chunk_id] = {
                'chunk_id': chunk_id,
                'url': chunk.get('url', ''),
                'title': chunk.get('title', ''),
                'content': chunk.get('content', ''),
                'total_score': 0,
                'query_matches': 0,
                'high_relevance_matches': 0,
                'average_score': 0,
                'top_queries': [],
                'query_details': []
            }
        
        # Score each chunk against each query
        for i, query_data in enumerate(fanout_queries, 1):
            query_text = query_data.get('query_text', '')
            if not query_text:
                continue
                
            logger.info(f"Scoring query {i}/{len(fanout_queries)}: {query_text[:50]}...")
            
            for chunk in chunks:
                chunk_id = chunk.get('id', 'unknown')
                content = chunk.get('content', '')
                
                # Score this chunk-query pair
                score_result = self.relevance_scorer.score_query_chunk_pair(query_text, content)
                
                if score_result and score_result.get('composite_score', 0) > 0:
                    score_data = {
                        'chunk_id': chunk_id,
                        'query_text': query_text,
                        'original_query': query_data.get('original_query', ''),
                        'query_type': query_data.get('query_type', 'unknown'),
                        'user_intent': query_data.get('user_intent', ''),
                        'composite_score': score_result.get('composite_score', 0),
                        'semantic_score': score_result.get('semantic_score', 0),
                        'token_overlap_score': score_result.get('token_overlap_score', 0),
                        'url': chunk.get('url', ''),
                        'title': chunk.get('title', ''),
                        'content_preview': content[:200] + '...' if len(content) > 200 else content
                    }
                    
                    all_scores.append(score_data)
                    
                    # Update chunk summary
                    chunk_scores[chunk_id]['total_score'] += score_result.get('composite_score', 0)
                    chunk_scores[chunk_id]['query_matches'] += 1
                    
                    if score_result.get('composite_score', 0) > 0.7:  # High relevance threshold
                        chunk_scores[chunk_id]['high_relevance_matches'] += 1
                    
                    # Track top queries for this chunk
                    chunk_scores[chunk_id]['query_details'].append({
                        'query_text': query_text,
                        'score': score_result.get('composite_score', 0),
                        'query_type': query_data.get('query_type', 'unknown')
                    })
        
        # Calculate averages and sort
        for chunk_id, chunk_data in chunk_scores.items():
            if chunk_data['query_matches'] > 0:
                chunk_data['average_score'] = chunk_data['total_score'] / chunk_data['query_matches']
                
                # Sort query details by score
                chunk_data['query_details'].sort(key=lambda x: x['score'], reverse=True)
                chunk_data['top_queries'] = chunk_data['query_details'][:5]  # Top 5 queries
        
        # Sort chunks by total score
        sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x['total_score'], reverse=True)
        
        # Get top K results
        top_results = sorted_chunks[:top_k]
        
        # Calculate overall statistics
        total_queries = len(fanout_queries)
        total_chunks = len(chunks)
        total_scores = len(all_scores)
        
        if total_scores > 0:
            avg_score = sum(score['composite_score'] for score in all_scores) / total_scores
            high_relevance_count = sum(1 for score in all_scores if score['composite_score'] > 0.7)
        else:
            avg_score = 0
            high_relevance_count = 0
        
        summary = {
            'total_chunks_scored': total_chunks,
            'total_queries_used': total_queries,
            'total_score_pairs': total_scores,
            'average_score': avg_score,
            'high_relevance_matches': high_relevance_count,
            'top_k_results': top_k,
            'scored_at': datetime.now().isoformat()
        }
        
        return {
            'summary': summary,
            'top_results': top_results,
            'all_scores': all_scores,
            'chunk_scores': chunk_scores
        }
    
    def generate_comprehensive_report(self, scoring_results: Dict[str, Any], 
                                    output_prefix: str = "data/comprehensive") -> Dict[str, str]:
        """Generate comprehensive reports from scoring results."""
        create_data_dir()
        
        report_files = {}
        
        # 1. Top Results CSV
        top_results = scoring_results.get('top_results', [])
        if top_results:
            df_top = pd.DataFrame(top_results)
            # Clean up content column for CSV
            if 'content' in df_top.columns:
                df_top['content'] = df_top['content'].apply(lambda x: x[:500] + '...' if len(str(x)) > 500 else str(x))
            
            top_csv_path = f"{output_prefix}_top_results.csv"
            df_top.to_csv(top_csv_path, index=False)
            report_files['top_results_csv'] = top_csv_path
        
        # 2. All Scores CSV
        all_scores = scoring_results.get('all_scores', [])
        if all_scores:
            df_scores = pd.DataFrame(all_scores)
            scores_csv_path = f"{output_prefix}_all_scores.csv"
            df_scores.to_csv(scores_csv_path, index=False)
            report_files['all_scores_csv'] = scores_csv_path
        
        # 3. Summary JSON
        summary = scoring_results.get('summary', {})
        summary_json_path = f"{output_prefix}_summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        report_files['summary_json'] = summary_json_path
        
        # 4. Detailed Results JSONL
        detailed_results = []
        for chunk_data in scoring_results.get('top_results', []):
            detailed_result = {
                'chunk_id': chunk_data['chunk_id'],
                'url': chunk_data['url'],
                'title': chunk_data['title'],
                'total_score': chunk_data['total_score'],
                'average_score': chunk_data['average_score'],
                'query_matches': chunk_data['query_matches'],
                'high_relevance_matches': chunk_data['high_relevance_matches'],
                'top_queries': chunk_data['top_queries'],
                'content_preview': chunk_data['content'][:300] + '...' if len(chunk_data['content']) > 300 else chunk_data['content']
            }
            detailed_results.append(detailed_result)
        
        if detailed_results:
            detailed_jsonl_path = f"{output_prefix}_detailed_results.jsonl"
            save_jsonl(detailed_results, detailed_jsonl_path)
            report_files['detailed_jsonl'] = detailed_jsonl_path
        
        logger.info(f"Generated comprehensive reports: {list(report_files.keys())}")
        return report_files
    
    def run_comprehensive_analysis(self, chunks_file: str = "data/chunks.jsonl",
                                 fanout_file: str = "data/query_fanout.jsonl",
                                 top_k: int = 10,
                                 output_prefix: str = "data/comprehensive") -> Dict[str, Any]:
        """Run the complete comprehensive analysis pipeline."""
        logger.info("Starting comprehensive content analysis...")
        
        # Load chunks
        try:
            chunks = load_jsonl(chunks_file)
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        except FileNotFoundError:
            logger.error(f"Chunks file not found: {chunks_file}")
            return {'error': f'Chunks file not found: {chunks_file}'}
        
        # Load fan-out queries
        try:
            fanout_results = load_jsonl(fanout_file)
            logger.info(f"Loaded {len(fanout_results)} fan-out results from {fanout_file}")
        except FileNotFoundError:
            logger.error(f"Fan-out file not found: {fanout_file}")
            return {'error': f'Fan-out file not found: {fanout_file}'}
        
        # Flatten fan-out queries
        fanout_generator = QueryFanoutGenerator()
        flattened_queries = fanout_generator.flatten_fanout_queries(fanout_results)
        logger.info(f"Flattened to {len(flattened_queries)} individual queries")
        
        # Score content against fan-out queries
        scoring_results = self.score_content_against_fanout(chunks, flattened_queries, top_k)
        
        # Generate reports
        report_files = self.generate_comprehensive_report(scoring_results, output_prefix)
        
        # Analyze fan-out coverage
        fanout_analysis = self.fanout_analyzer.analyze_fanout_coverage(fanout_results)
        
        # Compile final results
        final_results = {
            'scoring_results': scoring_results,
            'fanout_analysis': fanout_analysis,
            'report_files': report_files,
            'analysis_summary': {
                'chunks_analyzed': len(chunks),
                'queries_used': len(flattened_queries),
                'top_results': len(scoring_results.get('top_results', [])),
                'average_score': scoring_results.get('summary', {}).get('average_score', 0),
                'high_relevance_matches': scoring_results.get('summary', {}).get('high_relevance_matches', 0),
                'completed_at': datetime.now().isoformat()
            }
        }
        
        logger.info("Comprehensive analysis completed successfully!")
        return final_results 