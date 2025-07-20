"""
Command-line interface for the zero-click-compass pipeline.
"""
import argparse
import sys
import os
from typing import List, Optional

from .crawl import WebCrawler, crawl_single_page, crawl_multiple_pages
from .chunk import chunk_crawled_pages, SemanticChunker
from .embed import embed_and_index, load_existing_index, EmbeddingPipeline
from .expand import expand_query_simple, generate_intent_tree_simple, QueryExpander
from .score import score_query_chunks, analyze_content_performance_simple
from .channels import gather_social_chatter, analyze_social_impact
from .query_generator import ReverseQueryGenerator, QueryAnalyzer
from .query_fanout import QueryFanoutGenerator, FanoutAnalyzer
from .comprehensive_scorer import ComprehensiveContentScorer
from .xai_optimizer import XAIContentOptimizer, XAIOptimizationAnalyzer
from .channel_analyzer import ChannelAnalyzer, ChannelStrategyGenerator
from .utils import logger, create_data_dir, load_jsonl
import pandas as pd
import json
from datetime import datetime

def crawl_command(args):
    """Handle crawl command."""
    logger.info("Starting web crawling...")
    
    if args.single:
        # Crawl single page
        page = crawl_single_page(args.url)
        if page:
            logger.info(f"Successfully crawled: {args.url}")
        else:
            logger.error(f"Failed to crawl: {args.url}")
            sys.exit(1)
    else:
        # Crawl website
        crawler = WebCrawler(max_pages=args.max_pages, delay=args.delay)
        try:
            pages = crawler.crawl_website(args.url, max_depth=args.depth)
            logger.info(f"Successfully crawled {len(pages)} pages")
        finally:
            crawler.close()

def chunk_command(args):
    """Handle chunk command."""
    logger.info("Starting content chunking...")
    
    chunks = chunk_crawled_pages(
        input_file=args.input,
        output_file=args.output,
        use_semantic=not args.no_semantic
    )
    
    logger.info(f"Created {len(chunks)} chunks")

def embed_command(args):
    """Handle embed command."""
    logger.info("Starting embedding and indexing...")
    
    faiss_index = embed_and_index(
        input_file=args.input,
        output_prefix=args.output
    )
    
    if faiss_index:
        logger.info("Successfully created FAISS index")
    else:
        logger.error("Failed to create FAISS index")
        sys.exit(1)

def expand_command(args):
    """Handle expand command."""
    logger.info("Starting query expansion...")
    
    if args.intent_tree:
        expanded_queries = generate_intent_tree_simple(args.query)
    else:
        expanded_queries = expand_query_simple(args.query, args.max_expansions)
    
    logger.info(f"Generated {len(expanded_queries)} expanded queries")
    
    # Print expanded queries
    for i, query in enumerate(expanded_queries, 1):
        print(f"{i}. {query}")

def search_command(args):
    """Handle search command."""
    logger.info("Starting similarity search...")
    
    # Load FAISS index
    try:
        faiss_index = load_existing_index(args.index)
    except FileNotFoundError:
        logger.error(f"Index not found: {args.index}")
        sys.exit(1)
    
    # Search
    similar_chunks = faiss_index.search_similar(args.query, args.top_k)
    
    logger.info(f"Found {len(similar_chunks)} similar chunks")
    
    # Print results
    for i, chunk in enumerate(similar_chunks, 1):
        print(f"\n{i}. Score: {chunk['similarity_score']:.3f}")
        print(f"   URL: {chunk['url']}")
        print(f"   Content: {chunk['content'][:200]}...")

def score_command(args):
    """Handle score command."""
    logger.info("Starting content scoring...")
    
    # Load chunks
    data_dir = create_data_dir()
    chunks_path = os.path.join(data_dir, args.chunks)
    chunks = load_jsonl(chunks_path)
    
    if not chunks:
        logger.error(f"No chunks found in {chunks_path}")
        sys.exit(1)
    
    # Score chunks
    scored_chunks = score_query_chunks(args.query, chunks, args.top_k)
    
    logger.info(f"Scored {len(scored_chunks)} chunks")
    
    # Print results
    for i, result in enumerate(scored_chunks, 1):
        print(f"\n{i}. Composite Score: {result['scores']['composite']:.3f}")
        print(f"   Semantic: {result['scores']['semantic']:.3f}")
        print(f"   Keyword: {result['scores']['keyword']:.3f}")
        print(f"   URL: {result['url']}")
        print(f"   Content: {result['chunk_content'][:150]}...")

def social_command(args):
    """Handle social media command."""
    logger.info("Starting social media analysis...")
    
    social_data = gather_social_chatter(args.query, args.platforms)
    
    if args.analyze:
        analysis = analyze_social_impact(social_data)
        logger.info(f"Analyzed {analysis['total_content']} pieces of content")
        
        # Print top influencers
        print("\nTop Influencers:")
        for i, influencer in enumerate(analysis['top_influencers'][:5], 1):
            print(f"{i}. {influencer['username']} ({influencer['platform']}) - {influencer['engagement']:.0f} engagement")
    
    logger.info("Social media analysis completed")

def reverse_command(args):
    """Handle reverse query generation command."""
    logger.info("Starting reverse query generation...")
    
    # Load chunks
    try:
        chunks = load_jsonl(args.chunks_file)
        logger.info(f"Loaded {len(chunks)} chunks from {args.chunks_file}")
    except FileNotFoundError:
        logger.error(f"Chunks file not found: {args.chunks_file}")
        sys.exit(1)
    
    # Initialize query generator
    query_generator = ReverseQueryGenerator()
    
    # Generate queries for all chunks
    logger.info("Generating queries from content chunks...")
    summary = query_generator.generate_queries_for_all_chunks(chunks, args.output)
    
    print(f"\n=== Reverse Query Generation Results ===")
    print(f"Chunks processed: {summary['total_chunks_processed']}")
    print(f"Queries generated: {summary['total_queries_generated']}")
    print(f"Average queries per chunk: {summary['average_queries_per_chunk']:.1f}")
    print(f"Output file: {summary['output_file']}")
    
    # Analyze if requested
    if args.analyze:
        logger.info("Analyzing generated queries...")
        analyzer = QueryAnalyzer()
        
        # Load generated queries
        generated_queries = load_jsonl(args.output)
        
        # Analyze coverage
        coverage_analysis = query_generator.analyze_query_coverage(generated_queries)
        print(f"\n=== Query Coverage Analysis ===")
        print(f"Total queries: {coverage_analysis['total_queries']}")
        print(f"Unique chunks: {coverage_analysis['unique_chunks']}")
        print(f"Average relevance score: {coverage_analysis['average_relevance_score']:.1f}/10")
        
        print(f"\nTop categories:")
        for category, count in coverage_analysis['top_categories']:
            print(f"  {category}: {count}")
        
        print(f"\nTop intents:")
        for intent, count in coverage_analysis['top_intents']:
            print(f"  {intent}: {count}")
        
        # Find content gaps if target queries provided
        if args.target_queries:
            logger.info("Finding content gaps...")
            gap_analysis = analyzer.find_content_gaps(generated_queries, args.target_queries)
            print(f"\n=== Content Gap Analysis ===")
            print(f"Target queries analyzed: {gap_analysis['target_queries_analyzed']}")
            print(f"Content gaps found: {gap_analysis['gap_count']}")
            
            for gap in gap_analysis['content_gaps']:
                print(f"  ❌ {gap['target_query']} (priority: {gap['priority']})")
        
        # Get optimization recommendations
        optimization = analyzer.optimize_content_strategy(generated_queries)
        print(f"\n=== Content Optimization Recommendations ===")
        print(f"Recommendations: {optimization['recommendation_count']}")
        
        for rec in optimization['recommendations']:
            priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"  {priority_icon} {rec['message']}")
    
    logger.info("Reverse query generation completed!")

def fanout_command(args):
    """Handle fan-out generation command."""
    logger.info("Starting query fan-out generation...")
    
    # Initialize fan-out generator
    fanout_generator = QueryFanoutGenerator()
    
    # Generate fan-outs for all queries
    logger.info(f"Generating fan-outs for {len(args.queries)} queries in {args.mode} mode...")
    summary = fanout_generator.generate_fanout_for_queries(args.queries, args.mode, args.output)
    
    print(f"\n=== Query Fan-out Generation Results ===")
    print(f"Original queries: {summary['total_original_queries']}")
    print(f"Expanded queries: {summary['total_expanded_queries']}")
    print(f"Average expansion ratio: {summary['average_expansion_ratio']:.1f}")
    print(f"Mode used: {summary['mode_used']}")
    print(f"Output file: {summary['output_file']}")
    
    # Analyze if requested
    if args.analyze:
        logger.info("Analyzing fan-out results...")
        analyzer = FanoutAnalyzer()
        
        # Load fan-out results
        fanout_results = load_jsonl(args.output)
        
        # Analyze coverage
        coverage_analysis = analyzer.analyze_fanout_coverage(fanout_results)
        print(f"\n=== Fan-out Coverage Analysis ===")
        print(f"Total expanded queries: {coverage_analysis['total_expanded_queries']}")
        print(f"Average expansion per query: {coverage_analysis['average_expansion_per_query']:.1f}")
        
        print(f"\nQuery type distribution:")
        for query_type, count in coverage_analysis['top_query_types']:
            print(f"  {query_type}: {count}")
        
        print(f"\nUser intent distribution:")
        for intent, count in coverage_analysis['top_user_intents']:
            print(f"  {intent}: {count}")
        
        # Get optimization recommendations
        optimization = analyzer.optimize_fanout_strategy(fanout_results)
        print(f"\n=== Fan-out Optimization Recommendations ===")
        print(f"Recommendations: {optimization['recommendation_count']}")
        
        for rec in optimization['recommendations']:
            priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"  {priority_icon} {rec['message']}")
    
    logger.info("Query fan-out generation completed!")

def comprehensive_command(args):
    """Handle comprehensive scoring command."""
    logger.info("Starting comprehensive content scoring...")
    
    # Initialize comprehensive scorer
    scorer = ComprehensiveContentScorer()
    
    # Run comprehensive analysis
    results = scorer.run_comprehensive_analysis(
        chunks_file=args.chunks_file,
        fanout_file=args.fanout_file,
        top_k=args.top_k,
        output_prefix=args.output_prefix
    )
    
    if 'error' in results:
        logger.error(f"Comprehensive analysis failed: {results['error']}")
        sys.exit(1)
    
    # Display results
    analysis_summary = results['analysis_summary']
    print(f"\n=== Comprehensive Content Analysis Results ===")
    print(f"Chunks analyzed: {analysis_summary['chunks_analyzed']}")
    print(f"Queries used: {analysis_summary['queries_used']}")
    print(f"Top results: {analysis_summary['top_results']}")
    print(f"Average score: {analysis_summary['average_score']:.3f}")
    print(f"High relevance matches: {analysis_summary['high_relevance_matches']}")
    
    # Show top results
    scoring_results = results['scoring_results']
    top_results = scoring_results.get('top_results', [])
    
    print(f"\n=== Top {len(top_results)} Results ===")
    for i, result in enumerate(top_results, 1):
        print(f"\n{i}. Total Score: {result['total_score']:.3f}")
        print(f"   URL: {result['url']}")
        print(f"   Title: {result['title']}")
        print(f"   Query Matches: {result['query_matches']}")
        print(f"   High Relevance: {result['high_relevance_matches']}")
        print(f"   Average Score: {result['average_score']:.3f}")
        
        if result['top_queries']:
            print(f"   Top Queries:")
            for j, query in enumerate(result['top_queries'][:3], 1):
                print(f"     {j}. {query['query_text'][:60]}... (Score: {query['score']:.3f})")
    
    # Show report files
    report_files = results['report_files']
    print(f"\n=== Generated Reports ===")
    for report_type, file_path in report_files.items():
        print(f"  {report_type}: {file_path}")
    
    logger.info("Comprehensive content scoring completed!")

def xai_optimize_command(args):
    """Handle XAI optimization command."""
    logger.info("Starting XAI-powered content optimization...")
    
    # Load chunks and queries
    try:
        chunks = load_jsonl(args.chunks_file)
        logger.info(f"Loaded {len(chunks)} chunks from {args.chunks_file}")
    except FileNotFoundError:
        logger.error(f"Chunks file not found: {args.chunks_file}")
        sys.exit(1)
    
    try:
        fanout_results = load_jsonl(args.queries_file)
        logger.info(f"Loaded {len(fanout_results)} fan-out results from {args.queries_file}")
    except FileNotFoundError:
        logger.error(f"Queries file not found: {args.queries_file}")
        sys.exit(1)
    
    # Flatten fan-out queries
    from .query_fanout import QueryFanoutGenerator
    fanout_generator = QueryFanoutGenerator()
    flattened_queries = fanout_generator.flatten_fanout_queries(fanout_results)
    logger.info(f"Flattened to {len(flattened_queries)} individual queries")
    
    # Initialize XAI optimizer
    optimizer = XAIContentOptimizer()
    
    # Run optimization
    summary = optimizer.optimize_content_for_queries(
        chunks=chunks,
        queries=flattened_queries,
        target_queries=args.target_queries,
        output_file=args.output
    )
    
    print(f"\n=== XAI Content Optimization Results ===")
    print(f"Chunks optimized: {summary['total_chunks_optimized']}")
    print(f"Queries analyzed: {summary['total_queries_analyzed']}")
    print(f"Target queries checked: {summary['target_queries_checked']}")
    print(f"Output file: {summary['output_file']}")
    
    # Analyze if requested
    if args.analyze:
        logger.info("Analyzing XAI optimization results...")
        analyzer = XAIOptimizationAnalyzer()
        
        # Load optimization results
        optimization_results = load_jsonl(args.output)
        
        # Analyze results
        analysis = analyzer.analyze_optimization_results(optimization_results)
        
        print(f"\n=== XAI Optimization Analysis ===")
        print(f"Total chunks analyzed: {analysis['total_chunks_analyzed']}")
        
        if analysis.get('common_content_gaps'):
            print(f"\nCommon Content Gaps:")
            for gap in analysis['common_content_gaps'][:5]:
                print(f"  • {gap}")
        
        if analysis.get('common_optimizations'):
            print(f"\nCommon Optimizations:")
            for opt in analysis['common_optimizations'][:5]:
                print(f"  • {opt}")
        
        if analysis.get('optimization_priority'):
            priorities = analysis['optimization_priority']
            print(f"\nOptimization Priorities:")
            for priority_level, items in priorities.items():
                if items:
                    print(f"  {priority_level.upper()}:")
                    for item in items[:3]:
                        print(f"    • {item}")
    
    logger.info("XAI content optimization completed!")

def channel_analyze_command(args):
    """Handle channel analysis command."""
    logger.info("Starting channel analysis...")
    
    # Load queries
    try:
        fanout_results = load_jsonl(args.queries_file)
        logger.info(f"Loaded {len(fanout_results)} fan-out results from {args.queries_file}")
    except FileNotFoundError:
        logger.error(f"Queries file not found: {args.queries_file}")
        sys.exit(1)
    
    # Flatten fan-out queries
    from .query_fanout import QueryFanoutGenerator
    fanout_generator = QueryFanoutGenerator()
    flattened_queries = fanout_generator.flatten_fanout_queries(fanout_results)
    logger.info(f"Flattened to {len(flattened_queries)} individual queries")
    
    # Initialize channel analyzer
    analyzer = ChannelAnalyzer()
    
    # Run channel analysis
    summary = analyzer.analyze_queries_for_channels(
        queries=flattened_queries,
        output_file=args.output
    )
    
    print(f"\n=== Channel Analysis Results ===")
    print(f"Queries analyzed: {summary['total_queries_analyzed']}")
    print(f"Output file: {summary['output_file']}")
    
    # Generate strategy if requested
    if args.generate_strategy:
        logger.info("Generating comprehensive channel strategy...")
        strategy_generator = ChannelStrategyGenerator()
        
        # Load channel analyses
        channel_analyses = load_jsonl(args.output)
        
        # Generate strategy
        strategy = strategy_generator.generate_channel_strategy(channel_analyses)
        
        print(f"\n=== Channel Strategy ===")
        print(f"Total queries analyzed: {strategy['total_queries_analyzed']}")
        
        if strategy.get('channel_distribution'):
            print(f"\nChannel Distribution:")
            for platform, queries in strategy['channel_distribution'].items():
                print(f"  {platform.upper()}: {len(queries)} queries")
        
        if strategy.get('platform_strategies'):
            print(f"\nPlatform Strategies:")
            for platform, platform_data in strategy['platform_strategies'].items():
                print(f"  {platform.upper()}:")
                print(f"    Priority: {platform_data['priority']}")
                print(f"    Query count: {platform_data['query_count']}")
                print(f"    Engagement type: {platform_data['strategy']['engagement_type']}")
        
        if strategy.get('implementation_priority'):
            print(f"\nImplementation Priority:")
            for priority in strategy['implementation_priority'][:5]:
                print(f"  {priority['implementation_order']}. {priority['platform'].upper()} "
                      f"(Score: {priority['priority_score']}, Focus: {priority['recommended_focus']})")
        
        if strategy.get('agentic_engagement_roadmap'):
            roadmap = strategy['agentic_engagement_roadmap']
            print(f"\nAgentic Engagement Roadmap:")
            for timeline, opportunities in roadmap.items():
                if opportunities:
                    print(f"  {timeline.upper()}:")
                    for opportunity in opportunities[:3]:
                        print(f"    • {opportunity}")
        
        # Save strategy
        strategy_file = args.output.replace('.jsonl', '_strategy.json')
        with open(strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        print(f"\nStrategy saved to: {strategy_file}")
    
    logger.info("Channel analysis completed!")

def generate_visibility_csv(top_results, expanded_queries, original_query):
    """Generate visibility.csv with content performance data."""
    data_dir = create_data_dir()
    
    # Prepare data for CSV
    rows = []
    for i, result in enumerate(top_results, 1):
        row = {
            'rank': i,
            'url': result['url'],
            'similarity_score': result['similarity_score'],
            'content_preview': result['content'][:200] + '...' if len(result['content']) > 200 else result['content'],
            'chunk_id': result['id'],
            'original_query': original_query,
            'expanded_queries_count': len(expanded_queries),
            'timestamp': datetime.now().isoformat()
        }
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    csv_path = os.path.join(data_dir, 'visibility.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved visibility.csv to {csv_path}")

def generate_channels_json(social_data, analysis, original_query):
    """Generate channels.json with social media analysis data."""
    data_dir = create_data_dir()
    
    # Prepare data for JSON
    channels_data = {
        'query': original_query,
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_content': analysis['total_content'],
            'high_engagement_content': len(analysis['high_engagement_content']),
            'top_influencers_count': len(analysis['top_influencers'])
        },
        'platforms': social_data['platforms'],
        'top_influencers': analysis['top_influencers'][:10],  # Top 10
        'high_engagement_content': analysis['high_engagement_content'][:20]  # Top 20
    }
    
    # Save JSON
    json_path = os.path.join(data_dir, 'channels.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(channels_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved channels.json to {json_path}")

def pipeline_command(args):
    """Handle full pipeline command."""
    logger.info("Starting full zero-click-compass pipeline...")
    
    # Step 1: Crawl
    logger.info("Step 1: Crawling website...")
    crawler = WebCrawler(max_pages=args.max_pages, delay=args.delay)
    try:
        pages = crawler.crawl_website(args.url, max_depth=args.depth)
        logger.info(f"Crawled {len(pages)} pages")
    finally:
        crawler.close()
    
    # Step 2: Chunk
    logger.info("Step 2: Chunking content...")
    chunks = chunk_crawled_pages(use_semantic=not args.no_semantic)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 3: Embed and Index
    logger.info("Step 3: Creating embeddings and index...")
    faiss_index = embed_and_index()
    if not faiss_index:
        logger.error("Failed to create index")
        sys.exit(1)
    logger.info("Created FAISS index")
    
    # Step 4: Query Expansion
    logger.info("Step 4: Expanding queries...")
    expanded_queries = expand_query_simple(args.query, args.max_expansions)
    logger.info(f"Generated {len(expanded_queries)} expanded queries")
    
    # Step 5: Search and Score
    logger.info("Step 5: Searching and scoring...")
    all_results = []
    for query in expanded_queries:
        similar_chunks = faiss_index.search_similar(query, args.top_k)
        all_results.extend(similar_chunks)
    
    # Remove duplicates and get top results
    seen_chunks = set()
    unique_results = []
    for chunk in all_results:
        chunk_id = chunk['id']
        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            unique_results.append(chunk)
    
    # Sort by similarity score
    unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    top_results = unique_results[:args.top_k]
    
    logger.info(f"Found {len(top_results)} top results")
    
    # Step 6: Social Media (optional)
    if args.social:
        logger.info("Step 6: Analyzing social media...")
        social_data = gather_social_chatter(args.query)
        analysis = analyze_social_impact(social_data)
        logger.info(f"Found {analysis['total_content']} social media mentions")
    
    # Step 6: Social Media (optional)
    social_data = None
    analysis = None
    if args.social:
        logger.info("Step 6: Analyzing social media...")
        social_data = gather_social_chatter(args.query)
        analysis = analyze_social_impact(social_data)
        logger.info(f"Found {analysis['total_content']} social media mentions")
    
    # Generate visibility.csv
    logger.info("Generating visibility.csv...")
    generate_visibility_csv(top_results, expanded_queries, args.query)
    
    # Generate channels.json
    if social_data and analysis:
        logger.info("Generating channels.json...")
        generate_channels_json(social_data, analysis, args.query)
    
    # Step 7: Reverse Query Generation (optional)
    if hasattr(args, 'reverse_queries') and args.reverse_queries:
        logger.info("Step 7: Generating reverse queries from chunks...")
        query_generator = ReverseQueryGenerator()
        reverse_summary = query_generator.generate_queries_for_all_chunks(chunks, 'data/reverse_queries.jsonl')
        logger.info(f"Generated {reverse_summary['total_queries_generated']} reverse queries")
        
        # Analyze reverse queries
        analyzer = QueryAnalyzer()
        generated_queries = load_jsonl('data/reverse_queries.jsonl')
        coverage_analysis = query_generator.analyze_query_coverage(generated_queries)
        
        print(f"\n=== Reverse Query Analysis ===")
        print(f"Total queries your content answers: {coverage_analysis['total_queries']}")
        print(f"Average relevance score: {coverage_analysis['average_relevance_score']:.1f}/10")
        
        # Get optimization recommendations
        optimization = analyzer.optimize_content_strategy(generated_queries)
        if optimization['recommendations']:
            print(f"\nContent Optimization Recommendations:")
            for rec in optimization['recommendations']:
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                print(f"  {priority_icon} {rec['message']}")
    
    # Step 8: Query Fan-out Generation (optional)
    if hasattr(args, 'fanout_queries') and args.fanout_queries:
        logger.info("Step 8: Generating query fan-outs...")
        fanout_generator = QueryFanoutGenerator()
        fanout_summary = fanout_generator.generate_fanout_for_queries(
            args.fanout_queries, 
            args.fanout_mode, 
            'data/query_fanout.jsonl'
        )
        logger.info(f"Generated {fanout_summary['total_expanded_queries']} fan-out queries")
        
        print(f"\n=== Query Fan-out Results ===")
        print(f"Original queries: {fanout_summary['total_original_queries']}")
        print(f"Expanded queries: {fanout_summary['total_expanded_queries']}")
        print(f"Average expansion ratio: {fanout_summary['average_expansion_ratio']:.1f}")
    
    # Step 9: Comprehensive Scoring (optional)
    if hasattr(args, 'comprehensive_scoring') and args.comprehensive_scoring:
        logger.info("Step 9: Running comprehensive content scoring...")
        scorer = ComprehensiveContentScorer()
        
        # Check if fan-out file exists
        fanout_file = "data/query_fanout.jsonl"
        if not os.path.exists(fanout_file):
            logger.warning("No fan-out file found. Skipping comprehensive scoring.")
        else:
            comprehensive_results = scorer.run_comprehensive_analysis(
                chunks_file="data/chunks.jsonl",
                fanout_file=fanout_file,
                top_k=args.top_k,
                output_prefix="data/comprehensive"
            )
            
            if 'error' not in comprehensive_results:
                analysis_summary = comprehensive_results['analysis_summary']
                print(f"\n=== Comprehensive Scoring Results ===")
                print(f"Chunks analyzed: {analysis_summary['chunks_analyzed']}")
                print(f"Queries used: {analysis_summary['queries_used']}")
                print(f"Average score: {analysis_summary['average_score']:.3f}")
                print(f"High relevance matches: {analysis_summary['high_relevance_matches']}")
                
                # Show report files
                report_files = comprehensive_results['report_files']
                print(f"\nGenerated reports:")
                for report_type, file_path in report_files.items():
                    print(f"  {report_type}: {file_path}")
    
    # Step 10: XAI Content Optimization (optional)
    if hasattr(args, 'xai_optimize') and args.xai_optimize:
        logger.info("Step 10: Running XAI content optimization...")
        optimizer = XAIContentOptimizer()
        
        # Check if fan-out file exists
        fanout_file = "data/query_fanout.jsonl"
        if not os.path.exists(fanout_file):
            logger.warning("No fan-out file found. Skipping XAI optimization.")
        else:
            # Flatten fan-out queries
            fanout_generator = QueryFanoutGenerator()
            fanout_results = load_jsonl(fanout_file)
            flattened_queries = fanout_generator.flatten_fanout_queries(fanout_results)
            
            # Run optimization
            optimization_summary = optimizer.optimize_content_for_queries(
                chunks=chunks,
                queries=flattened_queries,
                output_file="data/xai_optimization.jsonl"
            )
            
            print(f"\n=== XAI Optimization Results ===")
            print(f"Chunks optimized: {optimization_summary['total_chunks_optimized']}")
            print(f"Queries analyzed: {optimization_summary['total_queries_analyzed']}")
            
            # Analyze results
            analyzer = XAIOptimizationAnalyzer()
            optimization_results = load_jsonl("data/xai_optimization.jsonl")
            analysis = analyzer.analyze_optimization_results(optimization_results)
            
            if analysis.get('common_content_gaps'):
                print(f"\nTop Content Gaps:")
                for gap in analysis['common_content_gaps'][:3]:
                    print(f"  • {gap}")
    
    # Step 11: Channel Analysis and Strategy (optional)
    if hasattr(args, 'channel_analyze') and args.channel_analyze:
        logger.info("Step 11: Running channel analysis and strategy...")
        analyzer = ChannelAnalyzer()
        
        # Check if fan-out file exists
        fanout_file = "data/query_fanout.jsonl"
        if not os.path.exists(fanout_file):
            logger.warning("No fan-out file found. Skipping channel analysis.")
        else:
            # Flatten fan-out queries
            fanout_generator = QueryFanoutGenerator()
            fanout_results = load_jsonl(fanout_file)
            flattened_queries = fanout_generator.flatten_fanout_queries(fanout_results)
            
            # Run channel analysis
            channel_summary = analyzer.analyze_queries_for_channels(
                queries=flattened_queries,
                output_file="data/channel_analysis.jsonl"
            )
            
            print(f"\n=== Channel Analysis Results ===")
            print(f"Queries analyzed: {channel_summary['total_queries_analyzed']}")
            
            # Generate strategy
            strategy_generator = ChannelStrategyGenerator()
            channel_analyses = load_jsonl("data/channel_analysis.jsonl")
            strategy = strategy_generator.generate_channel_strategy(channel_analyses)
            
            if strategy.get('channel_distribution'):
                print(f"\nChannel Distribution:")
                for platform, queries in strategy['channel_distribution'].items():
                    print(f"  {platform.upper()}: {len(queries)} queries")
            
            if strategy.get('implementation_priority'):
                print(f"\nTop Implementation Priorities:")
                for priority in strategy['implementation_priority'][:3]:
                    print(f"  {priority['implementation_order']}. {priority['platform'].upper()} "
                          f"(Score: {priority['priority_score']})")
    
    # Print results
    print(f"\n=== Zero-Click Compass Results for: {args.query} ===")
    print(f"Original Query: {args.query}")
    print(f"Expanded Queries: {len(expanded_queries)}")
    print(f"Top Results: {len(top_results)}")
    
    for i, result in enumerate(top_results, 1):
        print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
        print(f"   URL: {result['url']}")
        print(f"   Content: {result['content'][:200]}...")
    
    logger.info("Pipeline completed successfully!")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-Click Compass - LLM-first website performance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl a website
  python -m src.cli crawl https://example.com --max-pages 50
  
  # Run full pipeline
  python -m src.cli pipeline https://example.com "marketing strategies"
  
  # Search existing index
  python -m src.cli search "content marketing tips"
  
  # Analyze social media
  python -m src.cli social "SEO best practices" --analyze
  
  # Generate reverse queries from chunks
  python -m src.cli reverse --analyze --target-queries "marketing tips" "SEO guide"
  
  # Generate query fan-outs
  python -m src.cli fanout --queries "content marketing" "SEO strategies" --mode "AI Mode (complex)" --analyze
  
  # Run comprehensive scoring
  python -m src.cli comprehensive --top-k 15
  
  # XAI content optimization
  python -m src.cli xai-optimize --analyze --target-queries "best mattress" "sleep quality"
  
  # Channel analysis and strategy
  python -m src.cli channel-analyze --generate-strategy
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Crawl websites')
    crawl_parser.add_argument('url', help='URL to crawl')
    crawl_parser.add_argument('--single', action='store_true', help='Crawl single page only')
    crawl_parser.add_argument('--max-pages', type=int, default=100, help='Maximum pages to crawl')
    crawl_parser.add_argument('--depth', type=int, default=2, help='Crawl depth')
    crawl_parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests')
    crawl_parser.set_defaults(func=crawl_command)
    
    # Chunk command
    chunk_parser = subparsers.add_parser('chunk', help='Chunk crawled content')
    chunk_parser.add_argument('--input', default='crawled_pages.jsonl', help='Input file')
    chunk_parser.add_argument('--output', default='chunks.jsonl', help='Output file')
    chunk_parser.add_argument('--no-semantic', action='store_true', help='Disable semantic chunking')
    chunk_parser.set_defaults(func=chunk_command)
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Create embeddings and index')
    embed_parser.add_argument('--input', default='chunks.jsonl', help='Input file')
    embed_parser.add_argument('--output', default='faiss_index', help='Output prefix')
    embed_parser.set_defaults(func=embed_command)
    
    # Expand command
    expand_parser = subparsers.add_parser('expand', help='Expand queries')
    expand_parser.add_argument('query', help='Query to expand')
    expand_parser.add_argument('--intent-tree', action='store_true', help='Use intent tree expansion')
    expand_parser.add_argument('--max-expansions', type=int, default=20, help='Maximum expansions')
    expand_parser.set_defaults(func=expand_command)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search similar content')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--index', default='faiss_index', help='Index name')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    search_parser.set_defaults(func=search_command)
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Score content relevance')
    score_parser.add_argument('query', help='Query to score against')
    score_parser.add_argument('--chunks', default='chunks.jsonl', help='Chunks file')
    score_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    score_parser.set_defaults(func=score_command)
    
    # Social command
    social_parser = subparsers.add_parser('social', help='Analyze social media')
    social_parser.add_argument('query', help='Query to search for')
    social_parser.add_argument('--platforms', nargs='+', default=['reddit', 'twitter'], 
                              help='Platforms to search')
    social_parser.add_argument('--analyze', action='store_true', help='Analyze impact')
    social_parser.set_defaults(func=social_command)
    
    # Reverse query generation command
    reverse_parser = subparsers.add_parser('reverse', help='Generate queries from content chunks')
    reverse_parser.add_argument('--chunks-file', default='data/chunks.jsonl', help='Input chunks file')
    reverse_parser.add_argument('--output', default='data/generated_queries.jsonl', help='Output file for generated queries')
    reverse_parser.add_argument('--analyze', action='store_true', help='Analyze generated queries')
    reverse_parser.add_argument('--target-queries', nargs='+', help='Target queries to check for gaps')
    reverse_parser.set_defaults(func=reverse_command)
    
    # Fan-out generation command
    fanout_parser = subparsers.add_parser('fanout', help='Generate query fan-outs from base queries')
    fanout_parser.add_argument('--queries', nargs='+', required=True, help='Base queries to expand')
    fanout_parser.add_argument('--mode', choices=['AI Overview (simple)', 'AI Mode (complex)'], 
                              default='AI Overview (simple)', help='Fan-out mode')
    fanout_parser.add_argument('--output', default='data/query_fanout.jsonl', help='Output file for fan-out queries')
    fanout_parser.add_argument('--analyze', action='store_true', help='Analyze fan-out results')
    fanout_parser.set_defaults(func=fanout_command)
    
    # Comprehensive scoring command
    comprehensive_parser = subparsers.add_parser('comprehensive', help='Run comprehensive content scoring')
    comprehensive_parser.add_argument('--chunks-file', default='data/chunks.jsonl', help='Input chunks file')
    comprehensive_parser.add_argument('--fanout-file', default='data/query_fanout.jsonl', help='Input fan-out queries file')
    comprehensive_parser.add_argument('--top-k', type=int, default=10, help='Number of top results')
    comprehensive_parser.add_argument('--output-prefix', default='data/comprehensive', help='Output file prefix')
    comprehensive_parser.set_defaults(func=comprehensive_command)
    
    # XAI optimization command
    xai_parser = subparsers.add_parser('xai-optimize', help='XAI-powered content optimization')
    xai_parser.add_argument('--chunks-file', default='data/chunks.jsonl', help='Input chunks file')
    xai_parser.add_argument('--queries-file', default='data/query_fanout.jsonl', help='Input queries file')
    xai_parser.add_argument('--target-queries', nargs='+', help='Specific target queries to check')
    xai_parser.add_argument('--output', default='data/xai_optimization.jsonl', help='Output file')
    xai_parser.add_argument('--analyze', action='store_true', help='Analyze optimization results')
    xai_parser.set_defaults(func=xai_optimize_command)
    
    # Channel analysis command
    channel_parser = subparsers.add_parser('channel-analyze', help='Analyze query channels and engagement strategies')
    channel_parser.add_argument('--queries-file', default='data/query_fanout.jsonl', help='Input queries file')
    channel_parser.add_argument('--output', default='data/channel_analysis.jsonl', help='Output file')
    channel_parser.add_argument('--generate-strategy', action='store_true', help='Generate comprehensive channel strategy')
    channel_parser.set_defaults(func=channel_analyze_command)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('url', help='Website URL to analyze')
    pipeline_parser.add_argument('query', help='Query to analyze')
    pipeline_parser.add_argument('--max-pages', type=int, default=50, help='Maximum pages to crawl')
    pipeline_parser.add_argument('--depth', type=int, default=2, help='Crawl depth')
    pipeline_parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests')
    pipeline_parser.add_argument('--no-semantic', action='store_true', help='Disable semantic chunking')
    pipeline_parser.add_argument('--max-expansions', type=int, default=15, help='Maximum query expansions')
    pipeline_parser.add_argument('--top-k', type=int, default=10, help='Number of top results')
    pipeline_parser.add_argument('--social', action='store_true', help='Include social media analysis')
    pipeline_parser.add_argument('--reverse-queries', action='store_true', help='Generate reverse queries from chunks')
    pipeline_parser.add_argument('--fanout-queries', nargs='+', help='Generate fan-outs for these queries')
    pipeline_parser.add_argument('--fanout-mode', choices=['AI Overview (simple)', 'AI Mode (complex)'], 
                                default='AI Overview (simple)', help='Fan-out mode')
    pipeline_parser.add_argument('--comprehensive-scoring', action='store_true', help='Run comprehensive scoring')
    pipeline_parser.add_argument('--xai-optimize', action='store_true', help='Run XAI content optimization')
    pipeline_parser.add_argument('--channel-analyze', action='store_true', help='Run channel analysis and strategy')
    pipeline_parser.set_defaults(func=pipeline_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 