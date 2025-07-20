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