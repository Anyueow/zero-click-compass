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
from .query_generator import ReverseQueryGenerator, QueryAnalyzer
from .query_fanout import QueryFanoutGenerator, FanoutAnalyzer
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
    
    logger.info(f"Generated {summary['total_queries_generated']} queries")
    
    if args.analyze:
        # Analyze generated queries
        logger.info("Analyzing query coverage...")
        generated_queries = load_jsonl(args.output)
        coverage_analysis = query_generator.analyze_query_coverage(generated_queries)
        
        print(f"\n=== Query Coverage Analysis ===")
        print(f"Total queries your content answers: {coverage_analysis['total_queries']}")
        print(f"Average relevance score: {coverage_analysis['average_relevance_score']:.1f}/10")
        
        # Get optimization recommendations
        analyzer = QueryAnalyzer()
        optimization = analyzer.optimize_content_strategy(generated_queries)
        if optimization['recommendations']:
            print(f"\nContent Optimization Recommendations:")
            for rec in optimization['recommendations']:
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                print(f"  {priority_icon} {rec['message']}")

def fanout_command(args):
    """Handle query fan-out command."""
    logger.info("Starting query fan-out generation...")
    
    # Initialize fanout generator
    fanout_generator = QueryFanoutGenerator()
    
    # Generate fan-out for queries
    logger.info(f"Generating fan-out for {len(args.queries)} queries...")
    summary = fanout_generator.generate_fanout_for_queries(
        args.queries, 
        args.mode, 
        args.output
    )
    
    logger.info(f"Generated {summary['total_expanded_queries']} fan-out queries")
    
    print(f"\n=== Query Fan-out Results ===")
    print(f"Original queries: {summary['total_original_queries']}")
    print(f"Expanded queries: {summary['total_expanded_queries']}")
    print(f"Average expansion ratio: {summary['average_expansion_ratio']:.1f}")
    
    if args.analyze:
        # Analyze fan-out results
        logger.info("Analyzing fan-out coverage...")
        fanout_results = load_jsonl(args.output)
        analyzer = FanoutAnalyzer()
        coverage_analysis = analyzer.analyze_fanout_coverage(fanout_results)
        
        print(f"\n=== Fan-out Coverage Analysis ===")
        print(f"Query types covered: {len(coverage_analysis['query_types'])}")
        print(f"Average queries per original: {coverage_analysis['average_queries_per_original']:.1f}")
        
        # Get optimization recommendations
        optimization = analyzer.optimize_fanout_strategy(fanout_results)
        if optimization['recommendations']:
            print(f"\nFan-out Optimization Recommendations:")
            for rec in optimization['recommendations']:
                print(f"  • {rec}")

def generate_visibility_csv(top_results, expanded_queries, original_query):
    """Generate CSV with visibility scores."""
    data_dir = create_data_dir()
    csv_path = os.path.join(data_dir, "visibility.csv")
    
    # Create DataFrame
    data = []
    for i, result in enumerate(top_results, 1):
        data.append({
            'Rank': i,
            'Content': result.get('content', '')[:200] + '...',
            'URL': result.get('url', ''),
            'Similarity Score': f"{result.get('similarity_score', 0):.3f}",
            'Content Type': result.get('content_type', ''),
            'Tokens': result.get('tokens', 0)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved visibility.csv to {csv_path}")
    return csv_path

def pipeline_command(args):
    """Handle full zero-click-compass pipeline."""
    logger.info("Starting full zero-click-compass pipeline...")
    
    # Clear old cached data to ensure fresh processing
    logger.info("Clearing old cached data...")
    import os
    cached_files = [
        "data/chunks.jsonl",
        "data/embedded_chunks.jsonl", 
        "data/faiss_index.faiss",
        "data/faiss_index_chunks.jsonl",
        "data/reverse_queries.jsonl",
        "data/query_fanout.jsonl"
    ]
    for file_path in cached_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed cached file: {file_path}")
    
    # Step 1: Crawl (optimized for speed)
    logger.info("Step 1: Crawling website...")
    # Use faster crawling settings
    fast_delay = min(args.delay, 0.5)  # Cap delay at 0.5s for speed
    crawler = WebCrawler(max_pages=args.max_pages, delay=fast_delay)
    try:
        pages = crawler.crawl_website(args.url, max_depth=args.depth)
        logger.info(f"Crawled {len(pages)} pages in ~{fast_delay * len(pages) + 5:.1f}s")
    finally:
        crawler.close()
    
    # Step 2: Chunk (using fresh pages directly)
    logger.info("Step 2: Chunking content...")
    if not args.no_semantic:
        chunker = SemanticChunker(
            target_tokens=args.chunk_size,
            overlap_tokens=args.sliding_window
        )
        # Use fresh crawled pages directly
        chunks = chunker.chunk_pages(pages, max_chunks_per_page=5)
        
        # Also limit total chunks to max_chunks
        if len(chunks) > args.max_chunks:
            chunks = chunks[:args.max_chunks]
    else:
        chunks = chunk_crawled_pages(use_semantic=False)
        
        # Limit chunks to max_chunks
        if len(chunks) > args.max_chunks:
            chunks = chunks[:args.max_chunks]
    
    logger.info(f"Created {len(chunks)} chunks (limited to 5 per page)")
    
    # Step 3: Embed and Index (using fresh chunks directly)
    logger.info("Step 3: Creating embeddings and index...")
    embedding_pipeline = EmbeddingPipeline()
    faiss_index = embedding_pipeline.process_chunks(chunks, save_intermediate=False)
    if not faiss_index:
        logger.error("Failed to create index")
        sys.exit(1)
    logger.info("Created FAISS index")
    
    # Step 4: Query Expansion (Generate reverse queries from content) - OPTIMIZED
    logger.info("Step 4: Generating reverse queries from content...")
    query_generator = ReverseQueryGenerator()
    
    # Only process first 2 chunks for speed (or limit based on args)
    limited_chunks = chunks[:min(2, len(chunks))]
    logger.info(f"Processing {len(limited_chunks)} chunks for reverse queries (optimized for speed)")
    
    reverse_queries_summary = query_generator.generate_queries_for_all_chunks(
        limited_chunks, 
        output_file="data/reverse_queries.jsonl"
    )
    
    # Load the generated queries from file
    generated_queries = load_jsonl("data/reverse_queries.jsonl")
    
    # Get top 5 reverse queries for fan-out
    top_reverse_queries = []
    for chunk_result in generated_queries:
        chunk_queries = chunk_result.get('queries', [])
        # Sort by relevance score and take top queries
        sorted_queries = sorted(chunk_queries, key=lambda x: x.get('relevance_score', 0), reverse=True)
        top_reverse_queries.extend(sorted_queries[:args.max_reverse_queries])
    
    # Limit to top 5 reverse queries for fan-out
    top_reverse_queries = top_reverse_queries[:5]
    reverse_query_texts = [q.get('query_text', '') for q in top_reverse_queries if q.get('query_text')]
    
    logger.info(f"Generated {len(reverse_query_texts)} top reverse queries")
    
    # Step 5: Analysis and Expansion (Fan-out the reverse queries) - OPTIMIZED
    logger.info("Step 5: Expanding reverse queries with fan-out...")
    fanout_generator = QueryFanoutGenerator()
    all_fanout_queries = []
    
    # Only process first 2 reverse queries for speed
    limited_reverse_queries = reverse_query_texts[:min(2, len(reverse_query_texts))]
    logger.info(f"Processing {len(limited_reverse_queries)} reverse queries for fan-out (optimized for speed)")
    
    for reverse_query in limited_reverse_queries:
        fanout_queries = fanout_generator.generate_fanout(
            reverse_query, 
            mode="AI Overview (simple)"
        )
        # Extract just the query texts from the fanout result
        expanded_queries = fanout_queries.get('expanded_queries', [])
        query_texts = [q.get('query', '') for q in expanded_queries if q.get('query')]
        all_fanout_queries.extend(query_texts)
    
    # Use fan-out queries for search (not original query expansion)
    expanded_queries = all_fanout_queries
    
    logger.info(f"Generated {len(expanded_queries)} fan-out queries from {len(limited_reverse_queries)} reverse queries")
    
    # Step 6: Search and Score
    logger.info("Step 6: Searching and scoring...")
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
    
    # Step 7: Generate outputs
    logger.info("Step 7: Generating output files...")
    
    # Generate CSV with visibility scores
    csv_path = generate_visibility_csv(top_results, expanded_queries, args.query)
    logger.info(f"Generated visibility CSV: {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"URL: {args.url}")
    print(f"Query: {args.query}")
    print(f"Pages crawled: {len(pages)}")
    print(f"Chunks created: {len(chunks)} (limited to 5 per page)")
    print(f"Reverse queries generated: {len(reverse_query_texts)}")
    print(f"Fan-out queries created: {len(expanded_queries)}")
    print(f"Top results found: {len(top_results)}")
    print(f"Output files:")
    print(f"  - Visibility CSV: {csv_path}")
    print("="*60)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-Click Compass - Essential content analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m src.cli pipeline https://example.com "your query"
  
  # Individual steps
  python -m src.cli crawl https://example.com --max-pages 3
  python -m src.cli chunk --input crawled_pages.jsonl
  python -m src.cli embed --input chunks.jsonl
  python -m src.cli search "your query" --top-k 10
  python -m src.cli reverse --chunks-file chunks.jsonl
  python -m src.cli fanout --queries "query1" "query2"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Crawl website content')
    crawl_parser.add_argument('url', help='Website URL to crawl')
    crawl_parser.add_argument('--max-pages', type=int, default=3, help='Maximum pages to crawl')
    crawl_parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests')
    crawl_parser.add_argument('--depth', type=int, default=1, help='Crawl depth')
    crawl_parser.add_argument('--single', action='store_true', help='Crawl single page only')
    crawl_parser.set_defaults(func=crawl_command)
    
    # Chunk command
    chunk_parser = subparsers.add_parser('chunk', help='Chunk crawled content')
    chunk_parser.add_argument('--input', default='crawled_pages.jsonl', help='Input file')
    chunk_parser.add_argument('--output', default='chunks.jsonl', help='Output file')
    chunk_parser.add_argument('--no-semantic', action='store_true', help='Use basic chunking')
    chunk_parser.set_defaults(func=chunk_command)
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Create embeddings and index')
    embed_parser.add_argument('--input', default='chunks.jsonl', help='Input file')
    embed_parser.add_argument('--output', default='faiss_index', help='Output prefix')
    embed_parser.set_defaults(func=embed_command)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search similar content')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--index', default='faiss_index', help='Index name')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    search_parser.set_defaults(func=search_command)
    
    # Reverse command
    reverse_parser = subparsers.add_parser('reverse', help='Generate reverse queries')
    reverse_parser.add_argument('--chunks-file', required=True, help='Chunks file')
    reverse_parser.add_argument('--output', default='reverse_queries.jsonl', help='Output file')
    reverse_parser.add_argument('--analyze', action='store_true', help='Analyze results')
    reverse_parser.set_defaults(func=reverse_command)
    
    # Fanout command
    fanout_parser = subparsers.add_parser('fanout', help='Generate query fan-outs')
    fanout_parser.add_argument('--queries', nargs='+', required=True, help='Queries to expand')
    fanout_parser.add_argument('--mode', default='AI Overview (simple)', help='Fan-out mode')
    fanout_parser.add_argument('--output', default='query_fanout.jsonl', help='Output file')
    fanout_parser.add_argument('--analyze', action='store_true', help='Analyze results')
    fanout_parser.set_defaults(func=fanout_command)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('url', help='Website URL')
    pipeline_parser.add_argument('query', help='Target query')
    pipeline_parser.add_argument('--max-pages', type=int, default=3, help='Maximum pages')
    pipeline_parser.add_argument('--max-chunks', type=int, default=10, help='Maximum chunks')
    pipeline_parser.add_argument('--chunk-size', type=int, default=150, help='Chunk size in tokens')
    pipeline_parser.add_argument('--sliding-window', type=int, default=20, help='Sliding window overlap')
    pipeline_parser.add_argument('--max-reverse-queries', type=int, default=2, help='Max reverse queries per chunk')
    pipeline_parser.add_argument('--max-fanout-per-query', type=int, default=3, help='Max fan-out per query')
    pipeline_parser.add_argument('--top-k', type=int, default=10, help='Top results')
    pipeline_parser.add_argument('--delay', type=float, default=1.0, help='Crawl delay')
    pipeline_parser.add_argument('--depth', type=int, default=1, help='Crawl depth')
    pipeline_parser.add_argument('--no-semantic', action='store_true', help='Use basic chunking')
    pipeline_parser.set_defaults(func=pipeline_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main() 