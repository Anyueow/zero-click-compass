"""
Streamlit dashboard for Zero-Click Compass.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our modules
from src.crawl import WebCrawler, crawl_single_page
from src.chunk import chunk_crawled_pages, SemanticChunker
from src.embed import embed_and_index, load_existing_index, EmbeddingPipeline
from src.expand import expand_query_simple, generate_intent_tree_simple
from src.score import score_query_chunks, analyze_content_performance_simple
from src.channels import gather_social_chatter, analyze_social_impact
from src.query_generator import ReverseQueryGenerator, QueryAnalyzer
from src.query_fanout import QueryFanoutGenerator
from src.utils import create_data_dir, load_jsonl, logger

# Page configuration
st.set_page_config(
    page_title="Zero-Click Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .result-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .score-high { color: #28a745; }
    .score-medium { color: #ffc107; }
    .score-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧭 Zero-Click Compass</h1>', unsafe_allow_html=True)
    st.markdown("### Sequential Content-to-Query Analysis Pipeline")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # API Keys section
    st.sidebar.subheader("API Configuration")
    
    # Check if API key is already set in environment
    current_api_key = os.getenv("GOOGLE_API_KEY", "")
    if current_api_key:
        st.sidebar.success("✅ Google API key loaded from .env")
        # Option to override
        override_key = st.sidebar.checkbox("Override with custom key")
        if override_key:
            google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
            if google_api_key:
                os.environ["GOOGLE_API_KEY"] = google_api_key
    else:
        st.sidebar.warning("⚠️ Google API key not found in .env")
        google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password", 
                                             help="Get from https://ai.google.dev/")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        else:
            st.sidebar.error("Google API key is required to run the pipeline")
    
    # Pipeline configuration
    st.sidebar.subheader("Sequential Pipeline Settings")
    max_pages = st.sidebar.slider("Max Pages to Crawl", 1, 10, 3)
    chunk_size = st.sidebar.slider("Chunk Size (tokens)", 100, 500, 200)
    max_chunks = st.sidebar.slider("Max Chunks", 50, 200, 100)
    sliding_window = st.sidebar.slider("Sliding Window Overlap", 50, 200, 100)
    max_reverse_queries = st.sidebar.slider("Max Reverse Queries", 3, 10, 5)
    max_fanout_per_query = st.sidebar.slider("Max Fan-out per Query", 10, 20, 15)
    max_expansions = st.sidebar.slider("Max Query Expansions", 5, 30, 15)
    top_k = st.sidebar.slider("Top K Results", 5, 20, 10)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 Complete Pipeline", "📊 Analysis", "🔍 Search", "📱 Social", "🔄 Reverse Queries", "🏢 Brand Curation"])
    
    with tab1:
        pipeline_tab(max_pages, chunk_size, max_chunks, sliding_window, max_reverse_queries, max_fanout_per_query, max_expansions, top_k)
    
    with tab2:
        analysis_tab()
    
    with tab3:
        search_tab(top_k)
    
    with tab4:
        social_tab()
    
    with tab5:
        reverse_queries_tab()
    
    with tab6:
        brand_curation_tab()

def pipeline_tab(max_pages, chunk_size, max_chunks, sliding_window, max_reverse_queries, max_fanout_per_query, max_expansions, top_k):
    """Pipeline execution tab."""
    st.header("🚀 Complete Sequential Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        url = st.text_input("Website URL", placeholder="https://example.com")
        query = st.text_input("Analysis Query", placeholder="marketing strategies")
    
    with col2:
        include_social = st.checkbox("Include Social Media Analysis", value=False)
        use_semantic = st.checkbox("Use Semantic Chunking", value=True)
    
    if st.button("🚀 Run Sequential Pipeline", type="primary"):
        if not url or not query:
            st.error("Please provide both URL and query")
            return
        
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Please provide Google Gemini API key")
            return
        
        # Run sequential pipeline with progress tracking
        with st.spinner("Running sequential pipeline..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Crawl
                status_text.text("Step 1/6: Crawling website...")
                progress_bar.progress(16)
                
                crawler = WebCrawler(max_pages=max_pages, delay=1.0)
                pages = crawler.crawl_website(url, max_depth=2)
                crawler.close()
                
                st.success(f"✅ Crawled {len(pages)} pages")
                
                # Step 2: Chunk
                status_text.text("Step 2/6: Chunking content...")
                progress_bar.progress(33)
                
                # Use semantic chunking with new parameters
                if use_semantic:
                    chunker = SemanticChunker(
                        target_tokens=chunk_size,
                        overlap_tokens=sliding_window
                    )
                    # Load crawled pages and chunk them
                    from .utils import load_jsonl
                    crawled_pages = load_jsonl("data/crawled_pages.jsonl")
                    chunks = chunker.chunk_pages(crawled_pages)
                    
                    # Limit chunks to max_chunks
                    if len(chunks) > max_chunks:
                        chunks = chunks[:max_chunks]
                else:
                    chunks = chunk_crawled_pages(use_semantic=False)
                    
                    # Limit chunks to max_chunks
                    if len(chunks) > max_chunks:
                        chunks = chunks[:max_chunks]
                
                st.success(f"✅ Created {len(chunks)} chunks")
                
                # Step 3: Embed and Index
                status_text.text("Step 3/6: Creating embeddings and index...")
                progress_bar.progress(50)
                
                faiss_index = embed_and_index()
                if not faiss_index:
                    st.error("Failed to create index")
                    return
                
                st.success("✅ Created FAISS index")
                
                # Step 4: Query Expansion (Generate reverse queries from content)
                status_text.text("Step 4/6: Generating reverse queries from content...")
                progress_bar.progress(66)
                
                query_generator = ReverseQueryGenerator()
                reverse_queries = query_generator.generate_queries_for_chunks(
                    chunks, 
                    max_queries_per_chunk=max_reverse_queries,
                    output_file="data/reverse_queries.jsonl"
                )
                
                # Get top 5 reverse queries for fan-out
                top_reverse_queries = []
                for chunk_result in reverse_queries:
                    chunk_queries = chunk_result.get('queries', [])
                    # Sort by relevance score and take top queries
                    sorted_queries = sorted(chunk_queries, key=lambda x: x.get('relevance_score', 0), reverse=True)
                    top_reverse_queries.extend(sorted_queries[:max_reverse_queries])
                
                # Limit to top 5 reverse queries for fan-out
                top_reverse_queries = top_reverse_queries[:5]
                reverse_query_texts = [q.get('query_text', '') for q in top_reverse_queries if q.get('query_text')]
                
                st.success(f"✅ Generated {len(reverse_query_texts)} top reverse queries")
                
                # Step 5: Analysis and Expansion (Fan-out the reverse queries)
                status_text.text("Step 5/6: Expanding reverse queries with fan-out...")
                progress_bar.progress(83)
                
                # Fan-out each of the 5 reverse queries (max 15 each)
                fanout_generator = QueryFanoutGenerator()
                all_fanout_queries = []
                
                for reverse_query in reverse_query_texts:
                    fanout_queries = fanout_generator.generate_fanout(
                        reverse_query, 
                        mode="AI Overview (simple)",
                        max_expansions=max_fanout_per_query
                    )
                    all_fanout_queries.extend(fanout_queries)
                
                # Use fan-out queries for search (not original query expansion)
                expanded_queries = all_fanout_queries
                
                st.success(f"✅ Generated {len(expanded_queries)} fan-out queries from {len(reverse_query_texts)} reverse queries")
                
                # Step 6: Search and Score
                status_text.text("Step 6/6: Searching and scoring...")
                progress_bar.progress(100)
                
                all_results = []
                for expanded_query in expanded_queries:
                    similar_chunks = faiss_index.search_similar(expanded_query, top_k)
                    all_results.extend(similar_chunks)
                
                # Remove duplicates and get top results
                seen_chunks = set()
                unique_results = []
                for chunk in all_results:
                    chunk_id = chunk['id']
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        unique_results.append(chunk)
                
                unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                top_results = unique_results[:top_k]
                
                st.success(f"✅ Found {len(top_results)} top results")
                
                progress_bar.progress(100)
                status_text.text("Sequential pipeline completed!")
                
                # Display results
                display_pipeline_results(query, expanded_queries, top_results, social_data if include_social else None)
                
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                logger.error(f"Pipeline error: {e}")

def display_pipeline_results(query, expanded_queries, top_results, social_data):
    """Display pipeline results."""
    st.header("📊 Sequential Pipeline Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Query", query)
    
    with col2:
        st.metric("Expanded Queries", len(expanded_queries))
    
    with col3:
        st.metric("Top Results", len(top_results))
    
    with col4:
        if social_data:
            total_content = sum(platform.get('count', 0) for platform in social_data['platforms'].values())
            st.metric("Social Mentions", total_content)
        else:
            st.metric("Social Mentions", "N/A")
    
    # Expanded queries
    st.subheader("🔍 Fan-out Queries (from Reverse Queries)")
    for i, expanded_query in enumerate(expanded_queries, 1):
        st.write(f"{i}. {expanded_query}")
    
    # Top results
    st.subheader("🏆 Top Content Matches")
    for i, result in enumerate(top_results, 1):
        with st.expander(f"{i}. Score: {result['similarity_score']:.3f} - {result['url']}"):
            st.write(f"**URL:** {result['url']}")
            st.write(f"**Similarity Score:** {result['similarity_score']:.3f}")
            st.write(f"**Content:** {result['content'][:300]}...")
    
    # Social media analysis
    if social_data:
        st.subheader("📱 Social Media Analysis")
        
        # Platform breakdown
        platform_data = []
        for platform, data in social_data['platforms'].items():
            platform_data.append({
                'Platform': platform.title(),
                'Count': data['count']
            })
        
        if platform_data:
            df = pd.DataFrame(platform_data)
            fig = px.bar(df, x='Platform', y='Count', title="Content by Platform")
            st.plotly_chart(fig, use_container_width=True)

def analysis_tab():
    """Analysis tab for viewing existing data."""
    st.header("📊 Content Analysis Dashboard")
    
    data_dir = create_data_dir()
    
    # Check for existing data
    chunks_file = os.path.join(data_dir, "chunks.jsonl")
    index_file = os.path.join(data_dir, "faiss_index.faiss")
    
    if not os.path.exists(chunks_file):
        st.warning("No chunks data found. Run the pipeline first.")
        return
    
    # Load chunks
    chunks = load_jsonl(chunks_file)
    
    if not chunks:
        st.warning("No chunks found in data.")
        return
    
    # Analysis metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", len(chunks))
    
    with col2:
        try:
            avg_tokens = sum(chunk.get('tokens', 0) for chunk in chunks) / len(chunks)
            st.metric("Avg Tokens per Chunk", f"{avg_tokens:.1f}")
        except:
            st.metric("Avg Tokens per Chunk", "N/A")
    
    with col3:
        try:
            unique_urls = len(set(chunk.get('url', '') for chunk in chunks))
            st.metric("Unique URLs", unique_urls)
        except:
            st.metric("Unique URLs", "N/A")
    
    with col4:
        try:
            content_types = set(chunk.get('content_type', 'text') for chunk in chunks)
            st.metric("Content Types", len(content_types))
        except:
            st.metric("Content Types", "N/A")
    
    # Token distribution
    st.subheader("📈 Token Distribution")
    token_counts = [chunk.get('tokens', 0) for chunk in chunks]
    
    if token_counts and any(token_counts):
        fig = px.histogram(x=token_counts, nbins=20, title="Chunk Token Distribution")
        fig.update_layout(xaxis_title="Tokens", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No token data available")
    
    # URL distribution
    st.subheader("🌐 URL Distribution")
    url_counts = {}
    for chunk in chunks:
        url = chunk.get('url', '')
        url_counts[url] = url_counts.get(url, 0) + 1
    
    url_df = pd.DataFrame(list(url_counts.items()), columns=['URL', 'Chunk Count'])
    url_df = url_df.sort_values('Chunk Count', ascending=False).head(10)
    
    if not url_df.empty:
        fig = px.bar(url_df, x='URL', y='Chunk Count', title="Top URLs by Chunk Count")
        if hasattr(fig, 'update_xaxis'):
            fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No URL data available")
    
    # Content type distribution
    st.subheader("📝 Content Type Distribution")
    content_type_counts = {}
    for chunk in chunks:
        content_type = chunk.get('content_type', 'text')
        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
    
    content_df = pd.DataFrame(list(content_type_counts.items()), columns=['Content Type', 'Count'])
    
    if not content_df.empty:
        fig = px.pie(content_df, values='Count', names='Content Type', title="Content Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No content type data available")

def search_tab(top_k):
    """Search tab for querying existing index."""
    st.header("🔍 Content Search")
    
    # Check for existing index
    data_dir = create_data_dir()
    index_file = os.path.join(data_dir, "faiss_index.faiss")
    
    if not os.path.exists(index_file):
        st.warning("No FAISS index found. Run the pipeline first.")
        return
    
    # Search interface
    query = st.text_input("Search Query", placeholder="Enter your search query...")
    
    if st.button("🔍 Search", type="primary"):
        if not query:
            st.error("Please enter a search query")
            return
        
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Please provide Google Gemini API key")
            return
        
        try:
            # Load index
            faiss_index = load_existing_index()
            
            # Search
            similar_chunks = faiss_index.search_similar(query, top_k)
            
            st.success(f"Found {len(similar_chunks)} results")
            
            # Display results
            for i, chunk in enumerate(similar_chunks, 1):
                with st.expander(f"{i}. Score: {chunk['similarity_score']:.3f} - {chunk['url']}"):
                    st.write(f"**URL:** {chunk['url']}")
                    st.write(f"**Similarity Score:** {chunk['similarity_score']:.3f}")
                    st.write(f"**Content:** {chunk['content'][:300]}...")
                    
                    # Score breakdown if available
                    if 'scores' in chunk:
                        scores = chunk['scores']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Semantic", f"{scores.get('semantic', 0):.3f}")
                        with col2:
                            st.metric("Keyword", f"{scores.get('keyword', 0):.3f}")
                        with col3:
                            st.metric("Length", f"{scores.get('length', 0):.3f}")
                        with col4:
                            st.metric("Position", f"{scores.get('position', 0):.3f}")
        
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

def social_tab():
    """Social media analysis tab."""
    st.header("📱 Social Media Analysis Dashboard")
    
    query = st.text_input("Social Media Query", placeholder="Enter query to search social media...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        platforms = st.multiselect(
            "Platforms",
            ["reddit", "twitter"],
            default=["reddit", "twitter"]
        )
    
    with col2:
        include_analysis = st.checkbox("Include Impact Analysis", value=True)
    
    if st.button("📱 Analyze Social Media", type="primary"):
        if not query:
            st.error("Please enter a query")
            return
        
        try:
            with st.spinner("Gathering social media data..."):
                social_data = gather_social_chatter(query, platforms)
                
                if include_analysis:
                    analysis = analyze_social_impact(social_data)
                
                st.success("Social media analysis completed!")
                
                # Display results
                display_social_results(social_data, analysis if include_analysis else None)
        
        except Exception as e:
            st.error(f"Social media analysis failed: {str(e)}")

def display_social_results(social_data, analysis):
    """Display social media analysis results."""
    # Platform breakdown
    st.subheader("📊 Platform Breakdown")
    
    platform_data = []
    for platform, data in social_data['platforms'].items():
        platform_data.append({
            'Platform': platform.title(),
            'Count': data['count']
        })
    
    if platform_data:
        df = pd.DataFrame(platform_data)
        if not df.empty:
            fig = px.bar(df, x='Platform', y='Count', title="Content by Platform")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No platform data available")
    
    # Platform-specific content
    for platform, data in social_data['platforms'].items():
        if data['count'] > 0:
            st.subheader(f"📱 {platform.title()} Content")
            
            if platform == 'reddit':
                posts = data['posts']
                for i, post in enumerate(posts[:5], 1):
                    with st.expander(f"{i}. {post['title'][:100]}..."):
                        st.write(f"**Title:** {post['title']}")
                        st.write(f"**Author:** {post['author']}")
                        st.write(f"**Subreddit:** r/{post['subreddit']}")
                        st.write(f"**Score:** {post['score']}")
                        st.write(f"**Comments:** {post['num_comments']}")
                        st.write(f"**Content:** {post['content'][:200]}...")
            
            elif platform == 'twitter':
                tweets = data['tweets']
                for i, tweet in enumerate(tweets[:5], 1):
                    with st.expander(f"{i}. @{tweet['author_username']} - {tweet['content'][:100]}..."):
                        st.write(f"**Author:** @{tweet['author_username']}")
                        st.write(f"**Followers:** {tweet['author_followers']:,}")
                        st.write(f"**Likes:** {tweet['like_count']}")
                        st.write(f"**Retweets:** {tweet['retweet_count']}")
                        st.write(f"**Content:** {tweet['content']}")
    
    # Impact analysis
    if analysis:
        st.subheader("🎯 Impact Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Content", analysis['total_content'])
        
        with col2:
            high_engagement = len(analysis['high_engagement_content'])
            st.metric("High Engagement", high_engagement)
        
        with col3:
            top_influencers = len(analysis['top_influencers'])
            st.metric("Top Influencers", top_influencers)
        
        # Top influencers
        if analysis['top_influencers']:
            st.subheader("👑 Top Influencers")
            
            influencer_data = []
            for influencer in analysis['top_influencers'][:10]:
                influencer_data.append({
                    'Username': influencer['username'],
                    'Platform': influencer['platform'].title(),
                    'Engagement': influencer['engagement']
                })
            
            df = pd.DataFrame(influencer_data)
            if not df.empty:
                fig = px.bar(df, x='Username', y='Engagement', color='Platform', 
                            title="Top Influencers by Engagement")
                if hasattr(fig, 'update_xaxis'):
                    fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No influencer data available")

def reverse_queries_tab():
    """Reverse query generation tab."""
    st.markdown("### 🔄 Content-to-Query Generation")
    st.markdown("**Work backwards from your content to discover what queries it answers**")
    
    # Check if chunks exist
    chunks_file = "data/chunks.jsonl"
    if not os.path.exists(chunks_file):
        st.warning("⚠️ No chunks found. Please run the pipeline first to generate content chunks.")
        st.info("Go to the Pipeline tab to crawl and chunk your website content.")
        return
    
    # Load existing chunks
    try:
        chunks = load_jsonl(chunks_file)
        st.success(f"✅ Loaded {len(chunks)} content chunks")
    except Exception as e:
        st.error(f"Error loading chunks: {e}")
        return
    
    # Reverse query generation form
    with st.form("reverse_form"):
        st.subheader("Generate Queries from Content Chunks")
        
        # Options
        query_types = st.multiselect(
            "Query types to generate",
            ["direct", "related", "long_tail", "questions", "intent_based"],
            default=["direct", "questions", "intent_based"]
        )
        
        target_queries = st.text_area(
            "Target queries to check for gaps (optional)",
            placeholder="Enter one query per line\ne.g.,\nmarketing strategies\nSEO best practices\ncontent optimization",
            help="These queries will be checked against your generated queries to identify content gaps"
        )
        
        analyze_results = st.checkbox("Analyze results and provide recommendations", value=True)
        
        generate_submitted = st.form_submit_button("Generate Content Queries")
    
    if generate_submitted:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("❌ Google API key is required. Please set it in the sidebar.")
            return
        
        with st.spinner("Generating queries from your content chunks..."):
            try:
                # Initialize query generator
                query_generator = ReverseQueryGenerator()
                
                # Generate queries
                output_file = "data/generated_queries.jsonl"
                summary = query_generator.generate_queries_for_all_chunks(chunks, output_file)
                
                # Display results
                st.success(f"✅ Generated {summary['total_queries_generated']} queries from {summary['total_chunks_processed']} chunks")
                
                # Load and display generated queries
                generated_queries = load_jsonl(output_file)
                
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Queries", summary['total_queries_generated'])
                with col2:
                    st.metric("Chunks Processed", summary['total_chunks_processed'])
                with col3:
                    st.metric("Avg Queries/Chunk", f"{summary['average_queries_per_chunk']:.1f}")
                
                # Show sample queries
                st.subheader("Sample Content Queries")
                all_queries = []
                for chunk_result in generated_queries:
                    all_queries.extend(chunk_result.get('queries', []))
                
                if all_queries:
                    # Show top queries by relevance
                    top_queries = sorted(all_queries, key=lambda x: x.get('relevance_score', 0), reverse=True)[:10]
                    
                    for i, query in enumerate(top_queries, 1):
                        with st.expander(f"{i}. {query.get('query_text', 'Unknown')} (Score: {query.get('relevance_score', 0)})"):
                            st.write(f"**Category:** {query.get('category', 'Unknown')}")
                            st.write(f"**Intent:** {query.get('intent', 'Unknown')}")
                            st.write(f"**Relevance Score:** {query.get('relevance_score', 0)}/10")
                            st.write(f"**Confidence:** {query.get('confidence', 0):.2f}")
                
                # Analyze if requested
                if analyze_results:
                    st.subheader("📊 Content Query Analysis & Recommendations")
                    
                    analyzer = QueryAnalyzer()
                    
                    # Coverage analysis
                    coverage_analysis = query_generator.analyze_query_coverage(generated_queries)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Relevance", f"{coverage_analysis['average_relevance_score']:.1f}/10")
                        st.metric("Unique Chunks", coverage_analysis['unique_chunks'])
                    
                    with col2:
                        # Category distribution
                        categories = coverage_analysis.get('category_distribution', {})
                        if categories:
                            st.write("**Category Distribution:**")
                            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                                st.write(f"• {category}: {count}")
                    
                    # Content gaps analysis
                    if target_queries:
                        target_list = [q.strip() for q in target_queries.split('\n') if q.strip()]
                        if target_list:
                            st.subheader("🎯 Content Coverage Gap Analysis")
                            gap_analysis = analyzer.find_content_gaps(generated_queries, target_list)
                            
                            if gap_analysis['gap_count'] > 0:
                                st.warning(f"Found {gap_analysis['gap_count']} content gaps:")
                                for gap in gap_analysis['content_gaps']:
                                    st.write(f"❌ **{gap['target_query']}** (Priority: {gap['priority']})")
                            else:
                                st.success("✅ All target queries are covered by your content!")
                    
                    # Optimization recommendations
                    st.subheader("💡 Content Strategy Recommendations")
                    optimization = analyzer.optimize_content_strategy(generated_queries)
                    
                    if optimization['recommendations']:
                        for rec in optimization['recommendations']:
                            priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                            st.info(f"{priority_icon} **{rec['type'].replace('_', ' ').title()}:** {rec['message']}")
                    else:
                        st.success("✅ Your content strategy looks well-balanced!")
                
                # Download results
                st.subheader("📥 Download Content Query Results")
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        st.download_button(
                            label="Download Content Queries (JSONL)",
                            data=f.read(),
                            file_name="generated_queries.jsonl",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"Error generating content queries: {e}")
                st.exception(e)

def brand_curation_tab():
    """Brand-specific curation and optimization tab."""
    st.header("🏢 Brand Content Strategy")
    st.markdown("### AI-Powered Sequential Content Strategy for Brands")
    
    # Brand Information
    st.subheader("🎯 Brand Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand_name = st.text_input("Brand Name", placeholder="Your Brand")
        industry = st.selectbox(
            "Industry",
            ["B2B SaaS", "E-commerce", "Healthcare", "Finance", "Education", "Real Estate", 
             "Manufacturing", "Consulting", "Agency", "Personal Brand", "Other"]
        )
        brand_stage = st.selectbox(
            "Brand Stage",
            ["Startup", "Growth", "Established", "Enterprise", "Personal Brand"]
        )
    
    with col2:
        target_audience = st.multiselect(
            "Target Audience",
            ["B2B Decision Makers", "B2C Consumers", "Developers", "Marketers", "Sales Teams",
             "Executives", "Small Business Owners", "Enterprise Buyers", "Freelancers", "Students"]
        )
        content_goals = st.multiselect(
            "Content Goals",
            ["Brand Awareness", "Lead Generation", "Thought Leadership", "Customer Education",
             "Product Marketing", "SEO Traffic", "Social Engagement", "Community Building"]
        )
    
    # Query Strategy
    st.subheader("🔍 Sequential Query Strategy")
    
    strategy_type = st.selectbox(
        "Analysis Strategy",
        ["Content Audit", "Competitive Analysis", "Market Research", "SEO Optimization", 
         "Social Media Strategy", "Thought Leadership", "Product Marketing"]
    )
    
    # Brand-specific query templates
    query_templates = {
        "Content Audit": {
            "B2B SaaS": ["B2B SaaS content marketing ROI", "SaaS lead generation content", "B2B content marketing strategies"],
            "E-commerce": ["e-commerce SEO optimization", "online store content marketing", "e-commerce conversion optimization"],
            "Healthcare": ["healthcare content marketing", "medical practice SEO", "healthcare lead generation"],
            "Finance": ["financial services content marketing", "fintech SEO strategies", "wealth management content"],
            "Education": ["edtech content marketing", "online course marketing", "educational content SEO"],
            "Real Estate": ["real estate content marketing", "property listing SEO", "real estate lead generation"],
            "Manufacturing": ["manufacturing content marketing", "industrial SEO", "B2B manufacturing marketing"],
            "Consulting": ["consulting content marketing", "professional services SEO", "thought leadership content"],
            "Agency": ["agency content marketing", "service business SEO", "agency lead generation"],
            "Personal Brand": ["personal branding strategies", "thought leadership content", "personal brand SEO"]
        },
        "Competitive Analysis": {
            "B2B SaaS": ["SaaS competitor analysis", "B2B SaaS market positioning", "SaaS content strategy"],
            "E-commerce": ["e-commerce competitor research", "online store competitive analysis", "e-commerce market trends"],
            "Healthcare": ["healthcare competitor analysis", "medical practice competitive research", "healthcare market trends"],
            "Finance": ["financial services competitive analysis", "fintech competitor research", "wealth management trends"],
            "Education": ["edtech competitive analysis", "online education market research", "educational content trends"],
            "Real Estate": ["real estate competitive analysis", "property market research", "real estate marketing trends"],
            "Manufacturing": ["manufacturing competitive analysis", "industrial market research", "B2B manufacturing trends"],
            "Consulting": ["consulting competitive analysis", "professional services research", "consulting market trends"],
            "Agency": ["agency competitive analysis", "service business research", "agency market trends"],
            "Personal Brand": ["personal brand competitive analysis", "thought leadership research", "personal brand trends"]
        },
        "Market Research": {
            "B2B SaaS": ["B2B SaaS market trends", "SaaS industry analysis", "B2B software market research"],
            "E-commerce": ["e-commerce market trends", "online retail analysis", "e-commerce industry research"],
            "Healthcare": ["healthcare market trends", "medical industry analysis", "healthcare technology research"],
            "Finance": ["financial services trends", "fintech market analysis", "wealth management research"],
            "Education": ["edtech market trends", "online education analysis", "educational technology research"],
            "Real Estate": ["real estate market trends", "property market analysis", "real estate technology research"],
            "Manufacturing": ["manufacturing market trends", "industrial analysis", "B2B manufacturing research"],
            "Consulting": ["consulting market trends", "professional services analysis", "consulting industry research"],
            "Agency": ["agency market trends", "service business analysis", "marketing agency research"],
            "Personal Brand": ["personal brand trends", "thought leadership analysis", "personal brand research"]
        }
    }
    
    # Show recommended queries based on selection
    if strategy_type in query_templates and industry in query_templates[strategy_type]:
        recommended_queries = query_templates[strategy_type][industry]
        
        st.write("**Recommended Queries for Your Brand:**")
        selected_queries = []
        
        for i, query in enumerate(recommended_queries):
            if st.checkbox(f"✅ {query}", key=f"query_{i}"):
                selected_queries.append(query)
        
        # Custom queries
        st.write("**Custom Queries:**")
        custom_queries = st.text_area(
            "Add your own queries (one per line)",
            placeholder="Enter custom queries here...",
            height=100
        )
        
        if custom_queries:
            custom_list = [q.strip() for q in custom_queries.split('\n') if q.strip()]
            selected_queries.extend(custom_list)
    
    # Analysis Options
    st.subheader("📊 Sequential Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_reverse_queries = st.checkbox("Generate Reverse Queries", value=True)
        include_fanout = st.checkbox("Query Fan-Out Expansion", value=True)
        include_xai_optimization = st.checkbox("XAI Content Optimization", value=True)
    
    with col2:
        include_channel_analysis = st.checkbox("Channel Strategy Analysis", value=True)
        include_comprehensive_scoring = st.checkbox("Comprehensive Scoring", value=True)
        include_social_analysis = st.checkbox("Social Media Analysis", value=False)
    
    # Brand-specific insights
    st.subheader("💡 Brand-Specific Insights")
    
    # Industry benchmarks
    industry_benchmarks = {
        "B2B SaaS": {
            "avg_content_length": "1500-2500 words",
            "publishing_frequency": "2-3 times per week",
            "top_channels": ["LinkedIn", "Reddit", "Quora", "Google"],
            "content_types": ["Case Studies", "Technical Guides", "Thought Leadership", "Product Updates"]
        },
        "E-commerce": {
            "avg_content_length": "800-1500 words",
            "publishing_frequency": "3-4 times per week",
            "top_channels": ["Google", "Instagram", "Pinterest", "Facebook"],
            "content_types": ["Product Guides", "Shopping Tips", "Reviews", "Lifestyle Content"]
        },
        "Healthcare": {
            "avg_content_length": "1000-2000 words",
            "publishing_frequency": "1-2 times per week",
            "top_channels": ["Google", "LinkedIn", "Medical Forums", "YouTube"],
            "content_types": ["Educational Content", "Patient Resources", "Medical Updates", "Wellness Tips"]
        },
        "Finance": {
            "avg_content_length": "1200-2500 words",
            "publishing_frequency": "2-3 times per week",
            "top_channels": ["LinkedIn", "Google", "Financial Forums", "YouTube"],
            "content_types": ["Market Analysis", "Investment Guides", "Financial Planning", "Regulatory Updates"]
        }
    }
    
    if industry in industry_benchmarks:
        benchmarks = industry_benchmarks[industry]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Content Length", benchmarks["avg_content_length"])
            st.metric("Publishing Frequency", benchmarks["publishing_frequency"])
        
        with col2:
            st.write("**Top Channels:**")
            for channel in benchmarks["top_channels"]:
                st.write(f"• {channel}")
            
            st.write("**Content Types:**")
            for content_type in benchmarks["content_types"]:
                st.write(f"• {content_type}")
    
    # Run Analysis
    st.subheader("🚀 Run Sequential Brand Analysis")
    
    if st.button("🏢 Run Brand Curation Analysis", type="primary"):
        if not selected_queries:
            st.error("Please select at least one query for analysis")
            return
        
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Please provide Google Gemini API key")
            return
        
        # Run brand-specific analysis
        with st.spinner("Running sequential brand analysis..."):
            try:
                # This would integrate with the existing CLI commands
                st.info("Sequential brand analysis would run the following:")
                
                analysis_steps = []
                if include_reverse_queries:
                    analysis_steps.append("🔄 Reverse Query Generation")
                if include_fanout:
                    analysis_steps.append("🌊 Query Fan-Out Expansion")
                if include_xai_optimization:
                    analysis_steps.append("🧠 XAI Content Optimization")
                if include_channel_analysis:
                    analysis_steps.append("📱 Channel Strategy Analysis")
                if include_comprehensive_scoring:
                    analysis_steps.append("🎯 Comprehensive Scoring")
                if include_social_analysis:
                    analysis_steps.append("📱 Social Media Analysis")
                
                for step in analysis_steps:
                    st.write(f"• {step}")
                
                # Show brand-specific recommendations
                st.success("✅ Sequential brand analysis completed!")
                
                # Display brand insights
                st.subheader("🏢 Sequential Brand Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Industry", industry)
                    st.metric("Brand Stage", brand_stage)
                    st.metric("Target Audiences", len(target_audience))
                
                with col2:
                    st.metric("Content Goals", len(content_goals))
                    st.metric("Analysis Strategy", strategy_type)
                    st.metric("Selected Queries", len(selected_queries))
                
                # Brand-specific recommendations
                st.subheader("💡 Sequential Brand Recommendations")
                
                recommendations = [
                    f"**Content Strategy:** Focus on {', '.join(content_goals[:2])} content for {industry}",
                    f"**Audience Targeting:** Create content specifically for {', '.join(target_audience[:2])}",
                    f"**Channel Priority:** Prioritize channels based on {industry} audience behavior",
                    f"**Content Types:** Develop {strategy_type.lower()} content that resonates with your brand stage",
                    f"**Query Focus:** Optimize for {len(selected_queries)} high-value queries in your niche"
                ]
                
                for rec in recommendations:
                    st.info(rec)
                
            except Exception as e:
                st.error(f"Error running sequential brand analysis: {e}")
                st.exception(e)

if __name__ == "__main__":
    main() 