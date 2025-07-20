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

# Import our modules
from src.crawl import WebCrawler, crawl_single_page
from src.chunk import chunk_crawled_pages, SemanticChunker
from src.embed import embed_and_index, load_existing_index, EmbeddingPipeline
from src.expand import expand_query_simple, generate_intent_tree_simple
from src.score import score_query_chunks, analyze_content_performance_simple
from src.channels import gather_social_chatter, analyze_social_impact
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
    st.markdown("### LLM-First Website Performance Analysis")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # API Keys section
    st.sidebar.subheader("API Configuration")
    google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
    
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Pipeline configuration
    st.sidebar.subheader("Pipeline Settings")
    max_pages = st.sidebar.slider("Max Pages to Crawl", 10, 200, 50)
    chunk_size = st.sidebar.slider("Chunk Size (tokens)", 100, 300, 150)
    max_expansions = st.sidebar.slider("Max Query Expansions", 5, 30, 15)
    top_k = st.sidebar.slider("Top K Results", 5, 20, 10)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Pipeline", "📊 Analysis", "🔍 Search", "📱 Social"])
    
    with tab1:
        pipeline_tab(max_pages, chunk_size, max_expansions, top_k)
    
    with tab2:
        analysis_tab()
    
    with tab3:
        search_tab(top_k)
    
    with tab4:
        social_tab()

def pipeline_tab(max_pages, chunk_size, max_expansions, top_k):
    """Pipeline execution tab."""
    st.header("🚀 Run Complete Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        url = st.text_input("Website URL", placeholder="https://example.com")
        query = st.text_input("Analysis Query", placeholder="marketing strategies")
    
    with col2:
        include_social = st.checkbox("Include Social Media Analysis", value=False)
        use_semantic = st.checkbox("Use Semantic Chunking", value=True)
    
    if st.button("🚀 Run Pipeline", type="primary"):
        if not url or not query:
            st.error("Please provide both URL and query")
            return
        
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("Please provide Google Gemini API key")
            return
        
        # Run pipeline with progress tracking
        with st.spinner("Running pipeline..."):
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
                
                chunks = chunk_crawled_pages(use_semantic=use_semantic)
                
                st.success(f"✅ Created {len(chunks)} chunks")
                
                # Step 3: Embed and Index
                status_text.text("Step 3/6: Creating embeddings and index...")
                progress_bar.progress(50)
                
                faiss_index = embed_and_index()
                if not faiss_index:
                    st.error("Failed to create index")
                    return
                
                st.success("✅ Created FAISS index")
                
                # Step 4: Query Expansion
                status_text.text("Step 4/6: Expanding queries...")
                progress_bar.progress(66)
                
                expanded_queries = expand_query_simple(query, max_expansions)
                
                st.success(f"✅ Generated {len(expanded_queries)} expanded queries")
                
                # Step 5: Search and Score
                status_text.text("Step 5/6: Searching and scoring...")
                progress_bar.progress(83)
                
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
                
                # Step 6: Social Media (optional)
                if include_social:
                    status_text.text("Step 6/6: Analyzing social media...")
                    progress_bar.progress(100)
                    
                    social_data = gather_social_chatter(query)
                    analysis = analyze_social_impact(social_data)
                    
                    st.success(f"✅ Found {analysis['total_content']} social media mentions")
                
                progress_bar.progress(100)
                status_text.text("Pipeline completed!")
                
                # Display results
                display_pipeline_results(query, expanded_queries, top_results, social_data if include_social else None)
                
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                logger.error(f"Pipeline error: {e}")

def display_pipeline_results(query, expanded_queries, top_results, social_data):
    """Display pipeline results."""
    st.header("📊 Pipeline Results")
    
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
    st.subheader("🔍 Expanded Queries")
    for i, expanded_query in enumerate(expanded_queries, 1):
        st.write(f"{i}. {expanded_query}")
    
    # Top results
    st.subheader("🏆 Top Results")
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
    st.header("📊 Content Analysis")
    
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
    st.header("🔍 Search Content")
    
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
    st.header("📱 Social Media Analysis")
    
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

if __name__ == "__main__":
    main() 