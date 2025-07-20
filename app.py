"""
Zero-Click Compass - Streamlined Streamlit App
Essential pipeline for content analysis and optimization.
"""
import streamlit as st
import os
import sys
import time
import json
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append('src')

from src.crawl import WebCrawler
from src.chunk import SemanticChunker
from src.embed import EmbeddingPipeline
from src.query_generator import ReverseQueryGenerator
from src.query_fanout import QueryFanoutGenerator
from src.comprehensive_scorer import ComprehensiveScorer
from src.enhanced_ai_analyzer import EnhancedAIAnalyzer
from src.utils import logger, create_data_dir, load_jsonl, save_jsonl

# Page config
st.set_page_config(
    page_title="ZeroClix",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .log-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        max-height: 300px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

def log_to_streamlit(message, level="INFO"):
    """Log message to Streamlit interface."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {level}: {message}")
    logger.info(message)

def clear_logs():
    """Clear logs."""
    st.session_state.logs = []

def run_pipeline(url, query, max_pages=3, max_chunks_per_page=5):
    """Run the complete pipeline and return results."""
    results = {
        'crawled_pages': [],
        'chunks': [],
        'reverse_queries': [],
        'fanout_queries': [],
        'scores': [],
        'recommendations': []
    }
    
    try:
        # Step 1: Crawl
        log_to_streamlit(f"🕷️ Starting web crawl for: {url}")
        crawler = WebCrawler(max_pages=max_pages, delay=0.5)
        pages = crawler.crawl_website(url, max_depth=1)
        crawler.close()
        results['crawled_pages'] = pages
        log_to_streamlit(f"✅ Crawled {len(pages)} pages")
        
        # Step 2: Chunk
        log_to_streamlit("✂️ Chunking content...")
        chunker = SemanticChunker(target_tokens=150, overlap_tokens=20)
        chunks = chunker.chunk_pages(pages, max_chunks_per_page=max_chunks_per_page)
        results['chunks'] = chunks
        log_to_streamlit(f"✅ Created {len(chunks)} chunks (max {max_chunks_per_page} per page)")
        
        # Step 3: Embed
        log_to_streamlit("🔍 Creating embeddings and search index...")
        embedding_pipeline = EmbeddingPipeline()
        faiss_index = embedding_pipeline.process_chunks(chunks, save_intermediate=False)
        log_to_streamlit("✅ Created search index")
        
        # Step 4: Generate reverse queries
        log_to_streamlit("🔄 Generating reverse queries from content...")
        query_generator = ReverseQueryGenerator()
        
        # Process first 2 chunks for speed
        limited_chunks = chunks[:min(2, len(chunks))]
        log_to_streamlit(f"📝 Processing {len(limited_chunks)} chunks for reverse queries")
        
        reverse_queries_summary = query_generator.generate_queries_for_all_chunks(
            limited_chunks, 
            output_file="data/reverse_queries.jsonl"
        )
        
        # Load generated queries
        generated_queries = load_jsonl("data/reverse_queries.jsonl")
        results['reverse_queries'] = generated_queries
        
        # Extract top queries with full metadata (limit to 2 per chunk)
        all_queries = []
        for chunk_result in generated_queries:
            chunk_queries = chunk_result.get('queries', [])
            sorted_queries = sorted(chunk_queries, key=lambda x: x.get('relevance_score', 0), reverse=True)
            all_queries.extend(sorted_queries[:2])  # Top 2 per chunk (reduced from 3)
        
        top_queries = all_queries[:5]  # Top 5 overall
        log_to_streamlit(f"✅ Generated {len(top_queries)} top reverse queries")
        
        # Step 5: Fan-out expansion
        log_to_streamlit("🌊 Expanding queries with fan-out...")
        fanout_generator = QueryFanoutGenerator()
        all_fanout_queries = []
        
        # Process top 2 reverse queries
        reverse_query_texts = [q.get('query_text', '') for q in top_queries[:2] if q.get('query_text')]
        log_to_streamlit(f"🔄 Processing {len(reverse_query_texts)} reverse queries for fan-out")
        
        for i, reverse_query in enumerate(reverse_query_texts):
            fanout_result = fanout_generator.generate_fanout(
                reverse_query, 
                mode="AI Overview (simple)"
            )
            expanded_queries = fanout_result.get('expanded_queries', [])
            
            # Store fan-out queries with metadata
            for j, query_obj in enumerate(expanded_queries):
                if query_obj.get('query'):
                    fanout_query_data = {
                        'query': query_obj.get('query', ''),
                        'type': query_obj.get('type', ''),
                        'user_intent': query_obj.get('user_intent', ''),
                        'reasoning': query_obj.get('reasoning', ''),
                        'source_reverse_query': reverse_query,
                        'source_rank': i + 1,
                        'fanout_rank': j + 1
                    }
                    all_fanout_queries.append(fanout_query_data)
        
        results['fanout_queries'] = all_fanout_queries
        log_to_streamlit(f"✅ Generated {len(all_fanout_queries)} fan-out queries")
        
        # Step 6: Search and score
        log_to_streamlit("🎯 Searching and scoring content...")
        all_results = []
        query_scores = {}  # Track scores for each query
        
        for fanout_query_data in all_fanout_queries:
            query = fanout_query_data['query']
            similar_chunks = faiss_index.search_similar(query, 5)
            
            # Add query information to each result
            for chunk in similar_chunks:
                chunk['query'] = query  # Add the query that generated this result
            
            # Store the best score for this query
            if similar_chunks:
                best_score = max(chunk.get('similarity_score', 0) for chunk in similar_chunks)
                fanout_query_data['best_score'] = best_score
                query_scores[query] = best_score
            
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
        top_results = unique_results[:10]
        
        results['scores'] = top_results
        log_to_streamlit(f"✅ Found {len(top_results)} top results")
        
        # Step 7: Enhanced AI analysis with XAI and Mistral 7B
        log_to_streamlit("🎯 Running enhanced AI analysis with XAI and Mistral 7B...")
        
        # First run comprehensive scoring
        comprehensive_scorer = ComprehensiveScorer()
        comprehensive_results = comprehensive_scorer.score_chunks_against_queries(chunks, all_fanout_queries)
        
        # Then run enhanced AI analysis
        enhanced_analyzer = EnhancedAIAnalyzer()
        enhanced_results = enhanced_analyzer.generate_comprehensive_recommendations(
            chunks, all_fanout_queries, comprehensive_results['chunk_scores']
        )
        
        # Combine results
        results['comprehensive_analysis'] = comprehensive_results
        results['enhanced_analysis'] = enhanced_results
        
        # Log XAI recommendations
        if enhanced_results['xai_recommendations'].get('analysis'):
            log_to_streamlit("📊 XAI Analysis Results:")
            xai_analysis = enhanced_results['xai_recommendations']['analysis']
            # Split into lines and log each line
            for line in xai_analysis.split('\n'):
                if line.strip():
                    log_to_streamlit(f"  {line}")
        
        # Log Mistral recommendations
        if enhanced_results['mistral_keywords'].get('analysis'):
            log_to_streamlit("🤖 Mistral 7B Recommendations:")
            mistral_analysis = enhanced_results['mistral_keywords']['analysis']
            # Split into lines and log each line
            for line in mistral_analysis.split('\n'):
                if line.strip():
                    log_to_streamlit(f"  {line}")
        
        # Step 8: Generate recommendations
        log_to_streamlit("💡 Generating content and channel recommendations...")
        recommendations = generate_recommendations(top_results, all_fanout_queries, query)
        results['recommendations'] = recommendations
        log_to_streamlit("✅ Generated recommendations")
        
        log_to_streamlit("🎉 Pipeline completed successfully!")
        return results
        
    except Exception as e:
        log_to_streamlit(f"❌ Pipeline failed: {str(e)}", "ERROR")
        return None

def generate_recommendations(top_results, fanout_queries, original_query):
    """Generate content and channel recommendations."""
    recommendations = {
        'content_optimization': [],
        'channel_strategy': [],
        'priority_actions': []
    }
    
    # Content optimization based on top queries
    top_query_types = {}
    for query_data in fanout_queries[:10]:
        query = query_data.get('query', '')
        query_lower = query.lower()
        if 'how' in query_lower or 'what' in query_lower:
            top_query_types['how-to'] = top_query_types.get('how-to', 0) + 1
        elif 'best' in query_lower or 'top' in query_lower:
            top_query_types['best-of'] = top_query_types.get('best-of', 0) + 1
        elif 'vs' in query_lower or 'compare' in query_lower:
            top_query_types['comparison'] = top_query_types.get('comparison', 0) + 1
        else:
            top_query_types['informational'] = top_query_types.get('informational', 0) + 1
    
    # Content recommendations
    if top_query_types.get('how-to', 0) > 2:
        recommendations['content_optimization'].append({
            'type': 'how-to',
            'priority': 'high',
            'action': 'Create step-by-step guides and tutorials',
            'reason': f"{top_query_types['how-to']} how-to queries detected"
        })
    
    if top_query_types.get('comparison', 0) > 1:
        recommendations['content_optimization'].append({
            'type': 'comparison',
            'priority': 'medium',
            'action': 'Develop comparison content and product reviews',
            'reason': f"{top_query_types['comparison']} comparison queries detected"
        })
    
    # Channel strategy
    if len(fanout_queries) > 15:
        recommendations['channel_strategy'].append({
            'channel': 'SEO',
            'priority': 'high',
            'action': 'Optimize for long-tail keywords',
            'reason': 'High query diversity indicates strong SEO opportunity'
        })
    
    if any('social' in q.get('query', '').lower() or 'community' in q.get('query', '').lower() for q in fanout_queries):
        recommendations['channel_strategy'].append({
            'channel': 'Social Media',
            'priority': 'medium',
            'action': 'Engage with community discussions',
            'reason': 'Social/community queries detected'
        })
    
    # Priority actions
    recommendations['priority_actions'] = [
        f"Create content for top {min(5, len(fanout_queries))} queries",
        f"Optimize {len(top_results)} high-scoring content pieces",
        "Monitor query performance and iterate"
    ]
    
    return recommendations

def main():
    """Main Streamlit app."""
    # Initialize session state
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Header
    st.markdown('<h1 class="main-header">🧭 Zero-Click Compass</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <h3>Optimize your content for the zero-click world. Ensure your brand is discovered through AI overviews!</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        url = st.text_input(
            "Website URL",
            value="https://www.avocadogreenmattress.com",
            help="Enter the website URL to analyze"
        )
        
        query = st.text_input(
            "Target Query",
            value="organic mattress",
            help="Enter your target search query"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_pages = st.number_input("Max Pages", min_value=1, max_value=10, value=3)
        with col2:
            max_chunks = st.number_input("Max Chunks/Page", min_value=1, max_value=10, value=5)
        
        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
            clear_logs()
            with st.spinner("Running pipeline..."):
                st.session_state.results = run_pipeline(url, query, max_pages, max_chunks)
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.results = None
            clear_logs()
    
    # Main content
    if st.session_state.results:
        results = st.session_state.results
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pages Crawled", len(results['crawled_pages']))
        with col2:
            st.metric("Chunks Created", len(results['chunks']))
        with col3:
            st.metric("Reverse Queries", len(results['reverse_queries']))
        with col4:
            st.metric("Fan-out Queries", len(results['fanout_queries']))
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Queries", "🎯 Scores", "🎯 XAI Analysis", "💡 Recommendations", "📝 Logs"])
        
        with tab1:
            st.markdown('<h2 class="section-header">Generated Queries</h2>', unsafe_allow_html=True)
            
            # Reverse queries
            st.subheader("🔄 Reverse Queries from Content")
            reverse_queries_data = []
            for chunk_result in results['reverse_queries']:
                for query in chunk_result.get('queries', []):
                    # Get chunk preview from the chunks data
                    chunk_preview = ""
                    chunk_id = chunk_result.get('chunk_id', '')
                    for chunk in results['chunks']:
                        if chunk.get('id') == chunk_id:
                            chunk_text = chunk.get('text', '')
                            chunk_preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                            break
                    
                    reverse_queries_data.append({
                        'Query': query.get('query_text', ''),
                        'Category': query.get('category', ''),
                        'Relevance Score': query.get('relevance_score', 0),
                        'Intent': query.get('intent', ''),
                        'Source Chunk Preview': chunk_preview
                    })
            
            if reverse_queries_data:
                df_reverse = pd.DataFrame(reverse_queries_data)
                df_reverse = df_reverse.sort_values('Relevance Score', ascending=False)
                st.dataframe(df_reverse, use_container_width=True)
                
                # Show top reverse queries summary
                st.subheader("🏆 Top Reverse Queries Summary")
                top_reverse = df_reverse.head(5)
                for i, row in top_reverse.iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{i+1}. {row['Query']}</strong><br>
                        <small>Score: {row['Relevance Score']:.1f}/10 | Category: {row['Category']} | Intent: {row['Intent']}</small><br>
                        <small><strong>Source:</strong> {row['Source Chunk Preview']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Fan-out queries
            st.subheader("🌊 Fan-out Queries")
            fanout_queries_data = []
            for i, query_data in enumerate(results['fanout_queries']):
                fanout_queries_data.append({
                    'Query': query_data['query'],
                    'Type': query_data['type'],
                    'User Intent': query_data['user_intent'],
                    'Reasoning': query_data['reasoning'],
                    'Source Reverse Query': query_data['source_reverse_query'],
                    'Source Rank': query_data['source_rank'],
                    'Fanout Rank': query_data['fanout_rank'],
                    'Best Score': f"{query_data.get('best_score', 0):.3f}"
                })
            
            if fanout_queries_data:
                df_fanout = pd.DataFrame(fanout_queries_data)
                st.dataframe(df_fanout, use_container_width=True)
                
                # Show top fan-out queries summary
                st.subheader("🌊 Top Fan-out Queries Summary")
                # Sort by best score and show top 10
                df_fanout_sorted = df_fanout.copy()
                df_fanout_sorted['Best Score'] = df_fanout_sorted['Best Score'].astype(float)
                df_fanout_sorted = df_fanout_sorted.sort_values('Best Score', ascending=False)
                
                top_fanout = df_fanout_sorted.head(10)
                for i, row in top_fanout.iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{i+1}. {row['Query']}</strong><br>
                        <small>Score: {row['Best Score']:.3f} | Type: {row['Type']} | Intent: {row['User Intent']}</small><br>
                        <small>Source: "{row['Source Reverse Query']}" (Rank {row['Source Rank']})</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<h2 class="section-header">Content Scores</h2>', unsafe_allow_html=True)
            
            # Add context about what content scores mean
            st.markdown("""
            **📊 Content Scoring Analysis**
            
            This section shows how well your content matches the generated fan-out queries. Each piece of content is scored based on:
            - **Semantic similarity** to the target queries
            - **Keyword relevance** and content overlap
            - **Content quality** and comprehensiveness
            
            Higher scores indicate content that better addresses user search intent.
            """)
            
            scores_data = []
            for result in results['scores']:
                # Get more context about the content
                content_text = result.get('content', '')
                content_preview = content_text[:150] + "..." if len(content_text) > 150 else content_text
                
                # Get the fan-out query this score was calculated against
                fanout_query = result.get('query', 'N/A')  # The query used for similarity calculation
                
                # Determine content quality indicator
                similarity_score = result.get('similarity_score', 0)
                if similarity_score >= 0.8:
                    quality_indicator = "🟢 Excellent"
                elif similarity_score >= 0.6:
                    quality_indicator = "🟡 Good"
                elif similarity_score >= 0.4:
                    quality_indicator = "🟠 Fair"
                else:
                    quality_indicator = "🔴 Needs Improvement"
                
                scores_data.append({
                    'Content Preview': content_preview,
                    'Fan-out Query': fanout_query,
                    'URL': result.get('url', ''),
                    'Similarity Score': f"{similarity_score:.3f}",
                    'Quality': quality_indicator,
                    'Content Type': result.get('content_type', '')
                })
            
            if scores_data:
                df_scores = pd.DataFrame(scores_data)
                st.dataframe(df_scores, use_container_width=True)
                
                # Add summary statistics
                st.subheader("📈 Score Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_score = sum(float(row['Similarity Score']) for row in scores_data) / len(scores_data)
                    st.metric("Average Score", f"{avg_score:.3f}")
                
                with col2:
                    excellent_count = sum(1 for row in scores_data if "🟢" in row['Quality'])
                    st.metric("Excellent Content", excellent_count)
                
                with col3:
                    needs_improvement = sum(1 for row in scores_data if "🔴" in row['Quality'])
                    st.metric("Needs Improvement", needs_improvement)
        
        with tab3:
            st.markdown('<h2 class="section-header">🎯 Enhanced AI Analysis</h2>', unsafe_allow_html=True)
            
            if 'enhanced_analysis' in results:
                enhanced = results['enhanced_analysis']
                comprehensive = results.get('comprehensive_analysis', {})
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Chunks Analyzed", enhanced['summary']['total_chunks_analyzed'])
                with col2:
                    st.metric("Queries Analyzed", enhanced['summary']['total_queries_analyzed'])
                with col3:
                    st.metric("Channel Recs", enhanced['summary']['channel_recommendations_count'])
                with col4:
                    st.metric("Content Gaps", enhanced['summary']['content_gaps_count'])
                
                # Channel Recommendations (Cards at the top)
                st.subheader("📱 Channel Strategy Recommendations")
                if enhanced['channel_recommendations']:
                    for rec in enhanced['channel_recommendations']:
                        priority_color = "🔴" if rec['focus_level'] == 'high' else "🟡" if rec['focus_level'] == 'medium' else "🟢"
                        st.markdown(f"""
                        <div class="metric-card">
                            {priority_color} **{rec['platform']}** (Score: {rec['score']}, Focus: {rec['focus_level']})
                            <br><small><strong>Strategy:</strong> {rec['strategy']}</small>
                            <br><small><strong>Top Queries:</strong> {', '.join(rec['queries'][:3])}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Content Gaps Analysis (User-Friendly)
                st.subheader("📊 Content Gaps Analysis")
                if enhanced['content_gaps']['most_common_gaps']:
                    for gap_info in enhanced['content_gaps']['most_common_gaps']:
                        priority_color = "🔴" if gap_info['frequency'] > 20 else "🟡" if gap_info['frequency'] > 10 else "🟢"
                        st.markdown(f"""
                        <div class="metric-card">
                            {priority_color} **{gap_info['gap_type']}** (Found in {gap_info['frequency']} chunks - {gap_info['percentage']})
                            <br><small><strong>What this means:</strong> {gap_info['explanation']}</small>
                            <br><small><strong>How to fix:</strong> {gap_info['suggestion']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Mistral 7B Keyword & Content Recommendations
                st.subheader("🤖 AI Content Recommendations")
                if enhanced['mistral_keywords'].get('analysis'):
                    st.markdown(f'<div class="log-container">{enhanced["mistral_keywords"]["analysis"]}</div>', unsafe_allow_html=True)
                
            else:
                st.info("Run the pipeline to see enhanced AI analysis results.")
        
        with tab4:
            st.markdown('<h2 class="section-header">Recommendations</h2>', unsafe_allow_html=True)
            
            recommendations = results['recommendations']
            
            # Content optimization
            st.subheader("📝 Content Optimization")
            for rec in recommendations['content_optimization']:
                priority_color = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                st.markdown(f"""
                <div class="metric-card">
                    {priority_color} **{rec['action']}** ({rec['priority']} priority)
                    <br><small>{rec['reason']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Channel strategy
            st.subheader("📢 Channel Strategy")
            for rec in recommendations['channel_strategy']:
                priority_color = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                st.markdown(f"""
                <div class="metric-card">
                    {priority_color} **{rec['channel']}**: {rec['action']} ({rec['priority']} priority)
                    <br><small>{rec['reason']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Priority actions
            st.subheader("🎯 Priority Actions")
            for i, action in enumerate(recommendations['priority_actions'], 1):
                st.markdown(f"{i}. {action}")
        
        with tab5:
            st.markdown('<h2 class="section-header">Pipeline Logs</h2>', unsafe_allow_html=True)
            log_text = "\n".join(st.session_state.logs)
            st.markdown(f'<div class="log-container">{log_text}</div>', unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Get Started
        
        1. **Enter a website URL** to analyze
        2. **Set your target query** 
        3. **Configure settings** (optional)
        4. **Click "Run Pipeline"** to start analysis
        
        The pipeline will:
        - 🕷️ Crawl the website
        - ✂️ Chunk content into manageable pieces
        - 🔍 Create searchable embeddings
        - 🔄 Generate reverse queries from content
        - 🌊 Expand queries with fan-out
        - 🎯 Score content relevance
        - 💡 Provide optimization recommendations
        """)
        
        # Show logs if any
        if st.session_state.logs:
            st.markdown('<h2 class="section-header">Recent Logs</h2>', unsafe_allow_html=True)
            log_text = "\n".join(st.session_state.logs[-10:])  # Last 10 logs
            st.markdown(f'<div class="log-container">{log_text}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 