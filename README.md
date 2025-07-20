# 🧭 Zero-Click Compass

**LLM-First Website Performance Analysis Pipeline**

Zero-Click Compass is an end-to-end pipeline that analyzes how your website content performs in an LLM-first, zero-click world. It crawls your website, chunks and embeds content, builds semantic search indexes, expands user queries, scores content relevance, and surfaces influencer chatter—all so marketers can understand how their pages perform when users ask AI assistants instead of clicking through search results.

## 🚀 Features

- **🔍 Intelligent Web Crawling**: Downloads and sanitizes web pages with JavaScript support
- **📝 Semantic Chunking**: Splits content into ~150-token chunks with semantic boundaries
- **🧠 Gemini Embeddings**: Uses Google's Gemini API for high-quality embeddings
- **⚡ FAISS Indexing**: Fast similarity search with Facebook's FAISS library
- **🌳 Query Expansion**: Generates intent trees and related queries for comprehensive analysis
- **📊 Multi-Method Scoring**: Combines semantic, keyword, length, and position scoring
- **📱 Social Media Analysis**: Gathers influencer chatter from Reddit and X (Twitter)
- **🎯 Performance Analytics**: Comprehensive reports on content performance
- **🖥️ Beautiful Dashboard**: Streamlit interface for easy interaction
- **⚙️ CLI Interface**: Command-line tools for automation and scripting

## 📋 Requirements

- Python 3.8+
- Google Gemini API key
- Reddit API credentials (optional)
- Twitter/X API credentials (optional)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/zero-click-compass.git
   cd zero-click-compass
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Install Chrome/Chromium** (for web crawling):
   - macOS: `brew install chromium`
   - Ubuntu: `sudo apt-get install chromium-browser`
   - Windows: Download from https://www.chromium.org/

## 🔧 Configuration

Create a `.env` file with your API keys:

```env
# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key_here

# Twitter/X API (v2) - Optional
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

# Reddit API - Optional
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=your_reddit_user_agent_here

# Pipeline Configuration
CHUNK_SIZE=150
EMBEDDING_MODEL=gemini-1.5-flash
MAX_PAGES=100
MAX_RETRIES=3
REQUEST_DELAY=1.0
```

## 🚀 Quick Start

### Web Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` and:
1. Enter your Google Gemini API key
2. Input a website URL and analysis query
3. Click "Run Pipeline" to start the analysis

### Command Line

Run the complete pipeline:

```bash
python -m src.cli pipeline https://example.com "marketing strategies"
```

Or run individual steps:

```bash
# Crawl a website
python -m src.cli crawl https://example.com --max-pages 50

# Chunk content
python -m src.cli chunk

# Create embeddings and index
python -m src.cli embed

# Expand queries
python -m src.cli expand "content marketing"

# Search existing index
python -m src.cli search "SEO tips"

# Analyze social media
python -m src.cli social "digital marketing" --analyze
```

## 📊 Usage Examples

### 1. Analyze Marketing Website Performance

```python
from src.cli import pipeline_command
import argparse

# Analyze a marketing website
args = argparse.Namespace(
    url="https://marketingcompany.com",
    query="content marketing strategies",
    max_pages=50,
    depth=2,
    delay=1.0,
    no_semantic=False,
    max_expansions=15,
    top_k=10,
    social=True
)

pipeline_command(args)
```

### 2. Custom Query Expansion

```python
from src.expand import QueryExpander, IntentTree

# Basic expansion
expander = QueryExpander()
expanded = expander.expand_query("SEO best practices")

# Intent tree expansion
intent_tree = IntentTree()
tree = intent_tree.generate_intent_tree("content marketing")
flattened = intent_tree.flatten_intent_tree(tree)
```

### 3. Content Scoring

```python
from src.score import RelevanceScorer, QueryChunkRanker

scorer = RelevanceScorer()
ranker = QueryChunkRanker()

# Score individual query-chunk pairs
score_result = scorer.score_query_chunk_pair("marketing tips", chunk_data)

# Rank chunks for multiple queries
ranked_results = ranker.rank_chunks_for_queries(["query1", "query2"], chunks)
```

### 4. Social Media Analysis

```python
from src.channels import SocialMediaAnalyzer

analyzer = SocialMediaAnalyzer()
social_data = analyzer.gather_influencer_chatter("content marketing")
analysis = analyzer.analyze_influencer_impact(social_data)
```

## 📁 Project Structure

```
zero-click-compass/
│
├── src/
│   ├── crawl.py          # Web crawling and page extraction
│   ├── chunk.py          # Content chunking with semantic boundaries
│   ├── embed.py          # Gemini embeddings and FAISS indexing
│   ├── expand.py         # Query expansion and intent trees
│   ├── score.py          # Multi-method relevance scoring
│   ├── channels.py       # Social media data collection
│   ├── utils.py          # Utility functions and helpers
│   └── cli.py            # Command-line interface
│
├── app.py                # Streamlit dashboard
├── data/                 # Crawled pages, chunks, indexes, CSV exports
├── tests/                # Unit tests for each module
├── .env                  # API keys and configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔍 Pipeline Steps

1. **Crawl** (`crawl.py`): Download and sanitize web pages
2. **Chunk** (`chunk.py`): Split content into semantic chunks
3. **Embed** (`embed.py`): Generate embeddings and build FAISS index
4. **Expand** (`expand.py`): Create query variations and intent trees
5. **Score** (`score.py`): Rank content relevance using multiple methods
6. **Analyze** (`channels.py`): Gather social media insights

## 📈 Understanding Results

### Relevance Scores

- **Semantic Score** (40%): Embedding similarity between query and content
- **Keyword Score** (30%): Word overlap and Jaccard similarity
- **Length Score** (10%): Optimal chunk length (50-200 tokens)
- **Position Score** (10%): Content position on page
- **Content Type Score** (10%): Title/heading vs body text

### Performance Metrics

- **Query Coverage**: How many expanded queries your content answers
- **Average Relevance**: Mean composite score across all queries
- **Top Performers**: Highest-scoring content pieces
- **URL Distribution**: Which pages perform best
- **Social Engagement**: Influencer mentions and engagement

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test modules:

```bash
pytest tests/test_utils.py
pytest tests/test_chunk.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for embeddings and query expansion
- [Facebook FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Streamlit](https://streamlit.io/) for the beautiful dashboard
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing
- [Selenium](https://selenium-python.readthedocs.io/) for JavaScript rendering

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/zero-click-compass/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/zero-click-compass/discussions)
- **Email**: your-email@example.com

---

**Built with ❤️ for the LLM-first future of content marketing** 