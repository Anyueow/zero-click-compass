# 🧭 Zero-Click Compass

**Optimize your content for the zero-click world. Ensure your brand is discovered through AI overviews!**

A comprehensive content analysis and optimization pipeline that helps you understand how your content performs against user queries and provides actionable recommendations for improvement.

## 🎯 What It Does

Zero-Click Compass analyzes your website content to:
- **Generate reverse queries** from your existing content
- **Expand queries** with AI-powered fan-out analysis
- **Score content relevance** against target queries
- **Identify content gaps** and optimization opportunities
- **Provide channel-specific strategies** for 6 major platforms
- **Deliver actionable recommendations** for content improvement

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Crawler   │───▶│  Content Chunker│───▶│  Embedding      │
│                 │    │                 │    │  Pipeline       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Reverse Query   │    │ Query Fan-out   │    │ Comprehensive   │
│ Generator       │    │ Generator       │    │ Scorer          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Queries   │ │   Scores    │ │ XAI Analysis│ │Recommendations│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Pipeline Flow

### 1. Content Discovery & Processing
```
Website URL
    │
    ▼
┌─────────────┐
│ Web Crawler │───▶ Crawl pages (max 3 pages, 5 chunks/page)
└─────────────┘
    │
    ▼
┌─────────────┐
│   Chunker   │───▶ Semantic chunking (150 tokens, 20 overlap)
└─────────────┘
    │
    ▼
┌─────────────┐
│ Embeddings  │───▶ Create FAISS search index
└─────────────┘
```

### 2. Query Generation & Expansion
```
Content Chunks
    │
    ▼
┌─────────────┐
│   Reverse   │───▶ Generate 2 queries per chunk
│  Queries    │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Fan-out   │───▶ Expand top queries (28+ variations)
│  Generator  │
└─────────────┘
```

### 3. Analysis & Scoring
```
Fan-out Queries
    │
    ▼
┌─────────────┐
│   Search    │───▶ Find similar content chunks
└─────────────┘
    │
    ▼
┌─────────────┐
│Comprehensive│───▶ Score chunks against queries
│   Scorer    │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Channel     │───▶ Platform-specific strategies
│ Analysis    │
└─────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Conda environment (MLHW)
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd zero-click-compass
```

2. **Set up environment**
```bash
conda activate MLHW
pip install -r requirements.txt
```

3. **Configure API keys**
```bash
cp env.template .env
# Edit .env with your Google Gemini API key
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Access the dashboard**
```
http://localhost:8502
```

## 📊 Dashboard Features

### 📊 Queries Tab
- **🔄 Reverse Queries**: Queries generated from your content (2 per chunk)
- **🏆 Top Reverse Queries**: Best queries with relevance scores
- **🌊 Fan-out Queries**: AI-expanded query variations
- **🌊 Top Fan-out Queries**: Highest-scoring expanded queries

### 📈 Scores Tab
- **Content Scoring Analysis**: How well content matches queries
- **Quality Indicators**: 🟢 Excellent, 🟡 Good, 🟠 Fair, 🔴 Needs Improvement
- **Score Summary**: Average scores and content quality breakdown

### 🎯 XAI Analysis Tab
- **Summary Metrics**: Chunks analyzed, queries analyzed, optimization count
- **Content Gaps Analysis**: Missing content patterns
- **Channel Strategy**: Platform-specific recommendations
- **Detailed Chunk Analysis**: Individual scores with grades (A-F)
- **Optimization Recommendations**: Actionable suggestions

### 💡 Recommendations Tab
- **Content Optimization**: Specific improvement suggestions
- **Channel Strategy**: Platform-specific actions
- **Priority Actions**: High-impact recommendations

### 📝 Logs Tab
- **Real-time Pipeline Logs**: Detailed execution tracking

## 🎯 Supported Platforms

The system provides channel-specific strategies for:

| Platform | Focus | Content Type | Engagement Style |
|----------|-------|--------------|------------------|
| **Reddit** | Community participation | Detailed posts and comments | Community discussions |
| **Twitter/X** | Conversation participation | Threads and replies | Trending conversations |
| **Google** | SEO optimization | Comprehensive articles | Search visibility |
| **Yelp** | Review responses | Professional engagement | Business reviews |
| **Quora** | Expert answers | Detailed responses | Q&A platform |
| **LinkedIn** | Professional networking | Industry insights | Business networking |

## 📈 Example XAI Output

```
=== XAI Optimization Results ===
Chunks optimized: 15
Queries analyzed: 10

Top Content Gaps:
  • Missing expand content (mentioned in 8/15 chunks)
  • Missing improve content (mentioned in 6/15 chunks)
  • Missing clarify content (mentioned in 4/15 chunks)

=== Channel Strategy ===
Total queries analyzed: 10

Channel Distribution:
  GOOGLE: 6 queries
  REDDIT: 4 queries
  TWITTER: 3 queries

Top Implementation Priorities:
  1. GOOGLE (Score: 18, Focus: high)
  2. REDDIT (Score: 12, Focus: medium)
  3. TWITTER (Score: 9, Focus: medium)
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
GOOGLE_API_KEY=your_google_gemini_api_key

# Optional - for enhanced social media analysis
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

### Pipeline Settings
```bash
# Content chunking
CHUNK_SIZE=150
OVERLAP_TOKENS=20

# Web crawling
MAX_PAGES=3
MAX_CHUNKS_PER_PAGE=5

# Query expansion
MAX_EXPANSIONS=15
```

## 🎯 Usage Examples

### Basic Analysis
1. Enter website URL: `https://www.avocadogreenmattress.com`
2. Enter target query: `organic mattress benefits`
3. Click "Run Pipeline"
4. Review results across all tabs

### Advanced Analysis
- Adjust max pages and chunks per page in sidebar
- Explore detailed XAI analysis for comprehensive insights
- Use channel-specific recommendations for targeted optimization

## 📊 Performance Expectations

- **Crawling**: 2-5 seconds per page
- **Chunking**: 1-2 seconds for processing
- **Embedding**: 3-5 seconds for index creation
- **Query Generation**: 10-15 seconds for reverse queries
- **Fan-out Expansion**: 15-20 seconds for query expansion
- **Comprehensive Scoring**: 5-10 seconds for analysis
- **Total Pipeline**: 1-2 minutes for complete analysis

## 🛠️ Technical Stack

- **Web Crawling**: Custom crawler with BeautifulSoup
- **Content Processing**: Semantic chunking with tiktoken
- **Embeddings**: Google Gemini embeddings
- **Search**: FAISS vector similarity search
- **AI Generation**: Google Gemini for query generation
- **Frontend**: Streamlit dashboard
- **Analysis**: Custom comprehensive scoring engine

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the logs tab in the Streamlit app
2. Review the pipeline flow diagrams above
3. Ensure all API keys are properly configured
4. Verify the MLHW conda environment is active

---

**🧭 Zero-Click Compass** - Navigate the AI-powered content landscape with confidence! 