# 🧭 Zero-Click Compass

**Essential content analysis and optimization pipeline** - Streamlined for production use.

## 🎯 What It Does

Zero-Click Compass analyzes your website content and generates:
- **Reverse queries** from your content chunks
- **Fan-out sub-queries** for comprehensive coverage  
- **Content relevance scores** for optimization
- **Actionable recommendations** for content and channel strategy

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and setup
git clone <repository>
cd zero-click-compass

# Install dependencies
conda create -n MLHW python=3.10
conda activate MLHW
pip install -r requirements.txt

# Set environment variables
cp env.template .env
# Edit .env with your GOOGLE_API_KEY
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

### 3. Use CLI (Optional)
```bash
python -m src.cli pipeline https://example.com "your query" --max-pages 3
```

## 📊 What You'll See

### Streamlit Interface
- **Queries Tab**: All generated reverse queries and fan-out expansions
- **Scores Tab**: Content relevance scores and rankings
- **Recommendations Tab**: Content optimization and channel strategy
- **Logs Tab**: Detailed pipeline execution logs

### Key Metrics
- Pages crawled and chunks created
- Reverse queries generated from content
- Fan-out queries for comprehensive coverage
- Content relevance scores

## 🔧 Core Components

### Essential Files
- `app.py` - Streamlined Streamlit interface
- `src/cli.py` - Command-line pipeline
- `src/crawl.py` - Web crawling with Selenium
- `src/chunk.py` - Semantic content chunking
- `src/embed.py` - Embedding generation and FAISS indexing
- `src/query_generator.py` - Reverse query generation from content
- `src/query_fanout.py` - Query expansion and fan-out
- `src/utils.py` - Utility functions and logging

### Pipeline Flow
1. **Crawl** → Website content extraction
2. **Chunk** → Semantic content segmentation (5 chunks/page max)
3. **Embed** → Vector embeddings and search index
4. **Generate** → Reverse queries from content chunks
5. **Expand** → Fan-out query generation
6. **Score** → Content relevance scoring
7. **Recommend** → Optimization suggestions

## ⚙️ Configuration

### Streamlit Settings
- **Max Pages**: Limit pages to crawl (default: 3)
- **Max Chunks/Page**: Limit chunks per page (default: 5)
- **Target Query**: Your focus search term

### CLI Options
```bash
python -m src.cli pipeline <url> <query> \
  --max-pages 3 \
  --max-chunks 10 \
  --max-reverse-queries 2 \
  --max-fanout-per-query 3 \
  --delay 0.5
```

## 🎯 Use Cases

### Content Optimization
- Identify what queries your content answers
- Find content gaps and opportunities
- Optimize for high-relevance queries

### SEO Strategy
- Discover long-tail keyword opportunities
- Understand content performance
- Plan content expansion

### Channel Strategy
- Determine optimal content distribution
- Identify platform-specific opportunities
- Plan multi-channel content

## 📈 Performance

### Typical Pipeline Times
- **Crawling**: ~10s per page
- **Chunking**: ~0.1s
- **Embeddings**: ~2s for 5 chunks
- **Reverse Queries**: ~14s for 2 chunks
- **Fan-out**: ~30s for 2 queries
- **Total**: ~60s for complete analysis

### Optimizations
- Limited chunks per page (5 max)
- Process only top 2 chunks for reverse queries
- Process only top 2 reverse queries for fan-out
- Fresh data processing (no caching)

## 🔍 Example Output

### Generated Queries
```
🔄 Reverse Queries from Content:
- "organic mattress benefits" (Score: 9.2)
- "best organic mattress brands" (Score: 8.8)
- "organic vs conventional mattress" (Score: 8.5)

🌊 Fan-out Queries:
- "organic mattress for back pain"
- "certified organic mattress materials"
- "organic mattress price comparison"
```

### Content Scores
```
Content: "Our organic mattresses are made from..." (Score: 0.892)
Content: "Benefits of choosing organic..." (Score: 0.845)
```

### Recommendations
```
📝 Content Optimization:
🔴 Create step-by-step guides and tutorials (high priority)
🟡 Develop comparison content and product reviews (medium priority)

📢 Channel Strategy:
🔴 SEO: Optimize for long-tail keywords (high priority)
🟡 Social Media: Engage with community discussions (medium priority)
```

## 🛠️ Development

### Project Structure
```
zero-click-compass/
├── app.py                 # Streamlit interface
├── src/
│   ├── cli.py            # Command-line interface
│   ├── crawl.py          # Web crawling
│   ├── chunk.py          # Content chunking
│   ├── embed.py          # Embeddings and search
│   ├── query_generator.py # Reverse query generation
│   ├── query_fanout.py   # Query expansion
│   └── utils.py          # Utilities
├── data/                 # Output files
└── requirements.txt      # Dependencies
```

### Key Features
- **No caching**: Fresh data processing every run
- **Detailed logging**: Full pipeline visibility
- **Optimized performance**: Limited processing for speed
- **Production ready**: Streamlined for real-world use

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built for content creators and marketers who want data-driven optimization insights.** 