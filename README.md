# Zero-Click Compass 🧭

**LLM-first website performance analysis** - See how your content ranks in an AI-driven, zero-click world.

## What & Why

Traditional SEO focuses on getting clicks. But in the AI era, content needs to rank **inside** LLM responses where users never visit your site.

Zero-Click Compass analyzes your website content as if it were competing for ranking in AI-generated answers, not search results.

## How It Works

1. **Crawls** your website and chunks content into ~150-token semantic passages
2. **Embeds** chunks using Google Gemini and indexes with FAISS
3. **Expands** queries into intent trees (synonyms, questions, variations)
4. **Scores** relevance using 70% semantic similarity + 30% token overlap
5. **Discovers** social media chatter from Reddit and Twitter
6. **Outputs** `visibility.csv` (content performance) and `channels.json` (social analysis)

## Quick Start

### Install
```bash
conda activate MLHW  # or your preferred env
pip install -r requirements.txt
```

### Run Dashboard
```bash
streamlit run app.py
```

### Run Pipeline
```bash
python -m src.cli pipeline https://your-site.com "your topic" --social
```

## Output Files

- `data/visibility.csv` - Content performance rankings
- `data/channels.json` - Social media influencer analysis
- `data/faiss_index.faiss` - Vector search index

## CLI Commands

```bash
# Full pipeline
python -m src.cli pipeline https://example.com "marketing strategies"

# Individual steps
python -m src.cli crawl https://example.com
python -m src.cli expand "content marketing"
python -m src.cli social "SEO tips" --analyze
```

## Environment Variables

Create `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
TWITTER_BEARER_TOKEN=your_twitter_token
```

## Architecture

```
src/
├── crawl.py      # Web scraping with Selenium
├── chunk.py      # Semantic chunking with tiktoken
├── embed.py      # Gemini embeddings + FAISS indexing
├── expand.py     # Query expansion & intent trees
├── score.py      # Multi-method relevance scoring
├── channels.py   # Social media analysis
└── cli.py        # Pipeline orchestration
```

## Tests

```bash
python run_tests.py
``` 