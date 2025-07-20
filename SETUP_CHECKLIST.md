# Zero-Click Compass Setup Checklist 🚀

Complete list of everything you need to provide to get the system running.

## ✅ System Requirements

### Python Environment
- [ ] **Python 3.11+** (You have: Python 3.11.0 ✅)
- [ ] **Conda Environment** `MLHW` (Currently active ✅)
- [ ] **pip** for package installation

### Browser Dependencies (for Web Crawling)
- [ ] **Google Chrome** or **Chromium** installed
  - macOS: `brew install chromium` or download Chrome
  - Ubuntu: `sudo apt-get install chromium-browser`
  - Windows: Download from chrome.google.com
- [ ] **ChromeDriver** (auto-installed by webdriver-manager)

## 📦 Package Dependencies

### Install Required Packages
```bash
conda activate MLHW
pip install -r requirements.txt
```

**Core Dependencies:**
- [ ] `tiktoken>=0.5.0` - Token counting
- [ ] `google-generativeai>=0.3.0` - Gemini embeddings
- [ ] `faiss-cpu>=1.7.0` - Vector search
- [ ] `streamlit>=1.28.0` - Web dashboard
- [ ] `pandas>=2.0.0` - Data analysis
- [ ] `numpy>=1.24.0` - Numerical computing

**Web Scraping:**
- [ ] `selenium>=4.15.0` - Browser automation
- [ ] `webdriver-manager>=4.0.0` - ChromeDriver management
- [ ] `beautifulsoup4>=4.12.0` - HTML parsing
- [ ] `requests>=2.31.0` - HTTP requests

**Social Media APIs:**
- [ ] `praw>=7.7.0` - Reddit API
- [ ] `tweepy>=4.14.0` - Twitter API
- [ ] `requests-oauthlib>=1.3.0` - OAuth authentication

## 🔐 API Keys & Credentials (Required)

### 1. Google Gemini API Key (REQUIRED)
```bash
# Get from: https://ai.google.dev/
export GOOGLE_API_KEY="AIzaSyCMesgnSKQmDHvsLkH42DerPn5KUr1PCME"
```
- [ ] Sign up at [Google AI Studio](https://ai.google.dev/)
- [ ] Create new API key
- [ ] Set `GOOGLE_API_KEY` environment variable

### 2. Reddit API Credentials (Optional - for social analysis)
```bash
# Get from: https://www.reddit.com/prefs/apps
export REDDIT_CLIENT_ID="your_reddit_client_id_here"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret_here"
export REDDIT_USER_AGENT="ZeroClickCompass/1.0 by /u/yourusername"
```
- [ ] Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
- [ ] Create new app (type: "script")
- [ ] Get Client ID and Client Secret
- [ ] Set Reddit environment variables

### 3. Twitter/X API Credentials (Optional - for social analysis)
```bash
# Get from: https://developer.twitter.com/
export TWITTER_BEARER_TOKEN="your_twitter_bearer_token_here"
export TWITTER_API_KEY="your_twitter_api_key_here"
export TWITTER_API_SECRET="your_twitter_api_secret_here"
export TWITTER_ACCESS_TOKEN="your_twitter_access_token_here"
export TWITTER_ACCESS_TOKEN_SECRET="your_twitter_access_token_secret_here"
```
- [ ] Apply for [Twitter Developer Account](https://developer.twitter.com/)
- [ ] Create new app and get API keys
- [ ] Set Twitter environment variables

## 📁 Configuration Files

### 1. Environment Variables (.env file)
Create `.env` file in project root:
```bash
cp env.template .env
# Edit .env with your actual API keys
```
- [ ] Copy `env.template` to `.env`
- [ ] Fill in your actual API keys (especially `GOOGLE_API_KEY`)
- [ ] Adjust pipeline configuration values (optional)

### 2. Optional: JSON Configuration (config.json)
```json
{
  "embedding": {
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "batch_size": 10
  },
  "crawler": {
    "max_pages": 50,
    "delay": 1.0
  },
  "scoring": {
    "semantic_weight": 0.7,
    "token_overlap_weight": 0.3
  }
}
```

## 🗂️ Directory Structure
The system will auto-create these directories:
- [ ] `data/` - Generated data files
- [ ] `data/crawled_pages.jsonl` - Scraped website content
- [ ] `data/chunks.jsonl` - Semantic content chunks
- [ ] `data/faiss_index.faiss` - Vector search index
- [ ] `data/visibility.csv` - Content performance report
- [ ] `data/channels.json` - Social media analysis

## 🧪 Test Your Setup

### 1. Quick Smoke Test
```bash
python run_tests.py
```
- [ ] All imports work
- [ ] Basic functionality tests pass

### 2. Test API Connections
```bash
# New SOLID CLI
python -m src.presentation.cli status

# Legacy CLI
python -m src.cli status
```
- [ ] Google API key validates
- [ ] Reddit API connects (if configured)
- [ ] Twitter API connects (if configured)

### 3. Test Web Crawling
```bash
# Test single page crawl
python -c "from src.crawl import crawl_single_page; print('✅ Crawling works' if crawl_single_page('https://example.com') else '❌ Crawling failed')"
```
- [ ] Selenium WebDriver initializes
- [ ] Chrome/Chromium launches
- [ ] Can crawl web pages

## 🚀 Run the System

### Method 1: New SOLID CLI
```bash
python -m src.presentation.cli pipeline https://example.com "marketing strategies"
```

### Method 2: Legacy CLI
```bash
python -m src.cli pipeline https://example.com "marketing strategies" --social
```

### Method 3: Web Dashboard
```bash
streamlit run app.py
# Or use the helper script:
./run_app.sh
```

## 📊 Input Data Required

### For Pipeline Execution:
- [ ] **Website URL** - Target website to analyze
- [ ] **Analysis Query** - What topic to analyze (e.g., "marketing strategies")
- [ ] **Configuration Parameters** (optional):
  - Max pages to crawl (default: 50)
  - Max query expansions (default: 15)
  - Top K results (default: 10)
  - Include social media analysis (true/false)

### Example Command:
```bash
python -m src.presentation.cli pipeline \
  https://contentmarketing.com \
  "content marketing best practices" \
  --max-pages 100 \
  --social
```

## 🔧 Troubleshooting

### Common Issues:
- [ ] **Chrome not found**: Install Google Chrome or Chromium
- [ ] **API key invalid**: Check Google AI Studio for correct key
- [ ] **Import errors**: Ensure you're in the MLHW conda environment
- [ ] **Permission errors**: Check file permissions in data/ directory
- [ ] **Social media fails**: Reddit/Twitter APIs are optional, pipeline continues without them

### Debug Commands:
```bash
# Check environment
conda info --envs
python --version
which python

# Check imports
python -c "import tiktoken, google.generativeai, faiss, streamlit; print('✅ All imports work')"

# Check browser
python -c "from selenium import webdriver; from webdriver_manager.chrome import ChromeDriverManager; print('✅ Browser setup works')"
```

## 📋 Minimum Setup (Google API Only)

If you want to get started quickly with just the core functionality:

1. **Install packages**: `pip install -r requirements.txt`
2. **Get Google API key**: [ai.google.dev](https://ai.google.dev/)
3. **Set environment**: `export GOOGLE_API_KEY="your_key"`
4. **Run pipeline**: `python -m src.cli pipeline https://example.com "your query"`

This will work without Reddit/Twitter APIs (social analysis will be skipped).

## ✅ Success Indicators

You'll know it's working when:
- [ ] Smoke tests pass: `python run_tests.py`
- [ ] Status check passes: `python -m src.presentation.cli status`
- [ ] Pipeline runs successfully and generates:
  - `data/visibility.csv` with content rankings
  - `data/chunks.jsonl` with semantic chunks
  - `data/faiss_index.faiss` with vector index
- [ ] Streamlit dashboard loads at `http://localhost:8503`

---

**🎯 Ready to analyze your content for the LLM-first world!** 