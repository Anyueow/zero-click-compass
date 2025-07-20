# 🔐 Environment Variables Setup Guide

## Overview

Zero-Click Compass now **consistently** loads all API keys and configuration from `.env` files. No more manual exports or hardcoded values!

## ✅ What I Fixed

### 1. Consistent .env Loading
- **All modules** now use `load_dotenv()` to load environment variables
- **Streamlit app** loads .env on startup and shows status
- **Configuration providers** automatically load .env files
- **Legacy CLI** already used .env properly

### 2. Comprehensive .env Template
Created `env.template` with **all possible configuration options**:
- ✅ Google Gemini API (required)
- ✅ Reddit API (optional)
- ✅ Twitter API (optional)  
- ✅ Pipeline configuration
- ✅ System settings

### 3. Improved Streamlit Dashboard
- ✅ Automatically detects if API key is loaded from .env
- ✅ Shows green checkmark when .env is properly configured
- ✅ Option to override temporarily if needed
- ✅ Clear error messages if keys are missing

## 🚀 Quick Setup

### Step 1: Copy Template
```bash
cp env.template .env
```

### Step 2: Edit .env File
```bash
# Open in your editor
nano .env
# or
code .env
```

### Step 3: Fill in Required Values
**Minimum required:**
```
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

**For social media analysis (optional):**
```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

### Step 4: Test Setup
```bash
# Check status
python -m src.presentation.cli status

# Or run Streamlit (will show green checkmark if .env loaded)
streamlit run app.py
```

## 📋 Complete .env Template

```bash
# ===== REQUIRED =====
GOOGLE_API_KEY=your_google_gemini_api_key_here

# ===== OPTIONAL - SOCIAL MEDIA =====
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=ZeroClickCompass/1.0 by /u/yourusername

TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

# ===== OPTIONAL - PIPELINE CONFIGURATION =====
CHUNK_SIZE=150
OVERLAP_TOKENS=20
EMBEDDING_MODEL=gemini-1.5-flash
EMBEDDING_BATCH_SIZE=10
MAX_PAGES=50
CRAWL_DEPTH=2
REQUEST_DELAY=1.0
CRAWL_TIMEOUT=30
MAX_EXPANSIONS=15
SEMANTIC_WEIGHT=0.7
TOKEN_OVERLAP_WEIGHT=0.3
TOP_K_RESULTS=10
SOCIAL_SEARCH_LIMIT=100
MAX_RETRIES=3
RETRY_DELAY=1.0
DATA_DIRECTORY=data
LOG_LEVEL=INFO
```

## 🔍 How It Works

### Environment Loading Priority
1. **`.env` file** (highest priority)
2. **System environment variables** 
3. **Default values** (fallback)

### Where .env is Loaded
- ✅ `src/utils.py` - Legacy modules
- ✅ `src/infrastructure/config.py` - SOLID architecture  
- ✅ `app.py` - Streamlit dashboard
- ✅ All API integrations (Gemini, Reddit, Twitter)

### Verification Commands
```bash
# Test environment loading
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('✅' if os.getenv('GOOGLE_API_KEY') else '❌', 'Google API Key')"

# Check all status
python -m src.presentation.cli status

# Test imports
python -c "from src.embed import EmbeddingGenerator; print('✅ Embedding generator initialized')"
```

## 🐛 Troubleshooting

### Issue: "GOOGLE_API_KEY not set"
**Solution:**
1. Ensure `.env` file exists in project root
2. Check the file contains `GOOGLE_API_KEY=your_actual_key`
3. No spaces around the `=` sign
4. No quotes around the value

### Issue: "Module not found" errors  
**Solution:**
1. Ensure you're in the correct conda environment: `conda activate MLHW`
2. Install dependencies: `pip install -r requirements.txt`

### Issue: Social media APIs fail
**Solution:**
- Social APIs are **optional** - the pipeline continues without them
- Only fill them in if you want social media analysis
- Leave them as template values if not needed

### Issue: Streamlit shows "Override with custom key"
**Solution:**
- This is normal - it means .env is loaded correctly
- You can enter keys manually if needed for testing
- The checkbox lets you override .env values temporarily

## 🎯 Success Indicators

You'll know it's working when:

1. **Streamlit Dashboard:**
   - ✅ Green "Google API key loaded from .env" message
   - No red error messages about missing keys

2. **CLI Status Check:**
   ```bash
   python -m src.presentation.cli status
   # Should show:
   # ✅ Google API key configured
   # ✅ Reddit API configured (if set)
   # ✅ Twitter API configured (if set)
   ```

3. **Pipeline Runs Successfully:**
   ```bash
   python -m src.cli pipeline https://example.com "test query"
   # Should not show any "API key not set" errors
   ```

## 🔐 Security Notes

- ✅ `.env` is in `.gitignore` - won't be committed to git
- ✅ `env.template` is tracked - safe to commit  
- ✅ Never commit actual API keys to version control
- ✅ Use different .env files for development/production

---

**🎉 Your API keys are now properly managed and secure!** 