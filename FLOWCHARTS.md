# 🔄 Zero-Click Compass Flowcharts

Detailed flowcharts showing the complete pipeline process.

## 🏗️ Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ZERO-CLICK COMPASS                                │
│                        Content Analysis Pipeline                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT PHASE                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Website   │    │   Target    │    │   Pipeline  │                     │
│  │     URL     │    │   Query     │    │  Settings   │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTENT DISCOVERY                                 │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Web       │───▶│   Content   │───▶│  Embedding  │                     │
│  │  Crawler    │    │   Chunker   │    │  Pipeline   │                     │
│  │             │    │             │    │             │                     │
│  │ • Max 3     │    │ • 150 tokens│    │ • FAISS     │                     │
│  │   pages     │    │ • 20 overlap│    │   index     │                     │
│  │ • 5 chunks  │    │ • Semantic  │    │ • Vector    │                     │
│  │   per page  │    │   splitting │    │   search    │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY GENERATION                                    │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Reverse   │    │   Query     │    │   Fan-out   │                     │
│  │   Query     │───▶│   Ranking   │───▶│  Generator  │                     │
│  │ Generator   │    │             │    │             │                     │
│  │             │    │             │    │             │                     │
│  │ • 2 queries │    │ • Top 5     │    │ • 28+       │                     │
│  │   per chunk │    │   queries   │    │   variations│                     │
│  │ • AI-based  │    │ • Relevance │    │ • Platform- │                     │
│  │   generation│    │   scoring   │    │   specific  │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANALYSIS PHASE                                    │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Content   │    │Comprehensive│    │   Channel   │                     │
│  │   Search    │───▶│   Scorer    │───▶│  Strategy   │                     │
│  │             │    │             │    │             │                     │
│  │ • Similarity│    │ • Chunk     │    │ • 6 platforms│                     │
│  │   matching  │    │   scoring   │    │ • Platform- │                     │
│  │ • Top 10    │    │ • Gap       │    │   specific  │                     │
│  │   results   │    │   analysis  │    │   strategies│                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT DASHBOARD                                  │
│                                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │   Queries   │ │   Scores    │ │ XAI Analysis│ │Recommendations│ │  Logs  │ │
│  │             │ │             │ │             │ │             │ │         │ │
│  │ • Reverse   │ │ • Content   │ │ • Summary   │ │ • Content   │ │ • Real- │ │
│  │   queries   │ │   quality   │ │   metrics   │ │   optimiz.  │ │   time  │ │
│  │ • Fan-out   │ │ • Relevance │ │ • Gaps      │ │ • Channel   │ │   logs  │ │
│  │   queries   │ │   scores    │ │   analysis  │ │   strategy  │ │ • Debug │ │
│  │ • Rankings  │ │ • Quality   │ │ • Platform  │ │ • Actions   │ │   info  │ │
│  │             │ │   indicators│ │   priorities│ │             │ │         │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Detailed Pipeline Flow

### Phase 1: Content Discovery
```
Website URL
    │
    ▼
┌─────────────┐
│ Web Crawler │
│             │
│ • Selenium  │
│ • Beautiful │
│   Soup      │
│ • Max 3     │
│   pages     │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Chunker   │
│             │
│ • tiktoken  │
│ • 150 tokens│
│ • 20 overlap│
│ • Semantic  │
│   splitting │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Embeddings  │
│             │
│ • Gemini    │
│   embeddings│
│ • FAISS     │
│   index     │
│ • Vector    │
│   search    │
└─────────────┘
```

### Phase 2: Query Generation
```
Content Chunks
    │
    ▼
┌─────────────┐
│   Reverse   │
│  Queries    │
│             │
│ • AI-based  │
│   generation│
│ • 2 queries │
│   per chunk │
│ • Relevance │
│   scoring   │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Fan-out   │
│  Generator  │
│             │
│ • Top 2     │
│   queries   │
│ • 28+       │
│   variations│
│ • Platform- │
│   specific  │
└─────────────┘
```

### Phase 3: Analysis & Scoring
```
Fan-out Queries
    │
    ▼
┌─────────────┐
│   Search    │
│             │
│ • FAISS     │
│   search    │
│ • Top 10    │
│   results   │
│ • Similarity│
│   scores    │
└─────────────┘
    │
    ▼
┌─────────────┐
│Comprehensive│
│   Scorer    │
│             │
│ • Chunk     │
│   scoring   │
│ • Gap       │
│   analysis  │
│ • Grades    │
│   (A-F)     │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Channel     │
│ Analysis    │
│             │
│ • 6 platforms│
│ • Platform- │
│   specific  │
│ • Priority  │
│   scoring   │
└─────────────┘
```

## 📊 Data Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw HTML  │───▶│  Clean Text │───▶│  Chunks     │
│   Pages     │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Queries   │◀───│   Content   │◀───│  Embeddings │
│  Generated  │    │   Analysis  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Fan-out   │    │   Scores    │    │  Strategies │
│   Queries   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────┐
                    │  Dashboard  │
                    │   Results   │
                    └─────────────┘
```

## 🎯 Decision Flow

```
Start Pipeline
    │
    ▼
┌─────────────┐
│ Input Valid?│───No───▶ Error Message
│             │
└─────────────┘
    │ Yes
    ▼
┌─────────────┐
│ Crawl Pages │───Fail───▶ Retry/Continue
│             │
└─────────────┘
    │ Success
    ▼
┌─────────────┐
│ Create      │───Fail───▶ Use Fallback
│ Chunks?     │
└─────────────┘
    │ Success
    ▼
┌─────────────┐
│ Generate    │───Fail───▶ Use Default
│ Queries?    │
└─────────────┘
    │ Success
    ▼
┌─────────────┐
│ Expand      │───Fail───▶ Use Original
│ Queries?    │
└─────────────┘
    │ Success
    ▼
┌─────────────┐
│ Score       │───Fail───▶ Basic Scoring
│ Content?    │
└─────────────┘
    │ Success
    ▼
┌─────────────┐
│ Generate    │───Fail───▶ Basic Recs
│ Recs?       │
└─────────────┘
    │ Success
    ▼
Display Results
```

## 🔧 Error Handling Flow

```
Pipeline Step
    │
    ▼
┌─────────────┐
│ Try Execute │
│   Step      │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Success?    │───Yes───▶ Continue
│             │
└─────────────┘
    │ No
    ▼
┌─────────────┐
│ Log Error   │
│             │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Retry Count │───< 3───▶ Retry Step
│ < Max?      │
└─────────────┘
    │ >= 3
    ▼
┌─────────────┐
│ Use Fallback│
│   Method    │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Continue    │
│ Pipeline    │
└─────────────┘
```

## 📈 Performance Flow

```
Pipeline Start
    │
    ▼
┌─────────────┐
│ Crawling    │───▶ 2-5s per page
│ Phase       │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Chunking    │───▶ 1-2s total
│ Phase       │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Embedding   │───▶ 3-5s total
│ Phase       │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Query Gen   │───▶ 10-15s
│ Phase       │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Fan-out     │───▶ 15-20s
│ Phase       │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Scoring     │───▶ 5-10s
│ Phase       │
└─────────────┘
    │
    ▼
Total: 1-2 minutes
```

---

**These flowcharts provide a visual guide to understanding the complete Zero-Click Compass pipeline process.** 