# Backend Ingestion Guide

How to pull real-time news and tweets into the narrative vector database.

---

## Overview

There are two ways stories get ingested:

| Mode | Trigger | File |
|---|---|---|
| **Background pipeline** | Automatic, on a timer | `services/pipeline.py` |
| **Manual scrape** | `POST /api/ingest/scrape` | `api/routes/ingest.py` |

Both modes share the same dedup cache — the same story is never ingested twice in a session regardless of which mode fetches it.

The flow for every story is identical either way:

```
source (NewsAPI / Twitter)
        │
        ▼
  scraper.py          — fetch raw stories, filter by time window, deduplicate
        │
        ▼
  narrative_engine.py — embed story, find nearest narrative direction in ChromaDB
        │
        ├── distance < 0.40  →  update existing narrative's Surprise + Impact time series
        └── distance ≥ 0.40  →  create new narrative direction
```

---

## Setup

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Get API keys

**NewsAPI** (free tier — 100 req/day, articles up to 1 month old)
1. Register at https://newsapi.org/register
2. Copy your API key

**Twitter/X API v2** (free Basic tier — app-only auth, no user login)
1. Go to https://developer.twitter.com/en/portal/dashboard
2. Create a project → create an app → enable "Read" permissions
3. Copy the **Bearer Token** (not the API key — the bearer token)

### 3. Create your `.env`

```bash
cp .env.example ../.env   # from the backend/ directory
```

Fill in:

```env
NEWSAPI_KEY=your_newsapi_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
```

Leave a key blank to disable that source entirely — the scraper silently skips sources with no credentials.

### 4. Start the server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The background pipeline starts automatically on server boot.

---

## Manual Scraping — `POST /api/ingest/scrape`

This is the primary way to control exactly what gets ingested.

### Minimal request

```bash
curl -X POST http://localhost:8000/api/ingest/scrape \
  -H "Content-Type: application/json" \
  -d '{}'
```

Uses all defaults: last 60 minutes, 50 stories per source, all sources.

---

## Controlling Time Recency — `lookback_minutes`

This is the most important parameter. It sets how far back in time to look.

```bash
# Last 30 minutes — very fresh, low volume
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"lookback_minutes": 30}'

# Last hour (default)
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"lookback_minutes": 60}'

# Last 6 hours — broader sweep
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"lookback_minutes": 360}'

# Last 24 hours — max useful lookback for NewsAPI free tier
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"lookback_minutes": 1440}'
```

> **Note:** NewsAPI free tier only returns articles up to 1 month old, but rate-limits you to 100 requests/day. Twitter recent search covers the last 7 days.

---

## Controlling Volume — `max_per_source`

Caps how many items are fetched per source per run. Useful when you're testing or want to limit API usage.

```bash
# Small test batch — 10 from each source
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"lookback_minutes": 60, "max_per_source": 10}'

# Max volume — 100 from each source
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"lookback_minutes": 360, "max_per_source": 100}'
```

---

## Choosing Sources — `sources`

```bash
# NewsAPI only (no Twitter)
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"sources": ["newsapi"]}'

# Twitter only
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"sources": ["twitter"]}'

# Both (default)
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"sources": ["newsapi", "twitter"]}'
```

---

## Dry Run — Preview Without Ingesting

Fetches and deduplicates stories but does **not** write anything to ChromaDB.
Returns a preview of the first 20 stories that would be ingested.

Use this to tune your params before committing.

```bash
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"lookback_minutes": 60, "dry_run": true}' | python3 -m json.tool
```

Example output:

```json
{
  "fetched": 34,
  "duplicates_skipped": 6,
  "ingested": 0,
  "narratives_created": 0,
  "narratives_updated": 0,
  "dry_run": true,
  "stories_preview": [
    {
      "headline": "Fed signals rate cut may be delayed amid sticky inflation",
      "source": "newsapi:Reuters",
      "body": "Federal Reserve officials indicated Wednesday that..."
    },
    {
      "headline": "Oil prices surge after OPEC+ announces surprise production cut",
      "source": "newsapi:CNBC",
      "body": "Crude oil jumped more than 4% on Thursday..."
    }
  ]
}
```

---

## Custom Query Overrides

The default queries are broad financial/geopolitical filters. Override them for a focused ingestion run.

```bash
# Focus NewsAPI on banking stress only
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{
    "sources": ["newsapi"],
    "lookback_minutes": 360,
    "news_query": "\"bank run\" OR \"banking crisis\" OR \"bank collapse\" OR FDIC OR SVB"
  }'

# Focus Twitter on a specific event
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{
    "sources": ["twitter"],
    "lookback_minutes": 120,
    "twitter_query": "(tariffs OR \"trade war\" OR USMCA OR WTO) -is:retweet lang:en"
  }'
```

### Default queries (for reference)

**NewsAPI default:**
```
economy OR inflation OR recession OR "federal reserve" OR "interest rates"
OR "stock market" OR GDP OR "trade war" OR sanctions OR geopolitical
OR "supply chain" OR "central bank" OR "banking crisis" OR "credit risk"
```

**Twitter default:**
```
(economy OR inflation OR recession OR "federal reserve" OR "stock market"
OR sanctions OR geopolitical OR "banking crisis" OR "supply chain"
OR "credit risk" OR "rate hike") -is:retweet lang:en
```

---

## Reading the Response

A successful ingest run returns:

```json
{
  "fetched": 42,
  "duplicates_skipped": 8,
  "ingested": 41,
  "narratives_created": 3,
  "narratives_updated": 38,
  "errors": 1,
  "duration_seconds": 18.4,
  "dedup_cache_size": 312,
  "per_source": {
    "newsapi": 28,
    "twitter": 22
  },
  "narratives_touched": [
    {
      "id": "uuid",
      "name": "Federal Reserve monetary tightening",
      "action": "updated",
      "model_risk": 0.61,
      "event_count": 29
    },
    {
      "id": "uuid",
      "name": "OPEC production policy shift",
      "action": "created",
      "model_risk": 0.74,
      "event_count": 1
    }
  ]
}
```

| Field | What it tells you |
|---|---|
| `fetched` | Stories that passed dedup and are new this run |
| `duplicates_skipped` | Already seen in this session — correctly ignored |
| `narratives_created` | New narrative directions discovered this run |
| `narratives_updated` | Existing narratives that received new data points |
| `errors` | Stories that failed to process (check server logs) |
| `dedup_cache_size` | Session-wide unique stories seen so far |
| `narratives_touched` | Exactly which narratives changed and how |

---

## Staged Mode — Collect Now, Process Later

By default, `POST /api/ingest/scrape` embeds and routes stories immediately. Add `"buffer": true` to instead hold stories in memory and commit them to ChromaDB whenever you choose.

### Step 1 — Collect into the buffer

```bash
curl -X POST http://localhost:8000/api/ingest/scrape \
  -H "Content-Type: application/json" \
  -d '{"lookback_minutes": 60, "buffer": true}'
```

Response:

```json
{
  "fetched": 34,
  "duplicates_skipped": 6,
  "ingested": 0,
  "narratives_created": 0,
  "narratives_updated": 0,
  "buffer_mode": true,
  "buffer_size": 34,
  "dry_run": false
}
```

Nothing has been sent to Modal or ChromaDB yet. You can run this multiple times to accumulate stories from different queries or time windows.

### Step 2 — Inspect the buffer (optional)

```bash
curl http://localhost:8000/api/pipeline/buffer
```

```json
{
  "count": 34,
  "oldest_at": 1709120000.0,
  "newest_at": 1709123400.0,
  "preview": [
    {"headline": "Fed signals rate cut may be delayed...", "source": "newsapi:Reuters"},
    {"headline": "Oil prices surge after OPEC+ announcement", "source": "newsapi:CNBC"}
  ]
}
```

### Step 3 — Process when ready

```bash
curl -X POST http://localhost:8000/api/pipeline/process
```

This drains the buffer, sends all texts to Modal in **one** `embed_batch()` call, then routes each embedding into ChromaDB.

```json
{
  "processed": 34,
  "narratives_created": 4,
  "narratives_updated": 30,
  "errors": 0,
  "duration_seconds": 12.3,
  "buffer_remaining": 0,
  "narratives_touched": [
    {"id": "uuid", "name": "Federal Reserve monetary policy", "action": "updated", "model_risk": 0.61, "event_count": 29}
  ]
}
```

Optionally cap how many stories to process in one call (leaves the rest in the buffer):

```bash
curl -X POST http://localhost:8000/api/pipeline/process \
  -d '{"max_stories": 50}'
```

### Discard the buffer

```bash
curl -X DELETE http://localhost:8000/api/pipeline/buffer
```

```json
{"cleared": 34, "buffer_remaining": 0}
```

---

## Background Pipeline

The background pipeline is **disabled by default**. To enable automatic scraping on server boot, set in `.env`:

```env
AUTO_START_PIPELINE=true
```

Then configure the polling schedule:

```env
POLL_INTERVAL_SECONDS=300      # scrape every 5 minutes
PIPELINE_LOOKBACK_MINUTES=10   # only pull stories from last 10 minutes per poll
PIPELINE_MAX_PER_SOURCE=30     # keep auto-polls lightweight
PIPELINE_SOURCES=["newsapi","twitter"]
```

**Recommended relationship:** `PIPELINE_LOOKBACK_MINUTES ≈ POLL_INTERVAL_SECONDS / 60`
This keeps coverage continuous without overlap. At 5-minute polls looking back 10 minutes, you have a 5-minute overlap buffer to catch slow-propagating articles.

Check pipeline status at any time:

```bash
curl http://localhost:8000/api/pipeline/stats
```

```json
{
  "pipeline": {
    "stories_ingested": 482,
    "narratives_created": 23,
    "narratives_updated": 459,
    "errors": 2,
    "last_poll_at": 1709123400.0,
    "queue_size": 0
  },
  "narratives": {"total": 31, "active": 28},
  "events": {"total_events_ingested": 482}
}
```

---

## Tuning the Narrative Threshold

The threshold controls when a story creates a new narrative vs. updates an existing one.

```env
NEW_NARRATIVE_THRESHOLD=0.40   # default
```

| Value | Effect |
|---|---|
| `0.30` | Tighter clustering — fewer, broader narratives |
| `0.40` | Default — good balance |
| `0.50` | Looser clustering — more, narrower narratives |

**Symptoms of threshold being too high:** lots of single-event narratives, narrative count grows very fast relative to stories ingested.

**Symptoms of threshold being too low:** unrelated stories are merged into one narrative, descriptions become vague.

Monitor the ratio: `narratives_created / stories_ingested`. Healthy range is roughly 0.05–0.20 (1 new narrative per 5–20 stories).

---

## Workflow Recipes

### Initial database population (first run)

```bash
# Pull the last 24 hours at max volume to seed the database
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{"lookback_minutes": 1440, "max_per_source": 100}'
```

### Ongoing operation

Let the background pipeline handle it. Just keep the server running.

### Focused event ingestion (e.g., breaking news)

```bash
# Pull the last 2 hours with a targeted query
curl -X POST http://localhost:8000/api/ingest/scrape \
  -d '{
    "lookback_minutes": 120,
    "max_per_source": 100,
    "news_query": "\"Silicon Valley Bank\" OR SVB OR \"bank failure\" OR FDIC",
    "twitter_query": "(SVB OR \"bank run\" OR \"bank failure\") -is:retweet lang:en"
  }'
```

### Check what's in the database after ingesting

```bash
# List all narrative directions sorted by risk
curl "http://localhost:8000/api/narratives?sort_by=risk&active_only=true"

# Global risk index
curl http://localhost:8000/api/risk

# Semantic search
curl -X POST http://localhost:8000/api/narratives/search \
  -d '{"query": "central bank interest rate decisions"}'
```

---

## File Reference

| File | Purpose |
|---|---|
| `services/scraper.py` | `ScrapeParams`, `scrape_newsapi()`, `scrape_twitter()`, `scrape()`, `DeduplicatingCache` |
| `services/story_buffer.py` | Thread-safe in-memory buffer; `buffer` singleton |
| `services/pipeline.py` | Background polling loop, `pipeline_stats` |
| `services/narrative_engine.py` | `ingest_story()`, `route_with_embedding()` — routes stories into ChromaDB |
| `api/routes/ingest.py` | `POST /api/ingest/scrape` (with `buffer` flag), `POST /api/ingest`, `POST /api/ingest/batch` |
| `api/routes/pipeline.py` | `GET/DELETE /api/pipeline/buffer`, `POST /api/pipeline/process`, `GET /api/pipeline/stats` |
| `core/config.py` | All env-var settings with defaults |
| `.env` | Your actual credentials and config (gitignored) |
