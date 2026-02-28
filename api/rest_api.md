# REST API Reference

**Base URL:** `http://localhost:8000`
**Implemented by:** `backend/`
**Consumed by:** `frontend/`

All request and response bodies are JSON unless noted.
All timestamps are Unix epoch seconds (float).
All score values are floats in [0.0, 1.0].

---

## Health

### GET /health

```
Response 200:
{
  "status": "ok"
}
```

---

## Ingest

### POST /api/ingest

Manually ingest a single news story. The backend embeds it, routes it to the
nearest narrative direction (or creates a new one), and scores it.

```
Request:
{
  "headline": "Russia halts gas transit through Ukraine",   // required
  "body":     "Full article text...",                       // optional, default ""
  "source":   "manual"                                      // optional
}

Response 200:
{
  "action":               "created" | "updated",
  "narrative_id":         "uuid",
  "narrative_name":       "European energy supply shock",
  "best_distance":        0.18,        // null if DB was empty (forced create)
  "threshold":            0.40,        // cosine distance threshold in use
  "current_surprise":     0.82,
  "current_impact":       0.78,
  "model_risk":           0.80,        // sqrt(surprise × impact)
  "narrative_event_count": 1
}
```

### POST /api/ingest/scrape

Fetch fresh stories from live data sources and ingest them into ChromaDB.
**This is the primary way to populate the narrative database.**

All time-recency and volume knobs are exposed as request parameters.

```
Request:
{
  "lookback_minutes": 60,              // only pull stories from last N minutes
                                       //   30   → very fresh, low volume
                                       //   60   → last hour (default)
                                       //   360  → last 6 hours
                                       //   1440 → last 24 hours (NewsAPI max)

  "max_per_source":   50,              // max stories per source (1–100)

  "sources":          ["newsapi", "twitter"],   // which sources to use
                                                // omit a source to skip it

  "news_query":       null,            // override default NewsAPI keyword query
                                       // null = use the built-in financial query
  "twitter_query":    null,            // override default Twitter v2 query
                                       // null = use the built-in financial query

  "dry_run":          false            // if true: fetch + deduplicate but DON'T
                                       // ingest. Returns preview of what would
                                       // be ingested. Use to tune params.
}

Response 200:
{
  "fetched":              42,          // stories passing dedup (new this run)
  "duplicates_skipped":   8,           // already seen in this session
  "ingested":             41,          // successfully processed
  "narratives_created":   3,           // new narrative directions spawned
  "narratives_updated":   38,          // existing narratives updated
  "errors":               1,           // stories that failed to ingest
  "duration_seconds":     18.4,
  "dedup_cache_size":     312,         // total entries in session dedup cache
  "per_source": {
    "newsapi":  28,
    "twitter":  22
  },
  "narratives_touched": [
    {
      "id":         "uuid",
      "name":       "European energy supply shock",
      "action":     "updated",
      "model_risk": 0.77,
      "event_count": 41
    }
    // one entry per unique narrative touched this run
  ],
  "dry_run": false,
  "stories_preview": null              // populated only when dry_run=true
                                       // contains first 20 stories as
                                       // [{headline, source, body}]
}
```

**Recommended usage patterns:**

```bash
# Quick test — see what would be ingested without committing
POST /api/ingest/scrape  {"lookback_minutes": 60, "dry_run": true}

# Normal run — last hour from all sources
POST /api/ingest/scrape  {"lookback_minutes": 60}

# Bulk backfill — last 6 hours, max volume
POST /api/ingest/scrape  {"lookback_minutes": 360, "max_per_source": 100}

# News only, last 30 minutes
POST /api/ingest/scrape  {"lookback_minutes": 30, "sources": ["newsapi"]}

# Twitter only, custom topic
POST /api/ingest/scrape  {
  "sources": ["twitter"],
  "twitter_query": "(SVB OR 'silicon valley bank' OR 'bank run') -is:retweet lang:en"
}
```

### POST /api/ingest/batch

```
Request:
{
  "stories": [
    { "headline": "...", "body": "..." },
    { "headline": "...", "body": "..." }
  ],
  "max_stories": 50     // optional, default 50, max 100
}

Response 200:
{
  "processed":        2,
  "results":          [ /* same shape as single ingest response, one per story */ ],
  "errors":           [ "story headline: error message" ],
  "duration_seconds": 4.21
}
```

---

## Narratives

### GET /api/narratives

List narrative directions. Defaults to all narratives sorted by risk descending.

```
Query params:
  active_only  bool    false   only return narratives updated in last 48h
  sort_by      string  "risk"  "risk" | "events" | "updated" | "created"
  limit        int     50      max 200

Response 200:
{
  "narratives": [
    {
      "id":               "uuid",
      "name":             "European energy supply shock",
      "description":      "Ongoing deterioration of Russian gas exports...",
      "event_count":      41,
      "current_surprise": 0.81,
      "current_impact":   0.74,
      "model_risk":       0.77,
      "last_updated":     1709123456.789,
      "is_active":        true,
      "surprise_trend":   "rising",    // "rising" | "falling" | "stable" | null
      "impact_trend":     "stable"
    }
  ],
  "total": 1
}
```

### GET /api/narratives/{id}

Full narrative detail including complete time series.

```
Response 200:
{
  "id":               "uuid",
  "name":             "...",
  "description":      "...",
  "created_at":       1709000000.0,
  "last_updated":     1709123456.789,
  "event_count":      41,
  "surprise_series":  [ { "timestamp": 1709000000.0, "value": 0.72 }, ... ],
  "impact_series":    [ { "timestamp": 1709000000.0, "value": 0.65 }, ... ],
  "recent_headlines": [ "headline1", "headline2", ... ],
  "is_active":        true,
  "model_risk":       0.77,
  "current_surprise": 0.81,
  "current_impact":   0.74,
  "surprise_trend":   "rising",
  "impact_trend":     "stable"
}

Response 404:
{ "detail": "Narrative not found" }
```

### GET /api/narratives/{id}/history

Time series data ready for charting.

```
Response 200:
{
  "id":                 "uuid",
  "name":               "...",
  "description":        "...",
  "surprise_series":    [ { "timestamp": float, "value": float }, ... ],
  "impact_series":      [ { "timestamp": float, "value": float }, ... ],
  "model_risk_series":  [ { "timestamp": float, "value": float }, ... ],
  "event_count":        41,
  "recent_headlines":   [ "..." ]
}
```

`model_risk_series` is derived as `sqrt(surprise × impact)` at each timestamp.

### POST /api/narratives/search

Semantic search — find narrative directions closest in meaning to a query string.

```
Request:
{
  "query":       "sanctions affecting energy exports",   // required
  "n_results":   5,                                      // optional, default 5, max 20
  "active_only": false                                   // optional
}

Response 200:
{
  "query": "sanctions affecting energy exports",
  "results": [
    {
      "narrative":   { /* same shape as GET /api/narratives item */ },
      "distance":    0.22,    // cosine distance [0, 2]; lower = more similar
      "similarity":  0.89     // 1 - (distance / 2); easier to display
    }
  ],
  "total": 3
}
```

### DELETE /api/narratives/{id}

```
Response 200:
{ "deleted": true, "id": "uuid" }

Response 404:
{ "detail": "Narrative not found" }
```

---

## Risk Index

### GET /api/risk

Global Real-World Model Risk Index across all (or active) narratives.

```
Query params:
  method       string  "max"    "max" | "mean" | "weighted"
                                max      = single highest-risk narrative drives the index
                                mean     = average across narratives
                                weighted = impact-weighted mean
  active_only  bool    false    only include narratives updated in last 48h

Response 200:
{
  "model_risk_index":       0.77,         // null if no narratives exist
  "narrative_count":        23,
  "active_narrative_count": 11,
  "breakdown": [
    {
      "id":           "uuid",
      "name":         "European energy supply shock",
      "surprise":     0.81,
      "impact":       0.74,
      "model_risk":   0.77,
      "event_count":  41,
      "is_active":    true,
      "last_updated": 1709123456.789
    }
    // sorted by model_risk descending
  ],
  "aggregation_method": "max",
  "computed_at":        1709123456.789
}
```

**Interpretation:**
- `0.00 – 0.33` → LOW — markets behaving statistically, models reliable
- `0.34 – 0.66` → MEDIUM — narratives building, monitor closely
- `0.67 – 1.00` → HIGH — regime shift / model fragility

### GET /api/risk/history

Historical risk index for charting.

```
Query params:
  window      int   24    look-back in hours (1–168)
  resolution  int   100   number of data points (10–500)

Response 200:
{
  "history": [
    { "timestamp": 1709100000.0, "model_risk_index": 0.61 },
    { "timestamp": 1709100864.0, "model_risk_index": 0.68 },
    // ...100 points
  ],
  "window_hours": 24,
  "resolution":   100,
  "start_time":   1709037456.789,
  "end_time":     1709123456.789
}
```

`model_risk_index` is null for buckets where no events occurred in that window.

---

## Chat (RAG)

### POST /api/chat

Ask a natural language question. The backend retrieves the most semantically
similar narrative directions and uses the LLM to generate a grounded answer.

```
Request:
{
  "query":                 "Which narratives are most likely to break quant models right now?",
  "n_context_narratives":  5     // optional, default 5, max 10
}

Response 200:
{
  "query":  "Which narratives are most likely to break quant models right now?",
  "answer": "The three narratives currently posing the highest model risk are...",
  "context_narratives": [
    {
      "id":               "uuid",
      "name":             "...",
      "description":      "...",
      "current_surprise": 0.81,
      "current_impact":   0.74,
      "model_risk":       0.77,
      "recent_headlines": [ "..." ],
      "event_count":      41,
      "distance":         0.12
    }
  ]
}

Response 422:
{ "detail": "No narratives in database yet. Ingest some stories first." }
```

---

## Pipeline

### GET /api/pipeline/stats

Status of the automated news ingestion pipeline.

```
Response 200:
{
  "pipeline": {
    "stories_ingested":   482,
    "narratives_created": 23,
    "narratives_updated": 459,
    "errors":             2,
    "started_at":         1709050000.0,
    "last_ingested_at":   1709123400.0,
    "queue_size":         3
  },
  "narratives": {
    "total":  23,
    "active": 11
  },
  "events": {
    "total_events_ingested": 482
  },
  "timestamp": 1709123456.789
}
```

---

## Live Events (SSE)

See `sse_events.md` for the event stream format.

### GET /api/events/stream

Opens a persistent Server-Sent Events connection. The server pushes a message
after every story ingestion.

```
Response:  text/event-stream (persistent connection)
```
