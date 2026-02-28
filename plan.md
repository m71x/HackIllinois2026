# Real-World Model Risk Engine — Master Implementation Plan

> This document is the ground truth for building the system. Every section covers exact file paths,
> function signatures, data structures, prompts, API specs, and UI layout.
> Follow sections in order. Do not skip or summarize.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Ingestion & Scraping Pipeline](#2-data-ingestion--scraping-pipeline)
3. [Embedding Layer](#3-embedding-layer)
4. [ChromaDB & Narrative Direction Data Model](#4-chromadb--narrative-direction-data-model)
5. [Modal Model Integration](#5-modal-model-integration)
6. [Narrative Routing Engine](#6-narrative-routing-engine)
7. [Risk Index Computation](#7-risk-index-computation)
8. [FastAPI Backend — All Routes](#8-fastapi-backend--all-routes)
9. [Frontend UI](#9-frontend-ui)
10. [Configuration & Environment](#10-configuration--environment)
11. [End-to-End Data Flow Walkthrough](#11-end-to-end-data-flow-walkthrough)
12. [Design Decisions & Rationale](#12-design-decisions--rationale)
13. [Known Issues & Mitigations](#13-known-issues--mitigations)
14. [Complete File Map](#14-complete-file-map)

---

## 1. Architecture Overview

```
INTERNET NEWS SOURCES
  RSS feeds (CNBC, Reuters, MarketWatch, Yahoo Finance, Nasdaq)
  Reddit (r/stocks, r/investing, r/worldnews)
          │
          │ HTTP GET / Reddit API every 60s
          ▼
  scraper.py  (RawStory dataclass, DeduplicatingCache, async pollers)
          │
          │ RawStory objects → raw_queue (asyncio.Queue, maxsize=500)
          ▼
  pipeline.py  (N async workers pulling from raw_queue)
          │
          │ calls ingest_story(headline, body)
          ▼
  narrative_engine.py  ←──────────────────────────────────────────┐
    1. embedder.py: embed_text(story)                              │
    2. vector_store.py: query_nearest(embedding, n=5)             │
    3. if best_distance < threshold:                               │
         _update_narrative() → update ChromaDB                    │
       else:                                                       │
         _create_narrative() → add to ChromaDB                    │
          │                                                        │
          ▼                                                        │
  ChromaDB (./chroma_db/)                                         │
    Collection "narratives"                                        │
    Each doc = one NarrativeDirection                              │
    Fields: id, embedding[384], name, description,                │
            surprise_series (JSON), impact_series (JSON),         │
            event_count, recent_headlines, created_at, updated_at │
          │                                                        │
          ├─ GET /api/risk          → risk index + breakdown       │
          ├─ GET /api/narratives    → list all narrative dirs      │
          ├─ POST /api/narratives/search → semantic search         │
          ├─ GET /api/narratives/{id}/history → time series        │
          ├─ POST /api/ingest       → manual single story ingest   │
          ├─ POST /api/ingest/batch → manual batch ingest          │
          ├─ GET /api/pipeline/stats → scraper status              │
          ├─ POST /api/chat         → RAG Q&A over narratives      │
          └─ GET /api/events/stream → SSE live feed                │
                    │                                              │
                    ▼                                              │
          Frontend (index.html, styles.css, dashboard.js)          │
            Risk Gauge, 24h Chart, Narrative Table,               │
            Semantic Search, Live Feed, Manual Ingest             │
            SSE → live push updates ──────────────────────────────┘
```

---

## 2. Data Ingestion & Scraping Pipeline

### 2.1 RawStory Dataclass

**File:** `backend/services/scraper.py`

```python
from dataclasses import dataclass, field
import time

@dataclass
class RawStory:
    headline: str
    body: str
    source: str               # e.g. "cnbc_rss", "reuters_rss", "reddit_stocks"
    url: str = ""
    published_at: float = field(default_factory=time.time)
```

Every ingested item is first normalized into a `RawStory` before entering the queue.

### 2.2 DeduplicatingCache

**File:** `backend/services/scraper.py`

```python
import hashlib
from collections import OrderedDict

class DeduplicatingCache:
    """
    Fixed-capacity LRU cache of content hashes.
    Prevents the same story from being re-ingested across poll cycles.
    """
    def __init__(self, maxsize: int = 10_000):
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._maxsize = maxsize

    def _hash(self, headline: str, body: str) -> str:
        return hashlib.sha256(f"{headline}{body[:200]}".encode()).hexdigest()

    def is_seen(self, headline: str, body: str) -> bool:
        key = self._hash(headline, body)
        if key in self._cache:
            self._cache.move_to_end(key)
            return True
        return False

    def mark_seen(self, headline: str, body: str):
        key = self._hash(headline, body)
        self._cache[key] = True
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
```

### 2.3 RSS Poller

**File:** `backend/services/scraper.py`

RSS sources to poll (configurable in `config.py` as a list of strings):

```
https://www.cnbc.com/id/100003114/device/rss/rss.html          (markets)
https://feeds.a.dj.com/rss/RSSMarketsMain.xml                  (WSJ markets)
https://feeds.reuters.com/reuters/businessNews                  (Reuters business)
https://finance.yahoo.com/news/rssindex                        (Yahoo Finance)
https://www.investing.com/rss/news.rss                         (Investing.com)
https://feeds.marketwatch.com/marketwatch/topstories/          (MarketWatch)
```

```python
import asyncio
import aiohttp
import feedparser

async def rss_poller(
    sources: list[str],
    queue: asyncio.Queue,
    cache: DeduplicatingCache,
    poll_interval_seconds: int = 60,
):
    """
    Continuously polls RSS feeds and places new RawStory objects into queue.
    Runs forever; cancelled by pipeline shutdown.
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15)
    ) as session:
        while True:
            for url in sources:
                try:
                    async with session.get(url) as resp:
                        text = await resp.text()
                    feed = feedparser.parse(text)
                    for entry in feed.entries:
                        headline = entry.get("title", "").strip()
                        body = entry.get("summary", entry.get("description", "")).strip()
                        if not headline:
                            continue
                        if cache.is_seen(headline, body):
                            continue
                        cache.mark_seen(headline, body)
                        story = RawStory(
                            headline=headline,
                            body=body,
                            source=f"rss:{url.split('/')[2]}",
                            url=entry.get("link", ""),
                        )
                        try:
                            queue.put_nowait(story)
                        except asyncio.QueueFull:
                            pass  # drop if queue is full; prefer freshness
                except Exception:
                    pass  # silently skip failing feeds

            await asyncio.sleep(poll_interval_seconds)
```

### 2.4 Reddit Poller (optional, gated by config flag)

**File:** `backend/services/scraper.py`

Uses `praw` (Python Reddit API Wrapper). Only enabled if `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` are set.

```python
import praw

async def reddit_poller(
    subreddits: list[str],       # ["stocks", "investing", "worldnews"]
    queue: asyncio.Queue,
    cache: DeduplicatingCache,
    poll_interval_seconds: int = 90,
    posts_per_subreddit: int = 25,
):
    reddit = praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent="model-risk-engine/1.0",
    )
    loop = asyncio.get_event_loop()
    while True:
        for sub_name in subreddits:
            try:
                def fetch():
                    sub = reddit.subreddit(sub_name)
                    return list(sub.new(limit=posts_per_subreddit))
                posts = await loop.run_in_executor(None, fetch)
                for post in posts:
                    headline = post.title.strip()
                    body = (post.selftext or "")[:1000].strip()
                    if cache.is_seen(headline, body):
                        continue
                    cache.mark_seen(headline, body)
                    story = RawStory(
                        headline=headline,
                        body=body,
                        source=f"reddit:r/{sub_name}",
                        url=f"https://reddit.com{post.permalink}",
                        published_at=float(post.created_utc),
                    )
                    try:
                        queue.put_nowait(story)
                    except asyncio.QueueFull:
                        pass
            except Exception:
                pass
        await asyncio.sleep(poll_interval_seconds)
```

### 2.5 Pipeline Manager

**File:** `backend/services/pipeline.py`

```python
import asyncio
import time
from services.scraper import RawStory, DeduplicatingCache, rss_poller, reddit_poller
from services.narrative_engine import ingest_story
from core.config import settings

# Shared state accessible via /api/pipeline/stats
pipeline_stats = {
    "stories_ingested": 0,
    "narratives_created": 0,
    "narratives_updated": 0,
    "errors": 0,
    "started_at": None,
    "last_ingested_at": None,
}

raw_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
_cache = DeduplicatingCache(maxsize=10_000)
_tasks: list[asyncio.Task] = []


async def ingest_worker(worker_id: int):
    """
    Pulls RawStory objects from the queue and calls ingest_story().
    Runs in the asyncio event loop using run_in_executor for the
    blocking Cerebras + ChromaDB calls.
    """
    loop = asyncio.get_event_loop()
    while True:
        story: RawStory = await raw_queue.get()
        try:
            result = await loop.run_in_executor(
                None,
                lambda s=story: ingest_story(s.headline, s.body)
            )
            pipeline_stats["stories_ingested"] += 1
            pipeline_stats["last_ingested_at"] = time.time()
            if result["action"] == "created":
                pipeline_stats["narratives_created"] += 1
            else:
                pipeline_stats["narratives_updated"] += 1

            # Broadcast to SSE clients
            try:
                from api.routes.events import broadcast_event
                broadcast_event({"type": "ingest", "result": result, "timestamp": time.time()})
            except Exception:
                pass

        except Exception as e:
            pipeline_stats["errors"] += 1
        finally:
            raw_queue.task_done()


async def start_pipeline():
    pipeline_stats["started_at"] = time.time()
    num_workers = settings.pipeline_num_workers

    # Start RSS poller task
    _tasks.append(asyncio.create_task(
        rss_poller(
            sources=settings.rss_sources,
            queue=raw_queue,
            cache=_cache,
            poll_interval_seconds=settings.poll_interval_seconds,
        )
    ))

    # Start Reddit poller if configured
    if settings.reddit_client_id and settings.reddit_client_secret:
        _tasks.append(asyncio.create_task(
            reddit_poller(
                subreddits=settings.reddit_subreddits,
                queue=raw_queue,
                cache=_cache,
            )
        ))

    # Start ingest workers
    for i in range(num_workers):
        _tasks.append(asyncio.create_task(ingest_worker(i)))


async def stop_pipeline():
    for task in _tasks:
        task.cancel()
    _tasks.clear()
```

The pipeline is started and stopped via FastAPI's `lifespan` context manager in `main.py`:

```python
from contextlib import asynccontextmanager
from services.pipeline import start_pipeline, stop_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_pipeline()
    yield
    await stop_pipeline()

app = FastAPI(title="Real-World Model Risk Engine", lifespan=lifespan)
```

---

## 3. Embedding Layer

### 3.1 Model & Deployment

Model: `all-MiniLM-L6-v2` from `sentence-transformers`
- Output dimension: 384
- No HuggingFace token required (public model)
- Runs on Modal T4 GPU (faster than CPU, cheaper than A10G)

The embedding model is deployed as the `Embedder` class inside `model/modal_app.py`, alongside the `LLM` class. Both share the same Modal app (`model-risk-llm`) but use separate container classes with separate GPU types.

**Files:**
- `model/modal_app.py` — `Embedder` class definition (model team owns this)
- `backend/services/embedder.py` — thin client that calls Modal remotely (backend team owns this)

### 3.2 Embedder class (in modal_app.py)

```python
embed_image = modal.Image.debian_slim().pip_install("sentence-transformers")

@app.cls(gpu="T4", image=embed_image, container_idle_timeout=300)
class Embedder:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    @modal.method()
    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    @modal.method()
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()
```

### 3.3 embedder.py — Backend Client

```python
# backend/services/embedder.py
import modal
from core.config import settings

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = modal.Cls.lookup(settings.modal_app_name, "Embedder")
    return _embedder

def embed_text(text: str) -> list[float]:
    return _get_embedder()().embed.remote(text)

def embed_batch(texts: list[str]) -> list[list[float]]:
    return _get_embedder()().embed_batch.remote(texts)
```

The rest of the backend calls only `embed_text()` and `embed_batch()` — the Modal backend is transparent.

### 3.4 Embedding Strategy

Each narrative's stored embedding represents the **centroid** of all stories that have contributed to it. When a new story updates a narrative, the centroid is recomputed via online mean blending:

```
centroid_new = centroid_old * (n / (n + 1)) + story_vector * (1 / (n + 1))
```

This gives equal weight to all stories regardless of arrival order, keeping the centroid anchored to the full breadth of the narrative. The result is re-stored in ChromaDB via `collection.update()`.

The initial embedding for a new narrative is just the raw story embedding that triggered the narrative's creation.

---

## 4. ChromaDB & Narrative Direction Data Model

### 4.1 The Core Design Principle

**ChromaDB stores narrative directions, not individual news stories.**

A narrative direction is a persistent semantic category — a real-world story arc like "energy supply shock" or "China-US trade escalation". Individual news stories flow through the system but are not permanently stored. Their effect is captured by:
1. Updating the narrative centroid embedding (running average)
2. Appending a new point to the narrative's `surprise_series` and `impact_series`
3. Adding the headline to `recent_headlines` (capped at 10)

### 4.2 NarrativeDirection Pydantic Model

**File:** `backend/models/narrative.py`

```python
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import time

MAX_SERIES_LENGTH = 500  # cap time series at 500 points per narrative

class TimeSeriesPoint(BaseModel):
    timestamp: float   # unix epoch seconds
    value: float       # [0.0, 1.0]


class NarrativeDirection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str                        # short LLM-generated label (3–6 words)
    description: str                 # one-sentence LLM-generated summary
    created_at: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)
    event_count: int = 0             # total stories that have contributed

    surprise_series: list[TimeSeriesPoint] = []   # Surprise(t) over time
    impact_series: list[TimeSeriesPoint] = []     # Impact(t) over time
    recent_headlines: list[str] = []              # last 10 contributing headlines

    # --- Computed properties ---

    @property
    def current_surprise(self) -> Optional[float]:
        return self.surprise_series[-1].value if self.surprise_series else None

    @property
    def current_impact(self) -> Optional[float]:
        return self.impact_series[-1].value if self.impact_series else None

    @property
    def model_risk(self) -> Optional[float]:
        """
        Per-narrative risk = geometric mean of surprise and impact.
        High only when BOTH are high. Range [0, 1].
        sqrt(surprise * impact)
        """
        s = self.current_surprise
        i = self.current_impact
        if s is None or i is None:
            return None
        return (s * i) ** 0.5

    @property
    def is_active(self) -> bool:
        """
        A narrative is active if it received a story within the last 48 hours.
        Stale narratives remain in the DB but are filtered from active risk views.
        """
        if not self.last_updated:
            return False
        return (time.time() - self.last_updated) < (48 * 3600)

    @property
    def surprise_trend(self) -> Optional[str]:
        """
        Returns 'rising', 'falling', or 'stable' based on last 5 points.
        Returns None if fewer than 2 points exist.
        """
        if len(self.surprise_series) < 2:
            return None
        recent = [p.value for p in self.surprise_series[-5:]]
        delta = recent[-1] - recent[0]
        if delta > 0.05:
            return "rising"
        elif delta < -0.05:
            return "falling"
        return "stable"

    @property
    def impact_trend(self) -> Optional[str]:
        if len(self.impact_series) < 2:
            return None
        recent = [p.value for p in self.impact_series[-5:]]
        delta = recent[-1] - recent[0]
        if delta > 0.05:
            return "rising"
        elif delta < -0.05:
            return "falling"
        return "stable"

    def append_surprise(self, value: float, timestamp: float = None):
        point = TimeSeriesPoint(timestamp=timestamp or time.time(), value=value)
        self.surprise_series.append(point)
        if len(self.surprise_series) > MAX_SERIES_LENGTH:
            self.surprise_series = self.surprise_series[-MAX_SERIES_LENGTH:]
        self.last_updated = time.time()

    def append_impact(self, value: float, timestamp: float = None):
        point = TimeSeriesPoint(timestamp=timestamp or time.time(), value=value)
        self.impact_series.append(point)
        if len(self.impact_series) > MAX_SERIES_LENGTH:
            self.impact_series = self.impact_series[-MAX_SERIES_LENGTH:]
        self.last_updated = time.time()

    def add_headline(self, headline: str, max_recent: int = 10):
        self.recent_headlines.append(headline)
        self.recent_headlines = self.recent_headlines[-max_recent:]
        self.event_count += 1
        self.last_updated = time.time()
```

### 4.3 ChromaDB Collection Schema

Collection name: `"narratives"` (configurable via `CHROMA_COLLECTION` env var)
Distance metric: cosine (`hnsw:space = "cosine"`)

Each document stored in ChromaDB:

| Field | Type | Content |
|---|---|---|
| `id` | string | UUID (from NarrativeDirection.id) |
| `embedding` | float[384] | Narrative centroid vector |
| `document` | string | narrative.description (human-readable) |
| `metadata.name` | string | Short narrative label |
| `metadata.description` | string | One-sentence description |
| `metadata.created_at` | float | Unix epoch |
| `metadata.last_updated` | float | Unix epoch |
| `metadata.event_count` | int | Story count |
| `metadata.surprise_series` | string | JSON: `[{"timestamp": t, "value": v}, ...]` |
| `metadata.impact_series` | string | JSON: `[{"timestamp": t, "value": v}, ...]` |
| `metadata.recent_headlines` | string | JSON: `["headline1", "headline2", ...]` |

ChromaDB metadata values must be primitive types (str, int, float). The time series and headline list are serialized to JSON strings before storage and deserialized on read.

### 4.4 Full vector_store.py

**File:** `backend/db/vector_store.py`

```python
import json
import threading
import chromadb
from core.config import settings
from models.narrative import NarrativeDirection, TimeSeriesPoint

_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
collection = _client.get_or_create_collection(
    name=settings.chroma_collection,
    metadata={"hnsw:space": "cosine"},
)

# Write lock: prevents concurrent modifications from corrupting ChromaDB's SQLite
_write_lock = threading.Lock()


# --- Serialization ---

def _serialize(n: NarrativeDirection) -> dict:
    return {
        "name": n.name,
        "description": n.description,
        "created_at": n.created_at,
        "last_updated": n.last_updated,
        "event_count": n.event_count,
        "surprise_series": json.dumps([p.model_dump() for p in n.surprise_series]),
        "impact_series": json.dumps([p.model_dump() for p in n.impact_series]),
        "recent_headlines": json.dumps(n.recent_headlines),
    }

def _deserialize(id_: str, meta: dict) -> NarrativeDirection:
    return NarrativeDirection(
        id=id_,
        name=meta["name"],
        description=meta["description"],
        created_at=meta["created_at"],
        last_updated=meta["last_updated"],
        event_count=meta["event_count"],
        surprise_series=[TimeSeriesPoint(**p) for p in json.loads(meta["surprise_series"])],
        impact_series=[TimeSeriesPoint(**p) for p in json.loads(meta["impact_series"])],
        recent_headlines=json.loads(meta["recent_headlines"]),
    )


# --- Write operations (all guarded by _write_lock) ---

def add_narrative(narrative: NarrativeDirection, embedding: list[float]) -> str:
    with _write_lock:
        collection.add(
            ids=[narrative.id],
            embeddings=[embedding],
            metadatas=[_serialize(narrative)],
            documents=[narrative.description],
        )
    return narrative.id

def update_narrative(narrative: NarrativeDirection, new_embedding: list[float] = None):
    kwargs = dict(
        ids=[narrative.id],
        metadatas=[_serialize(narrative)],
        documents=[narrative.description],
    )
    if new_embedding is not None:
        kwargs["embeddings"] = [new_embedding]
    with _write_lock:
        collection.update(**kwargs)

def delete_narrative(narrative_id: str) -> bool:
    existing = collection.get(ids=[narrative_id])
    if not existing["ids"]:
        return False
    with _write_lock:
        collection.delete(ids=[narrative_id])
    return True


# --- Read operations ---

def get_narrative(narrative_id: str) -> NarrativeDirection | None:
    result = collection.get(ids=[narrative_id], include=["metadatas"])
    if not result["ids"]:
        return None
    return _deserialize(result["ids"][0], result["metadatas"][0])

def get_all_narratives() -> list[NarrativeDirection]:
    result = collection.get(include=["metadatas"])
    return [_deserialize(id_, meta) for id_, meta in zip(result["ids"], result["metadatas"])]

def query_nearest(
    embedding: list[float],
    n_results: int = 5,
) -> list[tuple[NarrativeDirection, float]]:
    """
    Returns list of (NarrativeDirection, cosine_distance) sorted by distance ascending.
    Cosine distance range: [0, 2]. Lower = more similar.
    Returns empty list if collection is empty.
    """
    count = collection.count()
    if count == 0:
        return []
    n_results = min(n_results, count)
    result = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["metadatas", "distances"],
    )
    return [
        (_deserialize(id_, meta), dist)
        for id_, meta, dist in zip(
            result["ids"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]

def narrative_count() -> int:
    return collection.count()

def get_embedding(narrative_id: str) -> list[float] | None:
    result = collection.get(ids=[narrative_id], include=["embeddings"])
    if not result["ids"]:
        return None
    return result["embeddings"][0]
```

---

## 5. Modal Model Integration

### 5.1 Architecture

LLM inference runs on Modal — a serverless GPU cloud platform. The system is split into two files:

| File | Role |
|---|---|
| `backend/modal_app.py` | Defines the Modal `App` and `LLM` class. Deployed once with `modal deploy`. |
| `backend/services/llm_client.py` | Thin client that looks up the deployed app and calls `.chat.remote()`. |

The public interface (`label_narrative`, `score_story`, `summarize_narrative_context`) is identical regardless of what runs inference — swapping backends only requires changing `llm_client.py`.

### 5.2 modal_app.py — Deployment Definition

**File:** `backend/modal_app.py`

```python
import modal

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
APP_NAME = "model-risk-llm"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.6", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME, image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")

@app.cls(gpu="A10G", timeout=600, container_idle_timeout=300, secrets=[hf_secret])
class LLM:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=MODEL_NAME, max_model_len=4096, dtype="bfloat16")
        self.SamplingParams = SamplingParams

    @modal.method()
    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.1) -> str:
        params = self.SamplingParams(max_tokens=max_tokens, temperature=temperature)
        outputs = self.llm.chat(messages, sampling_params=params)
        return outputs[0].outputs[0].text.strip()
```

**One-time setup:**
```bash
# 1. Install modal and authenticate
pip install modal
modal setup

# 2. Create HuggingFace secret (needed for gated Llama models)
modal secret create huggingface-secret HF_TOKEN=your_hf_token

# 3. Deploy (downloads model, builds container, deploys to Modal cloud)
modal deploy backend/modal_app.py
```

After deployment, the `LLM` class is reachable from anywhere via `modal.Cls.lookup("model-risk-llm", "LLM")`. Modal auto-scales and cold-starts as needed. The A10G GPU container stays warm for 300s after last use (`container_idle_timeout`).

### 5.3 llm_client.py — FastAPI-Side Client

**File:** `backend/services/llm_client.py`

Lazy singleton — the Modal app is only connected on first use:

```python
import modal
from core.config import settings

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = modal.Cls.lookup(settings.modal_app_name, "LLM")
    return _llm

def _chat(messages, max_tokens=256, temperature=0.1) -> str:
    for attempt in range(3):
        try:
            return _get_llm()().chat.remote(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
```

`_get_llm()().chat.remote(...)` — the double `()` is intentional:
- `_get_llm()` returns the `LLM` Modal class handle
- `()` instantiates it (routes to an available container)
- `.chat.remote(...)` calls the method on that container

### 5.4 label_narrative() — New Narrative Labeling

Called **only** when a story's best cosine distance exceeds the threshold (a new narrative direction is being created).

```python
_LABEL_PROMPT = """...identify the narrative direction...
Respond with ONLY valid JSON:
{{"name": "<3-6 word label>", "description": "<one sentence>"}}
News story: {story_text}"""

def label_narrative(story_text: str) -> dict:
    # max_tokens=256, temperature=0.1
    # fallback: {"name": "unclassified narrative", "description": story_text[:150]}
```

### 5.5 score_story() — Surprise & Impact Scoring

Called for every story (create or update path). Returns scores in [0, 1].

- **Surprise**: how unexpected this development is within the narrative arc (0=priced in, 1=sudden shock)
- **Impact**: economic significance — sector size, event type severity, geographic scope (0=negligible, 1=systemic)

When updating an existing narrative, the current surprise/impact values are included in the prompt so the model can calibrate whether this story escalates, continues, or de-escalates.

```python
def score_story(story_text, narrative_description, existing_surprise, existing_impact) -> dict:
    # max_tokens=64, temperature=0.1
    # fallback: {"surprise": 0.3, "impact": 0.3}
    # output clamped to [0.0, 1.0]
```

### 5.6 summarize_narrative_context() — RAG Chat

Called by `POST /api/chat`. Top-k nearest narratives retrieved from ChromaDB are formatted and passed as context to the LLM, which answers the user's question in prose.

```python
def summarize_narrative_context(narratives: list[dict], query: str) -> str:
    # max_tokens=512, temperature=0.3
```

### 5.7 Model & Runtime Configuration

| Setting | Value | Notes |
|---|---|---|
| Model | `meta-llama/Llama-3.3-70B-Instruct` | Configured in `modal_app.py` |
| GPU | `A10G` | Fits 70B bfloat16 with max_model_len=4096 |
| Temperature (scoring) | `0.1` | Near-deterministic JSON |
| Temperature (chat) | `0.3` | Slightly more fluent prose |
| Max tokens (score) | `64` | Just a JSON pair |
| Max tokens (label) | `256` | Name + description |
| Max tokens (chat) | `512` | Prose answer |
| Retry | 3 attempts, 1s/2s/4s backoff | In `_chat()` |
| JSON fallback | Extract `{...}` block → hardcoded defaults | In `_parse_json()` |


## 6. Narrative Routing Engine

### 6.1 The Core Decision

**File:** `backend/services/narrative_engine.py`

Every incoming story passes through a single function: `ingest_story(headline, body)`.

The routing decision:

```
story_embedding = embed_text(headline + "\n\n" + body)
nearest = query_nearest(story_embedding, n_results=5)

if nearest is empty OR nearest[0].distance >= NEW_NARRATIVE_THRESHOLD:
    → _create_narrative()   # story is too different from all existing narratives
else:
    → _update_narrative()   # story fits into best-matching existing narrative
```

The threshold `NEW_NARRATIVE_THRESHOLD = 0.40` is a cosine distance in [0, 2].
- Distance < 0.20: nearly identical topic (same event continuing)
- Distance 0.20–0.40: same general narrative direction (update → correct)
- Distance 0.40–0.70: loosely related but semantically distinct (create → correct)
- Distance > 0.70: completely unrelated topic

This threshold is configurable via `NEW_NARRATIVE_THRESHOLD` in `.env`.

### 6.2 _create_narrative()

```python
def _create_narrative(
    story_embedding: list[float],
    headline: str,
    full_text: str,
) -> NarrativeDirection:
    # 1. Ask Cerebras to label the new narrative direction
    label = label_narrative(full_text)

    # 2. Score the story for Surprise and Impact
    scores = score_story(
        story_text=full_text,
        narrative_description=label["description"],
        existing_surprise=None,
        existing_impact=None,
    )

    # 3. Build NarrativeDirection object
    now = time.time()
    narrative = NarrativeDirection(
        name=label["name"],
        description=label["description"],
        created_at=now,
        last_updated=now,
    )
    narrative.append_surprise(scores["surprise"], timestamp=now)
    narrative.append_impact(scores["impact"], timestamp=now)
    narrative.add_headline(headline)

    # 4. Store in ChromaDB using the raw story embedding as the initial centroid
    vector_store.add_narrative(narrative, embedding=story_embedding)

    return narrative
```

### 6.3 _update_narrative()

```python
def _update_narrative(
    narrative: NarrativeDirection,
    story_embedding: list[float],
    headline: str,
    full_text: str,
) -> NarrativeDirection:
    # 1. Score the story in context of the existing narrative
    scores = score_story(
        story_text=full_text,
        narrative_description=narrative.description,
        existing_surprise=narrative.current_surprise,
        existing_impact=narrative.current_impact,
    )

    # 2. Update time series and headline cache
    now = time.time()
    narrative.append_surprise(scores["surprise"], timestamp=now)
    narrative.append_impact(scores["impact"], timestamp=now)
    narrative.add_headline(headline)

    # 3. Blend story embedding into narrative centroid
    current_embedding = vector_store.get_embedding(narrative.id)
    updated_embedding = _blend_embedding(
        current=current_embedding,
        new=story_embedding,
        n=narrative.event_count,
    )

    # 4. Persist to ChromaDB
    vector_store.update_narrative(narrative, new_embedding=updated_embedding)

    return narrative
```

### 6.4 _blend_embedding() — Online Centroid Update

```python
def _blend_embedding(
    current: list[float],
    new: list[float],
    n: int,
) -> list[float]:
    """
    Incrementally update narrative centroid with a new story vector.
    Uses online mean formula: new_mean = old_mean * (n/(n+1)) + x * (1/(n+1))
    This gives equal weight to all historical stories.
    n is the event_count BEFORE adding the new story.
    """
    if n <= 0:
        return new
    w_old = n / (n + 1)
    w_new = 1.0 / (n + 1)
    blended = [c * w_old + v * w_new for c, v in zip(current, new)]
    # Re-normalize to unit length (cosine space requires unit vectors)
    magnitude = sum(x ** 2 for x in blended) ** 0.5
    if magnitude == 0:
        return blended
    return [x / magnitude for x in blended]
```

**Critical note:** `n` must be `narrative.event_count` BEFORE calling `narrative.add_headline()`. The order in `_update_narrative()` is: score → append_surprise → append_impact → get_embedding → blend (using pre-increment n) → add_headline → update_narrative. This ordering matters.

### 6.5 Thread Safety

The `ingest_story()` function is called from multiple async worker threads simultaneously. The `_write_lock` in `vector_store.py` prevents concurrent writes to ChromaDB. However, there is a TOCTOU (time-of-check to time-of-use) race condition in the routing logic:

```
Worker A: query_nearest → best_distance=0.50 → decide to CREATE
Worker B: query_nearest → best_distance=0.50 → decide to CREATE
Worker A: _create_narrative("energy supply shock")   ← adds to DB
Worker B: _create_narrative("energy supply shock")   ← creates duplicate!
```

Mitigation: add a narrative-level routing lock:

```python
_route_lock = threading.Lock()

def ingest_story(headline: str, body: str) -> dict:
    full_text = f"{headline}\n\n{body}"
    story_embedding = embed_text(full_text)

    with _route_lock:
        nearest = vector_store.query_nearest(story_embedding, n_results=5)
        if nearest and nearest[0][1] < NEW_NARRATIVE_THRESHOLD:
            best_narrative, best_distance = nearest[0]
            action = "updated"
            # Release lock before slow Cerebras call
    # ... Cerebras calls happen outside lock ...
```

For the hackathon scope with `pipeline_num_workers=2`, the TOCTOU window is small and acceptable. The lock approach above is the correct production fix.

### 6.6 ingest_story() — Full Function

```python
import time
import threading
from core.config import settings
from db import vector_store
from models.narrative import NarrativeDirection
from services.embedder import embed_text
from services.cerebras_client import score_story, label_narrative

NEW_NARRATIVE_THRESHOLD: float = settings.new_narrative_threshold
_route_lock = threading.Lock()


def ingest_story(headline: str, body: str) -> dict:
    full_text = f"{headline}\n\n{body}"
    story_embedding = embed_text(full_text)

    with _route_lock:
        nearest = vector_store.query_nearest(story_embedding, n_results=5)
        if nearest:
            best_narrative, best_distance = nearest[0]
        else:
            best_narrative, best_distance = None, float("inf")

        route_to_existing = (best_narrative is not None and
                             best_distance < NEW_NARRATIVE_THRESHOLD)

    if route_to_existing:
        action = "updated"
        narrative = _update_narrative(best_narrative, story_embedding, headline, full_text)
    else:
        action = "created"
        narrative = _create_narrative(story_embedding, headline, full_text)

    return {
        "action": action,
        "narrative_id": narrative.id,
        "narrative_name": narrative.name,
        "best_distance": round(best_distance, 4) if best_narrative else None,
        "threshold": NEW_NARRATIVE_THRESHOLD,
        "current_surprise": narrative.current_surprise,
        "current_impact": narrative.current_impact,
        "model_risk": narrative.model_risk,
        "narrative_event_count": narrative.event_count,
    }
```

---

## 7. Risk Index Computation

### 7.1 Per-Narrative Risk

```
model_risk(narrative) = sqrt(current_surprise * current_impact)
```

This is the geometric mean. Properties:
- If either dimension is 0, risk = 0 (a surprising but unimportant event is not dangerous)
- If both are 1.0, risk = 1.0 (maximum danger)
- Symmetric: surprise and impact contribute equally
- Computed in the `NarrativeDirection.model_risk` property on read — not stored

### 7.2 Global Model Risk Index

Three aggregation methods, selectable via query parameter:

```
method=max      → max(model_risk for all active narratives)
method=mean     → mean(model_risk for all active narratives)
method=weighted → weighted mean using impact as weights
```

**Default: `max`**

Rationale: Any single high-risk narrative should immediately elevate the overall index. If a bank collapses (narrative risk = 0.95), the fact that 15 routine narratives score 0.20 should not dilute the signal. This mirrors how financial risk actually works — one tail event dominates.

### 7.3 Risk History Endpoint

`GET /api/risk/history?window=24&resolution=100`

Reconstructs a historical risk index time series by scanning all narrative time series points that fall within the window, binning them into `resolution` buckets, and computing `max(risk)` per bucket.

The result is a JSON array of `{timestamp, model_risk_index}` objects suitable for charting with Chart.js.

### 7.4 Narrative Activity (is_active)

A narrative is "active" if `time.time() - last_updated < 48 * 3600` (48 hours). Stale narratives remain in ChromaDB permanently (they form the historical memory of the system) but are excluded from active risk views and the default risk index computation when `active_only=True`.

---

## 8. FastAPI Backend — All Routes

### 8.1 main.py (final)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routes import ingest, narratives, risk, chat, events, pipeline
from services.pipeline import start_pipeline, stop_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_pipeline()
    yield
    await stop_pipeline()

app = FastAPI(title="Real-World Model Risk Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router,     prefix="/api/ingest",     tags=["ingest"])
app.include_router(narratives.router, prefix="/api/narratives", tags=["narratives"])
app.include_router(risk.router,       prefix="/api/risk",       tags=["risk"])
app.include_router(chat.router,       prefix="/api/chat",       tags=["chat"])
app.include_router(events.router,     prefix="/api/events",     tags=["events"])
app.include_router(pipeline.router,   prefix="/api/pipeline",   tags=["pipeline"])

# Serve frontend from /
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

@app.get("/health")
def health():
    return {"status": "ok"}
```

### 8.2 Route Table

| Method | Path | Description | Request Body | Response |
|---|---|---|---|---|
| GET | /health | Health check | — | `{status}` |
| POST | /api/ingest | Ingest single story | `{headline, body, source?}` | `{action, narrative_id, narrative_name, best_distance, threshold, current_surprise, current_impact, model_risk, narrative_event_count}` |
| POST | /api/ingest/batch | Ingest multiple stories | `{stories: [...], max_stories?}` | `{processed, results, errors, duration_seconds}` |
| GET | /api/narratives | List all narrative directions | `?active_only&sort_by&limit` | `{narratives: [...], total}` |
| GET | /api/narratives/{id} | Get narrative detail | — | Full NarrativeDirection JSON |
| GET | /api/narratives/{id}/history | Get full time series | — | `{surprise_series, impact_series, model_risk_series, ...}` |
| POST | /api/narratives/search | Semantic search | `{query, n_results?, active_only?}` | `{query, results: [{narrative, distance, similarity}], total}` |
| DELETE | /api/narratives/{id} | Delete a narrative | — | `{deleted, id}` |
| GET | /api/risk | Global risk index | `?method&active_only` | `{model_risk_index, narrative_count, active_narrative_count, breakdown, aggregation_method, computed_at}` |
| GET | /api/risk/history | Risk index time series | `?window&resolution` | `{history: [{timestamp, model_risk_index}], ...}` |
| POST | /api/chat | RAG Q&A | `{query, n_context_narratives?}` | `{query, answer, context_narratives}` |
| GET | /api/events/stream | SSE live event stream | — | SSE stream |
| GET | /api/pipeline/stats | Pipeline status | — | `{pipeline, narratives, events, timestamp}` |

### 8.3 SSE Events Route

**File:** `backend/api/routes/events.py`

```python
import asyncio
import json
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

router = APIRouter()

_subscribers: list[asyncio.Queue] = []


def broadcast_event(data: dict):
    """Called from pipeline.py after each successful ingest."""
    payload = f"data: {json.dumps(data)}\n\n"
    dead = []
    for q in _subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


async def _event_generator(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    yield f"data: {json.dumps({'type': 'connected', 'timestamp': time.time()})}\n\n"
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield msg
            except asyncio.TimeoutError:
                # Send keepalive comment every 30s to prevent proxy disconnection
                yield ": keepalive\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        try:
            _subscribers.remove(queue)
        except ValueError:
            pass


@router.get("/stream")
async def stream_events():
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.append(queue)
    return StreamingResponse(
        _event_generator(queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

Each SSE event has structure:
```json
{
  "type": "ingest",
  "timestamp": 1709123456.789,
  "result": {
    "action": "created|updated",
    "narrative_name": "Energy supply shock",
    "narrative_id": "uuid",
    "model_risk": 0.62,
    "current_surprise": 0.71,
    "current_impact": 0.54
  }
}
```

### 8.4 Pipeline Stats Route

**File:** `backend/api/routes/pipeline.py`

```python
from fastapi import APIRouter
from services.pipeline import pipeline_stats, raw_queue
from db import vector_store
import time

router = APIRouter()

@router.get("/stats")
def get_pipeline_stats():
    all_narratives = vector_store.get_all_narratives()
    active_count = sum(1 for n in all_narratives if n.is_active)
    total_events = sum(n.event_count for n in all_narratives)

    return {
        "pipeline": {
            **pipeline_stats,
            "queue_size": raw_queue.qsize(),
        },
        "narratives": {
            "total": len(all_narratives),
            "active": active_count,
        },
        "events": {
            "total_events_ingested": total_events,
        },
        "timestamp": time.time(),
    }
```

---

## 9. Frontend UI

### 9.1 File Structure

```
frontend/
├── index.html      (single page; all content rendered here)
├── styles.css      (dark theme; CSS variables; responsive grid)
└── dashboard.js    (all logic: state, fetching, rendering, SSE, charts)
```

### 9.2 Dashboard Layout (ASCII Mockup)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  REAL-WORLD MODEL RISK ENGINE                    ● LIVE  [●] [AUTO 10s] │
├──────────────────┬──────────────────────────────┬────────────────────────┤
│                  │                              │                        │
│   RISK GAUGE     │     24H MODEL RISK INDEX     │   PIPELINE STATUS      │
│                  │                              │                        │
│    ┌──────┐      │  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆  │  Stories ingested: 482 │
│    │  0.73│      │                              │  Narratives total: 23  │
│    │ HIGH │      │  [Chart.js line chart]       │  Narratives active: 11 │
│    └──────┘      │                              │  Errors: 0             │
│                  │                              │  Queue: 3              │
│  LOW  MED  HIGH  │                              │  Started: 2h ago       │
├──────────────────┴──────────────────────────────┴────────────────────────┤
│  ACTIVE NARRATIVE DIRECTIONS                          [Sort: Risk ▼]     │
│ ┌──────────────────────────────────────────────────────────────────────┐ │
│ │ NAME                      SURPRISE   IMPACT    RISK    EVENTS TREND  │ │
│ │ Energy supply shock          0.81      0.74    0.77      41   ↑      │ │
│ │ Regional banking stress      0.65      0.71    0.68      29   →      │ │
│ │ China trade escalation       0.52      0.68    0.59      18   ↑      │ │
│ │ US debt ceiling pressure     0.44      0.63    0.53      12   ↓      │ │
│ │ ...                                                                  │ │
│ └──────────────────────────────────────────────────────────────────────┘ │
├────────────────────────────────┬─────────────────────────────────────────┤
│  SEMANTIC SEARCH               │  LIVE FEED                              │
│  ┌──────────────────────────┐  │  ┌───────────────────────────────────┐  │
│  │ sanctions energy exports │  │  │ ● [created] Energy supply shock   │  │
│  └──────────────────────────┘  │  │   risk=0.77  1m ago               │  │
│  [Search]                      │  │ ● [updated] Banking stress         │  │
│                                │  │   risk=0.68  3m ago               │  │
│  Results:                      │  │ ● [created] Sovereign debt risk    │  │
│  1. Energy supply shock (0.12) │  │   risk=0.52  7m ago               │  │
│     desc...                    │  │ ...                               │  │
│  2. Russia-Ukraine commodity   │  └───────────────────────────────────┘  │
│     shock (0.28) desc...       │                                         │
├────────────────────────────────┴─────────────────────────────────────────┤
│  MANUAL INGEST                                                           │
│  Headline: [_________________________________]                           │
│  Body:     [_________________________________]                           │
│            [_________________________________]                           │
│  [Ingest Story]                  Last result: created "Energy supply..." │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.3 index.html Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Real-World Model Risk Engine</title>
  <link rel="stylesheet" href="styles.css" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <header>
    <h1>Real-World Model Risk Engine</h1>
    <div id="live-indicator">● LIVE</div>
    <div id="last-updated"></div>
  </header>

  <main class="grid">

    <!-- Row 1: Risk gauge | 24h chart | Pipeline stats -->
    <section id="gauge-panel" class="card">
      <h2>Model Risk Index</h2>
      <div id="risk-value">—</div>
      <div id="risk-label">—</div>
      <div class="risk-bar">
        <div id="risk-fill"></div>
      </div>
      <div class="risk-bar-labels">
        <span>LOW</span><span>MED</span><span>HIGH</span>
      </div>
    </section>

    <section id="chart-panel" class="card">
      <h2>24h Risk History</h2>
      <canvas id="risk-chart"></canvas>
    </section>

    <section id="pipeline-panel" class="card">
      <h2>Pipeline Status</h2>
      <dl id="pipeline-stats"></dl>
    </section>

    <!-- Row 2: Narrative table (full width) -->
    <section id="narratives-panel" class="card wide">
      <h2>Active Narrative Directions
        <select id="sort-select">
          <option value="risk">Sort: Risk</option>
          <option value="events">Sort: Events</option>
          <option value="updated">Sort: Recent</option>
        </select>
      </h2>
      <table id="narratives-table">
        <thead>
          <tr>
            <th>Name</th><th>Description</th>
            <th>Surprise</th><th>Impact</th><th>Risk</th>
            <th>Events</th><th>Trend</th><th>Updated</th>
          </tr>
        </thead>
        <tbody id="narratives-tbody"></tbody>
      </table>
    </section>

    <!-- Row 3: Search | Live feed -->
    <section id="search-panel" class="card">
      <h2>Semantic Search</h2>
      <input id="search-input" type="text" placeholder="e.g. sanctions affecting energy exports" />
      <button id="search-btn">Search</button>
      <div id="search-results"></div>
    </section>

    <section id="feed-panel" class="card">
      <h2>Live Feed</h2>
      <ul id="live-feed"></ul>
    </section>

    <!-- Row 4: Manual ingest | Narrative detail modal -->
    <section id="ingest-panel" class="card wide">
      <h2>Manual Ingest</h2>
      <input id="ingest-headline" type="text" placeholder="Headline" />
      <textarea id="ingest-body" placeholder="Story body (optional)"></textarea>
      <button id="ingest-btn">Ingest Story</button>
      <div id="ingest-result"></div>
    </section>

  </main>

  <!-- Narrative detail modal (hidden by default) -->
  <div id="narrative-modal" class="modal hidden">
    <div class="modal-content">
      <button id="modal-close">×</button>
      <h2 id="modal-name"></h2>
      <p id="modal-description"></p>
      <canvas id="modal-chart"></canvas>
      <h3>Recent Headlines</h3>
      <ul id="modal-headlines"></ul>
    </div>
  </div>

  <script src="dashboard.js"></script>
</body>
</html>
```

### 9.4 dashboard.js — Full Logic

**File:** `frontend/dashboard.js`

The JS file is organized into these sections:

#### Constants & State

```javascript
const API = "http://localhost:8000/api";
let riskChart = null;
let modalChart = null;
let autoRefreshInterval = null;
const AUTO_REFRESH_MS = 10_000;
const FEED_MAX_ITEMS = 50;
```

#### API Functions

```javascript
async function fetchJSON(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}
```

#### Risk Gauge Rendering

```javascript
function renderGauge(indexValue) {
  const el = document.getElementById("risk-value");
  const labelEl = document.getElementById("risk-label");
  const fill = document.getElementById("risk-fill");

  if (indexValue === null || indexValue === undefined) {
    el.textContent = "—";
    labelEl.textContent = "No data";
    fill.style.width = "0%";
    fill.className = "risk-fill";
    return;
  }

  el.textContent = indexValue.toFixed(2);
  fill.style.width = `${(indexValue * 100).toFixed(1)}%`;

  if (indexValue < 0.33) {
    labelEl.textContent = "LOW — models reliable";
    fill.className = "risk-fill low";
  } else if (indexValue < 0.66) {
    labelEl.textContent = "MEDIUM — narratives building";
    fill.className = "risk-fill medium";
  } else {
    labelEl.textContent = "HIGH — regime shift / model fragility";
    fill.className = "risk-fill high";
  }
}
```

#### 24h Risk Chart

```javascript
async function initRiskChart() {
  const data = await fetchJSON("/risk/history?window=24&resolution=100");
  const labels = data.history.map(p =>
    new Date(p.timestamp * 1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"})
  );
  const values = data.history.map(p => p.model_risk_index);

  const ctx = document.getElementById("risk-chart").getContext("2d");
  riskChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Model Risk Index",
        data: values,
        borderColor: "#e74c3c",
        backgroundColor: "rgba(231,76,60,0.1)",
        tension: 0.3,
        pointRadius: 0,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 1, ticks: { color: "#aaa" }, grid: { color: "#333" } },
        x: { ticks: { color: "#aaa", maxTicksLimit: 8 }, grid: { color: "#333" } },
      },
      plugins: { legend: { display: false } },
    }
  });
}

async function updateRiskChart() {
  if (!riskChart) return;
  const data = await fetchJSON("/risk/history?window=24&resolution=100");
  riskChart.data.labels = data.history.map(p =>
    new Date(p.timestamp * 1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"})
  );
  riskChart.data.datasets[0].data = data.history.map(p => p.model_risk_index);
  riskChart.update("none");
}
```

#### Narrative Table Rendering

```javascript
function riskColor(value) {
  if (value === null || value === undefined) return "#888";
  if (value < 0.33) return "#2ecc71";
  if (value < 0.66) return "#f39c12";
  return "#e74c3c";
}

function trendArrow(trend) {
  if (trend === "rising") return "↑";
  if (trend === "falling") return "↓";
  if (trend === "stable") return "→";
  return "—";
}

function timeAgo(unixTs) {
  const diff = Date.now() / 1000 - unixTs;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff/60)}m ago`;
  if (diff < 86400) return `${Math.round(diff/3600)}h ago`;
  return `${Math.round(diff/86400)}d ago`;
}

async function refreshNarratives() {
  const sortBy = document.getElementById("sort-select").value;
  const data = await fetchJSON(`/narratives?active_only=true&sort_by=${sortBy}&limit=50`);
  const tbody = document.getElementById("narratives-tbody");
  tbody.innerHTML = "";
  for (const n of data.narratives) {
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    tr.innerHTML = `
      <td><strong>${n.name}</strong></td>
      <td class="desc">${n.description}</td>
      <td style="color:${riskColor(n.current_surprise)}">${(n.current_surprise??0).toFixed(2)}</td>
      <td style="color:${riskColor(n.current_impact)}">${(n.current_impact??0).toFixed(2)}</td>
      <td style="color:${riskColor(n.model_risk)};font-weight:bold">${(n.model_risk??0).toFixed(2)}</td>
      <td>${n.event_count}</td>
      <td>${trendArrow(n.surprise_trend)}</td>
      <td>${timeAgo(n.last_updated)}</td>
    `;
    tr.addEventListener("click", () => openNarrativeModal(n.id));
    tbody.appendChild(tr);
  }
}
```

#### Narrative Detail Modal

```javascript
async function openNarrativeModal(narrativeId) {
  const modal = document.getElementById("narrative-modal");
  const data = await fetchJSON(`/narratives/${narrativeId}/history`);

  document.getElementById("modal-name").textContent = data.name;
  document.getElementById("modal-description").textContent = data.description;

  // Render headlines
  const ul = document.getElementById("modal-headlines");
  ul.innerHTML = "";
  for (const h of (data.recent_headlines || [])) {
    const li = document.createElement("li");
    li.textContent = h;
    ul.appendChild(li);
  }

  // Render time series chart
  if (modalChart) { modalChart.destroy(); modalChart = null; }
  const ctx = document.getElementById("modal-chart").getContext("2d");
  const labels = data.surprise_series.map(p =>
    new Date(p.timestamp * 1000).toLocaleString()
  );
  modalChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Surprise",
          data: data.surprise_series.map(p => p.value),
          borderColor: "#e74c3c",
          tension: 0.3,
          pointRadius: 2,
        },
        {
          label: "Impact",
          data: data.impact_series.map(p => p.value),
          borderColor: "#3498db",
          tension: 0.3,
          pointRadius: 2,
        },
        {
          label: "Model Risk",
          data: data.model_risk_series.map(p => p.value),
          borderColor: "#f39c12",
          tension: 0.3,
          pointRadius: 2,
          borderDash: [5, 5],
        },
      ]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 1 },
      },
    }
  });

  modal.classList.remove("hidden");
}

document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("narrative-modal").classList.add("hidden");
  if (modalChart) { modalChart.destroy(); modalChart = null; }
});
```

#### Semantic Search

```javascript
document.getElementById("search-btn").addEventListener("click", async () => {
  const query = document.getElementById("search-input").value.trim();
  if (!query) return;
  const resultsDiv = document.getElementById("search-results");
  resultsDiv.textContent = "Searching...";
  try {
    const data = await postJSON("/narratives/search", { query, n_results: 5 });
    resultsDiv.innerHTML = data.results.map((r, i) => `
      <div class="search-result" onclick="openNarrativeModal('${r.narrative.id}')">
        <strong>${i+1}. ${r.narrative.name}</strong>
        <span class="distance">similarity: ${(r.similarity*100).toFixed(0)}%</span>
        <span style="color:${riskColor(r.narrative.model_risk)}">
          risk: ${(r.narrative.model_risk??0).toFixed(2)}
        </span>
        <p>${r.narrative.description}</p>
      </div>
    `).join("");
  } catch (e) {
    resultsDiv.textContent = `Error: ${e.message}`;
  }
});
```

#### Live Feed (SSE)

```javascript
function initSSE() {
  const feedEl = document.getElementById("live-feed");
  const es = new EventSource(`${API}/events/stream`);

  es.addEventListener("message", (e) => {
    const data = JSON.parse(e.data);
    if (data.type !== "ingest") return;

    const r = data.result;
    const li = document.createElement("li");
    li.className = `feed-item ${r.action}`;
    li.innerHTML = `
      <span class="feed-badge ${r.action}">${r.action}</span>
      <span class="feed-name">${r.narrative_name}</span>
      <span class="feed-risk" style="color:${riskColor(r.model_risk)}">
        risk=${(r.model_risk??0).toFixed(2)}
      </span>
      <span class="feed-time">just now</span>
    `;

    feedEl.insertBefore(li, feedEl.firstChild);
    while (feedEl.children.length > FEED_MAX_ITEMS) {
      feedEl.removeChild(feedEl.lastChild);
    }

    // Trigger a dashboard refresh after each new event
    refreshDashboard();
  });

  es.addEventListener("error", () => {
    // SSE will auto-reconnect; no manual action needed
  });
}
```

#### Manual Ingest Panel

```javascript
document.getElementById("ingest-btn").addEventListener("click", async () => {
  const headline = document.getElementById("ingest-headline").value.trim();
  const body = document.getElementById("ingest-body").value.trim();
  if (!headline) return;

  const resultEl = document.getElementById("ingest-result");
  resultEl.textContent = "Ingesting...";

  try {
    const data = await postJSON("/ingest", { headline, body });
    resultEl.innerHTML = `
      <span class="badge ${data.action}">${data.action}</span>
      <strong>${data.narrative_name}</strong>
      — risk: ${(data.model_risk??0).toFixed(2)},
      surprise: ${(data.current_surprise??0).toFixed(2)},
      impact: ${(data.current_impact??0).toFixed(2)}
      (distance: ${data.best_distance ?? "N/A"})
    `;
    document.getElementById("ingest-headline").value = "";
    document.getElementById("ingest-body").value = "";
  } catch (e) {
    resultEl.textContent = `Error: ${e.message}`;
  }
});
```

#### Auto-Refresh & Dashboard Init

```javascript
async function refreshDashboard() {
  try {
    const [riskData, pipelineData] = await Promise.all([
      fetchJSON("/risk"),
      fetchJSON("/pipeline/stats"),
    ]);

    renderGauge(riskData.model_risk_index);
    renderPipelineStats(pipelineData);
    await refreshNarratives();
    await updateRiskChart();

    document.getElementById("last-updated").textContent =
      `Updated: ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    console.error("Dashboard refresh failed:", e);
  }
}

function renderPipelineStats(data) {
  const dl = document.getElementById("pipeline-stats");
  const p = data.pipeline;
  dl.innerHTML = `
    <dt>Stories ingested</dt><dd>${p.stories_ingested ?? 0}</dd>
    <dt>Narratives created</dt><dd>${p.narratives_created ?? 0}</dd>
    <dt>Narratives updated</dt><dd>${p.narratives_updated ?? 0}</dd>
    <dt>Errors</dt><dd>${p.errors ?? 0}</dd>
    <dt>Queue size</dt><dd>${p.queue_size ?? 0}</dd>
    <dt>Active narratives</dt><dd>${data.narratives?.active ?? 0}</dd>
  `;
}

async function init() {
  await initRiskChart();
  await refreshDashboard();
  initSSE();
  autoRefreshInterval = setInterval(refreshDashboard, AUTO_REFRESH_MS);
}

document.addEventListener("DOMContentLoaded", init);
document.getElementById("sort-select").addEventListener("change", refreshNarratives);
```

### 9.5 styles.css

Dark theme with CSS variables. Key rules:

```css
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #c9d1d9;
  --text-muted: #8b949e;
  --accent-red: #e74c3c;
  --accent-blue: #3498db;
  --accent-yellow: #f39c12;
  --accent-green: #2ecc71;
  --radius: 8px;
}

body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; }
header { display: flex; align-items: center; gap: 1rem; padding: 1rem 2rem; border-bottom: 1px solid var(--border); }
h1 { font-size: 1.1rem; font-weight: 600; margin: 0; }

.grid {
  display: grid;
  grid-template-columns: 220px 1fr 220px;
  grid-template-rows: auto;
  gap: 1rem;
  padding: 1rem 2rem;
}
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; }
.wide { grid-column: 1 / -1; }

/* Risk gauge */
#risk-value { font-size: 3rem; font-weight: 700; text-align: center; }
#risk-label { text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.5rem; }
.risk-bar { height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
.risk-fill { height: 100%; transition: width 0.5s ease, background-color 0.5s ease; border-radius: 4px; }
.risk-fill.low { background: var(--accent-green); }
.risk-fill.medium { background: var(--accent-yellow); }
.risk-fill.high { background: var(--accent-red); }

/* Narrative table */
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { text-align: left; color: var(--text-muted); padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }
td { padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border); }
tr:hover { background: rgba(255,255,255,0.03); }
td.desc { color: var(--text-muted); font-size: 0.8rem; max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Live feed */
#live-feed { list-style: none; padding: 0; margin: 0; overflow-y: auto; max-height: 300px; }
.feed-item { display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; }
.feed-badge { padding: 2px 6px; border-radius: 3px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }
.feed-badge.created { background: rgba(46,204,113,0.2); color: var(--accent-green); }
.feed-badge.updated { background: rgba(52,152,219,0.2); color: var(--accent-blue); }
.feed-name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.feed-time { color: var(--text-muted); font-size: 0.75rem; }

/* Semantic search */
#search-input { width: 100%; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); margin-bottom: 0.5rem; }
.search-result { padding: 0.6rem; border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 0.5rem; cursor: pointer; }
.search-result:hover { background: rgba(255,255,255,0.03); }
.distance { color: var(--text-muted); font-size: 0.8rem; margin-left: 0.5rem; }

/* Manual ingest */
#ingest-headline { width: 100%; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); margin-bottom: 0.5rem; }
#ingest-body { width: 100%; height: 60px; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); resize: vertical; }
button { background: var(--accent-blue); color: white; border: none; padding: 0.5rem 1rem; border-radius: var(--radius); cursor: pointer; margin-top: 0.5rem; }
button:hover { opacity: 0.85; }

/* Modal */
.modal { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal.hidden { display: none; }
.modal-content { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 2rem; max-width: 800px; width: 90%; max-height: 80vh; overflow-y: auto; position: relative; }
#modal-close { position: absolute; top: 1rem; right: 1rem; background: none; border: none; color: var(--text); font-size: 1.5rem; cursor: pointer; padding: 0; }
```

---

## 10. Configuration & Environment

### 10.1 Full config.py

**File:** `backend/core/config.py`

```python
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Cerebras
    cerebras_api_key: str = ""
    cerebras_model: str = "llama-3.3-70b"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "narratives"

    # Narrative routing
    new_narrative_threshold: float = 0.40   # cosine distance in [0, 2]

    # Pipeline
    pipeline_num_workers: int = 2
    poll_interval_seconds: int = 60

    # RSS sources
    rss_sources: list[str] = [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ]

    # Reddit (optional)
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_subreddits: list[str] = ["stocks", "investing", "worldnews"]

    class Config:
        env_file = ".env"


settings = Settings()
```

### 10.2 Full .env.example

```
# Cerebras API
CEREBRAS_API_KEY=your_key_here
CEREBRAS_MODEL=llama-3.3-70b

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=narratives

# Narrative routing (cosine distance threshold, [0-2])
NEW_NARRATIVE_THRESHOLD=0.40

# Pipeline
PIPELINE_NUM_WORKERS=2
POLL_INTERVAL_SECONDS=60

# Reddit (optional — leave empty to disable)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
```

### 10.3 requirements.txt

```
fastapi
uvicorn[standard]
pydantic-settings
cerebras-cloud-sdk
chromadb
sentence-transformers
aiohttp
feedparser
praw
```

### 10.4 Running the Server

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend is served by FastAPI's StaticFiles mount at `/`. Navigate to `http://localhost:8000`.

---

## 11. End-to-End Data Flow Walkthrough

### Scenario A: New narrative created

1. RSS poller fetches CNBC feed. New entry: `"Russia cuts gas supply to Europe further"`
2. `DeduplicatingCache.is_seen()` → False. Story is new.
3. `RawStory` added to `raw_queue`
4. `ingest_worker` pulls story from queue
5. `ingest_story()` called:
   - `embed_text("Russia cuts gas supply to Europe further\n\n...")` → 384-dim vector
   - `query_nearest(embedding, n=5)` → DB is empty → returns `[]`
   - No nearest neighbors → `best_distance = inf` → exceeds threshold
   - `_create_narrative()` called:
     - `label_narrative(text)` → Cerebras: `{"name": "European energy supply shock", "description": "Ongoing deterioration of Russian gas exports to Europe disrupting energy markets"}`
     - `score_story(text, description, None, None)` → Cerebras: `{"surprise": 0.82, "impact": 0.78}`
     - `NarrativeDirection` object created with `id=uuid`, time series seeded
     - `add_narrative(narrative, embedding)` → ChromaDB stores document
6. Return: `{action: "created", narrative_name: "European energy supply shock", model_risk: 0.80, ...}`
7. `broadcast_event()` fires → SSE clients receive update → live feed shows new card

### Scenario B: Existing narrative updated

1. 30 minutes later: `"Germany activates emergency gas rationing plan"`
2. `embed_text(...)` → new 384-dim vector
3. `query_nearest(embedding, n=5)`:
   - DB now has "European energy supply shock" narrative
   - Cosine distance = 0.18 (very similar semantic direction)
4. `0.18 < 0.40` (threshold) → routes to existing narrative
5. `_update_narrative()` called:
   - `score_story(text, "Ongoing deterioration of Russian gas...", 0.82, 0.78)` → Cerebras: `{"surprise": 0.71, "impact": 0.85}` (less surprising, more impactful)
   - `narrative.append_surprise(0.71)` → series now has 2 points
   - `narrative.append_impact(0.85)` → series now has 2 points
   - `narrative.add_headline("Germany activates emergency gas rationing plan")` → event_count = 2
   - `get_embedding(narrative.id)` → retrieves current centroid
   - `_blend_embedding(current, new, n=1)` → updated centroid
   - `update_narrative(narrative, new_embedding)` → ChromaDB updated
6. Return: `{action: "updated", narrative_name: "European energy supply shock", model_risk: 0.78, ...}`

### Scenario C: Unrelated story creates second narrative

1. `"Federal Reserve signals third consecutive rate cut"`
2. `embed_text(...)` → different semantic direction
3. `query_nearest(embedding, n=5)`:
   - Best match: "European energy supply shock", distance = 0.74 (unrelated)
4. `0.74 >= 0.40` → creates new narrative: `"Federal Reserve monetary easing cycle"`
5. Now 2 narratives in DB

### Scenario D: Semantic search query

1. User types `"sanctions affecting energy exports"` in search panel
2. `embed_text("sanctions affecting energy exports")` → embedding
3. `query_nearest(embedding, n=5)` → returns nearest narratives with distances
4. "European energy supply shock" returns with distance=0.22 → similarity=89%
5. Results rendered in search panel with clickable narrative cards

---

## 12. Design Decisions & Rationale

### D1: max() for global risk index
Any single high-risk narrative fully elevates the index. If a bank collapses (risk=0.95), 15 routine narratives at 0.20 should NOT dilute the signal. Risk is not averaged away in reality.

### D2: sqrt(surprise × impact) for per-narrative risk
Geometric mean requires BOTH to be high. A very surprising but irrelevant event (surprise=0.9, impact=0.1) → risk=0.30. This correctly models the condition: markets break down when events are both unexpected AND large enough to move prices.

### D3: Cosine distance [0, 2] not similarity
ChromaDB HNSW returns distances not similarities. The threshold 0.40 means "within 40% of being identical vectors." This is human-interpretable.

### D4: Narrative directions, not individual stories
Keeps ChromaDB small, query-fast, and semantically stable. Individual stories are ephemeral inputs; narrative directions are the persistent risk memory.

### D5: Online mean for centroid blending
Equal weight for all historical stories regardless of time. The centroid should represent the full breadth of the narrative, not just recent stories. An EMA would drift toward recent events — acceptable tradeoff if desired.

### D6: Both embedding and LLM inference on Modal
Both `Embedder` (T4 GPU) and `LLM` (A10G GPU) run on Modal, deployed from `model/modal_app.py`. The backend holds only thin client wrappers (`embedder.py`, `llm_client.py`) with no local model weights. This keeps the backend lightweight, gives the model team full ownership of inference infrastructure, and means GPU selection can be tuned independently per model class.

### D7: SSE not WebSockets
Data flows strictly server → client for live updates. SSE is simpler, auto-reconnects, and works through proxies. WebSockets add unnecessary complexity for unidirectional push.

### D8: Temperature 0.1 for all Cerebras calls
Near-deterministic JSON output. Lower hallucination rate for structured responses. Crucial because malformed JSON breaks the ingestion pipeline.

---

## 13. Known Issues & Mitigations

### I1: ChromaDB concurrent write corruption
- Risk: Two ingest workers write simultaneously → SQLite corruption
- Fix: `_write_lock = threading.Lock()` in vector_store.py wrapping all `.add()`, `.update()`, `.delete()` calls

### I2: TOCTOU race in narrative routing
- Risk: Two workers both decide to create the same narrative direction
- Fix: `_route_lock = threading.Lock()` in narrative_engine.py wrapping the query+decision block (release before slow Cerebras call)

### I3: Cerebras 429 rate limiting
- Fix: Exponential backoff in `_chat()` (3 attempts, 1s/2s/4s delays). Fallback values used on final failure.

### I4: Narrative centroid drift
- Risk: After many updates, centroid drifts far from original direction
- Monitor: `query_nearest()` distances should stay < 0.30 for related stories
- Fix: If centroid drift is detected, optionally cap blending weight for old stories

### I5: Narrative proliferation (threshold too high)
- Risk: threshold too high → every story creates a new narrative → hundreds of single-story narratives
- Monitor: `pipeline_stats["narratives_created"] / pipeline_stats["stories_ingested"]` ratio. Should be < 0.30.
- Fix: Lower threshold to 0.35 via env var

### I6: RSS duplicate stories across polls
- Fix: DeduplicatingCache with SHA256 hash of `headline + body[:200]`, 10K capacity

### I7: Time series size growth
- Fix: `MAX_SERIES_LENGTH = 500` cap in `append_surprise()` and `append_impact()`. Oldest points trimmed.

### I8: JSON parse failure from Cerebras
- Fix: Three-level parse: `json.loads(raw)` → extract JSON block → use fallback dict

### I9: Frontend served from file:// (CORS)
- Fix: `allow_origins=["*"]` in FastAPI CORS middleware. Alternatively, serve frontend via FastAPI StaticFiles.

### I10: Large PDF/article body truncation
- Fix: Story body truncated to 1500 chars in all Cerebras prompts. Sufficient for narrative identification.

---

## 14. Complete File Map

```
hackillinois2026/
├── goal.md                         (project brief)
├── PLAN.md                         (this file — master implementation guide)
├── .env                            (gitignored — real keys)
├── .env.example                    (template)
├── .gitignore
│
├── backend/
│   ├── main.py                     [MODIFY: add lifespan, all 6 routers, StaticFiles]
│   ├── requirements.txt            [MODIFY: add aiohttp, feedparser, praw]
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               [MODIFY: add all new settings fields]
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── narrative.py            [MODIFY: add is_active, trends, series cap, add_headline]
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   └── vector_store.py         [MODIFY: add _write_lock, delete_narrative, get_embedding]
│   │
│   ├── modal_app.py                [NEW: Modal LLM deployment — deploy with `modal deploy`]
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_client.py           [NEW: Modal client — label_narrative, score_story, summarize_narrative_context]
│   │   ├── embedder.py             [EXISTS: no changes needed]
│   │   ├── narrative_engine.py     [MODIFY: add _route_lock, fix blend order, full ingest_story]
│   │   ├── scraper.py              [NEW: RawStory, DeduplicatingCache, rss_poller, reddit_poller]
│   │   └── pipeline.py             [NEW: raw_queue, pipeline_stats, ingest_worker, start/stop_pipeline]
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── ingest.py           [MODIFY: add batch endpoint, validation]
│   │       ├── narratives.py       [MODIFY: add query params, /history, DELETE, trends in response]
│   │       ├── risk.py             [MODIFY: add /history, method param, active_only]
│   │       ├── chat.py             [NEW: POST /api/chat RAG endpoint]
│   │       ├── events.py           [NEW: GET /api/events/stream SSE]
│   │       └── pipeline.py         [NEW: GET /api/pipeline/stats]
│   │
│   └── chroma_db/                  (auto-generated — gitignored)
│
└── frontend/
    ├── index.html                  [REWRITE: full dashboard HTML with all panels]
    ├── styles.css                  [NEW: dark theme, CSS variables, grid layout]
    └── dashboard.js                [NEW: all JS — state, fetch, render, SSE, Chart.js, modal]
```

---

*End of implementation plan. Build in this order: config → models → vector_store → cerebras_client → embedder → narrative_engine → scraper → pipeline → routes (ingest, narratives, risk, chat, events, pipeline) → main.py → frontend.*
