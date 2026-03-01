# NEXUS — Real-World Model Risk Engine

> HackIllinois 2026

Quantitative financial models assume the future resembles the recent past. During black-swan events — the 2008 crisis, the COVID crash, the Russia-Ukraine commodity shock — those assumptions break, and the models have no way of knowing it. **NEXUS answers: "How much should we trust our models right now?"**

NEXUS continuously monitors 90+ live news sources, identifies emerging real-world narratives (geopolitical tensions, banking crises, supply chain shocks), and computes a live **Model Risk Index** that tells traders when their statistical models are flying blind.

---

## What It Does

1. **Scrapes** 90+ RSS feeds and NewsAPI every 2 minutes
2. **Embeds** each headline into a 384-dimensional semantic vector (GPU inference on Modal)
3. **Routes** each story to a matching narrative direction (e.g., "Energy Supply Shock") or creates a new one if sufficiently novel
4. **Scores** each narrative for **Surprise** (how unexpected) and **Impact** (how economically significant)
5. **Computes** a composite **Model Risk Index** = √(surprise × impact) across all active narratives
6. **Broadcasts** live updates to the dashboard via Server-Sent Events

---

## AI Models in the System

### all-MiniLM-L6-v2 (Sentence Transformer)
- **Role:** Semantic embedding engine
- **Deployed on:** Modal cloud, NVIDIA T4 GPU
- **What it generates:** 384-dimensional L2-normalized semantic vectors for every news headline and narrative
- **Why it matters:** Enables similarity-based narrative routing and ticker search — a story about "Russian gas pipeline sabotage" and "European energy rationing" get routed to the same narrative without keyword matching
- **Optimization:** Thread-safe batching queue coalesces up to 32 stories per Modal call, reducing GPU round-trip overhead by ~97%

### Qwen 2.5 0.5B-Instruct
- **Role:** Narrative labeler
- **Deployed on:** Modal cloud, CPU
- **What it generates:** Concise 3–5 word financial narrative labels — e.g., "SVB Bank Run Collapse", "Fed Rate Hike", "China Chip Export Ban"
- **Why it matters:** Zero manual curation. Every narrative name on the dashboard was written autonomously by this model from a raw news headline using a few-shot financial analyst prompt
- **Fallback:** If Modal is unreachable, an instant keyword/heuristic labeler takes over with no latency penalty

### Heuristic Scoring Engine
- **Role:** Story surprise and impact scorer
- **What it generates:** Surprise (0–1) and Impact (0–1) scores per story, updated continuously via exponential moving average
- **Signals used:**
  - Semantic distance from narrative centroid (how novel is this story?)
  - Shock language detection ("unprecedented", "sudden", "reversal")
  - Magnitude language detection ("record", "all-time high", "40-year")
  - Staleness bonus (dormant narrative suddenly reactivated)
  - Directional inversion bonus (contradicts prior trend)

---

## AI Tools Used to Build It

### Claude Code (Anthropic)
Used throughout development for:
- Architecture design and API contract definitions
- Backend service implementation (scraper, embedder, narrative engine, pipeline)
- Prompt engineering for the Qwen labeler
- Frontend dashboard (HTML/CSS/JS, Three.js 3D visualization, Chart.js)
- Debugging and refactoring across the full stack
- Pitch script writing and demo preparation

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI, Uvicorn |
| **Embeddings** | all-MiniLM-L6-v2 on Modal (T4 GPU) |
| **Labeling** | Qwen 2.5 0.5B-Instruct on Modal (CPU) |
| **Vector DB** | ChromaDB (persistent, cosine distance) |
| **News Sources** | 90+ RSS feeds (CNBC, Reuters, BBC, Yahoo Finance, etc.), NewsAPI |
| **Stock Data** | yfinance |
| **Model Serving** | Modal (serverless GPU/CPU inference) |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js, Three.js, Phosphor Icons |
| **Real-Time** | Server-Sent Events (SSE) |

---

## Architecture

```
90+ Live News Sources (RSS, NewsAPI)
         │
         ▼
  scraper.py  ──  Dedup cache (SHA-256 LRU, 10k entries)
         │
         ▼
  story_buffer.py  ──  Thread-safe in-memory queue
         │
         ▼
  embedder.py → modal_app.py (Embedder)
         │         all-MiniLM-L6-v2, T4 GPU
         │         Batch queue (32 stories/call)
         ▼
  narrative_engine.py
         │  ├── Cosine similarity query (ChromaDB)
         │  ├── Route to existing narrative (threshold 0.40)
         │  │       └── Heuristic surprise/impact scoring
         │  └── Create new narrative
         │           └── Qwen 2.5 0.5B label generation (Modal)
         ▼
  ChromaDB vector store  ──  Persistent narrative directions
         │
         ▼
  FastAPI  ──  REST API + SSE live feed
         │
         ▼
  Frontend Dashboard  ──  Risk gauge, narrative table,
                          semantic search, 3D vector viz
```

---

## Repository Structure

```
HackIllinois2026/
├── api/                    # Shared API contracts (schemas, REST spec, SSE spec)
│   ├── README.md           # API contract documentation
│   ├── model_contract.py   # Python ABCs for Modal classes
│   ├── schemas.py          # Pydantic request/response models
│   ├── rest_api.md         # Every REST endpoint documented
│   ├── sse_events.md       # SSE event stream format
│   └── client.js           # Ready-to-use JS fetch wrappers
├── backend/
│   ├── main.py             # FastAPI app + startup/shutdown
│   ├── core/
│   │   ├── config.py       # Pydantic settings (env-based)
│   │   └── state.py        # Shared app state
│   ├── services/
│   │   ├── scraper.py      # RSS + NewsAPI scraper
│   │   ├── embedder.py     # Modal embedder client
│   │   ├── narrative_engine.py  # Routing + scoring brain
│   │   ├── llm_client.py   # Narrative labeler (Qwen / heuristic)
│   │   ├── pipeline.py     # Async background scrape loop
│   │   ├── story_buffer.py # Thread-safe story queue
│   │   └── ticker_service.py    # Ticker → narrative mapping
│   ├── api/                # FastAPI route handlers
│   ├── models/             # Pydantic data models
│   └── db/                 # ChromaDB vector store
├── model/
│   └── modal_app.py        # Embedder + Labeler Modal deployment
├── Frontend/
│   ├── index.html          # Main dashboard
│   ├── dashboard.js        # All frontend logic + Three.js 3D viz
│   ├── styles.css          # Glassmorphism dark theme
│   ├── login.html          # Auth gate
│   └── login.css
├── goal.md                 # Original problem statement
├── plan.md                 # Full implementation plan
├── walkthrough.md          # System architecture walkthrough
└── start.sh / start.ps1   # One-command startup scripts
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/risk` | Live Model Risk Index |
| GET | `/api/narratives` | All active narrative directions |
| POST | `/api/narratives/search` | Semantic search by query text |
| GET | `/api/narratives/{id}` | Single narrative + full time series |
| POST | `/api/ingest` | Submit a single story |
| POST | `/api/ingest/scrape` | Trigger live scrape |
| POST | `/api/tickers/relate` | Map ticker to narratives |
| GET | `/api/events/stream` | SSE live event feed |

---

## Running Locally

```bash
# 1. Install dependencies
cd backend && pip install -r requirements.txt

# 2. Deploy the Modal models (one-time)
modal deploy model/modal_app.py

# 3. Start the backend
uvicorn backend.main:app --reload

# 4. Open the frontend
open Frontend/index.html
```

Or use the startup script:

```bash
bash start.sh
```

---

## How the Risk Index Works

Each active narrative tracks two scores over time:

- **Surprise** — How unexpected recent developments are. High when stories are semantically distant from the narrative's historical center (novelty) or contain shock/magnitude language.
- **Impact** — How economically significant the narrative is. Driven by keywords indicating scale (sovereign default, military conflict, sanctions vs. routine earnings).

The composite **Model Risk** for a narrative = √(surprise × impact).

The dashboard **Model Risk Index** aggregates all active narratives, weighted by peak risk to prevent false calms during compound crises.

| Range | Label | Meaning |
|---|---|---|
| 0.00 – 0.33 | LOW | Markets behaving statistically. Models reliable. |
| 0.34 – 0.66 | MEDIUM | Narratives building. Monitor closely. |
| 0.67 – 1.00 | HIGH | Regime shift. Model fragility. Discount quant signals. |
