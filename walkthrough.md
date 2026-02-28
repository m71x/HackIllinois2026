# NEXUS — Real-World Model Risk Engine

> **HackIllinois 2026 Project**
> A system that monitors global news, identifies real-world narratives (geopolitical tensions, banking crises, supply chain shocks, etc.), and quantitatively measures how those narratives threaten the reliability of statistical trading models.

---

## The Core Idea

Quantitative financial models assume the future resembles the recent past. During black-swan events (2008 crisis, COVID crash, Russia-Ukraine commodity shock), those assumptions break — but models have no way of knowing it. **NEXUS answers: "How much should we trust our models right now?"**

It does this by:
1. **Scraping** live news and tweets
2. **Embedding** them into semantic vectors
3. **Routing** each story into a persistent *narrative direction* (e.g., "Energy supply shock")
4. **Scoring** each story for **Surprise** and **Impact**
5. **Computing** a composite **Model Risk Index**

---

## System Architecture

![NEXUS System Architecture](/home/mikexi/.gemini/antigravity/brain/7c32a9ae-eb08-4b01-bb07-c6f8c5e683f9/architecture_diagram_1772254837896.png)

---

## End-to-End Data Flow

![NEXUS End-to-End Data Flow](/home/mikexi/.gemini/antigravity/brain/7c32a9ae-eb08-4b01-bb07-c6f8c5e683f9/dataflow_diagram_1772254851764.png)

---

## Narrative Routing Logic

![Narrative Routing Decision Flow](/home/mikexi/.gemini/antigravity/brain/7c32a9ae-eb08-4b01-bb07-c6f8c5e683f9/routing_flowchart_1772254862780.png)

---

## Component Breakdown

### 1. Configuration — [config.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/core/config.py)

All settings loaded from [.env](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/.env) via Pydantic `BaseSettings`:

| Setting | Default | Purpose |
|---|---|---|
| `cerebras_api_key` | — | LLM inference key (Cerebras cloud) |
| `cerebras_model` | `llama-3.3-70b` | Which Cerebras model to use |
| `modal_app_name` | `model-risk-llm` | Modal deployment name for embeddings |
| `chroma_persist_dir` | `./chroma_db` | On-disk ChromaDB location |
| `new_narrative_threshold` | `0.40` | Cosine distance cutoff for narrative routing |
| `poll_interval_seconds` | `300` | Auto-pipeline scrape interval (5 min) |
| `newsapi_key` / `twitter_bearer_token` | — | API keys for data sources |

---

### 2. Scraper — [scraper.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/scraper.py)

- **NewsAPI** — article headlines + descriptions via `newsapi-python`, financial/geopolitical keywords
- **Twitter** — recent tweets via `tweepy`, Twitter v2 search, English non-retweet filter
- **Deduplication** — SHA-256 hash LRU cache (10k entries), prevents re-ingestion within a session

---

### 3. Story Buffer — [story_buffer.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/story_buffer.py)

Thread-safe in-memory holding area enabling **staged ingestion**: scrape → inspect → commit. Decouples scraping from expensive embedding/LLM calls.

---

### 4. Embedder — [embedder.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/embedder.py) + [modal_app.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/model/modal_app.py)

`all-MiniLM-L6-v2` (384-dim, L2-normalized) on **Modal** cloud GPU (T4). Supports single and batch embedding.

---

### 5. LLM Client — [llm_client.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/llm_client.py)

**Llama 3.3 70B** via **Cerebras** cloud API. Three functions:

| Function | Output |
|---|---|
| [label_narrative(story)](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/llm_client.py#139-151) | `{"name": "...", "description": "..."}` |
| [score_story(story, narrative)](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/llm_client.py#153-181) | `{"surprise": 0.7, "impact": 0.85}` |
| [summarize_narrative_context(narratives, query)](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/llm_client.py#183-204) | RAG-style prose answer |

Impact scoring uses an event severity hierarchy: *sovereign default > bank collapse > military conflict > sanctions > ...*

---

### 6. Narrative Engine — [narrative_engine.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/narrative_engine.py)

The **brain** of the system. Routes each story to an existing or new narrative based on cosine distance threshold (0.40). Updates narrative centroids via online mean blending: `new = old × n/(n+1) + story/(n+1)`.

---

### 7. Data Model — [narrative.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/models/narrative.py)

Each [NarrativeDirection](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/models/narrative.py#12-70) stores name, description, surprise/impact time series, recent headlines, and event count. Composite risk: `model_risk = √(surprise × impact)`.

---

### 8. Vector Store — [vector_store.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/db/vector_store.py)

**ChromaDB** with cosine distance. Each document is a narrative direction (not a raw story). Metadata holds all scalar fields + JSON-serialized time series.

---

### 9. Ticker Service — [ticker_service.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/ticker_service.py)

Connects stock tickers to narratives: fetch company metadata from **yfinance** → build rich embed text → embed with same model → query ChromaDB for nearest narratives. Cached with 24h TTL.

---

### 10. Pipeline — [pipeline.py](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/backend/services/pipeline.py)

Background `asyncio` task: scrape → ingest → broadcast SSE events. Runs every 5 min by default.

---

### 11. API Routes

| Prefix | Key Endpoints |
|---|---|
| `/api/ingest` | `POST /` (single), `POST /batch`, `POST /scrape` |
| `/api/narratives` | `GET /` (list), `GET /{id}`, `POST /search` |
| `/api/risk` | `GET /` (aggregate model risk index) |
| `/api/pipeline` | `GET /buffer`, `POST /process`, `DELETE /buffer`, `GET /stats` |
| `/api/tickers` | `POST /relate`, `POST /expose`, `GET /{symbol}` |

**Staged flow:** `POST /scrape {buffer: true}` → `GET /buffer` → `POST /process`

---

### 12. Frontend — [Frontend/](file:///home/mikexi/antigravity_projects_ubuntu/hackillinois2026/Frontend)

Dark-themed glassmorphism dashboard with two tabs:

- **Risk Dashboard:** Model Risk Gauge, 24h Chart, Pipeline Stats, Narrative Table, Semantic Search, Live Feed, Manual Ingest
- **Global Equities:** Market Clocks, Watchlist

Built with vanilla HTML/CSS/JS, Chart.js, Phosphor Icons, Outfit/Inter/JetBrains Mono fonts.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI, Uvicorn |
| **LLM** | Llama 3.3 70B via Cerebras Cloud |
| **Embeddings** | all-MiniLM-L6-v2 on Modal (T4 GPU) |
| **Vector DB** | ChromaDB (persistent, cosine distance) |
| **News Sources** | NewsAPI, Twitter/X API v2 |
| **Stock Data** | yfinance |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js, Phosphor Icons |
