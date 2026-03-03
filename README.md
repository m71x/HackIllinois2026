The Problem

  Quantitative trading models assume the future resembles the recent past. During black-swan events — the 2008 crisis, the COVID crash, the Russia-Ukraine       
  commodity shock — those assumptions shatter. The models keep trading as if the world hasn't changed, often with catastrophic results.

  **There is no systematic tool that answers: *"How much should we trust our models right now?"***

  NEXUS answers that question in real time.

  ---

  ## What It Does

  NEXUS continuously ingests 90+ live news sources, identifies emerging real-world narratives (geopolitical tensions, banking crises, supply chain shocks), and  
  computes a live **Model Risk Index** — a scalar signal that quantifies when statistical trading models are flying blind.

  | Step | What happens |
  |------|-------------|
  | **Scrape** | 5000+ RSS feeds + NewsAPI polled every 2 minutes |
  | **Embed** | Each headline → 384-dim semantic vector on Modal T4 GPU |
  | **Route** | Cosine similarity matches story to nearest narrative (threshold 0.40) |
  | **Score** | Surprise + Impact heuristics updated via exponential moving average |
  | **Index** | Model Risk Index = weighted aggregate of √(surprise × impact) |
  | **Broadcast** | Live updates pushed to the dashboard via Server-Sent Events |

  ---

  ## The Risk Index

  Each active narrative tracks two scores:

  - **Surprise** — How unexpected recent developments are. High when stories are semantically distant from the narrative's historical center, or contain shock   
  language ("unprecedented", "sudden reversal", "record high").
  - **Impact** — How economically significant the narrative is. Driven by scale keywords (sovereign default, military conflict, sanctions).

  **Narrative Model Risk** = √(surprise × impact)

  | Range | Label | Meaning |
  |-------|-------|---------|
  | 0.00 – 0.33 | STABLE | Markets behaving statistically. Models reliable. |
  | 0.34 – 0.66 | ELEVATED | Narratives building. Monitor closely. |
  | 0.67 – 1.00 | CRITICAL | Regime shift detected. Discount quant signals. |

  ---

  ## AI Models

  ### `all-MiniLM-L6-v2` (Sentence Transformers)
  Deployed on Modal (NVIDIA T4 GPU). Converts every headline and narrative into a 384-dimensional semantic vector. Narrative routing is purely semantic —        
  "Russian gas pipeline sabotage" and "European energy rationing" route to the same narrative with no keyword matching. A thread-safe batching queue coalesces up
   to 32 stories per GPU call, cutting round-trip overhead by ~97%.

  ### `Qwen 2.5 0.5B-Instruct`
  Deployed on Modal (CPU). Autonomously writes 3–5 word financial narrative labels from raw news headlines using a few-shot financial analyst prompt. Every      
  narrative name on the dashboard — "SVB Bank Run Collapse", "China Chip Export Ban" — was written by this model with zero manual curation. Falls back instantly 
  to a keyword heuristic labeler if Modal is unreachable.

  ---

  ## Tech Stack

  | Layer | Technology |
  |-------|-----------|
  | Backend | Python 3.11, FastAPI, Uvicorn |
  | Embeddings | all-MiniLM-L6-v2 · Modal T4 GPU |
  | Labeling | Qwen 2.5 0.5B-Instruct · Modal CPU |
  | Vector DB | ChromaDB (persistent, cosine distance) |
  | News Sources | 90+ RSS feeds, NewsAPI |
  | Stock Data | yfinance |
  | Frontend | Vanilla JS, Chart.js, Three.js, Phosphor Icons |
  | Real-Time | Server-Sent Events (SSE) |

  ---

  ## Getting Started

  ### Prerequisites
  - Python 3.11+
  - A [Modal](https://modal.com) account (free tier works)
  - *(Optional)* [NewsAPI](https://newsapi.org) key for broader coverage

  ### Setup

  ```bash
  # 1. Install dependencies
  cd backend && pip install -r requirements.txt

  # 2. Configure environment — copy and fill in .env
  cp .env.example .env

  # 3. Deploy models to Modal (one-time)
  modal deploy model/modal_app.py

  # 4. Start the backend (dashboard served at http://localhost:8000)
  uvicorn main:app --reload --port 8000

  Or use the one-command startup script:
  bash start.sh        # Linux/macOS
  ./start.ps1          # Windows

  Offline mode: If the backend isn't running, the dashboard automatically switches to mock data so the UI stays fully explorable.

  Environment Variables

  MODAL_APP_NAME=model-risk-llm     # must match APP_NAME in modal_app.py
  CHROMA_PERSIST_DIR=./chroma_db
  NEW_NARRATIVE_THRESHOLD=0.40      # cosine distance cutoff for new narrative creation
  NEWSAPI_KEY=                      # optional
  POLL_INTERVAL_SECONDS=120

  ---
  API Reference

  ┌────────┬────────────────────────┬───────────────────────────────────────────┐
  │ Method │        Endpoint        │                Description                │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ GET    │ /api/risk              │ Live Model Risk Index                     │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ GET    │ /api/risk/history      │ Historical risk time series               │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ GET    │ /api/narratives        │ All active narrative directions           │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ POST   │ /api/narratives/search │ Semantic search by query text             │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ GET    │ /api/narratives/graph  │ Clustered narrative graph (PCA + K-means) │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ POST   │ /api/ingest            │ Submit a story manually                   │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ POST   │ /api/ingest/scrape     │ Trigger an on-demand live scrape          │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ POST   │ /api/tickers/relate    │ Map ticker symbols to narratives          │
  ├────────┼────────────────────────┼───────────────────────────────────────────┤
  │ GET    │ /api/events/stream     │ SSE live event feed                       │
  └────────┴────────────────────────┴───────────────────────────────────────────┘

  ---
  Who This Is For

  ┌─────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │        User         │                                       Use case                                        │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Quant traders / PMs │ Model Risk Index as a regime indicator; drill into which narratives affect positions  │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Risk managers       │ Early warning of narrative-driven instability before volatility spikes                │
  ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Research teams      │ Semantic search of historical narrative data to study real-world → market propagation │
  └─────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Built at HackIllinois 2026

  Built with FastAPI · Modal · ChromaDB · Three.js · Chart.js · Claude Code
  ```

✻ Crunched for 37s

❯ make it more simple, readable, also for students and anyone interested in stocks that has no knowledge                                                                                                                                                        
● # NEXUS — Real-World Model Risk Engine                                                                                                                                                                                                                                                                                          
  > **Should you trust your trading algorithms right now?** NEXUS tells you.                                                                                        
  ---                                                                                                                                                                                                                                                                                                                             
  ## What is this?

  Wall Street runs on math. Banks and hedge funds use algorithms that look at
  historical price patterns to decide when to buy and sell. These algorithms
  work great — until something unexpected happens in the real world.

  A war breaks out. A major bank collapses. A pandemic shuts everything down.

  When that happens, the algorithms keep trading like everything is normal.
  They have no way of knowing the world just changed.

  **NEXUS fixes that.**

  It reads the news 24/7, detects when something big is happening, and gives
  traders a live score: *"How much should you trust your algorithms right now?"*

  ---

  ## How it works (in plain English)

  1. **Reads the news** — Pulls from 90+ sources (Reuters, CNBC, BBC, Yahoo Finance, etc.) every 2 minutes
  2. **Groups related stories** — "Russian gas pipeline cut" and "Europe energy rationing" get grouped into the same theme automatically, using AI
  3. **Scores each theme** — How surprising is this? How economically significant?
  4. **Outputs a Risk Index** — A single number from 0 to 1 that tells you how reliable trading algorithms are right now

  | Score | What it means |
  |-------|--------------|
  | 0.0 – 0.33 | 🟢 Normal. Algorithms are reliable. |
  | 0.34 – 0.66 | 🟡 Something is building. Pay attention. |
  | 0.67 – 1.00 | 🔴 Major event detected. Don't fully trust the algorithms. |

  ---

  ## What you can do with it

  - **See the live risk score** on a real-time dashboard
  - **Search news themes** by meaning, not just keywords (e.g. "banking stress")
  - **Look up a stock ticker** and see which news themes are affecting it
  - **Watch new narratives emerge** as events unfold

  ---

  ## The dashboard

  Built with a dark glassmorphism UI, the dashboard includes:
  - Live risk gauge with color-coded alerts
  - Narrative table ranked by risk level
  - Historical risk chart
  - 3D semantic map of how news themes relate to each other
  - Per-narrative time series (surprise and impact over time)
  - Ticker exposure tool — type in a stock and see related risks

  ---

  ## Tech used (for the curious)

  | What | How |
  |------|-----|
  | News reading | 90+ RSS feeds + NewsAPI |
  | AI that groups stories | `all-MiniLM-L6-v2` — runs on a GPU in the cloud |
  | AI that names the groups | `Qwen 2.5` — writes labels like "SVB Bank Run" automatically |
  | Database | ChromaDB — stores everything as semantic vectors |
  | Backend | Python + FastAPI |
  | Frontend | HTML, CSS, JavaScript, Chart.js, Three.js |
  | Cloud GPU | Modal (NVIDIA T4) |

  ---

  ## Run it yourself

  ```bash
  # Install dependencies
  cd backend && pip install -r requirements.txt

  # Add your API keys to .env (see .env.example)

  # Deploy the AI models to the cloud (free Modal account needed)
  modal deploy model/modal_app.py

  # Start the server — dashboard opens at http://localhost:8000
  uvicorn main:app --reload --port 8000
