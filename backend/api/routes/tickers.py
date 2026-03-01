"""
Ticker Routes
=============
Semantic association between stock tickers and narrative directions.

How it works
------------
For each ticker, yfinance supplies the company name, sector, industry, and
business description. These are concatenated into a rich text string that is
embedded using the same model and space as the narrative directions. The
resulting vector is queried against ChromaDB to find the closest narratives.

Endpoints
---------
POST /api/tickers/relate
    Given a list of tickers, return the narrative directions most semantically
    related to each one.  "Which narratives is JPM exposed to?"

POST /api/tickers/expose
    Given a narrative id and a list of tickers, rank the tickers by their
    semantic proximity to that narrative.  "Which stocks are most exposed to
    the 'regional banking stress' narrative?"

GET  /api/tickers/{symbol}
    Single-ticker convenience: company info + top related narratives.

DELETE /api/tickers/cache
    Evict the in-process yfinance cache (useful after corporate actions).
"""

import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.ticker_service import get_ticker_info, get_ticker_info_batch, invalidate_cache
from services.embedder import embed_text, embed_batch
from db import vector_store

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _narrative_to_dict(narrative, distance: float) -> dict:
    return {
        "id": narrative.id,
        "name": narrative.name,
        "description": narrative.description,
        "distance": round(distance, 4),
        "similarity": round(1 - (distance / 2), 4),   # [0,1], higher = more related
        "model_risk": narrative.model_risk,
        "current_surprise": narrative.current_surprise,
        "current_impact": narrative.current_impact,
        "event_count": narrative.event_count,
    }


# ---------------------------------------------------------------------------
# POST /api/tickers/relate — tickers → closest narratives
# ---------------------------------------------------------------------------

class RelateRequest(BaseModel):
    tickers: list[str] = Field(
        ..., min_length=1, max_length=50,
        description="List of ticker symbols, e.g. ['JPM', 'XOM', 'NVDA']",
    )
    n_results: int = Field(
        default=5, ge=1, le=20,
        description="Number of narrative directions to return per ticker.",
    )
    active_only: bool = Field(
        default=False,
        description="If true, exclude narratives that have gone inactive.",
    )


@router.post("/relate")
async def relate_tickers(req: RelateRequest):
    """
    Find which narrative directions each ticker is semantically exposed to.

    For each ticker:
      1. Fetch company metadata from yfinance (cached 24h)
      2. Build embed text: "{name} ({symbol}) — {sector}, {industry}\\n{description}"
      3. Embed that text (batched — one Modal round-trip for all tickers)
      4. Query ChromaDB for nearest narrative directions

    Response shape
    --------------
    {
      "results": [
        {
          "ticker": "JPM",
          "company_name": "JPMorgan Chase & Co.",
          "sector": "Financial Services",
          "industry": "Banks—Diversified",
          "embed_text": "...",
          "narratives": [
            {
              "id": "...",
              "name": "Regional banking stress",
              "distance": 0.18,
              "similarity": 0.91,
              "model_risk": 0.64,
              ...
            }
          ]
        }
      ],
      "errors": {"BADTICKER": "No data found for ticker 'BADTICKER'"}
    }
    """
    loop = asyncio.get_event_loop()

    # Deduplicate while preserving order
    seen = set()
    symbols = [s.upper() for s in req.tickers if s.upper() not in seen and not seen.add(s.upper())]

    # Fetch all ticker info (cached; errors per-ticker)
    info_map = await loop.run_in_executor(None, lambda: get_ticker_info_batch(symbols))

    # Split successes from errors
    good_symbols = [s for s in symbols if not isinstance(info_map[s], Exception)]
    errors = {
        s: str(info_map[s]) for s in symbols if isinstance(info_map[s], Exception)
    }

    if not good_symbols:
        return {"results": [], "errors": errors}

    # Batch-embed all valid tickers in one Modal call
    embed_texts = [info_map[s].embed_text for s in good_symbols]
    embeddings = await loop.run_in_executor(None, lambda: embed_batch(embed_texts))

    # Query ChromaDB per ticker
    results = []
    for symbol, embedding in zip(good_symbols, embeddings):
        info = info_map[symbol]
        nearest = await loop.run_in_executor(
            None,
            lambda emb=embedding: vector_store.query_nearest(emb, n_results=req.n_results)
        )

        narratives = []
        for narrative, distance, _emb in nearest:
            narratives.append(_narrative_to_dict(narrative, distance))

        results.append({
            "ticker": symbol,
            "company_name": info.name,
            "sector": info.sector,
            "industry": info.industry,
            "embed_text": info.embed_text,
            "narratives": narratives,
        })

    return {"results": results, "errors": errors}


# ---------------------------------------------------------------------------
# POST /api/tickers/expose — given a narrative, rank tickers by exposure
# ---------------------------------------------------------------------------

class ExposeRequest(BaseModel):
    narrative_id: str = Field(..., description="ID of the narrative direction.")
    tickers: list[str] = Field(
        ..., min_length=1, max_length=100,
        description="Tickers to rank by exposure to this narrative.",
    )


@router.post("/expose")
async def expose_narrative(req: ExposeRequest):
    """
    Given a narrative direction, rank a list of tickers by how semantically
    exposed they are to it.

    Useful for: "Given the 'energy supply shock' narrative is flaring up,
    which of my holdings are most at risk?"

    Similarity is cosine similarity [0, 1] between the ticker's company
    description embedding and the narrative's centroid embedding.

    Response shape
    --------------
    {
      "narrative_id": "...",
      "narrative_name": "Energy supply shock",
      "rankings": [
        {"ticker": "XOM", "company_name": "Exxon Mobil", "distance": 0.14, "similarity": 0.93},
        {"ticker": "CVX", "company_name": "Chevron",      "distance": 0.17, "similarity": 0.92},
        ...
      ],
      "errors": {}
    }
    """
    loop = asyncio.get_event_loop()

    # Get the narrative's stored centroid embedding
    narrative_embedding = await loop.run_in_executor(
        None, lambda: vector_store.get_embedding(req.narrative_id)
    )
    if not narrative_embedding:
        raise HTTPException(status_code=404, detail=f"Narrative '{req.narrative_id}' not found.")

    narrative = await loop.run_in_executor(
        None, lambda: vector_store.get_narrative(req.narrative_id)
    )

    # Deduplicate tickers
    seen = set()
    symbols = [s.upper() for s in req.tickers if s.upper() not in seen and not seen.add(s.upper())]

    # Fetch ticker info
    info_map = await loop.run_in_executor(None, lambda: get_ticker_info_batch(symbols))

    good_symbols = [s for s in symbols if not isinstance(info_map[s], Exception)]
    errors = {s: str(info_map[s]) for s in symbols if isinstance(info_map[s], Exception)}

    if not good_symbols:
        return {
            "narrative_id": req.narrative_id,
            "narrative_name": narrative.name if narrative else "unknown",
            "rankings": [],
            "errors": errors,
        }

    # Batch-embed all ticker descriptions
    embed_texts = [info_map[s].embed_text for s in good_symbols]
    embeddings = await loop.run_in_executor(None, lambda: embed_batch(embed_texts))

    # Compute cosine distance between each ticker and the narrative centroid
    # Both vectors are L2-normalized, so dot product = cosine similarity,
    # and distance = 1 - dot_product (ChromaDB cosine distance formula: 1 - cos_sim)
    rankings = []
    for symbol, ticker_emb in zip(good_symbols, embeddings):
        info = info_map[symbol]
        dot = sum(a * b for a, b in zip(narrative_embedding, ticker_emb))
        # Clamp to [-1, 1] to handle floating point drift
        dot = max(-1.0, min(1.0, dot))
        distance = 1.0 - dot         # cosine distance [0, 2] but dot∈[-1,1] → dist∈[0,2]
        similarity = round(dot, 4)   # dot product = cosine similarity for unit vectors

        rankings.append({
            "ticker": symbol,
            "company_name": info.name,
            "sector": info.sector,
            "industry": info.industry,
            "distance": round(distance, 4),
            "similarity": similarity,
        })

    # Sort by similarity descending (most exposed first)
    rankings.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "narrative_id": req.narrative_id,
        "narrative_name": narrative.name if narrative else "unknown",
        "rankings": rankings,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# GET /api/tickers/{symbol} — single ticker info + top narratives
# ---------------------------------------------------------------------------

@router.get("/{symbol}")
async def get_ticker(
    symbol: str,
    n_results: int = 5,
    active_only: bool = False,
):
    """
    Convenience endpoint: company metadata + top related narrative directions
    for a single ticker.
    """
    loop = asyncio.get_event_loop()
    symbol = symbol.upper()

    try:
        info = await loop.run_in_executor(None, lambda: get_ticker_info(symbol))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    embedding = await loop.run_in_executor(None, lambda: embed_text(info.embed_text))
    nearest = await loop.run_in_executor(
        None, lambda: vector_store.query_nearest(embedding, n_results=n_results)
    )

    narratives = []
    for narrative, distance, _emb in nearest:
        narratives.append(_narrative_to_dict(narrative, distance))

    return {
        "ticker": symbol,
        "company_name": info.name,
        "sector": info.sector,
        "industry": info.industry,
        "market_cap": info.market_cap,
        "embed_text": info.embed_text,
        "narratives": narratives,
    }


# ---------------------------------------------------------------------------
# DELETE /api/tickers/cache — evict yfinance cache
# ---------------------------------------------------------------------------

@router.delete("/cache")
def clear_ticker_cache(symbol: Optional[str] = None):
    """
    Evict the in-process yfinance company info cache.
    Pass ?symbol=JPM to evict one entry, or omit to clear everything.
    """
    invalidate_cache(symbol)
    return {
        "cleared": symbol.upper() if symbol else "all",
    }
