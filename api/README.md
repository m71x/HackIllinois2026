# API Contracts

This folder defines the interfaces between all three teams.
Read only the file(s) relevant to your boundary.

```
┌─────────────┐   model_contract.py   ┌─────────────┐   rest_api.md   ┌──────────────┐
│    model/   │ ───────────────────► │   backend/  │ ──────────────► │  frontend/   │
│  modal_app  │                       │  FastAPI    │                  │  HTML/JS     │
└─────────────┘                       └─────────────┘                  └──────────────┘
```

| File | Who reads it | What it defines |
|---|---|---|
| `model_contract.py` | **model team** (must implement), **backend team** (calls it) | Python ABCs for `LLM` and `Embedder` Modal classes |
| `rest_api.md` | **backend team** (must implement), **frontend team** (calls it) | Every REST endpoint: method, path, request, response |
| `sse_events.md` | **backend team** (must emit), **frontend team** (must handle) | SSE event stream format |

## Base URL

```
http://localhost:8000
```

## Shared Data Shape — NarrativeDirection

Both the backend API responses and the frontend UI are built around this object.

```json
{
  "id": "uuid-string",
  "name": "Energy supply shock",
  "description": "Ongoing deterioration of Russian gas exports disrupting European energy markets",
  "event_count": 41,
  "current_surprise": 0.81,
  "current_impact": 0.74,
  "model_risk": 0.77,
  "last_updated": 1709123456.789,
  "is_active": true,
  "surprise_trend": "rising",
  "impact_trend": "stable",
  "recent_headlines": [
    "Russia halts gas transit through Ukraine",
    "Germany activates emergency energy rationing"
  ]
}
```

`model_risk = sqrt(surprise × impact)` — computed on read, not stored.
`is_active = last_updated within past 48 hours`
`surprise_trend / impact_trend` = `"rising"` | `"falling"` | `"stable"` | `null`
