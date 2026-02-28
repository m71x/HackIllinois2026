# SSE Event Stream

**Endpoint:** `GET /api/events/stream`
**Implemented by:** `backend/api/routes/events.py`
**Consumed by:** `frontend/dashboard.js`

The server pushes one event per story ingested by the pipeline.
The frontend uses this to update the live feed and trigger dashboard refreshes
without polling.

---

## Connecting (JavaScript)

```javascript
const es = new EventSource("http://localhost:8000/api/events/stream");

es.addEventListener("message", (e) => {
  const data = JSON.parse(e.data);
  // handle by type
  if (data.type === "connected") { /* handshake */ }
  if (data.type === "ingest")    { /* new story processed */ }
});

// EventSource auto-reconnects on drop — no manual retry needed
```

---

## Event Types

### `connected` — sent once on connection open

```json
{
  "type":      "connected",
  "timestamp": 1709123456.789
}
```

### `ingest` — sent after every story processed by the pipeline

```json
{
  "type":      "ingest",
  "timestamp": 1709123456.789,
  "result": {
    "action":               "created" | "updated",
    "narrative_id":         "uuid",
    "narrative_name":       "European energy supply shock",
    "best_distance":        0.18,
    "threshold":            0.40,
    "current_surprise":     0.82,
    "current_impact":       0.78,
    "model_risk":           0.80,
    "narrative_event_count": 41
  }
}
```

**`action` values:**
- `"created"` — story was too different from all existing narratives; a new narrative direction was spawned
- `"updated"` — story was close enough to an existing narrative (`best_distance < threshold`); its metrics were updated

---

## Keepalive

The server sends a comment line every 30 seconds to prevent proxy disconnection:

```
: keepalive
```

This is invisible to `addEventListener` — it only prevents the connection from timing out.

---

## Recommended Frontend Behavior

On each `ingest` event:
1. Prepend a new item to the live feed list (show `action`, `narrative_name`, `model_risk`)
2. If `action === "created"`, briefly highlight the narrative table row
3. Re-fetch `GET /api/risk` and `GET /api/narratives` to refresh the dashboard

Do not re-fetch on every event if multiple events arrive in quick succession —
debounce refreshes by 2–3 seconds.
