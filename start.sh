#!/usr/bin/env bash
# ============================================================
# NEXUS Model Risk Engine — Full Startup Script
# Starts backend, ingests real financial news, serves frontend
# Usage: bash start.sh
# ============================================================

set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo "╔══════════════════════════════════════════════════╗"
echo "║  NEXUS — Real-World Model Risk Engine            ║"
echo "║  Starting full stack with real data...            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ----------------------------------------------------------
# 1. Check virtual environment
# ----------------------------------------------------------
if [ ! -f ".venv/bin/python" ]; then
  echo "❌ No .venv found. Run: python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi
echo "✅ Virtual environment found"

# ----------------------------------------------------------
# 2. Install dependencies (if missing)
# ----------------------------------------------------------
echo "📦 Checking dependencies..."
.venv/bin/pip install -q newsapi-python tweepy sentence-transformers feedparser 2>/dev/null
echo "✅ Dependencies ready"

# ----------------------------------------------------------
# 3. Kill any existing server on port 8000
# ----------------------------------------------------------
if lsof -ti:8000 >/dev/null 2>&1; then
  echo "🔄 Stopping existing server on port 8000..."
  lsof -ti:8000 | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# ----------------------------------------------------------
# 4. Clear old ChromaDB data (fresh start)
# ----------------------------------------------------------
echo "🗑️  Clearing old ChromaDB data..."
rm -rf backend/chroma_db

# ----------------------------------------------------------
# 5. Start FastAPI backend
# ----------------------------------------------------------
echo "🚀 Starting backend server..."
cd backend
../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
cd "$ROOT"

# Wait for server to be ready
echo -n "   Waiting for server"
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo " ✅"
    break
  fi
  echo -n "."
  sleep 1
  if [ $i -eq 30 ]; then
    echo " ❌ Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
  fi
done

# ----------------------------------------------------------
# 6. Check Modal GPU embedder status
# ----------------------------------------------------------
echo ""
MODAL_JSON=$(curl -s http://localhost:8000/api/modal/status)
MODAL_CONNECTED=$(echo "$MODAL_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('connected', False))" 2>/dev/null || echo "False")
MODAL_BACKEND=$(echo "$MODAL_JSON"  | python3 -c "import sys,json; print(json.load(sys.stdin).get('backend', 'unknown'))" 2>/dev/null || echo "unknown")

if [ "$MODAL_CONNECTED" = "True" ]; then
  echo "🟢 Modal GPU Embedder  ✓  CONNECTED  (backend: $MODAL_BACKEND)"
else
  echo "🟡 Modal GPU Embedder  ✗  OFFLINE    (backend: $MODAL_BACKEND — using local CPU)"
fi

# ----------------------------------------------------------
# 7. Bulk ingest — 90+ RSS feeds, 72 h lookback, 1000+ stories
# ----------------------------------------------------------
echo ""
echo "📰 Bulk ingesting financial news (90+ feeds, 72 h lookback)..."
echo "   ℹ️  The server auto-started bulk ingest on boot."
echo "      Triggering an additional pass now for maximum coverage..."

curl -s -X POST http://localhost:8000/api/ingest/bulk >/dev/null 2>&1 || true
echo "   🔄 Bulk ingest running in background — watch server logs for progress"
echo ""

# Brief wait then show initial pipeline stats
sleep 5
STATS=$(curl -s http://localhost:8000/api/pipeline/stats 2>/dev/null || echo "{}")
TOTAL=$(echo "$STATS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('narratives',{}).get('total',0))" 2>/dev/null || echo "0")
INGESTED=$(echo "$STATS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pipeline',{}).get('stories_ingested',0))" 2>/dev/null || echo "0")
echo "   📊 Early stats: $INGESTED stories ingested → $TOTAL narrative clusters so far"
echo "   (ingestion continues in background; refresh the dashboard in ~30 s)"

# ----------------------------------------------------------
# 8. Verify risk index
# ----------------------------------------------------------
echo ""
echo "📊 Verifying risk index..."
RISK=$(curl -s http://localhost:8000/api/risk | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Risk Index: {d[\"model_risk_index\"]}, Narratives: {d[\"narrative_count\"]}')" 2>/dev/null || echo "warming up...")
echo "   $RISK"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅ NEXUS is running!                             ║"
echo "║                                                   ║"
echo "║  🌐 Dashboard:  http://localhost:8000/             ║"
echo "║  📡 API Docs:   http://localhost:8000/docs         ║"
echo "║  🔑 Health:     http://localhost:8000/health       ║"
echo "║                                                   ║"
echo "║  Press Ctrl+C to stop the server                  ║"
echo "╚══════════════════════════════════════════════════╝"

# Keep running until Ctrl+C
wait $SERVER_PID
