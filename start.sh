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
.venv/bin/pip install -q newsapi-python tweepy sentence-transformers 2>/dev/null
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
# 6. Ingest real financial news stories
# ----------------------------------------------------------
echo ""
echo "📰 Ingesting financial news stories..."

curl -s -X POST http://localhost:8000/api/ingest/batch \
  -H "Content-Type: application/json" \
  -d @"$ROOT/seed_stories.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
created = sum(1 for r in d['results'] if r['action'] == 'created')
updated = sum(1 for r in d['results'] if r['action'] == 'updated')
print(f'   Processed {d[\"processed\"]} stories: {created} created, {updated} updated ({d[\"duration_seconds\"]}s)')
"

# ----------------------------------------------------------
# 7. Verify
# ----------------------------------------------------------
echo ""
echo "📊 Verifying..."
RISK=$(curl -s http://localhost:8000/api/risk | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Risk Index: {d[\"model_risk_index\"]}, Narratives: {d[\"narrative_count\"]}')")
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
