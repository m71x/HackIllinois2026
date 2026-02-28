# ============================================================
# NEXUS Model Risk Engine — Full Startup Script (Windows)
# Starts backend, ingests real financial news, serves frontend
# Usage: .\start.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  NEXUS - Real-World Model Risk Engine" -ForegroundColor Cyan
Write-Host "  Starting full stack with real data..." -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# ----------------------------------------------------------
# 1. Check virtual environment
# ----------------------------------------------------------
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "No .venv found. Creating one..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create .venv. Make sure Python is installed." -ForegroundColor Red
        exit 1
    }
}
Write-Host "[OK] Virtual environment found" -ForegroundColor Green

# ----------------------------------------------------------
# 2. Install dependencies
# ----------------------------------------------------------
Write-Host "[..] Checking dependencies..." -ForegroundColor Yellow
$ErrorActionPreference = "Continue"
& .venv\Scripts\pip.exe install -q -r backend\requirements.txt 2>&1 | Out-Null
$ErrorActionPreference = "Stop"
Write-Host "[OK] Dependencies ready" -ForegroundColor Green

# ----------------------------------------------------------
# 3. Kill any existing server on port 8000
# ----------------------------------------------------------
$existing = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($existing) {
    Write-Host "[..] Stopping existing server on port 8000..." -ForegroundColor Yellow
    foreach ($procId in $existing) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# ----------------------------------------------------------
# 4. Clear old ChromaDB data (fresh start)
# ----------------------------------------------------------
Write-Host "[..] Clearing old ChromaDB data..." -ForegroundColor Yellow
if (Test-Path "backend\chroma_db") {
    try {
        Remove-Item -Recurse -Force "backend\chroma_db" -ErrorAction Stop
    }
    catch {
        Write-Host "    Could not clear ChromaDB (file in use - will reuse existing)" -ForegroundColor Yellow
    }
}

# ----------------------------------------------------------
# 5. Start FastAPI backend
# ----------------------------------------------------------
Write-Host "[..] Starting backend server..." -ForegroundColor Cyan
$serverJob = Start-Process -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory "$ROOT\backend" `
    -PassThru -NoNewWindow

# Wait for server to be ready (two-phase: TCP port check, then HTTP health)
Write-Host -NoNewline "    Waiting for server"
$serverReady = $false
for ($i = 1; $i -le 60; $i++) {
    # Phase 1: Check if the TCP port is open
    $tcp = $null
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", 8000)
        $tcp.Close()
    }
    catch {
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 1
        continue
    }

    # Phase 2: Port is open - now confirm HTTP health endpoint responds
    try {
        $wc = New-Object System.Net.WebClient
        $resp = $wc.DownloadString("http://127.0.0.1:8000/health")
        if ($resp -match '"status"') {
            Write-Host " [OK]" -ForegroundColor Green
            $serverReady = $true
            break
        }
    }
    catch {
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 1
    }
}
if (-not $serverReady) {
    Write-Host " [FAILED]" -ForegroundColor Red
    Write-Host "    Server process may still be starting. Check if http://localhost:8000/health responds manually." -ForegroundColor Yellow
    Write-Host "    Not killing the server - it may just need more time." -ForegroundColor Yellow
}

# ----------------------------------------------------------
# 6. Ingest seed stories (if file exists)
# ----------------------------------------------------------
if (Test-Path "$ROOT\seed_stories.json") {
    Write-Host ""
    Write-Host "[..] Ingesting financial news stories..." -ForegroundColor Cyan
    try {
        $body = Get-Content "$ROOT\seed_stories.json" -Raw
        $result = Invoke-RestMethod -Uri "http://localhost:8000/api/ingest/batch" `
            -Method POST -ContentType "application/json" -Body $body -TimeoutSec 120
        $created = ($result.results | Where-Object { $_.action -eq "created" }).Count
        $updated = ($result.results | Where-Object { $_.action -eq "updated" }).Count
        Write-Host "    Processed $($result.processed) stories: $created created, $updated updated ($($result.duration_seconds)s)" -ForegroundColor Green
    }
    catch {
        Write-Host "    Ingestion failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# ----------------------------------------------------------
# 7. Verify
# ----------------------------------------------------------
Write-Host ""
Write-Host "[..] Verifying..." -ForegroundColor Cyan
try {
    $risk = Invoke-RestMethod -Uri "http://localhost:8000/api/risk" -TimeoutSec 10
    Write-Host "    Risk Index: $($risk.model_risk_index), Narratives: $($risk.narrative_count)" -ForegroundColor Green
}
catch {
    Write-Host "    Could not verify (API may still be warming up)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  NEXUS is running!" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "  Dashboard:  http://localhost:8000/" -ForegroundColor White
Write-Host "  API Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Health:     http://localhost:8000/health" -ForegroundColor White
Write-Host "" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Green

# Keep running until Ctrl+C
try {
    Wait-Process -Id $serverJob.Id
}
catch {
    Stop-Process -Id $serverJob.Id -Force -ErrorAction SilentlyContinue
}
