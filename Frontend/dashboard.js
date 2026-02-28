/* ============================================================
   NEXUS MODEL RISK ENGINE — dashboard.js
   Premium UI Logic | API Client | Mock Data Engine
   ============================================================ */

"use strict";

const API = "http://localhost:8000/api";
const AUTO_REFRESH_MS = 10_000;
const FEED_MAX_ITEMS = 50;

let riskChart = null;
let modalChart = null;
let sseSource = null;
let isMockMode = false;

// ============================================================
// SECTION 1 — API / Mock Data Engine
// ============================================================

const MOCK_DATA = {
  narratives: [
    { id: "1", name: "Global Energy Shock", description: "Geopolitical tensions causing massive disruption in LNG spot markets and European gas supply.", current_surprise: 0.88, current_impact: 0.92, model_risk: 0.89, event_count: 142, surprise_trend: "rising", last_updated: Date.now() / 1000 - 120 },
    { id: "2", name: "Regional Bank Contagion", description: "Rapid deposit flight in mid-tier regional banks triggering liquidity hoarding.", current_surprise: 0.72, current_impact: 0.65, model_risk: 0.68, event_count: 85, surprise_trend: "stable", last_updated: Date.now() / 1000 - 1500 },
    { id: "3", name: "Semiconductor Export Bans", description: "Bilateral trade restrictions halting flow of advanced chips and manufacturing equipment.", current_surprise: 0.45, current_impact: 0.81, model_risk: 0.60, event_count: 53, surprise_trend: "falling", last_updated: Date.now() / 1000 - 4500 },
    { id: "4", name: "AI Labor Displacement Shock", description: "Widespread automation affecting clerical labor unexpectedly fast, driving policy panic.", current_surprise: 0.91, current_impact: 0.45, model_risk: 0.64, event_count: 310, surprise_trend: "rising", last_updated: Date.now() / 1000 - 86400 },
    { id: "5", name: "Sovereign Debt Downgrade", description: "Major G7 economy facing credit downgrade warnings amid massive fiscal deficits.", current_surprise: 0.35, current_impact: 0.95, model_risk: 0.57, event_count: 22, surprise_trend: "stable", last_updated: Date.now() / 1000 - 172800 }
  ],
  riskIndex: 0.89,
  history: Array.from({ length: 24 }, (_, i) => ({ timestamp: Date.now() / 1000 - (24 - i) * 3600, model_risk_index: 0.5 + Math.random() * 0.4 })),
  pipeline: { stories_ingested: 14092, narratives_created: 18, narratives_updated: 241, queue_size: 14, errors: 0 }
};

async function fetchJSON(path) {
  try {
    const res = await fetch(`${API}${path}`);
    if (!res.ok) throw new Error("API failed");
    return res.json();
  } catch (e) {
    return handleMockMode(path);
  }
}

async function postJSON(path, body) {
  try {
    const res = await fetch(`${API}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error("API failed");
    return res.json();
  } catch (e) {
    isMockMode = true;
    return {
      action: "created",
      narrative_name: body.headline?.slice(0, 30) + " Trend" || "Unknown Output",
      model_risk: 0.77, current_surprise: 0.81, current_impact: 0.65, best_distance: 0.45
    };
  }
}

function handleMockMode(path) {
  if (!isMockMode) {
    isMockMode = true;
    document.getElementById("mock-badge").style.display = "flex";
    document.getElementById("conn-status").textContent = "MOCK";
    initMockSSE(); // Start fake real-time events
  }

  if (path.includes("/risk/history")) return { history: MOCK_DATA.history };
  if (path.includes("/risk")) return { model_risk_index: MOCK_DATA.riskIndex };
  if (path.includes("/narratives?")) return { narratives: MOCK_DATA.narratives };
  if (path.includes("/narratives/")) return { ...MOCK_DATA.narratives[0], surprise_series: MOCK_DATA.history.map(h => ({ timestamp: h.timestamp, value: h.model_risk_index - 0.2 })), impact_series: MOCK_DATA.history.map(h => ({ timestamp: h.timestamp, value: h.model_risk_index + 0.1 })), model_risk_series: MOCK_DATA.history.map(h => ({ timestamp: h.timestamp, value: h.model_risk_index })), recent_headlines: ["US blocks LNG exports to EU", "Gas futures collapse 40%", "emergency rationing invoked in Berlin", "CNBC market alert: Energy sector halts trading"] };
  if (path.includes("/pipeline/stats")) return { pipeline: MOCK_DATA.pipeline, narratives: { total: 42, active: 5 } };
  return {};
}

// ============================================================
// SECTION 2 — UI Renderers
// ============================================================

function riskColor(val) {
  if (val == null) return "var(--text-muted)";
  if (val < 0.33) return "var(--risk-low)";
  if (val < 0.66) return "var(--risk-medium)";
  return "var(--risk-high)";
}

function renderGauge(val) {
  const vEl = document.getElementById("risk-value");
  const lEl = document.getElementById("risk-label");
  const fill = document.getElementById("risk-fill");

  if (val == null) return;

  vEl.textContent = val.toFixed(2);
  fill.style.width = `${Math.min(val * 100, 100)}%`;

  if (val < 0.33) {
    lEl.textContent = "STABLE"; vEl.style.color = "var(--risk-low)"; fill.className = "risk-fill low";
  } else if (val < 0.66) {
    lEl.textContent = "ELEVATED"; vEl.style.color = "var(--risk-medium)"; fill.className = "risk-fill medium";
  } else {
    lEl.textContent = "CRITICAL LIMIT"; vEl.style.color = "var(--risk-high)"; fill.className = "risk-fill high";
  }
}

function renderPipelineStats(data) {
  const p = data.pipeline ?? {};
  const n = data.narratives ?? {};

  const animateNum = (id, val) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.textContent === String(val)) return;
    el.textContent = val;
    el.style.transform = "scale(1.2)";
    el.style.color = "#fff";
    setTimeout(() => { el.style.transform = "scale(1)"; el.style.color = ""; }, 300);
  };

  animateNum("stat-ingested", p.stories_ingested ?? 0);
  animateNum("stat-created", p.narratives_created ?? 0);
  animateNum("stat-updated", p.narratives_updated ?? 0);
  animateNum("stat-active", n.active ?? 0);
  animateNum("stat-queue", p.queue_size ?? 0);
  animateNum("stat-errors", p.errors ?? 0);
}

// ============================================================
// SECTION 3 — Narrative Table
// ============================================================

function timeAgo(unixTs) {
  const diff = Date.now() / 1000 - unixTs;
  if (diff < 60) return `Just now`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

function trendIcon(trend) {
  if (trend === "rising") return `<i class="ph-bold ph-trend-up text-accent-red"></i>`;
  if (trend === "falling") return `<i class="ph-bold ph-trend-down text-accent-cyan"></i>`;
  return `<i class="ph-bold ph-minus text-text-muted"></i>`;
}

async function refreshNarratives() {
  const sortBy = document.getElementById("sort-select").value;
  const data = await fetchJSON(`/narratives?active_only=true&sort_by=${sortBy}&limit=50`);
  const narratives = data.narratives ?? [];

  const tbody = document.getElementById("narratives-tbody");
  const emptyEl = document.getElementById("narratives-empty");

  if (narratives.length === 0) {
    emptyEl.classList.remove("hidden");
    tbody.innerHTML = "";
    return;
  }

  emptyEl.classList.add("hidden");

  // Client-side sort if mock mode
  if (isMockMode) {
    narratives.sort((a, b) => {
      if (sortBy === "risk") return b.model_risk - a.model_risk;
      if (sortBy === "events") return b.event_count - a.event_count;
      return b.last_updated - a.last_updated;
    });
  }

  tbody.innerHTML = narratives.map(n => `
    <tr data-id="${n.id}">
      <td><strong>${escapeHtml(n.name)}</strong></td>
      <td class="desc" title="${escapeHtml(n.description)}">${escapeHtml(n.description)}</td>
      <td class="num-col" style="color:${riskColor(n.current_surprise)}">${(n.current_surprise ?? 0).toFixed(2)}</td>
      <td class="num-col" style="color:${riskColor(n.current_impact)}">${(n.current_impact ?? 0).toFixed(2)}</td>
      <td class="num-col" style="color:${riskColor(n.model_risk)};font-weight:700">${(n.model_risk ?? 0).toFixed(2)}</td>
      <td class="num-col">${n.event_count ?? 0}</td>
      <td style="text-align:center">${trendIcon(n.surprise_trend)}</td>
      <td class="right-align data-number text-text-muted">${timeAgo(n.last_updated)}</td>
    </tr>
  `).join("");

  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => openNarrativeModal(tr.dataset.id));
  });
}

// ============================================================
// SECTION 4 — Charts
// ============================================================

async function updateRiskChart() {
  const data = await fetchJSON("/risk/history?window=24&resolution=100");
  const history = data.history ?? [];
  const ctx = document.getElementById("risk-chart").getContext("2d");

  const labels = history.map(p => new Date(p.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
  const values = history.map(p => p.model_risk_index);

  if (!riskChart) {
    // Create animated cyber gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, "rgba(239, 68, 68, 0.4)"); // high risk red
    gradient.addColorStop(0.5, "rgba(245, 158, 11, 0.1)");
    gradient.addColorStop(1, "rgba(16, 185, 129, 0.0)");

    riskChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: "Risk Index",
          data: values,
          borderColor: "#00f0ff", // Neon cyan line
          backgroundColor: gradient,
          borderWidth: 2,
          pointBackgroundColor: "#050811",
          pointBorderColor: "#00f0ff",
          pointBorderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 6,
          fill: true,
          tension: 0.4 // Smooth curves
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#8b949e", font: { family: "'JetBrains Mono'" } } },
          x: { grid: { display: false }, ticks: { color: "#8b949e", maxTicksLimit: 6, font: { family: "'JetBrains Mono'" } } }
        },
        plugins: { legend: { display: false }, tooltip: { backgroundColor: "rgba(16,22,38,0.9)", titleFont: { family: "'Outfit'" }, bodyFont: { family: "'JetBrains Mono'" }, borderColor: "rgba(0,240,255,0.3)", borderWidth: 1 } }
      }
    });
  } else {
    riskChart.data.labels = labels;
    riskChart.data.datasets[0].data = values;
    riskChart.update("none");
  }
}

// ============================================================
// SECTION 5 — SSE Live Feed & Mock Feed
// ============================================================

function appendFeedItem(r) {
  const feedEl = document.getElementById("live-feed");
  const li = document.createElement("li");
  li.className = `feed-item`;

  const icon = r.action === "created" ? "ph-sparkle text-accent-cyan" : "ph-arrows-clockwise text-accent-blue";

  li.innerHTML = `
    <span class="badge ${r.action}">${r.action}</span>
    <span class="feed-name"><i class="ph ${icon}"></i>  ${escapeHtml(r.narrative_name)}</span>
    <span class="feed-risk" style="color:${riskColor(r.model_risk)}">${(r.model_risk ?? 0).toFixed(2)}</span>
    <span class="feed-time">now</span>
  `;

  feedEl.insertBefore(li, feedEl.firstChild);
  while (feedEl.children.length > FEED_MAX_ITEMS) feedEl.removeChild(feedEl.lastChild);
}

function initSSE() {
  const statusEl = document.getElementById("feed-status");
  sseSource = new EventSource(`${API}/events/stream`);

  sseSource.addEventListener("message", e => {
    const data = JSON.parse(e.data);
    if (data.type === "connected") { statusEl.textContent = "Live Stream Active"; statusEl.style.color = "var(--risk-low)"; return; }
    if (data.type === "ingest") { appendFeedItem(data.result); refreshDashboard(); }
  });

  sseSource.addEventListener("error", () => {
    statusEl.innerHTML = `<i class="ph ph-warning-circle"></i> Disconnected`;
    statusEl.style.color = "var(--risk-high)";
  });
}

function initMockSSE() {
  document.getElementById("feed-status").innerHTML = `<i class="ph-fill ph-flask"></i> Mock Stream Active`;
  setInterval(() => {
    const headlines = ["Fed Signals Cuts", "Oil SPIKES on supply fear", "EU drafts new AI Rules", "Tech stocks plummet"];
    const action = Math.random() > 0.7 ? "created" : "updated";
    const name = MOCK_DATA.narratives[Math.floor(Math.random() * MOCK_DATA.narratives.length)].name;
    appendFeedItem({ action, narrative_name: name, model_risk: 0.4 + Math.random() * 0.5 });
    MOCK_DATA.pipeline.stories_ingested++;
    if (action === "created") MOCK_DATA.pipeline.narratives_created++;
    else MOCK_DATA.pipeline.narratives_updated++;
    renderPipelineStats({ pipeline: MOCK_DATA.pipeline, narratives: { active: 5 } });
  }, 4500);
}

// ============================================================
// SECTION 6 — Rest of Logic (Search, Modal, Ingest)
// ============================================================

document.getElementById("search-btn").addEventListener("click", runSearch);
document.getElementById("search-input").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });

async function runSearch() {
  const query = document.getElementById("search-input").value.trim().toLowerCase();
  const resDiv = document.getElementById("search-results");
  if (!query) return;

  resDiv.innerHTML = `<div class="spinner"></div>`;
  const data = await postJSON("/narratives/search", { query, n_results: 5 });
  
  let results = [];
  if (isMockMode) {
    // Filter mock data by text match
    const matches = MOCK_DATA.narratives.filter(n => 
      n.name.toLowerCase().includes(query) || 
      n.description.toLowerCase().includes(query)
    );
    
    // Sort relevance arbitrarily for mockup
    results = matches.map((n, i) => ({ 
      narrative: n, 
      similarity: 0.95 - (i * 0.05) 
    }));
    
    // Fallback if no exact text match (just return top 2 to simulate fuzzy finding)
    if (results.length === 0) {
      results = MOCK_DATA.narratives.slice(0, 2).map((n, i) => ({ 
        narrative: n, 
        similarity: 0.65 - (i * 0.10) 
      }));
    }
  } else {
    results = data.results;
  }

  resDiv.innerHTML = results.map((r, i) => `
    <div class="search-result" onclick="openNarrativeModal('${r.narrative.id}')">
      <strong>${i + 1}. ${escapeHtml(r.narrative.name)}</strong>
      <div style="display:flex; justify-content:space-between; margin-top:0.4rem;">
        <span class="distance">Match: <span class="text-accent-cyan">${((r.similarity) * 100).toFixed(0)}%</span></span>
        <span class="data-number" style="color:${riskColor(r.narrative.model_risk)}">Rsk: ${(r.narrative.model_risk).toFixed(2)}</span>
      </div>
    </div>
  `).join("");
}

document.getElementById("ingest-btn").addEventListener("click", async () => {
  const btn = document.getElementById("ingest-btn");
  const headline = document.getElementById("ingest-headline").value.trim();
  const resultEl = document.getElementById("ingest-result");

  if (!headline) return;

  btn.innerHTML = `<i class="ph ph-spinner ph-spin"></i> Processing...`;
  const data = await postJSON("/ingest", { headline, body: document.getElementById("ingest-body").value });

  resultEl.innerHTML = `
    <span class="badge ${data.action}">${data.action}</span>
    <span class="text-accent-cyan fw-bold">${escapeHtml(data.narrative_name)}</span>  
    <span class="data-number">R:${(data.model_risk).toFixed(2)}</span>
  `;
  document.getElementById("ingest-headline").value = ""; document.getElementById("ingest-body").value = "";
  btn.innerHTML = `<i class="ph-bold ph-check"></i> Done`;
  setTimeout(() => btn.innerHTML = `<i class="ph-bold ph-lightning"></i> Ingest Story`, 2000);
});

async function openNarrativeModal(id) {
  const data = await fetchJSON(`/narratives/${id}/history`);
  document.getElementById("modal-name").textContent = data.name;
  document.getElementById("modal-description").textContent = data.description;

  document.getElementById("modal-meta").innerHTML = `
    <div class="meta-card"><span class="meta-card-label">Overall Risk</span><span class="meta-card-value" style="color:${riskColor(data.model_risk)}">${data.model_risk.toFixed(2)}</span></div>
    <div class="meta-card"><span class="meta-card-label">Total Events</span><span class="meta-card-value text-accent-cyan">${data.event_count}</span></div>
    <div class="meta-card"><span class="meta-card-label">Last Updated</span><span class="meta-card-value">${timeAgo(data.last_updated)}</span></div>
  `;

  document.getElementById("modal-headlines").innerHTML = (data.recent_headlines || []).map(h => `<li>${escapeHtml(h)}</li>`).join("");

  if (modalChart) modalChart.destroy();
  const ctx = document.getElementById("modal-chart").getContext("2d");
  modalChart = new Chart(ctx, {
    type: "line", data: {
      labels: data.surprise_series.map(p => new Date(p.timestamp * 1000).toLocaleTimeString()),
      datasets: [
        { label: "Surprise", data: data.surprise_series.map(p => p.value), borderColor: "#f43f5e", tension: 0.3 },
        { label: "Impact", data: data.impact_series.map(p => p.value), borderColor: "#3b82f6", tension: 0.3 }
      ]
    }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#8b949e" } }, x: { display: false } } }
  });
  document.getElementById("narrative-modal").classList.remove("hidden");
}

document.getElementById("modal-close").addEventListener("click", () => document.getElementById("narrative-modal").classList.add("hidden"));
document.getElementById("modal-backdrop").addEventListener("click", () => document.getElementById("narrative-modal").classList.add("hidden"));
document.addEventListener("keydown", e => { if (e.key === "Escape") document.getElementById("narrative-modal").classList.add("hidden"); });
document.getElementById("sort-select").addEventListener("change", refreshNarratives);

async function refreshDashboard() {
  const [riskData, pipelineData] = await Promise.all([fetchJSON("/risk"), fetchJSON("/pipeline/stats")]);
  renderGauge(riskData.model_risk_index);
  renderPipelineStats(pipelineData);
  await refreshNarratives();
  await updateRiskChart();
  document.getElementById("last-updated").innerHTML = `<i class="ph ph-clock"></i> ${new Date().toLocaleTimeString()}`;
}

function escapeHtml(str) { return String(str || "").replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[m]); }

// ============================================================
// SECTION 7 — Equities & Global Markets
// ============================================================

const MARKETS = [
  { id: "ny", name: "New York (NYSE/NASDAQ)", timeZone: "America/New_York", openHr: 9, closeHr: 16 },
  { id: "ldn", name: "London (LSE)", timeZone: "Europe/London", openHr: 8, closeHr: 16 },
  { id: "tyo", name: "Tokyo (TSE)", timeZone: "Asia/Tokyo", openHr: 9, closeHr: 15 },
  { id: "syd", name: "Sydney (ASX)", timeZone: "Australia/Sydney", openHr: 10, closeHr: 16 }
];

const WATCHLIST = [
  { sym: "AAPL", name: "Apple Inc.", base: 185.20, vol: "52.1M" },
  { sym: "NVDA", name: "NVIDIA Corp.", base: 721.50, vol: "41.8M" },
  { sym: "MSFT", name: "Microsoft Corp.", base: 405.11, vol: "28.3M" },
  { sym: "TSLA", name: "Tesla Inc.", base: 193.81, vol: "110.2M" },
  { sym: "META", name: "Meta Platforms", base: 468.10, vol: "18.5M" },
  { sym: "GOOGL", name: "Alphabet Inc.", base: 142.30, vol: "25.6M" },
  { sym: "AMZN", name: "Amazon.com Inc.", base: 169.50, vol: "45.1M" },
  { sym: "XOM", name: "Exxon Mobil", base: 104.20, vol: "15.4M" },
  { sym: "JPM", name: "JPMorgan Chase", base: 175.40, vol: "12.1M" }
];

// Initialize stock prices with some random daily drift
WATCHLIST.forEach(s => {
  s.start = s.base * (1 + (Math.random() * 0.04 - 0.02)); // +/- 2% from base
  s.current = s.start;
});

function updateMarketClocks() {
  const container = document.getElementById("market-clocks");
  if (!container) return;

  const now = new Date();

  container.innerHTML = MARKETS.map(m => {
    // Format time in the target timezone
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: m.timeZone,
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: false, weekday: 'short', month: 'short', day: 'numeric'
    });

    // Check if market is open (crude check based on hours 9-16 roughly)
    const tzHourStr = new Intl.DateTimeFormat('en-US', { timeZone: m.timeZone, hour: 'numeric', hour12: false }).format(now);
    const tzHour = parseInt(tzHourStr, 10);
    const tzWeekday = new Intl.DateTimeFormat('en-US', { timeZone: m.timeZone, weekday: 'short' }).format(now);

    const isWeekend = tzWeekday === 'Sat' || tzWeekday === 'Sun';
    const isOpen = !isWeekend && (tzHour >= m.openHr && tzHour < m.closeHr);

    const parts = formatter.formatToParts(now);
    const tm = parts.filter(p => ["hour", "minute", "second", "literal"].includes(p.type) && p.value !== ', ').map(p => p.value).join('').trim();
    const dt = new Intl.DateTimeFormat('en-US', { timeZone: m.timeZone, weekday: 'short', month: 'short', day: 'numeric' }).format(now);

    return `
      <div class="clock-card ${isOpen ? 'open' : 'closed'}">
        <h3 class="market-name">${m.name}</h3>
        <div class="market-status ${isOpen ? 'open' : 'closed'}">
          <i class="ph-fill ${isOpen ? 'ph-check-circle' : 'ph-moon'}"></i> ${isOpen ? 'MARKET OPEN' : 'MARKET CLOSED'}
        </div>
        <div class="market-time">${tm}</div>
        <div class="market-date">${dt}</div>
      </div>
    `;
  }).join('');
}

function updateStocks() {
  const tbody = document.getElementById("stocks-tbody");
  if (!tbody) return;

  // Tick the stocks
  WATCHLIST.forEach(s => {
    // Random walk
    const change = s.current * (Math.random() * 0.004 - 0.002); // +/- 0.2% max tick
    s.current += change;
  });

  // Sort by highest volume for display, or just alphabetical. We'll stick to fixed for now.
  tbody.innerHTML = WATCHLIST.map(s => {
    const diff = s.current - s.start;
    const pct = (diff / s.start) * 100;
    const isUp = diff >= 0;
    const colorClass = isUp ? "text-green" : "text-red";
    const sign = isUp ? "+" : "";

    return `
      <tr>
        <td><strong>${s.sym}</strong></td>
        <td class="text-text-muted">${s.name}</td>
        <td class="num-col data-number" style="font-weight:700;">$${s.current.toFixed(2)}</td>
        <td class="num-col data-number ${colorClass}">${sign}${diff.toFixed(2)}</td>
        <td class="num-col data-number ${colorClass}">${sign}${pct.toFixed(2)}%</td>
        <td class="num-col data-number text-text-muted">${s.vol}</td>
      </tr>
    `;
  }).join('');
}

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
  // Tab Switching Logic
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      // Deactivate all
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      // Activate clicked
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-target");
      document.getElementById(targetId).classList.add("active");
    });
  });

  // Start Dashboard Features
  await refreshDashboard();
  initSSE();
  setInterval(refreshDashboard, AUTO_REFRESH_MS);

  // Start Equities & Clocks Features
  updateMarketClocks();
  setInterval(updateMarketClocks, 1000); // 1s tick for clocks

  updateStocks();
  setInterval(updateStocks, 2500); // 2.5s tick for stocks
});
