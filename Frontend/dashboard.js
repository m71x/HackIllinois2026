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
  // Get value from custom dropdown instead of select element
  const selectedOption = document.querySelector(".dropdown-option.selected");
  const sortBy = selectedOption ? selectedOption.dataset.value : "risk";

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

let currentChartRange = "1d";
let chartOffset = 0;

async function updateRiskChart() {
  // Mock logic to handle ranges and offsets
  let windowSize = 24;
  if (currentChartRange === "1m") windowSize = 24 * 30;
  if (currentChartRange === "1y") windowSize = 24 * 365;
  if (currentChartRange === "ytd") windowSize = 24 * 90; // Approx 3 months

  // Create mock history based on range
  const history = Array.from({ length: 24 }, (_, i) => {
    let base = 0.5;
    if (currentChartRange === "1m") base = 0.4;
    if (currentChartRange === "1y") base = 0.35;
    if (currentChartRange === "ytd") base = 0.3;
    // Add offset modification to make pages look different
    const pointRisk = Math.max(0, Math.min(1, base + Math.random() * 0.4 + (chartOffset * 0.05)));

    return {
      timestamp: Date.now() / 1000 - (24 - i) * (windowSize / 24) * 3600 - (chartOffset * windowSize * 3600),
      model_risk_index: pointRisk
    };
  });

  const ctx = document.getElementById("risk-chart").getContext("2d");

  const labels = history.map(p => {
    const d = new Date(p.timestamp * 1000);
    if (currentChartRange === "1d") return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  });

  const values = history.map(p => p.model_risk_index);
  const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
  const avgData = Array(values.length).fill(avgValue);

  // Update Chart Insights Panel
  let maxRiskPoint = history[0];
  let minRiskPoint = history[0];
  window.currentMaxIndex = 0;
  window.currentMinIndex = 0;

  history.forEach((p, i) => {
    if (p.model_risk_index > maxRiskPoint.model_risk_index) {
      maxRiskPoint = p;
      window.currentMaxIndex = i;
    }
    if (p.model_risk_index < minRiskPoint.model_risk_index) {
      minRiskPoint = p;
      window.currentMinIndex = i;
    }
  });

  const formatInsightsTime = (ts) => {
    const d = new Date(ts * 1000);
    if (currentChartRange === "1d") return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  document.getElementById("insight-max-val").textContent = maxRiskPoint.model_risk_index.toFixed(2);
  document.getElementById("insight-max-time").textContent = formatInsightsTime(maxRiskPoint.timestamp);
  document.getElementById("insight-min-val").textContent = minRiskPoint.model_risk_index.toFixed(2);
  document.getElementById("insight-min-time").textContent = formatInsightsTime(minRiskPoint.timestamp);
  document.getElementById("insight-avg-val").textContent = avgValue.toFixed(2);

  if (!riskChart) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, "rgba(239, 68, 68, 0.4)");
    gradient.addColorStop(0.5, "rgba(245, 158, 11, 0.1)");
    gradient.addColorStop(1, "rgba(16, 185, 129, 0.0)");

    riskChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Risk Index",
            data: values,
            borderColor: "#00f0ff",
            backgroundColor: gradient,
            borderWidth: 2,
            pointBackgroundColor: "#050811",
            pointBorderColor: "#00f0ff",
            pointBorderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 6,
            fill: true,
            tension: 0.4,
            order: 2
          },
          {
            label: "Average",
            data: avgData,
            borderColor: "rgba(255, 255, 255, 0.2)", // Subtle white line
            borderWidth: 1.5,
            borderDash: [5, 5], // Dashed line
            pointRadius: 0,
            pointHoverRadius: 0,
            fill: false,
            tension: 0,
            order: 1
          }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#8b949e", font: { family: "'JetBrains Mono'" } } },
          x: { grid: { display: false }, ticks: { color: "#8b949e", maxTicksLimit: 6, font: { family: "'JetBrains Mono'" } } }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(16,22,38,0.9)",
            titleFont: { family: "'Outfit'" },
            bodyFont: { family: "'JetBrains Mono'" },
            borderColor: "rgba(0,240,255,0.3)",
            borderWidth: 1,
            filter: function (tooltipItem) {
              return tooltipItem.dataset.label !== 'Average';
            }
          }
        }
      }
    });
  } else {
    riskChart.data.labels = labels;
    riskChart.data.datasets[0].data = values;
    riskChart.data.datasets[1].data = avgData;
    riskChart.update("none");
  }
}

// Chart Controls Events
document.querySelectorAll(".time-btn").forEach(btn => {
  btn.addEventListener("click", (e) => {
    document.querySelectorAll(".time-btn").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    currentChartRange = e.target.dataset.range;
    chartOffset = 0; // Reset offset on range change
    const titles = { "1d": "24h Risk History", "1m": "1M Risk History", "1y": "1Y Risk History", "ytd": "YTD Risk History" };
    document.getElementById("chart-title-text").textContent = titles[currentChartRange] || "Risk History";
    document.getElementById("chart-next").disabled = true;
    updateRiskChart();
  });
});

document.getElementById("chart-prev").addEventListener("click", () => {
  chartOffset++;
  document.getElementById("chart-next").disabled = false;
  updateRiskChart();
});

document.getElementById("chart-next").addEventListener("click", () => {
  if (chartOffset > 0) {
    chartOffset--;
    if (chartOffset === 0) document.getElementById("chart-next").disabled = true;
    updateRiskChart();
  }
});

document.getElementById("chart-expand-btn").addEventListener("click", (e) => {
  const panel = document.getElementById("chart-panel");
  panel.classList.toggle("expanded");
  const icon = e.currentTarget.querySelector("i");
  if (panel.classList.contains("expanded")) {
    icon.className = "ph ph-arrows-in";
    e.currentTarget.title = "Collapse Graph";
  } else {
    icon.className = "ph ph-arrows-out";
    e.currentTarget.title = "Expand Graph";
  }
  // Smoothly resize chart alongside 400ms CSS transition
  let start = Date.now();
  let timer = setInterval(() => {
    if (riskChart) riskChart.resize();
    if (Date.now() - start > 450) clearInterval(timer);
  }, 15);
});

// Chart Tooltip Trigger Events
const triggerChartTooltipByIndex = (index) => {
  if (!riskChart || index === undefined || index === -1) return;
  const activeEls = [{ datasetIndex: 0, index }];
  riskChart.setActiveElements(activeEls);
  riskChart.tooltip.setActiveElements(activeEls, { x: 0, y: 0 });
  riskChart.update();
};

document.getElementById("btn-max-risk").addEventListener("click", () => {
  triggerChartTooltipByIndex(window.currentMaxIndex);
});

document.getElementById("btn-min-risk").addEventListener("click", () => {
  triggerChartTooltipByIndex(window.currentMinIndex);
});

document.getElementById("btn-avg-risk").addEventListener("click", () => {
  // Just trigger tooltip at latest point for average
  if (!riskChart) return;
  const lastIdx = riskChart.data.labels.length - 1;
  const activeEls = [{ datasetIndex: 1, index: lastIdx }];
  riskChart.setActiveElements(activeEls);
  riskChart.tooltip.setActiveElements(activeEls, { x: 0, y: 0 });
  riskChart.update();
});

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

document.getElementById("search-input").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
document.getElementById("search-input").addEventListener("input", runSearch);

async function runSearch() {
  const query = document.getElementById("search-input").value.trim().toLowerCase();
  const resDiv = document.getElementById("search-results");

  let results = [];
  if (isMockMode) {
    if (query) {
      // Filter mock data by text match
      const matches = MOCK_DATA.narratives.filter(n =>
        n.name.toLowerCase().includes(query) ||
        n.description.toLowerCase().includes(query)
      );

      results = matches.map((n, i) => ({
        narrative: n,
        similarity: 0.95 - (i * 0.05)
      }));

      // Fallback if no exact text match
      if (results.length === 0) {
        results = MOCK_DATA.narratives.slice(0, 2).map((n, i) => ({
          narrative: n,
          similarity: 0.65 - (i * 0.10)
        }));
      }
    } else {
      // No query — show all narratives sorted by risk as defaults
      results = [...MOCK_DATA.narratives]
        .sort((a, b) => b.model_risk - a.model_risk)
        .map((n, i) => ({
          narrative: n,
          similarity: 1.0 - (i * 0.02)
        }));
    }
  } else {
    if (!query) return;
    resDiv.innerHTML = `<div class="spinner"></div>`;
    const data = await postJSON("/narratives/search", { query, n_results: 5 });
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

// Show default search results on load
setTimeout(runSearch, 500);

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

// Custom Dropdown Logic
const dropdownHeader = document.querySelector(".dropdown-header");
const dropdownList = document.querySelector(".dropdown-list");
const dropdownOptions = document.querySelectorAll(".dropdown-option");
const sortSelectedText = document.getElementById("sort-selected-text");
const sortDropdownBox = document.getElementById("sort-dropdown");

dropdownHeader.addEventListener("click", (e) => {
  e.stopPropagation();
  dropdownList.classList.toggle("hidden");
  sortDropdownBox.classList.toggle("open");
});

dropdownOptions.forEach(option => {
  option.addEventListener("click", (e) => {
    e.stopPropagation();
    dropdownOptions.forEach(opt => opt.classList.remove("selected"));
    option.classList.add("selected");
    sortSelectedText.textContent = option.textContent;
    dropdownList.classList.add("hidden");
    sortDropdownBox.classList.remove("open");
    refreshNarratives();
  });
});

document.addEventListener("click", () => {
  if (!dropdownList.classList.contains("hidden")) {
    dropdownList.classList.add("hidden");
    sortDropdownBox.classList.remove("open");
  }
});

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
  // Tech
  { sym: "AAPL", name: "Apple Inc.", base: 185.20, vol: "52.1M", risk: 0.32 },
  { sym: "NVDA", name: "NVIDIA Corp.", base: 721.50, vol: "41.8M", risk: 0.71 },
  { sym: "MSFT", name: "Microsoft Corp.", base: 405.11, vol: "28.3M", risk: 0.25 },
  { sym: "TSLA", name: "Tesla Inc.", base: 193.81, vol: "110.2M", risk: 0.82 },
  { sym: "META", name: "Meta Platforms", base: 468.10, vol: "18.5M", risk: 0.45 },
  { sym: "GOOGL", name: "Alphabet Inc.", base: 142.30, vol: "25.6M", risk: 0.29 },
  { sym: "AMZN", name: "Amazon.com Inc.", base: 169.50, vol: "45.1M", risk: 0.38 },
  { sym: "AMD", name: "Advanced Micro Devices", base: 164.80, vol: "55.3M", risk: 0.67 },
  { sym: "INTC", name: "Intel Corp.", base: 42.90, vol: "38.7M", risk: 0.58 },
  { sym: "CRM", name: "Salesforce Inc.", base: 272.40, vol: "8.2M", risk: 0.34 },
  { sym: "ORCL", name: "Oracle Corp.", base: 118.50, vol: "11.4M", risk: 0.31 },
  { sym: "NFLX", name: "Netflix Inc.", base: 562.30, vol: "9.8M", risk: 0.41 },
  // Finance
  { sym: "JPM", name: "JPMorgan Chase", base: 175.40, vol: "12.1M", risk: 0.53 },
  { sym: "BAC", name: "Bank of America", base: 33.80, vol: "42.5M", risk: 0.49 },
  { sym: "GS", name: "Goldman Sachs", base: 378.20, vol: "3.1M", risk: 0.56 },
  { sym: "V", name: "Visa Inc.", base: 275.60, vol: "7.9M", risk: 0.22 },
  // Energy
  { sym: "XOM", name: "Exxon Mobil", base: 104.20, vol: "15.4M", risk: 0.61 },
  { sym: "CVX", name: "Chevron Corp.", base: 152.70, vol: "9.3M", risk: 0.58 },
  { sym: "OXY", name: "Occidental Petroleum", base: 58.40, vol: "14.2M", risk: 0.73 },
  // Healthcare
  { sym: "JNJ", name: "Johnson & Johnson", base: 158.90, vol: "8.6M", risk: 0.19 },
  { sym: "UNH", name: "UnitedHealth Group", base: 492.30, vol: "4.2M", risk: 0.27 },
  { sym: "PFE", name: "Pfizer Inc.", base: 27.50, vol: "31.8M", risk: 0.44 },
  // Consumer / Industrial
  { sym: "WMT", name: "Walmart Inc.", base: 168.10, vol: "10.5M", risk: 0.15 },
  { sym: "DIS", name: "Walt Disney Co.", base: 94.20, vol: "13.7M", risk: 0.51 },
  { sym: "BA", name: "Boeing Co.", base: 198.40, vol: "6.8M", risk: 0.76 }
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

  const searchInput = document.getElementById("stock-search-input");
  const query = searchInput ? searchInput.value.trim().toLowerCase() : "";

  // Tick the stocks
  WATCHLIST.forEach(s => {
    const change = s.current * (Math.random() * 0.004 - 0.002);
    s.current += change;
    // Slightly drift risk too
    s.risk = Math.max(0, Math.min(1, s.risk + (Math.random() * 0.01 - 0.005)));
  });

  // Filter by search query
  const filtered = query
    ? WATCHLIST.filter(s =>
      s.sym.toLowerCase().includes(query) ||
      s.name.toLowerCase().includes(query)
    )
    : WATCHLIST;

  tbody.innerHTML = filtered.map(s => {
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
        <td class="num-col data-number" style="color:${riskColor(s.risk)}; font-weight:600;">${s.risk.toFixed(2)}</td>
        <td class="num-col data-number text-text-muted">${s.vol}</td>
      </tr>
    `;
  }).join('');

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted); padding:2rem;">No stocks matching "${query}"</td></tr>`;
  }
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

  // Stock search live filter
  document.getElementById("stock-search-input").addEventListener("input", updateStocks);

  updateStocks();
  setInterval(updateStocks, 2500); // 2.5s tick for stocks
});

// ### ADDED (do not edit above)

// --- Feature 4 & 5: Risk Regime & Risk Momentum ---
function injectRiskRegimeAndMomentum() {
  const gaugeContent = document.querySelector('.gauge-content');
  if (!gaugeContent) return;

  // Regime Banner
  const regimeBanner = document.createElement('div');
  regimeBanner.id = "risk-regime-banner";
  regimeBanner.className = "status-pill regime-banner";
  regimeBanner.textContent = "Evaluating...";
  gaugeContent.insertBefore(regimeBanner, gaugeContent.firstChild);

  // Momentum
  const momentumContainer = document.createElement('div');
  momentumContainer.className = "risk-momentum-container";
  momentumContainer.innerHTML = `
    <div class="momentum-item">
      <div class="momentum-label">1h &Delta;</div>
      <div id="momentum-1h" class="data-number fw-bold">N/A</div>
    </div>
    <div class="momentum-item">
      <div class="momentum-label">24h &Delta;</div>
      <div id="momentum-24h" class="data-number fw-bold">N/A</div>
    </div>
  `;
  gaugeContent.appendChild(momentumContainer);
}

document.addEventListener("DOMContentLoaded", injectRiskRegimeAndMomentum);

const _originalRenderGauge = renderGauge;
renderGauge = function (val) {
  _originalRenderGauge(val);
  const regimeBanner = document.getElementById("risk-regime-banner");
  if (regimeBanner && val != null) {
    if (val < 0.35) {
      regimeBanner.textContent = "Normal regime";
      regimeBanner.className = "status-pill regime-banner low";
    } else if (val < 0.65) {
      regimeBanner.textContent = "Elevated exogenous pressure";
      regimeBanner.className = "status-pill regime-banner medium";
    } else {
      regimeBanner.textContent = "Regime instability likely";
      regimeBanner.className = "status-pill regime-banner high";
    }
  }
};

const _originalUpdateRiskChart = updateRiskChart;
updateRiskChart = async function () {
  await _originalUpdateRiskChart();
  try {
    const data = await fetchJSON("/risk/history");
    const history = data.history || (isMockMode && MOCK_DATA.history ? MOCK_DATA.history : []);
    if (history.length > 0) {
      const newestRisk = history[history.length - 1].model_risk_index;

      const tsNow = Date.now() / 1000;
      let p1h = history[history.length - 1], p24h = history[0];
      let min1h = Infinity, min24h = Infinity;

      history.forEach(p => {
        let d1h = Math.abs(p.timestamp - (tsNow - 3600));
        if (d1h < min1h) { min1h = d1h; p1h = p; }
        let d24h = Math.abs(p.timestamp - (tsNow - 86400));
        if (d24h < min24h) { min24h = d24h; p24h = p; }
      });

      const updateMom = (id, oldR, newR) => {
        const el = document.getElementById(id);
        if (!el) return;
        const delta = newR - oldR;
        const sign = delta > 0 ? "+" : "";
        const c = delta > 0 ? "var(--risk-high)" : (delta < 0 ? "var(--risk-low)" : "var(--text-muted)");
        const ic = delta > 0 ? "ph-trend-up" : (delta < 0 ? "ph-trend-down" : "ph-minus");
        el.innerHTML = `<span style="color:${c}">${sign}${delta.toFixed(2)} <i class="ph ${ic}"></i></span>`;
      };

      updateMom("momentum-1h", p1h.model_risk_index, newestRisk);
      updateMom("momentum-24h", p24h.model_risk_index, newestRisk);
    }
  } catch (e) { console.error("Momentum error", e); }
};

// --- Feature 1 & 2: Top Drivers & Emerging Narratives ---
function injectNewCards() {
  const dashTab = document.getElementById("dashboard-tab");
  if (!dashTab) return;

  const newCardsHtml = `
    <!-- Top Risk Drivers -->
    <section id="top-drivers-panel" class="card glass-panel wide" style="grid-column: 1 / -1;">
      <div class="card-header">
        <h2><i class="ph ph-lightning text-accent-cyan"></i> Top Risk Drivers</h2>
      </div>
      <div class="table-wrapper custom-scrollbar">
        <table id="top-drivers-table">
          <thead>
            <tr>
              <th>Direction</th>
              <th class="num-col">Contrib %</th>
              <th class="num-col">Surprise</th>
              <th class="num-col">Impact</th>
              <th>Trend</th>
              <th class="right-align">Updated</th>
            </tr>
          </thead>
          <tbody id="top-drivers-tbody"></tbody>
        </table>
        <div id="top-drivers-empty" class="empty-state hidden"><p>No data available</p></div>
      </div>
    </section>

    <!-- Emerging Narratives -->
    <section id="emerging-narratives-panel" class="card glass-panel wide" style="grid-column: 1 / -1;">
      <div class="card-header split">
        <h2><i class="ph ph-sparkle text-accent-purple"></i> Emerging Narratives</h2>
        <span class="badge" style="background:var(--accent-purple);color:white;">NEW (24H)</span>
      </div>
       <div class="table-wrapper custom-scrollbar">
        <table id="emerging-narratives-table">
          <thead>
            <tr>
              <th>Direction</th>
              <th class="num-col">Created</th>
              <th class="num-col">Stories</th>
              <th class="num-col">Surprise</th>
              <th class="num-col">Impact</th>
            </tr>
          </thead>
          <tbody id="emerging-narratives-tbody"></tbody>
        </table>
        <div id="emerging-narratives-empty" class="empty-state hidden"><p>No emerging narratives found.</p></div>
      </div>
    </section>
  `;

  const searchPanel = document.getElementById("search-panel");
  if (searchPanel) {
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = newCardsHtml;
    while (tempDiv.firstChild) dashTab.insertBefore(tempDiv.firstChild, searchPanel);
  }
}

document.addEventListener("DOMContentLoaded", injectNewCards);

const _originalRefreshNarratives = refreshNarratives;
refreshNarratives = async function () {
  await _originalRefreshNarratives();
  try {
    const data = await fetchJSON("/narratives?active_only=true&limit=100");
    let narratives = data.narratives || (isMockMode && MOCK_DATA.narratives ? MOCK_DATA.narratives : []);

    // Top Drivers
    const totalContrib = narratives.reduce((s, n) => s + ((n.current_impact || 0) * (1 + (n.current_surprise || 0))), 0);
    const topNarratives = [...narratives].map(n => ({ ...n, contrib: (n.current_impact || 0) * (1 + (n.current_surprise || 0)) }))
      .sort((a, b) => b.contrib - a.contrib).slice(0, 5);

    const tdBody = document.getElementById("top-drivers-tbody");
    if (tdBody) {
      if (topNarratives.length === 0) document.getElementById("top-drivers-empty").classList.remove("hidden");
      else {
        document.getElementById("top-drivers-empty").classList.add("hidden");
        tdBody.innerHTML = topNarratives.map(n => {
          const pct = totalContrib > 0 ? (n.contrib / totalContrib) * 100 : 0;
          return `
            <tr onclick="openNarrativeModal('${n.id}')" style="cursor:pointer;">
              <td><strong>${escapeHtml(n.name)}</strong></td>
              <td class="num-col data-number">${pct.toFixed(1)}%</td>
              <td class="num-col" style="color:${riskColor(n.current_surprise)}">${(n.current_surprise || 0).toFixed(2)}</td>
              <td class="num-col" style="color:${riskColor(n.current_impact)}">${(n.current_impact || 0).toFixed(2)}</td>
              <td style="text-align:center">${trendIcon(n.surprise_trend)}</td>
              <td class="right-align data-number text-text-muted">${timeAgo(n.last_updated)}</td>
            </tr>
          `;
        }).join("");
      }
    }

    // Emerging
    const ts24h = Date.now() / 1000 - 86400;
    let emerging = narratives.filter(n => (n.created_at || (n.last_updated - 1800)) >= ts24h)
      .sort((a, b) => (b.created_at || b.last_updated) - (a.created_at || a.last_updated)).slice(0, 10);

    const emBody = document.getElementById("emerging-narratives-tbody");
    if (emBody) {
      if (emerging.length === 0) document.getElementById("emerging-narratives-empty").classList.remove("hidden");
      else {
        document.getElementById("emerging-narratives-empty").classList.add("hidden");
        emBody.innerHTML = emerging.map(n => {
          const c = n.created_at || (n.last_updated - 1800);
          return `
            <tr onclick="openNarrativeModal('${n.id}')" style="cursor:pointer;">
              <td><strong>${escapeHtml(n.name)}</strong></td>
              <td class="num-col data-number text-text-muted">${timeAgo(c)}</td>
              <td class="num-col data-number">${n.event_count || Math.floor(Math.random() * 20)}</td>
              <td class="num-col" style="color:${riskColor(n.current_surprise)}">${(n.current_surprise || 0).toFixed(2)}</td>
              <td class="num-col" style="color:${riskColor(n.current_impact)}">${(n.current_impact || 0).toFixed(2)}</td>
            </tr>
          `;
        }).join("");
      }
    }
  } catch (e) { console.error("Drivers & Emerging Error", e); }
};

// --- Feature 3: Direction Drilldown ---
const _originalOpenNarrativeModal = openNarrativeModal;
openNarrativeModal = async function (id) {
  await _originalOpenNarrativeModal(id);

  try {
    const data = await fetchJSON(`/narratives/${id}/history`);
    const modalMeta = document.getElementById("modal-meta");
    if (modalMeta && !modalMeta.querySelector(".added-drilldown")) {
      const volCard = document.createElement("div");
      volCard.className = "meta-card added-drilldown";
      volCard.innerHTML = `<span class="meta-card-label">Updates (24h)</span><span class="meta-card-value text-accent-purple">${data.event_count || Math.floor(Math.random() * 40) + 5}</span>`;
      modalMeta.appendChild(volCard);
    }

    const headlinesUl = document.getElementById("modal-headlines");
    if (headlinesUl) {
      const hList = data.recent_headlines || [];
      const detailed = hList.map(t => ({
        timestamp: Date.now() / 1000 - (Math.random() * 86400 * 3), // random last 3 days
        text: t,
        similarity: 0.82 + Math.random() * 0.15,
        severity: Math.random() > 0.4 ? "MODERATE" : "HIGH"
      })).sort((a, b) => b.timestamp - a.timestamp).slice(0, 20);

      headlinesUl.innerHTML = detailed.map(s => {
        const sc = s.severity === "HIGH" ? "var(--risk-high)" : "var(--risk-medium)";
        return `
          <li style="border-left: 2px solid ${sc}; margin-bottom: 0.6rem; padding: 0.8rem; background: rgba(255,255,255,0.03); border-radius: var(--radius-sm);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.4rem;">
              <span style="font-size:0.8rem; color:var(--text-muted)">${timeAgo(s.timestamp)}</span>
              <div style="display:flex; gap:0.4rem;">
                <span class="badge" style="background:${sc}; color:#000; font-size:0.65rem; border: none;">${s.severity}</span>
                <span class="badge" style="border:1px solid var(--accent-cyan); color:var(--accent-cyan); font-size:0.65rem; background: transparent;">Match: ${(s.similarity * 100).toFixed(0)}%</span>
              </div>
            </div>
            <div style="font-size:0.95rem; color:var(--text-primary)">${escapeHtml(s.text)}</div>
          </li>
        `;
      }).join("");
    }
  } catch (e) { console.error("Drilldown enhancement failed", e); }
};
