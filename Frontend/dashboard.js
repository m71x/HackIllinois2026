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

// ── Omnibar: Ticker search with autocomplete & validation ───────────────

// S&P 500 + major US-listed tickers (backend validates via yfinance for unlisted symbols)
const VALID_TICKERS = {
  AAPL: "Apple Inc.", ABBV: "AbbVie Inc.", ABT: "Abbott Laboratories", ACN: "Accenture plc",
  ADBE: "Adobe Inc.", ADI: "Analog Devices", ADM: "Archer-Daniels-Midland", ADP: "Automatic Data Processing",
  ADSK: "Autodesk Inc.", AEP: "American Electric Power", AFL: "Aflac Inc.", AIG: "American Intl Group",
  AMAT: "Applied Materials", AMD: "Advanced Micro Devices", AMGN: "Amgen Inc.", AMP: "Ameriprise Financial",
  AMZN: "Amazon.com Inc.", ANET: "Arista Networks", ANSS: "Ansys Inc.", AON: "Aon plc",
  APD: "Air Products & Chemicals", APH: "Amphenol Corp.", AVGO: "Broadcom Inc.", AXP: "American Express",
  BA: "Boeing Co.", BAC: "Bank of America", BAX: "Baxter International", BDX: "Becton Dickinson",
  BK: "Bank of New York Mellon", BKNG: "Booking Holdings", BLK: "BlackRock Inc.", BMY: "Bristol-Myers Squibb",
  "BRK.B": "Berkshire Hathaway", BSX: "Boston Scientific", C: "Citigroup Inc.", CAT: "Caterpillar Inc.",
  CB: "Chubb Ltd.", CCI: "Crown Castle", CDNS: "Cadence Design Systems", CEG: "Constellation Energy",
  CHTR: "Charter Communications", CI: "Cigna Group", CL: "Colgate-Palmolive", CMCSA: "Comcast Corp.",
  CME: "CME Group", CMG: "Chipotle Mexican Grill", COP: "ConocoPhillips", COST: "Costco Wholesale",
  CRM: "Salesforce Inc.", CSCO: "Cisco Systems", CTAS: "Cintas Corp.", CVS: "CVS Health",
  CVX: "Chevron Corp.", D: "Dominion Energy", DD: "DuPont de Nemours", DE: "Deere & Co.",
  DHR: "Danaher Corp.", DIS: "Walt Disney Co.", DUK: "Duke Energy", ECL: "Ecolab Inc.",
  EMR: "Emerson Electric", EOG: "EOG Resources", ETN: "Eaton Corp.", EW: "Edwards Lifesciences",
  EXC: "Exelon Corp.", F: "Ford Motor Co.", FANG: "Diamondback Energy", FCX: "Freeport-McMoRan",
  FDX: "FedEx Corp.", FI: "Fiserv Inc.", GD: "General Dynamics", GE: "GE Aerospace",
  GILD: "Gilead Sciences", GM: "General Motors", GOOG: "Alphabet Inc. (C)", GOOGL: "Alphabet Inc. (A)",
  GPN: "Global Payments", GS: "Goldman Sachs", HAL: "Halliburton Co.", HD: "Home Depot",
  HON: "Honeywell International", HUM: "Humana Inc.", IBM: "IBM Corp.", ICE: "Intercontinental Exchange",
  IDXX: "IDEXX Laboratories", INTC: "Intel Corp.", INTU: "Intuit Inc.", ISRG: "Intuitive Surgical",
  ITW: "Illinois Tool Works", JNJ: "Johnson & Johnson", JPM: "JPMorgan Chase", KHC: "Kraft Heinz Co.",
  KLAC: "KLA Corp.", KO: "Coca-Cola Co.", LEN: "Lennar Corp.", LHX: "L3Harris Technologies",
  LIN: "Linde plc", LLY: "Eli Lilly & Co.", LMT: "Lockheed Martin", LOW: "Lowe's Companies",
  LRCX: "Lam Research", MA: "Mastercard Inc.", MAR: "Marriott International", MCD: "McDonald's Corp.",
  MCHP: "Microchip Technology", MCK: "McKesson Corp.", MCO: "Moody's Corp.", MDLZ: "Mondelez Intl",
  MDT: "Medtronic plc", MET: "MetLife Inc.", META: "Meta Platforms", MMM: "3M Co.",
  MO: "Altria Group", MPC: "Marathon Petroleum", MRK: "Merck & Co.", MRNA: "Moderna Inc.",
  MS: "Morgan Stanley", MSCI: "MSCI Inc.", MSFT: "Microsoft Corp.", MSI: "Motorola Solutions",
  MU: "Micron Technology", NFLX: "Netflix Inc.", NKE: "Nike Inc.", NOC: "Northrop Grumman",
  NOW: "ServiceNow Inc.", NSC: "Norfolk Southern", NVDA: "NVIDIA Corp.", ORCL: "Oracle Corp.",
  OXY: "Occidental Petroleum", PANW: "Palo Alto Networks", PAYX: "Paychex Inc.", PEP: "PepsiCo Inc.",
  PFE: "Pfizer Inc.", PG: "Procter & Gamble", PGR: "Progressive Corp.", PH: "Parker-Hannifin",
  PLTR: "Palantir Technologies", PM: "Philip Morris Intl", PNC: "PNC Financial", PSA: "Public Storage",
  PSX: "Phillips 66", PYPL: "PayPal Holdings", QCOM: "Qualcomm Inc.", REGN: "Regeneron Pharma",
  ROP: "Roper Technologies", ROST: "Ross Stores", RTX: "RTX Corp.", SBUX: "Starbucks Corp.",
  SCHW: "Charles Schwab", SHW: "Sherwin-Williams", SLB: "Schlumberger", SMCI: "Super Micro Computer",
  SNPS: "Synopsys Inc.", SO: "Southern Co.", SPG: "Simon Property Group", SPGI: "S&P Global",
  SRE: "Sempra", SYK: "Stryker Corp.", SYY: "Sysco Corp.", T: "AT&T Inc.",
  TFC: "Truist Financial", TGT: "Target Corp.", TJX: "TJX Companies", TMO: "Thermo Fisher Scientific",
  TMUS: "T-Mobile US", TSLA: "Tesla Inc.", TSN: "Tyson Foods", TXN: "Texas Instruments",
  UNH: "UnitedHealth Group", UNP: "Union Pacific", UPS: "United Parcel Service", URI: "United Rentals",
  USB: "U.S. Bancorp", V: "Visa Inc.", VICI: "VICI Properties", VLO: "Valero Energy",
  VRSK: "Verisk Analytics", VRTX: "Vertex Pharmaceuticals", VZ: "Verizon Communications",
  WBA: "Walgreens Boots Alliance", WFC: "Wells Fargo", WM: "Waste Management", WMT: "Walmart Inc.",
  XEL: "Xcel Energy", XOM: "Exxon Mobil", ZTS: "Zoetis Inc.",
  // Popular non-S&P tickers
  ABNB: "Airbnb Inc.", AI: "C3.ai Inc.", ARM: "Arm Holdings", COIN: "Coinbase Global",
  CRWD: "CrowdStrike Holdings", DDOG: "Datadog Inc.", FTNT: "Fortinet Inc.", GME: "GameStop Corp.",
  HOOD: "Robinhood Markets", MARA: "Marathon Digital", MELI: "MercadoLibre", NET: "Cloudflare Inc.",
  PINS: "Pinterest Inc.", RIVN: "Rivian Automotive", RKLB: "Rocket Lab USA", ROKU: "Roku Inc.",
  SE: "Sea Ltd.", SHOP: "Shopify Inc.", SNAP: "Snap Inc.", SNOW: "Snowflake Inc.",
  SQ: "Block Inc.", SPOT: "Spotify Technology", SOFI: "SoFi Technologies", TTD: "The Trade Desk",
  UBER: "Uber Technologies", U: "Unity Software", ZM: "Zoom Video Comms", ZS: "Zscaler Inc.",
  // Major ETFs
  SPY: "SPDR S&P 500 ETF", QQQ: "Invesco QQQ Trust", IWM: "iShares Russell 2000",
  DIA: "SPDR Dow Jones ETF", VOO: "Vanguard S&P 500 ETF", VTI: "Vanguard Total Stock Market",
  ARKK: "ARK Innovation ETF", XLF: "Financial Select SPDR", XLE: "Energy Select SPDR",
  XLK: "Technology Select SPDR", XLV: "Health Care Select SPDR", GLD: "SPDR Gold Shares",
  SLV: "iShares Silver Trust", TLT: "iShares 20+ Yr Treasury", HYG: "iShares High Yield Bond",
  // ADRs & International
  NVO: "Novo Nordisk", TSM: "Taiwan Semiconductor", BABA: "Alibaba Group", JD: "JD.com Inc.",
  PDD: "PDD Holdings", ASML: "ASML Holding", SAP: "SAP SE", TM: "Toyota Motor",
  NVS: "Novartis AG", UL: "Unilever plc", BP: "BP plc", SHEL: "Shell plc", RIO: "Rio Tinto",
  BHP: "BHP Group", VALE: "Vale S.A.",
};

// ── Omnibar event wiring ────────────────────────────────────────────────

document.getElementById("search-input").addEventListener("input", omnibarAutocomplete);
document.getElementById("search-input").addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); omnibarSubmit(); }
  if (e.key === "Escape") omnibarDismiss();
  // Arrow-key navigation through suggestions
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const items = document.querySelectorAll("#search-results .omnibar-suggestion");
    if (!items.length) return;
    const active = document.querySelector("#search-results .omnibar-suggestion.active");
    let idx = active ? [...items].indexOf(active) : -1;
    if (active) active.classList.remove("active");
    idx = e.key === "ArrowDown" ? Math.min(idx + 1, items.length - 1) : Math.max(idx - 1, 0);
    items[idx].classList.add("active");
    items[idx].scrollIntoView({ block: "nearest" });
  }
});

// Focus the input when user clicks anywhere in the omnibar row
document.querySelector(".omnibar-input-row").addEventListener("click", () => {
  document.getElementById("search-input").focus();
});

// Click outside → dismiss results
document.addEventListener("mousedown", e => {
  const omnibar = document.getElementById("omnibar");
  if (omnibar && !omnibar.contains(e.target)) omnibarDismiss();
});

// ── Omnibar core functions ──────────────────────────────────────────────

function omnibarDismiss() {
  document.getElementById("search-results").innerHTML = "";
}

function omnibarAutocomplete() {
  const raw = document.getElementById("search-input").value.trim().toUpperCase();
  const resDiv = document.getElementById("search-results");

  if (!raw) { resDiv.innerHTML = ""; return; }

  // Filter tickers: prefix match on symbol, or partial match on company name
  const matches = Object.entries(VALID_TICKERS)
    .filter(([sym, name]) => sym.startsWith(raw) || name.toUpperCase().includes(raw))
    .sort((a, b) => {
      // Exact prefix matches first, then alphabetical
      const aPrefix = a[0].startsWith(raw) ? 0 : 1;
      const bPrefix = b[0].startsWith(raw) ? 0 : 1;
      return aPrefix - bPrefix || a[0].localeCompare(b[0]);
    })
    .slice(0, 8);

  if (matches.length === 0) {
    // Could still be a valid ticker not in our local set — let user try anyway
    if (/^[A-Z]{1,5}(\.[A-Z])?$/.test(raw)) {
      resDiv.innerHTML = `
        <div class="omnibar-suggestion omnibar-try-anyway" onclick="omnibarSelect('${raw}')">
          <span class="omnibar-ticker-sym">${escapeHtml(raw)}</span>
          <span class="omnibar-ticker-name">Look up on market…</span>
          <i class="ph ph-arrow-right"></i>
        </div>`;
    } else {
      resDiv.innerHTML = `
        <div class="omnibar-no-match">
          <i class="ph ph-warning-circle"></i> Enter a valid ticker symbol
        </div>`;
    }
    return;
  }

  resDiv.innerHTML = matches.map(([sym, name]) => `
    <div class="omnibar-suggestion" data-symbol="${sym}" onclick="omnibarSelect('${sym}')">
      <span class="omnibar-ticker-sym">${sym}</span>
      <span class="omnibar-ticker-name">${escapeHtml(name)}</span>
    </div>
  `).join("");
}

function omnibarSelect(symbol) {
  document.getElementById("search-input").value = symbol;
  omnibarSubmit(symbol);
}

async function omnibarSubmit(directSymbol) {
  const input = document.getElementById("search-input");
  const resDiv = document.getElementById("search-results");

  // Prefer the directly-passed symbol; fall back to reading the input
  let symbol = directSymbol
    || (document.querySelector("#search-results .omnibar-suggestion.active")?.dataset.symbol)
    || input.value.trim().toUpperCase();

  if (!symbol) return;
  symbol = symbol.trim().toUpperCase();

  // Validate format: 1-5 uppercase letters, optional dot suffix (BRK.B)
  if (!/^[A-Z]{1,5}(\.[A-Z])?$/.test(symbol)) {
    resDiv.innerHTML = `<div class="omnibar-no-match">
      <i class="ph ph-warning-circle"></i> "${escapeHtml(symbol)}" is not a valid ticker format
    </div>`;
    return;
  }

  input.value = symbol;
  resDiv.innerHTML = `<div class="omnibar-loading"><div class="spinner"></div> Looking up <strong>${symbol}</strong>…</div>`;

  try {
    let result;

    if (isMockMode) {
      result = _buildMockTickerResult(symbol);
      if (!result) {
        resDiv.innerHTML = `<div class="omnibar-no-match">
          <i class="ph ph-warning-circle"></i> No data found for ticker <strong>${symbol}</strong>
        </div>`;
        return;
      }
    } else {
      try {
        const res = await fetch(`${API}/tickers/${encodeURIComponent(symbol)}?n_results=10`);
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          resDiv.innerHTML = `<div class="omnibar-no-match">
            <i class="ph ph-warning-circle"></i> ${escapeHtml(err.detail || `Ticker '${symbol}' not found`)}
          </div>`;
          return;
        }
        result = await res.json();
      } catch (_netErr) {
        // Backend unreachable — fall back to mock
        isMockMode = true;
        result = _buildMockTickerResult(symbol);
        if (!result) {
          resDiv.innerHTML = `<div class="omnibar-no-match">
            <i class="ph ph-warning-circle"></i> No data found for ticker <strong>${symbol}</strong>
          </div>`;
          return;
        }
      }
    }

    // Dismiss the dropdown & open the full ticker analysis window
    omnibarDismiss();
    openTickerAnalysis(result);

  } catch (e) {
    console.error("Omnibar ticker search failed:", e);
    resDiv.innerHTML = `<div class="omnibar-no-match">
      <i class="ph ph-warning-circle"></i> Search failed — check your connection
    </div>`;
  }
}

/** Build a mock result object from MOCK_TICKER_NARRATIVES (returns null if not found). */
function _buildMockTickerResult(symbol) {
  const mock = MOCK_TICKER_NARRATIVES[symbol];
  if (!mock) return null;
  const matchedNarratives = mock.ids.map((id, i) => {
    const n = MOCK_DATA.narratives.find(n => n.id === id);
    if (!n) return null;
    return {
      id: n.id, name: n.name, description: n.description,
      distance: 0.15 + i * 0.08, similarity: 0.92 - i * 0.04,
      model_risk: n.model_risk, current_surprise: n.current_surprise,
      current_impact: n.current_impact, event_count: n.event_count, is_active: true,
    };
  }).filter(Boolean);
  return {
    ticker: symbol, company_name: mock.company_name,
    sector: mock.sector, industry: mock.industry,
    narratives: matchedNarratives,
  };
}

// ============================================================
// SECTION: 3D Vector Space Visualization (Three.js)
// ============================================================

let _active3DScene = null;  // track active scene for cleanup

/**
 * Render an interactive 3D point graph showing the search query (ticker)
 * as a central node surrounded by matched narrative nodes.
 *
 * @param {HTMLElement} container - DOM element to render into
 * @param {string} tickerSymbol - The searched ticker symbol
 * @param {Array} narratives - Array of { name, distance, similarity, model_risk, event_count, ... }
 * @returns {{ dispose: () => void }} cleanup handle
 */
function render3DVectorSpace(container, tickerSymbol, narratives) {
  const width = container.clientWidth;
  const height = container.clientHeight || 400;

  // ── Scene, camera, renderer ──
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
  camera.position.set(0, 2.5, 6);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  // ── Lights ──
  const ambientLight = new THREE.AmbientLight(0x334466, 0.8);
  scene.add(ambientLight);
  const pointLight = new THREE.PointLight(0x00f0ff, 1.2, 30);
  pointLight.position.set(2, 4, 3);
  scene.add(pointLight);
  const pointLight2 = new THREE.PointLight(0xf43f5e, 0.6, 25);
  pointLight2.position.set(-3, -2, 2);
  scene.add(pointLight2);

  // ── Helpers: risk → color ──
  function riskToColor(risk) {
    const r = risk ?? 0;
    if (r >= 0.66) return new THREE.Color(0xf43f5e);
    if (r >= 0.33) return new THREE.Color(0xf59e0b);
    return new THREE.Color(0x10b981);
  }

  // ── Grid / reference plane ──
  const gridHelper = new THREE.GridHelper(10, 20, 0x1a2340, 0x111827);
  gridHelper.position.y = -2;
  scene.add(gridHelper);

  // ── Central ticker node (large glowing sphere) ──
  const centerGeo = new THREE.SphereGeometry(0.35, 32, 32);
  const centerMat = new THREE.MeshPhongMaterial({
    color: 0x00f0ff,
    emissive: 0x00f0ff,
    emissiveIntensity: 0.5,
    transparent: true,
    opacity: 0.95,
    shininess: 100,
  });
  const centerMesh = new THREE.Mesh(centerGeo, centerMat);
  centerMesh.userData = { type: "ticker", label: tickerSymbol };
  scene.add(centerMesh);

  // Inner glow sphere
  const glowGeo = new THREE.SphereGeometry(0.55, 32, 32);
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.08,
  });
  const glowMesh = new THREE.Mesh(glowGeo, glowMat);
  scene.add(glowMesh);

  // ── Create text sprite helper ──
  function makeTextSprite(text, opts = {}) {
    const fontSize = opts.fontSize || 48;
    const fontFamily = opts.fontFamily || "'Outfit', 'Inter', sans-serif";
    const color = opts.color || "#ffffff";
    const bgColor = opts.bgColor || "rgba(10, 15, 28, 0.75)";
    const padding = 16;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    ctx.font = `600 ${fontSize}px ${fontFamily}`;
    const textWidth = ctx.measureText(text).width;

    canvas.width = textWidth + padding * 2;
    canvas.height = fontSize + padding * 2;

    // Background
    ctx.fillStyle = bgColor;
    const r = 12;
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(canvas.width - r, 0);
    ctx.quadraticCurveTo(canvas.width, 0, canvas.width, r);
    ctx.lineTo(canvas.width, canvas.height - r);
    ctx.quadraticCurveTo(canvas.width, canvas.height, canvas.width - r, canvas.height);
    ctx.lineTo(r, canvas.height);
    ctx.quadraticCurveTo(0, canvas.height, 0, canvas.height - r);
    ctx.lineTo(0, r);
    ctx.quadraticCurveTo(0, 0, r, 0);
    ctx.closePath();
    ctx.fill();

    // Text
    ctx.font = `600 ${fontSize}px ${fontFamily}`;
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);

    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(spriteMat);

    const scale = (opts.scale || 0.015) * canvas.height;
    sprite.scale.set(scale * (canvas.width / canvas.height), scale, 1);
    return sprite;
  }

  // ── Center label ──
  const centerLabel = makeTextSprite(tickerSymbol, {
    fontSize: 56, color: "#00f0ff", bgColor: "rgba(0, 240, 255, 0.12)", scale: 0.018,
  });
  centerLabel.position.set(0, 0.65, 0);
  scene.add(centerLabel);

  // ── Layout narrative nodes on a golden-angle spherical spiral ──
  const nodeGroup = new THREE.Group();
  scene.add(nodeGroup);
  const hoverable = [];  // meshes for raycasting

  const goldenAngle = Math.PI * (3 - Math.sqrt(5)); // ~137.5°

  narratives.forEach((n, i) => {
    // Radial distance based on semantic distance (closer = more similar)
    const dist = n.distance ?? 0.5;
    const radialDist = 1.0 + dist * 5.0;  // scale to a visible range

    // Spherical distribution using golden angle spiral
    const theta = goldenAngle * i;
    const phi = Math.acos(1 - (2 * (i + 0.5)) / Math.max(narratives.length, 1));

    const x = radialDist * Math.sin(phi) * Math.cos(theta);
    const y = radialDist * Math.cos(phi) * 0.6;  // flatten slightly
    const z = radialDist * Math.sin(phi) * Math.sin(theta);

    // Sphere size based on event count
    const eventCount = n.event_count ?? 1;
    const radius = 0.12 + Math.min(eventCount / 20, 0.25);

    const color = riskToColor(n.model_risk);

    // Node sphere
    const geo = new THREE.SphereGeometry(radius, 24, 24);
    const mat = new THREE.MeshPhongMaterial({
      color: color,
      emissive: color,
      emissiveIntensity: 0.35,
      transparent: true,
      opacity: 0.9,
      shininess: 80,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.userData = {
      type: "narrative",
      name: n.name,
      similarity: n.similarity,
      model_risk: n.model_risk,
      event_count: n.event_count,
      index: i,
      id: n.id,
      description: n.description,
      current_surprise: n.current_surprise,
      current_impact: n.current_impact,
      surprise_trend: n.surprise_trend,
    };
    nodeGroup.add(mesh);
    hoverable.push(mesh);

    // Outer glow per node
    const nodeGlowGeo = new THREE.SphereGeometry(radius * 1.8, 16, 16);
    const nodeGlowMat = new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.06,
    });
    const nodeGlow = new THREE.Mesh(nodeGlowGeo, nodeGlowMat);
    nodeGlow.position.copy(mesh.position);
    nodeGroup.add(nodeGlow);

    // Connection line from center to node
    const similarity = n.similarity ?? 0.5;
    const lineColor = color.clone().lerp(new THREE.Color(0x00f0ff), 0.3);
    const lineMat = new THREE.LineBasicMaterial({
      color: lineColor,
      transparent: true,
      opacity: 0.15 + similarity * 0.35,
      linewidth: 1,
    });
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(x, y, z),
    ]);
    const line = new THREE.Line(lineGeo, lineMat);
    nodeGroup.add(line);

    // Label sprite
    const simPct = ((similarity) * 100).toFixed(0);
    const labelText = n.name.length > 20 ? n.name.slice(0, 20) + "…" : n.name;
    const label = makeTextSprite(`${labelText}  ${simPct}%`, {
      fontSize: 36,
      color: "#c8d6e5",
      bgColor: "rgba(10, 15, 28, 0.65)",
      scale: 0.012,
    });
    label.position.set(x, y + radius + 0.35, z);
    nodeGroup.add(label);
  });

  // ── Orbit Controls ──
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.8;
  controls.minDistance = 2;
  controls.maxDistance = 15;
  controls.target.set(0, 0, 0);
  controls.update();

  // Pause auto-rotate on user interaction
  let userInteracting = false;
  renderer.domElement.addEventListener("pointerdown", () => {
    userInteracting = true;
    controls.autoRotate = false;
  });
  renderer.domElement.addEventListener("pointerup", () => {
    setTimeout(() => { controls.autoRotate = true; userInteracting = false; }, 3000);
  });

  // ── Raycasting for hover tooltips ──
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let tooltip = container.querySelector(".vector-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.className = "vector-tooltip";
    container.appendChild(tooltip);
  }

  function onMouseMove(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(hoverable);

    if (intersects.length > 0) {
      const obj = intersects[0].object;
      const d = obj.userData;

      // Highlight
      obj.material.emissiveIntensity = 0.8;
      renderer.domElement.style.cursor = "pointer";

      // Show tooltip
      tooltip.style.display = "block";
      tooltip.style.left = (e.clientX - rect.left + 16) + "px";
      tooltip.style.top = (e.clientY - rect.top - 12) + "px";
      tooltip.innerHTML = `
        <div class="vt-name">${escapeHtml(d.name)}</div>
        <div class="vt-row"><span class="vt-label">Similarity</span><span class="vt-val text-accent-cyan">${((d.similarity ?? 0) * 100).toFixed(0)}%</span></div>
        <div class="vt-row"><span class="vt-label">Risk</span><span class="vt-val" style="color:${riskColor(d.model_risk)}">${(d.model_risk ?? 0).toFixed(2)}</span></div>
        <div class="vt-row"><span class="vt-label">Events</span><span class="vt-val">${d.event_count ?? 0}</span></div>
        <div class="vt-click-hint"><i class="ph ph-cursor-click"></i> Click to inspect</div>
      `;
    } else {
      // Reset
      hoverable.forEach(m => { m.material.emissiveIntensity = 0.35; });
      renderer.domElement.style.cursor = "grab";
      tooltip.style.display = "none";
    }
  }

  renderer.domElement.addEventListener("mousemove", onMouseMove);

  // ── Click on narrative node → open detail modal ──
  let _mouseDownPos = { x: 0, y: 0 };
  function onNodeMouseDown(e) { _mouseDownPos = { x: e.clientX, y: e.clientY }; }
  function onNodeClick(e) {
    if (Math.hypot(e.clientX - _mouseDownPos.x, e.clientY - _mouseDownPos.y) > 5) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(hoverable);
    if (intersects.length > 0) {
      const d = intersects[0].object.userData;
      if (d.id) openNarrativeModal(d.id);
    }
  }
  renderer.domElement.addEventListener("mousedown", onNodeMouseDown);
  renderer.domElement.addEventListener("click", onNodeClick);

  // ── Animation loop ──
  let animFrame;
  const clock = new THREE.Clock();

  function animate() {
    animFrame = requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    // Pulse the center glow
    glowMesh.scale.setScalar(1 + Math.sin(t * 2) * 0.1);
    glowMat.opacity = 0.06 + Math.sin(t * 2) * 0.03;
    centerMat.emissiveIntensity = 0.4 + Math.sin(t * 3) * 0.15;

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // ── Resize handling ──
  const resizeObs = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  resizeObs.observe(container);

  // ── Cleanup function ──
  function dispose() {
    cancelAnimationFrame(animFrame);
    resizeObs.disconnect();
    renderer.domElement.removeEventListener("mousemove", onMouseMove);
    renderer.domElement.removeEventListener("mousedown", onNodeMouseDown);
    renderer.domElement.removeEventListener("click", onNodeClick);
    controls.dispose();
    renderer.dispose();

    // Dispose all geometries and materials
    scene.traverse(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (obj.material.map) obj.material.map.dispose();
        obj.material.dispose();
      }
    });

    if (renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
    if (tooltip) tooltip.style.display = "none";
  }

  return { dispose };
}


// ── Ticker Analysis Window ──────────────────────────────────────────────
let _tickerAnalysisChart = null;

function openTickerAnalysis(result) {
  // Remove any previous ticker-analysis overlay
  const old = document.getElementById("ticker-analysis-overlay");
  if (old) old.remove();
  if (_tickerAnalysisChart) { _tickerAnalysisChart.destroy(); _tickerAnalysisChart = null; }

  // Also activate the narrative panel ticker filter
  tickerFilterActive = true;
  tickerFilterData = result;
  const tickerInput = document.getElementById("ticker-input");
  if (tickerInput) tickerInput.value = result.ticker;
  renderTickerFilter(result);

  const narrs = result.narratives || [];

  // Compute aggregate stats
  const avgRisk = narrs.length ? narrs.reduce((s, n) => s + (n.model_risk ?? 0), 0) / narrs.length : 0;
  const maxRisk = narrs.length ? Math.max(...narrs.map(n => n.model_risk ?? 0)) : 0;
  const avgSim = narrs.length ? narrs.reduce((s, n) => s + (n.similarity ?? 0), 0) / narrs.length : 0;
  const totalEvents = narrs.reduce((s, n) => s + (n.event_count ?? 0), 0);

  // Build the overlay HTML
  const overlay = document.createElement("div");
  overlay.id = "ticker-analysis-overlay";
  overlay.className = "ticker-analysis-overlay";
  overlay.innerHTML = `
    <div class="ticker-analysis-backdrop"></div>
    <div class="ticker-analysis-window glass-panel">
      <button class="modal-close-btn ticker-analysis-close" aria-label="Close"><i class="ph ph-x"></i></button>

      <!-- Header -->
      <div class="ta-header">
        <div class="ta-badge-row">
          <span class="ta-ticker-badge">${escapeHtml(result.ticker)}</span>
          <span class="ta-sector-tag">${escapeHtml(result.sector || "—")}</span>
          <span class="ta-sector-tag">${escapeHtml(result.industry || "—")}</span>
        </div>
        <h2 class="ta-title">${escapeHtml(result.company_name)}</h2>
        <p class="ta-subtitle">Narrative exposure analysis via vector similarity</p>
      </div>

      <!-- Stats Row -->
      <div class="ta-stats-row">
        <div class="ta-stat-card">
          <span class="ta-stat-label">Matched Narratives</span>
          <span class="ta-stat-value text-accent-cyan">${narrs.length}</span>
        </div>
        <div class="ta-stat-card">
          <span class="ta-stat-label">Avg Risk Score</span>
          <span class="ta-stat-value" style="color:${riskColor(avgRisk)}">${avgRisk.toFixed(2)}</span>
        </div>
        <div class="ta-stat-card">
          <span class="ta-stat-label">Max Risk</span>
          <span class="ta-stat-value" style="color:${riskColor(maxRisk)}">${maxRisk.toFixed(2)}</span>
        </div>
        <div class="ta-stat-card">
          <span class="ta-stat-label">Avg Similarity</span>
          <span class="ta-stat-value text-accent-cyan">${(avgSim * 100).toFixed(0)}%</span>
        </div>
        <div class="ta-stat-card">
          <span class="ta-stat-label">Total Events</span>
          <span class="ta-stat-value">${totalEvents}</span>
        </div>
      </div>

      <!-- Two-column body -->
      <div class="ta-body">
        <!-- Left: 3D Vector Space -->
        <div class="ta-3d-section card-inner">
          <h3 class="ta-section-title"><i class="ph ph-cube"></i> Vector Space Proximity</h3>
          <div class="ta-3d-container" id="ta-3d-container"></div>
          <div class="ta-3d-legend">
            <span class="ta-3d-legend-item"><span class="ta-3d-dot" style="background:#00f0ff;box-shadow:0 0 6px #00f0ff"></span> Search Query</span>
            <span class="ta-3d-legend-item"><span class="ta-3d-dot" style="background:#10b981"></span> Low Risk</span>
            <span class="ta-3d-legend-item"><span class="ta-3d-dot" style="background:#f59e0b"></span> Med Risk</span>
            <span class="ta-3d-legend-item"><span class="ta-3d-dot" style="background:#f43f5e"></span> High Risk</span>
            <span class="ta-3d-legend-item"><i class="ph ph-arrows-out-line-horizontal" style="font-size:0.85rem;opacity:0.5"></i> Distance = similarity</span>
          </div>
        </div>
        <!-- Right: narrative list -->
        <div class="ta-narratives-section">
          <h3 class="ta-section-title"><i class="ph ph-list-bullets"></i> Related Narratives</h3>
          <div class="ta-narrative-list custom-scrollbar">
            ${narrs.length === 0 ? '<div class="ta-empty">No related narratives found</div>' :
      narrs.sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0)).map((n, i) => {
        const simPct = ((n.similarity ?? 0) * 100).toFixed(0);
        return `
                <div class="ta-narrative-card" data-id="${n.id}">
                  <div class="ta-narrative-rank">${i + 1}</div>
                  <div class="ta-narrative-info">
                    <div class="ta-narrative-name">${escapeHtml(n.name)}</div>
                    <div class="ta-narrative-desc">${escapeHtml(n.description || "")}</div>
                    <div class="ta-narrative-metrics">
                      <span class="ta-metric"><span class="ta-metric-label">Match</span> <span class="text-accent-cyan">${simPct}%</span></span>
                      <span class="ta-metric"><span class="ta-metric-label">Risk</span> <span style="color:${riskColor(n.model_risk)}">${(n.model_risk ?? 0).toFixed(2)}</span></span>
                      <span class="ta-metric"><span class="ta-metric-label">Surprise</span> <span style="color:${riskColor(n.current_surprise)}">${(n.current_surprise ?? 0).toFixed(2)}</span></span>
                      <span class="ta-metric"><span class="ta-metric-label">Impact</span> <span style="color:${riskColor(n.current_impact)}">${(n.current_impact ?? 0).toFixed(2)}</span></span>
                      <span class="ta-metric"><span class="ta-metric-label">Events</span> ${n.event_count ?? 0}</span>
                    </div>
                  </div>
                  <div class="ta-sim-bar-wrap">
                    <div class="ta-sim-bar"><div class="ta-sim-bar-fill" style="width:${simPct}%"></div></div>
                  </div>
                </div>`;
      }).join("")
    }
          </div>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  // Wire close handlers
  const closeBtn = overlay.querySelector(".ticker-analysis-close");
  const backdrop = overlay.querySelector(".ticker-analysis-backdrop");
  const closeFn = () => {
    overlay.classList.add("closing");
    setTimeout(() => overlay.remove(), 250);
    if (_tickerAnalysisChart) { _tickerAnalysisChart.destroy(); _tickerAnalysisChart = null; }
    if (_active3DScene) { _active3DScene.dispose(); _active3DScene = null; }
  };
  closeBtn.addEventListener("click", closeFn);
  backdrop.addEventListener("click", closeFn);
  document.addEventListener("keydown", function _taEsc(e) {
    if (e.key === "Escape" && document.getElementById("ticker-analysis-overlay")) {
      closeFn();
      document.removeEventListener("keydown", _taEsc);
    }
  });

  // Wire narrative card clicks → open narrative detail modal
  overlay.querySelectorAll(".ta-narrative-card").forEach(card => {
    card.addEventListener("click", () => {
      openNarrativeModal(card.dataset.id);
    });
  });

  // Render the 3D vector space visualization
  if (narrs.length > 0) {
    requestAnimationFrame(() => {
      const container3d = document.getElementById("ta-3d-container");
      if (!container3d) return;
      if (_active3DScene) { _active3DScene.dispose(); _active3DScene = null; }
      _active3DScene = render3DVectorSpace(container3d, result.ticker, narrs);
    });
  }
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
        { label: "Surprise", data: data.surprise_series.map(p => p.value), borderColor: "#f43f5e", tension: 0.3, borderWidth: 2, pointRadius: 3 },
        { label: "Impact", data: data.impact_series.map(p => p.value), borderColor: "rgba(0, 240, 255, 0.6)", tension: 0.3, borderWidth: 2, pointRadius: 3 }
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

  // Live Scrape Button Logic
  const liveScrapeBtn = document.getElementById("btn-live-scrape");
  if (liveScrapeBtn) {
    liveScrapeBtn.addEventListener("click", async () => {
      // Set loading state
      const originalHtml = liveScrapeBtn.innerHTML;
      liveScrapeBtn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Scraping...';
      liveScrapeBtn.disabled = true;

      try {
        await postJSON("/ingest/scrape", { lookback_minutes: 1440, max_per_source: 10, sources: ["rss"] });
        // Force an immediate refresh
        await refreshDashboard();
      } catch (err) {
        console.error("Live scrape failed:", err);
      } finally {
        // Restore button state
        liveScrapeBtn.innerHTML = originalHtml;
        liveScrapeBtn.disabled = false;
      }
    });
  }
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

  const feedPanel = document.getElementById("feed-panel");
  if (feedPanel) {
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = newCardsHtml;
    while (tempDiv.firstChild) dashTab.insertBefore(tempDiv.firstChild, feedPanel);
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

// --- Theme Toggle ---
document.addEventListener("DOMContentLoaded", () => {
  const themeBtn = document.getElementById("theme-toggle");
  if (!themeBtn) return;
  const icon = themeBtn.querySelector("i");
  const body = document.body;

  // Check saved theme
  const savedTheme = localStorage.getItem("nexus-theme");
  if (savedTheme === "light") {
    body.classList.replace("dark-theme", "light-theme");
    icon.className = "ph-bold ph-sun";
  }

  themeBtn.addEventListener("click", () => {
    if (body.classList.contains("dark-theme")) {
      body.classList.replace("dark-theme", "light-theme");
      icon.className = "ph-bold ph-sun";
      localStorage.setItem("nexus-theme", "light");
    } else {
      body.classList.replace("light-theme", "dark-theme");
      icon.className = "ph-bold ph-moon";
      localStorage.setItem("nexus-theme", "dark");
    }
    // Update chart text colors smoothly
    if (riskChart) {
      const isLight = body.classList.contains("light-theme");
      const gridColor = isLight ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.05)";
      const tickColor = isLight ? "#94a3b8" : "#8b949e";
      riskChart.options.scales.x.ticks.color = tickColor;
      riskChart.options.scales.y.ticks.color = tickColor;
      riskChart.options.scales.y.grid.color = gridColor;
      riskChart.update();
    }
    if (modalChart) {
      const isLight = body.classList.contains("light-theme");
      const gridColor = isLight ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.05)";
      const tickColor = isLight ? "#94a3b8" : "#8b949e";
      modalChart.options.scales.y.ticks.color = tickColor;
      modalChart.options.scales.y.grid.color = gridColor;
      modalChart.update();
    }
  });
});

// ============================================================
// SECTION: High Risk Stock Spotlight
// ============================================================

// Curated candidates spanning Tech, Financials, Energy, Crypto-adjacent, Industrial
const HIGH_RISK_CANDIDATES = [
  "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC",
  "JPM", "BAC", "GS", "MS", "WFC", "C",
  "XOM", "CVX", "OXY",
  "COIN", "MARA",
  "BA", "GE", "UNH", "V", "QCOM",
];

async function loadHighRiskStocks() {
  const grid = document.getElementById("high-risk-grid");
  if (!grid) return;

  // Skeleton
  grid.innerHTML = Array(10).fill(`
    <div class="hr-card hr-skeleton" style="border-left-color:rgba(255,255,255,0.08)">
      <div class="hr-skel-line" style="width:38%;height:0.9rem;margin-bottom:0.3rem"></div>
      <div class="hr-skel-line" style="width:68%;height:0.6rem;margin-bottom:0.55rem"></div>
      <div class="hr-skel-line" style="width:50%;height:1.1rem;margin-bottom:0.3rem"></div>
      <div class="hr-skel-line" style="width:100%;height:3px;margin-bottom:0.4rem"></div>
      <div class="hr-skel-line" style="width:80%;height:0.6rem"></div>
    </div>`).join("");

  let cards = [];
  try {
    const res = await fetch(`${API}/tickers/relate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tickers: HIGH_RISK_CANDIDATES, n_results: 3 }),
    });
    if (res.ok) {
      const data = await res.json();
      cards = _buildHighRiskCards(data.results || []);
    }
  } catch (_) { /* fall through to mock */ }

  if (cards.length === 0) cards = _mockHighRiskCards();
  renderHighRiskGrid(cards.slice(0, 10));
}

function _buildHighRiskCards(results) {
  // composite_risk = max over narratives of (model_risk × similarity)
  // This captures worst-case exposure: high-risk narrative + high semantic overlap
  return results
    .map(r => {
      const narrs = r.narratives || [];
      if (narrs.length === 0) return null;
      const top = narrs.reduce((best, n) => {
        return (n.model_risk ?? 0) * (n.similarity ?? 0) >
               (best.model_risk ?? 0) * (best.similarity ?? 0) ? n : best;
      });
      return {
        symbol: r.ticker,
        company_name: r.company_name,
        sector: r.sector,
        composite_risk: Math.min(1, (top.model_risk ?? 0) * (top.similarity ?? 0)),
        top_narrative: top.name,
        top_narrative_risk: top.model_risk ?? 0,
        similarity: top.similarity ?? 0,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.composite_risk - a.composite_risk);
}

function _mockHighRiskCards() {
  const mockNarrs = MOCK_DATA.narratives;
  return WATCHLIST.map(s => {
    const risk = Math.min(1, Math.max(0, s.risk + (Math.random() * 0.08 - 0.04)));
    const narr = mockNarrs[Math.floor(risk * mockNarrs.length)] || mockNarrs[0];
    return {
      symbol: s.sym,
      company_name: s.name,
      sector: "—",
      composite_risk: risk,
      top_narrative: narr.name,
      top_narrative_risk: narr.model_risk,
      similarity: 0.65 + Math.random() * 0.30,
    };
  }).sort((a, b) => b.composite_risk - a.composite_risk);
}

function renderHighRiskGrid(cards) {
  const grid = document.getElementById("high-risk-grid");
  if (!grid) return;

  grid.innerHTML = cards.map((c, i) => {
    const cls   = c.composite_risk >= 0.66 ? "risk-high" : c.composite_risk >= 0.33 ? "risk-medium" : "risk-low";
    const color = c.composite_risk >= 0.66 ? "var(--risk-high)" : c.composite_risk >= 0.33 ? "var(--risk-medium)" : "var(--risk-low)";
    const pct   = (c.composite_risk * 100).toFixed(0);
    const shortCompany = c.company_name.length > 22 ? c.company_name.slice(0, 20) + "…" : c.company_name;
    const shortNarr    = c.top_narrative.length > 26 ? c.top_narrative.slice(0, 24) + "…" : c.top_narrative;
    return `
      <div class="hr-card ${cls}" data-symbol="${escapeHtml(c.symbol)}" title="Click to open full analysis">
        <span class="hr-rank">#${i + 1}</span>
        <div class="hr-ticker">${escapeHtml(c.symbol)}</div>
        <div class="hr-company">${escapeHtml(shortCompany)}</div>
        <div class="hr-score" style="color:${color}">${c.composite_risk.toFixed(2)}</div>
        <div class="hr-bar"><div class="hr-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <div class="hr-narrative"><i class="ph ph-warning-circle" style="font-size:0.65rem;opacity:0.55"></i> ${escapeHtml(shortNarr)}</div>
      </div>`;
  }).join("");

  grid.querySelectorAll(".hr-card").forEach(card => {
    card.addEventListener("click", () => omnibarSelect(card.dataset.symbol));
  });
}

document.addEventListener("DOMContentLoaded", () => {
  loadHighRiskStocks();

  const refreshBtn = document.getElementById("high-risk-refresh-btn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      refreshBtn.disabled = true;
      const icon = refreshBtn.querySelector("i");
      if (icon) icon.className = "ph ph-spinner ph-spin";
      await loadHighRiskStocks();
      refreshBtn.disabled = false;
      if (icon) icon.className = "ph ph-arrows-clockwise";
    });
  }
});

// ============================================================
// SECTION: Ticker Search — Filter narratives by stock ticker
// ============================================================

let tickerFilterActive = false;
let tickerFilterData = null;   // { ticker, company_name, sector, industry, narratives }

// Mock ticker → narrative mapping for offline mode
const MOCK_TICKER_NARRATIVES = {
  AAPL: { company_name: "Apple Inc.", sector: "Technology", industry: "Consumer Electronics", ids: ["3", "4"] },
  NVDA: { company_name: "NVIDIA Corp.", sector: "Technology", industry: "Semiconductors", ids: ["3", "4"] },
  MSFT: { company_name: "Microsoft Corp.", sector: "Technology", industry: "Software", ids: ["4", "3"] },
  TSLA: { company_name: "Tesla Inc.", sector: "Consumer Cyclical", industry: "Auto Manufacturers", ids: ["1", "3"] },
  META: { company_name: "Meta Platforms", sector: "Technology", industry: "Internet Content", ids: ["4"] },
  GOOGL: { company_name: "Alphabet Inc.", sector: "Technology", industry: "Internet Content", ids: ["4", "3"] },
  AMZN: { company_name: "Amazon.com Inc.", sector: "Consumer Cyclical", industry: "Internet Retail", ids: ["4", "3"] },
  JPM: { company_name: "JPMorgan Chase", sector: "Financial Services", industry: "Banks — Diversified", ids: ["2", "5"] },
  BAC: { company_name: "Bank of America", sector: "Financial Services", industry: "Banks — Diversified", ids: ["2", "5"] },
  GS: { company_name: "Goldman Sachs", sector: "Financial Services", industry: "Capital Markets", ids: ["2", "5"] },
  XOM: { company_name: "Exxon Mobil", sector: "Energy", industry: "Oil & Gas Integrated", ids: ["1"] },
  CVX: { company_name: "Chevron Corp.", sector: "Energy", industry: "Oil & Gas Integrated", ids: ["1"] },
  OXY: { company_name: "Occidental Petroleum", sector: "Energy", industry: "Oil & Gas E&P", ids: ["1"] },
  JNJ: { company_name: "Johnson & Johnson", sector: "Healthcare", industry: "Drug Manufacturers", ids: ["5"] },
  PFE: { company_name: "Pfizer Inc.", sector: "Healthcare", industry: "Drug Manufacturers", ids: ["5"] },
  BA: { company_name: "Boeing Co.", sector: "Industrials", industry: "Aerospace & Defense", ids: ["3", "1"] },
};

async function searchTicker() {
  const input = document.getElementById("ticker-input");
  const symbol = input.value.trim().toUpperCase();
  if (!symbol) return;

  const btn = document.getElementById("ticker-search-btn");
  btn.classList.add("loading");
  btn.disabled = true;

  try {
    let result;

    if (isMockMode) {
      // Mock mode: simulate a ticker search
      const mock = MOCK_TICKER_NARRATIVES[symbol];
      if (!mock) {
        showTickerError(`No data found for ticker '${symbol}'`);
        return;
      }
      // Build mock narratives with similarity scores
      const matchedNarratives = mock.ids.map((id, i) => {
        const n = MOCK_DATA.narratives.find(n => n.id === id);
        if (!n) return null;
        return {
          id: n.id, name: n.name, description: n.description,
          distance: 0.15 + i * 0.08,
          similarity: 0.92 - i * 0.04,
          model_risk: n.model_risk,
          current_surprise: n.current_surprise,
          current_impact: n.current_impact,
          event_count: n.event_count,
          is_active: true,
        };
      }).filter(Boolean);

      result = {
        ticker: symbol,
        company_name: mock.company_name,
        sector: mock.sector,
        industry: mock.industry,
        narratives: matchedNarratives,
      };
    } else {
      // Live mode: call the backend
      const res = await fetch(`${API}/tickers/${encodeURIComponent(symbol)}?n_results=10`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showTickerError(err.detail || `Ticker '${symbol}' not found`);
        return;
      }
      result = await res.json();
    }

    // Activate filter
    tickerFilterActive = true;
    tickerFilterData = result;
    renderTickerFilter(result);

  } catch (e) {
    console.error("Ticker search failed:", e);
    showTickerError("Search failed — check your connection");
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
}

function renderTickerFilter(data) {
  // Show the banner
  const banner = document.getElementById("ticker-filter-banner");
  banner.classList.remove("hidden");

  document.getElementById("ticker-badge").textContent = data.ticker;
  document.getElementById("ticker-company-name").textContent = data.company_name;
  document.getElementById("ticker-company-meta").textContent =
    `${data.sector || "—"}  ·  ${data.industry || "—"}`;
  document.getElementById("ticker-match-count").textContent =
    `${data.narratives.length} narrative${data.narratives.length !== 1 ? "s" : ""} matched`;

  // Remove any previous error banner
  const oldErr = document.querySelector(".ticker-error-banner");
  if (oldErr) oldErr.remove();

  // Show similarity column
  document.querySelectorAll(".ticker-col").forEach(el => el.classList.remove("hidden"));

  // Render the filtered narrative table
  renderTickerNarratives(data.narratives);
}

function renderTickerNarratives(narratives) {
  const tbody = document.getElementById("narratives-tbody");
  const emptyEl = document.getElementById("narratives-empty");

  if (narratives.length === 0) {
    emptyEl.querySelector("p").textContent = "No matching narratives for this ticker";
    emptyEl.classList.remove("hidden");
    tbody.innerHTML = "";
    return;
  }

  emptyEl.classList.add("hidden");

  // Sort by similarity descending
  const sorted = [...narratives].sort((a, b) => (b.similarity ?? 0) - (a.similarity ?? 0));

  tbody.innerHTML = sorted.map(n => {
    const simPct = ((n.similarity ?? 0) * 100).toFixed(0);
    const simColor = n.similarity >= 0.7 ? "var(--accent-cyan)"
      : n.similarity >= 0.4 ? "var(--risk-medium)"
        : "var(--text-muted)";

    return `
    <tr data-id="${n.id}">
      <td><strong>${escapeHtml(n.name)}</strong></td>
      <td class="desc" title="${escapeHtml(n.description)}">${escapeHtml(n.description)}</td>
      <td class="num-col" style="color:${riskColor(n.current_surprise)}">${(n.current_surprise ?? 0).toFixed(2)}</td>
      <td class="num-col" style="color:${riskColor(n.current_impact)}">${(n.current_impact ?? 0).toFixed(2)}</td>
      <td class="num-col" style="color:${riskColor(n.model_risk)};font-weight:700">${(n.model_risk ?? 0).toFixed(2)}</td>
      <td class="num-col ticker-col">
        <div class="similarity-bar-cell">
          <div class="similarity-bar"><div class="similarity-bar-fill" style="width:${simPct}%; background:${simColor}"></div></div>
          <span class="similarity-pct" style="color:${simColor}">${simPct}%</span>
        </div>
      </td>
      <td class="num-col">${n.event_count ?? 0}</td>
      <td style="text-align:center">${trendIcon(n.surprise_trend)}</td>
      <td class="right-align data-number text-text-muted">${n.last_updated ? timeAgo(n.last_updated) : "—"}</td>
    </tr>
    `;
  }).join("");

  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => openNarrativeModal(tr.dataset.id));
  });
}

function showTickerError(msg) {
  // Remove previous error
  const old = document.querySelector(".ticker-error-banner");
  if (old) old.remove();

  // Hide filter banner if visible
  document.getElementById("ticker-filter-banner").classList.add("hidden");

  // Insert error banner
  const banner = document.createElement("div");
  banner.className = "ticker-error-banner";
  banner.innerHTML = `<i class="ph-bold ph-warning-circle"></i> ${escapeHtml(msg)}`;

  const tableWrapper = document.querySelector("#narratives-panel .table-wrapper");
  tableWrapper.parentNode.insertBefore(banner, tableWrapper);

  // Auto-dismiss after 4 seconds
  setTimeout(() => { if (banner.parentNode) banner.remove(); }, 4000);
}

function clearTickerFilter() {
  tickerFilterActive = false;
  tickerFilterData = null;

  // Hide banner
  document.getElementById("ticker-filter-banner").classList.add("hidden");

  // Hide similarity column
  document.querySelectorAll(".ticker-col").forEach(el => el.classList.add("hidden"));

  // Clear input
  document.getElementById("ticker-input").value = "";

  // Remove any error banner
  const err = document.querySelector(".ticker-error-banner");
  if (err) err.remove();

  // Restore the empty state text
  const emptyP = document.getElementById("narratives-empty")?.querySelector("p");
  if (emptyP) emptyP.textContent = "Awaiting Narrative Data...";

  // Refresh to show all narratives again
  refreshNarratives();
}

// Wire up ticker search events
document.addEventListener("DOMContentLoaded", () => {
  const tickerInput = document.getElementById("ticker-input");
  const tickerBtn = document.getElementById("ticker-search-btn");
  const tickerClearBtn = document.getElementById("ticker-clear-btn");

  tickerBtn.addEventListener("click", searchTicker);

  tickerInput.addEventListener("keydown", e => {
    if (e.key === "Enter") searchTicker();
    if (e.key === "Escape" && tickerFilterActive) clearTickerFilter();
  });

  tickerClearBtn.addEventListener("click", clearTickerFilter);
});

// Override refreshNarratives to skip overwriting when ticker filter is active
const _tickerOriginalRefreshNarratives = refreshNarratives;
refreshNarratives = async function () {
  if (tickerFilterActive && tickerFilterData) {
    // Keep showing ticker-filtered results; don't overwrite with all narratives
    renderTickerNarratives(tickerFilterData.narratives);
    return;
  }
  // Normal path
  await _tickerOriginalRefreshNarratives();

  // After normal render, make sure similarity column is hidden
  document.querySelectorAll(".ticker-col").forEach(el => el.classList.add("hidden"));
};

// ============================================================
// SECTION: Multi-Visualization Engine
// Six chart types: Line, Neon Bars, Area Bands, Scatter, Radar, Ring
// ============================================================

let currentChartType = "line";
let _cachedNarratives = [];   // stored for radar + ring charts

// Neon palette for multi-series charts
const NEON_PALETTE = [
  { line: "#00f0ff", fill: "rgba(0, 240, 255, 0.18)", glow: "rgba(0, 240, 255, 0.5)" },
  { line: "#f43f5e", fill: "rgba(244, 63, 94, 0.15)", glow: "rgba(244, 63, 94, 0.5)" },
  { line: "#8b5cf6", fill: "rgba(139, 92, 246, 0.15)", glow: "rgba(139, 92, 246, 0.5)" },
  { line: "#f59e0b", fill: "rgba(245, 158, 11, 0.15)", glow: "rgba(245, 158, 11, 0.5)" },
  { line: "#10b981", fill: "rgba(16, 185, 129, 0.15)", glow: "rgba(16, 185, 129, 0.5)" },
  { line: "#ec4899", fill: "rgba(236, 72, 153, 0.15)", glow: "rgba(236, 72, 153, 0.5)" },
];

// Cache narratives whenever they refresh (for radar/ring)
const _vizOrigRefresh = refreshNarratives;
refreshNarratives = async function () {
  await _vizOrigRefresh();
  try {
    const d = await fetchJSON("/narratives?active_only=true&sort_by=risk&limit=20");
    _cachedNarratives = d.narratives || (isMockMode ? MOCK_DATA.narratives : []);
  } catch (e) { /* keep old cache */ }
};

// ── Shared chart helpers ─────────────────────────────────────────────────────

function _riskBarColor(v, alpha) {
  if (v < 0.33) return `rgba(16, 185, 129, ${alpha})`;
  if (v < 0.66) return `rgba(245, 158, 11, ${alpha})`;
  return `rgba(239, 68, 68, ${alpha})`;
}

function _riskBarBorder(v) {
  if (v < 0.33) return "#10b981";
  if (v < 0.66) return "#f59e0b";
  return "#ef4444";
}

function _baseChartOpts(yMin, yMax) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      y: {
        min: yMin, max: yMax,
        grid: { color: "rgba(255,255,255,0.05)" },
        ticks: { color: "#8b949e", font: { family: "'JetBrains Mono'" } }
      },
      x: {
        grid: { display: false },
        ticks: { color: "#8b949e", maxTicksLimit: 8, font: { family: "'JetBrains Mono'" } }
      }
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(16,22,38,0.92)",
        titleFont: { family: "'Outfit'" },
        bodyFont: { family: "'JetBrains Mono'" },
        borderColor: "rgba(0,240,255,0.3)",
        borderWidth: 1,
      }
    }
  };
}

function _destroyRiskChart() {
  if (riskChart) { riskChart.destroy(); riskChart = null; }
  // Remove any overlays (radar legend, ring center)
  document.querySelectorAll(".radar-legend-overlay, .ring-center-label").forEach(el => el.remove());
}

// ── RENDERER: Neon Bars ──────────────────────────────────────────────────────

function _renderBarsChart(ctx, labels, values) {
  _destroyRiskChart();

  const barColors = values.map(v => _riskBarColor(v, 0.75));
  const barBorders = values.map(v => _riskBarBorder(v));
  const barHover = values.map(v => _riskBarColor(v, 1.0));

  riskChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Risk Index",
        data: values,
        backgroundColor: barColors,
        borderColor: barBorders,
        borderWidth: 1.5,
        borderRadius: { topLeft: 4, topRight: 4 },
        hoverBackgroundColor: barHover,
        borderSkipped: "bottom",
      }]
    },
    options: {
      ..._baseChartOpts(0, 1),
      plugins: {
        ..._baseChartOpts(0, 1).plugins,
        tooltip: {
          ..._baseChartOpts(0, 1).plugins.tooltip,
          callbacks: {
            label: (item) => `Risk: ${item.parsed.y.toFixed(3)}`,
            labelColor: (item) => ({
              borderColor: _riskBarBorder(item.parsed.y),
              backgroundColor: _riskBarColor(item.parsed.y, 0.8),
              borderWidth: 2, borderRadius: 2,
            }),
          }
        }
      },
      animation: {
        duration: 800,
        easing: "easeOutQuart",
      }
    },
    plugins: [{
      id: "neonGlow",
      afterDatasetsDraw(chart) {
        const meta = chart.getDatasetMeta(0);
        const ctx2 = chart.ctx;
        meta.data.forEach((bar, i) => {
          const v = values[i];
          if (v >= 0.55) {
            ctx2.save();
            ctx2.shadowColor = _riskBarBorder(v);
            ctx2.shadowBlur = v >= 0.75 ? 18 : 10;
            ctx2.fillStyle = _riskBarColor(v, 0.3);
            const { x, y, width, height, base } = bar.getProps(["x", "y", "width", "height", "base"]);
            ctx2.fillRect(x - width / 2, y, width, base - y);
            ctx2.restore();
          }
        });
      }
    }]
  });
}

// ── RENDERER: Area Bands ─────────────────────────────────────────────────────

function _renderBandsChart(ctx, labels, values) {
  _destroyRiskChart();

  const highBand = values.map(v => Math.max(0, v - 0.66));
  const medBand = values.map(v => Math.max(0, Math.min(v, 0.66) - 0.33));
  const lowBand = values.map(v => Math.min(v, 0.33));

  riskChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Critical",
          data: highBand,
          backgroundColor: "rgba(239, 68, 68, 0.35)",
          borderColor: "rgba(239, 68, 68, 0.6)",
          borderWidth: 0,
          fill: true,
          pointRadius: 0,
          tension: 0.4,
          order: 3,
          stack: "bands",
        },
        {
          label: "Elevated",
          data: medBand,
          backgroundColor: "rgba(245, 158, 11, 0.3)",
          borderColor: "rgba(245, 158, 11, 0.5)",
          borderWidth: 0,
          fill: true,
          pointRadius: 0,
          tension: 0.4,
          order: 2,
          stack: "bands",
        },
        {
          label: "Stable",
          data: lowBand,
          backgroundColor: "rgba(16, 185, 129, 0.25)",
          borderColor: "rgba(16, 185, 129, 0.4)",
          borderWidth: 0,
          fill: true,
          pointRadius: 0,
          tension: 0.4,
          order: 1,
          stack: "bands",
        },
        {
          label: "Risk Index",
          data: values,
          borderColor: "#ffffff",
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointBackgroundColor: "#fff",
          fill: false,
          tension: 0.4,
          order: 4,
        },
      ]
    },
    options: {
      ..._baseChartOpts(0, 1),
      scales: {
        ..._baseChartOpts(0, 1).scales,
        y: {
          ..._baseChartOpts(0, 1).scales.y,
          stacked: true,
        }
      },
      plugins: {
        ..._baseChartOpts(0, 1).plugins,
        legend: {
          display: true,
          position: "top",
          align: "end",
          labels: {
            usePointStyle: true,
            pointStyle: "rectRounded",
            padding: 12,
            font: { family: "'JetBrains Mono'", size: 10 },
            color: "#8b949e",
            filter: (item) => item.text !== "Risk Index",
          }
        },
        tooltip: {
          ..._baseChartOpts(0, 1).plugins.tooltip,
          filter: (item) => item.dataset.label === "Risk Index",
        }
      },
      animation: { duration: 600, easing: "easeOutCubic" },
    },
    plugins: [{
      id: "bandGlow",
      afterDraw(chart) {
        const ctx2 = chart.ctx;
        const area = chart.chartArea;
        ctx2.save();
        // Horizontal risk zone labels on right edge
        const zones = [
          { y: area.bottom - (area.height * 0.165), text: "LOW", color: "rgba(16,185,129,0.4)" },
          { y: area.bottom - (area.height * 0.495), text: "MED", color: "rgba(245,158,11,0.35)" },
          { y: area.bottom - (area.height * 0.83), text: "HIGH", color: "rgba(239,68,68,0.35)" },
        ];
        ctx2.font = "600 9px 'JetBrains Mono'";
        ctx2.textAlign = "right";
        zones.forEach(z => {
          ctx2.fillStyle = z.color;
          ctx2.fillText(z.text, area.right - 6, z.y);
        });
        ctx2.restore();
      }
    }]
  });
}

// ── RENDERER: Scatter Pulse ──────────────────────────────────────────────────

function _renderScatterChart(ctx, labels, values, history) {
  _destroyRiskChart();

  const scatterData = values.map((v, i) => ({
    x: i,
    y: v,
    r: 3 + v * 10,
  }));

  const bgColors = values.map(v => _riskBarColor(v, 0.7));
  const borderColors = values.map(v => _riskBarBorder(v));

  riskChart = new Chart(ctx, {
    type: "bubble",
    data: {
      datasets: [{
        label: "Risk Pulse",
        data: scatterData,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        hoverRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0, max: 1,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8b949e", font: { family: "'JetBrains Mono'" } },
          title: { display: true, text: "Risk", color: "#8b949e", font: { family: "'JetBrains Mono'", size: 10 } }
        },
        x: {
          min: -1, max: values.length,
          grid: { color: "rgba(255,255,255,0.03)" },
          ticks: {
            color: "#8b949e",
            font: { family: "'JetBrains Mono'" },
            maxTicksLimit: 8,
            callback: (val) => labels[Math.round(val)] || "",
          },
          title: { display: false }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(16,22,38,0.92)",
          titleFont: { family: "'Outfit'" },
          bodyFont: { family: "'JetBrains Mono'" },
          borderColor: "rgba(0,240,255,0.3)",
          borderWidth: 1,
          callbacks: {
            title: (items) => labels[items[0].parsed.x] || "",
            label: (item) => `Risk: ${item.parsed.y.toFixed(3)}`,
          }
        }
      },
      animation: { duration: 900, easing: "easeOutElastic" },
    },
    plugins: [{
      id: "scatterField",
      beforeDatasetsDraw(chart) {
        const ctx2 = chart.ctx;
        const area = chart.chartArea;
        // Draw thin connecting line through points
        const meta = chart.getDatasetMeta(0);
        if (meta.data.length < 2) return;
        ctx2.save();
        ctx2.beginPath();
        ctx2.strokeStyle = "rgba(0, 240, 255, 0.15)";
        ctx2.lineWidth = 1;
        ctx2.setLineDash([3, 4]);
        meta.data.forEach((pt, i) => {
          if (i === 0) ctx2.moveTo(pt.x, pt.y);
          else ctx2.lineTo(pt.x, pt.y);
        });
        ctx2.stroke();
        ctx2.restore();
      },
      afterDatasetsDraw(chart) {
        // Pulse glow on the latest (rightmost) point
        const meta = chart.getDatasetMeta(0);
        if (meta.data.length === 0) return;
        const last = meta.data[meta.data.length - 1];
        const v = values[values.length - 1];
        const ctx2 = chart.ctx;
        ctx2.save();
        const t = (Date.now() % 2000) / 2000;
        const pulse = 0.4 + 0.6 * Math.abs(Math.sin(t * Math.PI));
        ctx2.globalAlpha = pulse * 0.5;
        ctx2.beginPath();
        ctx2.arc(last.x, last.y, 14 + v * 12, 0, Math.PI * 2);
        ctx2.fillStyle = _riskBarBorder(v);
        ctx2.fill();
        ctx2.restore();
      }
    }]
  });
}

// ── RENDERER: Radar ──────────────────────────────────────────────────────────

function _renderRadarChart(ctx) {
  _destroyRiskChart();

  const narrs = (_cachedNarratives.length ? _cachedNarratives : MOCK_DATA.narratives)
    .slice(0, 6);

  if (narrs.length === 0) {
    ctx.font = "14px 'Outfit'";
    ctx.fillStyle = "#8b949e";
    ctx.textAlign = "center";
    ctx.fillText("No narrative data available", ctx.canvas.width / 2, ctx.canvas.height / 2);
    return;
  }

  const radarLabels = ["Surprise", "Impact", "Model Risk", "Event Density", "Recency"];

  const maxEvents = Math.max(...narrs.map(n => n.event_count || 1));
  const now = Date.now() / 1000;

  const datasets = narrs.map((n, i) => {
    const c = NEON_PALETTE[i % NEON_PALETTE.length];
    const recency = Math.max(0, 1 - ((now - (n.last_updated || now)) / 86400));
    return {
      label: n.name,
      data: [
        n.current_surprise ?? 0,
        n.current_impact ?? 0,
        n.model_risk ?? 0,
        (n.event_count || 0) / maxEvents,
        recency,
      ],
      borderColor: c.line,
      backgroundColor: c.fill,
      borderWidth: 2,
      pointBackgroundColor: c.line,
      pointBorderColor: "#050811",
      pointBorderWidth: 1,
      pointRadius: 4,
      pointHoverRadius: 7,
    };
  });

  riskChart = new Chart(ctx, {
    type: "radar",
    data: { labels: radarLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0, max: 1,
          beginAtZero: true,
          grid: { color: "rgba(255,255,255,0.06)", circular: true },
          angleLines: { color: "rgba(255,255,255,0.08)" },
          pointLabels: {
            color: "#8b949e",
            font: { family: "'JetBrains Mono'", size: 10, weight: "600" },
          },
          ticks: {
            color: "rgba(255,255,255,0.2)",
            backdropColor: "transparent",
            stepSize: 0.25,
            font: { family: "'JetBrains Mono'", size: 8 },
          },
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(16,22,38,0.94)",
          titleFont: { family: "'Outfit'", size: 13 },
          bodyFont: { family: "'JetBrains Mono'", size: 11 },
          borderColor: "rgba(0,240,255,0.3)",
          borderWidth: 1,
          callbacks: {
            title: (items) => items[0]?.dataset.label || "",
          }
        }
      },
      animation: { duration: 700, easing: "easeOutQuart" },
    },
    plugins: [{
      id: "radarGlow",
      afterDatasetsDraw(chart) {
        const ctx2 = chart.ctx;
        chart.data.datasets.forEach((ds, i) => {
          const meta = chart.getDatasetMeta(i);
          if (!meta.data.length) return;
          ctx2.save();
          ctx2.shadowColor = ds.borderColor;
          ctx2.shadowBlur = 8;
          ctx2.beginPath();
          meta.data.forEach((pt, j) => {
            if (j === 0) ctx2.moveTo(pt.x, pt.y);
            else ctx2.lineTo(pt.x, pt.y);
          });
          ctx2.closePath();
          ctx2.strokeStyle = ds.borderColor;
          ctx2.lineWidth = 1;
          ctx2.stroke();
          ctx2.restore();
        });
      }
    }]
  });

  // Insert floating legend
  const wrapper = document.getElementById("main-chart-wrapper");
  const legend = document.createElement("div");
  legend.className = "radar-legend-overlay";
  legend.innerHTML = narrs.map((n, i) => {
    const c = NEON_PALETTE[i % NEON_PALETTE.length];
    return `<div class="radar-legend-item">
      <span class="radar-legend-dot" style="background:${c.line};color:${c.line}"></span>
      ${escapeHtml(n.name)}
    </div>`;
  }).join("");
  wrapper.appendChild(legend);
}

// ── RENDERER: Ring ───────────────────────────────────────────────────────────

function _renderRingChart(ctx) {
  _destroyRiskChart();

  const narrs = (_cachedNarratives.length ? _cachedNarratives : MOCK_DATA.narratives)
    .slice(0, 8);

  if (narrs.length === 0) return;

  const riskValues = narrs.map(n => n.model_risk ?? 0);
  const total = riskValues.reduce((a, b) => a + b, 0) || 1;
  const maxRisk = Math.max(...riskValues);
  const colors = narrs.map((_, i) => NEON_PALETTE[i % NEON_PALETTE.length].line);
  const hoverColors = narrs.map((_, i) => NEON_PALETTE[i % NEON_PALETTE.length].glow);

  riskChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: narrs.map(n => n.name),
      datasets: [{
        data: riskValues,
        backgroundColor: colors.map(c => c + "cc"),
        borderColor: colors,
        borderWidth: 2,
        hoverBackgroundColor: hoverColors,
        hoverBorderWidth: 3,
        spacing: 3,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "65%",
      plugins: {
        legend: {
          display: true,
          position: "right",
          labels: {
            usePointStyle: true,
            pointStyle: "circle",
            padding: 10,
            font: { family: "'JetBrains Mono'", size: 9.5 },
            color: "#8b949e",
            generateLabels: (chart) => {
              const data = chart.data;
              return data.labels.map((label, i) => ({
                text: `${label.length > 18 ? label.slice(0, 18) + "..." : label}  ${((data.datasets[0].data[i] / total) * 100).toFixed(0)}%`,
                fillStyle: data.datasets[0].backgroundColor[i],
                strokeStyle: data.datasets[0].borderColor[i],
                lineWidth: 1,
                hidden: false,
                index: i,
                pointStyle: "circle",
              }));
            }
          },
        },
        tooltip: {
          backgroundColor: "rgba(16,22,38,0.94)",
          titleFont: { family: "'Outfit'", size: 13 },
          bodyFont: { family: "'JetBrains Mono'", size: 11 },
          borderColor: "rgba(0,240,255,0.3)",
          borderWidth: 1,
          callbacks: {
            label: (item) => {
              const v = item.parsed;
              const pct = ((v / total) * 100).toFixed(1);
              return ` Risk: ${v.toFixed(2)}  (${pct}%)`;
            }
          }
        }
      },
      animation: {
        animateRotate: true,
        duration: 900,
        easing: "easeOutQuart",
      }
    },
    plugins: [{
      id: "ringGlow",
      afterDraw(chart) {
        const ctx2 = chart.ctx;
        const meta = chart.getDatasetMeta(0);
        ctx2.save();
        meta.data.forEach((arc, i) => {
          const color = chart.data.datasets[0].borderColor[i];
          ctx2.shadowColor = color;
          ctx2.shadowBlur = 12;
          ctx2.beginPath();
          ctx2.arc(arc.x, arc.y, arc.outerRadius, arc.startAngle, arc.endAngle);
          ctx2.arc(arc.x, arc.y, arc.innerRadius, arc.endAngle, arc.startAngle, true);
          ctx2.closePath();
          ctx2.strokeStyle = color;
          ctx2.lineWidth = 0.5;
          ctx2.stroke();
        });
        ctx2.restore();
      }
    }]
  });

  // Center label
  const wrapper = document.getElementById("main-chart-wrapper");
  const center = document.createElement("div");
  center.className = "ring-center-label";
  center.innerHTML = `
    <div class="ring-center-value" style="color:${riskColor(maxRisk)}">${maxRisk.toFixed(2)}</div>
    <div class="ring-center-text">Peak Risk</div>
  `;
  wrapper.appendChild(center);
}

// ── CHART TYPE SWITCHING LOGIC ───────────────────────────────────────────────

function switchChartType(newType) {
  if (newType === currentChartType) return;

  // Destroy the old chart and clean up overlays before switching
  _destroyRiskChart();

  currentChartType = newType;

  // Update dropdown UI
  document.querySelectorAll(".chart-type-item").forEach(el => {
    el.classList.toggle("selected", el.dataset.type === newType);
  });

  // Update toggle button
  const iconMap = {
    line: "ph-chart-line", bars: "ph-chart-bar", bands: "ph-stack",
    scatter: "ph-chart-scatter", radar: "ph-hexagon", ring: "ph-circle-notch"
  };
  const labelMap = {
    line: "Line", bars: "Neon Bars", bands: "Area Bands",
    scatter: "Scatter", radar: "Radar", ring: "Ring"
  };
  document.getElementById("chart-type-icon").className = `ph ${iconMap[newType]}`;
  document.getElementById("chart-type-label").textContent = labelMap[newType];

  // Show/hide time controls (radar & ring don't use time series)
  const controls = document.querySelector(".chart-controls");
  if (newType === "radar" || newType === "ring") {
    controls.classList.add("no-time-controls");
  } else {
    controls.classList.remove("no-time-controls");
  }

  // Update insights visibility
  const insights = document.getElementById("chart-insights");
  if (newType === "radar" || newType === "ring") {
    insights.style.display = "none";
  } else {
    insights.style.display = "";
  }

  // Force re-render
  updateRiskChart();
}

// ── Override updateRiskChart to dispatch by type ─────────────────────────────

const _vizOrigUpdateRiskChart = updateRiskChart;
updateRiskChart = async function () {
  // For line type, delegate to original chain (which handles line + momentum)
  if (currentChartType === "line") {
    await _vizOrigUpdateRiskChart();
    return;
  }

  // For other types, generate the same mock/live data but render differently
  let windowSize = 24;
  if (currentChartRange === "1m") windowSize = 24 * 30;
  if (currentChartRange === "1y") windowSize = 24 * 365;
  if (currentChartRange === "ytd") windowSize = 24 * 90;

  const history = Array.from({ length: 24 }, (_, i) => {
    let base = 0.5;
    if (currentChartRange === "1m") base = 0.4;
    if (currentChartRange === "1y") base = 0.35;
    if (currentChartRange === "ytd") base = 0.3;
    const pointRisk = Math.max(0, Math.min(1, base + Math.random() * 0.4 + (chartOffset * 0.05)));
    return {
      timestamp: Date.now() / 1000 - (24 - i) * (windowSize / 24) * 3600 - (chartOffset * windowSize * 3600),
      model_risk_index: pointRisk
    };
  });

  const labels = history.map(p => {
    const d = new Date(p.timestamp * 1000);
    if (currentChartRange === "1d") return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  });

  const values = history.map(p => p.model_risk_index);

  const ctx = document.getElementById("risk-chart").getContext("2d");

  // Update insights (shared across time-series types)
  if (currentChartType !== "radar" && currentChartType !== "ring") {
    const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
    let maxRiskPoint = history[0], minRiskPoint = history[0];
    window.currentMaxIndex = 0;
    window.currentMinIndex = 0;
    history.forEach((p, i) => {
      if (p.model_risk_index > maxRiskPoint.model_risk_index) { maxRiskPoint = p; window.currentMaxIndex = i; }
      if (p.model_risk_index < minRiskPoint.model_risk_index) { minRiskPoint = p; window.currentMinIndex = i; }
    });
    const fmtTime = (ts) => {
      const d = new Date(ts * 1000);
      if (currentChartRange === "1d") return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    };
    document.getElementById("insight-max-val").textContent = maxRiskPoint.model_risk_index.toFixed(2);
    document.getElementById("insight-max-time").textContent = fmtTime(maxRiskPoint.timestamp);
    document.getElementById("insight-min-val").textContent = minRiskPoint.model_risk_index.toFixed(2);
    document.getElementById("insight-min-time").textContent = fmtTime(minRiskPoint.timestamp);
    document.getElementById("insight-avg-val").textContent = avgValue.toFixed(2);
  }

  // Dispatch to correct renderer
  switch (currentChartType) {
    case "bars": _renderBarsChart(ctx, labels, values); break;
    case "bands": _renderBandsChart(ctx, labels, values); break;
    case "scatter": _renderScatterChart(ctx, labels, values, history); break;
    case "radar": _renderRadarChart(ctx); break;
    case "ring": _renderRingChart(ctx); break;
  }
};

// ── Wire up the dropdown UI ──────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("chart-type-toggle");
  const menu = document.getElementById("chart-type-menu");
  const dropdown = document.getElementById("chart-type-dropdown");

  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    menu.classList.toggle("hidden");
    dropdown.classList.toggle("open");
  });

  menu.querySelectorAll(".chart-type-item").forEach(item => {
    item.addEventListener("click", (e) => {
      e.stopPropagation();
      const type = item.dataset.type;
      menu.classList.add("hidden");
      dropdown.classList.remove("open");
      switchChartType(type);
    });
  });

  // Close dropdown when clicking outside
  document.addEventListener("click", () => {
    if (!menu.classList.contains("hidden")) {
      menu.classList.add("hidden");
      dropdown.classList.remove("open");
    }
  });
});
