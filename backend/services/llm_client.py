"""
LLM client — narrative labeling via Modal LLM with heuristic fallback.

Public API:
    label_narrative(story_text)      -> {"name": str, "description": str}
    score_story(...)                 -> {"surprise": float, "impact": float}
    summarize_narrative_context(...) -> str

label_narrative() tries the Modal-deployed Qwen2.5-0.5B-Instruct Labeler
first (see model/modal_app.py).  If Modal is unavailable it falls back to
the keyword/heuristic approach instantly with no latency penalty.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal Labeler client  (mirrors the pattern in services/embedder.py)
# ---------------------------------------------------------------------------

_modal_labeler   = None   # cached modal.Cls handle
_labeler_offline = False  # set True permanently once Modal is unreachable


def _get_modal_labeler():
    """Return the cached Modal Labeler class handle, or None on failure."""
    global _modal_labeler, _labeler_offline
    if _modal_labeler is None and not _labeler_offline:
        try:
            import modal
            _modal_labeler = modal.Cls.from_name("model-risk-llm", "Labeler")
            logger.info("[llm_client] Connected to Modal Labeler")
        except Exception as exc:
            _labeler_offline = True
            logger.warning("[llm_client] Modal Labeler unavailable (%s), using heuristics", exc)
    return _modal_labeler


# ---------------------------------------------------------------------------
# Topic taxonomy — ordered by specificity (more specific first)
# ---------------------------------------------------------------------------

_TOPIC_MAP: list[tuple[frozenset[str], str]] = [
    (frozenset(["federal reserve", "fomc", "powell", "rate hike", "rate cut",
                "interest rate", "fed funds"]),                              "Monetary Policy"),
    (frozenset(["inflation", "cpi", "pce", "consumer price", "price index",
                "deflation", "stagflation"]),                                "Inflation Dynamics"),
    (frozenset(["oil", "crude", "opec", "petroleum", "lng", "natural gas",
                "energy supply"]),                                           "Energy Markets"),
    (frozenset(["bitcoin", "crypto", "ethereum", "blockchain", "defi",
                "stablecoin", "nft", "web3"]),                              "Crypto Assets"),
    (frozenset(["china", "beijing", "tariff", "trade war", "huawei",
                "xi jinping", "prc"]),                                       "China Trade Tensions"),
    (frozenset(["bank", "banking", "svb", "fdic", "credit", "lender",
                "deposit", "bank run"]),                                     "Banking Sector"),
    (frozenset(["earnings", "revenue", "profit", "eps", "quarterly results",
                "guidance", "outlook"]),                                     "Corporate Earnings"),
    (frozenset(["jobs", "unemployment", "payroll", "labor", "hiring",
                "layoff", "workforce", "nonfarm"]),                         "Labor Market"),
    (frozenset(["housing", "real estate", "mortgage", "home prices",
                "reit", "commercial property"]),                             "Real Estate"),
    (frozenset(["supply chain", "semiconductor", "chip", "shortage",
                "logistics", "reshoring"]),                                  "Supply Chain"),
    (frozenset(["ai", "artificial intelligence", "machine learning",
                "openai", "llm", "generative"]),                            "AI Technology"),
    (frozenset(["bond", "yield", "treasury", "debt", "deficit",
                "sovereign", "t-bill"]),                                     "Sovereign Debt"),
    (frozenset(["recession", "gdp", "growth slowdown", "contraction",
                "economic outlook"]),                                        "Economic Outlook"),
    (frozenset(["sanctions", "russia", "ukraine", "war", "conflict",
                "geopolit", "invasion"]),                                    "Geopolitical Risk"),
    (frozenset(["dollar", "yen", "euro", "forex", "currency",
                "exchange rate", "devaluation"]),                            "Currency Markets"),
    (frozenset(["merger", "acquisition", "ipo", "deal", "buyout",
                "private equity", "m&a"]),                                   "M&A Activity"),
    (frozenset(["sec", "regulation", "regulatory", "policy", "congress",
                "antitrust", "enforcement"]),                                "Regulatory Change"),
    (frozenset(["climate", "esg", "carbon", "green", "renewable",
                "sustainability", "net zero"]),                              "Climate Finance"),
    (frozenset(["stock", "equity", "s&p", "nasdaq", "dow",
                "market rally", "bull", "bear"]),                            "Equity Markets"),
    (frozenset(["commodity", "gold", "silver", "copper", "wheat",
                "agricultural", "futures"]),                                 "Commodities"),
]

_CRISIS_WORDS = frozenset(["collapse", "crisis", "crash", "fail", "bankrupt",
                            "panic", "turmoil", "meltdown", "contagion", "seized"])
_SURGE_WORDS  = frozenset(["surges", "surge", "rally", "soar", "boom",
                            "skyrocket", "jump", "spike", "record high"])
_PRESSURE_WORDS = frozenset(["falls", "drops", "decline", "slump", "plunge",
                              "tumble", "sinks", "contracts", "weakens"])
_RISK_WORDS   = frozenset(["warn", "risk", "threat", "concern", "fear",
                            "caution", "alarm", "uncertainty"])
_TIGHTEN_WORDS = frozenset(["tighten", "hike", "restrict", "sanction",
                             "ban", "curb", "crackdown", "freeze"])

# ---------------------------------------------------------------------------
# Named entity lookup — longest/most-specific keys first to prevent
# partial matches (e.g. "silicon valley bank" before "bank")
# ---------------------------------------------------------------------------

_KNOWN_ENTITIES: list[tuple[str, str]] = [
    ("silicon valley bank", "SVB"),
    ("credit suisse", "Credit Suisse"),
    ("first republic bank", "First Republic"),
    ("first republic", "First Republic"),
    ("signature bank", "Signature Bank"),
    ("federal reserve", "Fed"),
    ("european central bank", "ECB"),
    ("bank of england", "BoE"),
    ("bank of japan", "BoJ"),
    ("people's bank of china", "PBoC"),
    ("goldman sachs", "Goldman Sachs"),
    ("morgan stanley", "Morgan Stanley"),
    ("wells fargo", "Wells Fargo"),
    ("bank of america", "Bank of America"),
    ("jpmorgan chase", "JPMorgan"),
    ("jpmorgan", "JPMorgan"),
    ("white house", "White House"),
    ("10-year treasury", "10Y Treasury"),
    ("s&p 500", "S&P 500"),
    ("openai", "OpenAI"),
    ("alphabet", "Alphabet"),
    ("microsoft", "Microsoft"),
    ("amazon", "Amazon"),
    ("nvidia", "Nvidia"),
    ("tesla", "Tesla"),
    ("google", "Google"),
    ("blackrock", "BlackRock"),
    ("exxonmobil", "ExxonMobil"),
    ("exxon", "ExxonMobil"),
    ("boeing", "Boeing"),
    ("apple", "Apple"),
    ("meta", "Meta"),
    ("fomc", "FOMC"),
    ("opec+", "OPEC+"),
    ("opec", "OPEC"),
    ("powell", "Powell"),
    ("yellen", "Yellen"),
    ("china", "China"),
    ("russia", "Russia"),
    ("ukraine", "Ukraine"),
    ("israel", "Israel"),
    ("iran", "Iran"),
    ("bitcoin", "Bitcoin"),
    ("ethereum", "Ethereum"),
    ("nasdaq", "Nasdaq"),
    ("ftse", "FTSE"),
    ("imf", "IMF"),
    ("sec", "SEC"),
    ("huawei", "Huawei"),
    ("tsmc", "TSMC"),
    ("softbank", "SoftBank"),
    ("aramco", "Saudi Aramco"),
]

# Action phrases — most specific first; matched against headline + first 200 chars
_ACTION_MAP: list[tuple[str, str]] = [
    ("bank run", "Bank Run"),
    ("debt ceiling", "Debt Ceiling"),
    ("rate hike", "Rate Hike"),
    ("rate cut", "Rate Cut"),
    ("rate pause", "Rate Pause"),
    ("output cut", "Output Cut"),
    ("production cut", "Output Cut"),
    ("export ban", "Export Ban"),
    ("import ban", "Import Ban"),
    ("trade war", "Trade War"),
    ("earnings beat", "Earnings Beat"),
    ("earnings miss", "Earnings Miss"),
    ("job cuts", "Job Cuts"),
    ("mass layoff", "Mass Layoffs"),
    ("layoffs", "Layoffs"),
    ("layoff", "Layoffs"),
    ("debt default", "Debt Default"),
    ("default", "Default"),
    ("bankrupt", "Bankruptcy"),
    ("collapse", "Collapse"),
    ("contagion", "Contagion"),
    ("crackdown", "Crackdown"),
    ("sanctions", "Sanctions"),
    ("sanction", "Sanctions"),
    ("tariff hike", "Tariff Hike"),
    ("tariffs", "Tariffs"),
    ("tariff", "Tariffs"),
    ("downgrade", "Downgrade"),
    ("upgrade", "Upgrade"),
    ("ipo", "IPO"),
    ("merger", "Merger"),
    ("acquisition", "Acquisition"),
    ("buyout", "Buyout"),
    ("selloff", "Selloff"),
    ("sell-off", "Selloff"),
    ("market crash", "Market Crash"),
    ("market rally", "Market Rally"),
    ("recession", "Recession"),
    ("slowdown", "Slowdown"),
    ("inflation spike", "Inflation Spike"),
    ("inflation surge", "Inflation Surge"),
    ("inflation", "Inflation"),
    ("supply shortage", "Supply Shortage"),
    ("shortage", "Supply Shortage"),
    ("earnings", "Earnings"),
    ("rate decision", "Rate Decision"),
]

# Short names for topics, used when building entity+topic+direction names
_TOPIC_SHORT: dict[str, str] = {
    "Monetary Policy":      "Rate Policy",
    "Inflation Dynamics":   "Inflation",
    "Energy Markets":       "Energy",
    "Crypto Assets":        "Crypto",
    "China Trade Tensions": "Trade",
    "Banking Sector":       "Banking",
    "Corporate Earnings":   "Earnings",
    "Labor Market":         "Jobs",
    "Real Estate":          "Housing",
    "Supply Chain":         "Supply Chain",
    "AI Technology":        "AI",
    "Sovereign Debt":       "Debt",
    "Economic Outlook":     "Growth",
    "Geopolitical Risk":    "Geopolitics",
    "Currency Markets":     "FX",
    "M&A Activity":         "M&A",
    "Regulatory Change":    "Regulation",
    "Climate Finance":      "ESG",
    "Equity Markets":       "Equities",
    "Commodities":          "Commodities",
    "Market Development":   "Market",
}

_NAME_STOPWORDS = frozenset([
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "but", "with", "as", "is", "are", "was", "were", "be", "been", "by",
    "from", "up", "into", "after", "amid", "over", "new", "its", "their",
    "that", "this", "has", "have", "had", "will", "would", "could", "may",
    "says", "said", "report", "reports", "amid", "after", "despite",
])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def label_narrative(story_text: str) -> dict:
    """
    Label a new narrative direction.

    Tries the Modal Qwen2.5-0.5B-Instruct Labeler first for high-quality,
    entity-specific names (e.g. "SVB Bank Run", "Fed Rate Hike").
    Falls back to _heuristic_label() instantly if Modal is unreachable.
    """
    headline = next(
        (ln.strip() for ln in story_text.splitlines() if ln.strip()),
        story_text[:120],
    )
    labeler = _get_modal_labeler()
    if labeler is not None:
        try:
            result = labeler().label.remote(headline)
            if result and result.get("name"):
                logger.debug("[llm_client] Modal label → %r", result["name"])
                return result
        except Exception as exc:
            logger.warning("[llm_client] Modal label call failed (%s), falling back", exc)

    return _heuristic_label(story_text)


def _heuristic_label(story_text: str) -> dict:
    """
    Keyword-based narrative labeler — instant, no network call.

    Strategy:
      1. Extract dominant topic (keyword cluster) and direction (tone words).
      2. Extract a named entity from the headline (known entity list, then
         proper-noun fallback).
      3. Find the most specific action phrase in the headline.
      4. Compose: "{entity} {action}" > "{entity} {short_topic} {direction}"
         > "{short_topic} {action}" > "{topic} {direction}" (fallback).
    """
    text = story_text[:600].lower()
    headline = next((ln.strip() for ln in story_text.splitlines() if ln.strip()), story_text[:120])
    headline_lower = headline.lower()

    # ── 1. Topic (highest keyword-match score) ──────────────────────────────
    best_topic = "Market Development"
    best_score = 0
    for keywords, topic in _TOPIC_MAP:
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_topic = topic

    # ── 2. Direction ─────────────────────────────────────────────────────────
    if any(kw in text for kw in _CRISIS_WORDS):
        direction = "Crisis"
    elif any(kw in text for kw in _SURGE_WORDS):
        direction = "Surge"
    elif any(kw in text for kw in _PRESSURE_WORDS):
        direction = "Pressure"
    elif any(kw in text for kw in _RISK_WORDS):
        direction = "Risk"
    elif any(kw in text for kw in _TIGHTEN_WORDS):
        direction = "Tightening"
    else:
        direction = "Shift"

    # ── 3. Named entity (headline-anchored) ──────────────────────────────────
    subject: str | None = None
    for entity_key, entity_label in _KNOWN_ENTITIES:
        if entity_key in headline_lower:
            subject = entity_label
            break

    if subject is None:
        # Fallback: grab consecutive Title Case words from the headline
        # (skip position 0 which is often a verb in inverted news sentences)
        words = headline.split()
        proper: list[str] = []
        for i, w in enumerate(words):
            if i == 0:
                continue
            clean = w.strip('.,!?:;"\'()[]/-')
            if (len(clean) >= 3
                    and clean[0].isupper()
                    and not clean.isupper()           # skip ALL-CAPS (handled above)
                    and clean.lower() not in _NAME_STOPWORDS):
                proper.append(clean)
            elif proper:
                break  # stop at first gap
        if proper:
            subject = " ".join(proper[:2])

    # ── 4. Specific action phrase ─────────────────────────────────────────────
    search = headline_lower + " " + text[:200]
    action: str | None = None
    for phrase, label in _ACTION_MAP:
        if phrase in search:
            action = label
            break

    # ── 5. Compose name ───────────────────────────────────────────────────────
    short_topic = _TOPIC_SHORT.get(best_topic, best_topic)

    if subject and action:
        name = f"{subject} {action}"
    elif subject:
        name = f"{subject} {short_topic} {direction}"
    elif action:
        name = f"{short_topic} {action}"
    else:
        name = f"{best_topic} {direction}"

    if len(name) > 52:
        name = name[:49] + "…"

    description = headline[:200] if headline else story_text[:150]
    return {"name": name, "description": description}


def score_story(
    story_text: str,
    narrative_description: str,
    existing_surprise: float | None,
    existing_impact: float | None,
) -> dict:
    """
    Stub kept for backward compatibility with any callers that haven't been
    updated yet. Real scoring uses _heuristic_score() in narrative_engine.py.
    Returns neutral defaults.
    """
    return {"surprise": 0.3, "impact": 0.3}


def summarize_narrative_context(narratives: list[dict], query: str) -> str:
    """
    Template-based context summary — instant, no network call.
    Returns a structured plain-text answer from the narrative data.
    """
    if not narratives:
        return "No relevant narrative directions found for that query."

    lines = [f"Top {len(narratives)} narrative direction(s) related to '{query}':\n"]
    for i, n in enumerate(narratives, 1):
        recent = "; ".join((n.get("recent_headlines") or [])[-2:])
        lines.append(
            f"{i}. {n['name']}\n"
            f"   Risk={n.get('model_risk', 'N/A')}  "
            f"Surprise={n.get('current_surprise', 'N/A')}  "
            f"Impact={n.get('current_impact', 'N/A')}\n"
            f"   {n.get('description', '')}\n"
            f"   Recent: {recent or 'none'}"
        )
    return "\n".join(lines)
