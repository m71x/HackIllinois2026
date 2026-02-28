"""
LLM client — fully local, no external API calls.

All three public functions use fast keyword/heuristic logic:
    label_narrative(story_text)      -> {"name": str, "description": str}
    score_story(...)                 -> {"surprise": float, "impact": float}
    summarize_narrative_context(...) -> str
"""

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
# Public interface
# ---------------------------------------------------------------------------

def label_narrative(story_text: str) -> dict:
    """
    Keyword-based narrative labeler — instant, no network call.

    Finds the dominant financial topic from the taxonomy, then applies a
    directional modifier based on tone keywords to form a 3-5 word label.
    The description is the first line of story_text (usually the headline).
    """
    text = story_text[:600].lower()
    # First non-empty line is the headline
    headline = next((ln.strip() for ln in story_text.splitlines() if ln.strip()), story_text[:120])

    # Pick topic with the most keyword matches
    best_topic = "Market Development"
    best_score = 0
    for keywords, topic in _TOPIC_MAP:
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_topic = topic

    # Directional modifier
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

    name = f"{best_topic} {direction}"
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
