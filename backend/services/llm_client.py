"""
LLM client — calls the Cerebras API for all inference needs.

Set CEREBRAS_API_KEY in .env (or environment) before starting the backend.
Optionally override the model with CEREBRAS_MODEL (default: llama-3.3-70b).

Three public functions:
    label_narrative(story_text)     → {"name": str, "description": str}
    score_story(...)                → {"surprise": float, "impact": float}
    summarize_narrative_context(...)→ str
"""

import json
import time
from cerebras.cloud.sdk import Cerebras
from core.config import settings


# Lazy singleton — created on first use so startup doesn't fail if key is missing.
_client: Cerebras | None = None

def _get_client() -> Cerebras:
    global _client
    if _client is None:
        if not settings.cerebras_api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY is not set. Add it to .env or the environment."
            )
        _client = Cerebras(api_key=settings.cerebras_api_key)
    return _client


def _chat(messages: list[dict], max_tokens: int = 256, temperature: float = 0.1) -> str:
    """
    Call Cerebras chat completions with exponential backoff retry (3 attempts).
    """
    client = _get_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=settings.cerebras_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def _parse_json(raw: str, fallback: dict) -> dict:
    """Parse JSON from LLM output, with fallback for malformed responses."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return fallback


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_LABEL_PROMPT = """You are a financial risk analyst identifying persistent real-world narrative directions.

A "narrative direction" is a broad, ongoing real-world theme that can drive market model breakdown.
Examples:
  - "Energy supply shock" — ongoing constraints on energy availability affecting markets
  - "Regional banking stress" — deteriorating confidence in mid-size banks
  - "China trade policy tightening" — escalating restrictions on China-US trade
  - "Sovereign debt pressure" — rising concern about government default risk

A narrative direction is NOT a specific event. It is the underlying story arc.

Given the news story below, identify which narrative direction it belongs to.

Respond with ONLY valid JSON. No commentary. No markdown fences.

{{"name": "<3 to 6 word label>", "description": "<one sentence describing the persistent narrative direction>"}}

News story:
{story_text}"""

_SCORE_PROMPT = """You are a quantitative financial risk analyst.

Narrative direction: {narrative_description}
{context_block}
Score the following news story on two dimensions:

SURPRISE [0.0–1.0]: How unexpected or regime-breaking is this development within the narrative?
  0.0 = expected continuation already priced in
  0.5 = moderate escalation, partially surprising
  1.0 = sudden shock, reversal, or unprecedented development

IMPACT [0.0–1.0]: How economically significant is this event?
  Event severity (highest → lowest): sovereign default, bank collapse, military conflict,
  sanctions, credit crisis, supply chain shutdown, regulatory ban, commodity shock,
  rate shock, political coup, large-cap earnings, executive departure, minor regulation
  0.0 = negligible market relevance
  0.5 = affects one major sector or mid-cap companies
  1.0 = systemic, multi-sector, global significance

Respond with ONLY valid JSON. No commentary.
{{"surprise": <float 0.0–1.0>, "impact": <float 0.0–1.0>}}

News story:
{story_text}"""

_SCORE_CONTEXT_BLOCK = """
Current narrative state:
  - Surprise so far: {existing_surprise:.2f}
  - Impact so far:   {existing_impact:.2f}
Does this story ESCALATE, CONTINUE, or DE-ESCALATE the narrative?
"""

_CHAT_PROMPT = """You are a real-world market risk analyst with access to a live narrative database.
Answer the user's question using only the narrative data provided below.
Be specific and concise. If no narratives are relevant, say so.

User question: {query}

Relevant narrative directions:
{narratives_block}"""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def label_narrative(story_text: str) -> dict:
    """
    Generate a name and description for a new narrative direction.
    Called only when a story doesn't fit any existing narrative.
    Returns {"name": str, "description": str}.
    """
    prompt = _LABEL_PROMPT.format(story_text=story_text[:1500])
    raw = _chat([{"role": "user", "content": prompt}], max_tokens=256, temperature=0.1)
    return _parse_json(raw, fallback={
        "name": "unclassified narrative",
        "description": story_text[:150],
    })


def score_story(
    story_text: str,
    narrative_description: str,
    existing_surprise: float | None,
    existing_impact: float | None,
) -> dict:
    """
    Score a story for Surprise and Impact within its narrative context.
    Returns {"surprise": float [0,1], "impact": float [0,1]}.
    """
    context_block = ""
    if existing_surprise is not None and existing_impact is not None:
        context_block = _SCORE_CONTEXT_BLOCK.format(
            existing_surprise=existing_surprise,
            existing_impact=existing_impact,
        )

    prompt = _SCORE_PROMPT.format(
        narrative_description=narrative_description,
        context_block=context_block,
        story_text=story_text[:1500],
    )
    raw = _chat([{"role": "user", "content": prompt}], max_tokens=64, temperature=0.1)
    scores = _parse_json(raw, fallback={"surprise": 0.3, "impact": 0.3})
    return {
        "surprise": max(0.0, min(1.0, float(scores.get("surprise", 0.3)))),
        "impact": max(0.0, min(1.0, float(scores.get("impact", 0.3)))),
    }


def summarize_narrative_context(narratives: list[dict], query: str) -> str:
    """
    RAG-style summary: answer a user query using the top-k retrieved narratives.
    Returns a plain-text prose answer.
    """
    lines = []
    for i, n in enumerate(narratives, 1):
        headlines = "; ".join((n.get("recent_headlines") or [])[-3:])
        lines.append(
            f"{i}. [{n['name']}] risk={n.get('model_risk', 'N/A')}, "
            f"surprise={n.get('current_surprise', 'N/A')}, "
            f"impact={n.get('current_impact', 'N/A')}\n"
            f"   {n['description']}\n"
            f"   Recent: {headlines}"
        )

    prompt = _CHAT_PROMPT.format(
        query=query,
        narratives_block="\n\n".join(lines),
    )
    return _chat([{"role": "user", "content": prompt}], max_tokens=512, temperature=0.3)
