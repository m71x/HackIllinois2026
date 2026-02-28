"""
Cerebras LLM client.

Two main responsibilities:
  1. label_narrative()  — given a new story, generate a name + description for
                          the new narrative direction it represents.
  2. score_story()      — given a story and its matched narrative context,
                          return Surprise and Impact scores in [0, 1].
"""

import json
from cerebras.cloud.sdk import Cerebras
from core.config import settings

client = Cerebras(api_key=settings.cerebras_api_key)


def _chat(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=settings.cerebras_model,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def label_narrative(story_text: str) -> dict:
    """
    Generate a name and description for a new narrative direction.
    Returns {"name": str, "description": str}.
    """
    prompt = f"""You are a financial risk analyst. A news story has arrived that does not fit any existing narrative category.
Identify the persistent real-world narrative direction this story belongs to.

A narrative direction is a broad, ongoing theme (e.g. "energy supply shock", "regional banking stress", "China trade policy tightening").
It is NOT the specific event — it is the underlying story direction the event is part of.

Respond with valid JSON only:
{{
  "name": "<short label, 3-6 words>",
  "description": "<one sentence describing the narrative direction>"
}}

News story:
{story_text[:1500]}"""

    raw = _chat([{"role": "user", "content": prompt}])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON block if model added extra text
        start, end = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[start:end])


def score_story(
    story_text: str,
    narrative_description: str,
    existing_surprise: float | None,
    existing_impact: float | None,
) -> dict:
    """
    Score a news story against its matched narrative context.
    Returns {"surprise": float [0,1], "impact": float [0,1]}.

    Surprise: how unexpected/novel this development is within the narrative.
    Impact:   how economically significant this event is.
    """
    context_block = ""
    if existing_surprise is not None:
        context_block = f"""
Current narrative state:
  - Surprise so far: {existing_surprise:.2f}
  - Impact so far:   {existing_impact:.2f}
Consider whether this new story escalates, continues, or de-escalates the narrative."""

    prompt = f"""You are a quantitative financial risk analyst.

Narrative direction: {narrative_description}
{context_block}

Score the following news story on two dimensions:

1. Surprise [0.0 - 1.0]: How unexpected or regime-breaking is this development within the narrative?
   - 0.0 = fully expected continuation, markets have already priced this in
   - 1.0 = sudden shock, reversal, or unprecedented escalation

2. Impact [0.0 - 1.0]: How economically significant is this event?
   - Consider: size of affected economies/companies, number of sectors, severity of event type
   - 0.0 = negligible market relevance
   - 1.0 = systemic, multi-sector, global significance

Respond with valid JSON only:
{{"surprise": <float>, "impact": <float>}}

News story:
{story_text[:1500]}"""

    raw = _chat([{"role": "user", "content": prompt}])
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        scores = json.loads(raw[start:end])

    # Clamp to [0, 1]
    return {
        "surprise": max(0.0, min(1.0, float(scores["surprise"]))),
        "impact": max(0.0, min(1.0, float(scores["impact"]))),
    }


def complete(messages: list[dict]) -> str:
    """Generic completion for the /api/chat route."""
    return _chat(messages)
