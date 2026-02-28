"""
Model ↔ Backend Contract
========================
This file defines the interface that the model team MUST implement in
model/modal_app.py, and that the backend team calls via llm_client.py
and embedder.py.

DO NOT import this file at runtime. It is documentation only.

Deployment
----------
Both classes live in the same Modal app (APP_NAME = "model-risk-llm").
Deploy with:
    modal deploy model/modal_app.py

The backend looks them up by name:
    modal.Cls.lookup("model-risk-llm", "LLM")
    modal.Cls.lookup("model-risk-llm", "Embedder")
"""

from abc import ABC, abstractmethod


class LLMContract(ABC):
    """
    Implemented by:  model/modal_app.py :: class LLM
    Called by:       backend/services/llm_client.py :: _chat()

    GPU:             A10G
    Model:           meta-llama/Llama-3.3-70B-Instruct (via vLLM)
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        """
        Parameters
        ----------
        messages : list of {"role": "user"|"assistant"|"system", "content": str}
        max_tokens : int
            64   for score_story calls      (returns a tiny JSON pair)
            256  for label_narrative calls  (returns name + description)
            512  for chat/summarize calls   (returns prose answer)
        temperature : float
            0.1  for scoring/labeling  (near-deterministic JSON)
            0.3  for chat/summarize    (more fluent prose)

        Returns
        -------
        str
            The assistant's reply as a plain string (no role wrapper).
            Must not include leading/trailing whitespace.
        """
        ...


class EmbedderContract(ABC):
    """
    Implemented by:  model/modal_app.py :: class Embedder
    Called by:       backend/services/embedder.py

    GPU:             T4
    Model:           all-MiniLM-L6-v2 (sentence-transformers)
    Output dim:      384
    Normalization:   L2-normalized (unit vectors, required for cosine distance)
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed a single string.

        Parameters
        ----------
        text : str
            Raw text to embed. Typically "{headline}\\n\\n{body}" for stories,
            or a narrative description string for search queries.

        Returns
        -------
        list[float]
            384-dimensional L2-normalized float vector.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings in one forward pass.

        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        list[list[float]]
            One 384-dim L2-normalized vector per input string,
            in the same order as the input list.
        """
        ...


# ---------------------------------------------------------------------------
# Prompt contracts — what the backend sends, what the model must return
# ---------------------------------------------------------------------------
#
# label_narrative call
# --------------------
# Input:  single user message containing a news story (~1500 chars max)
# Output: valid JSON string — {"name": str, "description": str}
#         name:        3–6 word label for the narrative direction
#         description: one sentence describing the persistent narrative arc
#
# score_story call
# ----------------
# Input:  single user message containing narrative context + a news story
# Output: valid JSON string — {"surprise": float, "impact": float}
#         Both values clamped to [0.0, 1.0]
#         surprise: 0=fully expected, 1=sudden shock/regime break
#         impact:   0=negligible, 1=systemic multi-sector event
#
# summarize_narrative_context call
# --------------------------------
# Input:  single user message with user query + formatted narrative context
# Output: plain prose string answering the user's question
#         No JSON. No markdown. Just text.
