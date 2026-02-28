"""
ChromaDB interface for NarrativeDirection objects.

Storage model:
  - Each document in the collection is ONE narrative direction.
  - The embedding stored is the semantic direction vector (not a news story vector).
  - Metadata holds all scalar fields + JSON-serialized time series.

Routing logic (implemented in narrative_engine.py, not here):
  - Query nearest narratives for an incoming story embedding.
  - If best distance < SIMILARITY_THRESHOLD  → update that narrative.
  - If best distance >= SIMILARITY_THRESHOLD → create a new narrative here.
"""

import json
import chromadb
from core.config import settings
from models.narrative import NarrativeDirection, TimeSeriesPoint


_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
collection = _client.get_or_create_collection(
    name=settings.chroma_collection,
    metadata={"hnsw:space": "cosine"},  # cosine distance for semantic similarity
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize(narrative: NarrativeDirection) -> dict:
    """Flatten a NarrativeDirection into Chroma-compatible metadata (strings/ints/floats)."""
    return {
        "name": narrative.name,
        "description": narrative.description,
        "created_at": narrative.created_at,
        "last_updated": narrative.last_updated,
        "event_count": narrative.event_count,
        "surprise_series": json.dumps([p.model_dump() for p in narrative.surprise_series]),
        "impact_series": json.dumps([p.model_dump() for p in narrative.impact_series]),
        "recent_headlines": json.dumps(narrative.recent_headlines),
    }


def _deserialize(id: str, metadata: dict) -> NarrativeDirection:
    """Reconstruct a NarrativeDirection from Chroma metadata."""
    return NarrativeDirection(
        id=id,
        name=metadata["name"],
        description=metadata["description"],
        created_at=metadata["created_at"],
        last_updated=metadata["last_updated"],
        event_count=metadata["event_count"],
        surprise_series=[TimeSeriesPoint(**p) for p in json.loads(metadata["surprise_series"])],
        impact_series=[TimeSeriesPoint(**p) for p in json.loads(metadata["impact_series"])],
        recent_headlines=json.loads(metadata["recent_headlines"]),
    )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def add_narrative(narrative: NarrativeDirection, embedding: list[float]) -> str:
    """Store a new narrative direction. Returns its id."""
    collection.add(
        ids=[narrative.id],
        embeddings=[embedding],
        metadatas=[_serialize(narrative)],
        documents=[narrative.description],  # stored for human-readable Chroma queries
    )
    return narrative.id


def update_narrative(narrative: NarrativeDirection, new_embedding: list[float] = None):
    """
    Update an existing narrative's metadata (time series, headlines, etc.).
    Optionally update its embedding (e.g. rolling average of contributing story vectors).
    """
    kwargs = dict(
        ids=[narrative.id],
        metadatas=[_serialize(narrative)],
        documents=[narrative.description],
    )
    if new_embedding is not None:
        kwargs["embeddings"] = [new_embedding]
    collection.update(**kwargs)


def get_narrative(narrative_id: str) -> NarrativeDirection | None:
    result = collection.get(ids=[narrative_id], include=["metadatas"])
    if not result["ids"]:
        return None
    return _deserialize(result["ids"][0], result["metadatas"][0])


def get_all_narratives() -> list[NarrativeDirection]:
    result = collection.get(include=["metadatas"])
    return [_deserialize(id_, meta) for id_, meta in zip(result["ids"], result["metadatas"])]


def query_nearest(
    embedding: list[float],
    n_results: int = 5,
) -> list[tuple[NarrativeDirection, float]]:
    """
    Find the nearest narrative directions to the given embedding.
    Returns list of (NarrativeDirection, distance) sorted by distance ascending.
    Distance is cosine distance [0, 2]; lower = more similar.
    """
    if collection.count() == 0:
        return []

    n_results = min(n_results, collection.count())
    result = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    narratives = []
    for id_, meta, dist in zip(
        result["ids"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        narratives.append((_deserialize(id_, meta), dist))

    return narratives


def narrative_count() -> int:
    return collection.count()
