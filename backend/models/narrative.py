from pydantic import BaseModel, Field
from typing import Optional
import uuid
import time


class TimeSeriesPoint(BaseModel):
    timestamp: float  # unix epoch
    value: float


class NarrativeDirection(BaseModel):
    """
    A persistent semantic category stored in ChromaDB.
    Represents a real-world narrative (e.g. "energy supply shock",
    "regional banking stress") discovered from news flow.

    Individual news stories are NOT stored here. A story either
    updates an existing narrative's metrics or triggers creation
    of a new narrative if it doesn't fit any existing one.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str                        # LLM-generated short label
    description: str                 # LLM-generated summary of the narrative direction
    created_at: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)
    event_count: int = 0             # number of news stories that have updated this narrative

    # Time series stored as lists of points (serialized to JSON in Chroma metadata)
    surprise_series: list[TimeSeriesPoint] = []
    impact_series: list[TimeSeriesPoint] = []

    # Rolling context: last N headlines that contributed to this narrative
    recent_headlines: list[str] = []

    @property
    def current_surprise(self) -> Optional[float]:
        return self.surprise_series[-1].value if self.surprise_series else None

    @property
    def current_impact(self) -> Optional[float]:
        return self.impact_series[-1].value if self.impact_series else None

    @property
    def model_risk(self) -> Optional[float]:
        """Composite risk: high when both surprise and impact are high."""
        s = self.current_surprise
        i = self.current_impact
        if s is None or i is None:
            return None
        return (s * i) ** 0.5  # geometric mean keeps it in [0, 1]

    def append_surprise(self, value: float, timestamp: float = None):
        self.surprise_series.append(
            TimeSeriesPoint(timestamp=timestamp or time.time(), value=value)
        )
        self.last_updated = time.time()

    def append_impact(self, value: float, timestamp: float = None):
        self.impact_series.append(
            TimeSeriesPoint(timestamp=timestamp or time.time(), value=value)
        )
        self.last_updated = time.time()

    def add_headline(self, headline: str, max_recent: int = 10):
        self.recent_headlines.append(headline)
        self.recent_headlines = self.recent_headlines[-max_recent:]
        self.event_count += 1
        self.last_updated = time.time()
