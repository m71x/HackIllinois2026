"""
Story Buffer
============
Thread-safe in-memory holding area for scraped stories that have not yet
been embedded or routed into ChromaDB.

Lifecycle:
    1. scraper.py        → fills the buffer via add_batch()
    2. (user waits)
    3. pipeline route    → drains the buffer via drain(), batch-embeds,
                           then routes each embedding into narrative_engine

The buffer is a module-level singleton so it is shared between the scrape
endpoint and the process endpoint regardless of which worker handles the request.
"""

import threading
import time
from dataclasses import dataclass, field
from services.scraper import RawStory


@dataclass
class BufferStats:
    count: int
    oldest_at: float | None    # unix epoch of oldest buffered story
    newest_at: float | None    # unix epoch of newest buffered story
    preview: list[dict]        # first N stories as {headline, source}


class StoryBuffer:
    def __init__(self):
        self._stories: list[RawStory] = []
        self._lock = threading.Lock()

    def add(self, story: RawStory):
        with self._lock:
            self._stories.append(story)

    def add_batch(self, stories: list[RawStory]):
        with self._lock:
            self._stories.extend(stories)

    def drain(self) -> list[RawStory]:
        """Remove and return all buffered stories atomically."""
        with self._lock:
            out = self._stories[:]
            self._stories.clear()
            return out

    def peek(self, limit: int = 20) -> list[RawStory]:
        """Return up to `limit` stories without removing them."""
        with self._lock:
            return self._stories[:limit]

    def size(self) -> int:
        with self._lock:
            return len(self._stories)

    def clear(self):
        with self._lock:
            self._stories.clear()

    def stats(self, preview_limit: int = 10) -> BufferStats:
        with self._lock:
            if not self._stories:
                return BufferStats(count=0, oldest_at=None, newest_at=None, preview=[])
            return BufferStats(
                count=len(self._stories),
                oldest_at=min(s.published_at for s in self._stories),
                newest_at=max(s.published_at for s in self._stories),
                preview=[
                    {"headline": s.headline[:150], "source": s.source}
                    for s in self._stories[:preview_limit]
                ],
            )


# Module-level singleton — shared across all requests
buffer = StoryBuffer()
