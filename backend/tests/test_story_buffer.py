"""Tests for services/story_buffer.py — thread-safe buffer operations."""

import threading
import time
import pytest
from services.story_buffer import StoryBuffer
from services.scraper import RawStory


def make_story(headline: str = "Test headline", source: str = "test") -> RawStory:
    return RawStory(
        headline=headline,
        body="Test body content.",
        source=source,
        published_at=time.time(),
        url=f"https://example.com/{headline[:10]}",
    )


@pytest.fixture()
def buf() -> StoryBuffer:
    """Fresh buffer per test."""
    return StoryBuffer()


class TestBasicOps:
    def test_starts_empty(self, buf):
        assert buf.size() == 0

    def test_add_single(self, buf):
        buf.add(make_story("Story A"))
        assert buf.size() == 1

    def test_add_batch(self, buf):
        stories = [make_story(f"Story {i}") for i in range(10)]
        buf.add_batch(stories)
        assert buf.size() == 10

    def test_drain_returns_all(self, buf):
        buf.add_batch([make_story(f"S{i}") for i in range(5)])
        drained = buf.drain()
        assert len(drained) == 5

    def test_drain_empties_buffer(self, buf):
        buf.add_batch([make_story("X")] * 3)
        buf.drain()
        assert buf.size() == 0

    def test_drain_empty_buffer(self, buf):
        assert buf.drain() == []

    def test_clear(self, buf):
        buf.add_batch([make_story("X")] * 5)
        buf.clear()
        assert buf.size() == 0

    def test_peek_does_not_remove(self, buf):
        buf.add_batch([make_story(f"S{i}") for i in range(10)])
        peeked = buf.peek(limit=3)
        assert len(peeked) == 3
        assert buf.size() == 10   # still all there

    def test_peek_respects_limit(self, buf):
        buf.add_batch([make_story(f"S{i}") for i in range(20)])
        assert len(buf.peek(limit=5)) == 5

    def test_peek_on_empty(self, buf):
        assert buf.peek() == []


class TestStats:
    def test_empty_stats(self, buf):
        s = buf.stats()
        assert s.count == 0
        assert s.oldest_at is None
        assert s.newest_at is None
        assert s.preview == []

    def test_stats_count(self, buf):
        buf.add_batch([make_story(f"S{i}") for i in range(7)])
        assert buf.stats().count == 7

    def test_stats_oldest_newest(self, buf):
        t_old = 1000.0
        t_new = 2000.0
        buf.add(RawStory("Old", "", "src", "u1", t_old))
        buf.add(RawStory("New", "", "src", "u2", t_new))
        s = buf.stats()
        assert s.oldest_at == t_old
        assert s.newest_at == t_new

    def test_stats_preview_limit(self, buf):
        buf.add_batch([make_story(f"Story {i}") for i in range(20)])
        s = buf.stats(preview_limit=5)
        assert len(s.preview) == 5

    def test_stats_preview_contains_headline_and_source(self, buf):
        buf.add(make_story("Big Fed news", source="reuters"))
        preview = buf.stats(preview_limit=1).preview[0]
        assert "Big Fed news" in preview["headline"]
        assert preview["source"] == "reuters"


class TestThreadSafety:
    def test_concurrent_adds(self, buf):
        """Multiple threads adding simultaneously should not lose stories."""
        n_threads = 10
        n_per_thread = 100
        errors = []

        def add_stories():
            try:
                for i in range(n_per_thread):
                    buf.add(make_story(f"T{threading.get_ident()}-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_stories) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert buf.size() == n_threads * n_per_thread

    def test_concurrent_drain_and_add(self, buf):
        """Drain from one thread while another adds — no data corruption."""
        buf.add_batch([make_story(f"initial-{i}") for i in range(50)])
        drained_total = []
        errors = []

        def drain_repeatedly():
            for _ in range(5):
                drained_total.extend(buf.drain())
                time.sleep(0.001)

        def add_repeatedly():
            try:
                for i in range(50):
                    buf.add(make_story(f"added-{i}"))
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=drain_repeatedly)
        t2 = threading.Thread(target=add_repeatedly)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Drain whatever remains
        drained_total.extend(buf.drain())
        assert not errors
        # All 100 stories (50 initial + 50 added) must be accounted for
        assert len(drained_total) == 100
