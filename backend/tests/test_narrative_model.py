"""Tests for models/narrative.py — pure Python, no external deps."""

import time
import pytest
from models.narrative import NarrativeDirection, TimeSeriesPoint


def make_narrative(**kwargs) -> NarrativeDirection:
    defaults = dict(name="Test narrative", description="A test description")
    return NarrativeDirection(**{**defaults, **kwargs})


class TestNarrativeDirectionDefaults:
    def test_id_assigned(self):
        n = make_narrative()
        assert n.id and len(n.id) == 36  # UUID format

    def test_two_instances_have_different_ids(self):
        a, b = make_narrative(), make_narrative()
        assert a.id != b.id

    def test_event_count_starts_at_zero(self):
        assert make_narrative().event_count == 0

    def test_empty_series_gives_none_properties(self):
        n = make_narrative()
        assert n.current_surprise is None
        assert n.current_impact is None
        assert n.model_risk is None


class TestModelRisk:
    def test_geometric_mean(self):
        n = make_narrative()
        n.append_surprise(0.64)
        n.append_impact(1.0)
        assert abs(n.model_risk - 0.8) < 1e-9   # sqrt(0.64 * 1.0) = 0.8

    def test_zero_surprise_gives_zero_risk(self):
        n = make_narrative()
        n.append_surprise(0.0)
        n.append_impact(0.9)
        assert n.model_risk == 0.0

    def test_symmetric(self):
        n1, n2 = make_narrative(), make_narrative()
        n1.append_surprise(0.4); n1.append_impact(0.9)
        n2.append_surprise(0.9); n2.append_impact(0.4)
        assert abs(n1.model_risk - n2.model_risk) < 1e-9

    def test_both_max_gives_one(self):
        n = make_narrative()
        n.append_surprise(1.0)
        n.append_impact(1.0)
        assert abs(n.model_risk - 1.0) < 1e-9


class TestTimeSeries:
    def test_append_surprise_updates_current(self):
        n = make_narrative()
        n.append_surprise(0.3)
        n.append_surprise(0.7)
        assert n.current_surprise == 0.7

    def test_append_impact_updates_current(self):
        n = make_narrative()
        n.append_impact(0.5)
        assert n.current_impact == 0.5

    def test_series_accumulates(self):
        n = make_narrative()
        for v in [0.1, 0.2, 0.3]:
            n.append_surprise(v)
        assert len(n.surprise_series) == 3
        assert [p.value for p in n.surprise_series] == [0.1, 0.2, 0.3]

    def test_timestamps_monotonically_increasing(self):
        n = make_narrative()
        t1, t2, t3 = 1000.0, 1001.0, 1002.0
        n.append_surprise(0.1, timestamp=t1)
        n.append_surprise(0.2, timestamp=t2)
        n.append_surprise(0.3, timestamp=t3)
        timestamps = [p.timestamp for p in n.surprise_series]
        assert timestamps == sorted(timestamps)

    def test_last_updated_changes_on_append(self):
        n = make_narrative()
        before = n.last_updated
        time.sleep(0.01)
        n.append_surprise(0.5)
        assert n.last_updated > before


class TestAddHeadline:
    def test_adds_headline(self):
        n = make_narrative()
        n.add_headline("Markets fall on Fed news")
        assert "Markets fall on Fed news" in n.recent_headlines

    def test_increments_event_count(self):
        n = make_narrative()
        n.add_headline("Headline 1")
        n.add_headline("Headline 2")
        assert n.event_count == 2

    def test_recent_headlines_capped_at_max(self):
        n = make_narrative()
        for i in range(15):
            n.add_headline(f"Headline {i}", max_recent=10)
        assert len(n.recent_headlines) == 10
        # Should keep the LAST 10
        assert n.recent_headlines[-1] == "Headline 14"
        assert n.recent_headlines[0] == "Headline 5"

    def test_event_count_not_capped(self):
        n = make_narrative()
        for i in range(20):
            n.add_headline(f"H{i}", max_recent=10)
        assert n.event_count == 20  # count tracks all, not just recent
