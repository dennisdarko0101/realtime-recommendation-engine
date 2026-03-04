"""Tests for RealtimeUpdater and EventProcessor."""

import pytest

from src.serving.realtime import RealtimeUpdater, EventProcessor
from src.serving.cache import RecommendationCache
from src.data.store import DataStore
from src.data.schemas import Interaction, InteractionType, User, Item, Recommendation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> DataStore:
    s = DataStore()
    s.add_user(User(user_id="u1"))
    s.add_user(User(user_id="u2"))
    s.add_item(Item(item_id="i1"))
    s.add_item(Item(item_id="i2"))
    s.add_item(Item(item_id="i3"))
    return s


@pytest.fixture
def cache() -> RecommendationCache:
    return RecommendationCache()


@pytest.fixture
def updater(store: DataStore, cache: RecommendationCache) -> RealtimeUpdater:
    return RealtimeUpdater(store=store, cache=cache)


def _interaction(user_id: str = "u1", item_id: str = "i1") -> Interaction:
    return Interaction(
        user_id=user_id,
        item_id=item_id,
        interaction_type=InteractionType.CLICK,
    )


# ---------------------------------------------------------------------------
# RealtimeUpdater tests
# ---------------------------------------------------------------------------

class TestOnInteraction:
    def test_stores_interaction(self, updater: RealtimeUpdater, store: DataStore):
        ix = _interaction("u1", "i1")
        updater.on_interaction(ix)
        stored = store.get_user_interactions("u1")
        assert len(stored) == 1
        assert stored[0].item_id == "i1"

    def test_invalidates_cache(self, updater: RealtimeUpdater, cache: RecommendationCache):
        recs = [Recommendation(item_id="i1", score=1.0)]
        cache.set("u1", recs)
        assert cache.get("u1") is not None
        updater.on_interaction(_interaction("u1", "i1"))
        assert cache.get("u1") is None

    def test_tracks_pending_retrain_users(self, updater: RealtimeUpdater):
        updater.on_interaction(_interaction("u1", "i1"))
        pending = updater.get_pending_retrain_users()
        assert "u1" in pending

    def test_multiple_interactions_same_user(self, updater: RealtimeUpdater, store: DataStore):
        updater.on_interaction(_interaction("u1", "i1"))
        updater.on_interaction(_interaction("u1", "i2"))
        updater.on_interaction(_interaction("u1", "i3"))
        stored = store.get_user_interactions("u1")
        assert len(stored) == 3
        pending = updater.get_pending_retrain_users()
        assert "u1" in pending
        assert len(pending) == 1  # same user, only one entry


class TestInteractionCounter:
    def test_interactions_since_retrain_increments(self, updater: RealtimeUpdater):
        assert updater.interactions_since_retrain == 0
        updater.on_interaction(_interaction("u1", "i1"))
        assert updater.interactions_since_retrain == 1
        updater.on_interaction(_interaction("u2", "i2"))
        assert updater.interactions_since_retrain == 2


class TestShouldRetrain:
    def test_below_threshold_returns_false(self, updater: RealtimeUpdater):
        updater.on_interaction(_interaction("u1", "i1"))
        assert updater.should_retrain(threshold=5) is False

    def test_at_threshold_returns_true(self, updater: RealtimeUpdater):
        for i in range(5):
            updater.on_interaction(_interaction("u1", f"i{i}"))
        assert updater.should_retrain(threshold=5) is True

    def test_above_threshold_returns_true(self, updater: RealtimeUpdater):
        for i in range(10):
            updater.on_interaction(_interaction("u1", f"i{i}"))
        assert updater.should_retrain(threshold=5) is True


class TestClearPending:
    def test_clear_pending_resets_state(self, updater: RealtimeUpdater):
        updater.on_interaction(_interaction("u1", "i1"))
        updater.on_interaction(_interaction("u2", "i2"))
        assert updater.interactions_since_retrain == 2
        assert len(updater.get_pending_retrain_users()) == 2

        updater.clear_pending()
        assert updater.interactions_since_retrain == 0
        assert len(updater.get_pending_retrain_users()) == 0


# ---------------------------------------------------------------------------
# EventProcessor tests
# ---------------------------------------------------------------------------

class TestEventProcessorBuffer:
    def test_buffers_events(self, updater: RealtimeUpdater):
        processor = EventProcessor(updater, batch_size=10)
        processor.add_event(_interaction("u1", "i1"))
        assert processor.buffer_size == 1
        assert updater.interactions_since_retrain == 0  # not flushed yet

    def test_auto_flushes_at_batch_size(self, updater: RealtimeUpdater):
        processor = EventProcessor(updater, batch_size=3)
        processor.add_event(_interaction("u1", "i1"))
        processor.add_event(_interaction("u1", "i2"))
        assert processor.buffer_size == 2
        # This third event triggers flush
        processor.add_event(_interaction("u1", "i3"))
        assert processor.buffer_size == 0
        assert updater.interactions_since_retrain == 3

    def test_manual_flush(self, updater: RealtimeUpdater):
        processor = EventProcessor(updater, batch_size=100)
        processor.add_event(_interaction("u1", "i1"))
        processor.add_event(_interaction("u2", "i2"))
        count = processor.flush()
        assert count == 2
        assert processor.buffer_size == 0
        assert updater.interactions_since_retrain == 2

    def test_flush_empty_buffer_returns_zero(self, updater: RealtimeUpdater):
        processor = EventProcessor(updater, batch_size=10)
        count = processor.flush()
        assert count == 0
