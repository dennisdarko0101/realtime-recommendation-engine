"""Tests for RecommendationCache."""

import time
import threading

import pytest

from src.serving.cache import RecommendationCache
from src.data.schemas import Recommendation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recs(item_id: str = "item1", score: float = 1.0) -> list[Recommendation]:
    return [Recommendation(item_id=item_id, score=score)]


# ---------------------------------------------------------------------------
# Test: basic set / get
# ---------------------------------------------------------------------------

class TestSetAndGet:
    def test_set_and_get_returns_cached_value(self):
        cache = RecommendationCache()
        cache.set("user1", _recs("a"))
        result = cache.get("user1")
        assert result is not None
        assert result[0].item_id == "a"

    def test_returns_none_for_missing_key(self):
        cache = RecommendationCache()
        assert cache.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Test: TTL expiration
# ---------------------------------------------------------------------------

class TestTTLExpiration:
    def test_entry_expires_after_ttl(self):
        cache = RecommendationCache(default_ttl=1)
        cache.set("user1", _recs(), ttl=0.1)
        assert cache.get("user1") is not None
        time.sleep(0.15)
        assert cache.get("user1") is None

    def test_expired_entry_not_returned(self):
        cache = RecommendationCache()
        cache.set("user1", _recs(), ttl=0.05)
        time.sleep(0.1)
        result = cache.get("user1")
        assert result is None


# ---------------------------------------------------------------------------
# Test: LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_evicts_oldest_when_over_max_size(self):
        cache = RecommendationCache(max_size=2)
        cache.set("user1", _recs("a"))
        cache.set("user2", _recs("b"))
        cache.set("user3", _recs("c"))
        # user1 should have been evicted (oldest)
        assert cache.get("user1") is None
        assert cache.get("user2") is not None
        assert cache.get("user3") is not None

    def test_access_refreshes_lru_position(self):
        cache = RecommendationCache(max_size=2)
        cache.set("user1", _recs("a"))
        cache.set("user2", _recs("b"))
        # Access user1 to make it recently used
        cache.get("user1")
        # Now adding user3 should evict user2 (oldest after refresh)
        cache.set("user3", _recs("c"))
        assert cache.get("user1") is not None
        assert cache.get("user2") is None
        assert cache.get("user3") is not None


# ---------------------------------------------------------------------------
# Test: invalidate
# ---------------------------------------------------------------------------

class TestInvalidate:
    def test_invalidate_removes_entry(self):
        cache = RecommendationCache()
        cache.set("user1", _recs())
        assert cache.invalidate("user1") is True
        assert cache.get("user1") is None

    def test_invalidate_returns_false_for_missing_key(self):
        cache = RecommendationCache()
        assert cache.invalidate("no_such_user") is False


# ---------------------------------------------------------------------------
# Test: clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_cache(self):
        cache = RecommendationCache()
        cache.set("user1", _recs())
        cache.set("user2", _recs())
        cache.clear()
        assert cache.size == 0
        assert cache.get("user1") is None
        assert cache.get("user2") is None


# ---------------------------------------------------------------------------
# Test: size property
# ---------------------------------------------------------------------------

class TestSize:
    def test_size_reflects_number_of_entries(self):
        cache = RecommendationCache()
        assert cache.size == 0
        cache.set("user1", _recs())
        assert cache.size == 1
        cache.set("user2", _recs())
        assert cache.size == 2
        cache.invalidate("user1")
        assert cache.size == 1


# ---------------------------------------------------------------------------
# Test: thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_access_no_errors(self):
        cache = RecommendationCache(max_size=100)
        errors: list[Exception] = []

        def writer(start: int):
            try:
                for i in range(50):
                    cache.set(f"user_{start + i}", _recs(f"item_{start + i}"))
            except Exception as exc:
                errors.append(exc)

        def reader(start: int):
            try:
                for i in range(50):
                    cache.get(f"user_{start + i}")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(50,)),
            threading.Thread(target=reader, args=(0,)),
            threading.Thread(target=reader, args=(50,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_set_and_invalidate(self):
        cache = RecommendationCache(max_size=200)
        errors: list[Exception] = []

        def setter():
            try:
                for i in range(100):
                    cache.set(f"u{i}", _recs())
            except Exception as exc:
                errors.append(exc)

        def invalidator():
            try:
                for i in range(100):
                    cache.invalidate(f"u{i}")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=invalidator)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []
