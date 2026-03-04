"""In-memory LRU cache for recommendations."""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock

from src.data.schemas import Recommendation


class RecommendationCache:
    """Thread-safe LRU cache with TTL for recommendation results."""

    def __init__(self, max_size: int = 10_000, default_ttl: int = 300) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[list[Recommendation], float]] = OrderedDict()
        self._lock = Lock()

    def get(self, user_id: str) -> list[Recommendation] | None:
        """Get cached recommendations for user, or None if missing/expired."""
        with self._lock:
            entry = self._cache.get(user_id)
            if entry is None:
                return None
            recs, expires_at = entry
            if time.time() > expires_at:
                del self._cache[user_id]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(user_id)
            return recs

    def set(
        self, user_id: str, recommendations: list[Recommendation], ttl: int | None = None
    ) -> None:
        """Cache recommendations for a user."""
        if ttl is None:
            ttl = self._default_ttl

        expires_at = time.time() + ttl

        with self._lock:
            if user_id in self._cache:
                self._cache.move_to_end(user_id)
            self._cache[user_id] = (recommendations, expires_at)
            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, user_id: str) -> bool:
        """Remove cached recommendations for a user. Returns True if found."""
        with self._lock:
            if user_id in self._cache:
                del self._cache[user_id]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def _evict_expired(self) -> int:
        """Remove all expired entries. Returns count evicted."""
        now = time.time()
        evicted = 0
        with self._lock:
            expired_keys = [
                k for k, (_, exp) in self._cache.items() if now > exp
            ]
            for k in expired_keys:
                del self._cache[k]
                evicted += 1
        return evicted
