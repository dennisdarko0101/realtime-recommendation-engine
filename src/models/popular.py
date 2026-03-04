"""Popularity-based recommendation models (fallback/baseline)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

from src.data.schemas import Interaction, InteractionType, Item, Recommendation


class PopularityRecommender:
    """Popularity-based recommendations: trending, all-time, and per-category."""

    def __init__(self) -> None:
        self._interactions: list[Interaction] = []
        self._items: dict[str, Item] = {}
        self._item_scores: dict[str, float] = {}
        self._category_scores: dict[str, dict[str, float]] = defaultdict(dict)
        self._item_timestamps: dict[str, list[datetime]] = defaultdict(list)

    def fit(self, interactions: list[Interaction], items: list[Item]) -> None:
        self._interactions = interactions
        self._items = {it.item_id: it for it in items}
        self._compute_scores()

    def _compute_scores(self) -> None:
        # Weighted interaction counts
        scores: dict[str, float] = Counter()
        for ix in self._interactions:
            weight = {
                InteractionType.VIEW: 1.0,
                InteractionType.CLICK: 2.0,
                InteractionType.RATE: 3.0,
                InteractionType.PURCHASE: 5.0,
            }.get(ix.interaction_type, 1.0)

            if ix.value is not None:
                weight *= ix.value / 5.0  # Normalize ratings

            scores[ix.item_id] += weight
            self._item_timestamps[ix.item_id].append(ix.timestamp)

        self._item_scores = dict(scores)

        # Per-category scores
        for item_id, score in scores.items():
            item = self._items.get(item_id)
            if item:
                self._category_scores[item.category][item_id] = score

    def recommend_popular(self, k: int = 10) -> list[Recommendation]:
        """All-time most popular items."""
        sorted_items = sorted(self._item_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            Recommendation(
                item_id=iid,
                score=score,
                reason="Popular item (all-time)",
                model_used="popularity",
            )
            for iid, score in sorted_items[:k]
        ]

    def recommend_trending(
        self, k: int = 10, window_hours: int = 24, now: Optional[datetime] = None
    ) -> list[Recommendation]:
        """Trending items based on recent interaction volume."""
        if now is None:
            now = datetime.utcnow()
        cutoff = now - timedelta(hours=window_hours)

        recent_scores: dict[str, float] = Counter()
        for ix in self._interactions:
            if ix.timestamp >= cutoff:
                recent_scores[ix.item_id] += 1.0

        sorted_items = sorted(recent_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            Recommendation(
                item_id=iid,
                score=score,
                reason=f"Trending (last {window_hours}h)",
                model_used="trending",
            )
            for iid, score in sorted_items[:k]
        ]

    def recommend_by_category(self, category: str, k: int = 10) -> list[Recommendation]:
        """Most popular items within a category."""
        cat_scores = self._category_scores.get(category, {})
        sorted_items = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            Recommendation(
                item_id=iid,
                score=score,
                reason=f"Popular in {category}",
                model_used="category_popularity",
            )
            for iid, score in sorted_items[:k]
        ]
