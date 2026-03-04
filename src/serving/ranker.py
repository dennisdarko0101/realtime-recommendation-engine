"""Post-retrieval reranking with business rules."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from src.data.schemas import Item, Recommendation, User


class RecommendationRanker:
    """Rerank recommendations with business rules: diversity, freshness, dedup."""

    def __init__(
        self,
        max_same_category: int = 3,
        freshness_boost: float = 0.1,
        diversity_weight: float = 0.2,
    ) -> None:
        self.max_same_category = max_same_category
        self.freshness_boost = freshness_boost
        self.diversity_weight = diversity_weight

    def rerank(
        self,
        recommendations: list[Recommendation],
        user: Optional[User] = None,
        items: Optional[dict[str, Item]] = None,
        seen_item_ids: Optional[set[str]] = None,
        price_range: Optional[tuple[float, float]] = None,
    ) -> list[Recommendation]:
        """Apply business rules to rerank recommendations."""
        recs = list(recommendations)

        # 1. Remove already-seen items
        if seen_item_ids:
            recs = [r for r in recs if r.item_id not in seen_item_ids]

        # 2. Price range filter
        if price_range and items:
            lo, hi = price_range
            filtered = []
            for r in recs:
                item = items.get(r.item_id)
                if item is None:
                    filtered.append(r)
                    continue
                price = item.features.get("price")
                if price is None or (lo <= float(price) <= hi):
                    filtered.append(r)
            recs = filtered

        # 3. Category diversity enforcement
        if items:
            recs = self._enforce_category_diversity(recs, items)

        # 4. Freshness boost (items earlier in original list get slight boost)
        recs = self._apply_freshness_boost(recs)

        return recs

    def _enforce_category_diversity(
        self, recs: list[Recommendation], items: dict[str, Item]
    ) -> list[Recommendation]:
        """Ensure no more than max_same_category items from the same category."""
        result = []
        cat_counts: Counter[str] = Counter()

        for rec in recs:
            item = items.get(rec.item_id)
            cat = item.category if item else "unknown"
            if cat_counts[cat] < self.max_same_category:
                result.append(rec)
                cat_counts[cat] += 1

        return result

    def _apply_freshness_boost(self, recs: list[Recommendation]) -> list[Recommendation]:
        """Apply a small score boost to promote variety in ordering."""
        if not recs:
            return recs

        boosted = []
        max_score = max(r.score for r in recs) if recs else 1.0
        for i, rec in enumerate(recs):
            # Small random-ish boost based on position to break ties differently
            position_factor = 1.0 - (i / max(len(recs), 1)) * self.freshness_boost
            new_score = rec.score * position_factor
            boosted.append(
                Recommendation(
                    item_id=rec.item_id,
                    score=new_score,
                    reason=rec.reason,
                    model_used=rec.model_used,
                )
            )

        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted
