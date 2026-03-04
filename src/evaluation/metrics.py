"""Recommendation evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.data.schemas import MetricsReport, Recommendation


class RecommendationMetrics:
    """Compute standard recommendation quality metrics."""

    @staticmethod
    def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        """Fraction of top-k recommendations that are relevant."""
        rec_k = recommended[:k]
        if not rec_k:
            return 0.0
        hits = sum(1 for r in rec_k if r in relevant)
        return hits / len(rec_k)

    @staticmethod
    def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        """Fraction of relevant items that appear in top-k."""
        if not relevant:
            return 0.0
        rec_k = recommended[:k]
        hits = sum(1 for r in rec_k if r in relevant)
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        rec_k = recommended[:k]
        if not rec_k or not relevant:
            return 0.0

        dcg = 0.0
        for i, item in enumerate(rec_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)

        # Ideal DCG
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        """Mean Average Precision at k (for a single query)."""
        rec_k = recommended[:k]
        if not rec_k or not relevant:
            return 0.0

        score = 0.0
        hits = 0
        for i, item in enumerate(rec_k):
            if item in relevant:
                hits += 1
                score += hits / (i + 1)

        return score / min(len(relevant), k)

    @staticmethod
    def hit_rate(recommended: list[str], relevant: set[str]) -> float:
        """1 if any recommended item is relevant, 0 otherwise."""
        return 1.0 if any(r in relevant for r in recommended) else 0.0

    @staticmethod
    def mrr(recommended: list[str], relevant: set[str]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant item."""
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def coverage(
        all_recommendations: list[list[str]], total_items: int
    ) -> float:
        """Fraction of total catalog that appears in any recommendation list."""
        if total_items == 0:
            return 0.0
        unique_items = set()
        for recs in all_recommendations:
            unique_items.update(recs)
        return len(unique_items) / total_items

    @staticmethod
    def diversity(
        recommendations: list[str], item_categories: dict[str, str]
    ) -> float:
        """Intra-list diversity: fraction of unique categories in the list."""
        if not recommendations:
            return 0.0
        categories = [item_categories.get(r, "unknown") for r in recommendations]
        return len(set(categories)) / len(categories)

    @staticmethod
    def novelty(
        recommendations: list[str], item_popularity: dict[str, int], total_interactions: int
    ) -> float:
        """Average self-information of recommended items (less popular = more novel)."""
        if not recommendations or total_interactions == 0:
            return 0.0

        scores = []
        for item_id in recommendations:
            pop = item_popularity.get(item_id, 0)
            if pop > 0:
                scores.append(-np.log2(pop / total_interactions))
            else:
                scores.append(-np.log2(1 / total_interactions))

        return float(np.mean(scores))

    def evaluate(
        self,
        recommended_per_user: dict[str, list[str]],
        relevant_per_user: dict[str, set[str]],
        k: int = 10,
        total_items: int = 0,
        item_categories: dict[str, str] | None = None,
        item_popularity: dict[str, int] | None = None,
        total_interactions: int = 0,
        model_name: str = "",
    ) -> MetricsReport:
        """Evaluate a model across all users."""
        precisions, recalls, ndcgs, maps = [], [], [], []
        hits, mrrs = [], []
        all_rec_lists = []

        for user_id, recs in recommended_per_user.items():
            relevant = relevant_per_user.get(user_id, set())
            if not relevant:
                continue

            precisions.append(self.precision_at_k(recs, relevant, k))
            recalls.append(self.recall_at_k(recs, relevant, k))
            ndcgs.append(self.ndcg_at_k(recs, relevant, k))
            maps.append(self.map_at_k(recs, relevant, k))
            hits.append(self.hit_rate(recs, relevant))
            mrrs.append(self.mrr(recs, relevant))
            all_rec_lists.append(recs[:k])

        n = max(len(precisions), 1)

        cov = self.coverage(all_rec_lists, total_items) if total_items > 0 else 0.0

        div = 0.0
        if item_categories and all_rec_lists:
            divs = [self.diversity(r, item_categories) for r in all_rec_lists]
            div = float(np.mean(divs))

        nov = 0.0
        if item_popularity and all_rec_lists:
            novs = [
                self.novelty(r, item_popularity, total_interactions) for r in all_rec_lists
            ]
            nov = float(np.mean(novs))

        return MetricsReport(
            precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
            recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
            ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
            map_at_k=float(np.mean(maps)) if maps else 0.0,
            hit_rate=float(np.mean(hits)) if hits else 0.0,
            mrr=float(np.mean(mrrs)) if mrrs else 0.0,
            coverage=cov,
            diversity=div,
            novelty=nov,
            k=k,
            model_name=model_name,
        )
