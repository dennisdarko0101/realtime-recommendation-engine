"""Offline evaluation: temporal splits and model comparison."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

from src.data.schemas import Interaction, InteractionType, MetricsReport
from src.data.store import DataStore
from src.evaluation.metrics import RecommendationMetrics
from src.models.collaborative import CollaborativeFilter
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender
from src.models.popular import PopularityRecommender


class OfflineEvaluator:
    """Evaluate recommendation models with temporal train/test splits."""

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self._metrics = RecommendationMetrics()

    def temporal_split(
        self,
        interactions: list[Interaction],
        split_ratio: float = 0.8,
    ) -> tuple[list[Interaction], list[Interaction]]:
        """Split interactions temporally: earlier for training, later for testing."""
        sorted_ix = sorted(interactions, key=lambda x: x.timestamp)
        split_idx = int(len(sorted_ix) * split_ratio)
        return sorted_ix[:split_idx], sorted_ix[split_idx:]

    def build_relevant_sets(
        self,
        test_interactions: list[Interaction],
        min_interaction_type: InteractionType = InteractionType.CLICK,
    ) -> dict[str, set[str]]:
        """Build per-user relevant item sets from test interactions."""
        type_order = {
            InteractionType.VIEW: 0,
            InteractionType.CLICK: 1,
            InteractionType.RATE: 2,
            InteractionType.PURCHASE: 3,
        }
        min_order = type_order.get(min_interaction_type, 1)

        relevant: dict[str, set[str]] = {}
        for ix in test_interactions:
            ix_order = type_order.get(ix.interaction_type, 0)
            if ix_order >= min_order:
                relevant.setdefault(ix.user_id, set()).add(ix.item_id)

        return relevant

    def evaluate_collaborative(
        self,
        model: CollaborativeFilter,
        store: DataStore,
        test_interactions: list[Interaction],
        model_name: str = "collaborative",
    ) -> MetricsReport:
        """Evaluate a collaborative filtering model."""
        # Fit on training data (already in the store)
        matrix = store.get_interaction_matrix()
        user_index = store.get_user_index()
        item_index = store.get_item_index()
        model.fit(matrix, user_index, item_index)

        relevant = self.build_relevant_sets(test_interactions)
        items = store.list_items()
        item_categories = {it.item_id: it.category for it in items}
        item_pop = self._compute_popularity(store.get_all_interactions())

        recommended: dict[str, list[str]] = {}
        for user_id in relevant:
            recs = model.recommend(user_id, self.k)
            recommended[user_id] = [r.item_id for r in recs]

        return self._metrics.evaluate(
            recommended_per_user=recommended,
            relevant_per_user=relevant,
            k=self.k,
            total_items=store.n_items,
            item_categories=item_categories,
            item_popularity=item_pop,
            total_interactions=store.n_interactions,
            model_name=model_name,
        )

    def evaluate_hybrid(
        self,
        model: HybridRecommender,
        store: DataStore,
        test_interactions: list[Interaction],
        strategy: str = "weighted",
        model_name: str = "hybrid",
    ) -> MetricsReport:
        """Evaluate a hybrid model."""
        model.fit(store)

        relevant = self.build_relevant_sets(test_interactions)
        items = store.list_items()
        item_categories = {it.item_id: it.category for it in items}
        item_pop = self._compute_popularity(store.get_all_interactions())

        recommended: dict[str, list[str]] = {}
        for user_id in relevant:
            recs = model.recommend(user_id, self.k, strategy=strategy)
            recommended[user_id] = [r.item_id for r in recs]

        return self._metrics.evaluate(
            recommended_per_user=recommended,
            relevant_per_user=relevant,
            k=self.k,
            total_items=store.n_items,
            item_categories=item_categories,
            item_popularity=item_pop,
            total_interactions=store.n_interactions,
            model_name=model_name,
        )

    def compare_models(
        self,
        store: DataStore,
        test_interactions: list[Interaction],
    ) -> list[MetricsReport]:
        """Evaluate and compare multiple model types."""
        from src.models.collaborative import ItemBasedCF, MatrixFactorization, UserBasedCF

        reports = []

        # User-based CF
        ubcf = UserBasedCF(n_neighbors=20)
        reports.append(self.evaluate_collaborative(ubcf, store, test_interactions, "user_based_cf"))

        # Item-based CF
        ibcf = ItemBasedCF()
        reports.append(self.evaluate_collaborative(ibcf, store, test_interactions, "item_based_cf"))

        # Matrix Factorization
        mf = MatrixFactorization(n_factors=50)
        reports.append(
            self.evaluate_collaborative(mf, store, test_interactions, "matrix_factorization")
        )

        # Hybrid
        hybrid = HybridRecommender()
        reports.append(
            self.evaluate_hybrid(hybrid, store, test_interactions, "weighted", "hybrid_weighted")
        )

        return reports

    @staticmethod
    def _compute_popularity(interactions: list[Interaction]) -> dict[str, int]:
        return dict(Counter(ix.item_id for ix in interactions))
