"""Hybrid recommender combining collaborative and content-based approaches."""

from __future__ import annotations

from typing import Optional

from scipy.sparse import csr_matrix

from src.data.schemas import Item, Recommendation
from src.data.store import DataStore
from src.models.collaborative import CollaborativeFilter, MatrixFactorization, UserBasedCF
from src.models.content_based import ContentBasedFilter
from src.models.popular import PopularityRecommender


class HybridRecommender:
    """Combines collaborative and content-based filtering with multiple strategies.

    Strategies:
      - weighted: linear combination of collaborative and content-based scores
      - switching: use collaborative if enough data, else content-based
      - cascade: content-based filters candidates, collaborative ranks them
    """

    def __init__(
        self,
        collab_weight: float = 0.6,
        content_weight: float = 0.4,
        min_interactions_for_collab: int = 5,
    ) -> None:
        self.collab_weight = collab_weight
        self.content_weight = content_weight
        self.min_interactions_for_collab = min_interactions_for_collab

        self._collab: CollaborativeFilter | None = None
        self._content: ContentBasedFilter | None = None
        self._popularity: PopularityRecommender | None = None
        self._store: DataStore | None = None
        self._fitted = False

    def fit(
        self,
        store: DataStore,
        collab_model: Optional[CollaborativeFilter] = None,
        content_model: Optional[ContentBasedFilter] = None,
    ) -> None:
        """Fit hybrid model using data from the store."""
        self._store = store

        # Collaborative model
        if collab_model is not None:
            self._collab = collab_model
        else:
            self._collab = UserBasedCF(n_neighbors=20)

        interaction_matrix = store.get_interaction_matrix()
        user_index = store.get_user_index()
        item_index = store.get_item_index()
        self._collab.fit(interaction_matrix, user_index, item_index)

        # Content model
        if content_model is not None:
            self._content = content_model
        else:
            self._content = ContentBasedFilter()
            self._content.fit_tfidf(store.list_items())

        # Popularity fallback
        self._popularity = PopularityRecommender()
        self._popularity.fit(store.get_all_interactions(), store.list_items())

        self._fitted = True

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        strategy: str = "weighted",
    ) -> list[Recommendation]:
        if not self._fitted:
            return []

        if strategy == "weighted":
            return self._weighted(user_id, k)
        elif strategy == "switching":
            return self._switching(user_id, k)
        elif strategy == "cascade":
            return self._cascade(user_id, k)
        else:
            return self._weighted(user_id, k)

    def _get_user_history(self, user_id: str) -> list[str]:
        if self._store is None:
            return []
        interactions = self._store.get_user_interactions(user_id)
        return list({ix.item_id for ix in interactions})

    def _weighted(self, user_id: str, k: int) -> list[Recommendation]:
        """Linear combination of collaborative and content-based scores."""
        collab_recs = self._collab.recommend(user_id, k * 3) if self._collab else []
        history = self._get_user_history(user_id)
        content_recs = (
            self._content.recommend(user_id, k * 3, user_history=history)
            if self._content
            else []
        )

        # Merge scores
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}

        for rec in collab_recs:
            scores[rec.item_id] = scores.get(rec.item_id, 0) + self.collab_weight * rec.score
            reasons[rec.item_id] = rec.reason

        for rec in content_recs:
            scores[rec.item_id] = scores.get(rec.item_id, 0) + self.content_weight * rec.score
            if rec.item_id not in reasons:
                reasons[rec.item_id] = rec.reason

        if not scores:
            return self._fallback(user_id, k)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            Recommendation(
                item_id=iid,
                score=score,
                reason=reasons.get(iid, "Hybrid recommendation"),
                model_used="hybrid_weighted",
            )
            for iid, score in sorted_items
        ]

    def _switching(self, user_id: str, k: int) -> list[Recommendation]:
        """Use collaborative if enough interaction data, else content-based."""
        history = self._get_user_history(user_id)

        if len(history) >= self.min_interactions_for_collab and self._collab:
            recs = self._collab.recommend(user_id, k)
            if recs:
                return recs

        # Fall back to content-based
        if self._content and history:
            recs = self._content.recommend(user_id, k, user_history=history)
            if recs:
                return recs

        return self._fallback(user_id, k)

    def _cascade(self, user_id: str, k: int) -> list[Recommendation]:
        """Content-based filters candidates, collaborative ranks them."""
        history = self._get_user_history(user_id)

        # Stage 1: get broad content-based candidates
        candidates = (
            self._content.recommend(user_id, k * 5, user_history=history)
            if self._content and history
            else []
        )

        if not candidates:
            return self._fallback(user_id, k)

        # Stage 2: re-score with collaborative signal
        collab_recs = self._collab.recommend(user_id, k * 5) if self._collab else []
        collab_scores = {r.item_id: r.score for r in collab_recs}

        scored = []
        for rec in candidates:
            boost = collab_scores.get(rec.item_id, 0.0)
            scored.append(
                Recommendation(
                    item_id=rec.item_id,
                    score=rec.score + boost,
                    reason="Content candidate, collaborative ranked",
                    model_used="hybrid_cascade",
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def _fallback(self, user_id: str, k: int) -> list[Recommendation]:
        """Popularity-based fallback for cold-start users."""
        if self._popularity:
            return self._popularity.recommend_popular(k)
        return []
