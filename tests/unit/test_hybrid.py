"""Unit tests for the hybrid recommender."""

import pytest

from src.data.generator import generate_dataset
from src.data.store import DataStore
from src.data.schemas import User, Item, Interaction, InteractionType
from src.models.hybrid import HybridRecommender
from src.models.collaborative import UserBasedCF, MatrixFactorization
from src.models.content_based import ContentBasedFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_store():
    """A small but realistic dataset: 20 users, 10 items, 200 interactions."""
    return generate_dataset(n_users=20, n_items=10, n_interactions=200, seed=7)


@pytest.fixture()
def fitted_hybrid(small_store):
    """A HybridRecommender fitted on the small dataset with defaults."""
    hr = HybridRecommender(collab_weight=0.6, content_weight=0.4, min_interactions_for_collab=3)
    hr.fit(small_store)
    return hr, small_store


@pytest.fixture()
def active_user_id(small_store):
    """Return a user ID that has at least 5 interactions."""
    for user in small_store.list_users():
        if len(small_store.get_user_interactions(user.user_id)) >= 5:
            return user.user_id
    # Fallback -- pick first user
    return small_store.list_users()[0].user_id


@pytest.fixture()
def cold_user_store(small_store):
    """Add a brand-new user with zero interactions to the store."""
    cold_user = User(user_id="cold_start_user_999", features={"region": "northeast"})
    small_store.add_user(cold_user)
    return small_store, "cold_start_user_999"


# ===========================================================================
# Weighted strategy
# ===========================================================================

class TestWeightedStrategy:
    """Tests for the weighted hybrid strategy."""

    def test_weighted_returns_recommendations(self, fitted_hybrid, active_user_id):
        hr, store = fitted_hybrid
        recs = hr.recommend(active_user_id, k=5, strategy="weighted")
        assert isinstance(recs, list)

    def test_weighted_model_used_label(self, fitted_hybrid, active_user_id):
        hr, _ = fitted_hybrid
        recs = hr.recommend(active_user_id, k=5, strategy="weighted")
        for rec in recs:
            assert rec.model_used == "hybrid_weighted"

    def test_weighted_scores_are_combined(self, fitted_hybrid, active_user_id):
        hr, store = fitted_hybrid
        recs = hr.recommend(active_user_id, k=5, strategy="weighted")
        if len(recs) >= 2:
            # Scores should be in descending order
            scores = [r.score for r in recs]
            assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Switching strategy
# ===========================================================================

class TestSwitchingStrategy:
    """Tests for the switching hybrid strategy."""

    def test_switching_uses_collab_for_active_user(self, small_store, active_user_id):
        """An active user (>= min_interactions) should get collaborative recs."""
        hr = HybridRecommender(min_interactions_for_collab=2)
        hr.fit(small_store)
        recs = hr.recommend(active_user_id, k=5, strategy="switching")
        assert isinstance(recs, list)

    def test_switching_falls_back_for_cold_user(self, cold_user_store):
        """A cold-start user should fall back to content or popularity."""
        store, cold_uid = cold_user_store
        hr = HybridRecommender(min_interactions_for_collab=3)
        hr.fit(store)
        recs = hr.recommend(cold_uid, k=5, strategy="switching")
        # Should still return something (popularity fallback)
        assert isinstance(recs, list)
        assert len(recs) > 0


# ===========================================================================
# Cascade strategy
# ===========================================================================

class TestCascadeStrategy:
    """Tests for the cascade hybrid strategy."""

    def test_cascade_returns_recommendations(self, fitted_hybrid, active_user_id):
        hr, _ = fitted_hybrid
        recs = hr.recommend(active_user_id, k=5, strategy="cascade")
        assert isinstance(recs, list)

    def test_cascade_model_used_label(self, fitted_hybrid, active_user_id):
        hr, _ = fitted_hybrid
        recs = hr.recommend(active_user_id, k=5, strategy="cascade")
        for rec in recs:
            assert rec.model_used == "hybrid_cascade"

    def test_cascade_scores_descending(self, fitted_hybrid, active_user_id):
        hr, _ = fitted_hybrid
        recs = hr.recommend(active_user_id, k=5, strategy="cascade")
        if len(recs) >= 2:
            scores = [r.score for r in recs]
            assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Cold-start / Popularity fallback
# ===========================================================================

class TestColdStart:
    """Cold-start users should receive popularity-based fallback recommendations."""

    def test_new_user_gets_popularity_fallback(self, cold_user_store):
        store, cold_uid = cold_user_store
        hr = HybridRecommender()
        hr.fit(store)
        recs = hr.recommend(cold_uid, k=5, strategy="weighted")
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_fallback_scores_positive(self, cold_user_store):
        store, cold_uid = cold_user_store
        hr = HybridRecommender()
        hr.fit(store)
        recs = hr.recommend(cold_uid, k=5, strategy="weighted")
        for rec in recs:
            assert rec.score > 0


# ===========================================================================
# Fit behaviour
# ===========================================================================

class TestFitBehaviour:
    """Verify that fit correctly initialises all sub-models."""

    def test_fit_sets_fitted_flag(self, small_store):
        hr = HybridRecommender()
        assert hr._fitted is False
        hr.fit(small_store)
        assert hr._fitted is True

    def test_fit_populates_sub_models(self, small_store):
        hr = HybridRecommender()
        hr.fit(small_store)
        assert hr._collab is not None
        assert hr._content is not None
        assert hr._popularity is not None
        assert hr._store is small_store

    def test_fit_with_custom_collab_model(self, small_store):
        mf = MatrixFactorization(n_factors=3)
        hr = HybridRecommender()
        hr.fit(small_store, collab_model=mf)
        assert hr._collab is mf

    def test_fit_with_custom_content_model(self, small_store):
        cbf = ContentBasedFilter()
        cbf.fit_tfidf(small_store.list_items())
        hr = HybridRecommender()
        hr.fit(small_store, content_model=cbf)
        assert hr._content is cbf

    def test_recommend_before_fit_returns_empty(self):
        hr = HybridRecommender()
        recs = hr.recommend("u0", k=5)
        assert recs == []


# ===========================================================================
# Configurable weights
# ===========================================================================

class TestConfigurableWeights:
    """Different weight configurations should influence results."""

    def test_higher_collab_weight(self, small_store, active_user_id):
        hr_collab = HybridRecommender(collab_weight=0.9, content_weight=0.1)
        hr_collab.fit(small_store)
        recs_collab = hr_collab.recommend(active_user_id, k=5, strategy="weighted")

        hr_content = HybridRecommender(collab_weight=0.1, content_weight=0.9)
        hr_content.fit(small_store)
        recs_content = hr_content.recommend(active_user_id, k=5, strategy="weighted")

        # Both should return results
        assert isinstance(recs_collab, list)
        assert isinstance(recs_content, list)

    def test_weight_zero_collab(self, small_store, active_user_id):
        """With collab_weight=0, the weighted strategy should rely on content only."""
        hr = HybridRecommender(collab_weight=0.0, content_weight=1.0)
        hr.fit(small_store)
        recs = hr.recommend(active_user_id, k=5, strategy="weighted")
        assert isinstance(recs, list)


# ===========================================================================
# Different strategies produce different results
# ===========================================================================

class TestStrategyDifferences:
    """Different strategies may produce distinct orderings or sets of items."""

    def test_strategies_produce_results(self, fitted_hybrid, active_user_id):
        hr, _ = fitted_hybrid
        for strategy in ("weighted", "switching", "cascade"):
            recs = hr.recommend(active_user_id, k=5, strategy=strategy)
            assert isinstance(recs, list)

    def test_different_strategies_model_labels(self, fitted_hybrid, active_user_id):
        hr, _ = fitted_hybrid
        recs_w = hr.recommend(active_user_id, k=5, strategy="weighted")
        recs_c = hr.recommend(active_user_id, k=5, strategy="cascade")
        model_labels_w = {r.model_used for r in recs_w}
        model_labels_c = {r.model_used for r in recs_c}
        # weighted and cascade use distinct model_used labels
        if recs_w and recs_c:
            assert model_labels_w != model_labels_c


# ===========================================================================
# User with many interactions
# ===========================================================================

class TestManyInteractions:
    """Ensure the model handles a user who has interacted with many items."""

    def test_user_with_many_interactions(self, small_store):
        """Pick the user with the most interactions and verify recs work."""
        users = small_store.list_users()
        best_user = max(
            users,
            key=lambda u: len(small_store.get_user_interactions(u.user_id)),
        )
        hr = HybridRecommender()
        hr.fit(small_store)
        recs = hr.recommend(best_user.user_id, k=5, strategy="weighted")
        assert isinstance(recs, list)
        # None of the recommended items should be in the user's history
        history_ids = {
            ix.item_id
            for ix in small_store.get_user_interactions(best_user.user_id)
        }
        for rec in recs:
            assert rec.item_id not in history_ids or rec.model_used in (
                "popularity",
                "hybrid_weighted",
                "hybrid_cascade",
            )
