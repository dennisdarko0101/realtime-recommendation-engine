"""Integration tests for the end-to-end recommendation pipeline."""

from __future__ import annotations

import pytest
import numpy as np
from scipy.sparse import issparse

from src.data.generator import (
    generate_dataset,
    generate_interactions,
    generate_items,
    generate_users,
)
from src.data.schemas import Interaction, InteractionType, Item, User
from src.data.store import DataStore
from src.models.collaborative import ItemBasedCF, MatrixFactorization, UserBasedCF
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender
from src.models.popular import PopularityRecommender
from src.evaluation.offline_eval import OfflineEvaluator

# ---------------------------------------------------------------------------
# Small dataset parameters used across all pipeline tests
# ---------------------------------------------------------------------------
N_USERS = 20
N_ITEMS = 10
N_INTERACTIONS = 200
SEED = 123


@pytest.fixture
def small_users():
    return generate_users(N_USERS, seed=SEED)


@pytest.fixture
def small_items():
    return generate_items(N_ITEMS, seed=SEED)


@pytest.fixture
def small_interactions(small_users, small_items):
    return generate_interactions(small_users, small_items, N_INTERACTIONS, seed=SEED)


@pytest.fixture
def small_store(small_users, small_items, small_interactions):
    s = DataStore()
    s.add_users_batch(small_users)
    s.add_items_batch(small_items)
    s.add_interactions_batch(small_interactions)
    return s


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.integration
def test_generate_dataset_creates_correct_counts():
    store = generate_dataset(
        n_users=N_USERS,
        n_items=N_ITEMS,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
    )
    assert store.n_users == N_USERS
    assert store.n_items == N_ITEMS
    assert store.n_interactions == N_INTERACTIONS


@pytest.mark.integration
def test_datastore_save_load_roundtrip(tmp_path, small_store):
    """Save to disk and reload; counts and content must match."""
    data_dir = str(tmp_path / "rec_data")
    small_store.save(data_dir)

    loaded = DataStore()
    loaded.load(data_dir)

    assert loaded.n_users == small_store.n_users
    assert loaded.n_items == small_store.n_items
    assert loaded.n_interactions == small_store.n_interactions

    # Spot-check a user and item round-trip
    orig_user = small_store.list_users()[0]
    loaded_user = loaded.get_user(orig_user.user_id)
    assert loaded_user is not None
    assert loaded_user.user_id == orig_user.user_id

    orig_item = small_store.list_items()[0]
    loaded_item = loaded.get_item(orig_item.item_id)
    assert loaded_item is not None
    assert loaded_item.item_id == orig_item.item_id
    assert loaded_item.category == orig_item.category


@pytest.mark.integration
def test_interaction_matrix_has_correct_shape(small_store):
    matrix = small_store.get_interaction_matrix()
    assert matrix.shape == (N_USERS, N_ITEMS)


@pytest.mark.integration
def test_interaction_matrix_is_sparse(small_store):
    matrix = small_store.get_interaction_matrix()
    assert issparse(matrix)
    # The matrix should have fewer non-zero entries than total elements
    assert matrix.nnz < N_USERS * N_ITEMS


@pytest.mark.integration
def test_collaborative_user_based_pipeline(small_store):
    """UserBasedCF: fit on interaction matrix then recommend for a known user."""
    matrix = small_store.get_interaction_matrix()
    user_index = small_store.get_user_index()
    item_index = small_store.get_item_index()

    model = UserBasedCF(n_neighbors=5)
    model.fit(matrix, user_index, item_index)

    # Pick a user that has interactions
    user_id = small_store.list_users()[0].user_id
    recs = model.recommend(user_id, k=5)
    assert isinstance(recs, list)
    # Each recommendation must have the expected fields
    for rec in recs:
        assert rec.item_id
        assert isinstance(rec.score, float)
        assert rec.model_used == "user_based_cf"


@pytest.mark.integration
def test_collaborative_item_based_pipeline(small_store):
    """ItemBasedCF: fit then recommend."""
    matrix = small_store.get_interaction_matrix()
    user_index = small_store.get_user_index()
    item_index = small_store.get_item_index()

    model = ItemBasedCF()
    model.fit(matrix, user_index, item_index)

    user_id = small_store.list_users()[0].user_id
    recs = model.recommend(user_id, k=5)
    assert isinstance(recs, list)
    for rec in recs:
        assert rec.item_id
        assert isinstance(rec.score, float)


@pytest.mark.integration
def test_collaborative_matrix_factorization_pipeline(small_store):
    """MatrixFactorization: fit then recommend."""
    matrix = small_store.get_interaction_matrix()
    user_index = small_store.get_user_index()
    item_index = small_store.get_item_index()

    n_factors = min(5, min(N_USERS, N_ITEMS) - 1)
    model = MatrixFactorization(n_factors=n_factors)
    model.fit(matrix, user_index, item_index)

    user_id = small_store.list_users()[0].user_id
    recs = model.recommend(user_id, k=5)
    assert isinstance(recs, list)
    for rec in recs:
        assert rec.item_id
        assert isinstance(rec.score, float)
        assert rec.model_used == "matrix_factorization"


@pytest.mark.integration
def test_content_based_pipeline(small_store):
    """ContentBasedFilter: fit TF-IDF then get similar items."""
    items = small_store.list_items()
    model = ContentBasedFilter()
    model.fit_tfidf(items)

    item_id = items[0].item_id
    similar = model.get_similar_items(item_id, k=5)
    assert isinstance(similar, list)
    assert len(similar) <= 5
    for rec in similar:
        assert rec.item_id != item_id  # should not include self
        assert isinstance(rec.score, float)
        assert rec.score > 0


@pytest.mark.integration
def test_content_based_recommend_for_user(small_store):
    """ContentBasedFilter: recommend based on user history."""
    items = small_store.list_items()
    model = ContentBasedFilter()
    model.fit_tfidf(items)

    user_id = small_store.list_users()[0].user_id
    user_history = [
        ix.item_id for ix in small_store.get_user_interactions(user_id)
    ]
    recs = model.recommend(user_id, k=5, user_history=user_history)
    assert isinstance(recs, list)
    # Recommended items should not be in the user's history
    for rec in recs:
        assert rec.item_id not in user_history


@pytest.mark.integration
def test_hybrid_pipeline(small_store):
    """HybridRecommender: fit then recommend using different strategies."""
    hybrid = HybridRecommender()
    hybrid.fit(small_store)

    user_id = small_store.list_users()[0].user_id

    for strategy in ("weighted", "switching", "cascade"):
        recs = hybrid.recommend(user_id, k=5, strategy=strategy)
        assert isinstance(recs, list), f"Strategy {strategy} returned non-list"


@pytest.mark.integration
def test_popularity_pipeline(small_store):
    """PopularityRecommender: fit then get popular and trending."""
    interactions = small_store.get_all_interactions()
    items = small_store.list_items()

    model = PopularityRecommender()
    model.fit(interactions, items)

    popular = model.recommend_popular(k=5)
    assert isinstance(popular, list)
    assert len(popular) <= 5
    assert all(r.model_used == "popularity" for r in popular)

    trending = model.recommend_trending(k=5, window_hours=24 * 365 * 10)
    assert isinstance(trending, list)


@pytest.mark.integration
def test_offline_evaluation_produces_valid_metrics(small_store):
    """OfflineEvaluator: temporal split then evaluate should produce valid metrics."""
    evaluator = OfflineEvaluator(k=5)

    all_interactions = small_store.get_all_interactions()
    train_ix, test_ix = evaluator.temporal_split(all_interactions, split_ratio=0.8)

    assert len(train_ix) + len(test_ix) == len(all_interactions)
    assert len(train_ix) > 0
    assert len(test_ix) > 0

    # Build a training store
    train_store = DataStore()
    for u in small_store.list_users():
        train_store.add_user(u)
    for it in small_store.list_items():
        train_store.add_item(it)
    train_store.add_interactions_batch(train_ix)

    # Evaluate user-based CF
    model = UserBasedCF(n_neighbors=5)
    report = evaluator.evaluate_collaborative(model, train_store, test_ix, "user_based_cf")

    assert report.model_name == "user_based_cf"
    assert report.k == 5
    assert 0.0 <= report.precision_at_k <= 1.0
    assert 0.0 <= report.recall_at_k <= 1.0
    assert 0.0 <= report.ndcg_at_k <= 1.0
    assert 0.0 <= report.hit_rate <= 1.0
    assert 0.0 <= report.coverage <= 1.0


@pytest.mark.integration
def test_full_pipeline_generate_train_recommend_evaluate():
    """Complete pipeline: generate data -> train models -> recommend -> evaluate."""
    # 1. Generate dataset
    store = generate_dataset(
        n_users=N_USERS,
        n_items=N_ITEMS,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
    )
    assert store.n_users == N_USERS
    assert store.n_items == N_ITEMS

    # 2. Train models
    items = store.list_items()
    interactions = store.get_all_interactions()
    matrix = store.get_interaction_matrix()
    user_index = store.get_user_index()
    item_index = store.get_item_index()

    # Collaborative
    ubcf = UserBasedCF(n_neighbors=5)
    ubcf.fit(matrix, user_index, item_index)

    # Content-based
    content = ContentBasedFilter()
    content.fit_tfidf(items)

    # Popularity
    pop = PopularityRecommender()
    pop.fit(interactions, items)

    # Hybrid
    hybrid = HybridRecommender()
    hybrid.fit(store, content_model=content)

    # 3. Generate recommendations for every user
    users = store.list_users()
    for user in users:
        recs = hybrid.recommend(user.user_id, k=5)
        assert isinstance(recs, list)

    # 4. Evaluate
    evaluator = OfflineEvaluator(k=5)
    train_ix, test_ix = evaluator.temporal_split(interactions, 0.8)

    train_store = DataStore()
    for u in users:
        train_store.add_user(u)
    for it in items:
        train_store.add_item(it)
    train_store.add_interactions_batch(train_ix)

    reports = evaluator.compare_models(train_store, test_ix)
    assert len(reports) == 4  # user_cf, item_cf, mf, hybrid
    for report in reports:
        assert report.model_name != ""
        assert report.k == 5
        assert 0.0 <= report.precision_at_k <= 1.0
        assert 0.0 <= report.recall_at_k <= 1.0
