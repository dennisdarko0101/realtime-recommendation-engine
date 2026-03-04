"""Unit tests for content-based filtering recommendation models."""

import numpy as np
import pytest

from src.models.content_based import ContentBasedFilter, ItemEmbedder
from src.data.schemas import Item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_items(n: int = 5) -> list[Item]:
    """Create a small set of items with distinct categories and tags."""
    specs = [
        ("item_0", "electronics", ["laptop", "portable"], "A lightweight portable laptop for professionals"),
        ("item_1", "electronics", ["phone", "smart"], "A smart phone with excellent camera quality"),
        ("item_2", "clothing", ["jacket", "winter"], "A warm winter jacket with insulation"),
        ("item_3", "clothing", ["shoes", "running"], "Lightweight running shoes for daily training"),
        ("item_4", "food", ["organic", "snack"], "Organic trail-mix snack for healthy eating"),
    ]
    items = []
    for item_id, category, tags, description in specs[:n]:
        items.append(
            Item(
                item_id=item_id,
                category=category,
                tags=tags,
                description=description,
            )
        )
    return items


class MockEmbeddingModel:
    """A deterministic embedding model for testing.

    Returns a fixed-size embedding derived from the hash of each text so that
    identical texts always produce identical vectors.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        rng = np.random.RandomState(42)
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = hash(text) % (2**31)
            local_rng = np.random.RandomState(seed)
            embeddings[i] = local_rng.randn(self.dim).astype(np.float32)
        return embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def items():
    return _make_items(5)


@pytest.fixture()
def fitted_tfidf(items):
    cbf = ContentBasedFilter()
    cbf.fit_tfidf(items)
    return cbf, items


@pytest.fixture()
def fitted_features(items):
    cbf = ContentBasedFilter()
    cbf.fit_features(items)
    return cbf, items


@pytest.fixture()
def fitted_embeddings(items):
    cbf = ContentBasedFilter()
    model = MockEmbeddingModel(dim=32)
    cbf.fit_embeddings(items, model)
    return cbf, items


# ===========================================================================
# ContentBasedFilter -- TF-IDF
# ===========================================================================

class TestContentBasedTfidf:
    """Tests for TF-IDF based content filtering."""

    def test_fit_tfidf_populates_matrix(self, fitted_tfidf):
        cbf, items = fitted_tfidf
        assert cbf._tfidf_matrix is not None
        assert cbf._tfidf_matrix.shape[0] == len(items)

    def test_recommend_returns_items(self, fitted_tfidf):
        cbf, items = fitted_tfidf
        recs = cbf.recommend("u0", k=3, user_history=["item_0"])
        assert isinstance(recs, list)
        for rec in recs:
            assert hasattr(rec, "item_id")
            assert hasattr(rec, "score")

    def test_recommend_excludes_history(self, fitted_tfidf):
        cbf, items = fitted_tfidf
        recs = cbf.recommend("u0", k=5, user_history=["item_0", "item_1"])
        rec_ids = {r.item_id for r in recs}
        assert "item_0" not in rec_ids
        assert "item_1" not in rec_ids

    def test_recommend_empty_history_returns_empty(self, fitted_tfidf):
        cbf, _ = fitted_tfidf
        recs = cbf.recommend("u0", k=5, user_history=[])
        assert recs == []

    def test_recommend_none_history_returns_empty(self, fitted_tfidf):
        cbf, _ = fitted_tfidf
        recs = cbf.recommend("u0", k=5, user_history=None)
        assert recs == []

    def test_get_similar_items(self, fitted_tfidf):
        cbf, items = fitted_tfidf
        recs = cbf.get_similar_items("item_0", k=3, method="tfidf")
        assert isinstance(recs, list)
        assert len(recs) > 0
        # item_0 should not appear in its own similar items
        rec_ids = {r.item_id for r in recs}
        assert "item_0" not in rec_ids
        # All returned scores should be positive
        for rec in recs:
            assert rec.score > 0

    def test_get_similar_items_unknown_item(self, fitted_tfidf):
        cbf, _ = fitted_tfidf
        recs = cbf.get_similar_items("nonexistent", k=3, method="tfidf")
        assert recs == []

    def test_method_parameter_tfidf(self, fitted_tfidf):
        cbf, _ = fitted_tfidf
        recs = cbf.recommend("u0", k=3, user_history=["item_0"], method="tfidf")
        for rec in recs:
            assert rec.model_used == "content_tfidf"


# ===========================================================================
# ContentBasedFilter -- Features
# ===========================================================================

class TestContentBasedFeatures:
    """Tests for structured-feature based content filtering."""

    def test_fit_features_populates_matrix(self, fitted_features):
        cbf, items = fitted_features
        assert cbf._feature_matrix is not None
        assert cbf._feature_matrix.shape[0] == len(items)

    def test_recommend_via_features(self, fitted_features):
        cbf, items = fitted_features
        recs = cbf.recommend("u0", k=3, user_history=["item_0"], method="feature")
        assert isinstance(recs, list)

    def test_same_category_higher_similarity(self, fitted_features):
        cbf, items = fitted_features
        recs = cbf.get_similar_items("item_0", k=4, method="feature")
        if recs:
            # item_1 shares "electronics" category with item_0
            assert recs[0].item_id == "item_1"

    def test_feature_method_label(self, fitted_features):
        cbf, _ = fitted_features
        recs = cbf.recommend("u0", k=3, user_history=["item_2"], method="feature")
        for rec in recs:
            assert rec.model_used == "content_feature"


# ===========================================================================
# ContentBasedFilter -- Embeddings
# ===========================================================================

class TestContentBasedEmbeddings:
    """Tests for embedding-based content filtering."""

    def test_fit_embeddings_populates_matrix(self, fitted_embeddings):
        cbf, items = fitted_embeddings
        assert cbf._embedding_matrix is not None
        assert cbf._embedding_matrix.shape == (len(items), 32)

    def test_recommend_via_embeddings(self, fitted_embeddings):
        cbf, _ = fitted_embeddings
        recs = cbf.recommend("u0", k=3, user_history=["item_0"], method="embedding")
        assert isinstance(recs, list)

    def test_embedding_method_label(self, fitted_embeddings):
        cbf, _ = fitted_embeddings
        recs = cbf.recommend("u0", k=3, user_history=["item_0"], method="embedding")
        for rec in recs:
            assert rec.model_used == "content_embedding"


# ===========================================================================
# ItemEmbedder
# ===========================================================================

class TestItemEmbedder:
    """Tests for the ItemEmbedder vector search wrapper."""

    def test_build_index(self, items):
        embedder = ItemEmbedder(collection_name="test")
        model = MockEmbeddingModel(dim=16)
        embedder.build_index(items, model)
        assert embedder._embeddings is not None
        assert embedder._embeddings.shape == (len(items), 16)
        assert len(embedder._item_ids) == len(items)

    def test_query_returns_results(self, items):
        embedder = ItemEmbedder(collection_name="test")
        model = MockEmbeddingModel(dim=16)
        embedder.build_index(items, model)
        query_vec = np.random.RandomState(99).randn(16).astype(np.float32)
        recs = embedder.query(query_vec, k=3)
        assert isinstance(recs, list)
        assert len(recs) <= 3
        for rec in recs:
            assert rec.item_id in {it.item_id for it in items}
            assert rec.score > 0

    def test_query_before_build_returns_empty(self):
        embedder = ItemEmbedder()
        query_vec = np.random.randn(16).astype(np.float32)
        recs = embedder.query(query_vec, k=5)
        assert recs == []

    def test_query_k_limits_results(self, items):
        embedder = ItemEmbedder()
        model = MockEmbeddingModel(dim=16)
        embedder.build_index(items, model)
        query_vec = np.random.RandomState(7).randn(16).astype(np.float32)
        recs = embedder.query(query_vec, k=2)
        assert len(recs) <= 2

    def test_model_used_field(self, items):
        embedder = ItemEmbedder()
        model = MockEmbeddingModel(dim=16)
        embedder.build_index(items, model)
        query_vec = np.random.RandomState(7).randn(16).astype(np.float32)
        recs = embedder.query(query_vec, k=3)
        for rec in recs:
            assert rec.model_used == "embedding_cosine"


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestContentEdgeCases:
    """Edge cases for content-based filtering."""

    def test_no_items(self):
        """fit_tfidf on an empty list raises ValueError from TfidfVectorizer."""
        cbf = ContentBasedFilter()
        with pytest.raises(ValueError):
            cbf.fit_tfidf([])

    def test_single_item(self):
        items = _make_items(1)
        cbf = ContentBasedFilter()
        cbf.fit_tfidf(items)
        # Only one item and it is in history => nothing to recommend
        recs = cbf.recommend("u0", k=5, user_history=["item_0"])
        assert recs == []

    def test_get_similar_items_single_item(self):
        items = _make_items(1)
        cbf = ContentBasedFilter()
        cbf.fit_tfidf(items)
        recs = cbf.get_similar_items("item_0", k=3)
        assert recs == []

    def test_history_item_not_in_catalogue(self, fitted_tfidf):
        cbf, _ = fitted_tfidf
        recs = cbf.recommend("u0", k=3, user_history=["does_not_exist"])
        assert recs == []

    def test_no_matching_features_tfidf_raises(self):
        """Items with completely empty text cause TfidfVectorizer to raise."""
        items = [
            Item(item_id="a", category="", tags=[], description=""),
            Item(item_id="b", category="", tags=[], description=""),
        ]
        cbf = ContentBasedFilter()
        with pytest.raises(ValueError):
            cbf.fit_tfidf(items)

    def test_no_matching_features_fit_features_works(self):
        """fit_features should handle items with empty categories/tags gracefully."""
        items = [
            Item(item_id="a", category="", tags=[], description=""),
            Item(item_id="b", category="", tags=[], description=""),
        ]
        cbf = ContentBasedFilter()
        cbf.fit_features(items)
        recs = cbf.recommend("u0", k=3, user_history=["a"], method="feature")
        assert isinstance(recs, list)
