"""Unit tests for collaborative filtering recommendation models."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.collaborative import UserBasedCF, ItemBasedCF, MatrixFactorization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(data: list[list[float]]) -> csr_matrix:
    """Convert a dense 2-D list into a CSR sparse matrix."""
    return csr_matrix(np.array(data, dtype=np.float32))


def _user_index(n: int) -> dict[str, int]:
    return {f"u{i}": i for i in range(n)}


def _item_index(n: int) -> dict[str, int]:
    return {f"i{i}": i for i in range(n)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_matrix():
    """4 users x 5 items matrix with clear preference patterns.

    u0 and u1 share items i0, i1  (similar users)
    u2 prefers items i2, i3       (different cluster)
    u3 prefers items i3, i4       (overlaps with u2)
    """
    data = [
        [5, 4, 0, 0, 0],  # u0
        [4, 5, 0, 0, 0],  # u1
        [0, 0, 5, 4, 0],  # u2
        [0, 0, 0, 4, 5],  # u3
    ]
    mat = _make_matrix(data)
    return mat, _user_index(4), _item_index(5)


@pytest.fixture()
def sparse_matrix():
    """6 users x 8 items sparse interaction matrix."""
    data = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
    ]
    mat = _make_matrix(data)
    return mat, _user_index(6), _item_index(8)


# ===========================================================================
# UserBasedCF
# ===========================================================================

class TestUserBasedCF:
    """Tests for user-based collaborative filtering."""

    def test_fit_stores_matrix(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = UserBasedCF(n_neighbors=2)
        model.fit(mat, uidx, iidx)
        assert model._interaction_matrix is not None
        assert model._knn is not None

    def test_recommend_returns_list_of_recommendations(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = UserBasedCF(n_neighbors=2)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u0", k=3)
        assert isinstance(recs, list)
        for rec in recs:
            assert hasattr(rec, "item_id")
            assert hasattr(rec, "score")
            assert isinstance(rec.score, float)

    def test_recommend_excludes_seen_items(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = UserBasedCF(n_neighbors=3)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u0", k=5)
        rec_item_ids = {r.item_id for r in recs}
        # u0 interacted with i0, i1 -- those must not appear
        assert "i0" not in rec_item_ids
        assert "i1" not in rec_item_ids

    def test_recommend_unknown_user_returns_empty(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = UserBasedCF(n_neighbors=2)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("unknown_user", k=5)
        assert recs == []

    def test_fit_with_empty_matrix(self):
        mat = csr_matrix((0, 5))
        model = UserBasedCF(n_neighbors=3)
        model.fit(mat, {}, _item_index(5))
        recs = model.recommend("u0", k=5)
        assert recs == []

    def test_k_parameter_limits_results(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = UserBasedCF(n_neighbors=3)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u2", k=1)
        assert len(recs) <= 1

    def test_neighbor_similarity_correct(self, simple_matrix):
        """u0 and u1 are most similar -- u1's items should influence u0's recs."""
        mat, uidx, iidx = simple_matrix
        model = UserBasedCF(n_neighbors=3)
        model.fit(mat, uidx, iidx)
        recs_u0 = model.recommend("u0", k=5)
        recs_u2 = model.recommend("u2", k=5)
        # u0 should NOT get strong recs for i3/i4 because its neighbors are
        # u1 (similar) who also only has i0/i1.  u2 should get i4 from u3.
        u2_ids = {r.item_id for r in recs_u2}
        assert "i4" in u2_ids  # u3 is u2's neighbor and liked i4

    def test_single_user_matrix(self):
        """Only one user => no neighbors => empty recs."""
        mat = _make_matrix([[3, 0, 2]])
        model = UserBasedCF(n_neighbors=5)
        model.fit(mat, {"u0": 0}, _item_index(3))
        recs = model.recommend("u0", k=3)
        assert recs == []

    def test_n_neighbors_exceeds_users(self):
        """n_neighbors > n_users should still work (clamped internally)."""
        mat = _make_matrix([[1, 2], [2, 1]])
        model = UserBasedCF(n_neighbors=100)
        model.fit(mat, _user_index(2), _item_index(2))
        # Should not raise
        recs = model.recommend("u0", k=5)
        assert isinstance(recs, list)

    def test_model_used_field(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = UserBasedCF(n_neighbors=3)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u2", k=3)
        for r in recs:
            assert r.model_used == "user_based_cf"


# ===========================================================================
# ItemBasedCF
# ===========================================================================

class TestItemBasedCF:
    """Tests for item-based collaborative filtering."""

    def test_fit_builds_similarity_matrix(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = ItemBasedCF()
        model.fit(mat, uidx, iidx)
        assert model._item_similarity is not None
        assert model._item_similarity.shape == (5, 5)

    def test_recommend_returns_items(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = ItemBasedCF()
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u0", k=3)
        assert isinstance(recs, list)
        for rec in recs:
            assert rec.item_id.startswith("i")

    def test_excludes_interacted_items(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = ItemBasedCF()
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u0", k=5)
        rec_ids = {r.item_id for r in recs}
        assert "i0" not in rec_ids
        assert "i1" not in rec_ids

    def test_handles_sparse_data(self, sparse_matrix):
        mat, uidx, iidx = sparse_matrix
        model = ItemBasedCF()
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u0", k=3)
        assert isinstance(recs, list)

    def test_similarity_ordering(self):
        """Items co-interacted by same users should have higher similarity."""
        # i0 and i1 always appear together; i2 is independent
        data = [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
        mat = _make_matrix(data)
        model = ItemBasedCF()
        model.fit(mat, _user_index(3), _item_index(3))
        sim = model._item_similarity
        # similarity(i0, i1) > similarity(i0, i2)
        assert sim[0, 1] > sim[0, 2]

    def test_unknown_user_returns_empty(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = ItemBasedCF()
        model.fit(mat, uidx, iidx)
        recs = model.recommend("nonexistent", k=5)
        assert recs == []

    def test_model_used_field(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = ItemBasedCF()
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u2", k=3)
        for r in recs:
            assert r.model_used == "item_based_cf"

    def test_empty_item_matrix(self):
        mat = csr_matrix((3, 0))
        model = ItemBasedCF()
        model.fit(mat, _user_index(3), {})
        recs = model.recommend("u0", k=3)
        assert recs == []


# ===========================================================================
# MatrixFactorization
# ===========================================================================

class TestMatrixFactorization:
    """Tests for SVD-based matrix factorization."""

    def test_fit_performs_svd(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = MatrixFactorization(n_factors=2)
        model.fit(mat, uidx, iidx)
        assert model._user_factors is not None
        assert model._item_factors is not None
        assert model._sigma is not None

    def test_recommend_returns_items(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = MatrixFactorization(n_factors=2)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u0", k=3)
        assert isinstance(recs, list)
        for rec in recs:
            assert rec.item_id.startswith("i")

    def test_recommend_unknown_user(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = MatrixFactorization(n_factors=2)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("ghost", k=5)
        assert recs == []

    def test_n_factors_parameter(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = MatrixFactorization(n_factors=1)
        model.fit(mat, uidx, iidx)
        assert model._user_factors.shape[1] == 1
        assert model._item_factors.shape[1] == 1

    def test_scores_are_finite(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = MatrixFactorization(n_factors=2)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u0", k=5)
        for rec in recs:
            assert np.isfinite(rec.score)

    def test_excludes_seen_items(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = MatrixFactorization(n_factors=2)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u0", k=5)
        rec_ids = {r.item_id for r in recs}
        assert "i0" not in rec_ids
        assert "i1" not in rec_ids

    def test_model_used_field(self, simple_matrix):
        mat, uidx, iidx = simple_matrix
        model = MatrixFactorization(n_factors=2)
        model.fit(mat, uidx, iidx)
        recs = model.recommend("u2", k=3)
        for r in recs:
            assert r.model_used == "matrix_factorization"

    def test_n_factors_clamped(self):
        """n_factors larger than min(n_users, n_items)-1 should still work."""
        data = [[1, 2], [3, 4], [5, 6]]  # 3x2 => max factors = 1
        mat = _make_matrix(data)
        model = MatrixFactorization(n_factors=50)
        model.fit(mat, _user_index(3), _item_index(2))
        assert model._user_factors is not None
        assert model._user_factors.shape[1] == 1

    def test_fit_without_recommend(self, simple_matrix):
        """fit should succeed without calling recommend."""
        mat, uidx, iidx = simple_matrix
        model = MatrixFactorization(n_factors=2)
        model.fit(mat, uidx, iidx)
        assert model._user_factors.shape[0] == 4
        assert model._item_factors.shape[0] == 5


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases across all collaborative models."""

    @pytest.mark.parametrize("ModelClass", [UserBasedCF, ItemBasedCF])
    def test_all_zeros_matrix(self, ModelClass):
        """A matrix of all zeros should yield no recommendations."""
        data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        mat = _make_matrix(data)
        if ModelClass == UserBasedCF:
            model = ModelClass(n_neighbors=2)
        else:
            model = ModelClass()
        model.fit(mat, _user_index(3), _item_index(3))
        recs = model.recommend("u0", k=5)
        assert recs == []

    def test_all_zeros_matrix_mf(self):
        """MatrixFactorization on an all-zero matrix raises an ARPACK error
        because SVD cannot decompose a zero matrix.  Verify graceful handling."""
        from scipy.sparse.linalg import ArpackError

        data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        mat = _make_matrix(data)
        model = MatrixFactorization(n_factors=1)
        with pytest.raises(ArpackError):
            model.fit(mat, _user_index(3), _item_index(3))

    @pytest.mark.parametrize("ModelClass", [ItemBasedCF, MatrixFactorization])
    def test_single_item(self, ModelClass):
        """Single item means nothing new to recommend."""
        data = [[5], [3]]
        mat = _make_matrix(data)
        if ModelClass == MatrixFactorization:
            model = ModelClass(n_factors=1)
        else:
            model = ModelClass()
        model.fit(mat, _user_index(2), _item_index(1))
        recs = model.recommend("u0", k=5)
        # The only item was already seen, so no recs
        assert recs == []

    def test_single_user_item_based(self):
        """Single user with multiple items -- item similarity still works."""
        data = [[3, 0, 5]]
        mat = _make_matrix(data)
        model = ItemBasedCF()
        model.fit(mat, {"u0": 0}, _item_index(3))
        recs = model.recommend("u0", k=5)
        # u0 has i0=3, i2=5; i1 is unseen
        # item similarity between i1 and {i0,i2} is 0 because i1 has zero column
        assert isinstance(recs, list)

    def test_no_interactions_after_fit(self):
        """Matrix with shape but no non-zero entries."""
        mat = csr_matrix((5, 10))
        model = UserBasedCF(n_neighbors=3)
        model.fit(mat, _user_index(5), _item_index(10))
        recs = model.recommend("u0", k=5)
        assert recs == []

    def test_recommend_before_fit(self):
        """Calling recommend before fit should return empty list."""
        model = UserBasedCF(n_neighbors=3)
        recs = model.recommend("u0", k=5)
        assert recs == []

    def test_item_based_recommend_before_fit(self):
        model = ItemBasedCF()
        recs = model.recommend("u0", k=5)
        assert recs == []

    def test_mf_recommend_before_fit(self):
        model = MatrixFactorization(n_factors=5)
        recs = model.recommend("u0", k=5)
        assert recs == []
