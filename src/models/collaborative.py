"""Collaborative filtering recommendation models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from src.data.schemas import Recommendation


class CollaborativeFilter(ABC):
    """Base class for collaborative filtering models."""

    @abstractmethod
    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_index: dict[str, int],
        item_index: dict[str, int],
    ) -> None:
        ...

    @abstractmethod
    def recommend(self, user_id: str, k: int = 10) -> list[Recommendation]:
        ...

    def _index_to_item(self, idx: int) -> str:
        return self._idx_to_item.get(idx, f"unknown_{idx}")


class UserBasedCF(CollaborativeFilter):
    """User-based collaborative filtering using cosine similarity and KNN."""

    def __init__(self, n_neighbors: int = 20) -> None:
        self.n_neighbors = n_neighbors
        self._interaction_matrix: csr_matrix | None = None
        self._user_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._knn: NearestNeighbors | None = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_index: dict[str, int],
        item_index: dict[str, int],
    ) -> None:
        self._interaction_matrix = interaction_matrix
        self._user_index = user_index
        self._item_index = item_index
        self._idx_to_item = {v: k for k, v in item_index.items()}

        n_neighbors = min(self.n_neighbors, interaction_matrix.shape[0] - 1)
        if n_neighbors < 1:
            self._knn = None
            return

        self._knn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors, algorithm="brute")
        self._knn.fit(interaction_matrix)

    def recommend(self, user_id: str, k: int = 10) -> list[Recommendation]:
        if self._interaction_matrix is None or self._knn is None:
            return []
        if user_id not in self._user_index:
            return []

        user_idx = self._user_index[user_id]
        user_vec = self._interaction_matrix[user_idx]

        distances, indices = self._knn.kneighbors(user_vec, return_distance=True)

        # Aggregate neighbor preferences weighted by similarity
        n_items = self._interaction_matrix.shape[1]
        scores = np.zeros(n_items)

        for dist, neighbor_idx in zip(distances[0], indices[0]):
            similarity = 1.0 - dist  # cosine distance -> similarity
            if similarity <= 0:
                continue
            neighbor_vec = self._interaction_matrix[neighbor_idx].toarray().flatten()
            scores += similarity * neighbor_vec

        # Exclude items user already interacted with
        user_items = self._interaction_matrix[user_idx].toarray().flatten()
        scores[user_items > 0] = -np.inf

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append(
                Recommendation(
                    item_id=self._index_to_item(idx),
                    score=float(scores[idx]),
                    reason="Similar users liked this item",
                    model_used="user_based_cf",
                )
            )
        return results


class ItemBasedCF(CollaborativeFilter):
    """Item-based collaborative filtering using item-item cosine similarity."""

    def __init__(self) -> None:
        self._interaction_matrix: csr_matrix | None = None
        self._user_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._item_similarity: np.ndarray | None = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_index: dict[str, int],
        item_index: dict[str, int],
    ) -> None:
        self._interaction_matrix = interaction_matrix
        self._user_index = user_index
        self._item_index = item_index
        self._idx_to_item = {v: k for k, v in item_index.items()}

        # Compute item-item similarity (transpose so items are rows)
        item_matrix = interaction_matrix.T
        if item_matrix.shape[0] > 0:
            self._item_similarity = cosine_similarity(item_matrix)
        else:
            self._item_similarity = np.array([])

    def recommend(self, user_id: str, k: int = 10) -> list[Recommendation]:
        if self._interaction_matrix is None or self._item_similarity is None:
            return []
        if self._item_similarity.size == 0:
            return []
        if user_id not in self._user_index:
            return []

        user_idx = self._user_index[user_id]
        user_vec = self._interaction_matrix[user_idx].toarray().flatten()

        # Score each item based on similarity to items the user liked
        scores = self._item_similarity.dot(user_vec)

        # Exclude already-interacted items
        scores[user_vec > 0] = -np.inf

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append(
                Recommendation(
                    item_id=self._index_to_item(idx),
                    score=float(scores[idx]),
                    reason="Similar to items you liked",
                    model_used="item_based_cf",
                )
            )
        return results


class MatrixFactorization(CollaborativeFilter):
    """SVD-based matrix factorization for collaborative filtering."""

    def __init__(self, n_factors: int = 50) -> None:
        self.n_factors = n_factors
        self._user_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._interaction_matrix: csr_matrix | None = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_index: dict[str, int],
        item_index: dict[str, int],
    ) -> None:
        self._interaction_matrix = interaction_matrix
        self._user_index = user_index
        self._item_index = item_index
        self._idx_to_item = {v: k for k, v in item_index.items()}

        n_users, n_items = interaction_matrix.shape
        n_factors = min(self.n_factors, min(n_users, n_items) - 1)
        if n_factors < 1:
            return

        # Convert to float64 for SVD
        mat = interaction_matrix.astype(np.float64)
        U, sigma, Vt = svds(mat, k=n_factors)

        self._user_factors = U
        self._sigma = sigma
        self._item_factors = Vt.T  # (n_items, n_factors)

    def recommend(self, user_id: str, k: int = 10) -> list[Recommendation]:
        if self._user_factors is None or self._item_factors is None or self._sigma is None:
            return []
        if user_id not in self._user_index:
            return []

        user_idx = self._user_index[user_id]
        user_vec = self._user_factors[user_idx]  # (n_factors,)

        # Predicted scores: U[user] * Sigma * V^T
        scores = (user_vec * self._sigma) @ self._item_factors.T

        # Exclude already-interacted items
        if self._interaction_matrix is not None:
            user_items = self._interaction_matrix[user_idx].toarray().flatten()
            scores[user_items > 0] = -np.inf

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0 or np.isinf(scores[idx]):
                break
            results.append(
                Recommendation(
                    item_id=self._index_to_item(idx),
                    score=float(scores[idx]),
                    reason="Predicted from latent factor analysis",
                    model_used="matrix_factorization",
                )
            )
        return results
