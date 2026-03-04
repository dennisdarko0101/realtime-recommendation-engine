"""Content-based filtering using TF-IDF, embeddings, and feature similarity."""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.schemas import Item, Recommendation


class EmbeddingModel(Protocol):
    """Protocol for embedding models (allows mocking sentence-transformers)."""

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray: ...


class ContentBasedFilter:
    """Content-based recommendation using TF-IDF, embeddings, or structured features."""

    def __init__(self) -> None:
        self._items: list[Item] = []
        self._item_index: dict[str, int] = {}
        self._tfidf_matrix: np.ndarray | None = None
        self._embedding_matrix: np.ndarray | None = None
        self._feature_matrix: np.ndarray | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None

    def fit_tfidf(self, items: list[Item]) -> None:
        """Fit TF-IDF model on item descriptions and tags."""
        self._items = items
        self._item_index = {it.item_id: i for i, it in enumerate(items)}

        texts = []
        for it in items:
            text = f"{it.category} {' '.join(it.tags)} {it.description}"
            texts.append(text)

        self._tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts).toarray()

    def fit_embeddings(self, items: list[Item], model: EmbeddingModel) -> None:
        """Fit using sentence-transformer embeddings."""
        self._items = items
        self._item_index = {it.item_id: i for i, it in enumerate(items)}

        texts = [
            f"{it.category}: {' '.join(it.tags)}. {it.description}" for it in items
        ]
        self._embedding_matrix = model.encode(texts, show_progress_bar=False)

    def fit_features(self, items: list[Item], categories: Optional[list[str]] = None) -> None:
        """Fit using structured features (category one-hot + tag overlap)."""
        self._items = items
        self._item_index = {it.item_id: i for i, it in enumerate(items)}

        if categories is None:
            categories = sorted({it.category for it in items})
        all_tags = sorted({t for it in items for t in it.tags})

        cat_map = {c: i for i, c in enumerate(categories)}
        tag_map = {t: i for i, t in enumerate(all_tags)}

        n_features = len(categories) + len(all_tags)
        matrix = np.zeros((len(items), n_features))

        for i, it in enumerate(items):
            if it.category in cat_map:
                matrix[i, cat_map[it.category]] = 1.0
            for t in it.tags:
                if t in tag_map:
                    matrix[i, len(categories) + tag_map[t]] = 1.0

        self._feature_matrix = matrix

    def _get_similarity_matrix(self, method: str) -> np.ndarray | None:
        if method == "tfidf" and self._tfidf_matrix is not None:
            return cosine_similarity(self._tfidf_matrix)
        elif method == "embedding" and self._embedding_matrix is not None:
            return cosine_similarity(self._embedding_matrix)
        elif method == "feature" and self._feature_matrix is not None:
            return cosine_similarity(self._feature_matrix)
        return None

    def get_similar_items(
        self, item_id: str, k: int = 10, method: str = "tfidf"
    ) -> list[Recommendation]:
        if item_id not in self._item_index:
            return []

        sim_matrix = self._get_similarity_matrix(method)
        if sim_matrix is None:
            return []

        idx = self._item_index[item_id]
        scores = sim_matrix[idx]
        scores[idx] = -1  # Exclude self

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for i in top_indices:
            if scores[i] <= 0:
                break
            results.append(
                Recommendation(
                    item_id=self._items[i].item_id,
                    score=float(scores[i]),
                    reason=f"Content similar ({method})",
                    model_used=f"content_{method}",
                )
            )
        return results

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        user_history: Optional[list[str]] = None,
        method: str = "tfidf",
    ) -> list[Recommendation]:
        """Recommend items based on content similarity to user's history."""
        if not user_history:
            return []

        sim_matrix = self._get_similarity_matrix(method)
        if sim_matrix is None:
            return []

        history_indices = [
            self._item_index[iid] for iid in user_history if iid in self._item_index
        ]
        if not history_indices:
            return []

        # Average similarity to all items in user history
        scores = np.mean(sim_matrix[history_indices], axis=0)

        # Exclude items already in history
        for idx in history_indices:
            scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for i in top_indices:
            if scores[i] <= 0 or np.isinf(scores[i]):
                break
            results.append(
                Recommendation(
                    item_id=self._items[i].item_id,
                    score=float(scores[i]),
                    reason=f"Similar to items in your history ({method})",
                    model_used=f"content_{method}",
                )
            )
        return results


class ItemEmbedder:
    """Embed items into vector space for fast similarity search.

    Uses ChromaDB for approximate nearest neighbor when available,
    falls back to in-memory cosine similarity.
    """

    def __init__(self, collection_name: str = "items") -> None:
        self._collection_name = collection_name
        self._collection = None
        self._embeddings: np.ndarray | None = None
        self._item_ids: list[str] = []

    def build_index(
        self,
        items: list[Item],
        model: EmbeddingModel,
        chroma_client: object | None = None,
    ) -> None:
        """Build embedding index for items."""
        texts = [
            f"{it.category}: {' '.join(it.tags)}. {it.description}" for it in items
        ]
        embeddings = model.encode(texts, show_progress_bar=False)
        self._embeddings = embeddings
        self._item_ids = [it.item_id for it in items]

        if chroma_client is not None:
            try:
                collection = chroma_client.get_or_create_collection(self._collection_name)
                collection.upsert(
                    ids=self._item_ids,
                    embeddings=embeddings.tolist(),
                    metadatas=[{"category": it.category} for it in items],
                )
                self._collection = collection
            except Exception:
                pass  # Fall back to in-memory

    def query(self, embedding: np.ndarray, k: int = 10) -> list[Recommendation]:
        """Find nearest items to a query embedding."""
        if self._collection is not None:
            try:
                results = self._collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=k,
                )
                recs = []
                for iid, dist in zip(results["ids"][0], results["distances"][0]):
                    recs.append(
                        Recommendation(
                            item_id=iid,
                            score=float(1.0 / (1.0 + dist)),
                            reason="Embedding similarity (ChromaDB)",
                            model_used="embedding_ann",
                        )
                    )
                return recs
            except Exception:
                pass

        # Fallback: in-memory cosine similarity
        if self._embeddings is None:
            return []

        query_vec = embedding.reshape(1, -1)
        sims = cosine_similarity(query_vec, self._embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_indices:
            if sims[idx] <= 0:
                break
            results.append(
                Recommendation(
                    item_id=self._item_ids[idx],
                    score=float(sims[idx]),
                    reason="Embedding similarity",
                    model_used="embedding_cosine",
                )
            )
        return results
