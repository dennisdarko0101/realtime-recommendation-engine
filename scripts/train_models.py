#!/usr/bin/env python3
"""Train all recommendation models on the generated dataset."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.store import DataStore
from src.evaluation.offline_eval import OfflineEvaluator
from src.models.collaborative import ItemBasedCF, MatrixFactorization, UserBasedCF
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender
from src.models.popular import PopularityRecommender


def main() -> None:
    data_dir = Path("data/store")
    if not data_dir.exists():
        print("No data found. Run 'python scripts/generate_data.py' first.")
        return

    print("Loading data...")
    store = DataStore(str(data_dir))
    store.load()
    print(f"  Users: {store.n_users}, Items: {store.n_items}, Interactions: {store.n_interactions}")

    matrix = store.get_interaction_matrix()
    user_index = store.get_user_index()
    item_index = store.get_item_index()
    items = store.list_items()
    interactions = store.get_all_interactions()

    # Collaborative models
    for name, model in [
        ("UserBasedCF", UserBasedCF(n_neighbors=20)),
        ("ItemBasedCF", ItemBasedCF()),
        ("MatrixFactorization", MatrixFactorization(n_factors=50)),
    ]:
        t0 = time.time()
        model.fit(matrix, user_index, item_index)
        print(f"  {name} fitted in {time.time() - t0:.2f}s")

    # Content-based
    t0 = time.time()
    content = ContentBasedFilter()
    content.fit_tfidf(items)
    print(f"  ContentBased(TF-IDF) fitted in {time.time() - t0:.2f}s")

    # Popularity
    t0 = time.time()
    pop = PopularityRecommender()
    pop.fit(interactions, items)
    print(f"  Popularity fitted in {time.time() - t0:.2f}s")

    # Hybrid
    t0 = time.time()
    hybrid = HybridRecommender()
    hybrid.fit(store, content_model=content)
    print(f"  Hybrid fitted in {time.time() - t0:.2f}s")

    # Quick evaluation
    print("\nRunning offline evaluation...")
    evaluator = OfflineEvaluator(k=10)
    train_ix, test_ix = evaluator.temporal_split(interactions, 0.8)

    train_store = DataStore()
    for u in store.list_users():
        train_store.add_user(u)
    for it in store.list_items():
        train_store.add_item(it)
    train_store.add_interactions_batch(train_ix)

    reports = evaluator.compare_models(train_store, test_ix)
    print(f"\n{'Model':<25} {'P@10':<8} {'R@10':<8} {'NDCG@10':<8} {'Hit Rate':<8} {'Coverage':<8}")
    print("-" * 65)
    for r in reports:
        print(
            f"{r.model_name:<25} {r.precision_at_k:<8.4f} {r.recall_at_k:<8.4f} "
            f"{r.ndcg_at_k:<8.4f} {r.hit_rate:<8.4f} {r.coverage:<8.4f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
