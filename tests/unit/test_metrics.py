"""Tests for RecommendationMetrics."""

import numpy as np
import pytest

from src.evaluation.metrics import RecommendationMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metrics() -> RecommendationMetrics:
    return RecommendationMetrics()


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_precision(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert RecommendationMetrics.precision_at_k(recommended, relevant, k=3) == 1.0

    def test_partial_precision(self):
        recommended = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        # 2 hits in top 4 -> 0.5
        assert RecommendationMetrics.precision_at_k(recommended, relevant, k=4) == 0.5

    def test_no_hits(self):
        recommended = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert RecommendationMetrics.precision_at_k(recommended, relevant, k=3) == 0.0

    def test_k_larger_than_list(self):
        recommended = ["a", "b"]
        relevant = {"a"}
        # Only 2 items even though k=5, precision = 1/2
        assert RecommendationMetrics.precision_at_k(recommended, relevant, k=5) == 0.5

    def test_empty_recommendations(self):
        assert RecommendationMetrics.precision_at_k([], {"a"}, k=5) == 0.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert RecommendationMetrics.recall_at_k(recommended, relevant, k=3) == 1.0

    def test_partial_recall(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "d", "e", "f"}
        # 1 hit out of 4 relevant -> 0.25
        assert RecommendationMetrics.recall_at_k(recommended, relevant, k=3) == 0.25

    def test_empty_relevant_set(self):
        recommended = ["a", "b"]
        assert RecommendationMetrics.recall_at_k(recommended, set(), k=2) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ordering(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = RecommendationMetrics.ndcg_at_k(recommended, relevant, k=3)
        assert result == pytest.approx(1.0)

    def test_reversed_ordering(self):
        # All relevant items but checking less-than-perfect case
        # Recommend ["x", "a"] with relevant={"a"}: hit at position 2
        recommended = ["x", "a"]
        relevant = {"a"}
        result = RecommendationMetrics.ndcg_at_k(recommended, relevant, k=2)
        # DCG = 1/log2(3) = 0.6309..., IDCG = 1/log2(2) = 1.0
        expected = (1.0 / np.log2(3)) / (1.0 / np.log2(2))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_partial_hits(self):
        recommended = ["a", "x", "b"]
        relevant = {"a", "b"}
        result = RecommendationMetrics.ndcg_at_k(recommended, relevant, k=3)
        # DCG = 1/log2(2) + 1/log2(4)
        dcg = 1.0 / np.log2(2) + 1.0 / np.log2(4)
        # IDCG = 1/log2(2) + 1/log2(3) (ideal: both relevant at top)
        idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3)
        assert result == pytest.approx(dcg / idcg, rel=1e-4)

    def test_no_relevant_returns_zero(self):
        assert RecommendationMetrics.ndcg_at_k(["a", "b"], set(), k=2) == 0.0


# ---------------------------------------------------------------------------
# map_at_k
# ---------------------------------------------------------------------------

class TestMAPAtK:
    def test_perfect_map(self):
        recommended = ["a", "b"]
        relevant = {"a", "b"}
        # hit@1: prec=1/1, hit@2: prec=2/2 -> AP = (1 + 1)/2 = 1.0
        assert RecommendationMetrics.map_at_k(recommended, relevant, k=2) == pytest.approx(1.0)

    def test_known_map_value(self):
        recommended = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        # hit@1: prec=1/1=1.0, hit@3: prec=2/3
        # AP = (1.0 + 2/3) / min(2, 4) = (1.0 + 0.6667) / 2 = 0.8333...
        expected = (1.0 + 2.0 / 3.0) / 2.0
        assert RecommendationMetrics.map_at_k(recommended, relevant, k=4) == pytest.approx(
            expected, rel=1e-4
        )

    def test_no_hits_map(self):
        assert RecommendationMetrics.map_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------

class TestHitRate:
    def test_hit(self):
        assert RecommendationMetrics.hit_rate(["a", "b", "c"], {"b"}) == 1.0

    def test_miss(self):
        assert RecommendationMetrics.hit_rate(["a", "b"], {"x", "y"}) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_position(self):
        assert RecommendationMetrics.mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_middle_position(self):
        assert RecommendationMetrics.mrr(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)

    def test_no_hit(self):
        assert RecommendationMetrics.mrr(["x", "y"], {"a"}) == 0.0

    def test_second_position(self):
        assert RecommendationMetrics.mrr(["x", "a", "b"], {"a", "b"}) == 0.5


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_full_coverage(self):
        all_recs = [["a", "b"], ["c", "d"], ["e"]]
        assert RecommendationMetrics.coverage(all_recs, total_items=5) == 1.0

    def test_partial_coverage(self):
        all_recs = [["a", "b"], ["a", "c"]]
        # unique items: a, b, c -> 3/10
        assert RecommendationMetrics.coverage(all_recs, total_items=10) == pytest.approx(0.3)

    def test_zero_total_items(self):
        assert RecommendationMetrics.coverage([["a"]], total_items=0) == 0.0


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------

class TestDiversity:
    def test_all_same_category(self):
        item_cats = {"a": "books", "b": "books", "c": "books"}
        result = RecommendationMetrics.diversity(["a", "b", "c"], item_cats)
        # 1 unique category / 3 items
        assert result == pytest.approx(1.0 / 3.0)

    def test_all_different_categories(self):
        item_cats = {"a": "books", "b": "electronics", "c": "clothing"}
        result = RecommendationMetrics.diversity(["a", "b", "c"], item_cats)
        assert result == pytest.approx(1.0)

    def test_empty_recommendations(self):
        assert RecommendationMetrics.diversity([], {}) == 0.0


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------

class TestNovelty:
    def test_popular_items_low_novelty(self):
        # Very popular items -> low self-information
        popularity = {"a": 900, "b": 800}
        result = RecommendationMetrics.novelty(["a", "b"], popularity, total_interactions=1000)
        assert result < 1.0  # low novelty

    def test_rare_items_high_novelty(self):
        # Very rare items -> high self-information
        popularity = {"a": 1, "b": 2}
        result = RecommendationMetrics.novelty(["a", "b"], popularity, total_interactions=1000)
        assert result > 5.0  # high novelty

    def test_rare_more_novel_than_popular(self):
        popularity = {"popular": 500, "rare": 1}
        pop_novelty = RecommendationMetrics.novelty(
            ["popular"], popularity, total_interactions=1000
        )
        rare_novelty = RecommendationMetrics.novelty(
            ["rare"], popularity, total_interactions=1000
        )
        assert rare_novelty > pop_novelty

    def test_empty_recommendations_returns_zero(self):
        assert RecommendationMetrics.novelty([], {}, total_interactions=100) == 0.0

    def test_zero_total_interactions_returns_zero(self):
        assert RecommendationMetrics.novelty(["a"], {"a": 5}, total_interactions=0) == 0.0


# ---------------------------------------------------------------------------
# evaluate (combined)
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_metrics_report(self, metrics: RecommendationMetrics):
        recommended_per_user = {
            "u1": ["a", "b", "c"],
            "u2": ["d", "e", "f"],
        }
        relevant_per_user = {
            "u1": {"a", "c"},
            "u2": {"d"},
        }
        report = metrics.evaluate(
            recommended_per_user=recommended_per_user,
            relevant_per_user=relevant_per_user,
            k=3,
            total_items=10,
            item_categories={"a": "books", "b": "books", "c": "electronics",
                             "d": "clothing", "e": "clothing", "f": "food"},
            item_popularity={"a": 10, "b": 20, "c": 5, "d": 30, "e": 15, "f": 1},
            total_interactions=100,
            model_name="test_model",
        )
        assert report.k == 3
        assert report.model_name == "test_model"
        assert 0.0 <= report.precision_at_k <= 1.0
        assert 0.0 <= report.recall_at_k <= 1.0
        assert 0.0 <= report.ndcg_at_k <= 1.0
        assert 0.0 <= report.hit_rate <= 1.0
        assert 0.0 <= report.mrr <= 1.0
        assert report.coverage > 0.0
        assert report.diversity > 0.0
        assert report.novelty > 0.0

    def test_evaluate_perfect_recommendations(self, metrics: RecommendationMetrics):
        recommended_per_user = {
            "u1": ["a", "b"],
        }
        relevant_per_user = {
            "u1": {"a", "b"},
        }
        report = metrics.evaluate(
            recommended_per_user=recommended_per_user,
            relevant_per_user=relevant_per_user,
            k=2,
        )
        assert report.precision_at_k == pytest.approx(1.0)
        assert report.recall_at_k == pytest.approx(1.0)
        assert report.ndcg_at_k == pytest.approx(1.0)
        assert report.hit_rate == pytest.approx(1.0)
        assert report.mrr == pytest.approx(1.0)

    def test_evaluate_no_relevant_users_skipped(self, metrics: RecommendationMetrics):
        recommended_per_user = {
            "u1": ["a", "b"],
        }
        relevant_per_user = {
            "u1": set(),  # empty relevant set -> user is skipped
        }
        report = metrics.evaluate(
            recommended_per_user=recommended_per_user,
            relevant_per_user=relevant_per_user,
            k=2,
        )
        # All metrics default to 0 when no users have relevant items
        assert report.precision_at_k == 0.0
        assert report.recall_at_k == 0.0
