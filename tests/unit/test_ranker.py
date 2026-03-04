"""Tests for RecommendationRanker."""

import pytest

from src.serving.ranker import RecommendationRanker
from src.data.schemas import Recommendation, Item, User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(item_id: str, score: float = 1.0) -> Recommendation:
    return Recommendation(item_id=item_id, score=score)


def _item(item_id: str, category: str = "general", price: float | None = None) -> Item:
    features: dict = {}
    if price is not None:
        features["price"] = price
    return Item(item_id=item_id, category=category, features=features)


# ---------------------------------------------------------------------------
# Test: remove already-seen items
# ---------------------------------------------------------------------------

class TestSeenItemRemoval:
    def test_removes_seen_items(self):
        ranker = RecommendationRanker()
        recs = [_rec("a"), _rec("b"), _rec("c")]
        result = ranker.rerank(recs, seen_item_ids={"b"})
        result_ids = [r.item_id for r in result]
        assert "b" not in result_ids
        assert "a" in result_ids
        assert "c" in result_ids

    def test_removes_multiple_seen_items(self):
        ranker = RecommendationRanker()
        recs = [_rec("a"), _rec("b"), _rec("c"), _rec("d")]
        result = ranker.rerank(recs, seen_item_ids={"a", "c"})
        result_ids = [r.item_id for r in result]
        assert "a" not in result_ids
        assert "c" not in result_ids
        assert len(result_ids) == 2


# ---------------------------------------------------------------------------
# Test: category diversity
# ---------------------------------------------------------------------------

class TestCategoryDiversity:
    def test_enforces_max_same_category(self):
        ranker = RecommendationRanker(max_same_category=2)
        recs = [_rec("a", 5.0), _rec("b", 4.0), _rec("c", 3.0), _rec("d", 2.0)]
        items = {
            "a": _item("a", "electronics"),
            "b": _item("b", "electronics"),
            "c": _item("c", "electronics"),
            "d": _item("d", "electronics"),
        }
        result = ranker.rerank(recs, items=items)
        assert len(result) == 2

    def test_all_same_category_capped(self):
        ranker = RecommendationRanker(max_same_category=3)
        recs = [_rec(f"item{i}", 10.0 - i) for i in range(6)]
        items = {f"item{i}": _item(f"item{i}", "books") for i in range(6)}
        result = ranker.rerank(recs, items=items)
        assert len(result) == 3

    def test_different_categories_not_capped(self):
        ranker = RecommendationRanker(max_same_category=2)
        recs = [_rec("a", 5.0), _rec("b", 4.0), _rec("c", 3.0), _rec("d", 2.0)]
        items = {
            "a": _item("a", "electronics"),
            "b": _item("b", "books"),
            "c": _item("c", "clothing"),
            "d": _item("d", "food"),
        }
        result = ranker.rerank(recs, items=items)
        assert len(result) == 4

    def test_configurable_max_same_category(self):
        for cap in [1, 2, 5]:
            ranker = RecommendationRanker(max_same_category=cap)
            recs = [_rec(f"i{i}", 10.0 - i) for i in range(10)]
            items = {f"i{i}": _item(f"i{i}", "cat_a") for i in range(10)}
            result = ranker.rerank(recs, items=items)
            assert len(result) == cap


# ---------------------------------------------------------------------------
# Test: price range filtering
# ---------------------------------------------------------------------------

class TestPriceRangeFiltering:
    def test_filters_items_outside_price_range(self):
        ranker = RecommendationRanker()
        recs = [_rec("cheap"), _rec("mid"), _rec("expensive")]
        items = {
            "cheap": _item("cheap", price=5.0),
            "mid": _item("mid", price=50.0),
            "expensive": _item("expensive", price=500.0),
        }
        result = ranker.rerank(recs, items=items, price_range=(10.0, 100.0))
        result_ids = [r.item_id for r in result]
        assert "mid" in result_ids
        assert "cheap" not in result_ids
        assert "expensive" not in result_ids

    def test_keeps_items_without_price(self):
        ranker = RecommendationRanker()
        recs = [_rec("no_price"), _rec("has_price")]
        items = {
            "no_price": _item("no_price"),
            "has_price": _item("has_price", price=50.0),
        }
        result = ranker.rerank(recs, items=items, price_range=(10.0, 100.0))
        result_ids = [r.item_id for r in result]
        assert "no_price" in result_ids
        assert "has_price" in result_ids


# ---------------------------------------------------------------------------
# Test: freshness boost
# ---------------------------------------------------------------------------

class TestFreshnessBoost:
    def test_freshness_boost_modifies_scores(self):
        ranker = RecommendationRanker(freshness_boost=0.1)
        recs = [_rec("a", 10.0), _rec("b", 10.0), _rec("c", 10.0)]
        result = ranker.rerank(recs)
        # First item gets no penalty, later items get decreasing factor
        # So scores should differ despite starting equal
        scores = [r.score for r in result]
        assert scores[0] >= scores[-1]

    def test_zero_freshness_boost_preserves_equal_scores(self):
        ranker = RecommendationRanker(freshness_boost=0.0)
        recs = [_rec("a", 5.0), _rec("b", 5.0)]
        result = ranker.rerank(recs)
        scores = [r.score for r in result]
        assert scores[0] == pytest.approx(scores[1])


# ---------------------------------------------------------------------------
# Test: empty recommendations
# ---------------------------------------------------------------------------

class TestEmptyRecommendations:
    def test_empty_input_returns_empty(self):
        ranker = RecommendationRanker()
        result = ranker.rerank([])
        assert result == []

    def test_all_items_seen_returns_empty(self):
        ranker = RecommendationRanker()
        recs = [_rec("a"), _rec("b")]
        result = ranker.rerank(recs, seen_item_ids={"a", "b"})
        assert result == []


# ---------------------------------------------------------------------------
# Test: missing items dict
# ---------------------------------------------------------------------------

class TestMissingItemsDict:
    def test_no_items_dict_skips_category_and_price_filtering(self):
        ranker = RecommendationRanker(max_same_category=1)
        recs = [_rec("a", 5.0), _rec("b", 4.0), _rec("c", 3.0)]
        # Without items dict, category diversity cannot be enforced
        result = ranker.rerank(recs, items=None)
        assert len(result) == 3

    def test_items_dict_missing_some_items(self):
        ranker = RecommendationRanker(max_same_category=1)
        recs = [_rec("a", 5.0), _rec("b", 4.0), _rec("c", 3.0)]
        items = {"a": _item("a", "electronics")}
        # "b" and "c" not in items -> category "unknown"
        result = ranker.rerank(recs, items=items)
        # 1 electronics + 1 unknown (max), third unknown dropped
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test: combined rules
# ---------------------------------------------------------------------------

class TestCombinedRules:
    def test_seen_and_price_and_diversity(self):
        ranker = RecommendationRanker(max_same_category=1)
        recs = [
            _rec("a", 10.0),
            _rec("b", 9.0),
            _rec("c", 8.0),
            _rec("d", 7.0),
            _rec("e", 6.0),
        ]
        items = {
            "a": _item("a", "electronics", price=50.0),
            "b": _item("b", "electronics", price=200.0),
            "c": _item("c", "books", price=15.0),
            "d": _item("d", "books", price=20.0),
            "e": _item("e", "clothing", price=30.0),
        }
        result = ranker.rerank(
            recs,
            items=items,
            seen_item_ids={"a"},
            price_range=(10.0, 100.0),
        )
        result_ids = [r.item_id for r in result]
        # "a" removed (seen), "b" removed (price > 100)
        # remaining: c (books), d (books), e (clothing)
        # diversity: max 1 per category -> c (books), e (clothing)
        assert "a" not in result_ids
        assert "b" not in result_ids
        assert len(result_ids) == 2


# ---------------------------------------------------------------------------
# Test: order preservation
# ---------------------------------------------------------------------------

class TestOrderPreservation:
    def test_preserves_relative_order_with_no_filtering(self):
        ranker = RecommendationRanker(freshness_boost=0.0)
        recs = [_rec("a", 10.0), _rec("b", 8.0), _rec("c", 5.0)]
        result = ranker.rerank(recs)
        result_ids = [r.item_id for r in result]
        assert result_ids == ["a", "b", "c"]
