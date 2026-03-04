"""Tests for PopularityRecommender."""

from datetime import datetime, timedelta

import pytest

from src.data.schemas import Interaction, InteractionType, Item
from src.models.popular import PopularityRecommender


@pytest.fixture
def items() -> list[Item]:
    return [
        Item(item_id="i1", category="concrete"),
        Item(item_id="i2", category="concrete"),
        Item(item_id="i3", category="steel"),
        Item(item_id="i4", category="lumber"),
    ]


@pytest.fixture
def now() -> datetime:
    return datetime(2024, 6, 15, 12, 0)


@pytest.fixture
def interactions(now: datetime) -> list[Interaction]:
    old = now - timedelta(days=30)
    recent = now - timedelta(hours=2)
    return [
        # i1: 3 purchases (old) — high all-time
        Interaction(user_id="u1", item_id="i1", interaction_type=InteractionType.PURCHASE, timestamp=old),
        Interaction(user_id="u2", item_id="i1", interaction_type=InteractionType.PURCHASE, timestamp=old),
        Interaction(user_id="u3", item_id="i1", interaction_type=InteractionType.PURCHASE, timestamp=old),
        # i2: 5 recent views — trending
        Interaction(user_id="u1", item_id="i2", interaction_type=InteractionType.VIEW, timestamp=recent),
        Interaction(user_id="u2", item_id="i2", interaction_type=InteractionType.VIEW, timestamp=recent),
        Interaction(user_id="u3", item_id="i2", interaction_type=InteractionType.VIEW, timestamp=recent),
        Interaction(user_id="u4", item_id="i2", interaction_type=InteractionType.VIEW, timestamp=recent),
        Interaction(user_id="u5", item_id="i2", interaction_type=InteractionType.VIEW, timestamp=recent),
        # i3: 1 old click
        Interaction(user_id="u1", item_id="i3", interaction_type=InteractionType.CLICK, timestamp=old),
    ]


def test_popular_returns_items(items, interactions) -> None:
    pop = PopularityRecommender()
    pop.fit(interactions, items)
    recs = pop.recommend_popular(k=3)
    assert len(recs) == 3
    assert all(r.model_used == "popularity" for r in recs)


def test_popular_ordering(items, interactions) -> None:
    pop = PopularityRecommender()
    pop.fit(interactions, items)
    recs = pop.recommend_popular(k=10)
    # i1 has 3 purchases (weight 5 each = 15), should be top
    assert recs[0].item_id == "i1"


def test_trending(items, interactions, now) -> None:
    pop = PopularityRecommender()
    pop.fit(interactions, items)
    recs = pop.recommend_trending(k=3, window_hours=24, now=now)
    # i2 has 5 recent interactions
    assert recs[0].item_id == "i2"


def test_category_popular(items, interactions) -> None:
    pop = PopularityRecommender()
    pop.fit(interactions, items)
    recs = pop.recommend_by_category("concrete", k=2)
    assert all(items_dict[r.item_id].category == "concrete" for r in recs
               if (items_dict := {i.item_id: i for i in items}).get(r.item_id))


def test_empty_interactions(items) -> None:
    pop = PopularityRecommender()
    pop.fit([], items)
    assert pop.recommend_popular(k=5) == []


def test_trending_no_recent(items, interactions, now) -> None:
    pop = PopularityRecommender()
    pop.fit(interactions, items)
    future = now + timedelta(days=365)
    recs = pop.recommend_trending(k=5, window_hours=24, now=future)
    assert len(recs) == 0
