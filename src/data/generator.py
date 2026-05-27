"""Synthetic dataset generator for development, tests, and the demo.

Produces a plausible construction-supply catalog: users with a region and
segment, items in a handful of categories with tags and a text description,
and a log of interactions skewed by item popularity so the resulting user-item
matrix is realistically sparse. Output is clearly synthetic sample data, not
real usage.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from src.data.schemas import Interaction, InteractionType, Item, User
from src.data.store import DataStore

CATEGORIES = [
    "concrete", "steel", "lumber", "insulation", "roofing",
    "plumbing", "electrical", "drywall", "masonry", "hardware",
]

TAGS_BY_CATEGORY = {
    "concrete": ["ready-mix", "precast", "rebar", "high-strength"],
    "steel": ["structural", "rebar", "beam", "galvanized"],
    "lumber": ["framing", "treated", "plywood", "hardwood"],
    "insulation": ["fiberglass", "foam", "thermal", "acoustic"],
    "roofing": ["shingle", "membrane", "metal", "waterproof"],
    "plumbing": ["pipe", "fitting", "valve", "copper"],
    "electrical": ["wiring", "conduit", "breaker", "copper"],
    "drywall": ["gypsum", "board", "fire-rated", "moisture"],
    "masonry": ["brick", "block", "mortar", "stone"],
    "hardware": ["fastener", "bracket", "anchor", "galvanized"],
}

REGIONS = ["northeast", "southeast", "midwest", "southwest", "west"]
SEGMENTS = ["contractor", "wholesale", "diy", "enterprise"]

_INTERACTION_WEIGHTS = [
    (InteractionType.VIEW, 0.6),
    (InteractionType.CLICK, 0.22),
    (InteractionType.RATE, 0.1),
    (InteractionType.PURCHASE, 0.08),
]


def generate_users(n: int, seed: int | None = None) -> list[User]:
    rng = random.Random(seed)
    users = []
    for i in range(n):
        users.append(
            User(
                user_id=f"user_{i}",
                features={
                    "region": rng.choice(REGIONS),
                    "segment": rng.choice(SEGMENTS),
                },
            )
        )
    return users


def generate_items(n: int, seed: int | None = None) -> list[Item]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        pool = TAGS_BY_CATEGORY[category]
        tags = rng.sample(pool, k=min(2, len(pool)))
        grade = rng.choice(["standard", "premium", "industrial", "commercial"])
        description = (
            f"{grade} {category} material for construction projects, "
            f"{' and '.join(tags)} grade supply"
        )
        items.append(
            Item(
                item_id=f"item_{i}",
                category=category,
                tags=tags,
                description=description,
                features={"price": round(rng.uniform(10, 500), 2)},
            )
        )
    return items


def generate_interactions(
    users: list[User],
    items: list[Item],
    n: int,
    seed: int | None = None,
) -> list[Interaction]:
    """Generate ``n`` interactions skewed toward a popular subset of items.

    The popularity skew makes the user-item matrix sparse (many repeated pairs)
    rather than uniformly covering every cell.
    """
    rng = random.Random(seed)
    if not users or not items:
        return []

    # Popularity weights: a few items get most of the attention (long tail).
    pop_weights = [1.0 / (rank + 1) for rank in range(len(items))]

    base_time = datetime.utcnow()
    types, type_probs = zip(*_INTERACTION_WEIGHTS)

    interactions = []
    for _ in range(n):
        user = rng.choice(users)
        item = rng.choices(items, weights=pop_weights, k=1)[0]
        itype = rng.choices(types, weights=type_probs, k=1)[0]
        value = round(rng.uniform(1.0, 5.0), 1) if itype == InteractionType.RATE else None
        # Spread timestamps across the last 30 days for temporal splits.
        ts = base_time - timedelta(minutes=rng.randint(0, 60 * 24 * 30))
        interactions.append(
            Interaction(
                user_id=user.user_id,
                item_id=item.item_id,
                interaction_type=itype,
                value=value,
                timestamp=ts,
            )
        )
    return interactions


def generate_dataset(
    n_users: int = 1000,
    n_items: int = 500,
    n_interactions: int = 50_000,
    seed: int | None = None,
) -> DataStore:
    """Build a populated DataStore with the requested counts."""
    users = generate_users(n_users, seed=seed)
    items = generate_items(n_items, seed=seed)
    interactions = generate_interactions(users, items, n_interactions, seed=seed)

    store = DataStore()
    store.add_users_batch(users)
    store.add_items_batch(items)
    store.add_interactions_batch(interactions)
    return store
