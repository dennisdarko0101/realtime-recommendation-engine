"""Shared test fixtures."""

from __future__ import annotations

import pytest

from src.data.generator import generate_dataset
from src.data.schemas import Interaction, InteractionType, Item, User
from src.data.store import DataStore


@pytest.fixture
def small_store() -> DataStore:
    """A small dataset for fast tests: 20 users, 10 items, 200 interactions."""
    return generate_dataset(n_users=20, n_items=10, n_interactions=200, seed=42)


@pytest.fixture
def medium_store() -> DataStore:
    """A medium dataset: 50 users, 30 items, 1000 interactions."""
    return generate_dataset(n_users=50, n_items=30, n_interactions=1000, seed=42)


@pytest.fixture
def empty_store() -> DataStore:
    return DataStore()
