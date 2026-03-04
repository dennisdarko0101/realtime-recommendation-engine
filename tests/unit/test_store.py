"""Tests for DataStore."""

import pytest
import numpy as np
from scipy.sparse import issparse

from src.data.store import DataStore
from src.data.schemas import User, Item, Interaction, InteractionType


@pytest.fixture
def store() -> DataStore:
    s = DataStore()
    s.add_user(User(user_id="u1", features={"region": "northeast"}))
    s.add_user(User(user_id="u2", features={"region": "southeast"}))
    s.add_item(Item(item_id="i1", category="concrete", tags=["ready-mix"]))
    s.add_item(Item(item_id="i2", category="steel", tags=["structural"]))
    s.add_item(Item(item_id="i3", category="lumber", tags=["framing"]))
    s.add_interaction(Interaction(user_id="u1", item_id="i1", interaction_type=InteractionType.PURCHASE))
    s.add_interaction(Interaction(user_id="u1", item_id="i2", interaction_type=InteractionType.VIEW))
    s.add_interaction(Interaction(user_id="u2", item_id="i2", interaction_type=InteractionType.RATE, value=4.0))
    return s


def test_add_get_user(store: DataStore) -> None:
    user = store.get_user("u1")
    assert user is not None
    assert user.user_id == "u1"
    assert user.features["region"] == "northeast"


def test_get_missing_user(store: DataStore) -> None:
    assert store.get_user("nonexistent") is None


def test_list_users(store: DataStore) -> None:
    assert len(store.list_users()) == 2


def test_add_get_item(store: DataStore) -> None:
    item = store.get_item("i1")
    assert item is not None
    assert item.category == "concrete"


def test_list_items(store: DataStore) -> None:
    assert len(store.list_items()) == 3


def test_interaction_counts(store: DataStore) -> None:
    assert store.n_interactions == 3


def test_user_interactions(store: DataStore) -> None:
    ixs = store.get_user_interactions("u1")
    assert len(ixs) == 2


def test_item_interactions(store: DataStore) -> None:
    ixs = store.get_item_interactions("i2")
    assert len(ixs) == 2


def test_interaction_matrix_shape(store: DataStore) -> None:
    mat = store.get_interaction_matrix()
    assert mat.shape == (2, 3)
    assert issparse(mat)


def test_interaction_matrix_values(store: DataStore) -> None:
    mat = store.get_interaction_matrix()
    # u2 rated i2 with value 4.0
    user_idx = store.get_user_index()
    item_idx = store.get_item_index()
    assert mat[user_idx["u2"], item_idx["i2"]] == 4.0


def test_batch_operations() -> None:
    s = DataStore()
    users = [User(user_id=f"u{i}") for i in range(10)]
    items = [Item(item_id=f"i{i}", category="concrete") for i in range(5)]
    assert s.add_users_batch(users) == 10
    assert s.add_items_batch(items) == 5
    assert s.n_users == 10
    assert s.n_items == 5


def test_save_load(tmp_path) -> None:
    s = DataStore()
    s.add_user(User(user_id="u1"))
    s.add_item(Item(item_id="i1", category="steel"))
    s.add_interaction(Interaction(user_id="u1", item_id="i1"))
    s.save(str(tmp_path))

    s2 = DataStore()
    s2.load(str(tmp_path))
    assert s2.n_users == 1
    assert s2.n_items == 1
    assert s2.n_interactions == 1


def test_clear(store: DataStore) -> None:
    store.clear()
    assert store.n_users == 0
    assert store.n_items == 0
    assert store.n_interactions == 0


def test_empty_matrix() -> None:
    s = DataStore()
    mat = s.get_interaction_matrix()
    assert mat.shape == (0, 0)
