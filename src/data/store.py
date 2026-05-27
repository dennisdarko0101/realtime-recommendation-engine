"""In-memory data store for users, items, and interactions.

Holds the catalog and the interaction log, builds the sparse user-item matrix
used by the collaborative models, and can persist to / load from JSON on disk.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scipy.sparse import csr_matrix

from src.data.schemas import Interaction, Item, User


class DataStore:
    """Holds users, items, and interactions in memory."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._items: dict[str, Item] = {}
        self._interactions: list[Interaction] = []
        self._user_interactions: dict[str, list[Interaction]] = defaultdict(list)
        self._item_interactions: dict[str, list[Interaction]] = defaultdict(list)

    # --- Users ---

    def add_user(self, user: User) -> None:
        self._users[user.user_id] = user

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def list_users(self) -> list[User]:
        return list(self._users.values())

    def add_users_batch(self, users: list[User]) -> int:
        for u in users:
            self.add_user(u)
        return len(users)

    # --- Items ---

    def add_item(self, item: Item) -> None:
        self._items[item.item_id] = item

    def get_item(self, item_id: str) -> Item | None:
        return self._items.get(item_id)

    def list_items(self) -> list[Item]:
        return list(self._items.values())

    def add_items_batch(self, items: list[Item]) -> int:
        for it in items:
            self.add_item(it)
        return len(items)

    # --- Interactions ---

    def add_interaction(self, interaction: Interaction) -> None:
        self._interactions.append(interaction)
        self._user_interactions[interaction.user_id].append(interaction)
        self._item_interactions[interaction.item_id].append(interaction)

    def add_interactions_batch(self, interactions: list[Interaction]) -> int:
        for ix in interactions:
            self.add_interaction(ix)
        return len(interactions)

    def get_all_interactions(self) -> list[Interaction]:
        return list(self._interactions)

    def get_user_interactions(self, user_id: str) -> list[Interaction]:
        return list(self._user_interactions.get(user_id, []))

    def get_item_interactions(self, item_id: str) -> list[Interaction]:
        return list(self._item_interactions.get(item_id, []))

    # --- Counts ---

    @property
    def n_users(self) -> int:
        return len(self._users)

    @property
    def n_items(self) -> int:
        return len(self._items)

    @property
    def n_interactions(self) -> int:
        return len(self._interactions)

    # --- Index maps ---

    def get_user_index(self) -> dict[str, int]:
        """Map user_id to a stable row index (insertion order)."""
        return {uid: i for i, uid in enumerate(self._users)}

    def get_item_index(self) -> dict[str, int]:
        """Map item_id to a stable column index (insertion order)."""
        return {iid: i for i, iid in enumerate(self._items)}

    # --- Interaction matrix ---

    def get_interaction_matrix(self) -> csr_matrix:
        """Build the sparse user-item matrix.

        Each cell holds the strongest signal seen for that pair: an explicit
        rating value when present, otherwise an implicit weight of 1.0.
        """
        n_u, n_i = self.n_users, self.n_items
        if n_u == 0 or n_i == 0:
            return csr_matrix((n_u, n_i))

        user_idx = self.get_user_index()
        item_idx = self.get_item_index()

        cell: dict[tuple[int, int], float] = {}
        for ix in self._interactions:
            if ix.user_id in user_idx and ix.item_id in item_idx:
                weight = float(ix.value) if ix.value is not None else 1.0
                key = (user_idx[ix.user_id], item_idx[ix.item_id])
                cell[key] = max(cell.get(key, 0.0), weight)

        if not cell:
            return csr_matrix((n_u, n_i))

        rows = [k[0] for k in cell]
        cols = [k[1] for k in cell]
        data = [cell[k] for k in cell]
        return csr_matrix((data, (rows, cols)), shape=(n_u, n_i))

    # --- Persistence ---

    def save(self, path: str) -> None:
        """Write users, items, and interactions to JSON files under ``path``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "users.json").write_text(
            json.dumps([u.model_dump(mode="json") for u in self._users.values()])
        )
        (directory / "items.json").write_text(
            json.dumps([it.model_dump(mode="json") for it in self._items.values()])
        )
        (directory / "interactions.json").write_text(
            json.dumps([ix.model_dump(mode="json") for ix in self._interactions])
        )

    def load(self, path: str) -> None:
        """Replace current contents with data loaded from ``path``."""
        directory = Path(path)
        self.clear()
        for d in json.loads((directory / "users.json").read_text()):
            self.add_user(User(**d))
        for d in json.loads((directory / "items.json").read_text()):
            self.add_item(Item(**d))
        for d in json.loads((directory / "interactions.json").read_text()):
            self.add_interaction(Interaction(**d))

    def clear(self) -> None:
        self._users.clear()
        self._items.clear()
        self._interactions.clear()
        self._user_interactions.clear()
        self._item_interactions.clear()
