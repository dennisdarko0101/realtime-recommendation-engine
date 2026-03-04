"""Real-time update processing for the recommendation engine."""

from __future__ import annotations

import structlog

from src.data.schemas import Interaction
from src.data.store import DataStore
from src.models.popular import PopularityRecommender
from src.serving.cache import RecommendationCache

logger = structlog.get_logger()


class RealtimeUpdater:
    """Process new interactions in real-time and update system state."""

    def __init__(
        self,
        store: DataStore,
        cache: RecommendationCache,
        popularity: PopularityRecommender | None = None,
    ) -> None:
        self._store = store
        self._cache = cache
        self._popularity = popularity
        self._pending_retrain_users: set[str] = set()
        self._interaction_count_since_retrain = 0

    def on_interaction(self, interaction: Interaction) -> None:
        """Process a single new interaction."""
        # 1. Store the interaction
        self._store.add_interaction(interaction)

        # 2. Invalidate cached recommendations for this user
        self._cache.invalidate(interaction.user_id)

        # 3. Track that this user's model may need refreshing
        self._pending_retrain_users.add(interaction.user_id)
        self._interaction_count_since_retrain += 1

        logger.info(
            "interaction_processed",
            user_id=interaction.user_id,
            item_id=interaction.item_id,
            interaction_type=interaction.interaction_type,
        )

    def get_pending_retrain_users(self) -> set[str]:
        return set(self._pending_retrain_users)

    def clear_pending(self) -> None:
        self._pending_retrain_users.clear()
        self._interaction_count_since_retrain = 0

    @property
    def interactions_since_retrain(self) -> int:
        return self._interaction_count_since_retrain

    def should_retrain(self, threshold: int = 1000) -> bool:
        """Check if enough new interactions have accumulated to warrant retraining."""
        return self._interaction_count_since_retrain >= threshold


class EventProcessor:
    """Batch process interaction events."""

    def __init__(self, updater: RealtimeUpdater, batch_size: int = 100) -> None:
        self._updater = updater
        self._batch_size = batch_size
        self._buffer: list[Interaction] = []

    def add_event(self, interaction: Interaction) -> None:
        self._buffer.append(interaction)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> int:
        """Process all buffered events. Returns number processed."""
        count = len(self._buffer)
        for ix in self._buffer:
            self._updater.on_interaction(ix)
        self._buffer.clear()
        logger.info("batch_flushed", count=count)
        return count

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)
