"""Core data schemas for users, items, interactions, recommendations, metrics."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class InteractionType(str, Enum):
    """Type of a user-item interaction, ordered weakest to strongest signal."""

    VIEW = "view"
    CLICK = "click"
    RATE = "rate"
    PURCHASE = "purchase"


class User(BaseModel):
    user_id: str
    features: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Item(BaseModel):
    item_id: str
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    features: dict[str, Any] = Field(default_factory=dict)


class Interaction(BaseModel):
    user_id: str
    item_id: str
    interaction_type: InteractionType = InteractionType.VIEW
    value: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Recommendation(BaseModel):
    item_id: str
    score: float
    reason: str = ""
    model_used: str = ""


class MetricsReport(BaseModel):
    """Offline evaluation results for a single model."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    map_at_k: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0
    coverage: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0
    k: int = 10
    model_name: str = ""
