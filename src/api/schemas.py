"""API request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.data.schemas import InteractionType, MetricsReport


# --- Requests ---


class CreateUserRequest(BaseModel):
    user_id: str
    features: dict[str, str | int | float | list[str]] = Field(default_factory=dict)


class CreateItemRequest(BaseModel):
    item_id: str
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    features: dict[str, str | int | float | list[str]] = Field(default_factory=dict)


class RecordInteractionRequest(BaseModel):
    user_id: str
    item_id: str
    interaction_type: InteractionType = InteractionType.VIEW
    value: Optional[float] = None


class EvaluateRequest(BaseModel):
    split_ratio: float = 0.8
    k: int = 10


# --- Responses ---


class RecommendationResponse(BaseModel):
    item_id: str
    score: float
    reason: str
    model_used: str


class RecommendationsListResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationResponse]
    count: int
    cached: bool = False


class UserResponse(BaseModel):
    user_id: str
    features: dict
    created_at: datetime


class ItemResponse(BaseModel):
    item_id: str
    category: str
    tags: list[str]
    description: str
    features: dict


class InteractionResponse(BaseModel):
    user_id: str
    item_id: str
    interaction_type: str
    recorded: bool = True


class SimilarItemsResponse(BaseModel):
    item_id: str
    similar: list[RecommendationResponse]
    count: int


class TrendingResponse(BaseModel):
    items: list[RecommendationResponse]
    window_hours: int


class EvaluationResponse(BaseModel):
    reports: list[MetricsReport]


class HealthResponse(BaseModel):
    status: str = "healthy"
    n_users: int = 0
    n_items: int = 0
    n_interactions: int = 0
    models_fitted: bool = False
