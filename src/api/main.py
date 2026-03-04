"""FastAPI application for the recommendation engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.api.schemas import (
    CreateItemRequest,
    CreateUserRequest,
    EvaluateRequest,
    EvaluationResponse,
    HealthResponse,
    InteractionResponse,
    ItemResponse,
    RecommendationResponse,
    RecommendationsListResponse,
    RecordInteractionRequest,
    SimilarItemsResponse,
    TrendingResponse,
    UserResponse,
)
from src.config.settings import get_settings
from src.data.schemas import Interaction, Item, User
from src.data.store import DataStore
from src.evaluation.offline_eval import OfflineEvaluator
from src.models.collaborative import UserBasedCF
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender
from src.models.popular import PopularityRecommender
from src.serving.cache import RecommendationCache
from src.serving.ranker import RecommendationRanker
from src.serving.realtime import RealtimeUpdater

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter("rec_requests_total", "Total recommendation requests", ["endpoint"])
REQUEST_LATENCY = Histogram("rec_request_seconds", "Request latency", ["endpoint"])

# --- App State ---

store = DataStore()
cache = RecommendationCache()
ranker = RecommendationRanker()
hybrid = HybridRecommender()
popularity = PopularityRecommender()
content = ContentBasedFilter()
updater = RealtimeUpdater(store, cache, popularity)
models_fitted = False


def _fit_models() -> None:
    global models_fitted
    if store.n_users == 0 or store.n_items == 0:
        return

    items = store.list_items()
    interactions = store.get_all_interactions()

    content.fit_tfidf(items)
    popularity.fit(interactions, items)

    if store.n_interactions > 0:
        hybrid.fit(store, content_model=content)

    models_fitted = True
    logger.info("models_fitted", n_users=store.n_users, n_items=store.n_items)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info("starting", host=settings.API_HOST, port=settings.API_PORT)
    _fit_models()
    yield
    logger.info("shutting_down")


app = FastAPI(
    title="Realtime Recommendation Engine",
    description="Production recommendation system with collaborative, content-based, and hybrid approaches",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        n_users=store.n_users,
        n_items=store.n_items,
        n_interactions=store.n_interactions,
        models_fitted=models_fitted,
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/api/v1/recommend/{user_id}", response_model=RecommendationsListResponse)
async def get_recommendations(
    user_id: str,
    k: int = Query(default=10, ge=1, le=100),
    strategy: str = Query(default="weighted"),
) -> RecommendationsListResponse:
    REQUEST_COUNT.labels(endpoint="recommend").inc()

    if store.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Check cache
    cached_recs = cache.get(user_id)
    if cached_recs and len(cached_recs) >= k:
        return RecommendationsListResponse(
            user_id=user_id,
            recommendations=[
                RecommendationResponse(
                    item_id=r.item_id, score=r.score, reason=r.reason, model_used=r.model_used
                )
                for r in cached_recs[:k]
            ],
            count=k,
            cached=True,
        )

    # Generate fresh recommendations
    if models_fitted:
        recs = hybrid.recommend(user_id, k * 2, strategy=strategy)
    else:
        recs = []

    if not recs:
        recs = popularity.recommend_popular(k)

    # Rerank
    items_dict = {it.item_id: it for it in store.list_items()}
    seen = {ix.item_id for ix in store.get_user_interactions(user_id)}
    recs = ranker.rerank(recs, items=items_dict, seen_item_ids=seen)
    recs = recs[:k]

    cache.set(user_id, recs)

    return RecommendationsListResponse(
        user_id=user_id,
        recommendations=[
            RecommendationResponse(
                item_id=r.item_id, score=r.score, reason=r.reason, model_used=r.model_used
            )
            for r in recs
        ],
        count=len(recs),
        cached=False,
    )


@app.post("/api/v1/interact", response_model=InteractionResponse)
async def record_interaction(req: RecordInteractionRequest) -> InteractionResponse:
    REQUEST_COUNT.labels(endpoint="interact").inc()

    if store.get_user(req.user_id) is None:
        raise HTTPException(status_code=404, detail=f"User {req.user_id} not found")
    if store.get_item(req.item_id) is None:
        raise HTTPException(status_code=404, detail=f"Item {req.item_id} not found")

    interaction = Interaction(
        user_id=req.user_id,
        item_id=req.item_id,
        interaction_type=req.interaction_type,
        value=req.value,
    )
    updater.on_interaction(interaction)

    return InteractionResponse(
        user_id=req.user_id,
        item_id=req.item_id,
        interaction_type=req.interaction_type.value,
    )


@app.post("/api/v1/users", response_model=UserResponse, status_code=201)
async def create_user(req: CreateUserRequest) -> UserResponse:
    REQUEST_COUNT.labels(endpoint="create_user").inc()

    if store.get_user(req.user_id) is not None:
        raise HTTPException(status_code=409, detail=f"User {req.user_id} already exists")

    user = User(user_id=req.user_id, features=req.features)
    store.add_user(user)

    return UserResponse(
        user_id=user.user_id,
        features=user.features,
        created_at=user.created_at,
    )


@app.post("/api/v1/items", response_model=ItemResponse, status_code=201)
async def create_item(req: CreateItemRequest) -> ItemResponse:
    REQUEST_COUNT.labels(endpoint="create_item").inc()

    if store.get_item(req.item_id) is not None:
        raise HTTPException(status_code=409, detail=f"Item {req.item_id} already exists")

    item = Item(
        item_id=req.item_id,
        category=req.category,
        tags=req.tags,
        description=req.description,
        features=req.features,
    )
    store.add_item(item)

    return ItemResponse(
        item_id=item.item_id,
        category=item.category,
        tags=item.tags,
        description=item.description,
        features=item.features,
    )


@app.get("/api/v1/similar/{item_id}", response_model=SimilarItemsResponse)
async def get_similar_items(
    item_id: str,
    k: int = Query(default=10, ge=1, le=100),
) -> SimilarItemsResponse:
    REQUEST_COUNT.labels(endpoint="similar").inc()

    if store.get_item(item_id) is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    if not models_fitted:
        raise HTTPException(status_code=503, detail="Models not yet trained")

    recs = content.get_similar_items(item_id, k)
    return SimilarItemsResponse(
        item_id=item_id,
        similar=[
            RecommendationResponse(
                item_id=r.item_id, score=r.score, reason=r.reason, model_used=r.model_used
            )
            for r in recs
        ],
        count=len(recs),
    )


@app.get("/api/v1/trending", response_model=TrendingResponse)
async def get_trending(
    k: int = Query(default=10, ge=1, le=100),
    window_hours: int = Query(default=24, ge=1),
) -> TrendingResponse:
    REQUEST_COUNT.labels(endpoint="trending").inc()

    recs = popularity.recommend_trending(k, window_hours)
    if not recs:
        recs = popularity.recommend_popular(k)

    return TrendingResponse(
        items=[
            RecommendationResponse(
                item_id=r.item_id, score=r.score, reason=r.reason, model_used=r.model_used
            )
            for r in recs
        ],
        window_hours=window_hours,
    )


@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
async def evaluate_models(req: EvaluateRequest) -> EvaluationResponse:
    REQUEST_COUNT.labels(endpoint="evaluate").inc()

    if store.n_interactions == 0:
        raise HTTPException(status_code=400, detail="No interactions to evaluate")

    evaluator = OfflineEvaluator(k=req.k)
    train_ix, test_ix = evaluator.temporal_split(store.get_all_interactions(), req.split_ratio)

    # Build a training store
    train_store = DataStore()
    for u in store.list_users():
        train_store.add_user(u)
    for it in store.list_items():
        train_store.add_item(it)
    train_store.add_interactions_batch(train_ix)

    reports = evaluator.compare_models(train_store, test_ix)
    return EvaluationResponse(reports=reports)


def create_app() -> FastAPI:
    """Factory for creating the app (useful for testing)."""
    return app
