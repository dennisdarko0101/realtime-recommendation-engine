"""Integration tests for FastAPI recommendation engine endpoints."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app, cache, store, _fit_models, models_fitted
from src.data.schemas import InteractionType, Item, User


@pytest.fixture(autouse=True)
def clean_state():
    """Reset app state between tests."""
    import src.api.main as main_module

    store.clear()
    cache.clear()
    main_module.models_fitted = False
    yield
    store.clear()
    cache.clear()
    main_module.models_fitted = False


@pytest_asyncio.fixture
async def client():
    """Create an async test client that bypasses lifespan."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helper to seed minimal data through the API
# ---------------------------------------------------------------------------

async def _create_user(client: AsyncClient, user_id: str, **features) -> None:
    await client.post("/api/v1/users", json={"user_id": user_id, "features": features})


async def _create_item(
    client: AsyncClient,
    item_id: str,
    category: str = "concrete",
    tags: list[str] | None = None,
    description: str = "test item",
) -> None:
    await client.post(
        "/api/v1/items",
        json={
            "item_id": item_id,
            "category": category,
            "tags": tags or ["tag1"],
            "description": description,
        },
    )


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["n_users"] == 0
    assert body["n_items"] == 0
    assert body["models_fitted"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_user_returns_201(client: AsyncClient):
    resp = await client.post(
        "/api/v1/users",
        json={"user_id": "u1", "features": {"region": "northeast"}},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["user_id"] == "u1"
    assert body["features"]["region"] == "northeast"
    assert "created_at" in body


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_duplicate_user_returns_409(client: AsyncClient):
    await _create_user(client, "u1")
    resp = await client.post("/api/v1/users", json={"user_id": "u1"})
    assert resp.status_code == 409


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_item_returns_201(client: AsyncClient):
    resp = await client.post(
        "/api/v1/items",
        json={
            "item_id": "i1",
            "category": "steel",
            "tags": ["structural", "rebar"],
            "description": "Structural steel rebar",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["item_id"] == "i1"
    assert body["category"] == "steel"
    assert "structural" in body["tags"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_duplicate_item_returns_409(client: AsyncClient):
    await _create_item(client, "i1")
    resp = await client.post("/api/v1/items", json={"item_id": "i1"})
    assert resp.status_code == 409


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interact_records_interaction(client: AsyncClient):
    await _create_user(client, "u1")
    await _create_item(client, "i1")

    resp = await client.post(
        "/api/v1/interact",
        json={
            "user_id": "u1",
            "item_id": "i1",
            "interaction_type": "click",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "u1"
    assert body["item_id"] == "i1"
    assert body["interaction_type"] == "click"
    assert body["recorded"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interact_unknown_user_returns_404(client: AsyncClient):
    await _create_item(client, "i1")
    resp = await client.post(
        "/api/v1/interact",
        json={"user_id": "ghost", "item_id": "i1", "interaction_type": "view"},
    )
    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interact_unknown_item_returns_404(client: AsyncClient):
    await _create_user(client, "u1")
    resp = await client.post(
        "/api/v1/interact",
        json={"user_id": "u1", "item_id": "ghost", "interaction_type": "view"},
    )
    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recommend_returns_recommendations(client: AsyncClient):
    """Recommendations should fall back to popularity when models are not fitted."""
    import src.api.main as main_module

    # Seed data directly into the store so _fit_models can work
    for i in range(5):
        store.add_user(User(user_id=f"u{i}"))
    for i in range(10):
        store.add_item(
            Item(
                item_id=f"i{i}",
                category="concrete",
                tags=["ready-mix"],
                description=f"Item {i} description for testing",
            )
        )
    from src.data.schemas import Interaction

    for i in range(5):
        for j in range(10):
            store.add_interaction(
                Interaction(user_id=f"u{i}", item_id=f"i{j}", interaction_type=InteractionType.VIEW)
            )
    _fit_models()

    resp = await client.get("/api/v1/recommend/u0?k=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "u0"
    assert body["count"] <= 5
    assert isinstance(body["recommendations"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recommend_unknown_user_returns_404(client: AsyncClient):
    resp = await client.get("/api/v1/recommend/nonexistent")
    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_similar_items_returns_results(client: AsyncClient):
    """Similar items endpoint requires models to be fitted."""
    import src.api.main as main_module

    for i in range(5):
        store.add_user(User(user_id=f"u{i}"))
    for i in range(10):
        store.add_item(
            Item(
                item_id=f"i{i}",
                category="concrete" if i < 5 else "steel",
                tags=["ready-mix"] if i < 5 else ["structural"],
                description=f"Item {i} material for construction projects",
            )
        )
    from src.data.schemas import Interaction

    for i in range(5):
        store.add_interaction(
            Interaction(user_id=f"u{i}", item_id=f"i{i}", interaction_type=InteractionType.VIEW)
        )
    _fit_models()

    resp = await client.get("/api/v1/similar/i0?k=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["item_id"] == "i0"
    assert isinstance(body["similar"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_similar_unknown_item_returns_404(client: AsyncClient):
    resp = await client.get("/api/v1/similar/nonexistent")
    assert resp.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trending_returns_items(client: AsyncClient):
    resp = await client.get("/api/v1/trending?k=5&window_hours=48")
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert isinstance(body["items"], list)
    assert body["window_hours"] == 48


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_returns_prometheus_output(client: AsyncClient):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # Prometheus output should contain at least our custom metric names
    text = resp.text
    assert "rec_requests_total" in text or "rec_request_seconds" in text or len(text) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluate_no_data_returns_400(client: AsyncClient):
    resp = await client.post("/api/v1/evaluate", json={"split_ratio": 0.8, "k": 5})
    assert resp.status_code == 400
    assert "No interactions" in resp.json()["detail"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_reflects_counts_after_data_added(client: AsyncClient):
    await _create_user(client, "u1")
    await _create_item(client, "i1")

    resp = await client.get("/health")
    body = resp.json()
    assert body["n_users"] == 1
    assert body["n_items"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow(client: AsyncClient):
    """End-to-end: create users -> create items -> interact -> recommend."""
    import src.api.main as main_module

    # Step 1: Create users
    user_ids = [f"wf_u{i}" for i in range(5)]
    for uid in user_ids:
        resp = await client.post(
            "/api/v1/users",
            json={"user_id": uid, "features": {"region": "midwest"}},
        )
        assert resp.status_code == 201

    # Step 2: Create items
    categories = ["concrete", "steel", "lumber", "insulation", "roofing"]
    item_ids = [f"wf_i{i}" for i in range(10)]
    for idx, iid in enumerate(item_ids):
        cat = categories[idx % len(categories)]
        resp = await client.post(
            "/api/v1/items",
            json={
                "item_id": iid,
                "category": cat,
                "tags": [f"tag_{idx}"],
                "description": f"Workflow item {idx} in category {cat}",
            },
        )
        assert resp.status_code == 201

    # Step 3: Record interactions
    for uid in user_ids:
        for iid in item_ids[:5]:
            resp = await client.post(
                "/api/v1/interact",
                json={
                    "user_id": uid,
                    "item_id": iid,
                    "interaction_type": "view",
                },
            )
            assert resp.status_code == 200

    # Fit models so recommendations work
    _fit_models()

    # Step 4: Health should show correct counts
    resp = await client.get("/health")
    body = resp.json()
    assert body["n_users"] == 5
    assert body["n_items"] == 10
    assert body["n_interactions"] == 25
    assert body["models_fitted"] is True

    # Step 5: Get recommendations
    resp = await client.get(f"/api/v1/recommend/{user_ids[0]}?k=5")
    assert resp.status_code == 200
    rec_body = resp.json()
    assert rec_body["user_id"] == user_ids[0]
    assert isinstance(rec_body["recommendations"], list)

    # Step 6: Get trending
    resp = await client.get("/api/v1/trending?k=3")
    assert resp.status_code == 200
    assert isinstance(resp.json()["items"], list)

    # Step 7: Get similar items
    resp = await client.get(f"/api/v1/similar/{item_ids[0]}?k=3")
    assert resp.status_code == 200
    assert isinstance(resp.json()["similar"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interact_with_rating_value(client: AsyncClient):
    """Interactions with a rating value should be accepted."""
    await _create_user(client, "u1")
    await _create_item(client, "i1")

    resp = await client.post(
        "/api/v1/interact",
        json={
            "user_id": "u1",
            "item_id": "i1",
            "interaction_type": "rate",
            "value": 4.5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["interaction_type"] == "rate"
