# Realtime Recommendation Engine

Production recommendation system with collaborative filtering, content-based filtering, and hybrid approaches. Built for materials matching at GoMaterials.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      FastAPI Layer                        │
│  /recommend  /interact  /similar  /trending  /evaluate   │
└─────────┬───────────────────┬───────────────────┬────────┘
          │                   │                   │
    ┌─────▼─────┐     ┌──────▼──────┐     ┌─────▼──────┐
    │   Cache    │     │  Real-time  │     │   Ranker   │
    │  (LRU)    │     │  Updater    │     │ (business  │
    │           │     │             │     │  rules)    │
    └─────┬─────┘     └──────┬──────┘     └─────┬──────┘
          │                  │                   │
    ┌─────▼──────────────────▼───────────────────▼──────┐
    │              Hybrid Recommender                     │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
    │  │Collaborative │  │Content-Based │  │Popularity│ │
    │  │  - UserCF    │  │  - TF-IDF    │  │- Trending│ │
    │  │  - ItemCF    │  │  - Embedding │  │- All-time│ │
    │  │  - SVD       │  │  - Features  │  │- Category│ │
    │  └──────────────┘  └──────────────┘  └──────────┘ │
    └────────────────────────┬───────────────────────────┘
                             │
    ┌────────────────────────▼───────────────────────────┐
    │                    DataStore                        │
    │  Users │ Items │ Interactions │ Sparse Matrix       │
    └────────────────────────────────────────────────────┘
```

## Algorithms

### Collaborative Filtering
- **User-Based CF**: Finds users with similar interaction patterns using cosine similarity + KNN, then recommends items those neighbors liked.
- **Item-Based CF**: Builds an item-item similarity matrix from co-interaction patterns. Recommends items similar to what the user already liked.
- **Matrix Factorization (SVD)**: Decomposes the user-item interaction matrix into latent factors using truncated SVD. Predicts scores for unseen items from the latent space.

### Content-Based Filtering
- **TF-IDF**: Builds term-frequency vectors from item descriptions/tags, uses cosine similarity to find related items.
- **Embedding-Based**: Uses sentence-transformers to embed item descriptions into dense vectors. Supports ChromaDB for approximate nearest neighbor search.
- **Feature-Based**: One-hot encodes structured features (category, tags) and computes cosine similarity.

### Hybrid Approaches
- **Weighted**: `score = w1 * collab_score + w2 * content_score` — linear combination with configurable weights.
- **Switching**: Uses collaborative filtering when sufficient interaction data exists, falls back to content-based for cold-start users.
- **Cascade**: Content-based filtering generates broad candidates, collaborative filtering re-ranks them.

## Cold Start Handling

| Scenario | Strategy |
|----------|----------|
| New user, no history | Popularity-based (trending/all-time) |
| New user, stated preferences | Content-based from preferences |
| New user, few interactions | Switching hybrid (content → collab) |
| New item, no interactions | Content similarity to existing items |
| Established user | Full hybrid with all signals |

## Real-Time Update Pipeline

1. User interaction recorded via `POST /api/v1/interact`
2. Interaction stored in DataStore
3. User's cached recommendations invalidated
4. Popularity scores updated incrementally
5. Batch retrain triggered when interaction threshold reached

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate demo data
python scripts/generate_data.py

# Train models + evaluate
python scripts/train_models.py

# Start API server
make serve
# → http://localhost:8000/docs
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/recommend/{user_id}` | Get personalized recommendations |
| POST | `/api/v1/interact` | Record user interaction (triggers real-time update) |
| POST | `/api/v1/users` | Create a new user |
| POST | `/api/v1/items` | Create a new item |
| GET | `/api/v1/similar/{item_id}` | Find similar items |
| GET | `/api/v1/trending` | Get trending items |
| POST | `/api/v1/evaluate` | Run offline model evaluation |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

### Example: Get Recommendations

```bash
curl http://localhost:8000/api/v1/recommend/user_0001?k=5&strategy=weighted
```

```json
{
  "user_id": "user_0001",
  "recommendations": [
    {"item_id": "item_0042", "score": 3.21, "reason": "Similar users liked this item", "model_used": "hybrid_weighted"},
    {"item_id": "item_0118", "score": 2.85, "reason": "Content similar (tfidf)", "model_used": "hybrid_weighted"}
  ],
  "count": 5,
  "cached": false
}
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K | Fraction of relevant items found in top-K |
| NDCG@K | Normalized Discounted Cumulative Gain (position-aware) |
| MAP@K | Mean Average Precision |
| Hit Rate | Whether any relevant item appears in recommendations |
| MRR | Mean Reciprocal Rank of first relevant item |
| Coverage | Fraction of catalog appearing in any recommendation |
| Diversity | Category variety within recommendation lists |
| Novelty | How non-obvious (non-popular) recommendations are |

## A/B Testing

Built-in A/B testing framework for comparing recommendation strategies in production:
- Deterministic user assignment via hash-based bucketing
- Tracks CTR, conversion rate, diversity, and novelty per variant
- Two-proportion z-test for statistical significance
- Auto-concludes when significance threshold is met

## Project Structure

```
src/
├── config/         # Settings via environment variables
├── data/           # Schemas, DataStore, synthetic data generator
├── models/         # Collaborative, content-based, hybrid, popularity
├── serving/        # Cache, ranker, real-time updates, A/B testing
├── evaluation/     # Metrics, offline evaluation
├── api/            # FastAPI endpoints and request/response schemas
└── utils/          # Logging
tests/
├── unit/           # 100+ unit tests
└── integration/    # API and pipeline integration tests
scripts/            # Data generation and model training
docker/             # Dockerfile and docker-compose
docs/               # Architecture, deployment, handoff docs
```

## Development

```bash
make dev           # Install dev dependencies
make lint          # Run ruff linter
make type-check    # Run mypy
make test          # Run all tests
make test-cov      # Run tests with coverage report
```

## Docker

```bash
make docker-up     # Start production stack
make docker-down   # Stop stack
```

## License

MIT
