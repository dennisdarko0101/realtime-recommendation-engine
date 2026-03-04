# Architecture

## System Overview

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
    │  └──────┬───────┘  └──────┬───────┘  └────┬─────┘ │
    └─────────┼─────────────────┼────────────────┼───────┘
              │                 │                │
    ┌─────────▼─────────────────▼────────────────▼───────┐
    │                    DataStore                        │
    │  Users │ Items │ Interactions │ Sparse Matrix       │
    └────────────────────────────────────────────────────┘
```

## Recommendation Flow

1. **Request** arrives at `/recommend/{user_id}`
2. **Cache** checked for existing recommendations
3. **Hybrid model** generates candidates using collaborative + content signals
4. **Ranker** applies business rules (diversity, freshness, dedup)
5. **Response** returned and cached

## Real-time Pipeline

1. New interaction recorded via `/interact`
2. Stored in DataStore
3. User's cache invalidated
4. Popularity scores updated
5. Retrain triggered when threshold reached

## Cold Start Strategy

| Scenario | Strategy |
|----------|----------|
| New user, no history | Popularity-based recommendations |
| New user, some preferences | Content-based from stated preferences |
| New user, few interactions | Switching hybrid (content → collaborative) |
| New item | Content similarity to existing items |
| Existing user | Full hybrid (weighted/cascade) |

## A/B Testing

- Users deterministically assigned to variants via hash
- Metrics tracked per variant: CTR, conversion, diversity
- Two-proportion z-test for statistical significance
- Auto-concludes when significance threshold met
