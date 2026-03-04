# Handoff Document

## Project: Realtime Recommendation Engine

### Purpose
Production recommendation system for materials matching at GoMaterials. Provides personalized material recommendations using collaborative filtering, content-based filtering, and hybrid approaches.

### Key Decisions

1. **In-memory DataStore** — chosen for simplicity and fast iteration. Production should migrate to PostgreSQL + Redis.
2. **Sparse matrices** — scipy CSR format for efficient user-item interaction storage.
3. **SVD factorization** — scipy.sparse.linalg.svds for matrix factorization (no GPU required).
4. **LRU cache** — in-process cache avoids Redis dependency for development.
5. **Hybrid strategies** — weighted, switching, and cascade allow tuning per use case.

### Architecture

See `docs/ARCHITECTURE.md` for full system diagram.

### How to Extend

- **New model**: implement `CollaborativeFilter` ABC or add to `hybrid.py`
- **New features**: add fields to `Item` schema, update `ContentBasedFilter.fit_features()`
- **New API endpoint**: add to `src/api/main.py`, add schema to `src/api/schemas.py`
- **Production cache**: replace `RecommendationCache` with Redis-backed implementation

### Known Limitations

- DataStore is in-memory (restart loses state unless saved/loaded)
- Sentence-transformers not loaded by default (embedding-based filtering needs explicit setup)
- A/B test state is in-memory (not persisted across restarts)
- No authentication on API endpoints

### Testing

130+ tests covering all modules. Run with `make test`.
