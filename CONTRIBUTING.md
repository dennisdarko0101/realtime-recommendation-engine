# Contributing

## Setup

```bash
git clone <repo>
cd realtime-recommendation-engine
make dev
```

## Workflow

1. Create a feature branch
2. Write tests first
3. Implement the feature
4. Run `make lint` and `make test`
5. Open a pull request

## Code Style

- Python 3.11+, typed with type hints
- Formatted with `ruff format`
- Linted with `ruff check`
- Type-checked with `mypy`

## Testing

```bash
make test-unit        # Unit tests only
make test-integration # Integration tests
make test-cov         # With coverage report
```
