.PHONY: install dev lint type-check test test-unit test-integration generate train serve docker-up docker-down clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short -m integration

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --tb=short

generate:
	python scripts/generate_data.py

train:
	python scripts/train_models.py

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	rm -rf data/ .pytest_cache htmlcov .mypy_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
