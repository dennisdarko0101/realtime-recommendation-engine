FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pyproject declares readme = "README.md", so the build backend needs it present
# at install time to generate package metadata.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --timeout=300 --retries=10 .

FROM base AS production
COPY src/ src/
COPY scripts/ scripts/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dev
RUN pip install --no-cache-dir --timeout=300 --retries=10 ".[dev]"
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
