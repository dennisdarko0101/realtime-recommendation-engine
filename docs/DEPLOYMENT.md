# Deployment Guide

## Local Development

```bash
make dev              # Install with dev dependencies
make generate         # Generate synthetic data
make train            # Train models + evaluate
make serve            # Run API on localhost:8000
```

## Docker

```bash
# Production
docker compose -f docker/docker-compose.yml up -d

# Development (with hot reload)
docker compose -f docker/docker-compose.yml --profile dev up api-dev
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| REC_REDIS_URL | redis://localhost:6379/0 | Redis connection URL |
| REC_CHROMA_DIR | ./data/chroma | ChromaDB persistence directory |
| REC_EMBEDDING_MODEL | all-MiniLM-L6-v2 | Sentence-transformer model |
| REC_TOP_K_DEFAULT | 10 | Default recommendations count |
| REC_CACHE_TTL | 300 | Cache TTL in seconds |
| REC_DEBUG | false | Enable debug logging |

## Production Checklist

- [ ] Set appropriate `REC_CACHE_TTL` for your traffic
- [ ] Configure Redis for cache persistence
- [ ] Set up Prometheus scraping for `/metrics`
- [ ] Configure health checks on `/health`
- [ ] Load data and train models before accepting traffic
- [ ] Set up A/B tests for model comparison in production
