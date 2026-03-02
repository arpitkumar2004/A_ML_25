# Production SLOs and Latency Tiers

## Objective
Define realistic serving objectives for a multimodal ML system and separate synchronous low-latency traffic from heavy asynchronous scoring.

## Service Tiers

### Tier A: Online Synchronous Prediction (User Request Path)

- Use lightweight model path and cached/precomputed features.
- SLO targets:
  - Availability: 99.9%
  - Error rate: < 0.1%
  - Latency: p95 < 30ms, p99 < 80ms

### Tier B: Heavy Multimodal Scoring (Async)

- For expensive text/image embedding + ensemble reranking.
- SLO targets:
  - Queue processing success: 99.9%
  - End-to-end completion: p95 < 2s, p99 < 5s

## Important Note on "Microsecond" Latency

Microsecond latency is not practical for multimodal model inference in online serving. It is only feasible for memory lookups or cache hits with precomputed predictions.

Production design should use:

1. cache-first synchronous path,
2. lightweight fallback model,
3. asynchronous heavy scorer.

## P0 Operational Metrics

- Request QPS
- p50/p95/p99 latency
- HTTP 5xx and timeout rate
- Model version and feature schema version
- Warmup/reload status
