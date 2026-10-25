# API Latency Benchmark Report

Benchmark date: 2026-10-25 08:27:19 UTC

## Local Host (CPU, uvicorn single worker)

| Metric | Sequential (50 calls) | Concurrent (10 calls) |
|---|---|---|
| **p50** | 30.5 ms | 169.8 ms |
| **p95** | 40.6 ms | 177.7 ms |
| **Wall-clock** | - | 180.8 ms |
| **Budget** | p95 <= 1500 ms (PASS) | p95 <= 2500 ms (PASS) |

### Per-model Sequential Breakdown

| Model | Mean latency | Min | Max |
|---|---|---|---|
| **CONFLUX** | 31.1 ms | 15.5 ms | 44.4 ms |
| **QUORUM** | 28.1 ms | 15.3 ms | 41.6 ms |
| **NEXUS** | 28.8 ms | 14.7 ms | 40.6 ms |
