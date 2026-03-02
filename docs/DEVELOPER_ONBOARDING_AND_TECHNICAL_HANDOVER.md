# Developer Onboarding & Technical Handover

## Document Purpose
This document is the technical handover guide for engineers joining the project. It explains:

1. what is already implemented in this repository,
2. how each subsystem works end-to-end,
3. the target production architecture for 1M+ DAU with P99 < 1s,
4. how to start contributing safely from day one.

---

## 1) High-Level Architecture

## 1.1 End-to-End Data Flow (Current + Target)

### A. Raw Ingestion
- Product records are ingested as tabular datasets (`train.csv`, `test.csv`) and moved through raw/interim/processed layers.
- Current implementation is file-driven (`data/raw`, `data/processed`) and config-driven via YAML files under `configs/`.
- Target production extension uses **Kafka** as the event backbone for:
  - real-time catalog updates,
  - asynchronous feature refresh,
  - model inference events,
  - monitoring/feedback events.

### B. Parsing & Data Enrichment
- `src/data/parse_features.py` extracts structured attributes from text (e.g., parsed quantity/unit signals).
- This step feeds numeric/text feature builders and reduces free-text ambiguity.

### C. Feature Engineering
- `src/features/build_features.py` orchestrates multimodal feature construction:
  - text embeddings (`TextEmbedder`),
  - image embeddings (`ImageEmbedder`),
  - numeric features (`NumericBuilder`),
  - optional sparse/dense stacking.
- Feature artifacts can be cached to reduce recomputation.

### D. Training & Validation
- `src/pipelines/train_pipeline.py` drives train-time orchestration.
- `src/training/trainer.py` runs CV, computes metrics (RMSE/MAE/R2/SMAPE), persists fold models and OOF predictions.
- `src/models/stacker.py` supports meta-ensemble stacking.

### E. Inference
- Offline inference path: `src/pipelines/inference_pipeline.py` + `src/inference/predict.py`.
- Online API baseline: `src/serving/app.py` (FastAPI) exposing:
  - `/healthz`,
  - `/readyz`,
  - `/v1/warmup`,
  - `/v1/predict`.

### F. Production Serving Topology (Target)
For 1M+ DAU and P99 < 1s, production flow should be:

1. Request enters API Gateway / LB.
2. Serving service fetches hot features from **Redis** (sub-millisecond lookup).
3. Fast path model returns response within strict budget.
4. Heavy multimodal recalculation (if needed) runs async via Kafka workers.
5. Logs/metrics/traces stream to monitoring stack.

---

## 1.2 Microservices vs Modular Monolith

### Current State: Modular Monolith (Repository)
- The codebase is structured in clear domain modules (`data`, `features`, `models`, `training`, `inference`, `pipelines`).
- This is ideal for rapid iteration and lower operational overhead.

### Production Recommendation: Hybrid Evolution
- Keep training/experimentation as modular internal platform services.
- Split latency-critical inference into dedicated microservices:
  - **Online Inference Service** (FastAPI or gRPC),
  - **Feature Service** (Redis-backed online feature fetch),
  - **Async Scoring/Backfill Workers** (Kafka consumers),
  - **Monitoring/Drift Service**.

### Why this choice
- Modular monolith is simpler for dev velocity.
- Microservices are preferred for independent scaling and fault isolation at 1M+ DAU.
- Hybrid model gives best trade-off: engineering speed + operational resilience.

---

## 2) Feature Engineering & Feature Store

## 2.1 Current Feature Pipeline
- Feature orchestration entrypoint: `FeatureBuilder`.
- Text: SBERT/TF-IDF style embeddings (depending on config/runtime availability).
- Image: CLIP-style embedding pipeline (where dependencies/artifacts available).
- Numeric: parsed and scaled numeric features from extracted fields.
- Combined representation supports sparse and dense model families.

## 2.2 Online vs Offline Features (Production Design)

### Offline Features
- Computed in batch from historical data.
- Stored in offline store (e.g., parquet/lakehouse/warehouse tables).
- Used for training, backtesting, and CT data snapshots.

### Online Features
- Low-latency subset required for synchronous inference.
- Stored in **Redis** as key-value feature vectors keyed by entity ID (e.g., `product_id`).
- TTL and versioning enforce freshness and rollback safety.

## 2.3 Synchronization Logic
Use a single feature definition source and dual materialization:

1. Define feature transformations once (versioned spec).
2. Batch job materializes offline training datasets.
3. Streaming job materializes online Redis feature keys from Kafka events.
4. Every training run records `feature_schema_version` and `feature_view_hash`.
5. Inference validates incoming request + feature schema compatibility.

## 2.4 Preventing Training-Serving Skew
- Same transformation code path (or same generated feature definitions) for both online/offline.
- Schema contract checks at train and serve time.
- Point-in-time correct joins for offline labels.
- Periodic parity tests: offline vs online feature value comparison for sampled entities.
- Drift dashboards segmented by feature group.

---

## 3) Model Development & MLOps

## 3.1 Experimentation Phase

### Current
- Config-driven experiments under `configs/` and `src/experiments/`.
- Artifact outputs under `experiments/` (models, logs, reports, submissions).

### Target / Required for Production
Use **MLflow** or **Weights & Biases (W&B)** as first-class experiment tracker:
- run metadata (git SHA, config hash, data snapshot id),
- metrics by fold and global,
- model artifact lineage,
- comparison dashboards,
- model registry stages (Staging/Canary/Production).

## 3.2 CI/CD Pipeline for Models

### Current
- Baseline CI in `.github/workflows/ci.yml` includes:
  - dependency install,
  - syntax gate,
  - unit test gate (`ci_cd/tests`),
  - CLI smoke gate.

### Target Model CI/CD
Recommended stages:
1. Lint + type + tests.
2. Data contract validation.
3. Train candidate model (or run scheduled training).
4. Evaluate against quality gates (offline + shadow metrics).
5. Package model (ONNX/serialized artifacts).
6. Deploy to staging (shadow mode).
7. Automated promotion/canary if SLO + quality pass.
8. Auto rollback if SLO/error/drift thresholds breach.

## 3.3 Continuous Training (CT) Triggers
CT should run on event and schedule:
- data volume threshold reached,
- drift threshold exceeded,
- periodic retraining window (daily/weekly),
- model KPI degradation from online feedback,
- manual release trigger for emergency patches.

For each CT run record:
- training data snapshot id,
- feature schema version,
- code revision,
- reproducibility seed,
- eval report + approval status.

---

## 4) Inference Strategy

## 4.1 Serving Layer (FastAPI vs gRPC)

### FastAPI
- Faster to implement and integrate.
- Good for REST-first product teams.
- Current baseline implemented in `src/serving/app.py`.

### gRPC (recommended for ultra-low overhead)
- Better binary protocol efficiency.
- Strong interface contracts via protobuf.
- Preferred for high-throughput internal service-to-service calls.

Recommended pattern:
- external traffic: REST/FastAPI gateway,
- internal mesh: gRPC for hot path calls.

## 4.2 Model Optimization Techniques
To sustain P99 < 1s:
- **ONNX export** for optimized runtime portability.
- **Quantization** (dynamic/static) for lower latency and memory.
- **Pruning** for reducing model size and compute.
- Batch tuning + thread affinity + warm model pools.
- Precompute/cache expensive embeddings.

## 4.3 Traffic Management & Compute
- Deploy on **Docker** containers orchestrated by **Kubernetes**.
- Use L7 load balancers + HPA/KEDA autoscaling.
- Scale policies should use:
  - CPU/GPU utilization,
  - request concurrency,
  - p95/p99 latency,
  - queue lag (Kafka consumers).

---

## 5) Monitoring & Governance

## 5.1 Real-Time Drift Detection

### Data Drift
Monitor feature distribution shifts:
- PSI/KS/Jensen-Shannon for key features,
- null/missing spikes,
- cardinality explosion in categorical features.

### Concept Drift
Monitor prediction-target relationship decay:
- rolling performance windows,
- delayed-label evaluation,
- calibration drift,
- business KPI deltas.

## 5.2 Logging/Observability Stack
Use structured observability across layers:
- **Prometheus**: metrics scraping (latency, QPS, errors, drift scores).
- **Grafana**: SLO dashboards and alerting.
- **ELK** (Elasticsearch/Logstash/Kibana): searchable structured logs.
- Optional distributed tracing (OpenTelemetry) for hop-level latency attribution.

Minimum production tags on every prediction log:
- request_id,
- model_version,
- feature_schema_version,
- latency_ms,
- outcome_status,
- fallback_used,
- canary_bucket.

## 5.3 Automated Rollback Triggers
Rollback should be automatic when any guardrail is violated:
- P99 latency > SLO threshold for sustained window,
- 5xx/error-rate spikes,
- severe drift alert + quality degradation,
- shadow-vs-prod divergence over threshold,
- business KPI regression beyond control bounds.

Rollback mechanics:
- instant route switch to previous stable model,
- preserve canary telemetry for RCA,
- create incident ticket with snapshot context.

---

## 6) Developer Quick Start

## 6.1 Repository Mental Model
Key paths:
- `main.py`: CLI entrypoint.
- `src/data/`: loading/parsing.
- `src/features/`: feature construction and embeddings.
- `src/models/`: model wrappers.
- `src/training/`: CV trainer + metrics.
- `src/inference/`: prediction/postprocess.
- `src/pipelines/`: train/infer/ensemble orchestration.
- `src/serving/`: online inference API baseline.
- `configs/`: run-time configs.
- `ci_cd/tests/`: lightweight CI tests.
- `experiments/`: generated artifacts/reports/models.

## 6.2 Environment Setup
1. Create and activate Python 3.10 environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Sanity checks:
   - `python -m compileall src main.py`
   - `pytest -q ci_cd/tests`
   - `python main.py --help`

## 6.3 Core Commands
- Train:
  - `python main.py train --config configs/training/final_train.yaml`
- Train one model only:
  - `python main.py train --config configs/training/final_train.yaml --model lgbm`
- Offline inference:
  - `python main.py inference --config configs/inference/inference.yaml`
- Feature pipeline:
  - `python main.py features --config configs/features/all_features.yaml`

## 6.4 Local Shadow Inference Test (Recommended)
Goal: validate online service behavior without affecting user-facing decisions.

1. Start serving API:
   - `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000`
2. Warmup model cache:
   - `POST /v1/warmup`
3. Send sample requests copied from production-like payloads.
4. Compare shadow predictions with current baseline model outputs.
5. Log divergence metrics:
   - absolute error delta,
   - rank/order consistency,
   - latency differences.

---

## 7) What We Have Done in This Project (As-Built Summary)

This section is specifically for new developers joining before further improvements.

## 7.1 Implemented Foundations
- Modular ML code architecture across data/features/models/training/inference.
- Config-driven CLI entrypoint (`train`, `inference`, `features`, `ensemble`, `quickrun`).
- Cross-validation trainer with fold artifact persistence and metrics.
- Ensemble stacking support via OOF matrix + meta model.
- Inference pipeline with post-processing and artifact discovery.
- Baseline serving API with health/readiness/warmup/predict endpoints.
- CI quality gates (syntax/tests/smoke).
- SLO guidance document and production-readiness audit baseline.

## 7.2 Methods & Techniques Used
- Multimodal features (text + image + numeric).
- Model families: linear/forest/boosting (optional XGB/Cat depending on environment).
- CV-based evaluation + SMAPE-centric reporting.
- Feature/model artifact caching via joblib.
- Stacking ensemble for improved generalization.

## 7.3 Current Operational Gaps (Known)
- Full production feature store is not yet implemented.
- Drift/rollback automation not yet wired end-to-end.
- Dedicated Kafka/Redis/Kubernetes runtime topology is design target, not fully integrated in this repo yet.
- Serving optimization (ONNX/quantization/pruning) requires implementation pass.

These are the primary next workstreams for platform hardening.

---

## 8) Team Working Model for Fast Onboarding

## First 2 Days
- Run local environment and CI checks.
- Execute one train + one inference run.
- Read `src/pipelines/*` and `src/serving/app.py` fully.

## Week 1 Delivery Goal
- Pick one subsystem (feature quality, serving performance, or MLOps automation).
- Ship one measurable improvement with tests.
- Update this handover doc with architectural delta.

## Engineering Standards
- Prefer config-driven changes over hard-coded constants.
- Preserve train/serve schema contracts.
- Add tests for every bug fix and production incident class.
- Track model/data/version lineage for reproducibility.

---

## 9) Glossary
- **DAU**: Daily Active Users.
- **P99 latency**: 99th percentile request latency.
- **CT**: Continuous Training.
- **Training-serving skew**: mismatch between train-time and runtime features.
- **Canary**: partial rollout before full production promotion.
- **Shadow inference**: model runs in parallel without affecting user response.

---

## Final Notes
This repository now has a strong ML platform base and clear production direction. For the 1M+ DAU, P99 < 1s target, prioritize:

1. Redis-backed online features,
2. Kafka event pipelines,
3. gRPC-internal serving path,
4. ONNX/quantization optimization,
5. full drift/rollback automation on Kubernetes.

Once these are complete, the system will move from advanced pre-production to robust industry-grade operation.
