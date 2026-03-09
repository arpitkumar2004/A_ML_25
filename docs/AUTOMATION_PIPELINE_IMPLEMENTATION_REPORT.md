# Automation Pipeline Implementation Report (Recommendation Stack)

## 1. Objective
This report provides a complete implementation plan to automate the MLOps pipeline for this repository using a practical recommendation stack.

Primary goals:
- automate training, validation, and deployment decisions,
- track experiments and model lineage,
- monitor drift/latency and trigger retraining,
- make releases reproducible and rollback-safe.

## 2. Recommended Automation Stack
Use this stack in phases (from lowest risk to highest impact):

1. `GitHub Actions` for CI/CD orchestration.
2. `MLflow` for experiment tracking and model registry.
3. `DVC` for data and artifact versioning.
4. `Prefect` for scheduled and dependency-aware pipeline orchestration.
5. `Prometheus + Grafana` for service telemetry.
6. `Evidently` (or current PSI framework + webhook) for data/model drift reports.
7. `Docker` and optional `Kubernetes` for scalable runtime deployment.

## 3. Current State in This Repository
Already implemented:
- CI workflow: `.github/workflows/ci.yml`.
- Daily monitoring workflow: `.github/workflows/daily-monitoring.yml`.
- Training and inference pipelines with manifests:
  - `src/pipelines/train_pipeline.py`
  - `src/pipelines/inference_pipeline.py`
- Custom local model registry:
  - `src/registry/model_registry.py`
- Serving with health/readiness/metrics endpoints:
  - `src/serving/app.py`
- Monitoring report and alert checks:
  - `scripts/build_monitoring_dashboard.py`
  - `scripts/check_monitoring_alerts.py`
  - `scripts/retrain_orchestrator.py`

Partially implemented or missing:
- MLflow integration is a placeholder (`ci_cd/mlflow/mlflow.yml`).
- Dataset/artifact versioning is not governed by DVC yet.
- No full deployment workflow (`build image -> deploy staging -> smoke -> promote`).
- Alert delivery to Slack/Teams/PagerDuty is not wired.
- Pipeline scheduling and dependency management rely on scripts, not a dedicated orchestrator.

## 4. Target Architecture (End State)

1. Developer pushes code or data update.
2. CI runs tests, config checks, smoke jobs.
3. Orchestrator launches training flow.
4. Training logs params/metrics/artifacts to MLflow.
5. Candidate model is validated against quality gates.
6. Candidate is deployed to staging/canary.
7. Monitoring tracks latency, error, drift, fallback rates.
8. Policy engine promotes to production or rolls back.
9. Drift/latency triggers retraining flow automatically.

## 5. Implementation Plan (Step-by-Step)

## Phase 0: Baseline Freeze and Safety

Step 0.1: Pin environment and make runs reproducible
- keep using `.venv` and `requirements.txt`.
- add dependency lock generation process (example: `pip freeze > requirements-lock.txt`).
- capture `PYTHONHASHSEED`, training seed, and config hash in run manifests.

Step 0.2: Define quality gates before automation
- keep current gates in CI.
- add explicit minimum acceptance thresholds in config, for example:
  - max SMAPE,
  - max p95 latency,
  - max critical drift features.

Deliverables:
- `configs/training/final_train.yaml` includes `quality_gates` section.
- `configs/monitoring/retrain_policy.yaml` aligned with these gates.

## Phase 1: MLflow Tracking and Registry Integration

Step 1.1: Add dependencies
- add `mlflow` to `requirements.txt`.

Step 1.2: Instrument training pipeline
- in `src/pipelines/train_pipeline.py`:
  - open MLflow run with `run_id` and config tags,
  - log parameters from training config,
  - log metrics (RMSE, MAE, R2, SMAPE),
  - log artifacts (model report CSV, stacker summary, manifests).

Step 1.3: Instrument inference pipeline
- in `src/pipelines/inference_pipeline.py`:
  - log inference latency and rows,
  - log data quality policy result,
  - attach submission artifact path.

Step 1.4: Keep custom registry as fallback
- preserve `src/registry/model_registry.py`.
- add `mlflow_run_id` in registry manifest for cross-reference.

Step 1.5: Add MLflow service profile
- replace placeholder `ci_cd/mlflow/mlflow.yml` with runnable config.
- local command example:
  - `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`

Deliverables:
- training/inference runs visible in MLflow UI.
- each run mapped to local `run_id` and manifest path.

## Phase 2: DVC Data and Artifact Versioning

Step 2.1: Initialize DVC
- run:
  - `dvc init`
  - `dvc remote add -d storage <REMOTE_URI>`

Step 2.2: Track datasets and core artifacts
- track:
  - `data/raw/train.csv`
  - `data/raw/test.csv`
  - optional processed snapshots and reference baselines.

Step 2.3: Version training outputs for reproducibility
- track critical artifacts:
  - model bundles,
  - vectorizer/scaler,
  - OOF metadata,
  - monitoring baselines.

Step 2.4: Tie DVC revision to MLflow
- log DVC git revision hash as MLflow tag (`dvc_rev`).

Deliverables:
- reproducible `code + data + model` lineage.
- ability to recreate model from a specific commit.

## Phase 3: CI/CD Automation Upgrade

Step 3.1: Keep current CI and add model quality stage
- extend `.github/workflows/ci.yml` with:
  - config validation,
  - smoke train/predict,
  - quality-gate script that fails if thresholds are breached.

Step 3.2: Add deployment workflow
Create `.github/workflows/deploy.yml` with stages:
1. build Docker image,
2. run integration tests,
3. deploy to staging,
4. call `/readyz` and `/v1/predict` smoke endpoint,
5. promote canary to production if healthy.

Step 3.3: Add rollback action
- auto rollback when:
  - error rate exceeds threshold,
  - p95/p99 exceeds threshold for sustained window,
  - critical drift alert persists.

Deliverables:
- automated CI + CD with staged promotion and rollback.

## Phase 4: Orchestration with Prefect

Step 4.1: Create flow modules
- `src/orchestration/flows/train_flow.py`
- `src/orchestration/flows/inference_flow.py`
- `src/orchestration/flows/monitoring_flow.py`

Step 4.2: Wrap existing scripts as tasks
- use existing scripts as building blocks:
  - `scripts/validate_configs.py`
  - `scripts/smoke_train_predict.py`
  - `scripts/run_daily_monitoring.py`
  - `scripts/retrain_orchestrator.py`

Step 4.3: Define schedules
- daily monitoring schedule.
- weekly retraining schedule.
- event trigger for drift threshold breach.

Step 4.4: Add retries and alerts
- each task has retry policy.
- on failure, send alert webhook.

Deliverables:
- one orchestration control plane for all automation jobs.

## Phase 5: Monitoring and Alert Delivery

Step 5.1: Keep current metrics endpoints
- continue exposing `/metrics` and `/metrics/json` from `src/serving/app.py`.

Step 5.2: Add Prometheus scrape + Grafana dashboard
- include service metrics:
  - request count,
  - error count,
  - predict count,
  - p50/p95/p99 latency,
  - fallback feature rates.

Step 5.3: Add alert routing
- route alert payload from `experiments/monitoring/alert_payload.json` to:
  - Slack or Teams,
  - PagerDuty for critical severity.

Step 5.4: Add SLO burn-rate alerts
- add short-window and long-window burn-rate rules for error and latency.

Deliverables:
- actionable real-time alerting instead of manual checks.

## Phase 6: Controlled Promotion Strategy

Step 6.1: Formalize stages
- `staging -> canary -> production -> archived`.

Step 6.2: Define automatic promotion policy
- promote to production only when all are true:
  - offline metrics pass gates,
  - online canary health passes,
  - no critical monitoring alerts.

Step 6.3: Keep manual override path
- retain commands:
  - `python main.py promote ...`
  - `python main.py rollback ...`

Deliverables:
- deterministic, auditable release strategy.

## 6. File-Level Change Plan

Minimum files to add or update:
- `requirements.txt` (add `mlflow`, `dvc`, `prefect`, optional `prometheus-client`).
- `src/pipelines/train_pipeline.py` (MLflow logging).
- `src/pipelines/inference_pipeline.py` (MLflow logging).
- `ci_cd/mlflow/mlflow.yml` (real config, not placeholder).
- `.github/workflows/ci.yml` (quality gate extension).
- `.github/workflows/deploy.yml` (new deployment workflow).
- `scripts/quality_gate.py` (new gate evaluator).
- `src/orchestration/flows/*.py` (Prefect flows).
- `docs/PRODUCTION_RUNBOOK.md` (add new deployment/rollback commands).

## 7. Acceptance Criteria
A phase is complete only if all checks pass:

- Reproducibility:
  - same commit + same DVC revision + same config reproduces near-identical metrics.
- Observability:
  - metrics visible in dashboard and alerts fire on synthetic breach tests.
- Reliability:
  - canary failure triggers rollback automatically.
- Traceability:
  - every production model maps to config hash, data revision, and MLflow run.

## 8. Execution Timeline (Suggested)

Week 1:
- Phase 0 and Phase 1 (MLflow).

Week 2:
- Phase 2 (DVC) and CI quality gate hardening.

Week 3:
- Phase 3 deployment pipeline + rollback automation.

Week 4:
- Phase 4 orchestration + Phase 5 alert routing.

Week 5:
- stabilization, runbooks, and failure drills.

## 9. Risk Register and Mitigation

Risk: Training-serving feature mismatch.
- Mitigation: feature schema hash checks in train/infer/serve.

Risk: Alert noise or false positives.
- Mitigation: burn-rate logic and severity thresholds.

Risk: Slow deployment feedback loop.
- Mitigation: fast smoke tests before expensive stages.

Risk: Tooling complexity increase.
- Mitigation: phase rollout, keep custom registry until MLflow path is stable.

## 10. Operational Commands Reference

Core commands already used in this repo:
- `python scripts/validate_configs.py`
- `python main.py train --config configs/training/final_train.yaml`
- `python main.py inference --config configs/inference/inference.yaml`
- `python scripts/run_daily_monitoring.py --config configs/monitoring/alerts.yaml`
- `python scripts/retrain_orchestrator.py --policy configs/monitoring/retrain_policy.yaml`
- `python main.py list-registry --registry_dir experiments/registry`
- `python main.py rollback --to_run_id <RUN_ID> --registry_dir experiments/registry`

Proposed new commands after implementation:
- `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
- `dvc repro`
- `prefect deploy --all`

## 11. Final Recommendation
Implement in phases and do not replace all systems at once.

Practical order for this repository:
1. MLflow instrumentation,
2. DVC lineage,
3. deployment workflow with rollback,
4. orchestration and alert routing,
5. optional Kubernetes scale-out.

This order minimizes risk, preserves current functionality, and gives measurable automation value quickly.