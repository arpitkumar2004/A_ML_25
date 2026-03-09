# Production Runbook

## 1. Deployment Steps
- Ensure `.env` is present (copy from `.env.example`) and set `MODELS_DIR`, `DIM_CACHE`, `TFIDF_VECTORIZER_PATH`, `NUMERIC_SCALER_PATH`, optional `FEATURE_SELECTOR_PATH`.
- Start API: `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000`.
- Verify health and readiness:
  - `GET /healthz`
  - `GET /readyz`
- Verify metrics scrape endpoint:
  - `GET /metrics` (Prometheus text format)

## 2. Rollback Procedure
- List registry entries: `python main.py list-registry --registry_dir experiments/registry`.
- Roll back production pointer:
  - `python main.py rollback --to_run_id <RUN_ID> --registry_dir experiments/registry`
- Restart service after rollback so it picks intended artifacts.

## 3. Alert Triage
- Run dashboard build manually:
  - `python scripts/build_monitoring_dashboard.py --config configs/monitoring/alerts.yaml`
- Check critical alerts:
  - `python scripts/check_monitoring_alerts.py --dashboard experiments/monitoring/dashboard.json`
- Inspect payload with top drift features and PSI:
  - `experiments/monitoring/alert_payload.json`

## 4. SLO Breach Response
- Latency breach:
  - Review `/metrics` p95/p99 latency lines.
  - Check `experiments/monitoring/latency_events.jsonl`.
  - If persistent, route traffic to previous production run via rollback.
- Drift breach:
  - Inspect top PSI features in dashboard/payload.
  - Trigger policy-driven retraining dry run:
    - `python scripts/retrain_orchestrator.py --policy configs/monitoring/retrain_policy.yaml`
  - Execute retraining if approved:
    - `python scripts/retrain_orchestrator.py --policy configs/monitoring/retrain_policy.yaml --execute`

## 5. Data Quality Guardrails
- Batch inference applies reject/quarantine policy in `src/pipelines/inference_pipeline.py`.
- Serving inference applies reject/quarantine for bad request batches in `src/serving/app.py`.
- Quarantine outputs:
  - Batch: `experiments/quarantine/inference_quarantine.csv`
  - Serving: `experiments/quarantine/serving_quarantine.csv`

## 6. Daily Automation
- GitHub workflow `.github/workflows/daily-monitoring.yml` runs daily.
- It generates dashboard, appends drift snapshot history, emits alert payload, and uploads artifacts.

## 7. On-call Quick Commands
- Validate configs: `python scripts/validate_configs.py`
- Run smoke train+predict: `python scripts/smoke_train_predict.py`
- Build monitoring report: `python scripts/run_daily_monitoring.py`
- Retrain dry run: `python scripts/retrain_orchestrator.py`
