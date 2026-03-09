# Production Runbook

## Table of Contents

1. [Deployment & Serving](#1-deployment--serving)
2. [Rollback Procedures](#2-rollback-procedures)
3. [Alert Triage](#3-alert-triage)
4. [Phase 3: CI/CD Operations](#4-phase-3-cicd-operations)
5. [Emergency Response](#5-emergency-response)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Deployment & Serving

### Initial Deployment

- Ensure `.env` is present (copy from `.env.example`) and set:
  - `MODELS_DIR`: Path to serving model directory
  - `DIM_CACHE`: Path to dimensionality reduction cache
  - `TFIDF_VECTORIZER_PATH`: Path to fitted vectorizer
  - `NUMERIC_SCALER_PATH`: Path to numeric feature scaler

### Start API Server

```bash
# Activate venv
. .venv/Scripts/Activate.ps1

# Start serving
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### Verify Health

```bash
# Healthz endpoint (liveness)
curl http://localhost:8000/healthz

# Readyz endpoint (readiness)
curl http://localhost:8000/readyz

# Metrics (Prometheus format)
curl http://localhost:8000/metrics | grep inference
```

---

## 2. Rollback Procedures

### Quick Rollback (from CLI)

```bash
# List available runs in registry
python main.py list-registry --registry_dir experiments/registry

# Rollback to previous production
python scripts/rollback_deployment.py \
  --to-previous-production \
  --reason "Manual rollback: P95 latency spike"

# Restart serving to pick new model
# (for containerized: redeploy; for manual: restart uvicorn)
```

### Registry-based Rollback

```bash
# Check current production
cat experiments/registry/index.json | jq '.active_production_run_id'

# Find previous production (archived)
cat experiments/registry/index.json | jq '.runs[] | select(.status=="archived") | .run_id' | head -1

# Promote previous to production
python scripts/promote_model.py \
  --run-id <previous_run_id> \
  --target-stage production \
  --promoted-by "manual-rollback"

# Restart service
```

### Disaster Recovery (Full Restore)

```bash
# Restore registry from git
git checkout HEAD~5 -- experiments/registry/index.json
git commit -m "restore: recover registry"

# Restore model artifacts from DVC
dvc pull -r origin experiments/models/
```

---

## 3. Alert Triage

### Manual Health Check

```bash
# Run comprehensive health check
python scripts/health_check.py \
  --check-mlflow \
  --check-production-model \
  --check-registry \
  --check-inference \
  --output /tmp/health_report.json

# Review results
cat /tmp/health_report.json | jq '.'
```

### Build Monitoring Dashboard

```bash
python scripts/build_monitoring_dashboard.py \
  --config configs/monitoring/alerts.yaml \
  --output experiments/monitoring/dashboard.json
```

### Check Critical Alerts

```bash
python scripts/check_monitoring_alerts.py \
  --dashboard experiments/monitoring/dashboard.json \
  --severity critical
```

### Inspect Alert Payload

```bash
# Review detailed alert information
cat experiments/monitoring/alert_payload.json | jq '.critical_alerts'

# Check drift features
cat experiments/monitoring/alert_payload.json | jq '.drift_snapshot'
```

---

## 4. Phase 3: CI/CD Operations

### Automated Training Pipeline

**Manual Trigger** (Force Retrain):
```bash
gh workflow run training.yml -f force_retrain=true
```

**Check Training Status**:
```bash
gh run list -w training.yml --limit 1 --json startedAt,conclusion,startedAt
```

**Get Run ID from Successful Training**:
```bash
# Method 1: From MLflow
mlflow experiments get --experiment-name "First Experiment - A_ML_25" | grep run_id

# Method 2: From GitHub Actions
gh run view <run-id> --json title | jq '.'
```

### Model Promotion Workflow

**Promote to Staging** (Automatic or Manual):
```bash
gh workflow run promote.yml \
  -f run_id="<mlflow_run_id>" \
  -f target_stage="staging"
```

**Promote to Canary**:
```bash
gh workflow run promote.yml \
  -f run_id="<mlflow_run_id>" \
  -f target_stage="canary"
```

**Promote to Production** (Requires Approval):
```bash
gh workflow run promote.yml \
  -f run_id="<mlflow_run_id>" \
  -f target_stage="production"

# Then approve in GitHub Actions UI or environment
```

**Check Promotion Status**:
```bash
# View promotion history
cat experiments/registry/promotion_history.jsonl | jq '.'

# Find which run is currently production
cat experiments/registry/index.json | jq '.runs[] | select(.status=="production")'
```

### Deployment Workflow

**Deploy with Canary Strategy**:
```bash
gh workflow run deploy.yml \
  -f run_id="<run_id>" \
  -f deployment_strategy="canary" \
  -f canary_percent="10"
```

**Deploy with Blue-Green Strategy**:
```bash
gh workflow run deploy.yml \
  -f run_id="<run_id>" \
  -f deployment_strategy="blue_green"
```

**Monitor Canary Metrics**:
```bash
# Check ongoing canary config
cat experiments/registry/canary_config.json | jq '.'

# If auto-promoted, config will show:
# "canary_percent": 100,
# "promoted_to_production_time": "..."
```

### Health Check Workflow

**Manual Run**:
```bash
gh workflow run health-check.yml
```

**View Health Report**:
```bash
# Download latest health check artifact
gh run download <run-id> -n health-report
cat /tmp/health_report.json | jq '.checks'
```

**Automatic Trigger**: Runs every 6 hours (0, 6, 12, 18 UTC)

---

## 5. Emergency Response

### Incident: Model Latency Spike (P95 > 2s)

**Step 1: Verify Issue** (1 min)
```bash
# Check current production metrics
cat experiments/registry/production_tracker.json | jq '.metrics.latency_p95'

# If spike confirmed, proceed to Step 2
```

**Step 2: Immediate Action** (2 min)
```bash
# Option A: Auto-rollback (if within 15 min of deploy)
python scripts/rollback_deployment.py \
  --to-previous-production \
  --reason "Latency spike: P95 > 2.0s"

# Option B: If auto-rollback didn't trigger, manual rollback
python scripts/promote_model.py \
  --run-id <previous_run_id> \
  --target-stage production
```

**Step 3: Restart Service** (1 min)
```bash
# For containerized serving
kubectl rollout restart deployment/model-serving

# For manual serving
# Kill and restart uvicorn process
```

**Step 4: Verify Recovery** (1 min)
```bash
curl http://localhost:8000/healthz
# Should return 200

# Check latency
python scripts/health_check.py --check-inference --output /tmp/check.json
cat /tmp/check.json | jq '.metrics.latency_p95'
# Should be < 2s
```

### Incident: Model Accuracy Drop (< 0.65)

**Step 1: Pause Serving** (1 min)
```bash
# Disable inference endpoint (deployment-specific)
# E.g., Kubernetes: kubectl scale deployment/model-serving --replicas=0
```

**Step 2: Investigate** (5-10 min)
```bash
# Check data drift
python scripts/check_production_drift.py \
  --baseline experiments/registry/baseline_stats.json \
  --alert-threshold 0.25 \
  --output /tmp/drift.json
cat /tmp/drift.json | jq '.drift_detected'

# If drift detected: data problem
# If no drift: model regression
```

**Step 3: Resolve** (10-30 min)
```bash
# If data drift:
# 1. Investigate data source
# 2. Pause new data ingestion if needed
# 3. Escalate to data team
# 4. Retrain when issue resolved

# If model regression:
python scripts/rollback_deployment.py \
  --to-previous-production \
  --reason "Accuracy dropped from 0.72 to 0.63"
```

**Step 4: Resume Serving** (1 min)
```bash
kubectl scale deployment/model-serving --replicas=3  # or your deployment count
```

### Incident: Training Pipeline Fails

**Step 1: Check Error** (1 min)
```bash
gh run view <run-id> -v | tail -50
```

**Step 2: Common Fixes** (5-10 min)
```bash
# If credential error:
gh secret set MLFLOW_TRACKING_PASSWORD -b "new-token"

# If DVC pull fails:
dvc pull -r origin

# If syntax error:
git log --oneline -5
git show <commit>  # Review change

# If timeout:
# Edit .github/workflows/training.yml
# Change: timeout-minutes: 120 → 180
```

**Step 3: Retry** (120+ min)
```bash
gh workflow run training.yml -f force_retrain=true
```

### Incident: Registry Corruption

**Step 1: Detect** (1 min)
```bash
python -c "import json; json.load(open('experiments/registry/index.json'))" || echo "CORRUPTED"
```

**Step 2: Restore** (2 min)
```bash
# Find last good commit
git log --oneline -- experiments/registry/index.json | head -3

# Restore
git checkout <good-commit> -- experiments/registry/index.json
git add experiments/registry/index.json
git commit -m "restore: recover registry from corruption"
git push
```

---

## 6. Troubleshooting

### Common Issues & Solutions

#### Training exceeds 120 minutes
```bash
# Reduce data:
# Edit configs/training/final_train.yaml
# max_samples: 50000

# Or increase timeout:
# Edit .github/workflows/training.yml
# timeout-minutes: 180
```

#### MLflow connection fails
```bash
# Check URI and credentials
echo $MLFLOW_TRACKING_URI
echo $MLFLOW_TRACKING_USERNAME

# Test connection locally
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"

# Verify GitHub Secrets
gh secret list
```

#### DVC pull times out
```bash
# Check network
ping dagshub.com

# Try with explicit timeout
dvc pull -r origin --cd-list-timeout 30

# Or manually download
# E.g., download from S3 directly if using AWS
```

#### Inference latency high
```bash
# Profile model
python scripts/profile_model.py \
  --model-id <run_id> \
  --samples 1000 \
  --output /tmp/profile.json

# Check for bottlenecks
cat /tmp/profile.json | jq '.slowest_steps'
```

#### GitHub Actions quota exceeded
```bash
# Check usage
gh api repos/arpitkumar2004/A_ML_25/actions/runners

# Optimize:
# - Reduce training frequency
# - Use caching more aggressively
# - parallelize CI tests
# - Use self-hosted runners if available
```

---

## Quick Command Reference

```bash
# Training & Models
gh workflow run training.yml -f force_retrain=true
gh workflow run promote.yml -f run_id="abc123" -f target_stage="canary"
gh workflow run deploy.yml -f run_id="abc123" -f deployment_strategy="canary"

# Health & Diagnostics
python scripts/health_check.py --check-production-model --output /tmp/check.json
python scripts/validate_production_model.py
python scripts/check_data_drift.py --compare-against experiments/registry/baseline_stats.json --output /tmp/drift.json

# Registry Operations
cat experiments/registry/index.json | jq '.'
cat experiments/registry/promotion_history.jsonl | jq '.'
python scripts/rollback_deployment.py --to-previous-production --reason "Incident response"

# Monitoring & Alerts
python scripts/build_monitoring_dashboard.py --config configs/monitoring/alerts.yaml
python scripts/check_monitoring_alerts.py --dashboard experiments/monitoring/dashboard.json

# Debugging
git log --oneline -10
dvc status
dvc pull -r origin
gh run list -w training.yml --limit 5
```

---

## On-Call Schedule

| Weekend | Primary | Backup |
|---------|---------|--------|
| Monday-Friday | Team A | Team B |
| Saturday-Sunday | Team C | Team A |

**Escalation:**
- Tier 1 (on-call): Page immediately
- Tier 2 (backup): Call after 30 min if T1 no response
- Tier 3 (manager): Call after 60 min

**Contact**: PagerDuty or #ml-ops-incidents Slack channel

---

## SLOs & Targets

- **Availability**: 99.5% (acceptable downtime: ~3.6 hrs/month)
- **P95 Latency**: ≤ 2.0 seconds
- **Error Rate**: < 0.02 (2%)
- **Model Freshness**: ≤ 7 days since training
- Run smoke train+predict: `python scripts/smoke_train_predict.py`
- Build monitoring report: `python scripts/run_daily_monitoring.py`
- Retrain dry run: `python scripts/retrain_orchestrator.py`
