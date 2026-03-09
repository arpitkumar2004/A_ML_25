# Phase 3: CI/CD Automation Upgrade — Implementation Guide

## Overview

Phase 3 implements a production-grade, automated ML deployment pipeline following industry best practices. This phase introduces:

1. **Training Pipeline**: Automated retraining with drift detection
2. **Model Promotion**: Stage-based promotion (staging → canary → production)
3. **Deployment Strategies**: Canary, blue-green, and rolling deployments
4. **Health Monitoring**: Continuous validation of model and system health
5. **Rollback Safety**: Automatic rollback on degradation or failures
6. **Observability**: Comprehensive logging, metrics, and alerting
7. **GitOps**: Infrastructure and model versioning via Git + tags

---

## Architecture Overview

```
┌─────────────────┐
│   Code Push     │
│   (main/PR)     │
└────────┬────────┘
         │
         ├─→ CI Gate (syntax, tests, hygiene)
         │
         ├─→ Drift Check (production data patterns)
         │
         ├─→ Scheduled Training (daily 22:00 UTC or on-demand)
         │
         ├─→ Model Registry (staging stage)
         │
         ├─→ Manual Promotion (staging → canary/production)
         │
         ├─→ Health Checks Pre-Deployment
         │
         └─→ Deployment (canary or blue-green)
              │
              ├─→ Monitor Canary Metrics
              │
              ├─→ Auto-Promote or Rollback
              │
              └─→ Production (6-hr health checks, auto-rollback on failure)
```

---

## 1. GitHub Secrets Configuration

All sensitive credentials must be stored as **GitHub Secrets** (not `.env` in Git).

### Required Secrets

```
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
DAGSHUB_USERNAME           (if using DagsHub for remote)
DAGSHUB_TOKEN             (if using DagsHub for remote)
AWS_ACCESS_KEY_ID          (for S3 artifact storage)
AWS_SECRET_ACCESS_KEY      (for S3 artifact storage)
```

### How to Set GitHub Secrets

1. Go to **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Add each secret with exact name and value
4. **Never** commit `.env` file with real secrets!

### Local Development

Create `.env` file locally (not in Git):
```bash
cp .env.example .env
# Edit .env with actual values
# Add .env to .gitignore (already done)
```

### CI/CD Access

Workflows access secrets via:
```yaml
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
  MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
```

---

## 2. Branch Protection & Merge Strategy

### Production Branch Protection Rules

Enable on `main` branch (Settings → Branches → Add rule):

```
Branch name pattern: main

Require:
  ✓ Pull request reviews before merging (1 approver)
  ✓ Dismiss stale PR approvals when new commits pushed
  ✓ Require status checks to pass before merging
      - test-and-quality (from .github/workflows/ci.yml)
      - All required workflows must complete
  ✓ Require branches to be up to date before merging
  ✓ Require conversation resolution before merging
  ✓ Require signed commits
```

### Merge Strategy

- **Squash and merge**: Keep main history clean, one commit per feature
- **Delete head branch**: Auto-cleanup after merge
- **Require PR description**: Document why, what, impacts

### Workflow Triggers

- **CI (ci.yml)**: Runs on push to `main` and all PRs
- **Training (training.yml)**: Daily 22:00 UTC + manual trigger
- **Promotion (promote.yml)**: Manual dispatch only
- **Deployment (deploy.yml)**: Manual dispatch only
- **Health Check (health-check.yml)**: Every 6 hours + manual

---

## 3. Training Pipeline (`training.yml`)

### Trigger Modes

**Automatic (Daily):**
```yaml
schedule:
  - cron: '0 22 * * *'  # Daily at 22:00 UTC
```

**Manual (On-Demand):**
```bash
# Via GitHub Actions UI or CLI
gh workflow run training.yml -f force_retrain=true
```

### Pipeline Stages

#### Stage 1: Drift Check
- Compares current data distribution to baseline
- Decides if retraining is necessary
- Threshold: 15% Kullback-Leibler divergence
- Output: `drift_check.json`

#### Stage 2: Training
- Only runs if drift detected or `force_retrain=true`
- Downloads training data via `dvc pull` (optional)
- Runs `main.py train --config configs/training/final_train.yaml`
- Logs to MLflow with `MLFLOW_ENABLED=true`
- Timeout: **120 minutes** (2 hours)

#### Stage 3: Validation
- Checks model performance improvements
- Min SMAPE improvement: 0.02 (2 percentage points)
- Max training time: 3600 seconds (1 hour)
- Fails if metrics don't meet threshold

#### Stage 4: Registration
- Registers model in `experiments/registry`
- Creates manifest with artifacts reference
- Sets initial stage to `staging`
- Tags: `automated_training=true,drift_reason=<reason>`

#### Stage 5: Smoke Test
- Runs inference on sample data
- Validates model can be loaded and used
- Catches serialization/compatibility issues

### Output Artifacts

- `training-artifacts/`: Training logs and outputs (30-day retention)
- `drift-check-report/`: Drift analysis (7-day retention)
- MLflow run record: Metrics, params, artifacts

### Example Output

```json
{
  "run_id": "abc123def456",
  "training_duration_seconds": 2800,
  "smape": 0.28,
  "accuracy": 0.73,
  "error_rate": 0.001
}
```

---

## 4. Model Promotion Pipeline (`promote.yml`)

### Promotion Stages

```
staging → canary → production
```

### Pre-Promotion Checks

1. **Metric Validation**:
   - Accuracy ≥ 0.70
   - P95 Latency ≤ 2.0 seconds
   - Error rate ≤ 0.02 (2%)

2. **Production Eligibility** (for promotion to production):
   - Requires strict thresholds
   - Must pass all checks
   - Requires GitHub environment approval

### Manual Promotion Workflow

```bash
# Via GitHub Actions UI:
# 1. Go to Actions → Model Promotion
# 2. Click "Run workflow"
# 3. Enter:
#    - run_id: MLflow run ID (from training)
#    - target_stage: "staging" | "canary" | "production"
# 4. Review checks in workflow logs
# 5. (For production) Approve in environment protection
```

### Registry State Machine

```python
# After successful promotion
{
  "run_id": "abc123",
  "status": "canary",        # staging, canary, production, archived
  "created_utc": "2026-03-10T...",
  "updated_utc": "2026-03-10T...",
  "tracking": {
    "promoted_by": "github-actions",
    "promotion_url": "https://github.com/.../actions/runs/...",
    "target_stage": "canary"
  }
}
```

### Auto-Archiving

When promoting to production:
- Previous production: `status → archived`
- Current canary: `status → production`
- `active_production_run_id` updated

### Git Tagging

Promotion creates a git tag:
```
Tag: model/canary-abc123def456-20260310_221500
Ref: Promotion commit
```

For production releases:
```
Tag: v/abc123def456
Release: Model {run_id} → Production
```

---

## 5. Deployment Pipeline (`deploy.yml`)

### Deployment Strategies

#### Canary Deployment (10% traffic, 60 min monitoring)
1. Deploy new model to **canary environment** (10% of requests)
2. Monitor P95 latency, error rate for **60 seconds**
3. Auto-promote to 100% if metrics healthy
4. Auto-rollback if degradation detected

**Decision Logic:**
```python
if latency_p95 > 2.0s or error_rate > 5%:
    rollback()
else:
    promote_to_full()
```

#### Blue-Green Deployment (zero-downtime switch)
1. Deploy new model to **green environment**
2. Run full test suite against green
3. Switch **all traffic** to green
4. Keep blue available for instant rollback

#### Rolling Deployment (gradual rollout)
- 25% → 50% → 75% → 100% in stages
- Monitor between stages
- Rollback at any stage

### Pre-Deployment Checks

```
✓ Run integration tests
✓ Verify code coverage ≥ 80%
✓ Security scan (no vulnerabilities)
✓ Model inference on test data
✓ SLO thresholds validation
```

### Deployment Config

Create `experiments/registry/canary_config.json`:
```json
{
  "production_run_id": "<run_id>",
  "canary_enabled": true,
  "canary_percent": 10,
  "deployment_time": "2026-03-10T22:15:00Z",
  "deployed_by": "github-actions"
}
```

### Monitoring During Deployment

Scripts pull metrics from (or simulate):
- MLflow metrics
- Application logs (latency, errors)
- System metrics (CPU, memory)

### Rollback Procedure

**Automatic (on health check failure):**
```bash
python scripts/rollback_deployment.py \
  --to-previous-production \
  --reason "Canary metrics unhealthy"
```

**Manual:**
```bash
# Via GitHub Actions → Health Check workflow
# Click "Re-run jobs" after investigating
```

---

## 6. Health Check Pipeline (`health-check.yml`)

### Check Frequency

- **Scheduled**: Every 6 hours (0, 6, 12, 18 UTC)
- **Manual**: Anytime via `workflow_dispatch`
- **Post-Deployment**: Called by deploy workflow

### Health Check Categories

#### MLflow Connectivity
- Connect to tracking server
- List experiments
- Verify credentials

#### Production Model
- Load current production model
- Verify artifacts exist
- Check compatibility

#### Registry Integrity
- Validate index.json
- Confirm active production run exists
- Check state machine consistency

#### Inference Test
- Run sample prediction
- Verify latency < 2s
- Catch serialization issues

#### Data/Concept Drift
- Compare current data to baseline
- KL divergence threshold: 15%
- Alert if drift detected

#### Resource Usage
- Disk space: warn if > 80%
- Model cache: check for orphaned files
- Memory: monitor system health

### Escalation Path

**Health Status → Outcome:**
```
healthy   → Continue normal operations
degraded  → Create GitHub Issue (auto)
critical  → Auto-rollback + PagerDuty alert
```

### Alert Integration Points

```python
# Create issue if unhealthy
github.rest.issues.create({
    "title": "⚠️ Production System Health Alert",
    "labels": ["critical", "health-check"],
    "body": f"Issues: {health_failures}"
})

# Send to Slack (integration hook)
# Send to PagerDuty (if critical)
# Send to Email (team distribution)
```

---

## 7. Model Registry Structure

### Directory Layout

```
experiments/registry/
  ├── index.json                      # Model state machine
  ├── promotion_history.jsonl         # Line-delimited log
  ├── rollback_history.jsonl          # Rollback log
  ├── baseline_stats.json             # Data distribution baseline
  ├── canary_config.json              # Current canary settings
  ├── blue_green_config.json          # Blue-green state
  ├── production_tracker.json         # Production metrics
  └── deployment_manifest.json        # Deployment info
```

### index.json Schema

```json
{
  "runs": [
    {
      "run_id": "mlflow_run_id",
      "manifest_path": "experiments/models/mlflow_run_id/manifest.json",
      "stage": "training",
      "status": "production|staging|canary|archived",
      "tracking": {
        "promoted_by": "user|automation",
        "promotion_time": "2026-03-10T22:15:00Z",
        "training_duration_seconds": 2800,
        "smape": 0.28,
        "error_rate": 0.001
      },
      "created_utc": "2026-03-10T...",
      "updated_utc": "2026-03-10T..."
    }
  ],
  "active_production_run_id": "mlflow_run_id"
}
```

### Model Manifest

```json
{
  "run_id": "mlflow_run_id",
  "stage": "training",
  "registered_at": "2026-03-10T22:15:00Z",
  "artifacts": [
    {
      "path": "experiments/models/mlflow_run_id/model.pkl",
      "type": "sklearn_model",
      "framework": "scikit-learn"
    },
    {
      "path": "data/processed/features.joblib",
      "type": "feature_transformer"
    }
  ]
}
```

---

## 8. Observability & Alerting

### Metrics to Track

**Model Metrics:**
- Accuracy, precision, recall, F1
- SMAPE (mean absolute percentage error)
- Prediction latency (p50, p95, p99)
- Error rate / exception rate

**System Metrics:**
- API response time
- Throughput (requests/second)
- CPU, memory, disk usage
- Model cache hit rate

**Business Metrics:**
- Model freshness (days since training)
- Retraining frequency
- Deployment success rate
- Rollback frequency

### Alert Thresholds

```yaml
accuracy:
  warning: < 0.70
  critical: < 0.65

latency_p95:
  warning: > 2.0s
  critical: > 5.0s

error_rate:
  warning: > 0.02
  critical: > 0.05

drift_magnitude:
  warning: > 0.10
  critical: > 0.25
```

### Integration Points

```
GitHub Actions → Create Issue
             → Email (SMTP)
             → Slack (webhook)
             → PagerDuty (incident)
             → DataDog (APM)
             → CloudWatch (AWS)
             → Prometheus (open)
```

---

## 9. Rollback Procedures

### Automatic Rollback (15-minute rule)

```
if health_check_fails():
    if time_since_last_deployment < 15_minutes:
        auto_rollback_to_previous()
        alert(severity="critical")
```

### Manual Rollback

```bash
# Via GitHub Actions UI:
# 1. Go to Actions → Health Check
# 2. Check failure details
# 3. Run promote.yml manually with previous run_id

# OR via CLI:
python scripts/rollback_deployment.py \
  --to-previous-production \
  --reason "Manual: P95 latency spike detected"
```

### Rollback Record

```json
{
  "timestamp": "2026-03-10T22:30:00Z",
  "rolled_back_to": "previous_run_id",
  "reason": "Canary metrics unhealthy (error_rate=0.08)",
  "triggered_by": "automated_health_check",
  "recovered_at": "2026-03-10T23:00:00Z"
}
```

---

## 10. Cost Optimization

### Data Transfer Optimization
- Use DVC cache to avoid re-downloading
- Batch training data pulls
- Compress artifacts (gzip for JSON, ONNX for models)

### Compute Optimization
- **Training**: 120 min timeout, auto-fail if over budget
- **CI/CD**: Parallel jobs, share cache
- **Monitoring**: 6-hour cadence, not continuous

### Storage Optimization
- Archive old models (> 6 months)
- Clean DVC cache (`dvc gc --workspace`)
- Retention policies:
  - Training artifacts: 30 days
  - Drift reports: 7 days
  - Health reports: 90 days
  - Rollback history: permanent

### Example Cost Breakdown (AWS)

```
Training (1x per day, 2 hrs GPU): ~$2-5/day
Inference (small model, P95 <2s): ~$200-500/month
Storage (models + data): ~$10-50/month
DVC cache (S3): ~$20/month
Total: ~$300-700/month for moderate scale
```

---

## 11. Secrets Rotation Best Practices

### Rotation Schedule
- **MLflow tokens**: Quarterly (90 days)
- **AWS keys**: Semi-annually (180 days)
- **DagsHub tokens**: Quarterly (90 days)

### Rotation Procedure
1. Generate new credential
2. Update GitHub Secret
3. Test credentials in CI
4. Wait 24 hours for propagation
5. Revoke old credential

### Monitoring
```bash
# Check secret usage in runs
gh api repos/{owner}/{repo}/actions/secrets

# Audit trail via GitHub activity
gh api repos/{owner}/{repo}/activity
```

---

## 12. Disaster Recovery

### Backup Strategy

```
Registry Index (experiments/registry/index.json)
  → Committed to Git (version control)
  
Model Artifacts (experiments/models/{run_id}/)
  → DVC tracked (remote S3)
  
Training Data (data/raw/)
  → DVC tracked (remote S3)
```

### Recovery Scenarios

**Lost Model Artifacts:**
```bash
dvc checkout experiments/models/  # Restore from cache
dvc pull -r origin               # Or from remote
```

**Corrupted Registry:**
```bash
git checkout HEAD -- experiments/registry/index.json
git reset --hard HEAD
```

**Complete Failure:**
```bash
# Restore from last known good state
git tag | grep "model/production"  # Find last production tag
git checkout <tag>                  # Restore to that point
```

---

## 13. Quick Start Checklist

- [ ] Set all GitHub Secrets (MLFLOW*, AWS*, DAGSHUB*)
- [ ] Enable branch protection on `main`
- [ ] Create `.env` file locally (for development)
- [ ] Test training workflow on schedule
- [ ] Promote test model to staging
- [ ] Deploy canary and monitor metrics
- [ ] Verify health check runs and passes
- [ ] Test manual rollback procedure
- [ ] Document SLOs in runbook
- [ ] Set up Slack/PagerDuty integrations

---

## 14. Troubleshooting

### Training Fails Due to Drift Check Timeout

```bash
# Force retrain without drift check
gh workflow run training.yml -f force_retrain=true
```

### Promotion Blocked by Validation

```bash
# Check MLflow run metrics
mlflow ui  # Open at http://localhost:5000
```

### Deployment Canary Takes Too Long

Check `monitor_canary_metrics.py` duration setting (default 60s).

### Health Check Escalates to Auto-Rollback

1. Check CRITICAL issues in health report artifact
2. Investigate production metrics (latency, errors)
3. Review recent deployment changes
4. Manual rollback to known-good state

---

## 15. References

- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [GitHub Actions Workflows](https://docs.github.com/en/actions/using-workflows)
- [Blue-Green Deployments](https://martinfowler.com/bliki/BlueGreenDeployment.html)
- [Canary Releases](https://martinfowler.com/bliki/CanaryRelease.html)
