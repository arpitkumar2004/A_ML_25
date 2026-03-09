# GitHub Secrets Configuration Guide

## Overview

All sensitive credentials must be stored as **GitHub Secrets**, not in code or `.env` files checked into Git.

This guide walks through configuring all required secrets for Phase 3 CI/CD automation.

---

## Step 1: Access Repository Settings

1. Go to your repository: `https://github.com/<owner>/A_ML_25`
2. Click **Settings** (top right)
3. In left sidebar, click **Secrets and variables** → **Actions**

---

## Step 2: Add Each Secret

### MLflow Credentials

#### `MLFLOW_TRACKING_URI`
**Value**: Your MLflow tracking server URL
```
https://dagshub.com/<your-username>/<repo-name>.mlflow
```
**Example**: `https://dagshub.com/arpitkumar2004/A_ML_25.mlflow`

#### `MLFLOW_TRACKING_USERNAME`
**Value**: Your DagsHub username
```
arpitkumar2004
```

#### `MLFLOW_TRACKING_PASSWORD`
**Value**: Your DagsHub personal access token or password
```
8d2b2de07882ad3a0941a153762292bec82023e6
```

### DagsHub Storage Credentials (Optional, for DVC)

#### `DAGSHUB_USERNAME`
**Value**: Your DagsHub username
```
arpitkumar2004
```

#### `DAGSHUB_TOKEN`
**Value**: Your DagsHub personal access token
```
8d2b2de07882ad3a0941a153762292bec82023e6
```

### AWS S3 Credentials

#### `AWS_ACCESS_KEY_ID`
**Value**: AWS Access Key ID (or DagsHub bucket public key)
```
6efef6351287803cd8a277bad6e32ac43cf9e657
```

#### `AWS_SECRET_ACCESS_KEY`
**Value**: AWS Secret Access Key (or DagsHub bucket server token)
```
8d2b2de07882ad3a0941a153762292bec82023e6
```

---

## Step 3: UI Instructions

### Method 1: Using GitHub Web UI

1. Click **New repository secret** button
2. Enter **Name**: `MLFLOW_TRACKING_URI`
3. Enter **Secret**: `https://dagshub.com/arpitkumar2004/A_ML_25.mlflow`
4. Click **Add secret**

Repeat for each of the 8 secrets above.

### Method 2: Using GitHub CLI

```bash
# Install GitHub CLI if not already installed
# https://cli.github.com/

# Login
gh auth login

# Add secrets (run from repo directory)
gh secret set MLFLOW_TRACKING_URI -b "https://dagshub.com/arpitkumar2004/A_ML_25.mlflow"
gh secret set MLFLOW_TRACKING_USERNAME -b "arpitkumar2004"
gh secret set MLFLOW_TRACKING_PASSWORD -b "8d2b2de07882ad3a0941a153762292bec82023e6"
gh secret set DAGSHUB_USERNAME -b "arpitkumar2004"
gh secret set DAGSHUB_TOKEN -b "8d2b2de07882ad3a0941a153762292bec82023e6"
gh secret set AWS_ACCESS_KEY_ID -b "6efef6351287803cd8a277bad6e32ac43cf9e657"
gh secret set AWS_SECRET_ACCESS_KEY -b "8d2b2de07882ad3a0941a153762292bec82023e6"
```

### Method 3: Using PowerShell Script

Save as `setup_secrets.ps1`:

```powershell
param(
    [Parameter(Mandatory=$true)]
    [string]$Owner,
    [Parameter(Mandatory=$true)]
    [string]$Repo,
    [hashtable]$Secrets
)

foreach ($name in $Secrets.Keys) {
    $value = $Secrets[$name]
    gh secret set $name -b $value --repo "$Owner/$Repo"
    Write-Host "✓ Set secret: $name"
}

```

Run:
```powershell
$secrets = @{
    "MLFLOW_TRACKING_URI" = "https://dagshub.com/arpitkumar2004/A_ML_25.mlflow"
    "MLFLOW_TRACKING_USERNAME" = "arpitkumar2004"
    "MLFLOW_TRACKING_PASSWORD" = "your-token"
    "DAGSHUB_USERNAME" = "arpitkumar2004"
    "DAGSHUB_TOKEN" = "your-token"
    "AWS_ACCESS_KEY_ID" = "your-key-id"
    "AWS_SECRET_ACCESS_KEY" = "your-secret-key"
}

./setup_secrets.ps1 -Owner "arpitkumar2004" -Repo "A_ML_25" -Secrets $secrets
```

---

## Step 4: Verify Secrets

### Check Secrets List
```bash
gh secret list --repo arpitkumar2004/A_ML_25
```

Output:
```
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
DAGSHUB_USERNAME
DAGSHUB_TOKEN
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

### Verify Secret in Workflow
Add test step to CI workflow:
```yaml
- name: Verify secrets
  run: |
    echo "MLFLOW_TRACKING_URI=${{ secrets.MLFLOW_TRACKING_URI }}" | head -c 50
    echo "..."
```

---

## Step 5: Security Best Practices

### ✅ DO

- [ ] Use personal access tokens (not passwords)
- [ ] Rotate tokens every 90 days
- [ ] Use minimal scopes (only needed permissions)
- [ ] Store old tokens in secure password manager
- [ ] Audit secret usage in workflows
- [ ] Use separate tokens for dev and prod

### ❌ DON'T

- [ ] Copy secrets into Slack/Email/Teams
- [ ] Print secrets in workflow logs
- [ ] Commit secrets to Git (even in .env)
- [ ] Share credentials via unencrypted channels
- [ ] Use same token for multiple services
- [ ] Commit .env file with real values

---

## Step 6: Token Rotation

### Check Current Tokens

**DagsHub:**
1. Go to `https://dagshub.com`
2. Click Profile → Settings → Tokens
3. Note expiration dates

**AWS (if using):**
1. Go to AWS Console → IAM
2. Users → Security credentials
3. Check last rotated date

### Rotate a Token

1. Generate new token on service
2. Update GitHub Secret with new value
3. Test in CI (run a workflow)
4. Wait 24 hours for propagation
5. Revoke old token

**Example: Rotate DagsHub Token**

```bash
# 1. Generate new token on https://dagshub.com
#    (Settings → Tokens → Generate)

# 2. Copy new token

# 3. Update GitHub secret
gh secret set DAGSHUB_TOKEN -b "new-token-value"

# 4. Verify in workflow logs
gh workflow run ci.yml

# 5. Revoke old token on DagsHub
```

---

## Step 7: Troubleshooting

### Secret Not Found Error

```
Error: Resource not found when requesting
https://api.github.com/repos/arpitkumar2004/A_ML_25/actions/secrets/MLFLOW_TRACKING_URI
```

**Solution:**
- Ensure secret is created (check in Settings → Secrets)
- Check spelling (case-sensitive)
- Wait a few seconds for propagation
- Try accessing from clean terminal

### Authentication Failures in Workflows

```yaml
Error: 401 Unauthorized connecting to MLflow
```

**Solution:**
1. Check secret value is correct
2. Verify credentials haven't expired
3. Test credentials locally first:
   ```bash
   mlflow ui --backend-store-uri "your-uri" --username "your-user" --password "your-pass"
   ```

### Workflows Access Denied

```
Error: run_id invalid or not found
```

**Solution:**
- Your token might have insufficient scope
- Generate new token with broader permissions
- Test with the new token

---

## Step 8: Local Development Setup

### Create Local .env File

```bash
# Copy example
cp .env.example .env

# Edit with your secrets (never commit this)
# Use same values as GitHub Secrets
```

### .env Template

```bash
# MLflow
MLFLOW_TRACKING_URI=https://dagshub.com/arpitkumar2004/A_ML_25.mlflow
MLFLOW_TRACKING_USERNAME=arpitkumar2004
MLFLOW_TRACKING_PASSWORD=your-token

# DagsHub
DAGSHUB_USERNAME=arpitkumar2004
DAGSHUB_TOKEN=your-token

# AWS/DagsHub Storage
AWS_ACCESS_KEY_ID=your-key-id
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1

# Application
MLFLOW_ENABLED=true
```

### Use in Local Scripts

```python
import os
from dotenv import load_dotenv

load_dotenv('.env')

mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow_user = os.getenv('MLFLOW_TRACKING_USERNAME')
mlflow_pwd = os.getenv('MLFLOW_TRACKING_PASSWORD')
```

---

## Reference: Secret Usage in Workflows

### Current Workflows Use

```yaml
# Training Workflow
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
  MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}

# Deployment Workflow
- name: Deploy
  run: python scripts/deploy.py
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

### Adding New Secrets

If you add new integrations, follow the same pattern:

```yaml
new_integration:
  runs-on: ubuntu-latest
  steps:
    - name: Use new secret
      env:
        MY_NEW_SECRET: ${{ secrets.MY_NEW_SECRET }}
      run: |
        # Secret available as environment variable
        python scripts/use_secret.py
```

---

## Quick Reference

| Secret | Where to Get | Scope | Rotate |
|--------|-------------|-------|--------|
| MLFLOW_TRACKING_URI | DagsHub | Profile | N/A |
| MLFLOW_TRACKING_USERNAME | DagsHub | Profile | N/A |
| MLFLOW_TRACKING_PASSWORD | DagsHub → Tokens | MLflow | 90 days |
| DAGSHUB_USERNAME | DagsHub | Profile | N/A |
| DAGSHUB_TOKEN | DagsHub → Tokens | API | 90 days |
| AWS_ACCESS_KEY_ID | AWS IAM / DagsHub | S3 Upload | 180 days |
| AWS_SECRET_ACCESS_KEY | AWS IAM / DagsHub | S3 Upload | 180 days |

---

## Getting Help

- GitHub Secrets docs: https://docs.github.com/en/actions/security-guides/encrypted-secrets
- DagsHub tokens: https://dagshub.com/docs/getting-started/token
- AWS Access Keys: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html
