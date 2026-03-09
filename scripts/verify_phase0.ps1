param(
    [string]$PythonExe = ".venv/Scripts/python.exe",
    [string]$HashSeed = "42",
    [string[]]$ManifestGlobs = @(
        "experiments/registry/*_manifest.json",
        "experiments/smoke_check/registry/*_manifest.json"
    ),
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

Write-Output "[Phase0] Starting verification run"

Assert-True (Test-Path $PythonExe) "Python executable not found at '$PythonExe'"
$env:PYTHONPATH = "."
$env:PYTHONHASHSEED = $HashSeed
Write-Output "[Phase0] Using PYTHONHASHSEED=$($env:PYTHONHASHSEED)"

Write-Output "[Phase0] Step 1/6: Validate configs"
& $PythonExe scripts/validate_configs.py

if (-not $SkipSmoke) {
    Write-Output "[Phase0] Step 2/6: Run smoke train+predict"
    & $PythonExe scripts/smoke_train_predict.py
} else {
    Write-Output "[Phase0] Step 2/6: Skip smoke train+predict (requested)"
}

Write-Output "[Phase0] Step 3/6: Verify quality gate keys in configs"
$trainCfg = Get-Content "configs/training/final_train.yaml" -Raw
Assert-True ($trainCfg -match "quality_gates:") "quality_gates block missing in configs/training/final_train.yaml"
Assert-True ($trainCfg -match "max_smape:") "max_smape missing in configs/training/final_train.yaml"
Assert-True ($trainCfg -match "max_p95_latency_seconds:") "max_p95_latency_seconds missing in configs/training/final_train.yaml"
Assert-True ($trainCfg -match "max_critical_drift_features:") "max_critical_drift_features missing in configs/training/final_train.yaml"

$policyCfg = Get-Content "configs/monitoring/retrain_policy.yaml" -Raw
Assert-True ($policyCfg -match "quality_gates:") "quality_gates block missing in configs/monitoring/retrain_policy.yaml"

Write-Output "[Phase0] Step 4/6: Verify manifest reproducibility fields"
$manifestCandidates = @()
foreach ($glob in $ManifestGlobs) {
    $manifestCandidates += Get-ChildItem $glob -ErrorAction SilentlyContinue
}

$trainManifest = $manifestCandidates |
    Where-Object { $_.Name -like "*_train_manifest.json" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$manifest = $trainManifest
if (-not $manifest) {
    $manifest = $manifestCandidates |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

Assert-True ([bool]$manifest) ("No manifest found. Checked: " + ($ManifestGlobs -join ", "))

$manifestObj = Get-Content $manifest.FullName -Raw | ConvertFrom-Json
Assert-True ([bool]$manifestObj.reproducibility) "reproducibility block missing in manifest"
Assert-True (-not [string]::IsNullOrWhiteSpace($manifestObj.reproducibility.config_sha256)) "config_sha256 missing in manifest.reproducibility"

# Seed is mandatory for training manifests and optional for non-training manifests.
if ([string]$manifestObj.stage -eq "train") {
    $seedVal = [string]$manifestObj.reproducibility.seed
    Assert-True (-not [string]::IsNullOrWhiteSpace($seedVal)) "seed missing in manifest.reproducibility"
}
Write-Output "[Phase0] Latest manifest: $($manifest.FullName)"
Write-Output "[Phase0] config_sha256: $($manifestObj.reproducibility.config_sha256)"
Write-Output "[Phase0] seed: $($manifestObj.reproducibility.seed)"

Write-Output "[Phase0] Step 5/6: Retraining policy dry-run check"
& $PythonExe scripts/retrain_orchestrator.py --policy configs/monitoring/retrain_policy.yaml

Write-Output "[Phase0] Step 6/6: Export dependency lock file"
& powershell -ExecutionPolicy Bypass -File scripts/export_requirements_lock.ps1 -PythonExe $PythonExe -OutFile "requirements-lock.txt"
Assert-True (Test-Path "requirements-lock.txt") "requirements-lock.txt was not generated"

Write-Output "[Phase0] VERIFICATION PASSED"
