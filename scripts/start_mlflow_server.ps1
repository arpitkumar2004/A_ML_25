param(
    [string]$BackendStoreUri = "sqlite:///mlflow.db",
    [string]$ArtifactRoot = "./experiments/mlflow_artifacts",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 5000,
    [string]$PythonExe = ".venv/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at '$PythonExe'."
}

$env:MLFLOW_TRACKING_URI = "http://$BindHost`:$Port"
Write-Output "MLFLOW_TRACKING_URI=$env:MLFLOW_TRACKING_URI"

& $PythonExe -m mlflow server --backend-store-uri $BackendStoreUri --default-artifact-root $ArtifactRoot --host $BindHost --port $Port
