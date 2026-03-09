param(
    [string]$EnvFile = ".env",
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$DvcArgs
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EnvFile)) {
    throw "Env file '$EnvFile' not found. Create it from .env.example first."
}

function Set-EnvFromFile {
    param([string]$Path)

    $lines = Get-Content $Path
    foreach ($line in $lines) {
        $trim = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trim)) { continue }
        if ($trim.StartsWith("#")) { continue }

        $idx = $trim.IndexOf("=")
        if ($idx -lt 1) { continue }

        $key = $trim.Substring(0, $idx).Trim()
        $value = $trim.Substring($idx + 1).Trim()

        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

Set-EnvFromFile -Path $EnvFile

# Map DagsHub credentials to AWS env vars expected by DVC S3 backend.
if (-not $env:AWS_ACCESS_KEY_ID -and $env:DAGSHUB_USERNAME) {
    $env:AWS_ACCESS_KEY_ID = $env:DAGSHUB_USERNAME
}

if (-not $env:AWS_SECRET_ACCESS_KEY -and $env:DAGSHUB_TOKEN) {
    $env:AWS_SECRET_ACCESS_KEY = $env:DAGSHUB_TOKEN
}

# Fallback: many users keep DagsHub creds in MLflow vars.
if (-not $env:AWS_ACCESS_KEY_ID -and $env:MLFLOW_TRACKING_USERNAME) {
    $env:AWS_ACCESS_KEY_ID = $env:MLFLOW_TRACKING_USERNAME
}

if (-not $env:AWS_SECRET_ACCESS_KEY -and $env:MLFLOW_TRACKING_PASSWORD) {
    $env:AWS_SECRET_ACCESS_KEY = $env:MLFLOW_TRACKING_PASSWORD
}

if (-not $env:AWS_DEFAULT_REGION) {
    $env:AWS_DEFAULT_REGION = "us-east-1"
}

if (-not $DvcArgs -or $DvcArgs.Count -eq 0) {
    throw "No DVC arguments provided. Example: ./scripts/dvc_with_env.ps1 push -r origin --all-commits"
}

function Requires-RemoteCredentials {
    param([string[]]$Args)

    if (-not $Args -or $Args.Count -eq 0) { return $false }

    $cmd = $Args[0].ToLowerInvariant()
    return @("push", "pull", "fetch") -contains $cmd
}

if (Requires-RemoteCredentials -Args $DvcArgs) {
    if (-not $env:AWS_ACCESS_KEY_ID -or -not $env:AWS_SECRET_ACCESS_KEY) {
        throw (
            "Missing DVC remote credentials. Set DAGSHUB_USERNAME + DAGSHUB_TOKEN " +
            "(or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY) in .env."
        )
    }
}

$pythonExe = "c:/Users/RDRL/Desktop/A_ML_25/.venv/Scripts/python.exe"
if (Test-Path $pythonExe) {
    & $pythonExe -m dvc @DvcArgs
} else {
    & dvc @DvcArgs
}
