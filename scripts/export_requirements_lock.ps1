param(
    [string]$PythonExe = ".venv/Scripts/python.exe",
    [string]$OutFile = "requirements-lock.txt"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at '$PythonExe'. Activate venv or pass -PythonExe."
}

Write-Output "Using Python: $PythonExe"
& $PythonExe -m pip freeze | Out-File -FilePath $OutFile -Encoding ascii
Write-Output "Wrote lock file: $OutFile"
