$ErrorActionPreference = "Stop"

$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[setup] Creating virtual environment"
    python -m venv $venvPath
}

$activateScript = Join-Path $venvPath "Scripts\\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "Could not find activate script at $activateScript"
}

Write-Host "[setup] Activating virtual environment"
. $activateScript

pip install --upgrade pip
pip install -r requirements.txt

Write-Host "[run] Starting FastAPI with uvicorn"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
