param(
    [ValidateSet("slides", "animatediff")]
    [string]$VideoGenerator
)

$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith('#')) {
            return
        }
        if ($line -match '^\s*([^=]+)=(.*)$') {
            $key = $Matches[1].Trim()
            $value = $Matches[2]
            if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
                $value = $value.Substring(1, $value.Length - 2)
            } elseif ($value.StartsWith("'") -and $value.EndsWith("'") -and $value.Length -ge 2) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            [Environment]::SetEnvironmentVariable($key, $value)
        }
    }
    Write-Host "[env] Loaded variables from $Path"
}

Import-DotEnv (Join-Path $PSScriptRoot '.env')

if ($VideoGenerator) {
    $env:VIDEO_GENERATOR = $VideoGenerator
} elseif (-not $env:VIDEO_GENERATOR) {
    $env:VIDEO_GENERATOR = 'slides'
}
Write-Host "[env] VIDEO_GENERATOR = $($env:VIDEO_GENERATOR)"

if ($env:VIDEO_GENERATOR -eq 'animatediff' -and -not $env:HUGGINGFACE_TOKEN) {
    Write-Warning 'VIDEO_GENERATOR=animatediff but HUGGINGFACE_TOKEN is missing; private models may fail to download.'
}

$venvPath = Join-Path $PSScriptRoot '.venv'
if (-not (Test-Path $venvPath)) {
    Write-Host '[setup] Creating virtual environment'
    python -m venv $venvPath
}

$activateScript = Join-Path $venvPath 'Scripts\\Activate.ps1'
if (-not (Test-Path $activateScript)) {
    throw "Could not find activate script at $activateScript"
}

Write-Host '[setup] Activating virtual environment'
. $activateScript

pip install --upgrade pip
pip install -r requirements.txt

Write-Host '[run] Starting FastAPI with uvicorn'
uvicorn main:app --reload --host 127.0.0.1 --port 8000
