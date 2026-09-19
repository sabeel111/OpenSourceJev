$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating the Jev virtual environment..." -ForegroundColor Cyan
    py -3 -m venv (Join-Path $projectRoot ".venv")
}

Write-Host "Checking Python dependencies..." -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt") --disable-pip-version-check

Write-Host "Starting Jev at http://127.0.0.1:8000" -ForegroundColor Green
$server = Start-Process -FilePath $venvPython `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $projectRoot -PassThru -WindowStyle Hidden

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 1 -UseBasicParsing | Out-Null
            $ready = $true
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ready) {
        throw "Jev did not become ready within 30 seconds. Check that port 8000 is available."
    }

    Start-Process "http://127.0.0.1:8000"
    Wait-Process -Id $server.Id
} finally {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
