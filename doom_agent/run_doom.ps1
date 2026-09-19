# Jev DOOM Runner
# Run: .\run_doom.ps1
# Run quick target practice: .\run_doom.ps1 -Scenario basic
# Run headless: .\run_doom.ps1 -NoWindow

param(
    [string]$Scenario = "defend_the_center",
    [switch]$NoWindow,
    [int]$Episodes = 1
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = ".."

$windowFlag = ""
if ($NoWindow) {
    $windowFlag = "--no-window"
}

Write-Host "Launching Jev DOOM Slayer Agent ($Scenario, $Episodes episode)..." -ForegroundColor Green
& "..\.venv\Scripts\python.exe" "doom_agent.py" --scenario $Scenario $windowFlag --episodes $Episodes

if (-not $NoWindow) {
    Write-Host "`nMatch completed! Press Enter to exit..." -ForegroundColor Cyan
    Read-Host
}
