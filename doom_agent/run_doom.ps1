# OpenSourceJev DOOM Runner
# Run with Qwen 3.5 4B: .\run_doom.ps1 -Profile accuracy
# Run with Qwen 3 1.7B: .\run_doom.ps1 -Profile fast
# Run quick target practice: .\run_doom.ps1 -Scenario basic -Profile accuracy
# Run headless: .\run_doom.ps1 -NoWindow

param(
    [string]$Scenario = "defend_the_center",
    [string]$Profile = "accuracy",
    [string]$Model = "",
    [switch]$NoWindow,
    [int]$Episodes = 1
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = ".."

$windowFlag = ""
if ($NoWindow) {
    $windowFlag = "--no-window"
}

$modelArgs = @()
if ($Model) {
    $modelArgs += @("--model", $Model)
}

Write-Host "Launching OpenSourceJev DOOM Slayer Agent ($Scenario, Profile: $Profile, $Episodes episode)..." -ForegroundColor Green
& "..\.venv\Scripts\python.exe" "doom_agent.py" --scenario $Scenario --profile $Profile @modelArgs $windowFlag --episodes $Episodes

if (-not $NoWindow) {
    Write-Host "`nMatch completed! Press Enter to exit..." -ForegroundColor Cyan
    Read-Host
}
