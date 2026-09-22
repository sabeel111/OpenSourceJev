# OpenSourceJev DOOM Runner
# Run from project root: .\doom_agent\run_doom.ps1 -Scenario basic -Profile accuracy
# Run from doom_agent:   .\run_doom.ps1 -Scenario basic -Profile accuracy
# Run with Qwen 3.5 4B:  .\doom_agent\run_doom.ps1 -Profile accuracy
# Run with Qwen 3 1.7B:  .\doom_agent\run_doom.ps1 -Profile fast
# Run headless:          .\doom_agent\run_doom.ps1 -NoWindow

param(
    [string]$Scenario = "defend_the_center",
    [string]$Profile = "accuracy",
    [string]$Model = "",
    [switch]$NoWindow,
    [int]$Episodes = 1
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python.exe"
}

$DoomAgentScript = Join-Path $ScriptDir "doom_agent.py"
$env:PYTHONPATH = $ProjectRoot

$windowFlag = ""
if ($NoWindow) {
    $windowFlag = "--no-window"
}

$modelArgs = @()
if ($Model) {
    $modelArgs += @("--model", $Model)
}

Write-Host "Launching OpenSourceJev DOOM Slayer Agent ($Scenario, Profile: $Profile, $Episodes episode)..." -ForegroundColor Green
& $PythonExe $DoomAgentScript --scenario $Scenario --profile $Profile @modelArgs $windowFlag --episodes $Episodes

if (-not $NoWindow) {
    Write-Host "`nMatch completed! Press Enter to exit..." -ForegroundColor Cyan
    Read-Host
}
