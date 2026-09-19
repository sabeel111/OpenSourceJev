$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeDir = Join-Path $projectRoot "runtime\llama.cpp\bin"
$serverPath = Join-Path $runtimeDir "llama-server.exe"

if (-not (Test-Path $serverPath)) {
    throw "Prebuilt llama.cpp was not found at $serverPath. Download the official Windows CUDA 12.4 release and extract it there."
}

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidia) {
    throw "nvidia-smi was not found. Install an NVIDIA driver before enabling CUDA mode."
}

$gpu = & $nvidia.Source --query-gpu=name,memory.total,compute_cap --format=csv,noheader
$version = & $serverPath --version

Write-Host "Prebuilt llama.cpp runtime is ready." -ForegroundColor Green
$version
Write-Host "Detected NVIDIA GPU:" -ForegroundColor Cyan
$gpu
Write-Host "The native adapter reads logits directly from llama.dll; it does not use Ollama or llama-cpp-python." -ForegroundColor Cyan
