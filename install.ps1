#Requires -Version 5.1
<#
.SYNOPSIS
    Install the EPR PC application into a local virtual environment.

.PARAMETER Ml
    Include the recognition stack (PyTorch from PyPI). Works without an NVIDIA GPU.

.PARAMETER Cuda
    Include the CUDA 12.6 PyTorch wheels instead of the generic extra.
#>
param(
    [switch]$Ml,
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Creating virtual environment (.venv)"
$venv = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -eq 0) { $venv = $true }
    if (-not $venv) {
        py -3 -m venv .venv
        if ($LASTEXITCODE -eq 0) { $venv = $true }
    }
}
if (-not $venv) {
    python -m venv .venv
}
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Python 3.11 or newer is required. Install it from https://www.python.org/downloads/ and retry."
}

$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($Cuda) {
    & $python -m pip install -e ".[dev]"
    & $python -m pip install -r "pc\requirements-ml.txt"
} elseif ($Ml) {
    & $python -m pip install -e ".[dev,ml]"
} else {
    & $python -m pip install -e ".[dev]"
}

Write-Host ""
& ".\.venv\Scripts\epr.exe" --help
Write-Host ""
Write-Host "Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run:       epr acquire"
if (-not $Ml -and -not $Cuda) {
    Write-Host "Recognition (CNN/LNN) needs:  .\install.ps1 -Ml   or   .\install.ps1 -Cuda"
}
