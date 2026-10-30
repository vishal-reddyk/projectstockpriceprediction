$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Host "Create a Python 3.11 venv first: py -3.11 -m venv .venv"
    exit 1
}
.\.venv\Scripts\python.exe -m src.ingest
.\.venv\Scripts\python.exe -m src.preprocess
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.evaluate
Write-Host "Pipeline complete. Start API with: .\.venv\Scripts\uvicorn.exe src.api:app --port 8000"
