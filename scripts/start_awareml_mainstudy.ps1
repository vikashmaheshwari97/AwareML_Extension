$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
Write-Host "Starting AwareML Main Study..." -ForegroundColor Green
python -m streamlit run app.py
