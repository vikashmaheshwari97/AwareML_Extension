$ErrorActionPreference = "Stop"

function Require-File($path, $message) {
    if (-not (Test-Path $path)) {
        Write-Host $message -ForegroundColor Red
        exit 1
    }
}

Require-File ".venv\Scripts\python.exe" "MAIN environment is missing. Run .\setup.ps1 first."
Require-File ".venv-oaml\Scripts\python.exe" "OAML environment is missing. Run .\setup_oaml.ps1 first."
Require-File ".venv-evo\Scripts\python.exe" "EvoAutoML environment is missing. Run .\setup_evo.ps1 first."

$env:AWAREML_OAML_PYTHON = (Resolve-Path ".venv-oaml\Scripts\python.exe").Path
$env:AWAREML_EVO_PYTHON = (Resolve-Path ".venv-evo\Scripts\python.exe").Path

Write-Host "AwareML Extension" -ForegroundColor Cyan
Write-Host "  Main: $((Resolve-Path '.venv\Scripts\python.exe').Path)" -ForegroundColor DarkGray
Write-Host "  OAML: $env:AWAREML_OAML_PYTHON" -ForegroundColor DarkGray
Write-Host "  Evo : $env:AWAREML_EVO_PYTHON" -ForegroundColor DarkGray
Write-Host ""

& .\.venv\Scripts\python.exe -m streamlit run app.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
