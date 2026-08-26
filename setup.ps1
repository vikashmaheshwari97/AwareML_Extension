$ErrorActionPreference = "Stop"

Write-Host "AwareML Extension: Python 3.8 reproducible MAIN setup" -ForegroundColor Cyan
& py -3.8 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.8 is not installed. Install CPython 3.8.10 (64-bit)." }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.8 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv." }
}

$mainPython = ".\.venv\Scripts\python.exe"
& $mainPython -m pip install --upgrade "pip<25" "setuptools<76" wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade MAIN packaging tools." }

& $mainPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "MAIN dependency installation failed." }

Write-Host "Installing optional research/native MAIN backends..." -ForegroundColor DarkCyan
& $mainPython -m pip install -r requirements-optional.txt
if ($LASTEXITCODE -ne 0) { throw "Optional MAIN dependency installation failed." }

& $mainPython -c "import sys,river,pandas,numpy,sklearn,streamlit,codecarbon,shap,xgboost,flaml,vowpalwabbit; from lime.lime_tabular import LimeTabularExplainer; from flaml.onlineml.autovw import AutoVW; print('Python:',sys.version.split()[0]); print('River:',river.__version__); print('Pandas:',pandas.__version__); print('NumPy:',numpy.__version__); print('sklearn:',sklearn.__version__); print('Streamlit:',streamlit.__version__); print('MAIN ENVIRONMENT: READY')"
if ($LASTEXITCODE -ne 0) { throw "MAIN environment verification failed." }

Write-Host "MAIN AwareML environment is READY." -ForegroundColor Green
Write-Host "Next: .\setup_oaml.ps1 and .\setup_evo.ps1" -ForegroundColor Yellow
