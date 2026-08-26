$ErrorActionPreference = "Stop"

Write-Host "AwareML Extension: isolated native OAML environment" -ForegroundColor Cyan
& py -3.8 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.8 is not installed. Install CPython 3.8.10 (64-bit)." }

if (-not (Test-Path ".venv-oaml\Scripts\python.exe")) {
    & py -3.8 -m venv .venv-oaml
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv-oaml." }
}

$oamlPython = ".\.venv-oaml\Scripts\python.exe"
& $oamlPython -m pip install --upgrade "pip<25" "setuptools<76" wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade OAML packaging tools." }

& $oamlPython -m pip install -r requirements-oaml.txt
if ($LASTEXITCODE -ne 0) { throw "OAML dependency installation failed." }

& $oamlPython -c "import sys,river,pandas,numpy,scipy,sklearn,psutil,gama; from gama import GamaClassifier; print('Python:',sys.version.split()[0]); print('River:',river.__version__); print('Pandas:',pandas.__version__); print('NumPy:',numpy.__version__); print('SciPy:',scipy.__version__); print('sklearn:',sklearn.__version__); print('psutil:',psutil.__version__); print('GAMA: OK')"
if ($LASTEXITCODE -ne 0) { throw "OAML/GAMA verification failed." }

Write-Host "Native OAML environment is READY." -ForegroundColor Green
