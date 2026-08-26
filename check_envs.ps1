$ErrorActionPreference = "Stop"

function Check-Env($label, $python, $code) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    if (-not (Test-Path $python)) {
        Write-Host "$python missing" -ForegroundColor Red
        return
    }
    & $python -c $code
    if ($LASTEXITCODE -ne 0) { throw "$label verification failed." }
    Write-Host ""
}

Check-Env "MAIN ENVIRONMENT" ".venv\Scripts\python.exe" "import sys,river,numpy,pandas,scipy,sklearn,psutil; print('Python:',sys.version.split()[0]); print('River:',river.__version__); print('NumPy:',numpy.__version__); print('Pandas:',pandas.__version__); print('SciPy:',scipy.__version__); print('sklearn:',sklearn.__version__); print('psutil:',psutil.__version__)"
Check-Env "OAML ENVIRONMENT" ".venv-oaml\Scripts\python.exe" "import sys,river,numpy,pandas,scipy,sklearn,gama; from gama import GamaClassifier; print('Python:',sys.version.split()[0]); print('River:',river.__version__); print('NumPy:',numpy.__version__); print('Pandas:',pandas.__version__); print('SciPy:',scipy.__version__); print('sklearn:',sklearn.__version__); print('GAMA: installed')"
Check-Env "EVOAUTOML ENVIRONMENT" ".venv-evo\Scripts\python.exe" "import sys,river,numpy,pandas,scipy,sklearn; from EvOAutoML import classification,pipelinehelper; print('Python:',sys.version.split()[0]); print('River:',river.__version__); print('NumPy:',numpy.__version__); print('Pandas:',pandas.__version__); print('SciPy:',scipy.__version__); print('sklearn:',sklearn.__version__); print('EvoAutoML: installed')"
