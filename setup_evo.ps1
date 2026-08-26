$ErrorActionPreference = "Stop"

Write-Host "AwareML Extension: isolated EvoAutoML environment" -ForegroundColor Cyan

& py -3.8 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.8 is not installed. Install CPython 3.8.10 (64-bit)." }

if (-not (Test-Path ".venv-evo\Scripts\python.exe")) {
    & py -3.8 -m venv .venv-evo
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv-evo." }
}

$evoPython = ".\.venv-evo\Scripts\python.exe"
& $evoPython -m pip install --upgrade "pip<25" "setuptools<76" wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade EvoAutoML packaging tools." }

& $evoPython -m pip install -r requirements-evo.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install EvoAutoML environment requirements." }

& $evoPython -m pip install --no-deps "EvOAutoML==0.0.14"
if ($LASTEXITCODE -ne 0) { throw "Failed to install EvOAutoML 0.0.14." }

& $evoPython -c "from EvOAutoML import classification, pipelinehelper; from river import datasets; import river, sklearn, numpy, pandas; m=classification.EvolutionaryBaggingClassifier(seed=42); print('EvoAutoML: OK'); print('River:', river.__version__); print('sklearn:', sklearn.__version__); print('NumPy:', numpy.__version__); print('Pandas:', pandas.__version__)"
if ($LASTEXITCODE -ne 0) { throw "EvoAutoML verification failed." }

Write-Host "Native EvoAutoML environment is READY." -ForegroundColor Green
