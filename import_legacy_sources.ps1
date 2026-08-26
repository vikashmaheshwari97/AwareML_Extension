$ErrorActionPreference = "Stop"

param(
    [string]$LegacyProject = "C:\Users\maheshwari\PycharmProjects\AutoML_Stream"
)

if (-not (Test-Path $LegacyProject)) {
    throw "Legacy project not found: $LegacyProject"
}

$Target = Join-Path $PSScriptRoot "legacy_backends"
New-Item -ItemType Directory -Force -Path $Target | Out-Null

$Files = @(
    "AutoClass.py",
    "AutoStreamML.py",
    "OAML.py",
    "backend_stream.py",
    "fairness_engine.py",
    "explainability_engine.py",
    "ml_recommender.py",
    "meta_recommender.py",
    "metrics_stream.py"
)

foreach ($File in $Files) {
    $Source = Join-Path $LegacyProject $File
    if (Test-Path $Source) {
        Copy-Item $Source (Join-Path $Target $File) -Force
        Write-Host "Copied $File" -ForegroundColor Green
    } else {
        Write-Host "Skipped missing $File" -ForegroundColor Yellow
    }
}

Write-Host "Legacy sources copied to $Target for migration/reference." -ForegroundColor Cyan
Write-Host "The new UI should be developed in awareml/; do not edit the copied monoliths as the final architecture." -ForegroundColor Cyan
