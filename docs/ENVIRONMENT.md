# Reproducible Python Environment

## Published-experiment baseline

AwareML is kept compatible with the historical experiment family:

- CPython 3.8.10
- River 0.10.1 for AutoStreamML, AutoClass, EvoAutoML and ChaCha integration
- River 0.8.0 for native OAML

River 0.10.1 and 0.8.0 cannot coexist in the same virtual environment. The extension therefore uses two environments:

1. `.venv` — UI, analysis, recommender, AutoStreamML, AutoClass, EvoAutoML and ChaCha-side dependencies.
2. `.venv-oaml` — native OAML/GAMA execution only.

This separation is intentional. It prevents pip from silently replacing River and changing the behavior of the other streaming frameworks.

## Setup

```powershell
.\setup.ps1
.\setup_oaml.ps1
```

## Validate

Main environment:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -c "import river; print(river.__version__)"
```

Expected: Python 3.8.x and River 0.10.1.

OAML environment:

```powershell
.\.venv-oaml\Scripts\Activate.ps1
python --version
python -c "import river; print(river.__version__)"
```

Expected: Python 3.8.x and River 0.8.0.

## Legacy migration

The old project remains useful as the canonical implementation source while the extension is refactored into smaller modules. To copy the exact local source files into an ignored migration folder:

```powershell
.\import_legacy_sources.ps1
```

The copied files are for migration/reference and should not become the new monolithic architecture.
