# AwareML Extension dependency environments

AwareML must not install all five framework stacks into one virtual environment.
The accepted system combines frameworks whose original implementations depend on
incompatible River releases.

## Compatibility matrix

| Runtime | Frameworks / role | Python | River | Notes |
|---|---|---:|---:|---|
| `.venv` | UI, shared analysis, AutoStreamML, AutoClass, EvoAutoML integration, ChaCha integration | 3.8.10 | **0.10.1** | Main AwareML process |
| `.venv-oaml` | OAML worker / GAMA-compatible path | 3.8.10 | **0.8.0** | Isolated subprocess |

The project standardizes on CPython 3.8.10 because that is the experiment interpreter used for the accepted AwareML system. The original framework dependency stacks are older and should not be upgraded implicitly during extension experiments.

## Why two environments are required

`river==0.10.1` and `river==0.8.0` cannot be installed simultaneously in one Python
environment. Installing one will replace the other and can break class names, drift-detector
APIs, and framework behavior. OAML therefore runs behind a newline-delimited JSON worker
process. The main Streamlit process remains on River 0.10.1.

## Setup

From the project root in Windows PowerShell:

```powershell
.\setup.ps1
.\check_envs.ps1
.\run.ps1
```

`setup.ps1` creates the main environment with the Windows Python launcher (`py -3.8`). Run `setup_oaml.ps1` separately to create the isolated OAML environment, also with Python 3.8.
`run.ps1` automatically points AwareML to the OAML interpreter through
`AWAREML_OAML_PYTHON`.

## Publication experiments

Before collecting benchmark results, record the exact output of:

```powershell
.\check_envs.ps1
.\.venv\Scripts\python.exe -m pip freeze > artifacts\environment-main.txt
.\.venv-oaml\Scripts\python.exe -m pip freeze > artifacts\environment-oaml.txt
```

These files should be archived with experiment metadata but do not need to be committed.
