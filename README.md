# AwareML Extension

AwareML is an interactive system for transparent, controllable and sustainable streaming AutoML. It helps users express deployment needs, compare candidate frameworks, inspect streaming performance and review evidence behind recommendations.

## Repository structure

| Path | Contents |
| --- | --- |
| `app.py` | Main Streamlit dashboard |
| `awareml/` | Framework adapters, streaming execution, recommenders, analysis, workspaces and study support |
| `configs/` | Experiment configurations and dataset inventories |
| `data/Supplementary Data/stream_datasets/` | Development stream datasets |
| `data/Supplementary Data/External Evaluation Dataset/` | External evaluation and application testing datasets |
| `data/journal/` | Scenario benchmarks, annotations, study designs and research evidence |
| `data/meta/` | Versioned recommender assets and metadata |
| `prompts/` | Language-model prompt resources |
| `scripts/` | Preparation, validation and evaluation utilities |
| `tests/` | Automated checks |
| `hpc/` | Cluster execution and scheduling support |
| `docs/` | Protocol and participant documentation |
| `CD/` | Critical-difference figures |
| `study_cloud/` | Study deployment support |

## Features

- Compare AutoStreamML, AutoClass, EvoAutoML, OAML and ChaCha.
- Express preferences over accuracy, runtime, energy and CO2.
- Obtain dataset-aware recommendations from a labelled dataset prefix or inspect historical preference rankings.
- Monitor temporal performance and detected drift.
- Inspect fairness, feature attributions and explanation evidence.
- Review recommendations, record decisions and export run evidence.
- Access trust-calibration and information-seeking study workspaces.

Predicted and observed outcomes serve different purposes. Use their provenance and the relevant experiment configuration when interpreting results. Optional demonstration data stays local and is not distributed as study evidence.

## Requirements

The supplied dependency sets target **CPython 3.8.10**. The project specifies Python >=3.8.10 and <3.9.

OAML and EvoAutoML run in separate Python environments because their dependencies differ from the main application. Windows PowerShell setup scripts are included.

Some native dependencies require platform build tools. Install the required compiler and system libraries if a package cannot be installed from a wheel.

## Installation

Clone the repository and select the branch containing the current application:

```powershell
git clone --branch main https://github.com/vikashmaheshwari97/AwareML_Extension.git
cd AwareML_Extension
```

Create the main and isolated backend environments:

```powershell
.\setup.ps1
.\setup_oaml.ps1
.\setup_evo.ps1
```

The main setup installs `requirements.txt` and `requirements-optional.txt`. The other setup scripts install the OAML and EvoAutoML dependency sets into their own environments.

Create local configuration:

```powershell
Copy-Item .env.example .env
```

Edit `.env` for your machine. It includes the Ollama service URL, model name, country code, study database path and isolated backend interpreter paths.

## Launch

From the repository root:

```powershell
.\run.ps1
```

The launcher checks all three environments, sets the backend interpreter paths and starts Streamlit.

For an existing configured environment, the dashboard can also be started directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## First experiment

1. Open **Run Studio** and load an uploaded CSV or the built-in synthetic stream.
2. Select the target column and, when needed, a sensitive attribute and positive label.
3. Review the objectives and weights in the **Goal and Recommendation Panel**.
4. Configure the frameworks and run the experiment.
5. Inspect **3D Decision Space**, **Streaming Observatory**, **Responsible AI** and **Explanation Integrity Lab**.
6. Use **Export Center** to save the available evidence.

For the built-in synthetic example, the application suggests `target`, `group` and positive label `1`.

## Language-model configuration

The goal interpreter and language-assisted explanations use the configured Ollama endpoint. Run Ollama separately and select a model installed on that service.

Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env`. The example file provides defaults; the interface discovers available models.

## Additional entry points

Run these with the main environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run meta_recommender_v3_app.py
.\.venv\Scripts\python.exe -m streamlit run trust_calibration_app.py
.\.venv\Scripts\python.exe -m streamlit run information_seeking_app.py
.\.venv\Scripts\python.exe -m streamlit run assessor_review_app.py
```

Each command starts a separate Streamlit application. Study workspaces depend on their corresponding design and stimulus assets.

## Recommender and benchmark assets

A dataset-aware recommender requires a compatible trained model bundle. If the application reports that a bundle is missing or pending, check its configured location and prepare the required training artifacts.

The experiment command-line interface is available through:

```powershell
.\.venv\Scripts\python.exe scripts/awareml_v3.py --help
```

It provides preparation, execution, collection, training, evaluation and recommendation commands. Use the configuration and model manifest associated with a particular run; historical and current artifacts can use different protocols.

Required datasets, scenario annotations, inventories and research evidence remain versioned. Duplicate dataset copies and obsolete installation payloads are excluded. Generated results should be written under `artifacts/`.

## Development checks

Install development dependencies and run the checks from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-protocol-v3.in
.\.venv\Scripts\python.exe -m compileall -q awareml
.\.venv\Scripts\python.exe -m pytest -q
```

Some checks require additional research dependencies, trained bundles or generated experiment artifacts. Review any reported failures in that context.

## Repository maintenance

- Keep dependency requirement files and the root `README.md`.
- Keep CSV files required by the application, benchmark protocols and reproducibility.
- Store local run outputs, caches, installer payloads and backups outside versioned source.
- Keep local configuration and participant runtime data out of commits.
- Review `git status` and the staged diff before publishing changes.
