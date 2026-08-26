# AwareML Extension

> **Reproducibility note (Phase 1 foundation on V4):** use **CPython 3.8.10**. The main UI uses **River 0.10.1**. OAML/GAMA and EvoAutoML have isolated dependency environments so their historical stacks cannot downgrade the Streamlit environment. See `docs/V4_UPGRADE.md`.


**AwareML Extension** is a clean research implementation extending the accepted CIKM AwareML system while preserving its central idea: one interface for comparing **AutoStreamML, AutoClass, EvoAutoML, OAML, and ChaCha** under performance, drift, fairness, explainability, and sustainability objectives.

The extension is designed around the accepted paper's six-stage workflow and the reviewers' requested improvements: broader real-stream evaluation, stronger meta-recommender baselines, documented sustainability measurement, explicit uncertainty/correlation analysis, stronger fairness diagnostics, grounded LLM behavior, trust calibration, and information-seeking studies.

## Important dependency architecture

The five frameworks **must not be installed into one virtual environment**.

- **Main UI + AutoStreamML + AutoClass + ChaCha/AutoVW:** `river==0.10.1`
- **OAML/GAMA:** isolated `.venv-oaml`, `river==0.8.0`
- **EvoAutoML 0.0.14:** isolated `.venv-evo`, `river==0.10.1`, `scikit-learn==1.0.2`
- **Python:** the project standardizes on **CPython 3.8.10**, matching the accepted-system experiment environment. Newer Python versions are not used for the reproducibility environment.

Therefore the project uses three environments:

```text
AwareML_Extension/
├── .venv/          # Python 3.8.10 + River 0.10.1 (main UI, AutoStreamML, AutoClass, ChaCha)
├── .venv-oaml/     # Python 3.8.10 + River 0.8.0 (isolated OAML/GAMA worker)
└── .venv-evo/      # Python 3.8.10 + River 0.10.1 + EvoAutoML 0.0.14
```

OAML and EvoAutoML communicate with the main AwareML process through small JSON subprocess bridges. This preserves their required historical dependency stacks while keeping the Streamlit/UI environment stable.

See `docs/ENVIRONMENTS.md` for the full compatibility rationale.

## Windows / PyCharm setup

### 1. Extract the ZIP first

If the project ZIP is in your Downloads directory:

```powershell
cd C:\Users\maheshwari\Downloads
Expand-Archive -Path .\AwareML_Extension.zip -DestinationPath C:\Users\maheshwari\PycharmProjects -Force
cd C:\Users\maheshwari\PycharmProjects\AwareML_Extension
```

The earlier `cd ...\AwareML_Extension` error simply means that the ZIP had not yet been extracted into `PycharmProjects`.

### 2. Create the main reproducibility environment

```powershell
.\setup.ps1
```

The script requires the Windows launcher to find Python 3.8 via:

```powershell
py -3.8 --version
```

It creates:

```text
.venv       -> Python 3.8.10 + River 0.10.1
```

### 3. Create the isolated OAML environment

```powershell
.\setup_oaml.ps1
```

This creates:

```text
.venv-oaml  -> Python 3.8.10 + River 0.8.0 + OAML/GAMA dependencies
```

### 4. Create the isolated EvoAutoML environment

```powershell
.\setup_evo.ps1
```

This creates `.venv-evo` and verifies the upstream `EvolutionaryBaggingClassifier` path.

### 5. Verify exact versions

```powershell
.\check_envs.ps1
```

Expected River versions:

```text
MAIN:  River 0.10.1
OAML:  River 0.8.0
EVO:   River 0.10.1
```

### 6. Start AwareML

```powershell
.\run.ps1
```

or explicitly:

```powershell
$env:AWAREML_OAML_PYTHON = (Resolve-Path ".venv-oaml\Scripts\python.exe").Path
$env:AWAREML_EVO_PYTHON = (Resolve-Path ".venv-evo\Scripts\python.exe").Path
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 7. Optional research packages

Install optional explainability, sustainability, XGBoost, and FLAML components only in the main environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

Do **not** install `river==0.8.0` into `.venv`, and do **not** upgrade River in `.venv-oaml`.

## New research capabilities

- Advanced multi-page research UI with **Run Studio, Recommender Lab, Decision Lab, Drift & Temporal Lab, Fairness Lab, Explainability Lab, Sustainability Lab, Trust Calibration, Information-Seeking, and Research Protocol** workspaces.
- Formal objective-weight schema and utility calculation instead of implicit ranking rules.
- Explicit epsilon-Pareto analysis and objective-correlation warnings.
- Bootstrap uncertainty intervals for stream-level performance summaries.
- Stronger meta-recommender evaluation: historical mean, kNN, Ridge, Random Forest, Extra Trees, HistGradientBoosting, and optional XGBoost, evaluated with dataset-grouped holdouts.
- Sustainability protocol that stores hardware/software context, repetitions, region, and measurement status. Missing energy data is reported as **not measured**, not zero.
- Expanded fairness diagnostics: demographic parity, equal opportunity, equalized odds, predictive parity, and error-rate gaps with support/undefined states.
- Explanation diagnostics based on perturbation/resampling behavior; no random feature-importance fallback.
- Grounded local-first Ollama integration: discover installed models dynamically, use any selected local model, and send derived dashboard facts rather than raw rows by default.
- Human-readable YAML export of the exact streaming experiment protocol.
- Controlled trust-calibration study with matched explanation style across correct/weak/wrong conditions.
- Information-seeking study logging evidence requests, challenges, comparisons, clarification, follow-up depth, and first-answer acceptance.


## V4 legacy-to-extension design

V4 deliberately reuses the strongest scientific ideas from the original `AutoML_Stream` dashboard without copying its 450k-line monolithic frontend. The old behavior is ported into clean modules and shared session state:

- **Upload the dataset once in Run Studio.** The active DataFrame, target, sensitive attribute and positive label remain in `st.session_state` and are reused by Recommender, Decision, Drift, Fairness, Explainability and Sustainability workspaces.
- **Two pre-run recommenders are separate again:**
  1. **LLM-Assisted Recommender** — natural-language objective → validated objective weights → historical-evidence ranking, with optional local Ollama explanation.
  2. **ML Recommender** — explicit accuracy/runtime/energy/CO₂ preferences → Random-Forest meta-model trained from historical AwareML logs.
- **Historical meta evidence is bundled as research metadata** at `data/meta/meta_logs.json`; raw datasets are still not bundled. The loader treats non-positive historical CO₂ placeholders as missing rather than as physically exact zero emissions.
- **Fairness is a dedicated lab:** five complementary criteria, cross-framework heatmap/bar/radar views, temporal disparity traces, worst-window reporting and group-support diagnostics.
- **Explainability is a dedicated multi-level lab:** model-level importance/quality diagnostics, hyperparameter-level context, and system-level cross-framework feature-agreement analysis.
- **Drift & Temporal Lab** restores the old dashboard's strong temporal-analysis idea with prequential performance curves, ADWIN event markers and window-level heatmaps.
- **Ollama is detected through its local HTTP API.** If port `11434` is already bound by Ollama Desktop, do not start a second `ollama serve` process.

## Versioned meta-experiment foundation

The historical 47-dataset evidence is now frozen at `data/meta/legacy/meta_logs_v1_47.json`, with its integrity hash in `data/meta/legacy/manifest.json`. The exact V1 training set lives in `data/meta/manifests/train_v1_47.yaml`. New datasets must create a new manifest/snapshot version rather than being appended into V1.

Journal-scale runs use per-experiment artifacts under `artifacts/meta_experiments/<experiment_id>/` rather than a shared `meta_logs.json`. The V2 contract separates run summaries, temporal metrics, drift events, fairness snapshots, explainability snapshots and sustainability measurements. See `docs/PHASE1_META_FOUNDATION.md`.

The project history identifies 23 held-out V1 evaluation datasets, but their names are not available in the current source/meta-log snapshot. `test_v1_23.yaml` is therefore deliberately incomplete instead of inventing dataset identities. It must be populated and frozen before the held-out recommender evaluation is executed.

## Data policy

No raw datasets are bundled. Put local datasets under `data/local/`, which is ignored by Git. Historical experiment metadata in `data/meta/meta_logs.json` is intentionally versionable because it is used by the pre-run ML recommender and contains no raw dataset rows. A synthetic drift stream is included for development checks. `configs/datasets.yaml` records benchmark categories and temporal-order expectations without committing raw datasets.

## Framework integrity

The UI reports the backend actually used. Optional or compatibility backends must never be silently presented as a native implementation. The isolated OAML process is specifically a dependency boundary; native GAMA/OAML search work remains inside that environment so it can be upgraded without changing the Streamlit environment.

## Research integrity safeguards

- Deliberately wrong recommendations exist only inside the Trust Calibration study.
- Missing sustainability measurements remain `not measured`/`None`.
- Unsupported explanation metrics return `N/A` rather than invented values.
- Sensitive-attribute selection requires explicit confirmation.
- Raw dataset rows are not sent to the chat model by default.

For the scientific design, see `docs/REVIEWER_RESPONSE_MATRIX.md`, `docs/RESEARCH_PROTOCOL.md`, `docs/MIGRATION_AUDIT.md`, and `docs/PAPER_BASELINE.md`.


## Phase 2 — Streaming instrumentation

The research core now emits consistent prequential Accuracy/Macro-F1, rolling metrics, latency/throughput, drift degradation/recovery, temporal fairness with worst-group performance, explainability records, and framework-only sustainability measurements. See `docs/PHASE2_STREAMING_INSTRUMENTATION.md`.


## Phase 3 — Framework hardening

Phase 3 hardens the local five-framework path before Slurm/HPC generation. Model-level XAI now uses an auditable **SHAP → LIME → repeated-permutation** cascade; the method actually used and every failed/degenerate attempt are recorded. All-zero explanations are no longer displayed as valid evidence. LIME receives known categorical features from the streaming encoder.

The drift-recovery protocol now requires a post-drift assessment horizon, temporal fairness summaries preserve undefined windows, and protected attributes have an explicit model-access policy (`audit_only` by default or `include`). The ChaCha adapter follows FLAML AutoVW's predict-then-learn state contract and uses FLAML tuning domains. EvoAutoML/OAML workers use batched prediction APIs for lower XAI subprocess overhead.

See `docs/PHASE3_HARDENING.md` and run `python scripts/validate_phase3_hardening.py` before a recorded Phase-3 benchmark.

## Phase 3.1 pre-HPC hardening

Phase 3.1 fixes drift-summary consistency, aligns LIME categorical labels with the streaming encoder, lowers the configurable final-state XAI replay warning to 0.05, adds near-constant prediction diagnostics, and records dataset SHA-256/schema/class-distribution provenance. Before production Slurm runs, execute the full-Adult AutoClass multi-seed audit and the gradual-drift consistency validation described in `docs/PHASE3_1_HOTFIX.md`.
