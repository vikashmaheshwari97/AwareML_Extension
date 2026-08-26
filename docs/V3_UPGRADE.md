# AwareML Extension — V3 Upgrade

This upgrade focuses on research correctness and application robustness rather than visual changes alone.

## What changed

- Fairness and Explainability are separate research workspaces.
- Every Plotly element receives a unique Streamlit key to prevent `StreamlitDuplicateElementId` failures.
- CodeCarbon measurement is controlled directly by the Run Studio toggle; there is no second hidden environment-variable gate.
- Missing energy/CO2 stays `None`/`N/A` and carries measurement diagnostics.
- EvoAutoML runs through the isolated `.venv-evo` interpreter and the upstream `EvolutionaryBaggingClassifier` classification path.
- OAML can use the isolated River 0.8 online path or an experimental GAMA warm-up/refit bridge.
- ChaCha first attempts FLAML AutoVW/Vowpal Wabbit and records an explicit fallback if native execution fails.
- Ollama models are discovered dynamically from the local `/api/tags` endpoint. Users can select any installed local model.
- Natural-language objective parsing and grounded chat preserve deterministic fallbacks.
- Decision Lab can optimize against a selected fairness criterion, not only a single composite score.
- Run Studio exports a human-readable YAML experiment specification.
- Research Protocol translates human-centric static AutoML ideas into controlled streaming experiments.

## Human-centric LLM + AutoML inspiration

The 2025 KDIR paper *Integrating Large Language Models into Automated Machine Learning: A Human-Centric Approach* uses a local Ollama-hosted LLM, interactive configuration refinement, human-readable configuration summaries, and an AutoML execution layer. AwareML retains the human-centric interaction pattern but makes the experimental unit a temporally ordered stream: prequential evaluation, concept drift, fairness over windows, sustainability measurement, multi-objective ranking, trust calibration, and information-seeking behavior.

The paper's static train/test configuration workflow is therefore not copied directly. Its ideas are translated into:

1. local-LLM model discovery and user choice;
2. natural-language objective interpretation with schema validation;
3. auditable YAML protocol export;
4. explicit human confirmation of target/sensitive attribute/budgets;
5. grounded post-run explanations over derived evidence;
6. an evaluation design comparing manual, deterministic-assisted, and LLM-assisted configuration.

## Reproducible environments

- `.venv`: Python 3.8.10, River 0.10.1, Streamlit, FLAML/AutoVW, CodeCarbon, SHAP/LIME/XGBoost.
- `.venv-oaml`: Python 3.8.10, River 0.8.0, GAMA 22.0.0, sklearn 1.1.3.
- `.venv-evo`: Python 3.8.10, River 0.10.1, EvoAutoML 0.0.14, sklearn 1.0.2.

`run.ps1` wires the OAML and Evo interpreters automatically.
