# Migration audit: AutoML_Stream -> AwareML_Extension

## Ported or retained conceptually

| Original area | Extension location | Decision |
|---|---|---|
| `AutoStreamML.py` pipeline population + neighborhood search | `awareml/frameworks/autostreamml.py` | Keep core adaptive-search idea; remove duplicate commented implementations. |
| `AutoClass.py` population/mutation | `awareml/frameworks/autoclass.py` | Keep adaptive population; remove random explanation fallback. |
| EvoAutoML runner embedded in `backend_stream.py` | `awareml/frameworks/evoautoml.py` | Isolate adapter; expose native-package availability and transparent fallback. |
| `OAML.py` GAMA/River paths | `awareml/frameworks/oaml.py` + `awareml/workers/oaml_worker.py` | Keep OAML isolated in Python 3.9 / River 0.8.0; the main environment remains on River 0.10.1. |
| ChaCha / FLAML AutoVW in `backend_stream.py` | `awareml/frameworks/chacha.py` | Preserve AutoVW hook; use labeled UCB fallback when optional backend is absent. |
| `fairness_engine.py` | `awareml/analysis/fairness.py` | Expand metrics and support undefined states. |
| `explainability_engine.py` | `awareml/analysis/explainability.py` | Replace proxy/placeholders with perturbation/resampling diagnostics. |
| meta-recommenders / evaluation scripts | `awareml/recommender/` | Replace framework-favoring heuristics with auditable learned/baseline comparison. |
| Ollama integration | `awareml/llm/` | Remove user-specific executable path; configurable local endpoint. |

## Intentionally removed from the clean source tree

- The ~500 KB `backend_stream.py` monolith after responsibilities are separated.
- The ~460 KB `forntend.py` monolith and its typo-bound filename.
- Large blocks of commented obsolete implementations in framework/recommender files.
- Hard-coded `C:\\Users\\...\\Ollama\\ollama.exe` paths.
- Random feature importance used when a model lacks a native explanation.
- Framework-specific ranking bonuses that can bias a scientific comparison.
- Generated meta logs, evaluation CSVs, CodeCarbon outputs and datasets from source control.
- Silent `0.0` energy/CO2 values when no measurement was made.
- Patch/cleanup scripts that existed only to repair intermediate meta-log versions.

## Paper-code discrepancy to document

The accepted paper describes the meta-recommender's default supervised predictor as **XGBoost**. The current `ml_recommender.py` in `AutoML_Stream` uses **RandomForestRegressor** objects for accuracy, runtime, energy and CO2 prediction. The extension therefore treats model family as an explicit experimental factor and benchmarks both (when XGBoost is installed) rather than silently assuming the paper and code are identical.

## Dependency correction

The original framework stacks are not dependency-compatible in one environment. The extension now uses Python 3.9 with two virtual environments: `.venv` (`river==0.10.1`) and `.venv-oaml` (`river==0.8.0`). This is a deliberate architecture decision, not a temporary installation workaround.
