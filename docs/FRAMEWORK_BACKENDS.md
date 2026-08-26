# Framework backend policy

The original `AutoML_Stream` repository contains framework logic in separate files for AutoClass, AutoStreamML and OAML, while EvoAutoML and ChaCha integration is embedded in the large backend module. The extension exposes all five through one adapter contract while preserving dependency boundaries.

| Framework | Legacy source inspected | Required River | Extension behavior |
|---|---|---:|---|
| AutoStreamML | `AutoStreamML.py` | 0.10.1 | Main-process adapter preserving adaptive candidate population / neighborhood-search behavior. |
| AutoClass | `AutoClass.py` | 0.10.1 | Main-process adaptive population adapter; random feature-importance fallback is not used for research outputs. |
| EvoAutoML | imports/runners in `backend_stream.py` (`EvOAutoML`) | 0.10.1 | Native package availability is surfaced; native integration must be validated before publication claims. |
| **OAML** | `OAML.py` | **0.8.0** | Runs in `.venv-oaml` through a JSON worker process so River 0.8.0 never replaces River 0.10.1 in the dashboard environment. GAMA is kept in the same isolated stack. |
| ChaCha | `backend_stream.py` (`flaml.onlineml.autovw.AutoVW`) | 0.10.1 in the shared AwareML stack | AutoVW availability is probed; any fallback is explicitly labeled. |

## Why OAML is a separate process

The project cannot satisfy `river==0.10.1` and `river==0.8.0` in the same virtual environment. The OAML worker therefore has a strict process boundary. This is not only an installation workaround: it makes dependency versions auditable and prevents accidental upgrades from changing the behavior of another framework.

## Why backends are labeled

A scientific comparison must not silently substitute one algorithm for another. Every result records a `backend` string. Paper-grade experiments that claim results for native EvoAutoML, the full GAMA/OAML search path, or FLAML AutoVW/ChaCha must validate those upstream implementations and record exact versions in the experiment manifest.

## Native-integration rule

The adapter boundary is intentionally stable. Native framework integrations should preserve the same externally visible contract (`predict_one`, `learn_one`, `reset`, `get_params`) or use an isolated whole-run worker if the upstream package cannot support that contract. UI, ranking, fairness, and study pages should not need to know framework-specific dependency details.
