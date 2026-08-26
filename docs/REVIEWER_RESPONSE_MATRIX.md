# Reviewer-response implementation matrix

| Reviewer concern | Extension response | Where |
|---|---|---|
| Evaluation centered on Adult-as-stream | Benchmark manifest distinguishes native, temporal and ordered-as-stream datasets; benchmark runner supports any local CSV in temporal order. | `configs/datasets.yaml`, Run Studio |
| No strong meta-recommender baselines | Historical mean, kNN, Ridge, RF, Extra Trees, HGB, optional XGBoost with leave-one-dataset-out evaluation. | `awareml/recommender/` |
| Utility function under-specified | Objective weights are typed, normalized and applied to robustly normalized metrics. | `types.py`, `engine/pareto.py` |
| “Near-Pareto” unclear | Explicit epsilon-nondominance with visible epsilon. | `engine/pareto.py` |
| Energy and CO2 may be redundant | Spearman objective-correlation matrix and double-counting warning. | Decision Lab |
| Sustainability protocol insufficient | Repeats, seeds, hardware/software context, region, CodeCarbon status, null-not-zero policy. | `configs/sustainability_protocol.yaml` |
| SPD alone is narrow | DP, equal opportunity, equalized odds, predictive parity, error-rate gap plus group support. | `analysis/fairness.py` |
| LLM objective mapping/failure modes unclear | Pydantic schema validation, deterministic fallback, rejected malformed outputs. | `llm/objective_parser.py` |
| LLM grounding/privacy/overtrust | Derived-facts-only prompt, evidence keys, local-first Ollama, no raw rows, explicit trust study. | `llm/`, Trust Calibration page |
| Explainability metric validity | No random importance; measurable perturbation/resampling diagnostics, limitations shown. | `analysis/explainability.py` |
| Technical foundations/extensibility unclear | Common adapter interface and framework registry. | `frameworks/base.py`, `registry.py` |
| Usability lacks control condition | New controlled trust calibration study + qualitative information-seeking study. | `studies/`, study pages |
