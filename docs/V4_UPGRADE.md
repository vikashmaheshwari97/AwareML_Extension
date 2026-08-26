# AwareML Extension V4 — legacy-informed research UI

V4 is a structured migration of the strongest research interactions from the original `AutoML_Stream` dashboard into the modular extension architecture.

## What is intentionally reused

The original dashboard already contained useful research ideas that should not be discarded:

- one persistent uploaded dataset shared through `st.session_state`;
- two pre-run recommendation pathways: an LLM-assisted planner and an ML-based meta-recommender;
- streaming fairness over windows rather than a single static fairness number;
- multi-level explainability: model-level, hyperparameter-level, and system-level;
- drift/adaptation timelines;
- historical `meta_logs.json` used as recommender evidence;
- local Ollama for natural-language planning and summaries.

V4 ports those ideas into small modules instead of copying the old monolithic frontend/backend.

## New V4 workspaces

1. **Run Studio** — upload/load once, confirm target and sensitive attribute, inspect class distribution/correlation/missingness, run the shared prequential benchmark.
2. **Recommender Lab** — LLM-Assisted Recommender, ML Recommender, and Meta-Evidence & Validation tabs. The active dataset is reused automatically.
3. **Decision Lab** — post-run multi-objective ranking and Pareto evidence.
4. **Drift & Temporal Lab** — temporal accuracy, ADWIN drift markers, window heatmaps, drift counts.
5. **Fairness Lab** — DP, EO, Equalized Odds, Predictive Parity, Error-rate parity; temporal gaps; group support.
6. **Explainability Lab** — model-level, hyperparameter-level, system-level evidence.
7. **Sustainability Lab** — measured energy/CO2, runtime-energy and energy-CO2 relationships, measurement provenance.
8. **Trust Calibration** and **Information-Seeking Lab** — controlled human-study instrumentation.

## Historical meta logs

`data/meta/meta_logs.json` is historical experiment metadata, not a raw dataset. V4 uses it for pre-run framework recommendation. Non-positive historical carbon values are treated as missing before fitting the carbon model, because a stored zero can represent an unmeasured value.

## Ollama

The application talks to the Ollama HTTP API at `http://127.0.0.1:11434` by default. If Ollama Desktop is already running, `ollama serve` will fail with a port-bind error. That is expected; use `ollama list` or query `/api/tags` instead of starting a second server.

## Research-integrity boundary

V4 does not claim that historical meta-model predictions are the current stream's result. They are shown as pre-run estimates and must be verified through Run Studio. Likewise, hyperparameter values are shown as a snapshot unless a framework provides a genuine temporal tuning trace.
