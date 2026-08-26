# Phase 9.5 — Research UI Final Polish Guide

## Authoritative recommendation path

- **3D Decision Space** — authoritative Phase-6 pre-run ML Recommender V2.
- **Copilot Workspace** — natural-language goal → explicit four-objective weights + HCAI requirements → same Phase-6 recommender → human review.
- **Decision Lab** — post-run ranking using the outcomes actually measured in the current benchmark.
- **Legacy Recommender Lab** — historical paper baseline only. It may rank differently and should not be used as the main Phase-9 recommendation.

## 3D Decision Space

Two ranking evidence modes are available:

- **Point prediction** — uses central Phase-6 predicted values.
- **Conservative 90% bounds** — uses lower Accuracy and upper Runtime/Energy/CO2 bounds.

Preference sliders do **not** rerun AutoML. They rerank the same predicted candidate profiles.

Two 3D visualizations are available:

- **Raw objective space** — physical units.
- **Normalized desirability space** — each axis is 0–1 and higher is always better, making geometry easier to interpret.

## Drift and refit markers

- Red dotted marker = drift detector event.
- Green dashed marker = explicit refit/retrain event **only if recorded by the backend**.
- A shaded post-drift band indicates the immediate adaptation region.
- Continuous online learning is not falsely labeled as a refit.

## Faithfulness Lab

Keep the frozen Phase-8 benchmark as the primary validation view. This is important because it is comparable and reproducible across all demonstrations.

A future **Live Current-Dataset Faithfulness Probe** is a useful extension, but it should be displayed as a separate live experiment. It should not overwrite or replace the frozen benchmark.

## Explainability and AutoStreamML

If AutoStreamML reports `failed_or_degenerate`, this is not automatically a predictive-model failure. It means the current XAI methods did not return a trustworthy non-zero explanation signal.

Phase 9.5 improves the diagnostic presentation but intentionally does not fabricate feature importance. The underlying adapter should be diagnosed separately before Phase 10 if explanation coverage for AutoStreamML is a required publication result.

## Trust Calibration

Purpose: test whether participant trust tracks recommendation reliability. Correct/weak/wrong conditions are research manipulations only and never replace the operational recommendation.

## Information-Seeking Lab

Purpose: study what users do after receiving a recommendation. Follow-up questions are categorized into behaviors such as requesting evidence, challenging the result, comparing alternatives or asking for clarification. This provides HCAI evidence beyond simple satisfaction scores.
