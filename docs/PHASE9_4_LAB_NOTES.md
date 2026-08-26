# Phase 9.4 lab notes

## Why Copilot and the ML Recommender can disagree
- **3D Decision Space / ML Recommender V2**: the user directly sets the four objective weights (Accuracy, Runtime, Energy, CO2).
- **Copilot Workspace**: a natural-language goal is parsed first, then converted into weights plus HCAI requirements, then sent to the same frozen Phase-6 recommender.
- Therefore, the final recommended framework can differ when the parsed weights differ from the manually chosen weights.

## Decision Lab vs 3D Decision Space
- **3D Decision Space** is a **pre-run** prediction surface.
- **Decision Lab** is a **post-run** observed ranking surface.
- If Decision Lab recommends OAML, it means OAML had the highest observed weighted utility after the benchmark ran under the selected criterion and weights.

## Faithfulness Lab recommendation
- Keep the current page as the validated frozen benchmark.
- A future **current-dataset faithfulness probe** is a good addition, but it should be clearly labeled as a live experiment and shown separately from the benchmark evidence.

## Explainability Lab note
- `failed_or_degenerate` for a framework means AwareML did not obtain a trustworthy non-zero explanation signal.
- The safer UI choice is to show the diagnostic state rather than fabricate feature importance.
