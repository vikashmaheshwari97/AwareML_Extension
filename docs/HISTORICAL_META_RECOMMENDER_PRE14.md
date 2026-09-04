# AwareML Historical Meta-Recommender — pre-Phase-14 hotfix

This hotfix adds a second Copilot Workspace path for users who want an early framework recommendation **before uploading a dataset**.

## Two recommendation paths

### Goal Copilot (existing, unchanged in method)
- Natural-language scenario → LLaMA 3 8B + V3.1 objective interpretation.
- `equal_selected_v1` objective weights.
- Dataset meta-profile → frozen ML Recommender V2 → dataset-aware framework ranking.
- Dataset required for framework ranking.

### Historical Meta-Recommender (new)
- Dataset is not required.
- Uses the frozen Meta-Dataset V2 evidence: 47 datasets × 5 frameworks × 3 seeds = 705 historical runs.
- Ranks frameworks only from Accuracy / Runtime / Energy / CO2 preferences.
- Gives a **global historical prior**, not a prediction for a new dataset.
- The LLM never chooses the framework.

## Seed policy

Primary/default: **Stable 3-seed aggregate**. Uses the 235 dataset/framework profiles in which each profile is the mean of all three seeds.

Exploratory: **Best observed single seed**. For every dataset/framework pair, one actual seed is selected using the complete weighted preference score. The same seed supplies Accuracy, Runtime, Energy and CO2; metrics are never cherry-picked from different seeds.

The exploratory mode is an optimistic upper-bound view and should not replace stable 3-seed reporting.

## Research boundary

The new historical recommender does not replace or retrain ML Recommender V2. It does not modify Phase 12, Phase 13 or the V3.1 selector. Once a dataset is available, Goal Copilot / ML Recommender V2 remains the primary dataset-aware recommendation path.
