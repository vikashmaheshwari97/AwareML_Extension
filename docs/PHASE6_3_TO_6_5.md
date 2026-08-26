# AwareML Phase 6.3–6.5

## Phase 6.3 — final objective models and uncertainty

The Phase-6.2 selected models are fitted on all 47 development/meta datasets:

- accuracy: model selected by Phase 6.2
- runtime: model selected by Phase 6.2
- energy: model selected by Phase 6.2
- CO2: model selected by Phase 6.2

The selected model names are read from `recommender_v2_model_selection.json`; they are not hard-coded.

Uncertainty intervals are calibrated from the selected models' Leave-One-Dataset-Out residuals. They are explicitly described as empirical CV residual intervals, not exact conformal guarantees.

Artifacts are written under:

`data/meta/models/recommender_v2/`

## Phase 6.4 — preference-aware framework ranking

The recommender predicts all four objectives separately, converts each objective to a within-candidate normalized benefit score, and combines them only after the user supplies explicit weights.

Two ranking modes are supported:

- `point`: predicted objective values
- `conservative`: lower accuracy bound plus upper runtime/energy/CO2 bounds

Pareto-efficient candidates are marked.

Energy/CO2 correlation is surfaced as a warning when both objectives receive positive weight. It is not silently corrected or hidden.

Preference sensitivity is evaluated using the Phase-6.2 dataset-level OOF predictions for six predefined preference scenarios.

## Phase 6.5 — integration and freeze

`RecommendationService` remains backward compatible with the existing post-run `rank()` API and gains V2 pre-execution recommendation methods.

The freeze script:

- validates required artifacts exist
- creates a release manifest with hashes
- creates `data/meta/active_recommender_v2.txt`
- does NOT change legacy `data/meta/active_snapshot.txt`
- records that the 23-dataset frozen test split remains untouched until Phase 10

## One-command execution

After extracting the upgrade:

```powershell
python .\scripts\complete_phase6.py
```

Or run each phase individually using `APPLY_PHASE6_3_TO_6_5.txt`.
