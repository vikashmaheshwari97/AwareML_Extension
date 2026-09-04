# Phase 13 — Recommender & Multi-Objective Validation

Phase 13 validates three journal claims without consuming the reserved 23-dataset final test split.
All recommender comparisons use the frozen 47-dataset development meta-data and leave-one-dataset-out (LODO) predictions/folds.

## 13.1 Explicit recommender baselines

The frozen seven-model development search remains unchanged. Phase 13 adds ranking-level comparisons against:

- `historical_best_framework`: on each LODO fold, rank frameworks by their mean normalized multi-objective utility across the other 46 development datasets.
- `historical_framework_mean`: predict each objective by its framework-specific historical mean over the other 46 datasets, then apply the same preference utility.
- `accuracy_only`: rank by historical mean accuracy.
- `runtime_only`: rank by historical mean runtime (lower is better).
- `energy_only`: rank by historical mean energy (lower is better).
- `random_selection`: deterministic random ranking with seed 42.
- `ml_recommender_v2`: rank the selected objective-specific LODO OOF predictions from the frozen recommender.

Primary preference scenarios are predeclared as balanced, accuracy-focused, speed-focused and sustainability-focused.
The report contains Top-1, Top-3, normalized regret, achieved utility and ranking Spearman correlation per scenario plus an OVERALL row.

The question "Does ML Recommender V2 outperform simple selection rules?" is answered from the frozen table; the code does not force a positive conclusion.

## 13.2 Energy vs CO2 sensitivity

Three preference settings are evaluated on exactly the same 47 development datasets and selected LODO predictions:

1. Energy only
2. CO2 only
3. Energy + CO2 (0.5 / 0.5)

The report compares full-ranking changes, Top-1 changes, Kendall rank agreement, normalized regret and achieved canonical sustainability utility.
It also recomputes Spearman rho(Energy, CO2) from the development recommender table.

Decision rule:

- high correlation: `abs(rho) >= 0.95`;
- rank-equivalent: Energy-vs-CO2 Top-1 change rate <= 0.10 and mean Kendall tau >= 0.90.

If both hold, Phase 13 retains Energy and CO2 as distinct measured/user-selectable concepts but recommends avoiding interpretation of simultaneous positive Energy+CO2 weights as independent evidence. This prevents double counting while preserving both semantics.

## 13.3 Canonical near-Pareto definition

Journal specification ID: `epsilon_pareto_v1`.

Canonical epsilon: `0.05`.

Candidate objectives are transformed into a normalized all-higher-is-better space. For candidate `i` with normalized desirability vector `z_i`, candidate `j` epsilon-dominates `i` iff:

- `z_jk >= z_ik - epsilon` for every jointly available objective `k`; and
- `z_jk > z_ik + epsilon` for at least one objective `k`.

A candidate is epsilon-Pareto / near-Pareto if no other candidate epsilon-dominates it.
At epsilon = 0 the definition reduces to ordinary Pareto nondominance.

The canonical normalizer is `robust_quantile_05_95_all_higher_v1`: within the candidate set, each objective is clipped at the 5th/95th percentiles, scaled to [0,1], and cost/minimize objectives are inverted.

This definition is shared by:

- the pre-run ML Recommender V2 / 3D Decision Space;
- the observed post-run Decision Lab;
- journal reporting and paper text.

Interactive sensitivity may vary epsilon, but the journal result uses exactly 0.05.

## Inputs

Phase 13 reads only:

- `data/meta/snapshots/recommender_train_v2.parquet`
- `data/meta/snapshots/recommender_v2_oof_predictions.parquet`
- `data/meta/models/recommender_v2/manifest.json`

The reserved 23-dataset final test split is not read.

## Outputs

Generated under:

`data/journal/recommender_multiobjective_validation_v1/`

Results:

- `baseline_comparison_table.csv`
- `baseline_comparison_detail.csv`
- `baseline_conclusion.json`
- `baseline_validation_metadata.json`
- `energy_co2_sensitivity_report.csv`
- `energy_co2_sensitivity_detail.csv`
- `energy_co2_ranking_changes.csv`
- `energy_co2_decision.json`
- `near_pareto_specification.json`
- `near_pareto_specification.md`

Frozen release:

- `frozen/manifest.json`
- `frozen/manifest.json.sha256`
- `data/journal/active_recommender_validation.txt`

## Run

Recommended one-by-one during the first execution:

```powershell
pytest -q .\tests\test_phase13_recommender_validation.py
python .\scripts\run_phase13_baseline_validation.py
python .\scripts\run_phase13_energy_co2_sensitivity.py
python .\scripts\write_phase13_near_pareto_spec.py
python .\scripts\freeze_phase13_validation.py
python .\scripts\validate_phase13_complete.py
```

After debugging, the equivalent full runner is:

```powershell
python .\scripts\complete_phase13.py
```

## Completion gate

Phase 13 is complete only when all of the following exist and validation passes:

- baseline comparison table;
- Energy/CO2 sensitivity report and journal decision;
- canonical near-Pareto specification;
- tests;
- frozen manifest + SHA256;
- explicit confirmation that the reserved 23-dataset final test contents were not used.
