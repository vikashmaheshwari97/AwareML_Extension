# Phase 3.1 — Pre-HPC Research Hardening

Phase 3.1 is a narrow integrity hotfix applied after the Adult Phase-3 validation exposed several issues that should not enter the 47-dataset production meta-dataset.

## 1. Drift episode consistency

Drift degradation is now decided inside a fixed post-drift assessment horizon. If no material drop is observed by the end of that horizon, the episode is frozen as `degradation_observed=false`; later unrelated fluctuations can no longer change `min_accuracy_after`. If degradation is observed, the episode continues until recovery or the configured recovery horizon.

This guarantees the invariant:

- `degradation_observed=false` => `accuracy_drop <= tolerance`
- `degradation_observed=true` => `accuracy_drop > tolerance`

## 2. LIME + streaming categorical encoding

`StreamingEncoder` reserves code `0` for missing/unknown values and assigns categories from `1..K`. LIME categorical-name arrays now preserve those exact integer positions. The original human-readable category labels are passed from the live encoder into the LIME explainer.

## 3. XAI replay warning

The final-state-model replay warning is now configurable through `RunConfig.xai_replay_warning_threshold` and the CLI flag `--xai-replay-warning-threshold`. The journal default is `0.05` (five percentage points), reduced from the Phase-3 hard-coded `0.20`.

## 4. Prediction-degeneracy diagnostics

Every framework result now records prediction coverage, missing predictions, predicted-label counts, positive prediction rate, majority prediction fraction and a near-constant warning. The default near-constant threshold is `0.95` and is configurable with `--near-constant-threshold`.

This is especially important for fairness interpretation: a weak near-constant classifier can exhibit deceptively small group-rate gaps.

## 5. Dataset provenance

Recorded experiments now include:

- raw source-file SHA-256 when a file path is available;
- in-memory dataframe SHA-256;
- rows, columns and feature count;
- column names, dtypes, missing counts and cardinalities;
- target distribution;
- sensitive-attribute distribution;
- numeric/categorical feature lists;
- requested sample range.

The CLI records only the source file name, not the user's absolute local path.

## 6. AutoClass Adult multi-seed audit

Use `scripts/validate_autoclass_adult_multiseed.py` to run the full Adult stream for seeds 42/43/44 under test-then-train evaluation and `audit_only` sensitive-attribute handling. It also checks for exact target-copy columns and writes a reproducible JSON audit.

## 7. Gradual-drift integrity run

Use `scripts/validate_gradual_drift_phase31.py` on the gradual-drift Hyperplane stream. The script validates every drift episode and exits non-zero if the logical invariants above are violated.

## Production gate

Do not launch the 47-dataset Slurm production sweep until:

1. `validate_phase31_hotfix.py` passes;
2. the full-Adult AutoClass multi-seed audit is reviewed;
3. the gradual-drift validation reports zero inconsistent episodes.
