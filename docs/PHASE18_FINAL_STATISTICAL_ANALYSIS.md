# Phase 18 — Final Statistical Analysis and Journal-Evidence Gate

This package adds the final inferential/statistical layer required by the
Phase-18 held-out journal-evaluation roadmap.

## Scientific unit of analysis

The primary inferential unit is the **held-out dataset after seed aggregation**.

- 31 held-out datasets
- 5 frameworks
- 3 seeds per framework
- 465 raw framework executions
- 155 seed-aggregated dataset-framework profiles
- 3,100 primary preference cases

The script deliberately does **not** treat 465 seed runs as 465 independent
datasets.

## Statistical procedures

For framework comparisons:

1. 95% non-parametric percentile bootstrap confidence intervals over datasets.
2. Friedman omnibus test across the five paired frameworks.
3. Kendall's W as the omnibus effect size.
4. Two-sided Wilcoxon signed-rank tests for all 10 framework pairs.
5. Paired rank-biserial correlation as pairwise effect size.
6. Holm correction across the 10 pairwise comparisons within each metric.

For recommender results:

- Top-1, Top-3, normalized regret and Spearman receive 95% dataset-cluster
  bootstrap confidence intervals across the 31 held-out datasets.

For frozen objective predictions:

- MAE, RMSE, mean Spearman, objective-winner Top-1, and prediction-interval
  coverage (where interval columns are present) receive dataset-bootstrap CIs.

## Integrity protections

`phase18_statistical_analysis.py` is read-only with respect to all empirical
Phase-18 inputs. It refuses to consume files marked `SYNTHETIC_DEMO`.

`phase18_freeze_journal_evidence.py` is a later freeze gate. It also refuses
to freeze if a synthetic demo file exists inside the Phase-18 evidence tree.

The freeze gate includes a check for the previously identified
`health_insurance_Real` positive-label metadata inconsistency. If that
inconsistency remains in the frozen dataset manifest, the final journal freeze
will stop rather than silently certify the evidence.

## Workflow

### A. Apply and test locally in PyCharm

Extract this ZIP into the repository root:

`C:\Users\maheshwari\PycharmProjects\AwareML_Extension`

Then run:

```powershell
cd C:\Users\maheshwari\PycharmProjects\AwareML_Extension

python -m pytest -q .\tests\test_phase18_statistical_analysis.py

git status --short
```

Commit only the code/docs/test files from this package:

```powershell
git add `
  .\scripts\phase18_statistical_analysis.py `
  .\scripts\phase18_freeze_journal_evidence.py `
  .\tests\test_phase18_statistical_analysis.py `
  .\docs\PHASE18_FINAL_STATISTICAL_ANALYSIS.md

git commit -m "Add Phase 18 final statistical analysis"
git push origin main
```

### B. Pull the code on Rocket

```bash
cd ~/AwareML_Extension
git pull origin main

export AWAREML_PYTHON="$HOME/.conda/envs/awareml-main/bin/python"
export AWAREML_EVO_PYTHON="$HOME/.conda/envs/awareml-evo/bin/python"
export AWAREML_OAML_PYTHON="$HOME/.conda/envs/awareml-oaml/bin/python"
export AWAREML_OAML_MODE="online"
```

Run the existing campaign/recovery checks first:

```bash
"$AWAREML_PYTHON" scripts/phase18_prepare_recovery.py
"$AWAREML_PYTHON" hpc/production/phase18/resume_phase18_campaign.py
```

Both must still report zero missing/invalid work and 465/465 complete.

### C. Run final statistical analysis on Rocket

```bash
"$AWAREML_PYTHON" scripts/phase18_statistical_analysis.py
```

Expected final line:

`STATISTICAL ANALYSIS: PASS`

Outputs are written under:

`artifacts/phase18_final_heldout_31_v1/evaluation/statistical_analysis/`

including:

- `phase18_framework_descriptive_ci.csv`
- `phase18_omnibus_friedman.csv`
- `phase18_pairwise_framework_comparisons.csv`
- `phase18_effect_sizes.csv`
- `phase18_recommender_confidence_intervals.csv`
- `phase18_objective_prediction_confidence_intervals.csv`
- `phase18_confidence_intervals.csv`
- `phase18_statistical_summary.md`
- `phase18_statistical_analysis_manifest.json`

### D. Do not freeze immediately

First inspect and share:

```bash
cat artifacts/phase18_final_heldout_31_v1/evaluation/statistical_analysis/phase18_statistical_analysis_manifest.json

column -s, -t < artifacts/phase18_final_heldout_31_v1/evaluation/statistical_analysis/phase18_omnibus_friedman.csv | less -S

column -s, -t < artifacts/phase18_final_heldout_31_v1/evaluation/statistical_analysis/phase18_recommender_confidence_intervals.csv | less -S
```

Only after the statistical outputs and the Health Insurance metadata check have
been resolved should the final freeze command be run:

```bash
"$AWAREML_PYTHON" scripts/phase18_freeze_journal_evidence.py \
  --confirm FREEZE_PHASE18_JOURNAL_EVIDENCE
```

A successful freeze creates:

`artifacts/phase18_final_heldout_31_v1/evaluation/PHASE18_FROZEN_JOURNAL_EVIDENCE.json`
