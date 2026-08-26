# Phase 6.2 — ML Recommender V2 model benchmark

Phase 6.1 froze `recommender_train_v2.parquet` with 235 dataset/framework rows. Each row aggregates seeds 42/43/44 and retains mean and standard deviation for the measured outcomes.

Phase 6.2 compares candidate meta-models using **Leave-One-Dataset-Out (LODO)** evaluation.

## Why LODO

All five framework rows from one dataset are held out together. No row from the held-out dataset appears in training. This prevents the common leakage error of randomly splitting dataset/framework rows.

Each fold therefore trains on 46 datasets × 5 frameworks = 230 rows and predicts all 5 framework outcomes for the held-out dataset.

## Primary objectives

Models are evaluated independently for:

- accuracy — maximize
- runtime — minimize
- energy — minimize
- CO2 — minimize

The positive resource targets are learned in `log1p` space and predictions are transformed back to physical units.

## Candidate models

The benchmark includes:

- FrameworkMean
- Ridge
- kNN
- RandomForest
- ExtraTrees
- HistGradientBoosting
- XGBoost, when installed

XGBoost is optional because the project already keeps it in `requirements-optional.txt`.

## Metrics

Each target/model combination reports:

- MAE
- RMSE
- normalized MAE
- normalized RMSE
- R²
- mean per-dataset Spearman rank correlation
- Top-1 framework selection accuracy
- Top-3 framework coverage
- mean normalized regret

## Model selection

The selected model may differ by objective.

Selection prioritizes:

1. minimum normalized regret
2. maximum Top-1 framework accuracy
3. maximum Spearman correlation
4. minimum normalized MAE

This prioritizes the recommender's actual framework-selection task while retaining pointwise calibration as a tie-breaker.

## Artifacts

The benchmark writes:

- `data/meta/snapshots/recommender_v2_benchmark_metrics.parquet`
- `data/meta/snapshots/recommender_v2_oof_predictions.parquet`
- `data/meta/snapshots/recommender_v2_model_selection.json`
- a `.sha256` file for each artifact

## Run

```powershell
python .\scripts\run_recommender_v2_benchmark.py
```

If XGBoost is not installed and you want it included:

```powershell
python -m pip install xgboost==2.1.4
```

To intentionally skip XGBoost:

```powershell
python .\scripts\run_recommender_v2_benchmark.py --no-xgboost
```

## Validate

```powershell
python .\scripts\validate_recommender_v2_benchmark.py
```

## Tests

```powershell
pytest .\tests\test_recommender_v2_benchmark.py -q
```

After Phase 6.2 passes, Phase 6.3 fits the selected objective-specific models on all 47 development datasets and adds uncertainty estimation. Phase 6.4 then constructs preference-aware ranking from the four predicted objectives.
