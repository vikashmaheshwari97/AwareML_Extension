# Phase 6 — ML Recommender V2

## Phase 6.1: canonical training data

Phase 5 produced the frozen evidence file:

`data/meta/snapshots/meta_logs_v2.json`

The snapshot contains:

- 705 validated runs
- 47 datasets
- 5 frameworks
- 3 seeds per dataset/framework
- 141 runs per framework
- complete accuracy, runtime, energy, CO2, latency, throughput and provenance coverage

Phase 6.1 converts that evidence into two versioned Parquet snapshots.

### Canonical run table

`meta_logs_v2.parquet`

Granularity:

`(dataset_id, framework, seed)`

Expected rows: 705.

Nested JSON structures are flattened where possible. Remaining list/dictionary objects are serialized deterministically as JSON strings so that the Parquet schema is stable.

### Recommender training table

`recommender_train_v2.parquet`

Granularity:

`(dataset_id, framework)`

Expected rows: 235.

Seeds 42, 43 and 44 are treated as repeated measurements of the same experimental condition. For each measured outcome the table stores mean, standard deviation, minimum and maximum.

### Primary targets

The recommender predicts four targets separately:

- accuracy — maximize
- runtime — minimize
- energy — minimize
- CO2 — minimize

A fixed utility score is deliberately not stored in the training data. Utility is formed later using explicit user preferences.

### Meta-features

The Phase-6.1 builder derives non-performance dataset descriptors from frozen run provenance and the audited dataset registry:

- dataset sample count
- total feature count
- numeric feature count
- categorical feature count
- numeric/categorical feature fractions
- missing fraction
- number of classes
- majority class fraction
- minority class fraction
- class imbalance ratio
- normalized class entropy
- dataset family
- source type
- declared drift type
- protocol window and time budget

Run outcomes such as accuracy, runtime, energy, CO2, latency, drift detections, fairness results and XAI results are not used as dataset input features for the recommendation target being learned.

### Why seed aggregation is required

The 705 run rows are not 705 independent datasets. Three runs correspond to the same dataset/framework condition with different random seeds. Training directly on all 705 rows while randomly splitting rows would leak dataset identity across train and test folds.

Phase 6 therefore aggregates the three seeds before meta-model training and later uses dataset-level grouped evaluation.

### Build

```powershell
python .\scripts\build_meta_dataset_v2.py
```

### Validate

```powershell
python .\scripts\validate_meta_dataset_v2.py
```

### Tests

```powershell
pytest .\tests\test_meta_dataset_v2.py -q
```

Only after all Phase-6.1 validation gates pass should Phase 6.2 start model comparison and uncertainty estimation.
