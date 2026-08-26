# Versioned recommender snapshots

Generated recommender training snapshots belong here only after validation and explicit versioning. Do not overwrite an existing snapshot.

## Meta-Dataset V2

`meta_logs_v2.json` is the immutable, human-readable canonical evidence assembled from the validated Phase-5 campaigns.

Phase 6.1 derives the following artifacts without rerunning any framework:

- `meta_logs_v2.parquet` — 705 canonical run rows, one per `(dataset_id, framework, seed)`.
- `recommender_train_v2.parquet` — 235 dataset/framework rows after aggregating seeds 42/43/44.
- `meta_logs_v2_schema_report.json` — schema/type/null inventory for the canonical run table.
- `recommender_train_v2_schema.json` — explicit recommender feature/target contract.
- matching `.sha256` files for each generated artifact.

The four primary Phase-6 prediction targets are kept separate:

1. accuracy (maximize)
2. runtime (minimize)
3. energy (minimize)
4. CO2 (minimize)

No fixed utility target is stored in the training snapshot. Preference-aware utility is computed later from model predictions and explicit user weights.

Do not change `data/meta/active_snapshot.txt` to the V2 training snapshot until the Phase-6 recommender implementation is integrated and validated.
