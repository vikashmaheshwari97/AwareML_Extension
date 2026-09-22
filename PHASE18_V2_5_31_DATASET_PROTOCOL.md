# AwareML Phase 18 — 31-dataset final held-out protocol (v2.5)

This package implements the user's chosen 31-dataset design. `adult_binary_stream.csv` is intentionally excluded because its SHA256 exactly overlaps the V2 development dataset `adult`. The remaining 31 source CSVs are used.

Counts: 31 datasets = 23 historical + 8 extension; 5 frameworks; seeds 42/43/44; 465 framework executions; 155 seed-aggregated dataset/framework profiles; 100 frozen preferences per dataset = 3,100 preference cases.

## Explicit task policy

15 datasets already have low-cardinality classification targets and use those targets unchanged. 16 datasets whose source targets were previously flagged as high-cardinality numeric or ambiguous multiclass are retained by defining **derived classification tasks** before any framework outcome is generated. For those rows the package freezes an ordinal quantile transformation with at most five classes (20/40/60/80 percentiles, duplicate thresholds removed), drops the original source target from model features, and hashes the transform parameters. These are **not the original regression targets** and must be described as derived classification tasks in the journal.

The transform is benchmark construction, not recommender training. It is frozen before any AutoML run and applied identically to AutoStreamML, AutoClass, EvoAutoML, OAML, and ChaCha.

## Apply and validate

```powershell
cd "C:\Users\maheshwari\PycharmProjects\AwareML_Extension"
Expand-Archive -Path "$HOME\Downloads\AwareML_Phase18_31_Datasets_v2_5.zip" -DestinationPath . -Force
python .\scripts\validate_phase18_install.py
pytest -q .\tests\test_phase18_final_heldout.py
python .\scripts\validate_phase18_preflight.py
```

Expected pre-freeze result: 31/31 protocol-ready, 0 development overlaps.

## Freeze

Only after reading the complete audit:

```powershell
python .\scripts\phase18_prepare_heldout.py --freeze --confirm REVIEWED_31_DATASET_TASK_POLICY
```

The freeze creates `dataset_manifest_frozen.tsv`, `hpc_task_manifest_465.tsv`, `preference_manifest_3100.csv`, the environment snapshot, and the protocol lock under `data/journal/phase18_final_heldout_31_v1/frozen/`.

## HPC

The SLURM array is 0-464. Use the provided resume helper to inspect or submit missing tasks. After all 465 complete, run `scripts/phase18_collect_results.py`, then `scripts/phase18_evaluate_recommender.py`, then `scripts/phase18_build_journal_tables.py`.
