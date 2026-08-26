# Phase 1 — Meta-Experiment Foundation

This phase freezes the scientific data contract before new HPC experiments are launched.

## Why this exists

The historical AwareML metadata was a single appendable `meta_logs.json`. That was useful for local prototyping but it is unsafe for a journal-scale parallel experiment because:

- adding new datasets silently changes the training evidence;
- multiple Slurm tasks must not write to one shared JSON file;
- final metrics alone do not capture temporal streaming behavior;
- legacy zero CO2 values can mean "not measured" rather than zero emissions;
- held-out test datasets must be physically separated from recommender training evidence.

## Frozen V1 evidence

`data/meta/legacy/meta_logs_v1_47.json` is an immutable copy of the 47-dataset historical evidence. Its SHA256 is stored in `data/meta/legacy/manifest.json`.

`data/meta/manifests/train_v1_47.yaml` freezes the exact 47 dataset IDs. Never append a new dataset to this manifest. Future evidence becomes `train_v2`, `train_v3`, etc.

The project history states that V1 evaluation uses 23 held-out datasets. Their names are not present in the available V4 source/meta-log snapshot, so `data/meta/manifests/test_v1_23.yaml` is intentionally incomplete and unfrozen. The validator refuses to pretend that unknown names are known. Populate and freeze this split before journal evaluation.

## Dataset-family leakage

`data/meta/registry/datasets_v1_47.yaml` groups the 47 datasets into families using a clearly marked `heuristic_v1` rule. These family labels are aids for leakage analysis, not ground truth. They must be manually audited before family-held-out evaluation.

## Meta schema V2

Run-level data uses `RunSummaryRecord`. Temporal evidence is split into typed streams:

- `WindowMetricRecord`
- `DriftEventRecord`
- `FairnessSnapshotRecord`
- `ExplainabilitySnapshotRecord`
- `SustainabilitySnapshotRecord`

Important research-integrity rules are encoded in validation:

1. `not_measured` sustainability values use `None`, never decorative zeroes.
2. insufficient fairness support uses `None`, never a perfect fairness gap of `0.0`.
3. accuracy/F1/fairness quality values are validated in `[0, 1]`.
4. run IDs include dataset, framework, seed, protocol and a collision-resistant digest.

## HPC-safe storage contract

Each independent task owns one directory:

```text
artifacts/meta_experiments/<experiment_id>/
├── execution_manifest.json
├── run.json
├── windows.jsonl
├── drift_events.jsonl
├── fairness.jsonl
├── explainability.jsonl
└── sustainability.jsonl
```

No worker appends to a global meta-log. A later reducer validates and consolidates complete experiments into a versioned recommender snapshot.

This makes the future Slurm layout naturally map to:

```text
dataset × framework × seed × protocol
                  ↓
             one task
                  ↓
     one immutable run directory
```

## Compatibility with the current V4 recommender

`data/meta/active_snapshot.txt` currently points to the frozen V1 legacy evidence. The historical recommender checks this pointer before the compatibility `data/meta/meta_logs.json` path. Future V2 snapshots can be activated without overwriting V1.

## Validation commands

```powershell
python scripts\validate_meta_foundation.py
python scripts\audit_meta_foundation.py
python -m pytest tests -v
```

Expected foundation facts:

- 47 frozen meta-training datasets
- 234 historical run rows
- AutoStreamML/AutoClass/EvoAutoML/ChaCha: 47 historical rows each
- OAML: 46 historical rows
- V1 held-out target count: 23, names still to be registered
- no train/test dataset overlap in the registered manifests

## Next phase

Phase 2 instruments all five framework runners so each run emits this schema using true prequential Accuracy/Macro-F1, drift/recovery, temporal fairness, explainability snapshots and measured sustainability status. Only after that instrumentation is verified locally should the new 47-dataset meta-run be launched on HPC.
