# AwareML Phase-18 Runtime Recovery Hotfix v1

Additive recovery only: no frozen/checksummed Phase-18 source file is modified.

Fixes:
- EvOAutoML 0.0.14 categorical-string labels are encoded to internal integer IDs and decoded before AwareML metrics.
- ExperimentStore nested NumPy/pandas values are normalized for JSON/JSONL persistence.
- Missing sustainability measurements are detected from result.json and rerun serially (%1) to avoid CodeCarbon node-local lock contention.

The recovery script verifies the original frozen checksums before preparing or submitting any rerun and writes a recovery manifest with the frozen and hotfix Git heads.
