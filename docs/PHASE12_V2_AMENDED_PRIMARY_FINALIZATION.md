# Phase-12-v2 Amended Primary Finalizer

This overlay adds a separate finalization script. It does NOT replace `scripts/phase12_v2.py`.

Purpose:
- combine the original 170 eligible Phase-12-v2 ground-truth cases with the 14 eligible H041-H055 amendment cases;
- preserve the already-frozen original design and ground-truth manifests;
- select exactly 15 cases for each human-confirmed k'=1,2,3,4;
- maximize eligible human-written inclusion inside each fixed k' stratum;
- require >=24 human-written cases;
- use a deterministic SHA-256 scenario-ID tie-break with seed 20260914;
- never read model outputs or private generator intent;
- write a NEW `primary_benchmark_amended.csv`;
- write a NEW `amended_primary_manifest.json` and SHA256 sidecar.

Expected current pool:
- original eligible: 170
- amendment eligible: 14
- combined eligible: 184
- combined eligible human: k'=1:24, k'=2:11, k'=3:0, k'=4:0
- final primary: 60 cases, 15 per k'
- expected human inclusion under this policy: 26

Run from repository root:

    python .\scripts\phase12_v2_finalize_amended_primary.py validate

Inspect the JSON. Only if PASS:

    python .\scripts\phase12_v2_finalize_amended_primary.py freeze

The freeze command refuses to overwrite an existing amended-primary freeze.

It writes:
- `data/journal/objective_selection_benchmark_v2/amendments/human_enrichment_H041_H055/...`
- `data/journal/objective_selection_benchmark_v2/results/primary_benchmark_amended.csv`
- `data/journal/objective_selection_benchmark_v2/frozen/amended_primary_manifest.json`
- `data/journal/objective_selection_benchmark_v2/frozen/amended_primary_manifest.json.sha256`

After freezing, inspect Git status and stage only research artifacts explicitly. Do not use `git add .`.

Do NOT run V2/V3.2 confirmatory evaluation until the amended-primary freeze is verified.
