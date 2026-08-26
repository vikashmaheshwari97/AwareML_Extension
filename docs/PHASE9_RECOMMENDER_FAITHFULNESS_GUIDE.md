# Phase 9 Recommendation, Evidence and Faithfulness Guide

## Which recommender is authoritative?

For the current AwareML Extension:

1. **3D Decision Space / Phase-6 ML Recommender V2** is the authoritative pre-run framework recommender.
2. **Copilot Workspace / Phase 7** interprets the human goal, obtains four objective weights, calls the Phase-6 recommender, proposes a configuration, and requires human review.
3. **Legacy Recommender Lab** is retained only as the original-paper / historical baseline and may produce different rankings.

## Why can Ollama change the selected framework?

Ollama does not directly choose a framework. When enabled as the goal parser, it can interpret the same sentence into different weights for:

- Accuracy
- Runtime
- Energy
- CO2

The Phase-6 ML recommender then uses those weights. Therefore different parsed weights can change the ranking.

For reproducible demonstrations, use the deterministic parser.

## Evidence IDs

A string such as:

`evidence.before.candidates.EvoAutoML.accuracy`

is a machine-readable provenance key. It means:

- `evidence` — this is a grounded evidence item
- `before` — pre-run stage
- `candidates` — candidate framework table
- `EvoAutoML` — framework
- `accuracy` — predicted accuracy

These IDs are used to audit LLM claims and to compute faithfulness metrics.

## Energy/CO2 correlation

Spearman rho = 1.000 means the two objectives have exactly the same rank ordering across the relevant candidate/evaluation set.

That is not a performance score. It means the two sustainability signals are redundant in that context. Assigning both large independent weights can double-count essentially the same preference.

## Faithfulness Lab

The Phase-8 Faithfulness Lab is a development benchmark for explanation behavior. It is not automatically the current uploaded dataset.

`accuracy_evidence_flip` means:

1. start with the original pre-run recommendation,
2. deliberately modify accuracy evidence,
3. rerank the candidates,
4. regenerate the rationale,
5. check whether the explanation changes in the correct direction.

A faithful explanation should acknowledge the changed evidence and the new winner when the decision flips.

## Explainability Lab

`failed_or_degenerate` means the requested XAI method returned an unusable explanation signal, such as all-zero or non-finite feature importance.

AwareML intentionally preserves the model result but refuses to present the degenerate explanation as valid.
