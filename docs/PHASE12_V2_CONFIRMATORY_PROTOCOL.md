# AwareML Phase 11R / Phase 12 v2 confirmatory protocol

This upgrade preserves the frozen Phase-12-v1 artifact as the historical baseline and adds a new confirmatory line. It does **not** rewrite old results.

## What changes

**Phase 11R — Objective Selection V3.2 confirmatory freeze.** The exact locked `llama3:8b` remains the language model. The V3.2 prompt requires a separate decision and exact scenario-local evidence quote for every label. A compact concept-level guard validates the evidence. In benchmark mode the guard may reject unsupported LLM additions but never add an omitted label. Malformed JSON remains `malformed`; wrong model/digest remains a hard failure. Interactive semantic recovery exists only behind an explicit non-benchmark option and is audit-labelled.

**Phase 12 v2 — fresh paired benchmark.** Generate 150 new external-LLM candidates in five independent batches (25/30/45/50 intended k'=1/2/3/4), collect at least 30 genuine human-written candidates, realism-filter them without model outputs, annotate with three independent humans, compute agreement, and freeze ground truth. Final k' comes only from human majority labels.

The primary test is exactly 60 non-hard scenarios, 15 at each human-confirmed k'=1,2,3,4, with at least 24 human-written cases. Deterministic diversity selection uses objective-set/domain/style/source metadata only and runs before any evaluated-model output exists. It never looks at V2/V3.2 correctness.

Both the frozen Phase-11 V2 selector and frozen V3.2 selector run on the **same untouched 60 cases**. Primary metrics are per-objective precision/recall/F1, micro/macro F1, exact-set match, Jaccard, Hamming loss, over-selection, under-selection, malformed/status rates, k' breakdown, 2,000-sample bootstrap 95% CIs, and paired exact McNemar testing.

The paraphrase test uses 10 bases with distinct human ground-truth sets × 5 fresh externally generated paraphrases. Two humans review every variant before evaluation. Report correctness against human ground truth and prediction consistency separately.

The adversarial test uses 10–15 completely new cases with expected rejection status frozen before model execution. Report status accuracy, invalid non-empty selection rate, malformed failures, and a qualitative taxonomy.

## Why V3.1 results are not the final claim
V3.1 was developed after examining the old Phase-12-v1 errors. Its perfect post-hoc score on those cases is development evidence only. V3.2 is frozen before the new v2 human labels/outcomes are available, and the final journal claim comes from the fresh paired test.

## Existing protocol points retained unchanged

**Weighting:** `equal_selected_v1`: selected objectives receive `1/N`; unselected objectives receive zero.

**Near-Pareto:** `epsilon_pareto_v1`, epsilon = 0.05. Objectives are robustly normalized with 5th/95th percentile clipping to [0,1], minimize objectives are inverted, and a candidate is near-Pareto when no other candidate epsilon-dominates it under the existing formal rule in `awareml/engine/pareto_spec.py`.

## Anti-leakage gates enforced in code
1. Frozen Phase-12-v1 checksum must still validate.
2. V3.2 must be frozen before the v2 design can be frozen.
3. Fresh primary candidates cannot exactly/near-copy legacy primary/design scenarios.
4. Generator intent is kept in a private file and excluded from annotation packets.
5. Evaluated-model outputs are forbidden before design, ground truth, primary, paraphrase and adversarial freezes.
6. Final primary selection is based only on human labels and diversity metadata.
7. All evaluation input sets must be frozen before the first selector run.
8. Outputs are immutable by default; the run commands refuse to overwrite existing confirmatory outputs.
9. Final manifests record SHA-256 hashes for protected inputs, outputs, selector freezes and the legacy benchmark.
