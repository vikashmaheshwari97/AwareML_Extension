# Current repository audit and upgrade map

Audit basis: `vikashmaheshwari97/AwareML_Extension`, `main`, inspected 2026-09-14. GitHub HEAD at packaging time: `e4203972b79ec949b26898883ab2e525448cfa2a`.

## Existing state retained

- Fixed objective vocabulary is exactly `Accuracy`, `Runtime`, `Energy`, `CO2`.
- Phase-11 V2 already separates scenario-to-objective selection from `equal_selected_v1` weighting.
- Exact journal runtime remains `llama3:8b` through the strict journal client; wrong model/digest/Ollama version is a hard failure.
- Legacy Phase-12-v1 is frozen and remains untouched. Its 55-case result showed high recall but low precision/low exact match and no human-majority k'=4 cases.
- V3.1 is explicitly post-hoc development logic and its old-benchmark perfect diagnostic is not an independent journal test.
- `equal_selected_v1` already implements Step 8: selected labels receive 1/N and unselected labels zero.
- `epsilon_pareto_v1` already implements Step 9 with epsilon 0.05 and robust 5th/95th percentile normalization.

## Every recommended change implemented in this overlay

1. **Legacy v1 immutable:** v2 freeze gates verify the old v1 manifest against its checksum and never write into its directory.
2. **Phase-11 revision first:** V3.2 has a separate prompt/schema/source module plus a freeze script that records exact source hashes and strict LLaMA runtime before v2 design/annotation can freeze.
3. **Old Phase-12 is development only:** V3.2 lineage explicitly records this, and v2 candidate validation rejects exact/near-copy reuse of legacy primary/design scenarios.
4. **Fresh 150-case generated pool:** protocol fixes 150 external-LLM candidates over five independent batches with k' intent 25/30/45/50, oversampling k'=3/4. Intent stays private and never becomes truth.
5. **More human data:** at least 30 raw human-written candidates are required; the final 60 requires at least 24 human-written cases.
6. **Final 60 from humans only:** three independent annotators, agreement calculation, hard-case filtering, then exactly 15 human-confirmed k'=1/2/3/4 cases. Generator intent/model correctness cannot be used in selection.
7. **Paired comparison:** frozen V2 and frozen V3.2 run on the same immutable 60 scenarios.
8. **Expanded metrics:** per-label P/R/F1, micro/macro F1, exact match, Jaccard, Hamming, over/under-selection, strict superset/subset, malformed/status rates, k' breakdown, 2,000-sample bootstrap 95% CIs, paired exact McNemar.
9. **Fresh robustness tests:** 10 distinct-ground-truth bases × 5 fresh paraphrases with two human reviewers; fresh 10–15 adversarial cases with expected status frozen before evaluation.
10. **V3.2 concept guard:** compact semantic concepts replace the large phrase catalogue. In benchmark mode the guard only rejects unsupported additions; it never recovers omitted labels. Exact grounded evidence is mandatory.
11. **Malformed behavior:** benchmark malformed remains malformed with no semantic/deterministic rescue. Interactive recovery exists separately and is explicitly audit-labelled.
12. **Hash/provenance closure:** staged manifests freeze design, ground truth, primary set, paraphrase set, adversarial set and final results with SHA-256 protection.

## Important scientific interpretation

A higher v2 score is only credible if the final cases are selected without looking at V3.2 outputs. The code therefore treats “not enough human-confirmed k'=4 cases” as a data-collection failure, not permission to substitute private generator intent or choose easier cases.
