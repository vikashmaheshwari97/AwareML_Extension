# Phase 12 Changelog

## Added
- 120-scenario external stronger-model candidate pool.
- k'=1–4 balanced design.
- blinded human realism-filter workflow.
- optional genuine human-written scenario intake.
- 2–3 independent multilabel annotation workflow.
- Fleiss kappa and set-agreement analysis.
- majority-vote ground truth with explicit hard-case handling.
- strict Phase-10/11 `llama3:8b` evaluation runner.
- per-objective precision/recall/F1.
- micro-F1, macro-F1, exact-match, k' breakdown.
- 10 x 5 paraphrase robustness study.
- 14-case adversarial failure taxonomy.
- pre-evaluation design freeze.
- pre-LLM human ground-truth freeze.
- final `objective_selection_benchmark_v1` freeze + SHA256 validation.

## Explicit non-changes
- Phase-11 selector logic is not modified.
- ML Recommender V2 is not modified.
- 705 meta-logs are not used as Copilot training data.
- 23 held-out dataset contents are not read.
