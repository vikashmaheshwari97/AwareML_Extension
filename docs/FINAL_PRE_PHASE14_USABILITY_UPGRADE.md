# AwareML final pre-Phase-14 usability and research-safety upgrade

This patch does **not** change the frozen Phase-12 v1 benchmark, the frozen Phase-13 recommender validation, or the V3.1 objective selector. It upgrades the interaction layer around those artifacts.

## Copilot
- Default Simple View plus optional Research View.
- Beginner-friendly Copilot AutoML Plan summary: objectives, framework, algorithm, key configuration, why, and provenance.
- Renames preference utility so users do not confuse a normalized score with confidence.
- Corrects pre-run rationale language from “empirical” to “predicted”.
- Persists objective reviews to `artifacts/copilot/objective_reviews.jsonl`.
- Human objective correction can rerank ML Recommender V2 immediately without asking LLaMA again.
- Keeps all V1/V3/V3.1 research diagnostics under progressive disclosure.

## Dataset/fairness safeguards
- High-cardinality numeric target warning for classification misuse.
- Multiclass positive-label fairness is explicitly described as one-vs-rest.
- Sensitive-group N, positive N, positive rate and weak-support warnings.
- Zero/small disparity should not be interpreted as meaningful fairness when support is weak.

## 3D Decision Space
- Preference source toggle: use Copilot priorities or manual what-if priorities.
- Single-click framework inspection pills.
- Clear Recommended vs Currently Inspecting cards.
- Utility is explicitly not confidence.
- Pre-run predictions remain separate from observed post-run evidence.

## Decision Lab
- Explains why the weighted-utility winner can differ from the accuracy winner.
- Shows common-criteria fairness coverage.
- Explains Near-Pareto in plain language.
- Adds pre-run-vs-post-run recommendation comparison and predicted-vs-observed metric errors when available.

## Streaming Observatory
- Renames the total to framework-level drift alerts.
- Selected-framework, all-framework, and clustered-drift-episode views.
- Display-only drift clustering; raw events are never changed.
- Drift, explicit adaptation/refit, and observed recovery are kept semantically separate.
- Per-framework alert rate, adaptation count and median recovery lag summary.
