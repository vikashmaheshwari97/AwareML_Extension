# Phase 10 Changelog — Journal Protocol Lock

## Added

- Exact pre-Phase-10 engineering baseline lock.
- Baseline artifact SHA256 capture for Phases 5–9.
- Canonical 47-development / 23-held-out role resolution.
- Explicit block on final held-out dataset access before Phase 18.
- Frozen four-label objective vocabulary:
  - Accuracy
  - Runtime
  - Energy
  - CO2
- Frozen journal objective-selection JSON schema.
- Frozen journal objective-selection prompt template.
- Exact LLaMA 3 8B / Ollama runtime contract.
- Exact model-tag and model-digest verification.
- Ollama version capture.
- No-fallback journal-model policy.
- Journal Experimental Protocol v1 freeze/validation scripts.
- Phase-10 unit tests and protocol documentation.

## Intentionally unchanged

- Recommender V2
- 705 Meta-Dataset V2 runs
- Copilot v1 runtime behavior
- Faithfulness v1
- Research UI V2
- final 23 held-out dataset contents

## Deferred by design

- objective subset -> weight policy: Phase 11
- canonical near-Pareto journal definition: Phase 13
- held-out dataset execution: Phase 18
