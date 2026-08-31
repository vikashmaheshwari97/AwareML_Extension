# Phase 11 Changelog — Copilot Objective-Selection V2

## Added
- explicit scenario-to-objective-set schema
- fixed five-state failure handling
- separate HCAI requirements structure
- strict Phase-10 journal Ollama client
- exact tag/digest/version enforcement
- strict malformed-JSON behavior
- equal_selected_v1 weighting policy
- deterministic transparent interactive fallback
- updated Copilot metadata showing selection before weighting
- updated Copilot UI showing selected objectives before framework ranking
- Phase-11 validation/freeze/completion scripts
- Phase-11 unit tests and Phase-7 compatibility tests

## Changed
- Copilot no longer conceptually treats direct four-weight generation as the
  primary natural-language task.
- The framework recommender receives weights only after objective-set inference.

## Unchanged
- Phase-6 frozen Recommender V2 models
- 705 Meta-Dataset V2 framework runs
- 23 held-out dataset contents
- Phase-10 journal protocol
- human review requirement
- grounded evidence architecture

## Deferred
- human-annotated NL objective-selection benchmark -> Phase 12
- alternate/intensity-aware weighting policies -> separate future experiment
