# Phase 11 — Copilot Objective-Selection V2

## Purpose

Phase 11 realigns the Copilot with the journal roadmap and reviewer concern 1.1.

The old Copilot directly produced four continuous weights. Phase 11 separates
the task into two explicit problems:

**Problem A**
Natural-language scenario -> selected objective set.

**Problem B**
Selected objective set -> documented recommender weights.

The fixed journal objective vocabulary remains the Phase-10 lock:

- Accuracy
- Runtime
- Energy
- CO2

Fairness, explainability, and drift remain HCAI requirements and do not become
hidden primary-objective labels.

## Architecture

Scenario
-> frozen LLaMA 3 8B objective selector
-> selected objective set
-> equal_selected_v1 weighting
-> frozen ML Recommender V2
-> framework ranking
-> grounded Copilot explanation
-> human review

The LLM does not directly choose a framework.

## New objective-selection output

The primary output is an explicit subset such as:

```json
{
  "status": "valid",
  "selected_objectives": ["Accuracy", "Energy", "CO2"],
  "uncertainties": []
}
```

GoalInterpretation stores the selection status, selected objectives, source,
journal model, fallback status, weighting policy, generated four-objective
weights, and HCAI requirements.

## Equal-selected weighting policy

Phase 11 freezes:

`equal_selected_v1`

Example:

Selected:
- Accuracy
- Energy
- CO2

Weights:
- accuracy = 1/3
- runtime = 0
- energy = 1/3
- co2 = 1/3

This is intentionally simple and reproducible. Any later intensity-aware
weighting must be evaluated as a separate experiment rather than silently
changing the Phase-12 objective-selection benchmark.

## Failure handling

Five statuses are surfaced by the Phase-11 layer:

- valid
- ambiguous
- malformed
- out_of_scope
- contradictory

Malformed LLM JSON is never silently accepted.

For the interactive Copilot, a malformed LLM response may use the transparent
deterministic selector only when that fallback is explicitly labelled in the
interpretation and warnings.

Wrong model tag, wrong model digest, or wrong Ollama version always causes a
hard journal-model-lock failure.

Out-of-scope and contradictory scenarios do not silently become framework
recommendations.

## Exact journal LLM

Phase 11 consumes Journal Experimental Protocol v1 rather than creating a new
model configuration.

It therefore inherits:

- Ollama
- exact tag `llama3:8b`
- frozen model digest
- frozen Ollama version
- frozen prompt SHA256
- frozen JSON-schema SHA256
- temperature 0.0
- top_p 1.0
- seed 42
- JSON mode
- no silent model fallback

## HCAI boundary

The natural-language goal may also contain requirements such as:

- high drift sensitivity
- fairness auditing
- explainability
- runtime budget

These are preserved in `hcai_requirements`.

They are not counted as hidden labels in the four-objective journal benchmark.

## Phase-12 boundary

Phase 11 does not create or score the human-annotated NL benchmark.

Phase 12 will evaluate only Problem A using:
- per-objective precision/recall/F1
- micro/macro F1
- exact-match rate
- k'=1/2/3/4 breakdown
- paraphrase stability
- failure taxonomy

The 705 Meta-Dataset V2 framework runs remain recommender evidence and are not
Copilot objective-selection training examples.
