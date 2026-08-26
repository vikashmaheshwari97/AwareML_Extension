# Phase 7 — Human-Centric LLM Copilot

Phase 7 implements the HCAI layer described in the AwareML Extension plan.

## Scientific separation

The two recommenders remain different.

### LLM-Assisted Copilot

The Copilot interprets a natural-language goal, turns it into a reviewable configuration, explains measured evidence, supports before/during/after-run questions, and records human edits.

It does **not** train the empirical framework-performance models.

### ML Recommender V2

The frozen Phase-6 component predicts:

- Accuracy ↑
- Runtime ↓
- Energy ↓
- CO₂ ↓

for all five frameworks and performs the preference-aware empirical ranking.

Fairness and explainability remain HCAI requirements/evidence rather than being silently inserted into the four-objective utility.

## Before run

Flow:

Human goal
→ GoalParser
→ four primary preferences + HCAI requirements
→ frozen Phase-6 V2 predictions
→ framework ranking + uncertainty
→ configuration synthesis
→ grounded rationale
→ human review
→ configuration diff
→ approve / approve-with-edits / reject

The configuration is never considered approved automatically.

## During run

The Copilot can receive a `FrameworkResult` or equivalent dictionary and construct a compact evidence bundle containing:

- current accuracy/F1
- rolling metrics when available
- drift count and latest drift event
- drift recovery summary
- fairness status/metrics
- supported XAI summary and top features
- energy and CO₂ evidence

Raw dataset rows are excluded.

## After run

The Copilot can compare completed framework results using observed:

- accuracy
- macro-F1
- runtime
- energy
- CO₂
- drift/fairness/XAI summaries

Every grounded LLM answer must use valid `[evidence....]` keys. Unsupported or uncited LLM output falls back to a deterministic evidence answer.

## Preference counterfactual

`CopilotService.what_if_weights()` can answer preference questions such as:

> What if I care much more about energy?

It reranks the already-predicted framework outcomes. It does not rerun AutoML.

## Privacy

The local LLM receives only:

- the user's natural-language goal
- dataset meta-features
- framework predictions/results
- compact responsible-AI evidence

Raw dataset rows and participant rows are not placed in the evidence prompt.

## Ollama

The implementation is local-first through Ollama. If Ollama is unavailable or produces invalid JSON/evidence citations, deterministic fallbacks preserve functionality and expose the fallback source.

## Phase 8 boundary

Phase 7 provides grounding and evidence sensitivity primitives. Formal counterfactual explanation-fidelity experiments remain Phase 8.
