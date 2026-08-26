# Phase 8 — Faithfulness Layer

Phase 8 implements the practical, model-agnostic faithfulness layer planned for the AwareML Extension.

## Research motivation

Natural-language explanations can be plausible without being sensitive to the evidence that actually caused a recommendation.

AwareML therefore evaluates whether recommendation rationales respond appropriately when the recommendation evidence is deliberately changed.

The design is inspired by the counterfactual-evidence idea described in the project plan for FaithLM and by the broader rationale/evidence-alignment concern discussed for Faithfulness Serum.

## Important scientific boundary

This implementation **does not** claim to reproduce FaithLM or Faithfulness Serum.

It does **not** claim access to Ollama:

- attention tensors,
- hidden-state causal traces,
- PE-LRP,
- internal token attribution,
- attention-level interventions.

Instead it implements the practical AwareML version:

1. external evidence interventions,
2. recommendation reranking,
3. original vs counterfactual rationale generation,
4. evidence-citation validation,
5. decision/rationale alignment,
6. external objective-influence analysis,
7. irrelevant-evidence invariance.

A future GPU/Transformers experiment can add internal attribution guidance without changing this external evaluation protocol.

## Counterfactual evidence tests

For each development dataset, AwareML creates:

- accuracy evidence flip,
- runtime evidence flip,
- energy evidence flip,
- CO₂ evidence flip,
- joint sustainability evidence flip.

The intervention degrades the current winner on the selected objective and improves the runner-up, then reranks the same five candidate frameworks.

No AutoML framework is rerun.

## Decision influence

For each of the four objectives, the system neutralizes its cross-framework signal and observes how the recommendation changes.

This creates a normalized external objective-influence profile:

- accuracy influence,
- runtime influence,
- energy influence,
- CO₂ influence.

The textual rationale is then checked to see whether it cites the most decision-influential objective evidence.

This is called **external evidence-attribution alignment**. It must not be described as internal LLM attribution.

## AwareML Evidence Fidelity (AEF)

AEF is a project-defined transparent composite:

- 25% grounding validity,
- 20% decision/rationale alignment,
- 20% external evidence-attribution alignment,
- 25% counterfactual sensitivity,
- 10% irrelevant-evidence invariance.

AEF is **not** claimed to be a metric from FaithLM or Faithfulness Serum.

All components are retained separately and should be reported alongside the composite.

## Evaluation protocol

The deterministic faithful baseline is evaluated over all 47 frozen development/meta datasets.

When Ollama is enabled, a deterministic representative subset of development datasets is also evaluated using the live local model.

The frozen 23 held-out evaluation datasets are not used in Phase 8.

The larger held-out journal evaluation remains Phase 10.

## Output

Phase 8 writes:

- deterministic case-level Parquet,
- deterministic summary JSON,
- live Ollama case-level Parquet,
- live Ollama summary JSON,
- Phase-8 faithfulness report,
- SHA256 sidecars,
- frozen Phase-8 release manifest.
