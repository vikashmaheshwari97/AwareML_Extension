# Phase 8 change set

## Added

- `awareml/faithfulness/__init__.py`
- `awareml/faithfulness/schemas.py`
- `awareml/faithfulness/metrics.py`
- `awareml/faithfulness/attribution.py`
- `awareml/faithfulness/counterfactuals.py`
- `awareml/faithfulness/rationale.py`
- `awareml/faithfulness/evaluator.py`
- Phase-8 run/validate/freeze/complete scripts
- unit tests
- research documentation

## Scientific safeguards

- FaithLM/Serum are treated as inspiration, not copied implementations.
- AwareML Evidence Fidelity is explicitly project-defined.
- external objective influence is not mislabeled as internal LLM attribution.
- no PE-LRP or Ollama attention intervention is claimed.
- raw dataset rows are not required.
- the 23-dataset held-out evaluation split remains untouched.

## Phase boundary

Phase 8 establishes the faithfulness methodology and development evidence.
Phase 10 performs the frozen held-out journal evaluation.
