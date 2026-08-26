# Phase 7 change set

## Added

- `awareml/llm/client.py`
- `awareml/llm/schemas.py`
- `awareml/llm/goal_parser.py`
- `awareml/llm/config_diff.py`
- `awareml/llm/review.py`
- `awareml/llm/evidence.py`
- `awareml/llm/grounded_copilot.py`
- `awareml/llm/configuration.py`
- `awareml/llm/copilot.py`
- Phase-7 validation/freeze/orchestration scripts
- Phase-7 tests and documentation

## Updated

- `awareml/llm/__init__.py`

## Intentionally preserved

- legacy `parse_objective_text`
- legacy `GroundedChat`
- frozen Phase-6 recommender artifacts
- `data/meta/active_snapshot.txt`
- the 23-dataset held-out evaluation split
- UI code (Phase 9)

## Phase boundary

Formal counterfactual explanation fidelity and attribution-guidance research are intentionally deferred to Phase 8.
