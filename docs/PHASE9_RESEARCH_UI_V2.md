# Phase 9 — Research UI V2

Phase 9 turns the backend work from Phases 0–8 into one coherent Research OS.

## Signature experience

The central visualization is an interactive Plotly 3D multi-objective decision space:

- X = predicted Accuracy ↑
- Y = predicted Runtime ↓
- Z = predicted Energy ↓
- marker size = predicted CO₂
- marker tone = preference utility
- green outline = Pareto efficient
- white outline = inspected framework

The 3D scene uses only frozen Phase-6 predictions. It never invents decorative benchmark values.

## Workspaces

1. Command Center
2. Run Studio
3. 3D Decision Space
4. Streaming Observatory
5. Responsible AI
6. Copilot Workspace
7. Faithfulness Lab
8. Export Center
9. Advanced Labs

## Global state

The same `awareml_state` object is reused across the whole interface. Phase 9 extends it with the active Phase-6 profile/ranking, selected framework, user preference weights, ranking mode, Copilot proposal/evidence, and human review.

## Research integrity

Phase 9 does not:

- invent benchmark results for visual polish,
- use the frozen 23-dataset held-out test split,
- label Phase-6 predictions as observed outcomes,
- label Phase-8 AEF as a current-run metric,
- send raw dataset rows to the LLM/export bundle.

Phase 10 remains the frozen technical + HCAI journal evaluation.
