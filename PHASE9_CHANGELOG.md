# Phase 9 change set

## Updated
- `app.py`

## Added
- modular `awareml/ui_v2/` Research OS package
- Phase-9 validation/freeze scripts
- Phase-9 unit tests and documentation

## Signature additions
- Research Command Center
- interactive Plotly 3D Decision Space
- persistent global research state
- preference what-if ranking
- streaming observatory
- Responsible AI observatory
- Copilot + human review workspace
- Faithfulness Lab
- reproducibility/export center
- legacy specialist-lab bridge

## Research safeguards
- no decorative fake results
- no held-out 23-dataset use
- pre-run predictions labeled as predictions
- Phase-8 AEF labeled as development evidence
- no raw dataset rows in export bundles


## UI polish hotfix
- added appearance selector (System / Dark / Light)
- improved global contrast and readability
- redesigned Plotly chart palette and panel styling
- added stronger drift, refit, and adaptation markers in Streaming Observatory
- added user manual for demo / supervisor walkthrough

## Phase 9.2 clarity hotfix
- friendly evidence labels with raw IDs hidden in technical expanders
- deterministic Copilot goal parser is now the recommended/default demo path
- Copilot displays parsed objective weights and HCAI requirements explicitly
- Faithfulness Lab is clearly labeled as frozen Phase-8 development evidence
- Legacy Recommender Lab is renamed and marked non-authoritative
- Explainability Lab receives a clear degenerate-XAI interpretation notice


## Phase 9.3 generic dataset advisor
- removed Dutch-Census-specific setup panel from Run Studio
- added dataset-specific setup assistant for any uploaded/selected classification dataset
- suggests target candidates without auto-selecting them
- suggests possible fairness-audit attributes but never auto-enables them
- reports possible stream-order/time columns
- adds schema-level leakage and identifier warnings
- keeps Dutch Census only as an optional example dataset, not a UI assumption


## Phase 9.4 research-clarity and visualization hotfix
- upgraded temporal observatory plots with clearer drift/refit event rendering, stronger padding, and reduced annotation overlap
- added temporal-plot interpretation guidance and run-summary KPI cards
- improved 3D Decision Space explanation of point predictions vs conservative 90% bounds
- clarified that preference changes rerank frozen predictions rather than rerunning AutoML
- upgraded 3D chart scene ranges, margins, and captions for a more research-grade presentation
- cleaned Copilot rationale text so raw evidence IDs are replaced by human-readable labels
- added explicit Copilot vs ML Recommender explanation to reduce confusion when recommendations differ
- improved Faithfulness Lab messaging about frozen benchmark evidence vs future optional current-dataset probes
- expanded Advanced Labs guidance for Decision Lab, Trust Calibration, and Information-Seeking Lab


## Phase 9.5 final research-UI polish
- added native Phase-9 specialist pages for Decision, Drift, Fairness, Explainability, Sustainability, Trust and Information-Seeking
- removed dependence on older specialist plot layouts for the main Advanced Labs surfaces
- Decision Lab is now explicitly post-run observed ranking with utility-contribution and objective-correlation views
- temporal plots now use an event-strip design for drift/refit markers to avoid label overlap and clipping
- fairness temporal plots reuse the synchronized drift/refit event semantics and add worst-window summaries
- sustainability charts use horizontal scientific-notation bars and bounded margins to avoid plot overflow
- explainability now separates benchmark validity from XAI availability and shows prediction diagnostics for degenerate explanation cases
- Trust Calibration and Information-Seeking pages now explain their HCAI purpose directly in the UI
- 3D Decision Space adds raw vs normalized-desirability views and exact conservative-bound semantics
- current-dataset faithfulness recomputation is deliberately not mixed into the frozen Phase-8 benchmark; guidance for a future live probe is documented
