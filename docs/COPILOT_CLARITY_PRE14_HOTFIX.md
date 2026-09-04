# AwareML Copilot clarity hotfix before Phase 14

This incremental hotfix is applied after the final pre-Phase-14 usability/research-safety upgrade.

It does **not** modify Phase-12, Phase-13, or the V3.1 objective selector.

## Fixes

- removes the Streamlit nested-expander exception;
- Simple View no longer embeds the research renderer inside another expander;
- Research View preserves the full frozen Phase-12/V3/V3.1 diagnostics;
- dataset-aware Simple View follows a 1→2→3→4→5 workflow;
- `Why this framework?` becomes a deterministic, beginner-facing explanation from ML Recommender V2 predictions and current objective weights;
- raw evidence IDs are omitted from beginner prose but remain available in Research View;
- algorithm is shown in full in the plan table;
- `Fairness: disabled` is replaced by `Fairness constraint: Not requested by this goal`, with a separate statement explaining that post-run fairness auditing is still available when a sensitive attribute is configured;
- Simple View gets a persistent final plan approve / approve-with-edits / reject workflow;
- detailed technical proposal remains available in Research View instead of crowding Simple View.
