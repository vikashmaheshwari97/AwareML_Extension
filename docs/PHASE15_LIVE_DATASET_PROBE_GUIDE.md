# Phase 15 · Live Dataset Probe user guide

The Live Dataset Probe is exploratory and is not frozen journal evidence.

## Recommended workflow

1. Run **Quick check · all four live sources**.
2. Keep **Include one default faithfulness intervention for each source** enabled.
3. Read the simple source status table.
4. Open only sources marked `REVIEW` or `FAILED`.
5. Use **Inspect one source · optional drill-down** only when you want a deeper view or a different counterfactual.

## Explanation source meanings

- `B · Stage B · setup / recommendation`: pre-run recommendation evidence.
- `E · Stage E · fairness`: post-run fairness and calibration-gap evidence.
- `F_XAI · Stage F · XAI / feature attribution`: post-run attribution evidence. Generic feature importance is not relabelled as exact SHAP when literal SHAP values are unavailable.
- `F_CHAT · Stage F · conversational answer`: post-run answer grounded in active framework results/ranking.

## Explanation to verify

No typing is required in the normal workflow.

- **Generate with Ollama · recommended**: AwareML creates the explanation and verifies it.
- **Paste custom explanation · optional**: use this only to test an explanation you wrote or obtained elsewhere.

If the batch already generated the selected source, the single-source drill-down automatically reuses that explanation.

## Faithfulness

If the batch checkbox is enabled, one faithfulness intervention is already run for every source.

The manual faithfulness panel is therefore optional. Use it when you want to:
- change a different evidence key;
- choose a different counterfactual value;
- inspect the original/counterfactual text in detail.

Desired live behavior:
- changed evidence acknowledged = 1;
- numeric update accuracy = 1 when applicable;
- stale-claim rate = 0.

For Stage B and Stage F conversational evidence, the live intervention changes one evidence field but does not recompute the complete ranking/utility. Treat it as claim-level sensitivity rather than formal decision-flip evidence.
