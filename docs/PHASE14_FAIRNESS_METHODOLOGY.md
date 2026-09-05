# Phase 14 — Fairness methodology

## Existing hard-label criteria retained

AwareML reports:

- demographic-parity difference;
- equal-opportunity difference;
- equalized-odds gap;
- predictive-parity difference;
- error-rate gap;
- worst-group accuracy;
- worst-group Macro-F1.

Undefined or under-supported quantities remain unavailable (`None`/N/A).

## Calibration fairness

When valid prequential probabilities are exposed, AwareML additionally reports:

- Group Brier-score gap;
- Group Expected Calibration Error (ECE) gap.

Both are max-minus-min group disparities. Lower is better.

## Degenerate-prediction safeguard

A zero parity gap is not automatically evidence of a useful fair classifier.
If hard predictions are constant or near-constant, the raw gap is preserved for
auditability but the framework is excluded from the Fairness Lab's
"fairness-winner" claim.

This safeguard changes interpretation only; it does not alter the recorded
fairness values.

Likewise, a constant probability can mechanically produce a zero Group
Brier-score gap. The UI therefore exposes probability-behaviour diagnostics
and Group ECE alongside Brier disparity.

## Missing-value policy

Unavailable probabilities or undefined group rates remain N/A.
AwareML never substitutes zero for unavailable fairness evidence.
