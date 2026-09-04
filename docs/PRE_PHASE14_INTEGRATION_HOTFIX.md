# Pre-Phase-14 Integration / HCAI Hotfix

This patch corrects UI and integration issues discovered after the frozen Phase-12 and Phase-13 validations. It does **not** reopen, regenerate, or edit either frozen benchmark.

## Changes

1. Temporal Fairness maps Equal Opportunity to `equal_opportunity_diff`.
2. Fairness Lab adds a complete per-framework fairness table with explicit `N/A` values.
3. Composite fairness is cross-framework comparable: `1 - mean(common disparity criteria)` where the common criterion set is the intersection available for every valid framework result.
4. Decision Lab labels the composite score explicitly and hides internal `pareto_efficient` / spec fields from the journal-facing table.
5. Information-Seeking passes the classified follow-up intent to GroundedChat and deterministic responses now distinguish evidence, comparison, challenge, fairness, drift, XAI, sustainability, and recommendation questions.
6. Copilot reads frozen Phase-12 reliability evidence and shows it as aggregate calibration evidence, not current-sentence confidence.
7. Trust Calibration defaults to blinded participant mode. Manual reliability conditions are exposed only when the researcher explicitly launches Streamlit with `AWAREML_STUDY_RESEARCHER_MODE=1`.
8. Journal-facing Pareto language is normalized to canonical ε-Pareto terminology.

## Frozen-artifact rule

The installer hashes the Phase-12 and Phase-13 frozen manifests before and after installation and aborts if either changes.
