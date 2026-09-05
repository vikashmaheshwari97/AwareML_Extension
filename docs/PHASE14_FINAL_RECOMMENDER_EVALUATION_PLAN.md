# Phase 14 — Final held-out ML Recommender V2 evaluation

This predeclared scaffold does **not** open or use the 23 reserved outcomes.

## Ground truth is preference-conditioned

Ground truth is not `dataset -> one framework`.

It is:

`(held-out dataset, predeclared preference profile) -> observed five-framework ranking`

For every held-out dataset/framework, aggregate seeds 42/43/44 first. Then apply the frozen V2 ranking rule to the observed Accuracy, Runtime, Energy and CO2 outcomes under each frozen preference profile.

## Final run count

23 datasets × 5 frameworks × 3 seeds = **345 framework runs**.

## Protocol

Match the V2 development regime:
- prequential test-then-train
- max samples 30,000
- window size 1,000
- time budget 60 s
- seeds 42, 43, 44
- Energy/CO2 measurement enabled

Do not use the 5,000-sample interactive demo protocol for final V2 external evaluation.

## No tuning after opening held-out outcomes

Freeze:
1. V2 model bundle;
2. ranking rule;
3. preference profiles;
4. metrics;
5. run protocol.

Then open/run the 23 held-out datasets once and report the observed result.
