# Phase 15 protocol separation and robust live batch

Phase 15 now separates three purposes explicitly.

1. **Controlled benchmark** validates the verifier/intervention methodology with known ground truth.
2. **Live Dataset Probe** is exploratory UI/demo/debugging on the active dataset. It uses `phase15_live_guided_prompt_v1`.
3. **Empirical LLM benchmark** remains the journal-facing model evaluation path and uses `phase15_explanation_prompt_v4`.

The guided live prompt is deliberately **not** used for the empirical 24-case benchmark. This prevents repeated UI prompt fixes from silently changing the scientific evaluation protocol.

## Failure policy

A live LLM response that is empty, a clarification/meta-response, a placeholder response, or insufficiently grounded is recorded as a generation failure. No correctness or faithfulness score is fabricated.

A deterministic reference explanation can be displayed to show what evidence-grounded output looks like, but it is always labeled as a pipeline aid and never counted as LLM performance.

## All-stage live batch

The Live Dataset Probe can run Stage B, Stage E, Stage F XAI and Stage F conversational sources in one batch. Each stage is isolated. A failure in one stage does not stop the other stages. Results are persisted under:

`artifacts/phase15/live_batch_runs/<timestamp>__<dataset>/phase15_live_batch.json`

For Stage B and Stage F conversational live interventions, the UI changes one evidence field only and does not recompute the full production ranking. These remain claim-level sensitivity checks, not formal decision-flip evidence.
