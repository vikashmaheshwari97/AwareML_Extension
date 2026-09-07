# Phase 15 · Live Dataset Probe — simple workflow

After the five AwareML frameworks finish:

1. Open **Explanation Integrity · Phase 15 → Live Dataset Probe · exploratory**.
2. Leave **Also test faithfulness for all four sources** enabled.
3. Click **Run complete Phase-15 live check**.
4. Read the summary:
   - `OK` = completed and clean.
   - `REVIEW` = usable explanation, but inspect a correctness/faithfulness metric.
   - `FAILED` = no usable grounded output was produced.
5. Open only `REVIEW` or `FAILED` sources.

You do not need to run Stage B, Stage E, Stage F XAI and Stage F conversational
one by one.

## Why the previous run showed 0/4

The live grounding guard was too strict. It treated any contradicted or
unsupported claim as if the LLM had failed to generate an explanation at all.
That mixed up two different questions:

- **Generation validity:** did the LLM produce at least one source-grounded
  explanation claim?
- **Correctness quality:** are all extracted claims actually correct?

The final live guard now separates them. Imperfect LLM claims remain visible
and are scored as `REVIEW`; they are not discarded as `failed_grounding`.

## Faithfulness

When the batch checkbox is enabled, faithfulness is already tested once for all
four sources. The collapsed **Advanced manual verification** section is only for
researchers who want to paste a custom explanation or try a different
counterfactual value.

Stage B and Stage F conversational interventions remain claim-level sensitivity
checks in the live UI because one changed field does not recompute the full
production ranking.
