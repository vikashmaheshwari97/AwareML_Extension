# Phase 15 benchmark-validity correction

The first four-source Llama 3 8B smoke test revealed two benchmark-side issues
that must be corrected before a journal-facing 24-case run.

## 1. Decision interventions must be internally coherent

The original Stage-B and Stage-F-chat counterfactuals changed AutoClass
accuracy, but left its derived utility/rank and the selected framework
unchanged.

An explanation that continued to discuss the unchanged ranking could therefore
be reasonable, even though the old evaluator treated failure to mention the
changed accuracy as unfaithful.

The corrected intervention:

1. perturbs AutoClass accuracy;
2. recomputes the controlled point utility;
3. recomputes framework ranks;
4. updates the top recommendation consistently; and
5. records the expected new decision.

For the first controlled context:

- original top framework: AutoClass
- original accuracy: 0.86
- counterfactual accuracy: 0.69
- counterfactual AutoClass utility: 0.265
- ChaCha utility: 0.271
- counterfactual top framework: ChaCha

This is now a genuine decision-flip intervention.

## 2. Fairness list output must be parsed line-by-line

Llama emitted fairness metrics as Markdown list rows such as:

`AutoClass: dp_diff = 0.155, equal_opportunity_diff = ...`

The older sentence parser joined multiple list rows into one large span and
therefore missed the changed DP value even though the LLM clearly updated it.

The corrected parser:

- treats line boundaries as claim boundaries;
- preserves decimal numbers and evidence keys;
- recognizes literal `dp_diff`, `equal_opportunity_diff`,
  `equalized_odds_gap`, `group_brier_score_gap`, and `group_ece_gap`.

The exact Stage-E output from the first four-source run is recognized as
updating DP from 0.035 to 0.155.

## Interpretation of the earlier smoke test

The run saved under `20260905T194221Z__llama3-8b` remains useful as a diagnostic
record, but its aggregate faithfulness score must **not** be reported as the
final model result because Stage B/F-chat interventions were not coherent
decision interventions and Stage E contained a parser false negative.

Stage-F-XAI already behaved correctly and reached faithfulness 1.0.

Rerun the 1-case and 4-case checks after this correction. Only then scale the
24-case experiment to HPC.
