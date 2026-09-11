# Trust Calibration Study Guide

## What this study is for

Trust Calibration tests whether a person's trust in an AI explanation changes
appropriately when the explanation is actually correct or incorrect.

A well-calibrated user should generally:
- trust / accept correct explanations more;
- reduce reliance on incorrect explanations;
- avoid treating fluent or confident language as proof of correctness.

The study is within-subject, randomized and blinded. The same participant sees
a balanced mixture of correct and incorrect explanation stimuli, but never sees
the researcher correctness label.

## Participant Study

Participants do NOT need to run the full AwareML dashboard first.

The Participant Study is self-contained. For each item the participant receives:
1. the task given to the AI;
2. a readable AI explanation;
3. reference evidence that the explanation is supposed to summarize;
4. trust, correctness, fluency and confidence ratings;
5. an Accept / Override / Reject reliance decision.

Technical provenance tags such as `[evidence....]` are removed from the
participant-facing rendering. Common metric abbreviations are expanded. The
underlying frozen stimulus bank and all numeric claims remain unchanged.

The reference-evidence panel is derived from the already-verified
`evidence_summary` in the frozen source bank. It does NOT reveal the
correct/incorrect label.

## Pilot Study versus Main Study

Pilot Study:
- checks wording, usability and the data pipeline;
- is stored separately;
- is not journal evidence;
- never becomes Main Study data.

Main Study:
- starts from zero;
- remains locked until the design is finalized and frozen;
- contains only real final-study participants.

Zero Main Study counts are therefore expected before finalization.

## Study readiness

Before Main Study can open, the research team must finalize:
- the overall study protocol;
- validated trust instrument: name, citation, exact items and scoring;
- power / sample-size calculation and required completed N;
- ethics determination and reference where applicable;
- final participant instructions and consent text;
- design freeze.

After these are genuinely complete:

```powershell
python -m scripts.freeze_phase16_design
```

Then configure deployment-only secrets:

```text
AWAREML_PHASE16_COLLECTION_MODE=final
AWAREML_PHASE16_FINAL_ARMED=YES
AWAREML_PHASE16_ID_SALT=<private value of at least 16 characters>
```

Do not commit these secrets to Git.

## Calibration analysis

Trust gap:
mean trust(correct) minus mean trust(incorrect).
Positive is better; near zero means participants barely distinguish correctness.

Over-trust:
fraction of incorrect explanations that were accepted.
Lower is better.

Appropriate reliance:
accepting correct explanations plus overriding/rejecting incorrect explanations.
Higher is better.

Valid participants:
participants who completed the balanced within-subject assignment.

Pilot analysis is diagnostic only. It should not be reported as a journal result.

## Important UI bug fixed

Earlier, the Researcher Workspace used one shared Streamlit session key for the
analysis result. This allowed a Pilot Study analysis to remain visible after
switching the dataset selector to Main Study, even when Main Study had zero
records.

The updated UI stores analysis results separately by collection mode:
- Pilot analysis stays under Pilot Study.
- Main analysis stays under Main Study.
- Main analysis is disabled when no completed Main Study participant exists.

The statistical analysis implementation itself is unchanged.

## Scientific core unchanged

This usability patch does NOT change:
- frozen source stimuli;
- correctness labels;
- balanced randomization;
- participant pseudonymization;
- append-only storage;
- response schema;
- trust-analysis formulas;
- final design gate;
- result-freezing logic.

The Participant presentation is improved before the design is frozen. Review
this rendering during Pilot Study before Main Study begins.
