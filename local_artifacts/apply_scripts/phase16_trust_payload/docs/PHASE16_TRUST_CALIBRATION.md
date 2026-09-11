# Phase 16 — Trust Calibration Study

## Research question

**Does user trust track actual explanation correctness, or merely how fluent/confident the explanation sounds?**

Phase 16 is Track 2 of the AwareML journal-extension roadmap. It must start only after the Phase-15 explanation-integrity work and correct-vs-incorrect stimulus bank are frozen.

## What changed

The earlier `TrustCalibrationStudy` scaffold manipulated the top-, second-, and lowest-utility framework. That implementation is retired. Phase 16 now consumes the frozen Phase-15 Track-2 stimulus bank at:

`data/journal/trust_stimulus_bank_v1/frozen/`

The bank is hash-verified before use. The participant-facing file contains no correctness labels; the separate researcher file supplies `known_correct` / `known_incorrect` ground truth server-side.

The current frozen bank is a **controlled Track-2 stimulus bank**. It is appropriate for the human calibration manipulation, but it must not be described as an empirical LLM-performance estimate.

## Study design implemented

- within-subject
- randomized
- participant-blinded
- default 20 explanations per participant; technical support for 16, 18, 20, 22, or 24
- exact 50/50 correct/incorrect allocation per participant
- one variant per underlying correct/incorrect pair per participant
- source-stage balancing across B, E, F_XAI and F_CHAT
- maximum three consecutive items from the same correctness condition
- opaque participant-facing item IDs
- separate Track-2 participants: anyone who authored Track-1 scenarios is ineligible
- pilot and final records separated in the same append-only SQLite store

## Measurements

Each trial records:

- finalized trust instrument score (provisional pilot trust item until Morten finalizes the measure)
- perceived factual correctness
- perceived fluency
- perceived confidence / authoritativeness
- `Accept`, `Override`, or `Reject`
- derived recommendation acceptance
- automatic response time
- hidden actual correctness condition
- expertise group and self-rating
- randomized order
- opaque participant item ID
- researcher stimulus ID/source stage in the protected database

The validated trust instrument is data-driven from `protocol.json`. Multiple items and reverse-coded items are supported. The final study is blocked until Morten marks the trust instrument and power calculation as finalized.

## Participant isolation

Real participants should **not** use the normal `app.py` navigation. Use:

```powershell
streamlit run phase16_participant_app.py
```

That entry point exposes only the Phase-16 study and creates no navigation into Advanced Labs or Phase-15 ground truth.

The normal AwareML Advanced Research Labs still contain **Trust Calibration**, but that page is a researcher workspace with a no-save blinded participant preview and a protected researcher console.

## Researcher mode

Recommended:

```powershell
$env:AWAREML_STUDY_RESEARCHER_KEY="choose-a-local-researcher-secret"
streamlit run app.py
```

Open **Advanced Research Labs → Trust Calibration**, enter the key in the Researcher console tab, and unlock the ground-truth view.

The previous development flag remains supported:

```powershell
$env:AWAREML_STUDY_RESEARCHER_MODE="1"
streamlit run app.py
```

Return to normal mode with:

```powershell
Remove-Item Env:AWAREML_STUDY_RESEARCHER_MODE -ErrorAction SilentlyContinue
```

A global participant/researcher toggle is intentionally not used for real data collection because it could let participants expose correctness labels.

## Pilot collection

Pilot is the default mode. It is safe for technical testing before the trust measure and power calculation are finalized.

```powershell
$env:AWAREML_PHASE16_COLLECTION_MODE="pilot"
streamlit run phase16_participant_app.py
```

Pilot records are never included by the final analysis/freezing commands unless explicitly analyzed as pilot.

## Final design gate

Before final recruitment, edit:

`data/journal/trust_calibration_phase16_v1/design/protocol.json`

The following must be finalized:

1. `status = "final"`
2. Morten's validated `trust_measure`
3. Morten's `power_calculation`, including `required_completed_participants`
4. institutional ethics determination
5. final participant instructions and consent text

Then freeze the design exactly once:

```powershell
python scripts/freeze_phase16_design.py
```

The design freeze refuses to overwrite an existing frozen design.

## Final collection

Use a private participant-ID salt. Do not commit the salt.

```powershell
$env:AWAREML_PHASE16_ID_SALT="replace-with-a-private-random-secret-at-least-16-characters"
$env:AWAREML_PHASE16_COLLECTION_MODE="final"
$env:AWAREML_PHASE16_FINAL_ARMED="YES"
streamlit run phase16_participant_app.py
```

Raw participant/session codes are HMAC-pseudonymized and are not stored in the study database.

## Analysis

Run:

```powershell
python scripts/analyze_phase16_trust.py --mode final
```

Primary calibration estimand:

`mean trust(correct) - mean trust(incorrect)` within participant.

The analysis reports:

- participant-level bootstrap 95% CI
- paired sign-flip permutation test
- mean trust and acceptance by correctness condition
- over-trust = accepting known-incorrect stimuli
- under-trust = overriding/rejecting known-correct stimuli
- appropriate reliance
- fluency/trust association
- perceived-confidence/trust association
- participant-centered standardized regression comparing correctness vs style cues
- expertise-group results
- order effects
- Phase-15 source-stage results

## Completion freeze

After the power target is met and all participant records are complete:

```powershell
python scripts/freeze_phase16_results.py
```

The final freeze creates:

- `study_protocol.json`
- `stimulus_randomization.csv`
- `participant_summary.csv`
- `participant_responses.csv`
- exact `analysis_script.py`
- `analysis_results.json`
- `calibration_results.json`
- `overtrust_results.json`
- cryptographic `manifest.json`

under:

`data/journal/trust_calibration_phase16_v1/frozen/`

The command refuses to freeze when the power target is not met, participant records are incomplete/unbalanced, or the design is not frozen.
