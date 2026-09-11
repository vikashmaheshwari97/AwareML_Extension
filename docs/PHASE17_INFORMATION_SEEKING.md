# Information-Seeking Study

## Research question

**When do users ask for more information instead of accepting the initial explanation?**

This implements the smaller/optional Information-Seeking human-study track.

## What is logged

For each session the implementation captures:

- whether a follow-up occurred;
- number of follow-ups;
- first follow-up category;
- time before the first follow-up;
- evidence areas viewed;
- final recommendation decision: Accept / Override / Reject;
- final decision confidence;
- optional participant rationale.

Follow-up questions are deterministically classified as:

- Evidence request
- Explanation probe
- Challenge
- Counterfactual / comparison
- Clarification
- Other follow-up

**The classifier is an analysis aid only. It does not replace qualitative coding.**

## Participant flow

The Information-Seeking Study uses the current AwareML measured run and the
observed post-run ranking from Decision Lab.

1. Researcher prepares/runs the five-framework benchmark.
2. Open Decision Lab once so an observed ranking is available.
3. Participant enters Information-Seeking Study.
4. Participant sees the initial recommendation.
5. Participant may inspect evidence and/or ask follow-ups.
6. Participant finishes whenever they feel they have enough information.
7. Accept / Override / Reject is recorded.

Zero follow-ups is a valid behavioral outcome.

## Researcher flow

The protected Researcher Workspace provides:

- Pilot/Main collection counts;
- follow-up rate and depth;
- median time to first follow-up;
- evidence-view summaries;
- final recommendation decisions;
- automated descriptive behavior patterns;
- manual qualitative coding;
- manual-theme counts;
- CSV/JSON exports.

The local unfrozen Pilot convenience key is the same one currently used by the
Trust Calibration Pilot:

```text
phase16-local-test-key
```

For final deployment, configure a private:

```text
AWAREML_STUDY_RESEARCHER_KEY=<private key>
```

## Pilot versus Main Study

Default collection is Pilot. Pilot data is stored as:

```text
information_seeking_pilot
```

Final data is stored separately as:

```text
information_seeking_final
```

Main Study cannot activate unless:
- the study design has been frozen;
- `AWAREML_INFORMATION_SEEKING_COLLECTION_MODE=final`;
- `AWAREML_INFORMATION_SEEKING_FINAL_ARMED=YES`.

## Finalization

The package ships a DRAFT protocol because the suggested 10–15 participants,
ethics status, final participant materials and qualitative coding scheme should
be reviewed before real recruitment.

After review, edit:

```text
data/journal/information_seeking_v1/design/protocol.json
data/journal/information_seeking_v1/design/coding_scheme.json
```

Then:

```powershell
python -m scripts.freeze_phase17_design
```

For analysis:

```powershell
python -m scripts.analyze_phase17_information_seeking --mode pilot
python -m scripts.analyze_phase17_information_seeking --mode final
```

After final recruitment reaches the frozen target:

```powershell
python -m scripts.freeze_phase17_results
```

## Completion gate

The implementation supports all required deliverables:

- qualitative coding scheme;
- behavior logs;
- theme analysis from manual codes;
- representative behavior patterns.

Representative patterns are descriptive structural summaries and are kept
separate from manually coded qualitative themes.


## Pilot workflow after usability/grounding fix

For a local internal Pilot test, the researcher may:

1. Open **Run Studio**.
2. Run the agreed five-framework benchmark (for the current pilot, the Dutch Census setup can be reused).
3. Open **Decision Lab** once so the observed post-run ranking is created.
4. Return to **Information-Seeking Study**.

For repeated participants, do **not** require every participant to rerun AwareML.
Open the protected Researcher Workspace and use:

**Save current measured run as Pilot study context**

This stores only derived measured framework evidence and the observed ranking in
`artifacts/information_seeking_context_pilot.json`. Raw dataset rows are not saved.
Later participant sessions on the same deployment can use that standardized context.

Before Main Study, use one finalized/frozen study context rather than allowing
participant-by-participant benchmark differences unless such variation is explicitly
part of the approved study design.

## Multi-topic grounded questions

Participant questions can mention several information needs at once. For example:

> Explain OAML accuracy, fairness, and sustainability.

The updated deterministic grounding layer detects all requested supported topics and
answers each one. A multi-topic question is still assigned one primary follow-up
behavior category for logging, but the requested evidence topics are stored separately.

Participant-visible grounded answers no longer show internal provenance strings such
as `[frameworks.OAML.fairness]`. The underlying values still come only from the measured
study context.

## Participant blinding / behavioral priming

The automated follow-up category is no longer displayed to participants during the
session. It remains available in researcher logs. Example quick questions are collapsed
under an optional expander so they are less likely to steer participant behavior.

The Participant page also displays the current Pilot consent/information text before
the consent checkbox.
