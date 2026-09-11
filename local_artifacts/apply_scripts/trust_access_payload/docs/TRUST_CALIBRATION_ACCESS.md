# Trust Calibration access model

## Normal use: one AwareML application

Start AwareML normally:

```powershell
streamlit run app.py
```

Then open **Advanced Labs → Trust Calibration**.

The Trust Calibration workspace presents two clear choices:

1. **Participant Study**
   - Runs the real blinded participant flow inside the normal AwareML app.
   - Pilot Study vs Main Study is decided by the server configuration, not by the participant.
   - Dashboard navigation is hidden while the participant study view is active.
   - The participant can exit the study view; progress is preserved.
   - After completion, the participant can return to the AwareML dashboard.

2. **Researcher Workspace**
   - Contains study readiness, ground truth, response counts, exports and analysis.
   - Requires a configured researcher key.
   - The key should be configured once by the app administrator in `.env` or deployment secrets.
   - Participants never see or choose researcher access without the key.

## Participant-only deployment

For external recruitment, where participants must not see the AwareML dashboard at all:

```powershell
streamlit run phase16_participant_app.py
```

or:

```powershell
.\START_PARTICIPANT_ONLY.ps1
```

A special port such as `8502` is needed only when the normal AwareML app and the
participant-only app are running at the same time on the same machine.

## Researcher key

For local development, configure once in the repository's ignored `.env` file:

```text
AWAREML_STUDY_RESEARCHER_KEY=replace-with-a-private-key
```

`app.py` already loads `.env`. After that, normal use remains:

```powershell
streamlit run app.py
```

The legacy `AWAREML_STUDY_RESEARCHER_MODE` flag remains internally compatible
for development, but it is no longer part of the normal workflow.

## Pilot Study vs Main Study

Participants do not select the collection mode.

- Before the design is formally frozen, the UI resolves to **Pilot Study**.
- Main Study remains protected by the existing final-collection gates.
- After the design is frozen, the research team configures the server for Main Study.
- Pilot and Main Study responses remain physically separated in the existing append-only store.

The scientific study logic, stimulus bank, randomization, response schema,
analysis and freeze gates are unchanged by this access/UI simplification.
