# Trust Calibration — final UI/access notes

## Researcher access during Pilot Study

For local Pilot Study development, the familiar convenience key is available:

```text
phase16-local-test-key
```

No `.streamlit/secrets.toml` file is required for Pilot Study, and the UI no
longer probes `st.secrets`, so Streamlit does not display the red
"No secrets found" message.

This convenience key is intentionally disabled after the final study design is
frozen. Before Main Study deployment, configure a private key:

```text
AWAREML_STUDY_RESEARCHER_KEY=<private-key>
```

The application already loads the local `.env` file through `app.py`.

## Main Study timing

Do not freeze/start Main Study yet if the validated trust instrument, power
calculation, ethics determination, participant materials, or stimulus-diversity
decision are still open.

The software implementation can be considered complete after tests pass.
The scientific study design should be frozen only after the research team
(including Morten for the trust measure and power calculation) finalizes those
items.

## Stimulus repetition

The current Trust Calibration assignment explicitly balances:
- correct vs incorrect condition;
- explanation source stages;
- one variant from each underlying matched pair.

It does NOT currently add a separate balancing constraint for:
- AutoClass vs AutoStreamML vs OAML vs EvoAutoML vs ChaCha;
- SHAP vs LIME.

Therefore repeated framework names or SHAP-oriented items can occur.

This patch does not change that logic. A researcher-facing note now makes the
issue explicit so the team can decide before design freeze whether broader
framework / XAI-method diversity is required. If the study design changes,
change and revalidate the stimulus bank before Main Study, not after data
collection starts.

## UI spacing

A small shared presentation module now adds conservative spacing between cards,
columns, forms, expanders, buttons and major sections across AwareML. It does
not alter Streamlit state or scientific logic.
