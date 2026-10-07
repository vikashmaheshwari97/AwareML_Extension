"""Local supplementary pilot collector; never writes to the frozen Phase-16 store."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

from awareml.studies.behavioral_calibration import validate_pilot_bank
from awareml.studies.behavioral_pilot_store import BehavioralPilotStore, utc_now


st.set_page_config(page_title="AwareML supplementary decision pilot", layout="centered")
st.title("AwareML decision-support pilot")
st.caption("Supplementary pilot · separate from Phase-16 final collection")
root = Path(__file__).resolve().parent
bank_path = Path(os.environ.get("AWAREML_XAI_PILOT_BANK", str(root / "artifacts/xai_pilot/reviewed_bank.json")))
db_path = Path(os.environ.get("AWAREML_XAI_PILOT_DB", str(root / "artifacts/xai_pilot/responses.sqlite")))
if not bank_path.exists():
    st.info("Researcher setup required: place a reviewed case bank at artifacts/xai_pilot/reviewed_bank.json. "
            "See docs/MULTILEVEL_XAI_QUICKSTART.md for the schema. No demonstration trials are mixed into responses.")
    st.stop()
try:
    bank = validate_pilot_bank(json.loads(bank_path.read_text(encoding="utf-8-sig")))
except (ValueError, KeyError, TypeError) as exc:
    st.error("Invalid study bank: " + str(exc))
    st.stop()

store = BehavioralPilotStore(db_path)
if "participant_code" not in st.session_state:
    st.write("You will make an initial decision, review advice, then decide again. Some advice or explanations may be inaccurate. "
             "Participation is voluntary. Responses are saved locally under a participant code; do not enter your name or email. "
             "You may stop at any time. Contact the researcher with your code to request removal of your records.")
    with st.form("consent"):
        code = st.text_input("Participant code supplied by the researcher", max_chars=80)
        consent = st.checkbox("I understand this pilot and agree to participate.")
        start = st.form_submit_button("Start or resume")
    if start:
        if not consent or not code.strip():
            st.error("Consent and a participant code are required.")
        else:
            try:
                assignment = store.register(code.strip(), bank)
                st.session_state.participant_code = code.strip()
                st.session_state.assignment = assignment
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.stop()

participant = st.session_state.participant_code
assignment = st.session_state.assignment
current = None
for item in assignment:
    initial, final = store.get_trial(participant, item["case"]["case_id"])
    if final is None:
        current = item
        break
if current is None:
    st.success("All pilot trials are complete. Thank you.")
    st.write("This pilot studies how people use advice and explanations. Some trials intentionally use incorrect advice or "
             "explanations. A confident explanation is not a guarantee of correctness.")
    st.download_button("Download your recorded responses", pd.DataFrame(store.export_participant(participant)).to_csv(index=False),
                       "pilot_responses.csv", "text/csv")
    st.stop()

case, condition = current["case"], current["condition"]
trial_id = case["case_id"]
st.progress(len(store.export_participant(participant)) / len(assignment))
st.subheader(case["question"])
st.write(case["task_evidence"])
if initial is None:
    start_key = "initial_started_" + trial_id
    if start_key not in st.session_state:
        st.session_state[start_key] = utc_now()
    with st.form("initial_" + trial_id):
        initial_answer = st.radio("Your initial decision", ["Select an answer"] + case["choices"])
        submitted = st.form_submit_button("Record decision and reveal advice")
    if submitted:
        if initial_answer == "Select an answer":
            st.error("Choose an answer before revealing advice.")
        else:
            store.save_initial(participant, trial_id, {"answer": initial_answer, "recorded_at": utc_now(),
                                                      "started_at": st.session_state[start_key]})
            st.rerun()
    st.stop()

st.write("**Your initial decision:**", initial["answer"])
st.write("**AI advice:**", case["ai_answer"])
if condition in ("rationale", "rationale_and_diagnostics"):
    st.write(case["rationale"])
if condition == "rationale_and_diagnostics":
    st.write("**Explanation evidence and limitations**")
    st.write(case["diagnostics"])
with st.form("final_" + trial_id):
    answer = st.radio("Your final decision", ["Select an answer"] + case["choices"])
    probability = st.number_input("Probability that the AI advice is correct (0–100%)", 0, 100, 50)
    trust = st.slider("How much do you trust this advice? (1–7)", 1, 7, 4)
    final_submitted = st.form_submit_button("Save response and continue")
if final_submitted:
    if answer == "Select an answer":
        st.error("Choose a final answer.")
    else:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(initial["recorded_at"])).total_seconds()
        store.save_final(participant, trial_id, {
            "participant_id": participant, "trial_id": trial_id, "dataset_id": case["dataset_id"],
            "condition": condition, "ground_truth": case["ground_truth"], "ai_answer": case["ai_answer"],
            "order_index": assignment.index(current),
            "bank_sha256": hashlib.sha256(json.dumps(bank, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "initial_answer": initial["answer"], "final_answer": answer,
            "ai_correct_probability": probability / 100, "trust_rating": trust,
            "post_advice_elapsed_sec": elapsed, "submitted_at": utc_now(), "collection_mode": "supplementary_pilot",
            "ground_truth_policy": case["ground_truth_policy"],
        })
        st.rerun()
