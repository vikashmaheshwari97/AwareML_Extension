from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from awareml.studies.trust import (
    BalancedWithinSubjectRandomizer,
    EligibilityError,
    Phase15StimulusBank,
    Phase16Error,
    Phase16Store,
    ProtocolGateError,
    TrustCalibrationStudy,
    score_trust_items,
    validate_protocol_for_design_freeze,
)
from awareml.studies.trust_analysis import analyze_phase16_frames
from awareml.studies.trust_freeze import freeze_phase16_design


STAGES = ["B", "E", "F_XAI", "F_CHAT"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bank(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    participant = []
    researcher = []
    for i in range(24):
        stage = STAGES[i % len(STAGES)]
        pair_id = "PAIR_{:02d}_{}".format(i + 1, stage)
        for condition in ("correct", "incorrect"):
            stimulus_id = "{}_{}".format(pair_id, condition.upper())
            explanation = "Explanation {} {} with matched fluent wording.".format(i + 1, condition)
            p = {
                "stimulus_id": stimulus_id,
                "source": "Stage {} explanation".format(stage),
                "scenario": "scenario_{:02d}".format(i + 1),
                "prompt": "Evaluate explanation {}.".format(i + 1),
                "explanation": explanation,
            }
            participant.append(p)
            researcher.append(
                {
                    "stimulus_id": stimulus_id,
                    "pair_id": pair_id,
                    "dataset_id": "scenario_{:02d}".format(i + 1),
                    "source_stage": stage,
                    "source_name": "Stage {} explanation".format(stage),
                    "prompt": p["prompt"],
                    "explanation": explanation,
                    "researcher_label": "known_correct" if condition == "correct" else "known_incorrect",
                    "error_type": None if condition == "correct" else "numeric_error",
                    "participant_version": p,
                    "provenance": {
                        "explanation_sha256": hashlib.sha256(explanation.encode("utf-8")).hexdigest()
                    },
                }
            )
    files = {
        "stimuli_participant.json": participant,
        "stimuli_researcher.json": researcher,
        "summary.json": {"correct": 24, "incorrect": 24, "pairs": 24},
        "benchmark_metadata.json": {"controlled_benchmark": True},
    }
    for name, payload in files.items():
        (root / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "artifact": "trust_stimulus_bank_v1",
        "status": "frozen",
        "schema_version": "trust_stimulus_bank_v1",
        "files": {
            name: {"sha256": _sha(root / name), "size_bytes": (root / name).stat().st_size}
            for name in files
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def draft_protocol() -> dict:
    return {
        "artifact": "phase16_trust_calibration_protocol_v1",
        "schema_version": "phase16_trust_calibration_v1",
        "phase": 16,
        "track": 2,
        "status": "draft",
        "central_rq": "Does user trust track actual explanation correctness, or merely how fluent/confident the explanation sounds?",
        "design": {
            "within_subject": True,
            "randomized": True,
            "blinded": True,
            "items_per_participant": 20,
            "shared_pool_min_per_condition": 20,
            "track1_authors_excluded": True,
        },
        "trust_measure": {
            "status": "pending_morten",
            "instrument_name": None,
            "scale_min": 1,
            "scale_max": 7,
            "scoring_rule": "mean_after_reverse_coding",
            "items": [
                {
                    "id": "pilot_trust",
                    "text": "How much do you trust this explanation?",
                    "reverse": False,
                    "pilot_only": True,
                }
            ],
        },
        "power_calculation": {
            "status": "pending_morten",
            "required_completed_participants": None,
        },
        "ethics": {"status": "pending", "reference": None},
        "participant_materials": {
            "status": "draft",
            "instructions": "Pilot instructions",
            "consent_text": "Pilot consent",
        },
        "randomization": {"max_same_condition_run": 3},
    }


def finalized_protocol(required_n: int = 2) -> dict:
    protocol = copy.deepcopy(draft_protocol())
    protocol["status"] = "final"
    protocol["trust_measure"] = {
        "status": "final",
        "instrument_name": "Finalized test instrument",
        "citation": "Test citation",
        "scale_min": 1,
        "scale_max": 7,
        "scoring_rule": "mean_after_reverse_coding",
        "items": [
            {"id": "trust_a", "text": "Trust item A", "reverse": False, "pilot_only": False},
            {"id": "trust_b", "text": "Trust item B", "reverse": True, "pilot_only": False},
        ],
    }
    protocol["power_calculation"] = {
        "status": "final",
        "required_completed_participants": required_n,
        "alpha": 0.05,
        "target_power": 0.8,
    }
    protocol["ethics"] = {"status": "not_required", "reference": None}
    protocol["participant_materials"] = {
        "status": "final",
        "instructions": "Final instructions",
        "consent_text": "Final consent",
    }
    return protocol


def write_protocol(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def make_study(tmp_path: Path, protocol: dict = None) -> TrustCalibrationStudy:
    bank = make_bank(tmp_path / "bank")
    protocol_path = write_protocol(tmp_path / "protocol.json", protocol or draft_protocol())
    return TrustCalibrationStudy(
        protocol_path=protocol_path,
        stimulus_root=bank,
        db_path=tmp_path / "study.sqlite",
        id_salt="pilot-test-salt-123456789",
        verify_stimulus_hashes=True,
    )


def test_bank_has_24_correct_and_24_incorrect(tmp_path):
    bank = Phase15StimulusBank(make_bank(tmp_path / "bank"))
    summary = bank.validate_design_pool()
    assert summary["correct"] == 24
    assert summary["incorrect"] == 24
    assert summary["pairs"] == 24


def test_bank_hash_tampering_is_rejected(tmp_path):
    root = make_bank(tmp_path / "bank")
    path = root / "stimuli_participant.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(Exception):
        Phase15StimulusBank(root)


def test_randomization_is_exactly_balanced_and_unique(tmp_path):
    bank = Phase15StimulusBank(make_bank(tmp_path / "bank"))
    randomizer = BalancedWithinSubjectRandomizer(bank)
    assignment = randomizer.build("participant-abc", 20)
    conditions = [item.correctness_condition for item in assignment]
    assert conditions.count("correct") == 10
    assert conditions.count("incorrect") == 10
    assert len({item.pair_id for item in assignment}) == 20


def test_randomization_limits_condition_runs(tmp_path):
    bank = Phase15StimulusBank(make_bank(tmp_path / "bank"))
    assignment = BalancedWithinSubjectRandomizer(bank, max_same_condition_run=3).build("participant-run", 24)
    run = 0
    last = None
    max_run = 0
    for item in assignment:
        if item.correctness_condition == last:
            run += 1
        else:
            run = 1
            last = item.correctness_condition
        max_run = max(max_run, run)
    assert max_run <= 3


def test_randomization_is_deterministic_per_participant(tmp_path):
    bank = Phase15StimulusBank(make_bank(tmp_path / "bank"))
    randomizer = BalancedWithinSubjectRandomizer(bank)
    a = [x.to_dict() for x in randomizer.build("same-person", 20)]
    b = [x.to_dict() for x in randomizer.build("same-person", 20)]
    assert a == b


def test_different_participants_receive_different_orders(tmp_path):
    bank = Phase15StimulusBank(make_bank(tmp_path / "bank"))
    randomizer = BalancedWithinSubjectRandomizer(bank)
    a = [x.stimulus_id for x in randomizer.build("person-a", 20)]
    b = [x.stimulus_id for x in randomizer.build("person-b", 20)]
    assert a != b


def test_participant_preview_contains_no_condition_or_pair_id(tmp_path):
    study = make_study(tmp_path)
    preview = study.build_preview_assignment("preview")
    assert len(preview) == 20
    assert all("condition" not in item for item in preview)
    assert all("correctness_condition" not in item for item in preview)
    assert all("pair_id" not in item for item in preview)


def test_track1_authors_are_ineligible(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(EligibilityError):
        study.register(
            "P001",
            "Novice / student",
            2,
            True,
            False,
            collection_mode="pilot",
        )


def test_raw_participant_code_is_not_stored(tmp_path):
    study = make_study(tmp_path)
    reg = study.register("SECRET-PARTICIPANT-CODE", "Practitioner", 3, True, True, "pilot")
    assert reg.participant_hash != "SECRET-PARTICIPANT-CODE"
    raw = (tmp_path / "study.sqlite").read_bytes()
    assert b"SECRET-PARTICIPANT-CODE" not in raw


def test_current_trial_is_blinded(tmp_path):
    study = make_study(tmp_path)
    reg = study.register("P002", "Practitioner", 3, True, True, "pilot")
    trial = study.current_trial(reg.participant_hash, "pilot")
    assert trial is not None
    assert "correctness_condition" not in trial
    assert "stimulus_id" not in trial
    assert trial["item_id"].startswith("T16-")


def test_response_is_append_only(tmp_path):
    study = make_study(tmp_path)
    reg = study.register("P003", "ML / AutoML expert", 5, True, True, "pilot")
    trial = study.current_trial(reg.participant_hash, "pilot")
    study.submit_trial(reg.participant_hash, trial["order"], {"pilot_trust": 5}, 5, 6, 6, "Accept", 2.5, "pilot")
    with pytest.raises(Phase16Error):
        study.store.save_response(
            "pilot", reg.participant_hash, trial["order"], 5.0, {"pilot_trust": 5}, 5, 6, 6, "Accept", 2.0
        )


def test_reverse_coded_multi_item_trust_score():
    protocol = finalized_protocol()
    # trust_a=7, trust_b=1 and reverse coding makes trust_b=7; mean=7.
    score = score_trust_items(protocol, {"trust_a": 7, "trust_b": 1}, "final")
    assert score == 7.0


def test_design_freeze_gate_blocks_pending_morten():
    errors = validate_protocol_for_design_freeze(draft_protocol())
    joined = " ".join(errors)
    assert "Morten gate" in joined
    assert "power_calculation" in joined


def test_design_freeze_succeeds_only_after_finalization(tmp_path):
    protocol_path = write_protocol(tmp_path / "protocol.json", finalized_protocol())
    bank = make_bank(tmp_path / "bank")
    out = tmp_path / "frozen_design"
    manifest = freeze_phase16_design(protocol_path, out, bank)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen"
    assert payload["source_stimulus_bank"]["summary"]["pairs"] == 24


def synthetic_analysis_rows(n_participants: int = 4):
    participants = []
    responses = []
    for p in range(n_participants):
        ph = "p{:02d}".format(p)
        participants.append(
            {
                "participant_hash": ph,
                "expertise_group": "Novice / student" if p < n_participants / 2 else "ML / AutoML expert",
                "n_items": 20,
                "completed_at": "2026-09-07T00:00:00+00:00",
            }
        )
        for order in range(1, 21):
            correct = order <= 10
            responses.append(
                {
                    "participant_hash": ph,
                    "expertise_group": participants[-1]["expertise_group"],
                    "order_index": order,
                    "correctness_condition": "correct" if correct else "incorrect",
                    "trust_rating": 6.0 if correct else 2.0,
                    "perceived_correctness": 6 if correct else 2,
                    "fluency_rating": 5,
                    "perceived_confidence": 5,
                    "decision_action": "Accept" if correct else "Reject",
                    "accepted": 1 if correct else 0,
                    "response_time_sec": 4.0 + order / 10.0,
                    "source_stage": STAGES[(order - 1) % 4],
                }
            )
    return participants, responses


def test_analysis_recovers_positive_calibration_gap():
    participants, responses = synthetic_analysis_rows()
    result = analyze_phase16_frames(participants, responses, finalized_protocol(required_n=4))
    assert result["status"] == "ok"
    assert result["primary_calibration"]["paired_gap"]["mean_gap"] == pytest.approx(4.0)
    assert result["overtrust"]["overtrust_rate"] == 0.0
    assert result["reliance"]["appropriate_reliance_rate"] == 1.0


def test_analysis_counts_overtrust():
    participants, responses = synthetic_analysis_rows(2)
    # Turn one incorrect trial into acceptance.
    for row in responses:
        if row["participant_hash"] == "p00" and row["correctness_condition"] == "incorrect":
            row["accepted"] = 1
            row["decision_action"] = "Accept"
            break
    result = analyze_phase16_frames(participants, responses, finalized_protocol(required_n=2))
    assert result["overtrust"]["incorrect_acceptance_count"] == 1
    assert result["overtrust"]["overtrust_rate"] == pytest.approx(1.0 / 20.0)
