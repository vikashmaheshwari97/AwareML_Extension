from pathlib import Path
import hashlib

from awareml.studies.trust import (
    BalancedWithinSubjectRandomizer,
    PARTICIPANT_UNJUDGEABLE_ERROR_TYPES,
    Phase15StimulusBank,
)

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "awareml" / "ui_v2" / "phase16_trust_calibration.py"

def test_readable_participant_assignment_excludes_hidden_citation_only_errors():
    bank = Phase15StimulusBank()
    randomizer = BalancedWithinSubjectRandomizer(bank)
    participant_hash = hashlib.sha256(b"phase16-judgeability-regression-test").hexdigest()[:24]
    assignment = randomizer.build(participant_hash, 20)

    assert len(assignment) == 20
    assert len({item.pair_id for item in assignment}) == 20
    assert sum(item.correctness_condition == "correct" for item in assignment) == 10
    assert sum(item.correctness_condition == "incorrect" for item in assignment) == 10
    assert all(
        str(item.error_type or "") not in PARTICIPANT_UNJUDGEABLE_ERROR_TYPES
        for item in assignment
    )

    by_stage = {}
    for item in assignment:
        by_stage[item.source_stage] = by_stage.get(item.source_stage, 0) + 1
    assert sorted(by_stage.values()) == [5, 5, 5, 5]

def test_researcher_ground_truth_tracks_preview_item():
    source = UI.read_text(encoding="utf-8")
    assert "Researcher-only ground truth for current preview" in source
    assert "stimulus = study.bank.get(preview_stimulus_id)" in source
    assert "This panel follows the selected Preview item automatically." in source
    assert "Verified reference evidence for this exact item" in source

def test_participant_help_explains_override_without_requiring_ui_editing():
    source = UI.read_text(encoding="utf-8")
    assert "You do not edit the text in this study; Override records that intended action." in source
    assert "additional claim that is not supported by the reference" in source

def test_frozen_phase15_bank_is_not_modified_by_fix():
    source = UI.read_text(encoding="utf-8")
    assert "The frozen Phase-15 bank is not edited; only participant eligibility is filtered." in source
