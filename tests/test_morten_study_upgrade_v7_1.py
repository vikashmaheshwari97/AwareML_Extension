from pathlib import Path
import json

from awareml.studies.trust_morten_v2 import StimulusBankV2, TrustCalibrationStudyV2
from awareml.ui_v2.phase16_morten_v2 import _evidence_answer


def test_bank_has_20_unique_scenario_aligned_pairs():
    bank = StimulusBankV2()
    assert len(bank.pairs) == 20
    scenarios = [bank.pairs[p]["correct"].scenario_id for p in bank.pair_order]
    assert len(set(scenarios)) == 20
    frameworks = [bank.pairs[p]["correct"].framework_context for p in bank.pair_order]
    assert {fw: frameworks.count(fw) for fw in set(frameworks)} == {
        "AutoStreamML": 4, "AutoClass": 4, "EvoAutoML": 4, "OAML": 4, "ChaCha": 4
    }
    stages = [bank.pairs[p]["correct"].source_stage for p in bank.pair_order]
    assert {stage: stages.count(stage) for stage in set(stages)} == {
        "OBJ_FIT": 5, "COMPARE": 5, "TRADEOFF": 5, "F_CHAT": 5
    }


def test_scenario_task_prompt_uses_only_frozen_track1_objectives():
    rows = json.loads(Path("data/journal/trust_stimulus_bank_v2/frozen/stimuli_researcher.json").read_text(encoding="utf-8"))
    for row in rows:
        objectives = row["track1_ground_truth_objectives"]
        prompt = row["prompt"]
        for objective in objectives:
            assert objective in prompt
        assert "group-disparity" not in prompt.lower()
        assert "shap" not in prompt.lower()
        assert row["evidence_summary"]["scenario_objectives"] == objectives
        assert len(row["evidence_summary"]["frameworks"]) == 5


def test_h026_is_accuracy_runtime_not_fairness_and_rank_accuracy_question_is_answered():
    bank = StimulusBankV2()
    stimulus = next(
        bank.pairs[p]["correct"] for p in bank.pair_order
        if bank.pairs[p]["correct"].scenario_id == "H026"
    )
    trial = stimulus.participant_dict("TEST-H026", 1)
    assert "Accuracy" in stimulus.prompt and "Runtime" in stimulus.prompt
    assert "fair" not in stimulus.prompt.lower()
    answer = _evidence_answer("Please provide the rank also the accuracy of AutoClass", trial)
    assert "AutoClass" in answer
    assert "rank #" in answer
    assert "accuracy" in answer.lower()
    fairness = _evidence_answer("What is the fairness of AutoClass?", trial)
    assert "not part of this trial" in fairness.lower()
    assert "will not invent" in fairness.lower()


def test_counterbalance_remains_10_10_and_two_correct_per_framework(tmp_path, monkeypatch):
    monkeypatch.setenv("AWAREML_PHASE16_DB", str(tmp_path / "trust.sqlite"))
    study = TrustCalibrationStudyV2(id_salt="0123456789abcdef012345")
    for arm in ["A", "B"]:
        items = study._assignment("participant-" + arm, arm)
        assert len(items) == 20
        assert sum(x.correctness_condition == "correct" for x in items) == 10
        for fw in ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"]:
            subset = [x for x in items if x.framework_context == fw]
            assert len(subset) == 4
            assert sum(x.correctness_condition == "correct" for x in subset) == 2


def test_information_seeking_v21_protocol_requires_scenario_alignment_and_richer_behavior_analysis():
    protocol = json.loads(Path("data/journal/information_seeking_v2/frozen/protocol.json").read_text(encoding="utf-8"))
    assert protocol["schema_version"] == "information_seeking_v2_1"
    assert protocol["study_context"]["researcher_alignment_attestation_required"] is True
    assert "repeat" in protocol["analysis"]["researcher_visualization"].lower()
    src = Path("awareml/ui_v2/phase17_morten_v2.py").read_text(encoding="utf-8")
    for marker in ["Behavior overview", "Evidence use", "Decision & timing", "Sequences", "Clusters", "Qualitative coding"]:
        assert marker in src


def test_trust_researcher_workspace_restores_visual_analytics():
    protocol = json.loads(Path("data/journal/trust_calibration_phase16_v2/frozen_design/protocol.json").read_text(encoding="utf-8"))
    assert protocol["schema_version"] == "phase16_trust_calibration_v2_1"
    assert "scenario-task-evidence" in protocol["revision_basis"].lower()
    src = Path("awareml/ui_v2/phase16_morten_v2.py").read_text(encoding="utf-8")
    for marker in ["Participant-level paired trust profile", "Accept / Override / Reject by correctness", "Trial-family exposure", "Trial-order diagnostics", "Expertise groups"]:
        assert marker in src
