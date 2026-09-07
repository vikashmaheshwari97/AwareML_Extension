from awareml.explanation_integrity.stimuli import (
    build_trust_stimulus_bank,
    participant_bank,
    stimulus_counts,
)


def test_stimulus_bank_has_at_least_20_per_condition():
    records = build_trust_stimulus_bank()
    counts = stimulus_counts(records)

    assert counts["labels"]["known_correct"] >= 20
    assert counts["labels"]["known_incorrect"] >= 20
    assert counts["dataset_contexts"] >= 3
    assert len(counts["sources"]) == 4


def test_participant_bank_does_not_leak_ground_truth_label():
    records = build_trust_stimulus_bank()
    participant = participant_bank(records)

    assert participant
    for row in participant:
        assert "researcher_label" not in row
        assert "error_type" not in row
        assert "verifier_metrics" not in row


def test_researcher_pairs_are_balanced():
    records = build_trust_stimulus_bank()
    by_pair = {}
    for row in records:
        by_pair.setdefault(row.pair_id, []).append(row.researcher_label)

    assert len(by_pair) >= 20
    for labels in by_pair.values():
        assert sorted(labels) == ["known_correct", "known_incorrect"]
