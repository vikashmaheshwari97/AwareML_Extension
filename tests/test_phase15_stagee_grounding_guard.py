from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_live_grounding_is_still_separate_from_correctness():
    source = (
        ROOT / "awareml" / "explanation_integrity" / "live_guided.py"
    ).read_text(encoding="utf-8")

    assert "generation-validity" in source
    assert "Correctness quality belongs to GeneralEvidenceVerifier metrics" in source
