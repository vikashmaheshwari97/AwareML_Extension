from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_phase14_is_integrated_into_existing_labs():
    advanced = _read("awareml/ui_v2/pages_advanced.py")
    specialist = _read("awareml/ui_v2/pages_specialist.py")

    assert "Phase 14 · Fairness + Sustainability Hardening" not in advanced
    assert '"Fairness Lab": fairness_v2_page' in advanced
    assert '"Sustainability Lab": sustainability_v2_page' in advanced
    assert "render_phase14_fairness_details(results)" in specialist
    assert "render_phase14_sustainability_details(results)" in specialist


def test_repeatability_runner_resolves_downloads_and_dutch_schema():
    runner = _read("scripts/run_phase14_repeatability.py")
    assert 'Path.home() / "Downloads"' in runner
    assert 'DUTCH_TARGET = "occupation_binary"' in runner
    assert 'DUTCH_SENSITIVE = "sex"' in runner
