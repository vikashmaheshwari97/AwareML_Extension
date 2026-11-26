from pathlib import Path

from awareml.types import RunConfig

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_hotfix_runner_is_wired():
    source = (ROOT / "awareml/engine/runner.py").read_text(encoding="utf-8")
    assert "HybridDriftController" in source
    assert "adapt_framework" in source
    assert "refit_events=refit_events" in source
    assert "recent_training.append((dict(x), y))" in source


def test_temporal_hotfix_run_studio_controls_are_wired():
    source = (ROOT / "awareml/ui/pages.py").read_text(encoding="utf-8")
    assert "Hybrid · ADWIN + Page-Hinkley + degradation confirmation" in source
    assert "Adaptive reset + recent-window replay" in source
    assert "drift_detector_mode=drift_detector_mode" in source


def test_temporal_hotfix_preserves_historical_defaults():
    cfg = RunConfig(target="y")
    assert cfg.drift_detector_mode == "adwin"
    assert cfg.drift_action_policy == "monitor_only"


def test_multilevel_stale_contract_is_removed_if_test_exists():
    """Validate the integrated XAI contract without self-scanning test literals.

    v3.1 accidentally searched this *test file* for the old Ollama tag.  The
    compatibility test itself intentionally contains that literal while
    asserting that the production page does not, so the assertion could never
    pass.  Check the production page and the actual retired UI contract instead.
    """
    test_path = ROOT / "tests/test_multilevel_xai_ui.py"
    page_path = ROOT / "awareml/ui_v2/pages_multilevel_explainability.py"

    if not test_path.exists():
        return

    test_source = test_path.read_text(encoding="utf-8")
    page_source = page_path.read_text(encoding="utf-8")

    assert "integrated_explainability_page" in test_source or "def integrated_explainability_page" in page_source
    assert "multi_level_explainability_page" not in test_source
    assert "len(empty.tabs) == 5" not in test_source

    # Model-tag validation belongs on production code, not on the source of a
    # test that may legitimately mention the historical tag in an assertion.
    assert "llama3:8b" in page_source
    assert "llama3.1:8b" not in page_source
