from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    path = ROOT / path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _json(path):
    path = ROOT / path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_ok(relative):
    manifest = _json(relative)
    if not isinstance(manifest, dict):
        return False
    base = (ROOT / relative).parent
    for filename, meta in (manifest.get("files") or {}).items():
        path = base / filename
        if not path.exists():
            return False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta.get("sha256"):
            return False
    return manifest.get("status") == "frozen"


def main():
    advanced = _read("awareml/ui_v2/pages_advanced.py")
    ui = _read("awareml/ui_v2/phase15_explanation_integrity.py")
    claims = _read("awareml/explanation_integrity/claims.py")
    verifier = _read("awareml/explanation_integrity/verifier.py")
    faith = _read("awareml/explanation_integrity/faithfulness.py")
    live = _read("awareml/explanation_integrity/live.py")

    stimulus_summary = _json(
        "data/journal/trust_stimulus_bank_v1/frozen/summary.json"
    ) or {}
    labels = stimulus_summary.get("labels") or {}
    sources = stimulus_summary.get("sources") or {}

    correctness_summary = _json(
        "data/journal/explanation_correctness_v1/frozen/summary.json"
    ) or {}
    faith_summary = _json(
        "data/journal/faithfulness_v2/frozen/summary.json"
    ) or {}

    completion = _json(
        "data/journal/phase15_completion_v1/manifest.json"
    ) or {}

    checks = {
        "phase15_ui_integrated": (
            "Explanation Integrity · Phase 15" in advanced
            and "phase15_explanation_integrity_page" in advanced
        ),
        "correctness_faithfulness_separate_in_ui": (
            "Correctness · factual agreement with structured evidence" in ui
            and "Faithfulness · response to evidence interventions" in ui
        ),
        "four_explanation_sources": all(
            token in live
            for token in [
                'source_stage="B"',
                'source_stage="E"',
                'source_stage="F_XAI"',
                'source_stage="F_CHAT"',
            ]
        ),
        "required_claim_families": all(
            token in claims
            for token in [
                '"accuracy"',
                '"runtime"',
                '"energy"',
                '"co2"',
                '"dp"',
                '"eo"',
                '"eodds"',
                '"brier_gap"',
                '"ece_gap"',
                '"shap"',
            ]
        ),
        "required_correctness_metrics": all(
            token in verifier
            for token in [
                '"claim_precision"',
                '"supported_claim_rate"',
                '"numeric_correctness"',
                '"citation_validity"',
                '"decision_consistency"',
                '"unsupported_claim_rate"',
                '"contradiction_rate"',
            ]
        ),
        "faithlm_style_external_intervention": (
            "apply_intervention" in faith
            and "changed_evidence_acknowledged" in faith
            and "stale_claim_rate" in faith
        ),
        "live_probe_marked_exploratory": (
            "Exploratory Live Dataset Probe" in ui
            and "NOT frozen journal evidence" in ui
        ),
        "stimulus_bank_balanced": (
            int(labels.get("known_correct", 0)) >= 20
            and int(labels.get("known_incorrect", 0)) >= 20
        ),
        "stimulus_bank_four_sources": len(sources) == 4,
        "stimulus_bank_multiple_contexts": (
            int(stimulus_summary.get("dataset_contexts", 0)) >= 3
        ),
        "known_correct_verifier_control": (
            (
                (correctness_summary.get("by_label") or {})
                .get("known_correct", {})
                .get("claim_precision")
            ) == 1.0
        ),
        "faithful_vs_sticky_separation": (
            float(
                (faith_summary.get("known_faithful") or {})
                .get("mean_faithfulness_score", 0.0)
            )
            >
            float(
                (faith_summary.get("known_unfaithful_sticky") or {})
                .get("mean_faithfulness_score", 1.0)
            )
        ),
        "explanation_correctness_v1_frozen": _manifest_ok(
            "data/journal/explanation_correctness_v1/frozen/manifest.json"
        ),
        "faithfulness_v2_frozen": _manifest_ok(
            "data/journal/faithfulness_v2/frozen/manifest.json"
        ),
        "trust_stimulus_bank_v1_frozen": _manifest_ok(
            "data/journal/trust_stimulus_bank_v1/frozen/manifest.json"
        ),
        "phase15_completion_manifest": (
            completion.get("status") == "frozen_controlled_completion_gate"
            and completion.get("correctness_and_faithfulness_kept_separate") is True
            and len(completion.get("artifacts") or {}) == 3
        ),
    }

    print("=" * 100)
    print("AwareML Phase-15 explanation correctness + faithfulness V2 validation")
    print("=" * 100)

    failed = []
    for name, ok in checks.items():
        print("{:<68} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    print("=" * 100)
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("Phase-15 implementation + controlled completion gate: PASS")
    print(
        "Note: frozen controlled artifacts validate methodology/stimulus ground "
        "truth. Actual LLM performance claims require a reviewed model run."
    )


if __name__ == "__main__":
    main()
