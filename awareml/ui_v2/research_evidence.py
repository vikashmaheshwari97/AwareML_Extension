from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_phase12_objective_selection_reliability(
    root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load aggregate, frozen Phase-12 calibration evidence for the Copilot UI.

    This function is read-only. It never changes the frozen benchmark and the
    returned values are aggregate benchmark evidence, not confidence estimates
    for the current natural-language sentence.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    manifest = (
        base
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "frozen"
        / "manifest.json"
    )
    if not manifest.exists():
        return None

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None

    metrics = payload.get("metrics") or {}
    paraphrase = payload.get("paraphrase_summary") or {}
    adversarial = payload.get("adversarial_summary") or {}
    taxonomy = adversarial.get("taxonomy_counts") or {}
    primary_failure = None
    if taxonomy:
        primary_failure = sorted(
            taxonomy.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[0][0]

    consistency = paraphrase.get("paraphrase_prediction_consistency_rate")
    set_change = None
    if consistency is not None:
        try:
            set_change = 1.0 - float(consistency)
        except Exception:
            set_change = None

    return {
        "artifact": payload.get("artifact"),
        "model": payload.get("journal_model"),
        "benchmark_cases": metrics.get("n") or payload.get("primary_benchmark_n"),
        "micro_f1": metrics.get("micro_f1"),
        "macro_f1": metrics.get("macro_f1"),
        "exact_match_rate": metrics.get("exact_match_rate"),
        "micro_precision": metrics.get("micro_precision"),
        "micro_recall": metrics.get("micro_recall"),
        "paraphrase_consistency_rate": consistency,
        "paraphrase_set_change_rate": set_change,
        "paraphrase_evaluations": paraphrase.get("paraphrase_evaluations"),
        "adversarial_cases": adversarial.get("n"),
        "primary_failure_tendency": primary_failure,
        "release_status": payload.get("release_status"),
        "human_ground_truth": payload.get("human_ground_truth"),
        "manifest_sha256": _sha256(manifest),
    }


def load_phase12_v3_posthoc_diagnostic(
    root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load optional V3 post-hoc development diagnostics for UI comparison.

    These values must never be presented as an independent replacement for the
    frozen Phase-12 v1 benchmark. They exist only to show whether the improved
    evidence-grounded selector is moving in the intended direction before a
    fresh independently annotated evaluation is created.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    path = (
        base
        / "data"
        / "journal"
        / "objective_selection_v3_diagnostic"
        / "results"
        / "v3_comparison_report.json"
    )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    v3 = dict(payload.get("v3_posthoc") or {})
    para = dict(payload.get("paraphrase_v3") or {})
    adv = dict(payload.get("adversarial_v3") or {})
    return {
        "micro_precision": v3.get("micro_precision"),
        "micro_recall": v3.get("micro_recall"),
        "micro_f1": v3.get("micro_f1"),
        "macro_f1": v3.get("macro_f1"),
        "exact_match_rate": v3.get("exact_match_rate"),
        "paraphrase_consistency_rate": para.get(
            "paraphrase_prediction_consistency_rate"
        ),
        "adversarial_over_selection_count": adv.get("over_selection_count"),
        "adversarial_cases": adv.get("cases"),
        "evaluation_role": payload.get("evaluation_role"),
        "warning": payload.get("warning"),
        "report_sha256": _sha256(path),
    }


def load_phase12_v31_posthoc_diagnostic(
    root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load V3.1 post-hoc development diagnostics for transparent UI display."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    path = (
        base
        / "data"
        / "journal"
        / "objective_selection_v31_diagnostic"
        / "results"
        / "v31_comparison_report.json"
    )
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    v31 = dict(payload.get("v31_posthoc") or {})
    para = dict(payload.get("paraphrase_v31") or {})
    adv = dict(payload.get("adversarial_v31") or {})
    return {
        "micro_precision": v31.get("micro_precision"),
        "micro_recall": v31.get("micro_recall"),
        "micro_f1": v31.get("micro_f1"),
        "macro_f1": v31.get("macro_f1"),
        "exact_match_rate": v31.get("exact_match_rate"),
        "paraphrase_consistency_rate": para.get(
            "paraphrase_prediction_consistency_rate"
        ),
        "paraphrase_evaluations": para.get("paraphrase_evaluations"),
        "adversarial_over_selection_count": adv.get("over_selection_count"),
        "adversarial_cases": adv.get("cases"),
        "evaluation_role": payload.get("evaluation_role"),
        "warning": payload.get("warning"),
        "report_sha256": _sha256(path),
    }
