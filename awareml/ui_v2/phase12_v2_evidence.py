from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_phase12_v2_confirmatory_evidence(root: Path) -> Optional[Dict[str, Any]]:
    """Load only finalized, checksum-valid Phase-12-v2 confirmatory evidence."""
    root = Path(root)
    manifest = (
        root
        / "data"
        / "journal"
        / "objective_selection_benchmark_v2"
        / "frozen"
        / "manifest.json"
    )
    sha_path = Path(str(manifest) + ".sha256")
    if not manifest.exists() or not sha_path.exists():
        return None

    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = _sha256(manifest)
    if not expected or expected.lower() != actual.lower():
        return None

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("release_status") != "frozen":
        return None

    baseline = dict(payload.get("baseline_primary_metrics") or {})
    v32 = dict(payload.get("v32_primary_metrics") or {})
    paired = dict(payload.get("paired_primary_comparison") or {})
    if not baseline or not v32:
        return None

    return {
        "release_status": payload.get("release_status"),
        "frozen_at_utc": payload.get("frozen_at_utc"),
        "benchmark_cases": payload.get("primary_benchmark_n"),
        "ground_truth_source": payload.get("ground_truth_source"),
        "manifest_sha256": actual,
        "baseline": {
            "role": "Primary confirmatory evidence",
            "name": "V2 fresh replay",
            "reporting_name": "Phase-11 V2 method replay under Phase-11R confirmatory runtime",
            "exact_match_rate": baseline.get("exact_match_rate"),
            "micro_precision": baseline.get("micro_precision"),
            "micro_recall": baseline.get("micro_recall"),
            "micro_f1": baseline.get("micro_f1"),
            "macro_f1": baseline.get("macro_f1"),
            "mean_jaccard": baseline.get("mean_jaccard"),
            "over_selection_any_fp_rate": baseline.get("over_selection_any_fp_rate"),
            "under_selection_any_fn_rate": baseline.get("under_selection_any_fn_rate"),
        },
        "v32": {
            "role": "Fresh confirmatory candidate",
            "name": "V3.2",
            "exact_match_rate": v32.get("exact_match_rate"),
            "micro_precision": v32.get("micro_precision"),
            "micro_recall": v32.get("micro_recall"),
            "micro_f1": v32.get("micro_f1"),
            "macro_f1": v32.get("macro_f1"),
            "mean_jaccard": v32.get("mean_jaccard"),
            "malformed_rate": v32.get("malformed_rate"),
            "valid_status_rate": v32.get("valid_status_rate"),
        },
        "paired": {
            "mcnemar_exact": paired.get("mcnemar_exact"),
            "deltas_v32_minus_baseline": paired.get("deltas_v32_minus_baseline"),
        },
        "v31_evidence_role": "Development/post-hoc only",
        "v33_evidence_role": "Interactive development successor; not confirmatory evidence",
    }
