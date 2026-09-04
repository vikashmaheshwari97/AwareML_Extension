from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


OBJECTIVES = ("Accuracy", "Runtime", "Energy", "CO2")
DIAGNOSTIC_ARTIFACT = "objective_selection_v31_posthoc_development_diagnostic"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def diagnostic_root(root: Path) -> Path:
    return root / "data" / "journal" / "objective_selection_v31_diagnostic"


def frozen_phase12_manifest(root: Path) -> Path:
    return (
        root
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "frozen"
        / "manifest.json"
    )


def ground_truth_path(root: Path) -> Path:
    return (
        root
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "results"
        / "ground_truth.jsonl"
    )


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def multilabel_metrics(
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    pred_by_id = {str(row["scenario_id"]): row for row in prediction_rows}
    per_counts = {
        objective: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for objective in OBJECTIVES
    }
    exact = 0
    jaccards = []
    by_k = {}

    for truth in truth_rows:
        sid = str(truth["scenario_id"])
        if sid not in pred_by_id:
            raise ValueError("Missing V3.1 prediction for {}".format(sid))
        true_set = set(truth.get("ground_truth_objectives") or [])
        pred_set = set(pred_by_id[sid].get("selected_objectives") or [])
        is_exact = true_set == pred_set
        exact += int(is_exact)
        union = true_set | pred_set
        jac = len(true_set & pred_set) / float(len(union)) if union else 1.0
        jaccards.append(jac)

        k = int(truth.get("k_prime") or len(true_set))
        bucket = by_k.setdefault(k, {"n": 0, "exact": 0, "jaccards": []})
        bucket["n"] += 1
        bucket["exact"] += int(is_exact)
        bucket["jaccards"].append(jac)

        for objective in OBJECTIVES:
            t = objective in true_set
            p = objective in pred_set
            if t and p:
                per_counts[objective]["tp"] += 1
            elif (not t) and p:
                per_counts[objective]["fp"] += 1
            elif t and (not p):
                per_counts[objective]["fn"] += 1
            else:
                per_counts[objective]["tn"] += 1

    per_objective = {}
    total_tp = total_fp = total_fn = 0
    for objective in OBJECTIVES:
        counts = per_counts[objective]
        score = _prf(counts["tp"], counts["fp"], counts["fn"])
        per_objective[objective] = {**counts, **score}
        total_tp += counts["tp"]
        total_fp += counts["fp"]
        total_fn += counts["fn"]

    micro = _prf(total_tp, total_fp, total_fn)
    macro_f1 = sum(row["f1"] for row in per_objective.values()) / len(OBJECTIVES)
    n = len(truth_rows)
    return {
        "artifact": DIAGNOSTIC_ARTIFACT,
        "n": n,
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1": macro_f1,
        "exact_match_rate": exact / float(n) if n else 0.0,
        "mean_jaccard": sum(jaccards) / float(len(jaccards)) if jaccards else 0.0,
        "per_objective": per_objective,
        "by_k_prime": {
            str(k): {
                "n": row["n"],
                "exact_match_rate": row["exact"] / float(row["n"]),
                "mean_jaccard": sum(row["jaccards"]) / float(len(row["jaccards"])),
            }
            for k, row in sorted(by_k.items())
        },
        "interpretation": "posthoc_development_diagnostic_not_independent_test",
    }


def baseline_metrics(root: Path) -> Dict[str, Any]:
    path = frozen_phase12_manifest(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(payload.get("metrics") or {})
    paraphrase = dict(payload.get("paraphrase_summary") or {})
    adversarial = dict(payload.get("adversarial_summary") or {})
    return {
        "manifest_sha256": sha256_file(path),
        "micro_precision": metrics.get("micro_precision"),
        "micro_recall": metrics.get("micro_recall"),
        "micro_f1": metrics.get("micro_f1"),
        "macro_f1": metrics.get("macro_f1"),
        "exact_match_rate": metrics.get("exact_match_rate"),
        "paraphrase_consistency_rate": paraphrase.get(
            "paraphrase_prediction_consistency_rate"
        ),
        "adversarial_taxonomy_counts": adversarial.get("taxonomy_counts") or {},
        "release_status": payload.get("release_status"),
    }
