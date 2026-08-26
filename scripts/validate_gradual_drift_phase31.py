from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.engine.runner import run_benchmark
from awareml.experiments.provenance import build_dataset_provenance
from awareml.types import RunConfig


def _inconsistencies(result: dict) -> list[dict]:
    bad = []
    summary = result.get("drift_summary") or {}
    tol = float(summary.get("tolerance") or 0.0)
    for ep in summary.get("episodes") or []:
        drop = ep.get("accuracy_drop")
        flag = ep.get("degradation_observed")
        if drop is None or flag is None:
            continue
        drop = float(drop)
        if flag is False and drop > tol + 1e-12:
            bad.append({"reason": "false_flag_with_drop_above_tolerance", **ep})
        if flag is True and drop <= tol + 1e-12:
            bad.append({"reason": "true_flag_without_material_drop", **ep})
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description="Run one Phase-3.1 gradual-drift validation and assert drift-summary consistency.")
    p.add_argument("csv")
    p.add_argument("--target", required=True)
    p.add_argument("--sensitive", default=None)
    p.add_argument("--frameworks", nargs="+", default=["AutoStreamML", "ChaCha"])
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--time-budget", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="artifacts/validation/gradual_drift_phase31.json")
    p.add_argument("--experiment-root", default="artifacts/meta_experiments")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    sensitive = args.sensitive if args.sensitive and args.sensitive in df.columns else None
    provenance = build_dataset_provenance(df, args.target, sensitive, source_path=args.csv)
    cfg = RunConfig(
        target=args.target,
        sensitive_attribute=sensitive,
        sensitive_feature_policy="audit_only",
        window_size=max(50, int(args.window)),
        max_samples=min(len(df), int(args.max_samples)),
        seed=int(args.seed),
        time_budget_sec=float(args.time_budget),
        track_sustainability=False,
        xai_method="permutation",
        drift_min_assessment_samples=max(25, min(100, int(args.window) // 5)),
    )
    results = run_benchmark(
        df,
        cfg,
        frameworks=args.frameworks,
        record_experiments=True,
        experiment_root=args.experiment_root,
        dataset_id="hyperplane_gradual_drift_validation",
        protocol_version="meta-v2-phase3.1-drift-audit",
        experiment_nonce="seed-%d" % int(args.seed),
        dataset_provenance=provenance,
    )
    payload = []
    total_bad = 0
    for result in results:
        d = result.to_dict()
        bad = _inconsistencies(d)
        total_bad += len(bad)
        payload.append({
            "framework": result.framework,
            "backend": result.backend,
            "status": result.status,
            "accuracy": result.accuracy,
            "f1_macro": result.f1_macro,
            "drift_summary": result.drift_summary,
            "inconsistencies": bad,
        })
        print("%-14s status=%-7s drift_events=%d inconsistencies=%d" % (
            result.framework, result.status, len(result.drift_events or []), len(bad)
        ))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset_provenance": provenance, "results": payload}, indent=2, default=str), encoding="utf-8")
    print("Saved:", out)
    if total_bad:
        raise SystemExit("Phase 3.1 drift validation FAILED: %d inconsistent episode(s)." % total_bad)
    print("Phase 3.1 gradual-drift validation: OK")


if __name__ == "__main__":
    main()
