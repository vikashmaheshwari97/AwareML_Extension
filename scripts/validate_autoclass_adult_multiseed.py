from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.engine.runner import run_benchmark
from awareml.experiments.provenance import build_dataset_provenance
from awareml.types import RunConfig


def _exact_target_copy_columns(df: pd.DataFrame, target: str) -> list[str]:
    target_values = df[target].astype(str).fillna("<NA>")
    suspicious = []
    for col in df.columns:
        if col == target:
            continue
        try:
            if df[col].astype(str).fillna("<NA>").equals(target_values):
                suspicious.append(str(col))
        except Exception:
            continue
    return suspicious


def main() -> None:
    p = argparse.ArgumentParser(description="Full-Adult, multi-seed AutoClass prequential audit for Phase 3.1.")
    p.add_argument("csv")
    p.add_argument("--target", required=True)
    p.add_argument("--sensitive", default="sex")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--max-samples", type=int, default=0, help="0 means the full dataset.")
    p.add_argument("--time-budget", type=float, default=900.0, help="Per-seed time budget in seconds.")
    p.add_argument("--output", default="artifacts/validation/autoclass_adult_multiseed.json")
    p.add_argument("--experiment-root", default="artifacts/meta_experiments")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit("Target column %r not found. Columns=%r" % (args.target, list(df.columns)))
    sensitive = args.sensitive if args.sensitive in df.columns else None
    max_samples = len(df) if int(args.max_samples) <= 0 else min(len(df), int(args.max_samples))
    provenance = build_dataset_provenance(df, args.target, sensitive, source_path=args.csv)
    suspicious = _exact_target_copy_columns(df, args.target)

    rows = []
    for seed in args.seeds:
        cfg = RunConfig(
            target=args.target,
            sensitive_attribute=sensitive,
            sensitive_feature_policy="audit_only",
            window_size=max(50, int(args.window)),
            max_samples=max_samples,
            seed=int(seed),
            time_budget_sec=float(args.time_budget),
            track_sustainability=False,
            xai_method="permutation",
            xai_replay_warning_threshold=0.05,
        )
        result = run_benchmark(
            df,
            cfg,
            frameworks=["AutoClass"],
            record_experiments=True,
            experiment_root=args.experiment_root,
            dataset_id="adult_autoclass_validation",
            protocol_version="meta-v2-phase3.1-autoclass-audit",
            experiment_nonce="seed-%d" % int(seed),
            dataset_provenance=provenance,
        )[0]
        rows.append(result.to_dict())
        print("seed=%d status=%s samples=%d accuracy=%.6f macro_f1=%.6f" % (
            seed, result.status, result.samples, result.accuracy, result.f1_macro
        ))

    accs = [float(r["accuracy"]) for r in rows if r.get("status") == "ok"]
    f1s = [float(r["f1_macro"]) for r in rows if r.get("status") == "ok"]
    summary = {
        "dataset_provenance": provenance,
        "audit": {
            "test_then_train_protocol": True,
            "sensitive_feature_policy": "audit_only" if sensitive else None,
            "exact_target_copy_columns": suspicious,
            "requested_seeds": [int(s) for s in args.seeds],
            "full_dataset_rows": int(len(df)),
            "requested_max_samples": int(max_samples),
        },
        "aggregate": {
            "n_successful": len(accs),
            "accuracy_mean": statistics.mean(accs) if accs else None,
            "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else 0.0 if accs else None,
            "macro_f1_mean": statistics.mean(f1s) if f1s else None,
            "macro_f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0 if f1s else None,
            "high_accuracy_review_flag": bool(accs and statistics.mean(accs) >= 0.95),
        },
        "runs": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("Saved:", out)
    if suspicious:
        print("WARNING: exact target-copy feature(s) detected:", suspicious)
    if summary["aggregate"]["high_accuracy_review_flag"]:
        print("REVIEW: mean prequential accuracy remains >= 0.95. Treat it as a result requiring dataset/protocol interpretation, not as an automatic failure.")


if __name__ == "__main__":
    main()
