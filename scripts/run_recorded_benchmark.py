from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.engine.runner import run_benchmark
from awareml.experiments.provenance import build_dataset_provenance
from awareml.types import RunConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Run a Phase-3.1 recorded AwareML benchmark.")
    p.add_argument("csv")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--sensitive")
    p.add_argument(
        "--sensitive-feature-policy",
        choices=["audit_only", "include"],
        default="audit_only",
        help="audit_only excludes the protected attribute from model inputs while retaining it for fairness measurement.",
    )
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--time-budget", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frameworks", nargs="*", default=None)
    p.add_argument("--track-sustainability", action="store_true")
    p.add_argument("--xai-method", choices=["auto", "shap", "lime", "permutation"], default="auto")
    p.add_argument("--xai-max-rows", type=int, default=250)
    p.add_argument("--xai-replay-warning-threshold", type=float, default=0.05, help="Warn when final-model replay accuracy differs from rolling prequential accuracy by at least this amount.")
    p.add_argument("--near-constant-threshold", type=float, default=0.95, help="Flag prediction streams whose dominant predicted class reaches this fraction.")
    p.add_argument("--drift-assessment", type=int, default=None, help="Minimum post-drift samples before recovery can be declared.")
    p.add_argument("--oaml-mode", choices=["online", "gama"], default=None)
    p.add_argument("--oaml-gama-budget", type=int, default=None)
    p.add_argument("--experiment-root", default="artifacts/meta_experiments")
    p.add_argument("--protocol-version", default="meta-v2-phase3.1")
    p.add_argument("--nonce", default=None)
    p.add_argument("--output", default="artifacts/phase31_last_results.json")
    args = p.parse_args()

    if args.oaml_mode:
        os.environ["AWAREML_OAML_MODE"] = args.oaml_mode
    if args.oaml_gama_budget is not None:
        os.environ["AWAREML_OAML_GAMA_BUDGET"] = str(max(2, int(args.oaml_gama_budget)))

    df = pd.read_csv(args.csv)
    provenance = build_dataset_provenance(
        df,
        target=args.target,
        sensitive_attribute=args.sensitive,
        source_path=args.csv,
    )
    cfg = RunConfig(
        target=args.target,
        sensitive_attribute=args.sensitive,
        sensitive_feature_policy=args.sensitive_feature_policy,
        window_size=args.window,
        max_samples=args.max_samples,
        seed=args.seed,
        time_budget_sec=args.time_budget,
        track_sustainability=args.track_sustainability,
        xai_method=args.xai_method,
        xai_max_rows=max(30, int(args.xai_max_rows)),
        xai_replay_warning_threshold=max(0.0, float(args.xai_replay_warning_threshold)),
        prediction_near_constant_threshold=float(args.near_constant_threshold),
        drift_min_assessment_samples=args.drift_assessment,
    )
    results = run_benchmark(
        df,
        cfg,
        frameworks=args.frameworks,
        record_experiments=True,
        experiment_root=args.experiment_root,
        dataset_id=args.dataset_id,
        protocol_version=args.protocol_version,
        experiment_nonce=args.nonce,
        dataset_provenance=provenance,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Recorded {len(results)} framework result(s).")
    for r in results:
        print(
            f"{r.framework:14s} status={r.status:7s} acc={r.accuracy:.4f} "
            f"f1={r.f1_macro:.4f} p95={r.p95_prediction_latency_ms} "
            f"exp={r.experiment_id}"
        )
    print(f"Summary JSON: {out}")
    print(f"Experiment root: {args.experiment_root}")


if __name__ == "__main__":
    main()
