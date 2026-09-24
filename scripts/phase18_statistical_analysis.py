#!/usr/bin/env python3
"""
Phase 18 final statistical analysis.

This script is READ-ONLY with respect to the Phase-18 empirical inputs.
It never changes framework outcomes, oracle labels, recommender predictions,
or preference-evaluation rows. It only creates new statistical summaries.

Primary inferential unit:
    one seed-aggregated dataset result (31 held-out datasets)

Main procedures:
    * 95% non-parametric bootstrap CIs over datasets
    * Friedman omnibus test across the five frameworks
    * Kendall's W omnibus effect size
    * paired Wilcoxon signed-rank comparisons
    * paired rank-biserial effect sizes
    * Holm correction within each metric
    * dataset-cluster bootstrap CIs for recommender metrics
    * dataset-cluster bootstrap CIs for objective-prediction metrics
"""

from __future__ import print_function

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as exc:
    raise SystemExit(
        "Phase-18 statistical analysis requires scipy. "
        "Install scipy in the AWAREML_PYTHON environment before running."
    ) from exc


PROTOCOL_ID = "phase18_final_heldout_31_v1"
EXPECTED_DATASETS = 31
EXPECTED_FRAMEWORKS = [
    "AutoStreamML",
    "AutoClass",
    "EvoAutoML",
    "OAML",
    "ChaCha",
]
EXPECTED_GROUND_TRUTH_ROWS = 155
EXPECTED_PREFERENCE_CASES = 3100

METRICS = {
    "accuracy": {
        "aliases": ["accuracy_mean", "accuracy"],
        "direction": "higher",
        "required": True,
    },
    "f1_macro": {
        "aliases": ["f1_macro_mean", "f1_macro", "macro_f1_mean", "macro_f1"],
        "direction": "higher",
        "required": True,
    },
    "runtime_sec": {
        "aliases": ["runtime_sec_mean", "runtime_sec", "runtime_mean_sec", "runtime"],
        "direction": "lower",
        "required": True,
    },
    "throughput_samples_sec": {
        "aliases": [
            "throughput_samples_sec_mean",
            "throughput_samples_sec",
            "throughput_mean",
            "throughput",
        ],
        "direction": "higher",
        "required": False,
    },
    "mean_prediction_latency_ms": {
        "aliases": [
            "mean_prediction_latency_ms_mean",
            "mean_prediction_latency_ms",
            "prediction_latency_ms_mean",
        ],
        "direction": "lower",
        "required": False,
    },
    "p95_prediction_latency_ms": {
        "aliases": [
            "p95_prediction_latency_ms_mean",
            "p95_prediction_latency_ms",
            "prediction_latency_p95_ms",
        ],
        "direction": "lower",
        "required": False,
    },
    "energy_kwh": {
        "aliases": ["energy_kwh_mean", "energy_kwh", "energy"],
        "direction": "lower",
        "required": True,
    },
    "co2_kg": {
        "aliases": ["co2_kg_mean", "co2_kg", "co2"],
        "direction": "lower",
        "required": True,
    },
}

PREDICTION_TARGETS = {
    "accuracy": ("accuracy", "higher"),
    "runtime": ("runtime_sec", "lower"),
    "energy": ("energy_kwh", "lower"),
    "co2": ("co2_kg", "lower"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
        ).strip()
    except Exception:
        return None


def find_existing(base, relative_candidates):
    for rel in relative_candidates:
        p = base / rel
        if p.exists():
            return p
    return None


def load_table(path):
    if path is None:
        raise FileNotFoundError("Input path is missing.")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError("Unsupported input format: %s" % path)


def resolve_column(df, aliases):
    for name in aliases:
        if name in df.columns:
            return name
    # Conservative suffix fallback, useful when collector prefixes a metric.
    for alias in aliases:
        matches = [c for c in df.columns if c.lower().endswith(alias.lower())]
        if len(matches) == 1:
            return matches[0]
    return None


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def bootstrap_ci(values, stat_name="mean", reps=10000, seed=18031):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), float(x[0])

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(reps, n))
    samples = x[idx]
    if stat_name == "mean":
        vals = np.mean(samples, axis=1)
    elif stat_name == "median":
        vals = np.median(samples, axis=1)
    else:
        raise ValueError("Unsupported bootstrap statistic: %s" % stat_name)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_custom_rows(df, estimator, reps=10000, seed=18031):
    """
    Dataset-cluster bootstrap. df must have one row per dataset cluster.
    estimator receives a bootstrapped dataframe and returns a scalar.
    """
    n = len(df)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        v = float(estimator(df.copy()))
        return v, v

    rng = np.random.RandomState(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        take = rng.randint(0, n, size=n)
        sample = df.iloc[take].reset_index(drop=True)
        vals[i] = float(estimator(sample))
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def rank_biserial_paired(oriented_diff, zero_tol=1e-15):
    d = np.asarray(oriented_diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > zero_tol]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d), method="average")
    w_plus = float(np.sum(ranks[d > 0]))
    w_minus = float(np.sum(ranks[d < 0]))
    denom = w_plus + w_minus
    if denom == 0:
        return 0.0
    return (w_plus - w_minus) / denom


def validate_ground_truth(df):
    required = ["dataset_id", "framework"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Ground-truth table missing columns: %s" % missing)

    if len(df) != EXPECTED_GROUND_TRUTH_ROWS:
        raise RuntimeError(
            "Expected %d ground-truth rows, found %d."
            % (EXPECTED_GROUND_TRUTH_ROWS, len(df))
        )

    datasets = df["dataset_id"].astype(str).nunique()
    if datasets != EXPECTED_DATASETS:
        raise RuntimeError(
            "Expected %d held-out datasets, found %d."
            % (EXPECTED_DATASETS, datasets)
        )

    dup = df.duplicated(["dataset_id", "framework"]).sum()
    if dup:
        raise RuntimeError("Duplicate dataset/framework rows found: %d" % dup)

    per_dataset = df.groupby("dataset_id")["framework"].nunique()
    bad = per_dataset[per_dataset != 5]
    if len(bad):
        raise RuntimeError(
            "Every dataset must contain five frameworks. Bad datasets: %s"
            % bad.to_dict()
        )

    observed_frameworks = set(df["framework"].astype(str).unique())
    expected = set(EXPECTED_FRAMEWORKS)
    if observed_frameworks != expected:
        raise RuntimeError(
            "Unexpected framework set. Expected %s; observed %s"
            % (sorted(expected), sorted(observed_frameworks))
        )


def resolve_metric_columns(gt):
    resolved = {}
    missing_required = []
    for metric, spec in METRICS.items():
        col = resolve_column(gt, spec["aliases"])
        if col is None:
            if spec["required"]:
                missing_required.append((metric, spec["aliases"]))
            continue
        resolved[metric] = col

    if missing_required:
        raise RuntimeError(
            "Required Phase-18 metrics are missing from ground truth: %s"
            % missing_required
        )
    return resolved


def framework_statistics(gt, metric_columns, reps):
    descriptive_rows = []
    omnibus_rows = []
    pairwise_rows = []

    for metric_index, metric in enumerate(metric_columns):
        column = metric_columns[metric]
        direction = METRICS[metric]["direction"]
        higher = direction == "higher"

        work = gt[["dataset_id", "framework", column]].copy()
        work[column] = pd.to_numeric(work[column], errors="coerce")

        # Framework-level descriptive summaries and dataset bootstrap CIs.
        for fw_index, fw in enumerate(EXPECTED_FRAMEWORKS):
            vals = work.loc[work["framework"] == fw, column].dropna().to_numpy(float)
            ci_mean = bootstrap_ci(
                vals, "mean", reps=reps, seed=18031 + metric_index * 100 + fw_index
            )
            ci_median = bootstrap_ci(
                vals, "median", reps=reps, seed=28031 + metric_index * 100 + fw_index
            )
            descriptive_rows.append(
                {
                    "scope": "combined_31",
                    "metric": metric,
                    "source_column": column,
                    "direction": direction,
                    "framework": fw,
                    "n_datasets": int(len(vals)),
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                    "median": float(np.median(vals)) if len(vals) else np.nan,
                    "mean_ci95_low": ci_mean[0],
                    "mean_ci95_high": ci_mean[1],
                    "median_ci95_low": ci_median[0],
                    "median_ci95_high": ci_median[1],
                    "ci_method": "nonparametric_dataset_bootstrap_percentile",
                    "bootstrap_reps": reps,
                }
            )

        pivot = work.pivot(index="dataset_id", columns="framework", values=column)
        pivot = pivot.reindex(columns=EXPECTED_FRAMEWORKS)
        complete = pivot.dropna(axis=0, how="any")
        n = len(complete)
        k = len(EXPECTED_FRAMEWORKS)

        if n >= 2:
            fried = stats.friedmanchisquare(
                *[complete[fw].to_numpy(float) for fw in EXPECTED_FRAMEWORKS]
            )
            chi2 = float(fried.statistic)
            pvalue = float(fried.pvalue)
            kendalls_w = chi2 / (n * (k - 1.0))
        else:
            chi2 = np.nan
            pvalue = np.nan
            kendalls_w = np.nan

        omnibus_rows.append(
            {
                "scope": "combined_31",
                "metric": metric,
                "source_column": column,
                "direction": direction,
                "n_complete_datasets": int(n),
                "frameworks": k,
                "friedman_chi2": chi2,
                "df": k - 1,
                "p_value": pvalue,
                "kendalls_w": kendalls_w,
                "alpha": 0.05,
                "significant": bool(pvalue < 0.05) if np.isfinite(pvalue) else False,
            }
        )

        metric_pairs = []
        for pair_index, (fw_a, fw_b) in enumerate(itertools.combinations(EXPECTED_FRAMEWORKS, 2)):
            pair = pivot[[fw_a, fw_b]].dropna()
            a = pair[fw_a].to_numpy(float)
            b = pair[fw_b].to_numpy(float)
            raw_diff = a - b
            oriented = raw_diff if higher else -raw_diff

            if len(oriented) == 0:
                stat_value = np.nan
                raw_p = np.nan
            elif np.all(np.abs(oriented) <= 1e-15):
                stat_value = 0.0
                raw_p = 1.0
            else:
                try:
                    w = stats.wilcoxon(
                        oriented,
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                        mode="auto",
                    )
                    stat_value = float(w.statistic)
                    raw_p = float(w.pvalue)
                except ValueError:
                    stat_value = 0.0
                    raw_p = 1.0

            rbc = rank_biserial_paired(oriented)
            mean_ci = bootstrap_ci(
                oriented,
                "mean",
                reps=reps,
                seed=38031 + metric_index * 100 + pair_index,
            )
            med_ci = bootstrap_ci(
                oriented,
                "median",
                reps=reps,
                seed=48031 + metric_index * 100 + pair_index,
            )

            wins = int(np.sum(oriented > 1e-15))
            ties = int(np.sum(np.abs(oriented) <= 1e-15))
            losses = int(np.sum(oriented < -1e-15))

            metric_pairs.append(
                {
                    "scope": "combined_31",
                    "metric": metric,
                    "source_column": column,
                    "direction": direction,
                    "framework_a": fw_a,
                    "framework_b": fw_b,
                    "n_paired_datasets": int(len(pair)),
                    "raw_mean_difference_a_minus_b": (
                        float(np.mean(raw_diff)) if len(raw_diff) else np.nan
                    ),
                    "better_oriented_mean_difference": (
                        float(np.mean(oriented)) if len(oriented) else np.nan
                    ),
                    "better_oriented_median_difference": (
                        float(np.median(oriented)) if len(oriented) else np.nan
                    ),
                    "better_oriented_mean_diff_ci95_low": mean_ci[0],
                    "better_oriented_mean_diff_ci95_high": mean_ci[1],
                    "better_oriented_median_diff_ci95_low": med_ci[0],
                    "better_oriented_median_diff_ci95_high": med_ci[1],
                    "wilcoxon_statistic": stat_value,
                    "p_value_raw": raw_p,
                    "paired_rank_biserial": rbc,
                    "a_wins": wins,
                    "ties": ties,
                    "a_losses": losses,
                    "positive_effect_means": "framework_a_better",
                }
            )

        valid_p = [r["p_value_raw"] for r in metric_pairs]
        adjusted = holm_adjust(valid_p)
        for row, adj in zip(metric_pairs, adjusted):
            row["p_value_holm"] = float(adj)
            row["significant_holm_0_05"] = bool(adj < 0.05)
            row["multiplicity_family"] = "all_10_framework_pairs_within_metric"
        pairwise_rows.extend(metric_pairs)

    return (
        pd.DataFrame(descriptive_rows),
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(pairwise_rows),
    )


def recommender_confidence_intervals(pref, reps):
    required = [
        "dataset_id",
        "top1_match",
        "top3_contains_oracle",
        "normalized_regret_span",
        "spearman_utility",
    ]
    missing = [c for c in required if c not in pref.columns]
    if missing:
        raise RuntimeError("Preference evaluation missing columns: %s" % missing)

    if len(pref) != EXPECTED_PREFERENCE_CASES:
        raise RuntimeError(
            "Expected %d primary preference cases, found %d."
            % (EXPECTED_PREFERENCE_CASES, len(pref))
        )
    if pref["dataset_id"].astype(str).nunique() != EXPECTED_DATASETS:
        raise RuntimeError("Primary preference file does not contain 31 datasets.")

    # One row per dataset cluster. Each held-out dataset has 100 preference cases.
    counts = pref.groupby("dataset_id").size()
    if not np.all(counts.to_numpy() == 100):
        raise RuntimeError(
            "Expected exactly 100 preference cases per dataset. Observed: %s"
            % counts.value_counts().to_dict()
        )

    dataset = (
        pref.groupby("dataset_id", as_index=False)
        .agg(
            top1=("top1_match", "mean"),
            top3=("top3_contains_oracle", "mean"),
            normalized_regret_mean=("normalized_regret_span", "mean"),
            spearman_mean=("spearman_utility", "mean"),
        )
    )

    specs = [
        ("top1", "top1", "higher"),
        ("top3", "top3", "higher"),
        ("normalized_regret_mean", "normalized_regret_mean", "lower"),
        ("spearman_mean", "spearman_mean", "higher"),
    ]
    rows = []
    for i, (metric, col, direction) in enumerate(specs):
        vals = dataset[col].to_numpy(float)
        estimate = float(np.nanmean(vals))
        lo, hi = bootstrap_ci(vals, "mean", reps=reps, seed=58031 + i)
        rows.append(
            {
                "analysis": "primary_preference_recommender",
                "metric": metric,
                "direction": direction,
                "estimate": estimate,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_dataset_clusters": EXPECTED_DATASETS,
                "cases": EXPECTED_PREFERENCE_CASES,
                "cluster_size": 100,
                "ci_method": "dataset_cluster_bootstrap_percentile",
                "bootstrap_reps": reps,
            }
        )
    return pd.DataFrame(rows)


def objective_prediction_confidence_intervals(gt, pred, metric_columns, winners, reps):
    required_pred = ["dataset_id", "framework", "accuracy", "runtime", "energy", "co2"]
    missing = [c for c in required_pred if c not in pred.columns]
    if missing:
        raise RuntimeError("Frozen prediction table missing columns: %s" % missing)
    if len(pred) != EXPECTED_GROUND_TRUTH_ROWS:
        raise RuntimeError("Expected 155 frozen prediction rows, found %d." % len(pred))

    merge_cols = ["dataset_id", "framework"]
    merged = gt.merge(pred, on=merge_cols, how="inner", suffixes=("_real", "_pred"))
    if len(merged) != EXPECTED_GROUND_TRUTH_ROWS:
        raise RuntimeError(
            "Ground truth/prediction merge produced %d rows; expected 155." % len(merged)
        )

    rows = []
    for i, (target, (gt_metric, direction)) in enumerate(PREDICTION_TARGETS.items()):
        gt_col = metric_columns.get(gt_metric)
        if gt_col is None:
            raise RuntimeError("Cannot find ground-truth metric for %s." % target)
        pred_col = target

        work = merged[["dataset_id", "framework", gt_col, pred_col]].copy()
        work[gt_col] = pd.to_numeric(work[gt_col], errors="coerce")
        work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
        work = work.dropna()

        def mae_est(sample_clusters):
            selected = work[work["dataset_id"].isin(sample_clusters["dataset_id"])]
            # To preserve multiplicity of resampled clusters, calculate at the
            # per-dataset level first and then average selected dataset values.
            per_ds = work.assign(
                abs_error=np.abs(work[pred_col].to_numpy() - work[gt_col].to_numpy())
            ).groupby("dataset_id")["abs_error"].mean()
            vals = per_ds.loc[sample_clusters["dataset_id"].tolist()].to_numpy(float)
            return np.mean(vals)

        def rmse_est(sample_clusters):
            sq = work.assign(
                sq_error=(work[pred_col].to_numpy() - work[gt_col].to_numpy()) ** 2
            ).groupby("dataset_id")["sq_error"].mean()
            vals = sq.loc[sample_clusters["dataset_id"].tolist()].to_numpy(float)
            return np.sqrt(np.mean(vals))

        cluster_df = pd.DataFrame(
            {"dataset_id": sorted(work["dataset_id"].astype(str).unique())}
        )

        per_ds_abs = work.assign(
            abs_error=np.abs(work[pred_col] - work[gt_col])
        ).groupby("dataset_id")["abs_error"].mean()
        per_ds_sq = work.assign(
            sq_error=(work[pred_col] - work[gt_col]) ** 2
        ).groupby("dataset_id")["sq_error"].mean()
        mae = float(per_ds_abs.mean())
        rmse = float(np.sqrt(per_ds_sq.mean()))

        mae_lo, mae_hi = bootstrap_custom_rows(
            cluster_df, mae_est, reps=reps, seed=68031 + i * 10
        )
        rmse_lo, rmse_hi = bootstrap_custom_rows(
            cluster_df, rmse_est, reps=reps, seed=68032 + i * 10
        )

        rows.append(
            {
                "target": target,
                "metric": "mae",
                "direction": "lower",
                "estimate": mae,
                "ci95_low": mae_lo,
                "ci95_high": mae_hi,
                "n_dataset_clusters": EXPECTED_DATASETS,
                "rows": len(work),
                "ci_method": "dataset_cluster_bootstrap_percentile",
                "bootstrap_reps": reps,
            }
        )
        rows.append(
            {
                "target": target,
                "metric": "rmse",
                "direction": "lower",
                "estimate": rmse,
                "ci95_low": rmse_lo,
                "ci95_high": rmse_hi,
                "n_dataset_clusters": EXPECTED_DATASETS,
                "rows": len(work),
                "ci_method": "dataset_cluster_bootstrap_percentile",
                "bootstrap_reps": reps,
            }
        )

        # Mean within-dataset Spearman ranking correlation.
        spearmans = []
        for dataset_id, part in work.groupby("dataset_id"):
            if len(part) < 2:
                continue
            corr = stats.spearmanr(
                part[gt_col].to_numpy(float), part[pred_col].to_numpy(float)
            ).correlation
            if np.isfinite(corr):
                spearmans.append(float(corr))
        sp = np.asarray(spearmans, dtype=float)
        sp_est = float(np.mean(sp)) if len(sp) else np.nan
        sp_lo, sp_hi = bootstrap_ci(
            sp, "mean", reps=reps, seed=78031 + i
        )
        rows.append(
            {
                "target": target,
                "metric": "spearman_mean",
                "direction": "higher",
                "estimate": sp_est,
                "ci95_low": sp_lo,
                "ci95_high": sp_hi,
                "n_dataset_clusters": int(len(sp)),
                "rows": len(work),
                "ci_method": "dataset_cluster_bootstrap_percentile",
                "bootstrap_reps": reps,
            }
        )

        # Use the official winner table for winner Top-1 CI if available.
        if winners is not None:
            target_w = winners[winners["objective"].astype(str) == target].copy()
            if len(target_w) == EXPECTED_DATASETS and "top1_match" in target_w.columns:
                top1_vals = pd.to_numeric(
                    target_w["top1_match"], errors="coerce"
                ).dropna().to_numpy(float)
                est = float(np.mean(top1_vals))
                lo, hi = bootstrap_ci(
                    top1_vals, "mean", reps=reps, seed=88031 + i
                )
                rows.append(
                    {
                        "target": target,
                        "metric": "objective_winner_top1",
                        "direction": "higher",
                        "estimate": est,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_dataset_clusters": int(len(top1_vals)),
                        "rows": int(len(top1_vals)),
                        "ci_method": "dataset_bootstrap_percentile",
                        "bootstrap_reps": reps,
                    }
                )

        # Prediction interval coverage where lower/upper columns exist.
        lower = target + "_lower"
        upper = target + "_upper"
        if lower in pred.columns and upper in pred.columns:
            cov_merge = gt[["dataset_id", "framework", gt_col]].merge(
                pred[["dataset_id", "framework", lower, upper]],
                on=["dataset_id", "framework"],
                how="inner",
            )
            cov_merge["covered"] = (
                (cov_merge[gt_col] >= cov_merge[lower])
                & (cov_merge[gt_col] <= cov_merge[upper])
            ).astype(float)
            per_ds_cov = cov_merge.groupby("dataset_id")["covered"].mean()
            vals = per_ds_cov.to_numpy(float)
            est = float(np.mean(vals))
            lo, hi = bootstrap_ci(
                vals, "mean", reps=reps, seed=98031 + i
            )
            rows.append(
                {
                    "target": target,
                    "metric": "prediction_interval_coverage",
                    "direction": "higher",
                    "estimate": est,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_dataset_clusters": int(len(vals)),
                    "rows": int(len(cov_merge)),
                    "ci_method": "dataset_cluster_bootstrap_percentile",
                    "bootstrap_reps": reps,
                }
            )

    return pd.DataFrame(rows)


def build_effect_sizes(omnibus, pairwise):
    rows = []
    for _, r in omnibus.iterrows():
        rows.append(
            {
                "scope": r["scope"],
                "metric": r["metric"],
                "comparison": "all_5_frameworks",
                "effect_name": "kendalls_w",
                "effect_value": r["kendalls_w"],
                "positive_direction": "not_applicable",
                "n_datasets": r["n_complete_datasets"],
            }
        )
    for _, r in pairwise.iterrows():
        rows.append(
            {
                "scope": r["scope"],
                "metric": r["metric"],
                "comparison": "%s_vs_%s" % (r["framework_a"], r["framework_b"]),
                "effect_name": "paired_rank_biserial",
                "effect_value": r["paired_rank_biserial"],
                "positive_direction": "first_framework_better",
                "n_datasets": r["n_paired_datasets"],
            }
        )
    return pd.DataFrame(rows)


def build_combined_ci(descriptive, recommender_ci, objective_ci):
    rows = []
    for _, r in descriptive.iterrows():
        rows.append(
            {
                "analysis": "framework_descriptive",
                "metric": r["metric"],
                "subgroup": r["framework"],
                "estimate": r["mean"],
                "ci95_low": r["mean_ci95_low"],
                "ci95_high": r["mean_ci95_high"],
                "n_dataset_clusters": r["n_datasets"],
                "method": r["ci_method"],
            }
        )
    for _, r in recommender_ci.iterrows():
        rows.append(
            {
                "analysis": r["analysis"],
                "metric": r["metric"],
                "subgroup": "overall",
                "estimate": r["estimate"],
                "ci95_low": r["ci95_low"],
                "ci95_high": r["ci95_high"],
                "n_dataset_clusters": r["n_dataset_clusters"],
                "method": r["ci_method"],
            }
        )
    for _, r in objective_ci.iterrows():
        rows.append(
            {
                "analysis": "objective_prediction",
                "metric": "%s:%s" % (r["target"], r["metric"]),
                "subgroup": "overall",
                "estimate": r["estimate"],
                "ci95_low": r["ci95_low"],
                "ci95_high": r["ci95_high"],
                "n_dataset_clusters": r["n_dataset_clusters"],
                "method": r["ci_method"],
            }
        )
    return pd.DataFrame(rows)


def write_summary_md(path, omnibus, pairwise, recommender_ci, objective_ci):
    sig_omnibus = int(omnibus["significant"].sum()) if len(omnibus) else 0
    sig_pair = (
        int(pairwise["significant_holm_0_05"].sum()) if len(pairwise) else 0
    )
    lines = [
        "# Phase 18 final statistical analysis",
        "",
        "Primary inferential unit: the 31 held-out datasets after seed aggregation.",
        "The 465 seed executions are not treated as 465 independent datasets.",
        "",
        "## Methods",
        "",
        "- Framework-wide omnibus comparison: Friedman test.",
        "- Omnibus effect size: Kendall's W.",
        "- Paired framework comparisons: two-sided Wilcoxon signed-rank tests.",
        "- Pairwise effect size: paired rank-biserial correlation.",
        "- Multiplicity control: Holm correction across the 10 framework pairs within each metric.",
        "- Confidence intervals: 95% non-parametric bootstrap over held-out datasets.",
        "- Recommender intervals: dataset-cluster bootstrap over the 31 held-out datasets.",
        "",
        "## Validation",
        "",
        "- Held-out datasets: 31",
        "- Frameworks: 5",
        "- Seed-aggregated dataset-framework rows: 155",
        "- Primary preference cases: 3,100",
        "- Significant Friedman omnibus tests at alpha=0.05: %d" % sig_omnibus,
        "- Holm-significant pairwise comparisons across all metric families: %d" % sig_pair,
        "",
        "This file summarizes statistical outputs only. It does not alter any Phase-18 empirical result.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        default=None,
        help=(
            "Override artifacts/phase18_final_heldout_31_v1. "
            "Normally leave unset when running from the repository root."
        ),
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
        help="Bootstrap replicates (default: 10000).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_reps < 1000:
        raise SystemExit("--bootstrap-reps must be at least 1000.")

    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = (
        Path(args.artifact_root).resolve()
        if args.artifact_root
        else repo_root / "artifacts" / PROTOCOL_ID
    )
    evaluation = artifact_root / "evaluation"
    reduced = artifact_root / "reduced"
    out_dir = evaluation / "statistical_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = find_existing(
        artifact_root,
        [
            "reduced/phase18_framework_ground_truth_155.parquet",
            "reduced/phase18_framework_ground_truth_155.csv",
        ],
    )
    if gt_path is None:
        raise SystemExit("Cannot find phase18_framework_ground_truth_155 parquet/csv.")

    pref_path = find_existing(
        artifact_root,
        [
            "evaluation/phase18_preference_eval_primary_3100.csv",
            "evaluation/phase18_preference_eval_primary_3100.parquet",
        ],
    )
    pred_path = find_existing(
        artifact_root,
        [
            "evaluation/phase18_frozen_v2_predictions_155.parquet",
            "evaluation/phase18_frozen_v2_predictions_155.csv",
        ],
    )
    winner_path = find_existing(
        artifact_root,
        [
            "evaluation/journal_tables/phase18_predicted_vs_observed_objective_winners.csv",
            "evaluation/phase18_predicted_vs_observed_objective_winners.csv",
        ],
    )

    for label, p in [
        ("ground truth", gt_path),
        ("primary preference evaluation", pref_path),
        ("frozen V2 predictions", pred_path),
    ]:
        if p is None or not p.exists():
            raise SystemExit("Required %s input is missing." % label)

    gt = load_table(gt_path)
    pref = load_table(pref_path)
    pred = load_table(pred_path)
    winners = load_table(winner_path) if winner_path is not None else None

    # Refuse to consume synthetic demo material in journal statistics.
    for p in [gt_path, pref_path, pred_path, winner_path]:
        if p is not None and "SYNTHETIC_DEMO" in p.name.upper():
            raise SystemExit(
                "Refusing to use synthetic demo file for journal statistics: %s" % p
            )
    if "data_status" in pref.columns:
        statuses = set(pref["data_status"].dropna().astype(str).str.upper())
        if "SYNTHETIC_DEMO" in statuses:
            raise SystemExit(
                "Refusing to use preference data marked SYNTHETIC_DEMO."
            )

    validate_ground_truth(gt)
    metric_columns = resolve_metric_columns(gt)

    descriptive, omnibus, pairwise = framework_statistics(
        gt, metric_columns, args.bootstrap_reps
    )
    recommender_ci = recommender_confidence_intervals(
        pref, args.bootstrap_reps
    )
    objective_ci = objective_prediction_confidence_intervals(
        gt, pred, metric_columns, winners, args.bootstrap_reps
    )
    effects = build_effect_sizes(omnibus, pairwise)
    combined_ci = build_combined_ci(descriptive, recommender_ci, objective_ci)

    outputs = {
        "phase18_framework_descriptive_ci.csv": descriptive,
        "phase18_omnibus_friedman.csv": omnibus,
        "phase18_pairwise_framework_comparisons.csv": pairwise,
        "phase18_effect_sizes.csv": effects,
        "phase18_recommender_confidence_intervals.csv": recommender_ci,
        "phase18_objective_prediction_confidence_intervals.csv": objective_ci,
        "phase18_confidence_intervals.csv": combined_ci,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False)

    summary_path = out_dir / "phase18_statistical_summary.md"
    write_summary_md(
        summary_path, omnibus, pairwise, recommender_ci, objective_ci
    )

    output_hashes = {}
    for name in list(outputs.keys()) + ["phase18_statistical_summary.md"]:
        output_hashes[name] = sha256_file(out_dir / name)

    manifest = {
        "schema_version": "1.0",
        "phase": 18,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(repo_root),
        "analysis_status": "FINAL_STATISTICAL_ANALYSIS",
        "inputs_are_read_only": True,
        "synthetic_demo_inputs_allowed": False,
        "primary_inferential_unit": "held_out_dataset_after_seed_aggregation",
        "expected_counts": {
            "datasets": EXPECTED_DATASETS,
            "frameworks": 5,
            "seed_aggregated_ground_truth_rows": EXPECTED_GROUND_TRUTH_ROWS,
            "primary_preference_cases": EXPECTED_PREFERENCE_CASES,
        },
        "methods": {
            "confidence_intervals": (
                "95% non-parametric percentile bootstrap over datasets; "
                "preference results use dataset-cluster bootstrap"
            ),
            "omnibus_test": "Friedman test across five paired frameworks",
            "omnibus_effect_size": "Kendall's W",
            "pairwise_test": "two-sided Wilcoxon signed-rank",
            "pairwise_effect_size": "paired rank-biserial correlation",
            "multiple_comparison_correction": (
                "Holm correction across 10 framework pairs within each metric"
            ),
            "alpha": 0.05,
            "bootstrap_reps": args.bootstrap_reps,
        },
        "inputs": {
            str(gt_path): sha256_file(gt_path),
            str(pref_path): sha256_file(pref_path),
            str(pred_path): sha256_file(pred_path),
        },
        "outputs": output_hashes,
        "note": (
            "This analysis adds inferential/statistical summaries only. "
            "It does not modify Phase-18 empirical outcomes, oracle labels, "
            "frozen recommender predictions, or preference-evaluation rows."
        ),
    }
    if winner_path is not None:
        manifest["inputs"][str(winner_path)] = sha256_file(winner_path)

    manifest_path = out_dir / "phase18_statistical_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("AwareML Phase 18 — final statistical analysis")
    print("=" * 78)
    print("Ground truth rows:", len(gt))
    print("Held-out datasets:", gt["dataset_id"].nunique())
    print("Frameworks:", gt["framework"].nunique())
    print("Primary preference cases:", len(pref))
    print("Resolved metrics:", ", ".join(metric_columns.keys()))
    print("Bootstrap reps:", args.bootstrap_reps)
    print("Output directory:", out_dir)
    print("")
    print("Created:")
    for name in list(outputs.keys()) + [
        "phase18_statistical_summary.md",
        "phase18_statistical_analysis_manifest.json",
    ]:
        print("  ", out_dir / name)
    print("")
    print("STATISTICAL ANALYSIS: PASS")
    print("Empirical inputs modified: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
