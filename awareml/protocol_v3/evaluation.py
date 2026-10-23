from __future__ import annotations

import itertools
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from .campaign import preference_bank
from .collection import load_collection
from .common import (OBJECTIVES, TARGETS, desirability, digest, read_json, sha256,
                     utc_now, utilities, weights_vector, write_json)
from .models import load_bundle, predict

FIXED_DEVELOPMENT_PROFILES = [
    {"preference_id": "accuracy", "accuracy": .70, "runtime": .10, "energy": .10, "co2": .10},
    {"preference_id": "balanced", "accuracy": .25, "runtime": .25, "energy": .25, "co2": .25},
    {"preference_id": "speed", "accuracy": .20, "runtime": .50, "energy": .15, "co2": .15},
    {"preference_id": "sustainability", "accuracy": .20, "runtime": .10, "energy": .35, "co2": .35},
]


def observed_array(frame, frameworks):
    block = frame.set_index("framework").loc[list(frameworks)]
    return block[[TARGETS[t] + "_mean" for t in OBJECTIVES]].to_numpy(dtype=float)


def training_baselines(reference, frameworks, sbs_preferences):
    matrices = [observed_array(part, frameworks) for _, part in reference.groupby("dataset_id", sort=True)]
    normalized = np.mean([desirability(a) for a in matrices], axis=0)
    raw = np.mean(matrices, axis=0)
    weights = np.asarray([weights_vector(p) for p in sbs_preferences])
    expected_utility = normalized.dot(weights.T).mean(axis=1)
    return raw, normalized, int(np.argmax(expected_utility))


def recommendation_cases(observed, predicted, reference, frameworks, preferences, sbs_preferences, development=False):
    rows = []
    for dataset_id, part in observed.groupby("dataset_id", sort=True):
        ref = reference.loc[~reference.dataset_id.eq(dataset_id)] if development else reference
        if set(ref.dataset_id) & {dataset_id}:
            raise ValueError("Evaluation dataset leaked into baseline fitting.")
        means, prior_scores, sbs = training_baselines(ref, frameworks, sbs_preferences)
        true = observed_array(part, frameworks)
        pred = predicted.loc[predicted.dataset_id.eq(dataset_id)].set_index("framework").loc[list(frameworks)]
        pv = pred[list(OBJECTIVES)].to_numpy(dtype=float)
        demo = pred[["demo_" + t for t in OBJECTIVES]].to_numpy(dtype=float)
        for preference in preferences:
            w = weights_vector(preference)
            true_u = utilities(true, w)
            oracle = int(np.argmax(true_u))
            span = float(np.ptp(true_u))
            candidate_scores = {"AwareML_V3": utilities(pv, w), "FrameworkMean": utilities(means, w),
                                "Prior": prior_scores.dot(w), "DemoXGBoostRefit": utilities(demo, w)}
            orders = {name: np.argsort(-values, kind="stable") for name, values in candidate_scores.items()}
            orders["SBS"] = np.asarray([sbs] + [i for i in range(len(frameworks)) if i != sbs])
            orders["VBS"] = np.argsort(-true_u, kind="stable")
            for name in list(orders) + ["Random"]:
                if name == "Random":
                    chosen, value = None, float(np.mean(true_u))
                    top1, top3 = 1 / len(frameworks), min(3, len(frameworks)) / len(frameworks)
                    tie_aware = float(np.isclose(true_u, true_u[oracle], atol=1e-12, rtol=0).sum()) / len(frameworks)
                    rho = 0.0 if span > 1e-12 else None
                else:
                    order = orders[name]
                    chosen = int(order[0])
                    value = float(true_u[chosen])
                    top1, top3 = float(chosen == oracle), float(oracle in order[:3])
                    tie_aware = float(abs(value - true_u[oracle]) <= 1e-12)
                    # A fixed SBS specifies no meaningful ordering of unselected
                    # alternatives, so no full-ranking correlation is assigned.
                    score = candidate_scores.get(name)
                    if name == "VBS":
                        score = true_u
                    rho = float(stats.spearmanr(score, true_u).statistic) if score is not None and np.ptp(score) > 1e-12 and span > 1e-12 else None
                rows.append({"dataset_id": dataset_id, "preference_id": preference["preference_id"], "method": name,
                             "recommended_framework": frameworks[chosen] if chosen is not None else "uniform_expectation",
                             "oracle_framework": frameworks[oracle], "selected_utility": value,
                             "oracle_utility": float(true_u[oracle]), "sbs_utility": float(true_u[sbs]),
                             "top1_exact": top1, "top1_tie_aware": tie_aware, "top3": top3,
                             "normalized_regret": (float(true_u[oracle]) - value) / span if span > 1e-12 else 0.0,
                             "spearman": rho})
    return pd.DataFrame(rows)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    result = np.empty(len(p))
    largest = 0.0
    for i, idx in enumerate(np.argsort(p)):
        largest = max(largest, min(1.0, (len(p) - i) * p[idx]))
        result[idx] = largest
    return result


def bootstrap_mean(values, repetitions=5000, seed=42):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None, None
    rng = np.random.RandomState(seed)
    means = a[rng.randint(0, len(a), size=(int(repetitions), len(a)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def recommender_statistics(cases):
    metrics = ["selected_utility", "top1_exact", "top1_tie_aware", "top3", "normalized_regret", "spearman"]
    dataset_means = cases.groupby(["dataset_id", "method"], as_index=False)[metrics].mean()
    summaries = []
    for name, part in dataset_means.groupby("method", sort=True):
        mean_u, oracle, sbs = float(part.selected_utility.mean()), float(cases.oracle_utility.mean()), float(cases.sbs_utility.mean())
        gap = (mean_u - sbs) / (oracle - sbs) if abs(oracle - sbs) > 1e-12 else None
        for metric in metrics:
            lo, hi = bootstrap_mean(part[metric])
            summaries.append({"method": name, "metric": metric, "mean": float(part[metric].mean()) if part[metric].notna().any() else None,
                              "ci95_low": lo, "ci95_high": hi, "n_datasets": len(part), "closed_gap": gap})
    comparisons = []
    for metric in ("selected_utility", "top1_exact", "normalized_regret"):
        pivot = dataset_means.pivot(index="dataset_id", columns="method", values=metric)
        family = []
        for baseline in pivot.columns:
            if baseline in ("AwareML_V3", "VBS"):
                continue
            difference = (pivot.AwareML_V3 - pivot[baseline]).to_numpy(dtype=float)
            difference[np.abs(difference) < 1e-12] = 0.0
            test = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto") if np.any(difference) else None
            family.append({"metric": metric, "baseline": baseline, "n_datasets": len(difference),
                           "mean_V3_minus_baseline": float(difference.mean()), "p_value": float(test.pvalue) if test else 1.0,
                           "W": float(test.statistic) if test else 0.0})
        adjusted = holm([r["p_value"] for r in family])
        for row, p in zip(family, adjusted):
            comparisons.append({**row, "p_holm": float(p)})
    return dataset_means, pd.DataFrame(summaries), pd.DataFrame(comparisons)


def framework_statistics(aggregates, frameworks, output):
    output = Path(output)
    omnibus, pairwise = [], []
    for metric in ("accuracy", "f1_macro", "runtime_sec", "energy_kwh", "co2_kg", "peak_process_tree_rss_mib"):
        pivot = aggregates.pivot(index="dataset_id", columns="framework", values=metric + "_mean").loc[:, list(frameworks)]
        matrix = pivot.to_numpy(dtype=float)
        if len(matrix) < 2 or not np.isfinite(matrix).all() or len(frameworks) < 3:
            continue
        maximize = metric in ("accuracy", "f1_macro")
        ranks = np.asarray([stats.rankdata(-a if maximize else a, method="average") for a in matrix])
        means = ranks.mean(axis=0)
        all_tied = bool(np.all(np.ptp(matrix, axis=1) < 1e-12))
        fried = None if all_tied else stats.friedmanchisquare(*[matrix[:, i] for i in range(len(frameworks))])
        chi2, p = (0.0, 1.0) if fried is None else (float(fried.statistic), float(fried.pvalue))
        n, k = matrix.shape
        se = math.sqrt(k * (k + 1) / (6.0 * n))
        cd = float(stats.studentized_range.ppf(.95, k, np.inf) / math.sqrt(2) * se)
        omnibus.append({"metric": metric, "n_datasets": n, "frameworks": k, "friedman_chi2": chi2, "p_value": p,
                        "kendall_W": chi2 / (n * (k - 1)), "CD_alpha_0_05": cd})
        for i, j in itertools.combinations(range(k), 2):
            q = abs(float(means[i] - means[j])) / se * math.sqrt(2)
            pairwise.append({"metric": metric, "framework_a": frameworks[i], "framework_b": frameworks[j],
                             "mean_rank_a": means[i], "mean_rank_b": means[j],
                             "nemenyi_p": float(stats.studentized_range.sf(q, k, np.inf)),
                             "posthoc_after_significant_friedman": p < .05})
        draw_cd(means, frameworks, cd, output / (metric + "_critical_difference.svg"), p)
    return pd.DataFrame(omnibus), pd.DataFrame(pairwise)


def draw_cd(means, frameworks, cd, path, omnibus_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(means)
    scores = np.asarray(means)[order]
    labels = [frameworks[i] for i in order]
    k = len(scores)
    groups = []
    for start in range(k):
        end = max(j for j in range(start, k) if scores[j] - scores[start] <= cd + 1e-12)
        if end > start:
            groups.append((start, end))
    groups = [g for g in groups if not any(h != g and h[0] <= g[0] and h[1] >= g[1] for h in groups)]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot([1, k], [0, 0], color="black", linewidth=1)
    for rank in range(1, k + 1):
        ax.plot([rank, rank], [-.035, .035], color="black")
        ax.text(rank, -.13, str(rank), ha="center", va="top")
    left_count = (k + 1) // 2
    for i, (score, label) in enumerate(zip(scores, labels)):
        left = i < left_count
        level = i if left else k - 1 - i
        y = .58 + .32 * level
        edge = .55 if left else k + .45
        ax.plot([score, score, edge], [0, y, y], color="#405766", linewidth=.9)
        ax.text(edge, y + .09, "{}  {:.2f}".format(label, score), ha="left" if left else "right", fontsize=10)
    for i, (a, b) in enumerate(groups):
        y = .15 + .10 * i
        ax.plot([scores[a], scores[b]], [y, y], color="#172f43", linewidth=4, solid_capstyle="butt")
    # The CD can exceed the rank-axis width on very small dataset cohorts.
    ax.plot([1, 1 + cd], [-.50, -.50], color="#172f43", linewidth=2)
    for x in (1, 1 + cd):
        ax.plot([x, x], [-.55, -.45], color="#172f43")
    ax.text(1 + cd / 2, -.62, "CD = {:.3f}".format(cd), ha="center", fontsize=10)
    ax.set_xlim(.35, max(k + .65, 1 + cd + .2))
    ax.set_ylim(1.65, -.85)
    ax.axis("off")
    ax.set_title("Average rank (lower is better) · Friedman p = {:.3g}".format(omnibus_p), loc="left", fontsize=12)
    fig.text(.5, .025, "Thick bars join maximal groups with rank difference no larger than the Nemenyi CD (α = .05).", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def objective_statistics(observed, predicted, frameworks, manifest, external=True):
    rows = []
    joined = observed.merge(predicted, on=["dataset_id", "framework"], validate="one_to_one")
    for target in OBJECTIVES:
        errors = joined[target].to_numpy() - joined[TARGETS[target] + "_mean"].to_numpy()
        joined["absolute"] = np.abs(errors)
        joined["squared"] = errors ** 2
        radius = manifest["uncertainty"][target]["absolute_radius"]
        lo = np.maximum(0, joined[target].to_numpy() - radius)
        hi = joined[target].to_numpy() + radius
        if target == "accuracy":
            hi = np.minimum(1, hi)
        truth = joined[TARGETS[target] + "_mean"].to_numpy()
        joined["covered"] = (truth >= lo) & (truth <= hi)
        per_dataset = joined.groupby("dataset_id")[["absolute", "squared", "covered"]].mean()
        mae_low, mae_high = bootstrap_mean(per_dataset.absolute)
        rows.append({"objective": target, "MAE": float(per_dataset.absolute.mean()),
                     "RMSE": float(np.sqrt(per_dataset.squared.mean())), "MAE_ci95_low": mae_low, "MAE_ci95_high": mae_high,
                     "interval_coverage": float(per_dataset.covered.mean()) if external else None, "mean_interval_width": float(np.mean(hi - lo)),
                     "nominal_coverage": .90, "n_datasets": len(per_dataset)})
    return pd.DataFrame(rows)


def evaluate(bundle_dir, output, evaluation_collection=None, complete_case=False):
    manifest, models = load_bundle(bundle_dir)
    bundle_dir, output = Path(bundle_dir), Path(output)
    if output.exists():
        raise ValueError("Evaluation directory exists; choose a new path.")
    metadata = manifest["feature_set"] != "prefix"
    frameworks = manifest["frameworks"]
    reference = pd.read_csv(bundle_dir / "development_aggregates.csv", keep_default_na=False)
    cfg = manifest["protocol"]
    bank = preference_bank(cfg["preferences_per_dataset"], cfg["preference_seed"])
    if evaluation_collection is None:
        observed = reference
        predicted = pd.read_csv(bundle_dir / "nested_predictions.csv")
        preferences = FIXED_DEVELOPMENT_PROFILES
        development = True
        role = "nested_LODO_development_four_predeclared_profiles"
    else:
        observed, payload = load_collection(evaluation_collection, complete_case)
        if set(observed.cohort) != {"eval"}:
            raise ValueError("External evaluation accepts evaluation-only collections.")
        if digest(payload["frozen"]["protocol"]) != digest(cfg):
            raise ValueError("Training and evaluation benchmark protocols differ.")
        if digest(payload["frozen"]["measurement_hardware"]) != digest(manifest["measurement_hardware"]):
            raise ValueError("Development and evaluation measurements use different CPU models or thread allocations.")
        dev_sources = {d["canonical_sha256"] for d in manifest["training_sources"]}
        test_sources = {d["source_canonical_sha256"] for d in payload["frozen"]["dataset_provenance"]}
        if dev_sources & test_sources or set(observed.dataset_id) & set(manifest["training_dataset_ids"]):
            raise ValueError("Development/evaluation overlap detected.")
        observed = observed.loc[observed.framework.isin(frameworks)].copy()
        if set(observed.framework) != set(frameworks) or not all(observed.groupby("dataset_id").framework.nunique().eq(len(frameworks))):
            raise ValueError("Evaluation collection does not contain the complete model portfolio.")
        predicted = observed[["dataset_id", "framework"]].copy()
        for target in OBJECTIVES:
            predicted[target] = predict(models[target], observed, target, metadata)
            entry = manifest["demo_models"][target]
            demo = joblib.load(bundle_dir / entry["file"])
            predicted["demo_" + target] = predict(demo, observed, target, metadata)
        preferences, development = bank, False
        role = "revised_external_evaluation_of_previously_evaluated_cohort"
    cases = recommendation_cases(observed, predicted, reference, frameworks, preferences, bank, development)
    dataset_means, summary, comparisons = recommender_statistics(cases)
    output.mkdir(parents=True)
    cases.to_csv(output / "preference_cases.csv", index=False)
    predicted.to_csv(output / "predictions.csv", index=False)
    dataset_means.to_csv(output / "dataset_method_means.csv", index=False)
    summary.to_csv(output / "baseline_summary.csv", index=False)
    comparisons.to_csv(output / "paired_wilcoxon_holm.csv", index=False)
    omnibus, nemenyi = framework_statistics(observed, frameworks, output)
    omnibus.to_csv(output / "framework_friedman.csv", index=False)
    nemenyi.to_csv(output / "framework_nemenyi.csv", index=False)
    objective_statistics(observed, predicted, frameworks, manifest, external=not development).to_csv(output / "objective_errors_and_coverage.csv", index=False)
    strata = []
    if evaluation_collection is not None:
        # Derived regression-to-classification tasks must remain distinguishable
        # from the native classification cohort, including in combined reports.
        provenance = payload["frozen"]["dataset_provenance"]
        native_ids = {d["dataset_id"] for d in provenance if d["task_policy"] == "native_classification"}
        for label, ids in (("native_classification", native_ids),
                           ("derived_classification", set(observed.dataset_id) - native_ids)):
            part = cases.loc[cases.dataset_id.isin(ids)]
            if part.empty:
                continue
            directory = output / label
            directory.mkdir()
            per_dataset, strat_summary, strat_tests = recommender_statistics(part)
            per_dataset.to_csv(directory / "dataset_method_means.csv", index=False)
            strat_summary.to_csv(directory / "baseline_summary.csv", index=False)
            strat_tests.to_csv(directory / "paired_wilcoxon_holm.csv", index=False)
            strata.append({"cohort": label, "n_datasets": int(part.dataset_id.nunique())})
    evaluation = {"created_utc": utc_now(), "role": role, "bundle_manifest_sha256": sha256(bundle_dir / "manifest.json"),
                  "preference_bank_sha256": digest(preferences),
                  "observed_outcomes_sha256": digest(observed.sort_values(["dataset_id", "framework"])[["dataset_id", "framework"] + [TARGETS[t] + "_mean" for t in OBJECTIVES]].to_dict("records")),
                  "feature_set": manifest["feature_set"], "portfolio": manifest["portfolio"],
                  "n_datasets": int(observed.dataset_id.nunique()), "preferences_per_dataset": len(preferences),
                  "n_preference_cases_per_method": int(observed.dataset_id.nunique()) * len(preferences),
                  "task_policy_strata": strata,
                  "inferential_unit": "dataset", "tie_rule": "first in frozen framework order; exact and tie-aware Top1 both exported",
                  "random_rule": "exact uniform expectation; no Monte Carlo noise",
                  "closed_gap": "ratio of dataset-weighted mean utilities; undefined denominator is null; negative values retained",
                  "file_hashes": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}}
    write_json(output / "evaluation.json", evaluation)
    return evaluation


def compare_ablation(prefix_eval, metadata_eval, output):
    a, b = read_json(Path(prefix_eval) / "evaluation.json"), read_json(Path(metadata_eval) / "evaluation.json")
    if (a["role"], a["portfolio"], a["n_datasets"], a["preferences_per_dataset"]) != (b["role"], b["portfolio"], b["n_datasets"], b["preferences_per_dataset"]):
        raise ValueError("Ablation evaluation protocols differ.")
    for key in ("preference_bank_sha256", "observed_outcomes_sha256"):
        if a[key] != b[key]:
            raise ValueError("Ablation did not use the same frozen preferences and observed outcomes.")
    left = pd.read_csv(Path(prefix_eval) / "preference_cases.csv")
    right = pd.read_csv(Path(metadata_eval) / "preference_cases.csv")
    left = left.loc[left.method.eq("AwareML_V3")]
    right = right.loc[right.method.eq("AwareML_V3")]
    joined = left.merge(right, on=["dataset_id", "preference_id"], suffixes=("_prefix", "_metadata"), validate="one_to_one")
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError("Ablation cases do not match exactly.")
    rows = []
    for metric in ("top1_exact", "normalized_regret", "selected_utility"):
        joined["difference"] = joined[metric + "_prefix"] - joined[metric + "_metadata"]
        differences = joined.groupby("dataset_id").difference.mean()
        lo, hi = bootstrap_mean(differences)
        rows.append({"metric": metric, "mean_prefix_minus_metadata": differences.mean(), "ci95_low": lo,
                     "ci95_high": hi, "n_datasets": len(differences)})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return rows
