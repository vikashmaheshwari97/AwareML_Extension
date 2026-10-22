from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOPIC_LABELS = {
    "performance": "Accuracy & performance",
    "fairness": "Fairness",
    "sustainability": "Sustainability",
    "drift": "Drift & recovery",
    "explainability": "Explainability",
}


def detect_requested_topics(question: str) -> List[str]:
    q = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    topics: List[str] = []

    def add(name: str) -> None:
        if name not in topics:
            topics.append(name)

    if any(token in q for token in ["accuracy", "f1", "performance", "runtime", "latency", "throughput"]):
        add("performance")
    if any(token in q for token in ["fair", "parity", "equal opportunity", "equalized odds", "brier", "ece", "worst-group"]):
        add("fairness")
    if any(token in q for token in ["sustain", "energy", "co2", "co₂", "carbon", "emission", "green"]):
        add("sustainability")
    if any(token in q for token in ["drift", "recovery", "temporal", "adwin"]):
        add("drift")
    if any(token in q for token in ["xai", "explainability", "shap", "lime", "feature importance", "fidelity", "stability"]):
        add("explainability")
    return topics


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number) != 0 and (abs(number) < 0.001 or abs(number) >= 10000):
        return "{:.4g}".format(number)
    return ("{:.%df}" % digits).format(number).rstrip("0").rstrip(".")


def _accuracy(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
        return "{} ({:.1f}%)".format(_fmt(number, 4), number * 100.0)
    except Exception:
        return str(value)


def _ranking_rows(ranking: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(ranking or [])
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("rank")) if row.get("rank") is not None else 1e9,
            -float(row.get("utility")) if row.get("utility") is not None else 0.0,
        ),
    )


def _mentioned_frameworks(question: str, results: List[Dict[str, Any]]) -> List[str]:
    q = str(question or "").lower()
    names = [str(row.get("framework")) for row in results if row.get("framework")]
    return [name for name in names if name.lower() in q]


def _target_frameworks(
    question: str,
    results: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
    comparison: bool,
) -> List[str]:
    mentioned = _mentioned_frameworks(question, results)
    available = [str(row.get("framework")) for row in results if row.get("framework")]
    ordered = [
        str(row.get("framework"))
        for row in _ranking_rows(ranking)
        if row.get("framework") in available
    ]

    targets: List[str] = []
    for name in mentioned + ordered + available:
        if name and name not in targets:
            targets.append(name)
        if len(targets) >= (2 if comparison else 1):
            break
    return targets


def _row_for(results: List[Dict[str, Any]], framework: str) -> Dict[str, Any]:
    for row in results:
        if str(row.get("framework")) == str(framework):
            return row
    return {}


def _rank_row(ranking: List[Dict[str, Any]], framework: str) -> Dict[str, Any]:
    for row in ranking or []:
        if str(row.get("framework")) == str(framework):
            return row
    return {}


def _topic_text(topic: str, framework: str, run: Dict[str, Any], rank: Dict[str, Any]) -> str:
    if topic == "performance":
        return (
            "**Accuracy & performance — {fw}.** Accuracy: {acc}; Macro-F1: {f1}; "
            "runtime: {runtime} s. Higher accuracy/F1 is better; lower runtime is better."
        ).format(
            fw=framework,
            acc=_accuracy(run.get("accuracy", rank.get("accuracy"))),
            f1=_fmt(run.get("f1_macro"), 4),
            runtime=_fmt(run.get("runtime_sec", rank.get("runtime_sec")), 4),
        )

    if topic == "fairness":
        fair = run.get("fairness") or {}
        return (
            "**Fairness — {fw}.** Demographic parity gap: {dp}; equal opportunity gap: {eo}; "
            "equalized odds gap: {eod}; predictive parity gap: {pp}; error-rate gap: {erg}; "
            "group Brier-score gap: {brier}; group calibration-error gap: {ece}. "
            "For these disparity gaps, values closer to 0 generally indicate smaller group differences. "
            "Unavailable values remain N/A."
        ).format(
            fw=framework,
            dp=_fmt(fair.get("dp_diff"), 5),
            eo=_fmt(fair.get("equal_opportunity_diff"), 5),
            eod=_fmt(fair.get("equalized_odds_gap"), 5),
            pp=_fmt(fair.get("predictive_parity_diff"), 5),
            erg=_fmt(fair.get("error_rate_gap"), 5),
            brier=_fmt(fair.get("group_brier_score_gap"), 5),
            ece=_fmt(fair.get("group_ece_gap"), 5),
        )

    if topic == "sustainability":
        return (
            "**Sustainability — {fw}.** Measured energy: {energy} kWh; CO₂: {co2} kg; "
            "runtime: {runtime} s. Lower measured energy, CO₂ and runtime indicate lower resource use "
            "for this run; missing measurements are reported as N/A rather than zero."
        ).format(
            fw=framework,
            energy=_fmt(run.get("energy_kwh", rank.get("energy_kwh")), 7),
            co2=_fmt(run.get("co2_kg", rank.get("co2_kg")), 7),
            runtime=_fmt(run.get("runtime_sec", rank.get("runtime_sec")), 4),
        )

    if topic == "drift":
        drift = run.get("drift_summary") or {}
        events = run.get("drift_events")
        if isinstance(events, (list, tuple)):
            count = len(events)
        elif events is None:
            count = None
        else:
            count = events
        return (
            "**Drift & recovery — {fw}.** Recorded drift events: {count}; recovery rate: {rate}; "
            "median recovery samples: {median}; mean accuracy drop: {drop}. "
            "N/A means the current run did not provide that recovery measurement."
        ).format(
            fw=framework,
            count="N/A" if count is None else count,
            rate=_fmt(drift.get("recovery_rate"), 4),
            median=_fmt(drift.get("median_recovery_samples"), 2),
            drop=_fmt(drift.get("mean_accuracy_drop"), 4),
        )

    exp = run.get("explainability") or {}
    return (
        "**Explainability — {fw}.** Method: {method}; status: {status}; fidelity: {fidelity}; "
        "stability: {stability}; consistency: {consistency}; top feature: {top}. "
        "Only values recorded in the current run are shown."
    ).format(
        fw=framework,
        method=exp.get("method") or "N/A",
        status=exp.get("status") or "N/A",
        fidelity=_fmt(exp.get("fidelity"), 4),
        stability=_fmt(exp.get("stability"), 4),
        consistency=_fmt(exp.get("consistency"), 4),
        top=exp.get("top_feature") or "N/A",
    )


def _core_summary(framework: str, run: Dict[str, Any], rank: Dict[str, Any]) -> str:
    return (
        "{fw}: utility {utility}; accuracy {acc}; runtime {runtime} s; energy {energy} kWh; CO₂ {co2} kg."
    ).format(
        fw=framework,
        utility=_fmt(rank.get("utility"), 4),
        acc=_accuracy(run.get("accuracy", rank.get("accuracy"))),
        runtime=_fmt(run.get("runtime_sec", rank.get("runtime_sec")), 4),
        energy=_fmt(run.get("energy_kwh", rank.get("energy_kwh")), 7),
        co2=_fmt(run.get("co2_kg", rank.get("co2_kg")), 7),
    )


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _pct_change(lower: Optional[float], higher: Optional[float]) -> Optional[float]:
    if lower is None or higher is None or higher == 0:
        return None
    return (higher - lower) / abs(higher) * 100.0


def _asks_for_alternatives(question: str) -> bool:
    q = str(question or "").lower()
    return any(
        token in q
        for token in [
            "alternatives",
            "alternative options",
            "other options",
            "other frameworks",
            "what else",
            "next choices",
        ]
    )


def _asks_energy_co2_relationship(question: str) -> bool:
    q = str(question or "").lower()
    has_energy = "energy" in q
    has_co2 = "co2" in q or "co₂" in q or "carbon" in q or "emission" in q
    relation = any(
        token in q
        for token in [
            "which is better",
            "what is better",
            "better co2",
            "better energy",
            "co2 or energy",
            "co₂ or energy",
            "energy or co2",
            "energy or co₂",
            "difference between",
            "what is the difference",
        ]
    )
    return has_energy and has_co2 and relation


def _simple_recommendation_explanation(
    framework: str,
    run: Dict[str, Any],
    rank: Dict[str, Any],
) -> str:
    return (
        "**In simple terms:** AwareML is not saying {fw} is best at every single metric. "
        "It is ranked first because, under the current Decision Lab preference weights, it has the "
        "highest combined utility ({utility}). In this measured run it achieved {acc} accuracy, "
        "took {runtime} s, used {energy} kWh, and produced {co2} kg CO₂. "
        "You can inspect any one of those objectives separately or compare {fw} with another framework."
    ).format(
        fw=framework,
        utility=_fmt(rank.get("utility"), 4),
        acc=_accuracy(run.get("accuracy", rank.get("accuracy"))),
        runtime=_fmt(run.get("runtime_sec", rank.get("runtime_sec")), 4),
        energy=_fmt(run.get("energy_kwh", rank.get("energy_kwh")), 7),
        co2=_fmt(run.get("co2_kg", rank.get("co2_kg")), 7),
    )


def _sustainability_relationship_text(
    framework: str,
    run: Dict[str, Any],
    rank: Dict[str, Any],
) -> str:
    energy = run.get("energy_kwh", rank.get("energy_kwh"))
    co2 = run.get("co2_kg", rank.get("co2_kg"))
    runtime = run.get("runtime_sec", rank.get("runtime_sec"))
    return (
        "**Energy and CO₂ answer different questions — neither is universally 'better'.**\n\n"
        "- **Energy** is how much electricity the framework consumed. Lower is better when your goal is resource efficiency.\n"
        "- **CO₂** is the estimated climate impact associated with that energy under the recorded carbon intensity. "
        "Lower is better when your goal is emissions reduction.\n\n"
        "For **{fw}** in this run: energy = **{energy} kWh**, CO₂ = **{co2} kg**, runtime = **{runtime} s**. "
        "The energy and CO₂ numbers have different units, so they should not be compared numerically with each other. "
        "Use energy to compare electricity use and CO₂ to compare carbon impact."
    ).format(
        fw=framework,
        energy=_fmt(energy, 7),
        co2=_fmt(co2, 7),
        runtime=_fmt(runtime, 4),
    )


def _pairwise_comparison_text(
    left: str,
    right: str,
    results: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
) -> str:
    lrun, rrun = _row_for(results, left), _row_for(results, right)
    lrank, rrank = _rank_row(ranking, left), _rank_row(ranking, right)

    lacc = _num(lrun.get("accuracy", lrank.get("accuracy")))
    racc = _num(rrun.get("accuracy", rrank.get("accuracy")))
    lruntime = _num(lrun.get("runtime_sec", lrank.get("runtime_sec")))
    rruntime = _num(rrun.get("runtime_sec", rrank.get("runtime_sec")))
    lenergy = _num(lrun.get("energy_kwh", lrank.get("energy_kwh")))
    renergy = _num(rrun.get("energy_kwh", rrank.get("energy_kwh")))
    lco2 = _num(lrun.get("co2_kg", lrank.get("co2_kg")))
    rco2 = _num(rrun.get("co2_kg", rrank.get("co2_kg")))

    lines = [
        "**Top-2 measured comparison — {left} vs {right}.**".format(left=left, right=right),
        "- **Utility:** {left} {lu} vs {right} {ru}. The ranking uses the current Decision Lab weights.".format(
            left=left, right=right, lu=_fmt(lrank.get("utility"), 4), ru=_fmt(rrank.get("utility"), 4)
        ),
    ]
    if lacc is not None and racc is not None:
        diff_pp = (lacc - racc) * 100.0
        leader = left if diff_pp >= 0 else right
        lines.append(
            "- **Accuracy:** {left} {la:.1f}% vs {right} {ra:.1f}% — {leader} is higher by {d:.2f} percentage points.".format(
                left=left, right=right, la=lacc * 100.0, ra=racc * 100.0, leader=leader, d=abs(diff_pp)
            )
        )
    if lruntime is not None and rruntime is not None and min(lruntime, rruntime) > 0:
        low_name, low, high_name, high = (left, lruntime, right, rruntime) if lruntime <= rruntime else (right, rruntime, left, lruntime)
        lines.append(
            "- **Runtime:** {left} {lv:.4g} s vs {right} {rv:.4g} s — {low_name} is about {ratio:.2f}× faster on this run.".format(
                left=left, right=right, lv=lruntime, rv=rruntime, low_name=low_name, ratio=high / low
            )
        )
    if lenergy is not None and renergy is not None and max(lenergy, renergy) > 0:
        low_name, low, high_name, high = (left, lenergy, right, renergy) if lenergy <= renergy else (right, renergy, left, lenergy)
        reduction = _pct_change(low, high)
        lines.append(
            "- **Energy:** {left} {lv} kWh vs {right} {rv} kWh — {low_name} uses {pct:.1f}% less measured energy.".format(
                left=left, right=right, lv=_fmt(lenergy, 7), rv=_fmt(renergy, 7), low_name=low_name, pct=0.0 if reduction is None else reduction
            )
        )
    if lco2 is not None and rco2 is not None and max(lco2, rco2) > 0:
        low_name, low, high_name, high = (left, lco2, right, rco2) if lco2 <= rco2 else (right, rco2, left, lco2)
        reduction = _pct_change(low, high)
        lines.append(
            "- **CO₂:** {left} {lv} kg vs {right} {rv} kg — {low_name} produces {pct:.1f}% less measured CO₂.".format(
                left=left, right=right, lv=_fmt(lco2, 7), rv=_fmt(rco2, 7), low_name=low_name, pct=0.0 if reduction is None else reduction
            )
        )
    lines.append(
        "**How to read this:** the first-ranked framework is the best match to the current weighted objectives, not necessarily the best on every individual metric."
    )
    return "\n".join(lines)


def _alternatives_text(
    results: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    ordered = _ranking_rows(ranking)
    if len(ordered) <= 1:
        return "No additional ranked alternatives are available in this study context.", []
    top_name = str(ordered[0].get("framework") or "the current recommendation")
    alternatives: List[str] = []
    lines = ["**Ranked alternatives to {}.**".format(top_name)]
    for row in ordered[1:]:
        framework = str(row.get("framework") or "")
        if not framework:
            continue
        alternatives.append(framework)
        lines.append(
            "- **#{rank} {fw}:** {summary}".format(
                rank=row.get("rank") if row.get("rank") is not None else len(alternatives) + 1,
                fw=framework,
                summary=_core_summary(framework, _row_for(results, framework), row),
            )
        )
    lines.append(
        "These are the other frameworks in the current measured ranking. They are alternatives, not new recommendations; you can ask to compare any one of them with {}.".format(top_name)
    )
    return "\n".join(lines), alternatives


def grounded_information_answer(
    question: str,
    category: Optional[str],
    results: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
) -> Tuple[str, List[str], List[str]]:
    """Create a deterministic, participant-readable grounded response.

    This v2 router distinguishes a pairwise comparison from a request for all
    alternatives and answers common conceptual questions in plain language while
    remaining grounded in the measured study context.
    """
    q = str(question or "").strip()
    ql = q.lower()
    topics = detect_requested_topics(q)
    ordered = _ranking_rows(ranking)

    if not results or not ordered:
        return (
            "No measured framework evidence is available for this study context.",
            topics,
            [],
        )

    if _asks_for_alternatives(q):
        text, alternatives = _alternatives_text(results, ranking)
        return text, topics, alternatives

    comparison = (
        category == "counterfactual_or_comparison"
        or "compare" in ql
        or " versus " in ql
        or " vs " in ql
        or "top 2" in ql
        or "top two" in ql
        or "second best" in ql
    )
    targets = _target_frameworks(q, results, ranking, comparison=comparison)
    if not targets:
        return (
            "No measured framework evidence is available for this study context.",
            topics,
            targets,
        )

    if _asks_energy_co2_relationship(q) and len(targets) == 1:
        framework = targets[0]
        return (
            _sustainability_relationship_text(
                framework,
                _row_for(results, framework),
                _rank_row(ranking, framework),
            ),
            ["sustainability"],
            targets,
        )

    if topics:
        blocks: List[str] = []
        for topic in topics:
            if comparison and len(targets) >= 2:
                blocks.append("### {}".format(TOPIC_LABELS.get(topic, topic.title())))
                for framework in targets[:2]:
                    blocks.append(
                        _topic_text(
                            topic,
                            framework,
                            _row_for(results, framework),
                            _rank_row(ranking, framework),
                        )
                    )
            else:
                framework = targets[0]
                blocks.append(
                    _topic_text(
                        topic,
                        framework,
                        _row_for(results, framework),
                        _rank_row(ranking, framework),
                    )
                )
        return "\n\n".join(blocks), topics, targets[:2] if comparison else targets

    top_name = str(ordered[0].get("framework"))
    top_run = _row_for(results, top_name)
    top_rank = _rank_row(ranking, top_name)

    if comparison:
        compare_targets = targets[:2]
        if len(compare_targets) < 2 and len(ordered) >= 2:
            compare_targets = [str(ordered[0].get("framework")), str(ordered[1].get("framework"))]
        if len(compare_targets) >= 2:
            return (
                _pairwise_comparison_text(compare_targets[0], compare_targets[1], results, ranking),
                topics,
                compare_targets,
            )

    if category == "challenge":
        text = "**Current recommendation.** {}".format(_core_summary(top_name, top_run, top_rank))
        if len(ordered) > 1:
            alt = str(ordered[1].get("framework"))
            text += "\n\n**Strongest ranked alternative.** {}".format(
                _core_summary(alt, _row_for(results, alt), _rank_row(ranking, alt))
            )
        text += (
            "\n\nThe recommendation is not treated as unquestionably correct. It reflects the current "
            "weights and measured evidence, so a different priority can favor another framework."
        )
        return text, topics, [top_name]

    if category == "clarification":
        if "utility" in ql:
            return (
                "**Utility** is the combined score used by Decision Lab after applying the current objective weights. "
                "It is not the same as accuracy and it does not mean a framework is best on every metric.",
                topics,
                [top_name],
            )
        return (
            _simple_recommendation_explanation(top_name, top_run, top_rank),
            topics,
            [top_name],
        )

    if category == "evidence_request":
        return (
            "**Measured evidence supporting the current recommendation.** {}\n\n"
            "These are observed values from the standardized AwareML study context. You can ask for fairness, "
            "sustainability, drift/recovery, explainability, or a comparison with another framework.".format(
                _core_summary(top_name, top_run, top_rank)
            ),
            topics,
            [top_name],
        )

    return (
        "**Why this framework is currently ranked first.** {}\n\n"
        "It has the highest utility under the current Decision Lab weights. That is a weighted trade-off, "
        "not a claim that it wins every individual metric."
    ).format(_core_summary(top_name, top_run, top_rank)), topics, [top_name]
