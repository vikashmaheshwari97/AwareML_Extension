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


def grounded_information_answer(
    question: str,
    category: Optional[str],
    results: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
) -> Tuple[str, List[str], List[str]]:
    """Create a deterministic, participant-readable grounded response.

    Multiple requested evidence topics are answered independently so a question
    asking about accuracy, fairness and sustainability cannot collapse to only
    one branch of a keyword router.
    """
    q = str(question or "").strip()
    topics = detect_requested_topics(q)
    comparison = (
        category == "counterfactual_or_comparison"
        or "compare" in q.lower()
        or " versus " in q.lower()
        or " vs " in q.lower()
    )
    targets = _target_frameworks(q, results, ranking, comparison=comparison)

    if not targets:
        return (
            "No measured framework evidence is available for this study context.",
            topics,
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
        return "\n\n".join(blocks), topics, targets

    ordered = _ranking_rows(ranking)
    top_name = targets[0]
    top_run = _row_for(results, top_name)
    top_rank = _rank_row(ranking, top_name)

    if comparison and len(targets) >= 2:
        left, right = targets[:2]
        return (
            "**Current-run comparison.** {}\n\n{} Lower runtime, energy and CO₂ are better; "
            "higher accuracy is better. Utility reflects the current Decision Lab weights."
        ).format(
            _core_summary(left, _row_for(results, left), _rank_row(ranking, left)),
            _core_summary(right, _row_for(results, right), _rank_row(ranking, right)),
        ), topics, targets[:2]

    if category == "challenge":
        text = "**Current recommendation.** {}".format(_core_summary(top_name, top_run, top_rank))
        if len(ordered) > 1:
            alt = str(ordered[1].get("framework"))
            text += "\n\n**Strongest ranked alternative.** {}".format(
                _core_summary(alt, _row_for(results, alt), _rank_row(ranking, alt))
            )
        text += "\n\nThe recommendation is not treated as unquestionably correct; you can inspect or compare the measured evidence before deciding."
        return text, topics, targets

    if category == "clarification":
        ql = q.lower()
        if "utility" in ql:
            return (
                "Utility is the normalized weighted multi-objective score used by Decision Lab to rank frameworks. "
                "It is not the same as accuracy; it combines the currently selected objectives and weights.",
                topics,
                targets,
            )
        return (
            "The recommendation summarizes measured evidence from the current AwareML run. "
            "You can ask about accuracy/performance, fairness, sustainability, drift/recovery, "
            "or explainability, and the study will show only the evidence available for this run.",
            topics,
            targets,
        )

    return (
        "**Why this framework is currently ranked first.** {}\n\n"
        "This describes the current observed run under the current Decision Lab weights. "
        "You may request a specific evidence area or compare it with another framework."
    ).format(_core_summary(top_name, top_run, top_rank)), topics, targets
