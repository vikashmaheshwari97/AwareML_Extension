from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .claims import ClaimExtractor, extract_citations
from .evidence import (
    LOWER_IS_BETTER,
    evidence_numeric_candidates,
    flatten_evidence,
    key_metric,
    normalize_key,
    numeric_close,
    ranking_rows,
    resolve_flat_key,
    top_feature,
    top_ranked_framework,
)
from .schemas import (
    Claim,
    ClaimVerification,
    CorrectnessReport,
    EvidenceCase,
)


def _safe_mean(values):
    clean = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    if not clean:
        return None
    return float(np.mean(clean))


def _ratio(numerator: int, denominator: int):
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _citation_expected_value(
    flat: Mapping[str, Any],
    claim: Claim,
) -> Optional[Tuple[str, float]]:
    for key in claim.cited_keys:
        normalized = normalize_key(key)
        resolved_key = resolve_flat_key(flat, normalized)
        if resolved_key is None:
            continue
        if claim.metric and key_metric(resolved_key) != claim.metric:
            continue
        value = flat.get(resolved_key)
        try:
            numeric = float(value)
        except Exception:
            continue
        if not np.isfinite(numeric):
            continue
        if claim.entity and claim.entity.lower() not in normalized.lower():
            # SHAP citations are feature-based instead of framework-based.
            if claim.metric != "shap":
                continue
        if claim.feature and claim.feature.lower() not in normalized.lower():
            feature_ok = False
            if claim.metric == "feature_importance":
                parts = resolved_key.split(".")
                if "feature_importance" in parts:
                    try:
                        idx = parts.index("feature_importance")
                    except ValueError:
                        idx = -1
                    if (
                        idx >= 0
                        and idx + 1 < len(parts)
                        and parts[idx + 1].isdigit()
                    ):
                        prefix = ".".join(parts[: idx + 2])
                        sibling_feature = flat.get(prefix + ".feature")
                        sibling_name = flat.get(prefix + ".name")
                        feature_ok = (
                            str(
                                sibling_feature
                                or sibling_name
                                or ""
                            ).lower()
                            == claim.feature.lower()
                        )
            if not feature_ok:
                continue
        return resolved_key, numeric
    return None


def _resolve_numeric(
    claim: Claim,
    flat: Mapping[str, Any],
) -> Optional[Tuple[str, float]]:
    cited = _citation_expected_value(flat, claim)
    if cited is not None:
        return cited

    candidates = evidence_numeric_candidates(
        flat,
        claim.metric or "",
        entity=claim.entity,
        feature=claim.feature,
    )
    if len(candidates) == 1:
        return candidates[0]

    # Fairness/calibration live evidence can contain both a canonical
    # framework-level fairness value and nested copies in temporal summaries.
    # Prefer the direct canonical fairness path before generic tie-breaking.
    if (
        candidates
        and claim.entity
        and claim.metric in {"dp", "eo", "eodds", "brier_gap", "ece_gap"}
    ):
        prefix = (
            "evidence.frameworks.{}.fairness."
            .format(str(claim.entity))
            .lower()
        )
        direct = [
            (key, value)
            for key, value in candidates
            if key.lower().startswith(prefix)
            and key.lower().count(".") == prefix.count(".")
        ]
        if len(direct) == 1:
            return direct[0]

        if direct and claim.numeric_value is not None:
            ranked_direct = sorted(
                direct,
                key=lambda item: abs(
                    float(item[1]) - float(claim.numeric_value)
                ),
            )
            if len(ranked_direct) == 1:
                return ranked_direct[0]
            if (
                abs(
                    float(ranked_direct[0][1])
                    - float(claim.numeric_value)
                )
                + 1e-15
                <
                abs(
                    float(ranked_direct[1][1])
                    - float(claim.numeric_value)
                )
            ):
                return ranked_direct[0]

    # If multiple candidates exist, prefer exact suffix/entity matches.
    if candidates:
        scored = []
        for key, value in candidates:
            score = 0
            lowered = key.lower()
            if claim.entity and claim.entity.lower() in lowered:
                score += 3
            if claim.feature and claim.feature.lower() in lowered:
                score += 3
            if claim.metric and key_metric(key) == claim.metric:
                score += 2
            scored.append((score, key, value))
        scored.sort(reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1], scored[0][2]

        # If structural scores tie, use the reported numeric value only as a
        # disambiguator inside the already-matched metric/entity family. This
        # avoids declaring a correct canonical point claim unsupported merely
        # because the evidence tree contains duplicated summary copies.
        if scored and claim.numeric_value is not None:
            best_score = scored[0][0]
            tied = [
                (key, value)
                for score, key, value in scored
                if score == best_score
            ]
            if tied:
                tied.sort(
                    key=lambda item: abs(
                        float(item[1]) - float(claim.numeric_value)
                    )
                )
                if len(tied) == 1:
                    return tied[0]
                first_distance = abs(
                    float(tied[0][1]) - float(claim.numeric_value)
                )
                second_distance = abs(
                    float(tied[1][1]) - float(claim.numeric_value)
                )
                if first_distance + 1e-15 < second_distance:
                    return tied[0]

                # If duplicate keys carry effectively the same expected value,
                # choosing the shortest canonical path is deterministic and
                # does not change the truth value of the claim.
                same_value = all(
                    numeric_close(
                        tied[0][1],
                        item[1],
                        abs_tol=1e-12,
                        rel_tol=1e-9,
                    )
                    for item in tied[1:]
                )
                if same_value:
                    tied.sort(key=lambda item: len(item[0]))
                    return tied[0]
    return None


def _ranking_expected_rank(
    evidence: Mapping[str, Any],
    entity: str,
) -> Optional[int]:
    rows = ranking_rows(evidence)
    for row in rows:
        if str(row.get("framework", "")).lower() != str(entity).lower():
            continue
        rank = row.get("rank")
        if rank is not None:
            try:
                return int(rank)
            except Exception:
                pass

    top = top_ranked_framework(evidence)
    if top and top.lower() == str(entity).lower():
        return 1
    return None


def _metric_value_for_entity(
    flat: Mapping[str, Any],
    metric: str,
    entity: str,
) -> Optional[float]:
    candidates = evidence_numeric_candidates(
        flat,
        metric,
        entity=entity,
    )
    if not candidates:
        return None
    # Prefer shortest key as the least nested canonical metric.
    candidates = sorted(candidates, key=lambda item: len(item[0]))
    return float(candidates[0][1])


def _comparative_truth(
    claim: Claim,
    flat: Mapping[str, Any],
) -> Optional[bool]:
    left = claim.entity
    right = claim.feature  # compact schema: second framework is stored here.
    metric = claim.metric
    comparator = (claim.comparator or "").lower()
    if not left or not right or not metric:
        return None

    left_value = _metric_value_for_entity(flat, metric, left)
    right_value = _metric_value_for_entity(flat, metric, right)
    if left_value is None or right_value is None:
        return None

    if comparator in {"higher", "greater"}:
        return bool(left_value > right_value)
    if comparator in {"lower", "smaller"}:
        return bool(left_value < right_value)
    if comparator == "faster":
        return bool(left_value < right_value)
    if comparator == "slower":
        return bool(left_value > right_value)
    if comparator == "better":
        if metric in LOWER_IS_BETTER:
            return bool(left_value < right_value)
        return bool(left_value > right_value)
    if comparator == "worse":
        if metric in LOWER_IS_BETTER:
            return bool(left_value > right_value)
        return bool(left_value < right_value)
    return None


class GeneralEvidenceVerifier:
    """Claim-level factual verifier for AwareML explanation evidence.

    Correctness is evaluated only against structured evidence supplied with the
    explanation case. Faithfulness is intentionally handled by a separate
    intervention evaluator.
    """

    def __init__(
        self,
        abs_tolerance: float = 1e-4,
        relative_tolerance: float = 0.01,
        extractor: Optional[ClaimExtractor] = None,
    ):
        self.abs_tolerance = float(abs_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.extractor = extractor or ClaimExtractor()

    def verify(
        self,
        case: EvidenceCase,
        explanation: str,
    ) -> CorrectnessReport:
        evidence = case.evidence
        flat = flatten_evidence(evidence)
        claims = self.extractor.extract(explanation, evidence)
        citations = extract_citations(explanation)
        valid_keys = set(flat.keys())

        verified: List[ClaimVerification] = []

        for claim in claims:
            citation_valid = None
            if claim.cited_keys:
                citation_valid = all(
                    resolve_flat_key(flat, key) is not None
                    for key in claim.cited_keys
                )

            if claim.claim_type == "numeric":
                resolved = _resolve_numeric(claim, flat)
                if resolved is None:
                    verified.append(
                        ClaimVerification(
                            claim=claim,
                            status="unsupported",
                            supported=False,
                            contradicted=False,
                            unsupported=True,
                            numeric_correct=None,
                            citation_valid=citation_valid,
                            reason=(
                                "No unique structured evidence value matched "
                                "this metric/entity claim."
                            ),
                        )
                    )
                    continue

                key, expected = resolved
                correct = numeric_close(
                    claim.numeric_value,
                    expected,
                    abs_tol=self.abs_tolerance,
                    rel_tol=self.relative_tolerance,
                )
                verified.append(
                    ClaimVerification(
                        claim=claim,
                        status="supported" if correct else "contradicted",
                        supported=bool(correct),
                        contradicted=not bool(correct),
                        unsupported=False,
                        numeric_correct=bool(correct),
                        citation_valid=citation_valid,
                        matched_evidence_key=key,
                        expected_value=expected,
                        observed_value=claim.numeric_value,
                        reason=(
                            None
                            if correct
                            else "Numeric claim does not match structured evidence."
                        ),
                    )
                )
                continue

            if claim.claim_type == "availability":
                if claim.metric == "calibration":
                    calibration_values = []
                    for key, value in flat.items():
                        if key_metric(key) not in {"brier_gap", "ece_gap"}:
                            continue
                        try:
                            numeric = float(value)
                        except Exception:
                            continue
                        if np.isfinite(numeric):
                            calibration_values.append((key, numeric))

                    expected_available = bool(calibration_values)
                    observed_available = (
                        str(claim.comparator or "").lower() == "available"
                    )
                    consistent = observed_available == expected_available

                    verified.append(
                        ClaimVerification(
                            claim=claim,
                            status="supported" if consistent else "contradicted",
                            supported=bool(consistent),
                            contradicted=not bool(consistent),
                            unsupported=False,
                            citation_valid=citation_valid,
                            decision_consistent=None,
                            expected_value=(
                                "available"
                                if expected_available
                                else "unavailable"
                            ),
                            observed_value=claim.comparator,
                            reason=(
                                None
                                if consistent
                                else (
                                    "Calibration-availability claim contradicts "
                                    "structured Brier/ECE evidence."
                                )
                            ),
                        )
                    )
                    continue

            if claim.claim_type == "categorical":
                if claim.metric == "ranking_mode":
                    expected = None
                    recommendation = (
                        evidence.get("recommendation")
                        if isinstance(evidence, Mapping)
                        else None
                    )
                    if isinstance(recommendation, Mapping):
                        expected = recommendation.get("ranking_mode")

                    if expected is None:
                        verified.append(
                            ClaimVerification(
                                claim=claim,
                                status="unsupported",
                                supported=False,
                                contradicted=False,
                                unsupported=True,
                                citation_valid=citation_valid,
                                reason="No structured ranking-mode evidence is available.",
                            )
                        )
                    else:
                        observed = str(claim.comparator or "")
                        consistent = observed.lower() == str(expected).lower()
                        verified.append(
                            ClaimVerification(
                                claim=claim,
                                status="supported" if consistent else "contradicted",
                                supported=bool(consistent),
                                contradicted=not bool(consistent),
                                unsupported=False,
                                citation_valid=citation_valid,
                                decision_consistent=bool(consistent),
                                matched_evidence_key="evidence.recommendation.ranking_mode",
                                expected_value=expected,
                                observed_value=observed,
                                reason=(
                                    None
                                    if consistent
                                    else "Ranking-mode claim contradicts structured evidence."
                                ),
                            )
                        )
                    continue

            if claim.claim_type == "framework_rank":
                expected_rank = _ranking_expected_rank(
                    evidence,
                    claim.entity or "",
                )
                if expected_rank is None:
                    verified.append(
                        ClaimVerification(
                            claim=claim,
                            status="unsupported",
                            supported=False,
                            contradicted=False,
                            unsupported=True,
                            citation_valid=citation_valid,
                            decision_consistent=None,
                            reason="No structured ranking matched this framework.",
                        )
                    )
                    continue
                consistent = int(claim.rank_value or -1) == int(expected_rank)
                verified.append(
                    ClaimVerification(
                        claim=claim,
                        status="supported" if consistent else "contradicted",
                        supported=bool(consistent),
                        contradicted=not bool(consistent),
                        unsupported=False,
                        citation_valid=citation_valid,
                        decision_consistent=bool(consistent),
                        expected_value=expected_rank,
                        observed_value=claim.rank_value,
                        reason=(
                            None
                            if consistent
                            else "Framework rank contradicts structured ranking."
                        ),
                    )
                )
                continue

            if claim.claim_type == "feature_rank":
                expected = top_feature(evidence)
                if expected is None:
                    verified.append(
                        ClaimVerification(
                            claim=claim,
                            status="unsupported",
                            supported=False,
                            contradicted=False,
                            unsupported=True,
                            citation_valid=citation_valid,
                            decision_consistent=None,
                            reason="No feature-ranking evidence is available.",
                        )
                    )
                    continue
                consistent = (
                    claim.rank_value == 1
                    and str(claim.feature).lower() == str(expected).lower()
                )
                verified.append(
                    ClaimVerification(
                        claim=claim,
                        status="supported" if consistent else "contradicted",
                        supported=bool(consistent),
                        contradicted=not bool(consistent),
                        unsupported=False,
                        citation_valid=citation_valid,
                        decision_consistent=bool(consistent),
                        expected_value=expected,
                        observed_value=claim.feature,
                        reason=(
                            None
                            if consistent
                            else "Feature-ranking claim contradicts structured attribution evidence."
                        ),
                    )
                )
                continue

            if claim.claim_type == "comparative":
                truth = _comparative_truth(claim, flat)
                if truth is None:
                    verified.append(
                        ClaimVerification(
                            claim=claim,
                            status="unsupported",
                            supported=False,
                            contradicted=False,
                            unsupported=True,
                            citation_valid=citation_valid,
                            reason="Comparative claim could not be resolved from evidence.",
                        )
                    )
                else:
                    verified.append(
                        ClaimVerification(
                            claim=claim,
                            status="supported" if truth else "contradicted",
                            supported=bool(truth),
                            contradicted=not bool(truth),
                            unsupported=False,
                            citation_valid=citation_valid,
                            decision_consistent=bool(truth),
                            reason=(
                                None
                                if truth
                                else "Comparative claim reverses structured evidence."
                            ),
                        )
                    )
                continue

            # Explicitly retained unparsed factual claims become unsupported.
            verified.append(
                ClaimVerification(
                    claim=claim,
                    status="unsupported",
                    supported=False,
                    contradicted=False,
                    unsupported=True,
                    citation_valid=citation_valid,
                    reason=(
                        "Factual-looking statement falls outside the supported "
                        "Phase-15 claim grammar or lacks resolvable evidence."
                    ),
                )
            )

        total = len(verified)
        supported = sum(item.supported for item in verified)
        contradicted = sum(item.contradicted for item in verified)
        unsupported = sum(item.unsupported for item in verified)
        verifiable = total - unsupported

        numeric_rows = [
            item for item in verified
            if item.claim.claim_type == "numeric"
            and item.numeric_correct is not None
        ]
        decision_rows = [
            item for item in verified
            if item.decision_consistent is not None
        ]

        valid_citations = sum(
            resolve_flat_key(flat, key) is not None
            for key in citations
        )

        metrics: Dict[str, Optional[float]] = {
            "claim_precision": _ratio(supported, total),
            "supported_claim_rate": _ratio(supported, verifiable),
            "numeric_correctness": _ratio(
                sum(bool(item.numeric_correct) for item in numeric_rows),
                len(numeric_rows),
            ),
            "citation_validity": _ratio(valid_citations, len(citations)),
            "citation_coverage": _ratio(
                sum(bool(item.claim.cited_keys) for item in verified),
                total,
            ),
            "decision_consistency": _ratio(
                sum(bool(item.decision_consistent) for item in decision_rows),
                len(decision_rows),
            ),
            "unsupported_claim_rate": _ratio(unsupported, total),
            "contradiction_rate": _ratio(contradicted, total),
            "claims_total": float(total),
            "claims_supported": float(supported),
            "claims_unsupported": float(unsupported),
            "claims_contradicted": float(contradicted),
        }

        return CorrectnessReport(
            case_id=case.case_id,
            dataset_id=case.dataset_id,
            source_stage=case.source_stage,
            source_name=case.source_name,
            explanation=str(explanation),
            claims=verified,
            metrics=metrics,
            evidence_key_count=len(flat),
            explanation_sha256=hashlib.sha256(
                str(explanation).encode("utf-8")
            ).hexdigest(),
        )
