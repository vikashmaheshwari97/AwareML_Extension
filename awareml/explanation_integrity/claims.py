from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .evidence import (
    canonical_metric,
    flatten_evidence,
    key_entity,
    key_metric,
    normalize_key,
    resolve_flat_key,
)
from .schemas import Claim


CITATION_RE = re.compile(
    r"\[((?:evidence|frameworks|ranking)[A-Za-z0-9_.:/-]*)\]",
    flags=re.IGNORECASE,
)

NUMBER_RE = r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_])"
# Do not parse digits embedded in tokens such as "CO2" as numeric claims.



METRIC_TEXT = {
    "accuracy": r"(?:accuracy|acc)",
    "runtime": r"(?:runtime|latency|time)",
    "energy": r"(?:energy(?:\s+use)?|energy_kwh)",
    "co2": r"(?:co2|co₂|carbon(?:\s+emissions?)?)",
    "dp": r"(?:dp(?:_diff|\s+difference)?|spd|demographic\s+parity(?:\s+difference)?|statistical\s+parity(?:\s+difference)?)",
    "eo": r"(?:eo|equal[_\s]+opportunity(?:_diff|[_\s]+difference)?)",
    "eodds": r"(?:eodds|eod|equalized[_\s]+odds(?:_gap|[_\s]+gap)?|equalised[_\s]+odds(?:_gap|[_\s]+gap)?)",
    "brier_gap": r"(?:group[_\s]+brier[_\s]+score[_\s]+gap|(?:group\s+)?brier(?:[-\s]+score)?(?:\s+gap)?)",
    "ece_gap": r"(?:group[_\s]+ece[_\s]+gap|(?:group\s+)?ece(?:\s+gap)?|calibration(?:\s+gap)?)",
    "utility": r"(?:utility|preference\s+utility)",
}


def _sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split prose and markdown/list output without breaking decimals/evidence keys.

    LLM fairness explanations commonly emit one metric row per line. Treat
    newlines as hard boundaries first, then split sentence-ending punctuation
    only when followed by whitespace/end. This preserves values such as 0.155
    and keys such as evidence.frameworks.AutoClass.fairness.dp_diff.
    """
    spans: List[Tuple[str, int, int]] = []
    offset = 0

    for raw_line in str(text).splitlines(True):
        line = raw_line.rstrip("\r\n")
        line_start = offset
        offset += len(raw_line)

        # Strip markdown/list prefixes while preserving source offsets.
        prefix_match = re.match(r"^\s*(?:[-*+]|\d+[.)])?\s*", line)
        prefix_len = prefix_match.end() if prefix_match else 0
        content = line[prefix_len:].strip()
        if not content:
            continue

        content_pos = line.find(content, prefix_len)
        absolute_start = line_start + max(0, content_pos)

        pattern = re.compile(
            r".+?(?:[.!?](?=\s|$)|$)",
            flags=re.DOTALL,
        )
        for match in pattern.finditer(content):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            spans.append(
                (
                    sentence,
                    absolute_start + match.start(),
                    absolute_start + match.end(),
                )
            )

    # Single-line text without a terminal newline is covered above. Empty input
    # simply yields no spans.
    return spans


def _citation_keys(sentence: str) -> List[str]:
    return sorted(
        {
            normalize_key(match.group(1))
            for match in CITATION_RE.finditer(sentence)
        }
    )


def _entities_from_evidence(evidence: Mapping[str, Any]) -> List[str]:
    flat = flatten_evidence(evidence)
    entities = set()
    for key in flat:
        entity = key_entity(key)
        if entity:
            entities.add(str(entity))

    ranking = evidence.get("ranking") if isinstance(evidence, Mapping) else None
    if isinstance(ranking, list):
        for row in ranking:
            if isinstance(row, Mapping) and row.get("framework") is not None:
                entities.add(str(row["framework"]))

    frameworks = evidence.get("frameworks") if isinstance(evidence, Mapping) else None
    if isinstance(frameworks, Mapping):
        entities.update(str(name) for name in frameworks)

    candidates = evidence.get("candidates") if isinstance(evidence, Mapping) else None
    if isinstance(candidates, Mapping):
        entities.update(str(name) for name in candidates)

    return sorted(entities, key=len, reverse=True)


def _features_from_evidence(evidence: Mapping[str, Any]) -> List[str]:
    features = set()

    def inspect(node):
        if isinstance(node, Mapping):
            shap = node.get("shap_values")
            if isinstance(shap, Mapping):
                features.update(str(name) for name in shap)

            importance = node.get("feature_importance")
            if isinstance(importance, Mapping):
                features.update(str(name) for name in importance)
            elif isinstance(importance, list):
                for item in importance:
                    if isinstance(item, Mapping):
                        name = item.get("feature") or item.get("name")
                        if name:
                            features.add(str(name))

            top = node.get("top_features")
            if isinstance(top, list):
                for item in top:
                    if isinstance(item, Mapping):
                        name = item.get("feature") or item.get("name")
                        if name:
                            features.add(str(name))
                    elif isinstance(item, (list, tuple)) and item:
                        features.add(str(item[0]))
                    elif isinstance(item, str):
                        features.add(item)
            for child in node.values():
                inspect(child)
        elif isinstance(node, list):
            for child in node:
                inspect(child)

    inspect(evidence)
    return sorted(features, key=len, reverse=True)


def _find_entity(sentence: str, entities: Sequence[str]) -> Optional[str]:
    lowered = sentence.lower()
    for entity in entities:
        if entity.lower() in lowered:
            return entity
    return None


def _find_feature(sentence: str, features: Sequence[str]) -> Optional[str]:
    lowered = sentence.lower()
    for feature in features:
        if feature.lower() in lowered:
            return feature
    return None


def _claim_id(index: int) -> str:
    return "C{:03d}".format(index)


def _feature_from_evidence_key(key: str) -> Optional[str]:
    parts = normalize_key(key).split(".")
    if "shap_values" in parts:
        idx = parts.index("shap_values")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _citation_numeric_claims(
    sentence: str,
    citations: Sequence[str],
    evidence: Mapping[str, Any],
    start: int,
    end: int,
) -> List[Claim]:
    """Extract numeric claims directly from exact structured-evidence citations.

    This is deliberately citation-first. When an LLM writes:
        DP difference: 0.035 [evidence.frameworks.AutoClass.fairness.dp_diff]
    the citation already disambiguates metric and framework even though the
    framework name appears only inside the citation.
    """
    flat = flatten_evidence(evidence)
    claims: List[Claim] = []

    # Remove bracketed evidence paths before scanning numbers so indices such
    # as ranking.0.rank do not get mistaken for reported numeric values.
    visible = CITATION_RE.sub(" ", sentence)
    number_matches = list(re.finditer(NUMBER_RE, visible))
    visible_numbers = []
    for match in number_matches:
        try:
            visible_numbers.append(float(match.group(0)))
        except Exception:
            continue

    for citation in citations:
        resolved = resolve_flat_key(flat, citation)
        if resolved is None:
            continue

        metric = key_metric(resolved)
        if metric not in {
            "accuracy",
            "runtime",
            "energy",
            "co2",
            "dp",
            "eo",
            "eodds",
            "brier_gap",
            "ece_gap",
            "shap",
            "utility",
            "rank",
        }:
            continue

        expected = flat.get(resolved)
        try:
            expected_numeric = float(expected)
        except Exception:
            continue

        if not visible_numbers:
            continue

        # Prefer a number matching the cited structured value. This supports
        # both "value 0.31 [key]" and "[key], 0.31" without using an LLM judge.
        observed = min(
            visible_numbers,
            key=lambda value: abs(float(value) - expected_numeric),
        )

        entity = key_entity(resolved)
        feature = _feature_from_evidence_key(resolved)

        claims.append(
            Claim(
                claim_id="",
                text=sentence,
                claim_type="numeric",
                metric=metric,
                entity=entity,
                feature=feature,
                numeric_value=float(observed),
                cited_keys=list(citations),
                span_start=start,
                span_end=end,
            )
        )

    return claims



class ClaimExtractor:
    """Deterministic claim extractor for AwareML explanation domains.

    It deliberately focuses on the factual claim families required by Phase 15:
    objective metrics, fairness/calibration, SHAP values/ranking, and framework
    ranking. It does not use an LLM judge, which keeps correctness measurement
    independent of the explanation model under evaluation.
    """

    def extract(
        self,
        text: str,
        evidence: Mapping[str, Any],
    ) -> List[Claim]:
        text = str(text or "")
        entities = _entities_from_evidence(evidence)
        features = _features_from_evidence(evidence)
        claims: List[Claim] = []

        for sentence, start, end in _sentences(text):
            citations = _citation_keys(sentence)
            entity = _find_entity(sentence, entities)
            feature = _find_feature(sentence, features)
            sentence_claims = []

            # Exact evidence citations are the strongest available grounding
            # signal. Parse them before free-form surface grammar.
            sentence_claims.extend(
                _citation_numeric_claims(
                    sentence,
                    citations,
                    evidence,
                    start,
                    end,
                )
            )

            # Framework ranking / recommendation claims.
            # IMPORTANT: "is <number>" is NOT a rank claim. The word
            # "ranked" (or "ranks") is required here so metric sentences such
            # as "AutoClass is 0.72" cannot become a fake framework rank.
            if entity:
                ranked = re.search(
                    re.escape(entity)
                    + r"\s+(?:(?:is|was|remains)\s+ranked|ranks?)\s*#?\s*(\d+)\b",
                    sentence,
                    flags=re.IGNORECASE,
                )
                if ranked:
                    sentence_claims.append(
                        Claim(
                            claim_id="",
                            text=sentence,
                            claim_type="framework_rank",
                            entity=entity,
                            metric="rank",
                            rank_value=int(ranked.group(1)),
                            cited_keys=citations,
                            span_start=start,
                            span_end=end,
                        )
                    )
                elif re.search(
                    re.escape(entity)
                    + r".{0,40}(?:top recommendation|ranked first|ranked #?1|best-ranked|highest-ranked|rank #?1)",
                    sentence,
                    flags=re.IGNORECASE,
                ):
                    sentence_claims.append(
                        Claim(
                            claim_id="",
                            text=sentence,
                            claim_type="framework_rank",
                            entity=entity,
                            metric="rank",
                            rank_value=1,
                            cited_keys=citations,
                            span_start=start,
                            span_end=end,
                        )
                    )

            # Natural recommendation / rank phrasing used by LLMs.
            if entity:
                if re.search(
                    r"(?:recommended(?:\s+framework)?\s+(?:is|being)\s+"
                    + re.escape(entity)
                    + r"|top[-\s]*ranked\s+framework\s+(?:is|being)\s+"
                    + re.escape(entity)
                    + r"|"
                    + re.escape(entity)
                    + r".{0,40}(?:is|was)\s+(?:the\s+)?(?:top[-\s]*ranked|recommended)\s+framework)",
                    sentence,
                    flags=re.IGNORECASE,
                ):
                    sentence_claims.append(
                        Claim(
                            claim_id="",
                            text=sentence,
                            claim_type="framework_rank",
                            entity=entity,
                            metric="rank",
                            rank_value=1,
                            cited_keys=citations,
                            span_start=start,
                            span_end=end,
                        )
                    )

            # Additional natural rank-one forms observed in real Llama output.
            if entity:
                rank_one_patterns = [
                    r"(?:the\s+)?framework\s+ranked\s+first\s+is\s+[\"']?"
                    + re.escape(entity),
                    r"(?:ranked\s+first|ranking\s+first)\s+is\s+[\"']?"
                    + re.escape(entity),
                    re.escape(entity)
                    + r".{0,55}(?:with\s+)?(?:a\s+)?(?:rank|ranking)\s+(?:of\s+)?1\b",
                    re.escape(entity)
                    + r".{0,80}(?:has|with|indicates|shows)?.{0,20}(?:the\s+)?highest\s+ranking\b",
                    r"specifically\s+[\"']?"
                    + re.escape(entity)
                    + r"[\"']?.{0,30}(?:rank|ranking)\s+(?:of\s+)?1\b",
                ]
                if any(
                    re.search(pattern, sentence, flags=re.IGNORECASE)
                    for pattern in rank_one_patterns
                ):
                    sentence_claims.append(
                        Claim(
                            claim_id="",
                            text=sentence,
                            claim_type="framework_rank",
                            entity=entity,
                            metric="rank",
                            rank_value=1,
                            cited_keys=citations,
                            span_start=start,
                            span_end=end,
                        )
                    )

            # Categorical ranking-mode claim.
            mode_match = re.search(
                r"ranking\s+mode.{0,20}?(?:is\s+set\s+to|is|being|=|:)\s*[\"']?([A-Za-z0-9_-]+)",
                sentence,
                flags=re.IGNORECASE,
            )
            if mode_match:
                sentence_claims.append(
                    Claim(
                        claim_id="",
                        text=sentence,
                        claim_type="categorical",
                        metric="ranking_mode",
                        comparator=mode_match.group(1),
                        cited_keys=citations,
                        span_start=start,
                        span_end=end,
                    )
                )

            # Metric numeric claims. Two common word orders are supported.
            for metric, metric_pattern in METRIC_TEXT.items():
                patterns = []
                if entity:
                    patterns.extend([
                        (
                            re.escape(entity)
                            + r".{0,320}?"
                            + metric_pattern
                            + r"(?:\s+(?:value|score|consumption))?"
                            + r"\s*(?:=|:|is|was|of)?\s*\(?\s*(" + NUMBER_RE + r")"
                        ),
                        (
                            metric_pattern
                            + r"(?:\s+(?:value|score|consumption))?"
                            + r".{0,160}?"
                            + re.escape(entity)
                            + r".{0,30}?(?:=|:|is|was|of)?\s*\(?\s*(" + NUMBER_RE + r")"
                        ),
                    ])
                else:
                    patterns.append(
                        metric_pattern
                        + r"\s*(?:gap|value|score|consumption)?\s*(?:=|:|is|was|of)?\s*\(?\s*(" + NUMBER_RE + r")"
                    )

                numeric_match = None
                for pattern in patterns:
                    numeric_match = re.search(
                        pattern,
                        sentence,
                        flags=re.IGNORECASE,
                    )
                    if numeric_match:
                        break

                if numeric_match:
                    try:
                        value = float(numeric_match.group(1))
                    except Exception:
                        continue
                    sentence_claims.append(
                        Claim(
                            claim_id="",
                            text=sentence,
                            claim_type="numeric",
                            metric=metric,
                            entity=entity,
                            numeric_value=value,
                            cited_keys=citations,
                            span_start=start,
                            span_end=end,
                        )
                    )

            # Generic feature-importance numeric claims.
            if (
                feature
                and not re.search(r"\bshap\b", sentence, flags=re.IGNORECASE)
                and re.search(
                    r"\b(?:feature\s+importance|importance|attribution)\b",
                    sentence,
                    flags=re.IGNORECASE,
                )
            ):
                importance_patterns = [
                    re.escape(feature)
                    + r".{0,45}?(?:importance|attribution)"
                    + r".{0,20}?(?:=|:|is|was)?\s*("
                    + NUMBER_RE
                    + r")",
                    r"(?:importance|attribution).{0,30}?"
                    + re.escape(feature)
                    + r".{0,20}?(?:=|:|is|was)?\s*("
                    + NUMBER_RE
                    + r")",
                ]
                importance_match = None
                for pattern in importance_patterns:
                    importance_match = re.search(
                        pattern,
                        sentence,
                        flags=re.IGNORECASE,
                    )
                    if importance_match:
                        break
                if importance_match:
                    try:
                        value = float(importance_match.group(1))
                    except Exception:
                        value = None
                    if value is not None:
                        sentence_claims.append(
                            Claim(
                                claim_id="",
                                text=sentence,
                                claim_type="numeric",
                                metric="feature_importance",
                                feature=feature,
                                numeric_value=value,
                                cited_keys=citations,
                                span_start=start,
                                span_end=end,
                            )
                        )

            # SHAP numeric claims.
            if feature and re.search(r"\bshap\b", sentence, flags=re.IGNORECASE):
                shap_patterns = [
                    re.escape(feature)
                    + r".{0,40}?\bshap\b.{0,20}?(?:=|:|is|was)?\s*(" + NUMBER_RE + r")",
                    r"\bshap\b.{0,30}?"
                    + re.escape(feature)
                    + r".{0,20}?(?:=|:|is|was)?\s*(" + NUMBER_RE + r")",
                ]
                shap_match = None
                for pattern in shap_patterns:
                    shap_match = re.search(pattern, sentence, flags=re.IGNORECASE)
                    if shap_match:
                        break
                if shap_match:
                    try:
                        value = float(shap_match.group(1))
                    except Exception:
                        value = None
                    if value is not None:
                        sentence_claims.append(
                            Claim(
                                claim_id="",
                                text=sentence,
                                claim_type="numeric",
                                metric="shap",
                                feature=feature,
                                numeric_value=value,
                                cited_keys=citations,
                                span_start=start,
                                span_end=end,
                            )
                        )

            # Top-feature / feature-ranking claims.
            if feature and re.search(
                r"(?:top|most important|most influential|highest|largest|ranked first|#1).{0,30}(?:feature|shap)|"
                + r"(?:most influential feature\s+(?:is|was)\s+[\"']?"
                + re.escape(feature)
                + r"[\"']?)|"
                + re.escape(feature)
                + r".{0,45}(?:top feature|most important|most influential|largest shap|highest shap)",
                sentence,
                flags=re.IGNORECASE,
            ):
                sentence_claims.append(
                    Claim(
                        claim_id="",
                        text=sentence,
                        claim_type="feature_rank",
                        metric=(
                            "shap"
                            if re.search(
                                r"\bshap\b",
                                sentence,
                                flags=re.IGNORECASE,
                            )
                            else "feature_importance"
                        ),
                        feature=feature,
                        rank_value=1,
                        cited_keys=citations,
                        span_start=start,
                        span_end=end,
                    )
                )

            # Calibration-availability claims.
            if re.search(
                r"\bcalibration\s+fairness\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                availability = None
                if re.search(
                    r"\b(?:unavailable|not available|not explicitly mentioned|missing)\b",
                    sentence,
                    flags=re.IGNORECASE,
                ):
                    availability = "unavailable"
                elif re.search(
                    r"\b(?:available|reported|provided)\b",
                    sentence,
                    flags=re.IGNORECASE,
                ):
                    availability = "available"

                if availability is not None:
                    sentence_claims.append(
                        Claim(
                            claim_id="",
                            text=sentence,
                            claim_type="availability",
                            metric="calibration",
                            comparator=availability,
                            cited_keys=citations,
                            span_start=start,
                            span_end=end,
                        )
                    )

            # Comparative claims: X higher/lower metric than Y.
            if len(entities) >= 2:
                mentioned = [
                    item for item in entities if item.lower() in sentence.lower()
                ]
                if len(mentioned) >= 2:
                    for metric, metric_pattern in METRIC_TEXT.items():
                        comp = re.search(
                            r"\b(higher|lower|greater|smaller|faster|slower|better|worse)\b.{0,30}"
                            + metric_pattern
                            + r"|"
                            + metric_pattern
                            + r".{0,30}\b(higher|lower|greater|smaller|faster|slower|better|worse)\b",
                            sentence,
                            flags=re.IGNORECASE,
                        )
                        if comp:
                            comparator = comp.group(1) or comp.group(2)
                            sentence_claims.append(
                                Claim(
                                    claim_id="",
                                    text=sentence,
                                    claim_type="comparative",
                                    metric=metric,
                                    entity=mentioned[0],
                                    feature=mentioned[1],  # second entity kept here for compact schema
                                    comparator=str(comparator).lower(),
                                    cited_keys=citations,
                                    span_start=start,
                                    span_end=end,
                                )
                            )
                            break

            # If the sentence makes an apparent factual claim but none of the
            # required families could be extracted, keep it as an unsupported
            # factual claim rather than silently dropping it.
            factual_signal = (
                bool(re.search(NUMBER_RE, sentence))
                or bool(citations)
                or bool(
                    re.search(
                        r"\b(?:rank|best|highest|lowest|better|worse|fairer|"
                        r"accuracy|runtime|energy|co2|co₂|parity|opportunity|"
                        r"odds|calibration|brier|ece|shap|fair|proves|guaranteed|causal|ranking|drift|necessarily)\b",
                        sentence,
                        flags=re.IGNORECASE,
                    )
                )
            )
            if factual_signal and not sentence_claims:
                sentence_claims.append(
                    Claim(
                        claim_id="",
                        text=sentence,
                        claim_type="unparsed_factual",
                        entity=entity,
                        feature=feature,
                        cited_keys=citations,
                        span_start=start,
                        span_end=end,
                    )
                )

            for claim in sentence_claims:
                # Avoid exact duplicates produced by overlapping regexes.
                signature = (
                    claim.claim_type,
                    claim.metric,
                    claim.entity,
                    claim.feature,
                    claim.numeric_value,
                    claim.rank_value,
                    claim.comparator,
                    claim.text,
                )
                existing = {
                    (
                        item.claim_type,
                        item.metric,
                        item.entity,
                        item.feature,
                        item.numeric_value,
                        item.rank_value,
                        item.comparator,
                        item.text,
                    )
                    for item in claims
                }
                if signature in existing:
                    continue
                claim.claim_id = _claim_id(len(claims) + 1)
                claims.append(claim)

        return claims


def extract_citations(text: str) -> List[str]:
    return [
        normalize_key(match.group(1))
        for match in CITATION_RE.finditer(str(text or ""))
    ]
