from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from awareml.llm.client import OllamaClient

from .evidence import normalize_key, top_ranked_framework
from .schemas import EvidenceCase
from .verifier import GeneralEvidenceVerifier


LIVE_GUIDED_PROMPT_VERSION = "phase15_live_guided_prompt_v3"


class LiveGroundingError(RuntimeError):
    """Exploratory live generation failed deterministic grounding."""


def _fmt(value: Any) -> str:
    try:
        return "{:.10g}".format(float(value))
    except Exception:
        return str(value)


def _focus_framework(case: EvidenceCase) -> Optional[str]:
    focus = case.metadata.get("focus_framework")
    if focus:
        return str(focus)

    recommendation = case.evidence.get("recommendation") or {}
    if recommendation.get("top_framework"):
        return str(recommendation.get("top_framework"))

    if case.evidence.get("framework"):
        return str(case.evidence.get("framework"))

    frameworks = case.evidence.get("frameworks") or {}
    if frameworks:
        return str(next(iter(frameworks.keys())))

    return None


def _importance_rows(exp: Mapping[str, Any]):
    raw = exp.get("feature_importance")
    rows = []

    if isinstance(raw, Mapping):
        for name, value in raw.items():
            try:
                rows.append((str(name), float(value), str(name)))
            except Exception:
                continue

    elif isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            name = item.get("feature") or item.get("name")
            value = item.get("importance")
            if name is None or value is None:
                continue
            try:
                rows.append((str(name), float(value), index))
            except Exception:
                continue

    rows.sort(key=lambda row: abs(row[1]), reverse=True)
    return rows


def _intervention_info(case: EvidenceCase):
    raw = case.metadata.get("phase15_intervention")
    return raw if isinstance(raw, Mapping) else None


class LiveReferenceExplanationGenerator:
    """Deterministic reference for UI inspection only.

    This output is never counted as LLM performance.
    """

    source = "phase15-live-reference"
    model = None

    def generate(self, case: EvidenceCase):
        ev = case.evidence
        stage = case.source_stage

        if stage == "B":
            top = (ev.get("recommendation") or {}).get("top_framework")
            row = (ev.get("candidates") or {}).get(top) or {}
            parts = [
                "{} is the current top recommendation "
                "[evidence.recommendation.top_framework].".format(top)
            ]
            for metric in ("utility", "accuracy", "runtime", "energy", "co2"):
                if row.get(metric) is not None:
                    parts.append(
                        "{} predicted {} = {} "
                        "[evidence.candidates.{}.{}].".format(
                            top,
                            metric,
                            _fmt(row[metric]),
                            top,
                            metric,
                        )
                    )
            return " ".join(parts), {
                "source": self.source,
                "model": None,
            }

        if stage == "E":
            focus = _focus_framework(case)
            fair = (
                ((ev.get("frameworks") or {}).get(focus) or {})
                .get("fairness")
                or {}
            )
            parts = []
            for key, label in (
                ("dp_diff", "DP/SPD gap"),
                ("equal_opportunity_diff", "Equal Opportunity gap"),
                ("equalized_odds_gap", "Equalized Odds gap"),
                ("group_brier_score_gap", "Group Brier gap"),
                ("group_ece_gap", "Group ECE gap"),
            ):
                if fair.get(key) is not None:
                    parts.append(
                        "{} {} = {} "
                        "[evidence.frameworks.{}.fairness.{}].".format(
                            focus,
                            label,
                            _fmt(fair[key]),
                            focus,
                            key,
                        )
                    )
            return " ".join(parts), {
                "source": self.source,
                "model": None,
            }

        if stage == "F_XAI":
            framework = ev.get("framework")
            exp = ev.get("explainability") or {}
            shap = exp.get("shap_values")

            if isinstance(shap, Mapping) and shap:
                name, value = sorted(
                    (
                        (str(key), float(value))
                        for key, value in shap.items()
                    ),
                    key=lambda row: abs(row[1]),
                    reverse=True,
                )[0]
                return (
                    "{} is the top SHAP feature with value {} "
                    "[evidence.explainability.shap_values.{}]."
                    .format(name, _fmt(value), name)
                ), {
                    "source": self.source,
                    "model": None,
                }

            rows = _importance_rows(exp)
            if rows:
                name, value, locator = rows[0]
                if isinstance(locator, int):
                    key = (
                        "evidence.explainability.feature_importance."
                        "{}.importance".format(locator)
                    )
                else:
                    key = (
                        "evidence.explainability.feature_importance."
                        "{}".format(name)
                    )
                return (
                    "Literal SHAP values are unavailable. "
                    "{} has generic feature importance {} [{}]."
                    .format(name, _fmt(value), key)
                ), {
                    "source": self.source,
                    "model": None,
                }

            return (
                "{} XAI evidence is unavailable.".format(framework),
                {"source": self.source, "model": None},
            )

        top = top_ranked_framework(ev)
        row = (ev.get("frameworks") or {}).get(top) or {}
        parts = ["{} is ranked first.".format(top)]
        for label, key in (
            ("accuracy", "accuracy"),
            ("runtime", "runtime_sec"),
            ("energy", "energy_kwh"),
            ("CO2", "co2_kg"),
        ):
            if row.get(key) is not None:
                parts.append(
                    "{} {} = {} [evidence.frameworks.{}.{}].".format(
                        top,
                        label,
                        _fmt(row[key]),
                        top,
                        key,
                    )
                )
        return " ".join(parts), {
            "source": self.source,
            "model": None,
        }


class LiveGuidedOllamaGenerator:
    """Guided generator used only by the exploratory Live Dataset Probe.

    The empirical 24-case benchmark continues to use its frozen,
    separate prompt protocol.
    """

    source = "ollama-phase15-live-guided"
    prompt_version = LIVE_GUIDED_PROMPT_VERSION

    def __init__(self, timeout_sec=300.0, network_retries=1):
        self.client = OllamaClient(timeout_sec=float(timeout_sec))
        self.network_retries = max(0, int(network_retries))
        self.model = self.client.model
        self.verifier = GeneralEvidenceVerifier()

    def _facts(self, case):
        ev = case.evidence
        stage = case.source_stage
        facts = []

        if stage == "B":
            top = (ev.get("recommendation") or {}).get("top_framework")
            facts.append(
                (
                    "top framework",
                    top,
                    "evidence.recommendation.top_framework",
                )
            )
            row = (ev.get("candidates") or {}).get(top) or {}
            for metric in ("utility", "accuracy", "runtime", "energy", "co2"):
                if row.get(metric) is not None:
                    facts.append(
                        (
                            "predicted {}".format(metric),
                            row[metric],
                            "evidence.candidates.{}.{}".format(
                                top,
                                metric,
                            ),
                        )
                    )
            return facts

        if stage == "E":
            focus = _focus_framework(case)
            facts.append(("framework", focus, None))
            fair = (
                ((ev.get("frameworks") or {}).get(focus) or {})
                .get("fairness")
                or {}
            )
            for key, label in (
                ("dp_diff", "DP/SPD gap"),
                ("equal_opportunity_diff", "Equal Opportunity gap"),
                ("equalized_odds_gap", "Equalized Odds gap"),
                ("group_brier_score_gap", "Group Brier gap"),
                ("group_ece_gap", "Group ECE gap"),
            ):
                if fair.get(key) is not None:
                    facts.append(
                        (
                            label,
                            fair[key],
                            "evidence.frameworks.{}.fairness.{}".format(
                                focus,
                                key,
                            ),
                        )
                    )
            return facts

        if stage == "F_XAI":
            framework = ev.get("framework")
            exp = ev.get("explainability") or {}
            facts.extend([
                ("framework", framework, None),
                ("method", exp.get("method"), None),
            ])

            shap = exp.get("shap_values")
            if isinstance(shap, Mapping) and shap:
                for name, value in sorted(
                    shap.items(),
                    key=lambda item: abs(float(item[1])),
                    reverse=True,
                )[:3]:
                    facts.append(
                        (
                            "{} SHAP value".format(name),
                            value,
                            "evidence.explainability.shap_values.{}".format(
                                name
                            ),
                        )
                    )
            else:
                for name, value, locator in _importance_rows(exp)[:3]:
                    if isinstance(locator, int):
                        key = (
                            "evidence.explainability.feature_importance."
                            "{}.importance".format(locator)
                        )
                    else:
                        key = (
                            "evidence.explainability.feature_importance."
                            "{}".format(name)
                        )
                    facts.append(
                        (
                            "{} generic feature importance".format(name),
                            value,
                            key,
                        )
                    )
            return facts

        top = top_ranked_framework(ev)
        facts.append(("top framework", top, None))
        row = (ev.get("frameworks") or {}).get(top) or {}
        for label, key in (
            ("accuracy", "accuracy"),
            ("runtime", "runtime_sec"),
            ("energy", "energy_kwh"),
            ("CO2", "co2_kg"),
        ):
            if row.get(key) is not None:
                facts.append(
                    (
                        label,
                        row[key],
                        "evidence.frameworks.{}.{}".format(top, key),
                    )
                )
        return facts

    def _task(self, case):
        intervention = _intervention_info(case)
        if intervention:
            return (
                "This is the counterfactual response for a faithfulness "
                "intervention. You MUST report the changed evidence value "
                "using its exact key and current counterfactual value."
            )

        if case.source_stage == "B":
            return (
                "Explain the current pre-run recommendation using the listed "
                "PREDICTED point evidence. Do not describe these values as "
                "training results, observed outcomes, or achieved performance."
            )

        if case.source_stage == "E":
            return (
                "Explain the selected framework's fairness gaps using the "
                "listed actual values. Do not label the model simply fair "
                "or biased."
            )

        if case.source_stage == "F_XAI":
            return (
                "Explain the recorded feature-attribution evidence. If literal "
                "SHAP values are absent, call the listed values generic feature "
                "importance and do not relabel them as exact SHAP values."
            )

        return (
            "Answer which framework is ranked first and report the listed "
            "post-run supporting metrics."
        )

    def _prompt(self, case, repair=False, previous=None):
        lines = []
        for label, value, key in self._facts(case):
            if key:
                lines.append(
                    "- {} = {} | [{}]".format(
                        label,
                        _fmt(value),
                        key,
                    )
                )
            else:
                lines.append("- {} = {}".format(label, value))

        intervention = _intervention_info(case)
        intervention_block = ""
        if intervention:
            key = normalize_key(
                str(intervention.get("evidence_key") or "")
            )
            new_value = intervention.get("counterfactual_value")
            intervention_block = (
                "\nINTERVENTION TARGET:\n"
                "- changed evidence key = {key}\n"
                "- counterfactual value = {value}\n"
                "The response MUST explicitly state this changed value and "
                "cite [{key}].\n"
            ).format(
                key=key,
                value=_fmt(new_value),
            )

        repair_text = ""
        if repair:
            repair_text = (
                "\nThe previous answer was not sufficiently grounded:\n"
                "{}\n"
                "Rewrite it from FACTS only. Use one evidence-backed fact "
                "per line.\n"
            ).format(str(previous or "")[:1200])

        return """You are generating an EXPLORATORY AwareML live explanation.
This guided prompt is NOT the frozen empirical LLM benchmark.

TASK:
{task}
{intervention}
{repair}
FACTS:
{facts}

RULES:
- Use only FACTS.
- Write one evidence-backed factual statement per line.
- Do not add a generic introduction or conclusion.
- State concrete values, not generic metric definitions.
- Use each fact's OWN supplied [evidence....] key; never reuse another fact's key.
- Do not invent missing values or use placeholders.
- Do not ask the user for more information.
- Do not claim causality.
- For Stage B, say "predicted" rather than "achieved" or "during training".
- For Stage E, do not simply label the model fair/biased.
- For XAI, never call generic feature importance an exact SHAP value.
- Maximum 120 words.

ANSWER:
""".format(
            task=self._task(case),
            intervention=intervention_block,
            repair=repair_text,
            facts="\n".join(lines),
        )

    def _call(self, prompt):
        last_error = None
        for attempt in range(self.network_retries + 1):
            try:
                text, meta = self.client.generate_text(prompt)
                meta = dict(meta or {})
                self.model = meta.get("model") or self.model
                return text, meta
            except Exception as exc:
                last_error = exc
                if attempt >= self.network_retries:
                    break
                time.sleep(min(4.0, 1.5 * (2 ** attempt)))
        raise last_error

    def _grounded(self, case, text):
        """Check only whether the live response is usable enough to score.

        Important scientific separation:
        - This function is a *generation-validity* guard.
        - Correctness quality belongs to GeneralEvidenceVerifier metrics.
        - A response with a supported core claim plus an extra unsupported or
          contradicted claim is still a valid LLM output and must be scored,
          not discarded as "failed_grounding".
        """
        lowered = str(text or "").lower()
        if not lowered.strip():
            return False, "empty response"

        if any(marker in lowered for marker in (
            "[insert value]",
            "please provide",
            "need more context",
            "i'm ready",
            "i am ready",
        )):
            return False, "meta/placeholder response"

        report = self.verifier.verify(case, text)

        supported = [
            row
            for row in report.claims
            if row.supported
        ]
        supported_numeric = [
            row
            for row in supported
            if row.claim.claim_type == "numeric"
        ]
        supported_metrics = {
            row.claim.metric
            for row in supported
            if row.claim.metric
        }

        # Counterfactual validity is intentionally narrow: the changed evidence
        # itself must be acknowledged correctly. Other imperfections are left
        # in the faithfulness/correctness metrics rather than converted into a
        # generation failure.
        intervention = _intervention_info(case)
        if intervention:
            changed_key = normalize_key(
                str(intervention.get("evidence_key") or "")
            )
            changed_rows = []

            for row in report.claims:
                matched = normalize_key(
                    str(row.matched_evidence_key or "")
                )
                cited = {
                    normalize_key(str(key))
                    for key in row.claim.cited_keys
                }
                if matched == changed_key or changed_key in cited:
                    changed_rows.append(row)

            if not changed_rows:
                return False, (
                    "counterfactual response did not mention the changed "
                    "evidence key"
                )

            if not any(
                row.supported
                and (
                    row.numeric_correct is None
                    or bool(row.numeric_correct)
                )
                for row in changed_rows
            ):
                return False, (
                    "counterfactual response did not correctly update the "
                    "changed evidence value"
                )

            return True, None

        # Original live explanation validity: require a minimal source-relevant
        # grounded signal. Do NOT demand perfect correctness here; imperfect
        # claims are exactly what the correctness metrics are supposed to show.
        if case.source_stage == "B":
            valid = bool(
                supported
                and (
                    supported_numeric
                    or "rank" in supported_metrics
                    or "utility" in supported_metrics
                )
            )

        elif case.source_stage == "E":
            valid = bool(
                supported_numeric
                and (
                    supported_metrics
                    & {
                        "dp",
                        "eo",
                        "eodds",
                        "brier_gap",
                        "ece_gap",
                    }
                )
            )

        elif case.source_stage == "F_XAI":
            valid = bool(
                supported
                and (
                    supported_metrics
                    & {
                        "shap",
                        "feature_importance",
                    }
                    or any(
                        row.claim.claim_type == "feature_rank"
                        and row.supported
                        for row in report.claims
                    )
                )
            )

        else:
            valid = bool(
                supported
                and (
                    supported_numeric
                    or "rank" in supported_metrics
                )
            )

        if not valid:
            return False, "no supported source-relevant claim was extracted"

        return True, None

    def generate(self, case):
        text, meta = self._call(self._prompt(case))
        ok, reason = self._grounded(case, text)
        repaired = False

        if not ok:
            repaired = True
            text, second_meta = self._call(
                self._prompt(
                    case,
                    repair=True,
                    previous=text,
                )
            )
            meta.update(dict(second_meta or {}))
            ok, reason = self._grounded(case, text)

        meta["source"] = self.source
        meta["model"] = meta.get("model") or self.model
        meta["prompt_version"] = self.prompt_version
        meta["guided_live_mode"] = True
        meta["corrective_reprompt"] = bool(repaired)

        if not ok:
            raise LiveGroundingError(
                "Guided live generation remained ungrounded after one repair "
                "attempt: {}. No LLM score is fabricated.".format(reason)
            )

        return text, meta
