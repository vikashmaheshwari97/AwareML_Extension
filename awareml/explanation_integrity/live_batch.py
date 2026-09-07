from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from .evidence import flatten_evidence, top_ranked_framework
from .faithfulness import FaithfulnessV2Evaluator
from .live import build_live_cases
from .live_guided import (
    LiveGroundingError,
    LiveGuidedOllamaGenerator,
    LiveReferenceExplanationGenerator,
)
from .schemas import EvidenceCase, Intervention
from .verifier import GeneralEvidenceVerifier


def _as_float(value):
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _default_intervention(case: EvidenceCase):
    flat = flatten_evidence(case.evidence)

    if case.source_stage == "B":
        top = (case.evidence.get("recommendation") or {}).get("top_framework")
        candidates = [
            "evidence.candidates.{}.accuracy".format(top),
            "evidence.candidates.{}.runtime".format(top),
        ]
    elif case.source_stage == "E":
        focus = case.metadata.get("focus_framework")
        candidates = [
            "evidence.frameworks.{}.fairness.dp_diff".format(focus),
            "evidence.frameworks.{}.fairness.equal_opportunity_diff".format(focus),
            "evidence.frameworks.{}.fairness.equalized_odds_gap".format(focus),
        ]
    elif case.source_stage == "F_XAI":
        candidates = [
            key
            for key in flat
            if ".shap_values." in key
        ]
        if not candidates:
            candidates = [
                key
                for key in flat
                if ".feature_importance." in key
                and key.endswith(".importance")
            ]
        if not candidates:
            candidates = [
                key
                for key in (
                    "evidence.explainability.fidelity",
                    "evidence.explainability.stability",
                    "evidence.explainability.consistency",
                )
                if key in flat
            ]
    else:
        top = top_ranked_framework(case.evidence)
        candidates = [
            "evidence.frameworks.{}.accuracy".format(top),
            "evidence.frameworks.{}.runtime_sec".format(top),
        ]

    for key in candidates:
        if key not in flat:
            continue
        value = _as_float(flat[key])
        if value is None:
            continue
        counterfactual = value * 0.8 if abs(value) > 1e-12 else 0.1
        return Intervention(
            intervention_id="LIVE_BATCH",
            kind="live_numeric_counterfactual",
            evidence_key=key,
            original_value=value,
            counterfactual_value=counterfactual,
            relevant=True,
            metadata={
                "exploratory_live_probe": True,
                "claim_level_sensitivity": (
                    case.source_stage in {"B", "F_CHAT"}
                ),
            },
        )
    return None


def _stage_label(stage):
    return {
        "B": "Stage B · recommendation",
        "E": "Stage E · fairness",
        "F_XAI": "Stage F · XAI",
        "F_CHAT": "Stage F · conversational",
    }.get(stage, stage)


def run_live_batch(
    state: Mapping[str, Any],
    include_interventions=True,
    timeout_sec=300.0,
    network_retries=1,
):
    cases = build_live_cases(state)
    generator = LiveGuidedOllamaGenerator(
        timeout_sec=timeout_sec,
        network_retries=network_retries,
    )
    reference = LiveReferenceExplanationGenerator()
    verifier = GeneralEvidenceVerifier()

    rows = []

    for case in cases:
        row = {
            "case_id": case.case_id,
            "stage": case.source_stage,
            "stage_label": _stage_label(case.source_stage),
            "source_name": case.source_name,
            "dataset_id": case.dataset_id,
            "reference_explanation": None,
            "llm_generation_status": "not_run",
            "llm_explanation": None,
            "generation_meta": None,
            "correctness": None,
            "intervention": None,
            "faithfulness_status": "not_run",
            "faithfulness": None,
            "error": None,
        }

        try:
            ref_text, _ = reference.generate(case)
            row["reference_explanation"] = ref_text
        except Exception as exc:
            row["reference_explanation"] = (
                "Reference unavailable: {}: {}"
                .format(type(exc).__name__, exc)
            )

        try:
            text, meta = generator.generate(case)
            report = verifier.verify(case, text)
            row["llm_generation_status"] = "ok"
            row["llm_explanation"] = text
            row["generation_meta"] = dict(meta or {})
            row["correctness"] = report.to_dict()
        except LiveGroundingError as exc:
            row["llm_generation_status"] = "failed_grounding"
            row["error"] = str(exc)
            rows.append(row)
            continue
        except Exception as exc:
            row["llm_generation_status"] = "failed_runtime"
            row["error"] = "{}: {}".format(type(exc).__name__, exc)
            rows.append(row)
            continue

        if include_interventions:
            intervention = _default_intervention(case)
            if intervention is None:
                row["faithfulness_status"] = "no_numeric_intervention"
            else:
                row["intervention"] = intervention.to_dict()
                try:
                    evaluator = FaithfulnessV2Evaluator()
                    record = evaluator.evaluate(
                        case,
                        intervention,
                        generator,
                        original_explanation=text,
                    )
                    row["faithfulness_status"] = "ok"
                    row["faithfulness"] = record.to_dict()
                except LiveGroundingError as exc:
                    row["faithfulness_status"] = "failed_grounding"
                    row["error"] = (
                        (row.get("error") + " | " if row.get("error") else "")
                        + str(exc)
                    )
                except Exception as exc:
                    row["faithfulness_status"] = "failed_runtime"
                    row["error"] = (
                        (row.get("error") + " | " if row.get("error") else "")
                        + "{}: {}".format(type(exc).__name__, exc)
                    )

        rows.append(row)

    stage_count = len(rows)
    generation_ok = sum(
        row["llm_generation_status"] == "ok"
        for row in rows
    )
    faithfulness_ok = sum(
        row["faithfulness_status"] == "ok"
        for row in rows
    )

    return {
        "schema_version": "phase15_live_batch_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": str(state.get("dataset_name") or "active_dataset"),
        "scope": (
            "exploratory live-dataset batch; not frozen journal evidence and "
            "not the empirical LLM benchmark"
        ),
        "guided_live_prompt_version": generator.prompt_version,
        "empirical_protocol": (
            "The empirical 24-case benchmark remains separate and continues "
            "to use OllamaEvidenceExplanationGenerator / "
            "phase15_explanation_prompt_v4."
        ),
        "include_interventions": bool(include_interventions),
        "stage_count": stage_count,
        "generation_ok_count": generation_ok,
        "faithfulness_ok_count": faithfulness_ok,
        "complete_without_stage_failure": bool(
            stage_count > 0
            and generation_ok == stage_count
            and (
                not include_interventions
                or faithfulness_ok == stage_count
            )
        ),
        "stages": rows,
    }


def save_live_batch(result, root: Path):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset = str(result.get("dataset_name") or "dataset")
    safe = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in dataset
    )

    out_dir = (
        Path(root)
        / "artifacts"
        / "phase15"
        / "live_batch_runs"
        / "{}__{}".format(timestamp, safe)
    )
    suffix = 0
    candidate = out_dir
    while candidate.exists():
        suffix += 1
        candidate = Path(str(out_dir) + "__{}".format(suffix))
    candidate.mkdir(parents=True, exist_ok=False)

    path = candidate / "phase15_live_batch.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path
