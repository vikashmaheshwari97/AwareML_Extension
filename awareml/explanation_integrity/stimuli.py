from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional

from .controlled import (
    build_controlled_cases,
    correct_explanation,
    incorrect_explanation,
)
from .schemas import EvidenceCase, StimulusRecord
from .verifier import GeneralEvidenceVerifier


def _evidence_summary(case: EvidenceCase) -> Dict[str, Any]:
    """Compact researcher-facing evidence summary; raw rows are never included."""
    evidence = case.evidence
    if case.source_stage == "B":
        top = evidence.get("recommendation", {}).get("top_framework")
        row = (evidence.get("candidates") or {}).get(top, {})
        return {
            "top_framework": top,
            "top_accuracy": row.get("accuracy"),
            "top_runtime": row.get("runtime"),
            "top_energy": row.get("energy"),
            "top_co2": row.get("co2"),
        }

    if case.source_stage == "E":
        focus = case.metadata.get("focus_framework")
        return {
            "focus_framework": focus,
            "fairness": (
                (evidence.get("frameworks") or {})
                .get(focus, {})
                .get("fairness", {})
            ),
        }

    if case.source_stage == "F_XAI":
        exp = evidence.get("explainability") or {}
        return {
            "method": exp.get("method"),
            "top_feature": exp.get("top_feature"),
            "shap_values": exp.get("shap_values"),
        }

    ranking = evidence.get("ranking") or []
    frameworks = evidence.get("frameworks") or {}
    top = ranking[0].get("framework") if ranking else None
    return {
        "top_framework": top,
        "top_framework_metrics": frameworks.get(top, {}),
    }


def build_trust_stimulus_bank(
    verifier: Optional[GeneralEvidenceVerifier] = None,
) -> List[StimulusRecord]:
    verifier = verifier or GeneralEvidenceVerifier()
    records: List[StimulusRecord] = []
    created = datetime.now(timezone.utc).isoformat()

    for pair_index, case in enumerate(build_controlled_cases()):
        correct = correct_explanation(case)
        incorrect, error_type = incorrect_explanation(case, pair_index // 4)

        for label, explanation, error in [
            ("known_correct", correct, None),
            ("known_incorrect", incorrect, error_type),
        ]:
            suffix = "CORRECT" if label == "known_correct" else "INCORRECT"
            stimulus_id = "{}_{}".format(case.case_id, suffix)
            report = verifier.verify(case, explanation)
            participant_version = {
                "stimulus_id": stimulus_id,
                "source": case.source_name,
                "scenario": case.dataset_id,
                "prompt": case.prompt,
                "explanation": explanation,
                # Intentionally no correctness label or verifier score.
            }
            records.append(
                StimulusRecord(
                    stimulus_id=stimulus_id,
                    pair_id=case.case_id,
                    dataset_id=case.dataset_id,
                    source_stage=case.source_stage,
                    source_name=case.source_name,
                    prompt=case.prompt,
                    explanation=explanation,
                    evidence_summary=_evidence_summary(case),
                    researcher_label=label,
                    error_type=error,
                    verifier_metrics=report.metrics,
                    participant_version=participant_version,
                    provenance={
                        "created_utc": created,
                        "generator": "phase15_controlled_stimulus_builder_v1",
                        "controlled_fixture": True,
                        "explanation_sha256": hashlib.sha256(
                            explanation.encode("utf-8")
                        ).hexdigest(),
                    },
                )
            )

    return records


def participant_bank(records: List[StimulusRecord]) -> List[Dict[str, Any]]:
    return [dict(record.participant_version) for record in records]


def stimulus_counts(records: List[StimulusRecord]) -> Dict[str, Any]:
    labels: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    datasets = set()
    for record in records:
        labels[record.researcher_label] = (
            labels.get(record.researcher_label, 0) + 1
        )
        sources[record.source_name] = (
            sources.get(record.source_name, 0) + 1
        )
        datasets.add(record.dataset_id)
    return {
        "total": len(records),
        "labels": labels,
        "sources": sources,
        "dataset_contexts": len(datasets),
    }
