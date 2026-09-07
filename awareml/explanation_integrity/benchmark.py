from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .controlled import (
    build_controlled_cases,
    correct_explanation,
    incorrect_explanation,
    intervention_from_case,
)
from .faithfulness import (
    ControlledExplanationGenerator,
    FaithfulnessV2Evaluator,
    OllamaEvidenceExplanationGenerator,
    StickyExplanationGenerator,
)
from .schemas import EvidenceCase, FaithfulnessRecord, Intervention
from .stimuli import (
    build_trust_stimulus_bank,
    participant_bank,
    stimulus_counts,
)
from .verifier import GeneralEvidenceVerifier


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [
        float(value)
        for value in values
        if value is not None
    ]
    if not clean:
        return None
    return float(mean(clean))


def _metric_summary(reports: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    metric_names = [
        "claim_precision",
        "supported_claim_rate",
        "numeric_correctness",
        "citation_validity",
        "citation_coverage",
        "decision_consistency",
        "unsupported_claim_rate",
        "contradiction_rate",
    ]
    return {
        name: _safe_mean(
            (report.get("metrics") or {}).get(name)
            for report in reports
        )
        for name in metric_names
    }


def correctness_controlled_benchmark() -> Dict[str, Any]:
    verifier = GeneralEvidenceVerifier()
    cases = build_controlled_cases()
    reports: List[Dict[str, Any]] = []

    for index, case in enumerate(cases):
        correct = correct_explanation(case)
        incorrect, error_type = incorrect_explanation(case, index // 4)

        for label, explanation, error in [
            ("known_correct", correct, None),
            ("known_incorrect", incorrect, error_type),
        ]:
            report = verifier.verify(case, explanation).to_dict()
            report["researcher_label"] = label
            report["error_type"] = error
            reports.append(report)

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for report in reports:
        by_label[str(report["researcher_label"])].append(report)
        by_source[str(report["source_name"])].append(report)

    return {
        "schema_version": "explanation_correctness_v1",
        "created_utc": _now(),
        "scope": (
            "controlled verifier-validation benchmark; "
            "not an actual LLM performance estimate"
        ),
        "case_count": len(cases),
        "explanation_count": len(reports),
        "reports": reports,
        "summary": {
            "overall": _metric_summary(reports),
            "by_label": {
                label: _metric_summary(rows)
                for label, rows in sorted(by_label.items())
            },
            "by_source": {
                source: _metric_summary(rows)
                for source, rows in sorted(by_source.items())
            },
        },
    }


def _irrelevant_intervention(case: EvidenceCase) -> Intervention:
    return Intervention(
        intervention_id="CTRL_" + case.case_id,
        kind="irrelevant_control",
        evidence_key="evidence.control_metadata.display_token",
        original_value="CONTROL_A",
        counterfactual_value="CONTROL_B",
        relevant=False,
        expected_decision_change=False,
        metadata={},
    )


def faithfulness_controlled_benchmark() -> Dict[str, Any]:
    evaluator = FaithfulnessV2Evaluator()
    faithful = ControlledExplanationGenerator()
    cases = build_controlled_cases()
    records: List[Dict[str, Any]] = []

    for case in cases:
        relevant = intervention_from_case(case)
        original_text, _ = faithful.generate(case)

        faithful_record = evaluator.evaluate(
            case,
            relevant,
            faithful,
            original_explanation=original_text,
        ).to_dict()
        faithful_record["control_condition"] = "known_faithful"
        records.append(faithful_record)

        sticky = StickyExplanationGenerator(original_text)
        sticky_record = evaluator.evaluate(
            case,
            relevant,
            sticky,
            original_explanation=original_text,
        ).to_dict()
        sticky_record["control_condition"] = "known_unfaithful_sticky"
        records.append(sticky_record)

        irrelevant = _irrelevant_intervention(case)
        invariant_record = evaluator.evaluate(
            case,
            irrelevant,
            faithful,
            original_explanation=original_text,
        ).to_dict()
        invariant_record["control_condition"] = "irrelevant_control"
        records.append(invariant_record)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["control_condition"])].append(row)

    summary = {}
    for condition, rows in sorted(grouped.items()):
        summary[condition] = {
            "n": len(rows),
            "mean_faithfulness_score": _safe_mean(
                row.get("faithfulness_score") for row in rows
            ),
            "mean_changed_evidence_acknowledged": _safe_mean(
                row.get("changed_evidence_acknowledged") for row in rows
            ),
            "mean_numeric_update_accuracy": _safe_mean(
                row.get("numeric_update_accuracy") for row in rows
            ),
            "mean_stale_claim_rate": _safe_mean(
                row.get("stale_claim_rate") for row in rows
            ),
            "mean_irrelevant_invariance": _safe_mean(
                row.get("irrelevant_invariance") for row in rows
            ),
        }

    return {
        "schema_version": "faithfulness_v2",
        "created_utc": _now(),
        "scope": (
            "controlled external evidence-intervention evaluator validation; "
            "not an actual LLM faithfulness estimate"
        ),
        "case_count": len(cases),
        "record_count": len(records),
        "records": records,
        "summary": summary,
        "method_note": (
            "FaithLM-style external counterfactual evidence intervention: "
            "change structured evidence, regenerate the explanation, and test "
            "whether claims update. A separate irrelevant control measures invariance."
        ),
    }


def trust_stimulus_benchmark() -> Dict[str, Any]:
    records = build_trust_stimulus_bank()
    return {
        "schema_version": "trust_stimulus_bank_v1",
        "created_utc": _now(),
        "scope": (
            "controlled Track-2 trust-calibration stimuli with researcher "
            "ground truth separated from participant-facing versions"
        ),
        "records": [record.to_dict() for record in records],
        "participant_records": participant_bank(records),
        "summary": stimulus_counts(records),
    }


def run_controlled_phase15() -> Dict[str, Any]:
    return {
        "correctness": correctness_controlled_benchmark(),
        "faithfulness": faithfulness_controlled_benchmark(),
        "stimuli": trust_stimulus_benchmark(),
    }


class _RetryingGenerator:
    """Retry wrapper for slow/cold local Ollama generations."""

    def __init__(self, base_generator, retries: int = 1):
        self.base_generator = base_generator
        self.retries = max(0, int(retries))

    @property
    def source(self):
        return getattr(self.base_generator, "source", "unknown")

    @property
    def model(self):
        return getattr(self.base_generator, "model", None)

    @property
    def prompt_version(self):
        return getattr(
            self.base_generator,
            "prompt_version",
            None,
        )

    def generate(self, case):
        import time

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                return self.base_generator.generate(case)
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(5.0, 1.5 * (2 ** attempt)))
        raise last_error


def run_llm_phase15(
    cases: Optional[List[EvidenceCase]] = None,
    max_cases: Optional[int] = None,
    timeout_sec: float = 300.0,
    retries: int = 1,
    fail_fast: bool = False,
    generator=None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run model-facing Phase-15 evaluation robustly.

    Local Ollama generation can exceed 60 seconds on the first request while
    the model is loading into RAM/VRAM. Phase 15 therefore uses a longer
    benchmark-specific timeout, retries transient failures, and records failed
    cases instead of losing the entire run.

    These outputs remain empirical model evidence and are NOT auto-frozen.
    """
    cases = list(cases or build_controlled_cases())
    if max_cases is not None:
        cases = cases[: max(0, int(max_cases))]

    model_status: Dict[str, Any]

    if generator is None:
        from awareml.llm.client import OllamaClient

        client = OllamaClient(timeout_sec=float(timeout_sec))
        model_status = client.status()

        if not model_status.get("reachable"):
            raise RuntimeError(
                "Ollama is not reachable at {}. Start Ollama and verify the "
                "configured model before running Phase 15. Details: {}"
                .format(
                    getattr(client, "base_url", "configured Ollama URL"),
                    model_status.get("error"),
                )
            )

        base_generator = OllamaEvidenceExplanationGenerator(client=client)
    else:
        base_generator = generator
        model_status = {
            "reachable": None,
            "source": "injected-generator",
            "resolved_model": getattr(generator, "model", None),
        }

    retrying_generator = _RetryingGenerator(
        base_generator,
        retries=retries,
    )

    verifier = GeneralEvidenceVerifier()
    evaluator = FaithfulnessV2Evaluator(verifier=verifier)

    correctness = []
    faithfulness = []
    failures = []

    for index, case in enumerate(cases, start=1):
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "event": "case_start",
                        "index": index,
                        "total": len(cases),
                        "case_id": case.case_id,
                        "source_stage": case.source_stage,
                        "source_name": case.source_name,
                    }
                )
            except Exception:
                pass

        try:
            text, meta = retrying_generator.generate(case)
        except Exception as exc:
            failures.append({
                "case_id": case.case_id,
                "dataset_id": case.dataset_id,
                "source_stage": case.source_stage,
                "source_name": case.source_name,
                "step": "original_explanation_generation",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "case_index": index,
            })
            if fail_fast:
                raise
            continue

        report = verifier.verify(case, text).to_dict()
        report["generation_meta"] = meta
        correctness.append(report)

        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "event": "original_complete",
                        "index": index,
                        "total": len(cases),
                        "case_id": case.case_id,
                    }
                )
            except Exception:
                pass

        try:
            intervention = intervention_from_case(case)
            faith = evaluator.evaluate(
                case,
                intervention,
                retrying_generator,
                original_explanation=text,
            ).to_dict()
            faithfulness.append(faith)

            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "event": "case_complete",
                            "index": index,
                            "total": len(cases),
                            "case_id": case.case_id,
                            "faithfulness_score": faith.get("faithfulness_score"),
                        }
                    )
                except Exception:
                    pass
        except Exception as exc:
            failures.append({
                "case_id": case.case_id,
                "dataset_id": case.dataset_id,
                "source_stage": case.source_stage,
                "source_name": case.source_name,
                "step": "counterfactual_explanation_generation",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "case_index": index,
            })
            if fail_fast:
                raise

    requested = len(cases)
    completed_correctness = len(correctness)
    completed_faithfulness = len(faithfulness)

    complete = (
        completed_correctness == requested
        and completed_faithfulness == requested
        and not failures
    )

    return {
        "schema_version": "phase15_llm_run_v2",
        "created_utc": _now(),
        "scope": "empirical LLM explanation correctness + faithfulness run",
        "model": getattr(retrying_generator, "model", None),
        "explanation_prompt_version": getattr(
            retrying_generator,
            "prompt_version",
            None,
        ),
        "model_status": model_status,
        "runtime_config": {
            "timeout_sec": float(timeout_sec),
            "retries": int(retries),
            "fail_fast": bool(fail_fast),
        },
        "case_count": requested,
        "completed_correctness_count": completed_correctness,
        "completed_faithfulness_count": completed_faithfulness,
        "failure_count": len(failures),
        "complete": bool(complete),
        "correctness_reports": correctness,
        "faithfulness_records": faithfulness,
        "failures": failures,
        "summary": {
            "correctness": _metric_summary(correctness),
            "faithfulness": {
                "mean_faithfulness_score": _safe_mean(
                    row.get("faithfulness_score") for row in faithfulness
                ),
                "mean_changed_evidence_acknowledged": _safe_mean(
                    row.get("changed_evidence_acknowledged")
                    for row in faithfulness
                ),
                "mean_numeric_update_accuracy": _safe_mean(
                    row.get("numeric_update_accuracy")
                    for row in faithfulness
                ),
                "mean_stale_claim_rate": _safe_mean(
                    row.get("stale_claim_rate")
                    for row in faithfulness
                ),
            },
            "completion": {
                "requested_cases": requested,
                "completed_correctness_cases": completed_correctness,
                "completed_faithfulness_cases": completed_faithfulness,
                "failure_count": len(failures),
                "complete": bool(complete),
            },
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def freeze_controlled_artifact(
    artifact_root: Path,
    artifact_name: str,
    payload: Dict[str, Any],
    files: Dict[str, Any],
    overwrite: bool = False,
) -> Dict[str, Any]:
    frozen = artifact_root / artifact_name / "frozen"
    if frozen.exists() and any(frozen.iterdir()) and not overwrite:
        raise FileExistsError(
            "Frozen artifact already exists: {}. "
            "Refusing to overwrite without explicit overwrite=True."
            .format(frozen)
        )
    frozen.mkdir(parents=True, exist_ok=True)

    written = []
    for filename, content in files.items():
        path = frozen / filename
        _write_json(path, content)
        written.append(path)

    manifest = {
        "artifact": artifact_name,
        "status": "frozen",
        "created_utc": _now(),
        "scope": payload.get("scope"),
        "schema_version": payload.get("schema_version"),
        "controlled_benchmark": True,
        "actual_llm_performance_claim": False,
        "files": {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in written
        },
    }
    manifest_path = frozen / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "directory": str(frozen),
        "manifest": manifest,
    }


def freeze_all_controlled(
    root: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    results = run_controlled_phase15()

    correctness = results["correctness"]
    faithfulness = results["faithfulness"]
    stimuli = results["stimuli"]

    outputs = {}
    outputs["explanation_correctness_v1"] = freeze_controlled_artifact(
        root,
        "explanation_correctness_v1",
        correctness,
        {
            "correctness_reports.json": correctness["reports"],
            "summary.json": correctness["summary"],
            "benchmark_metadata.json": {
                key: value
                for key, value in correctness.items()
                if key not in {"reports", "summary"}
            },
        },
        overwrite=overwrite,
    )

    outputs["faithfulness_v2"] = freeze_controlled_artifact(
        root,
        "faithfulness_v2",
        faithfulness,
        {
            "faithfulness_records.json": faithfulness["records"],
            "summary.json": faithfulness["summary"],
            "benchmark_metadata.json": {
                key: value
                for key, value in faithfulness.items()
                if key not in {"records", "summary"}
            },
        },
        overwrite=overwrite,
    )

    outputs["trust_stimulus_bank_v1"] = freeze_controlled_artifact(
        root,
        "trust_stimulus_bank_v1",
        stimuli,
        {
            "stimuli_researcher.json": stimuli["records"],
            "stimuli_participant.json": stimuli["participant_records"],
            "summary.json": stimuli["summary"],
            "benchmark_metadata.json": {
                key: value
                for key, value in stimuli.items()
                if key not in {"records", "participant_records", "summary"}
            },
        },
        overwrite=overwrite,
    )

    return outputs
