from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from awareml.explanation_integrity.controlled import (
    build_controlled_cases,
    intervention_from_case,
)
from awareml.explanation_integrity.faithfulness import (
    FaithfulnessV2Evaluator,
)
from awareml.explanation_integrity.verifier import (
    GeneralEvidenceVerifier,
)


def _safe_mean(values):
    values = [float(v) for v in values if v is not None]
    return float(mean(values)) if values else None


class ReplayGenerator:
    source = "phase15-rescore-replay"

    def __init__(self, text, model=None):
        self.text = str(text)
        self.model = model

    def generate(self, case):
        return self.text, {
            "source": self.source,
            "model": self.model,
        }


def rescore(path: Path):
    raw = path.read_bytes()
    source = json.loads(raw.decode("utf-8"))

    cases = {
        case.case_id: case
        for case in build_controlled_cases()
    }

    verifier = GeneralEvidenceVerifier()
    evaluator = FaithfulnessV2Evaluator(verifier=verifier)

    correctness_reports = []
    faithfulness_records = []
    per_case = []

    source_correctness = {
        row["case_id"]: row
        for row in source.get("correctness_reports") or []
    }
    source_faithfulness = {
        row["case_id"]: row
        for row in source.get("faithfulness_records") or []
    }

    ordered_case_ids = [
        row["case_id"]
        for row in source.get("faithfulness_records") or []
    ]

    for case_id in ordered_case_ids:
        if case_id not in cases:
            raise KeyError(
                "Saved run case {!r} does not exist in the current controlled "
                "benchmark definition.".format(case_id)
            )

        case = cases[case_id]
        old_correctness = source_correctness.get(case_id) or {}
        old_faith = source_faithfulness.get(case_id) or {}

        original_text = (
            old_correctness.get("explanation")
            or old_faith.get("original_explanation")
            or ""
        )
        counterfactual_text = old_faith.get("counterfactual_explanation") or ""

        correctness = verifier.verify(case, original_text).to_dict()
        correctness["generation_meta"] = old_correctness.get("generation_meta")
        correctness_reports.append(correctness)

        replay = ReplayGenerator(
            counterfactual_text,
            model=source.get("model"),
        )
        faith = evaluator.evaluate(
            case,
            intervention_from_case(case),
            replay,
            original_explanation=original_text,
        ).to_dict()
        faithfulness_records.append(faith)

        per_case.append({
            "case_id": case_id,
            "source_name": case.source_name,
            "claim_precision": correctness["metrics"].get("claim_precision"),
            "numeric_correctness": correctness["metrics"].get("numeric_correctness"),
            "citation_validity": correctness["metrics"].get("citation_validity"),
            "citation_coverage": correctness["metrics"].get("citation_coverage"),
            "decision_consistency": correctness["metrics"].get("decision_consistency"),
            "unsupported_claim_rate": correctness["metrics"].get("unsupported_claim_rate"),
            "contradiction_rate": correctness["metrics"].get("contradiction_rate"),
            "faithfulness_score": faith.get("faithfulness_score"),
            "changed_evidence_acknowledged": faith.get(
                "changed_evidence_acknowledged"
            ),
            "numeric_update_accuracy": faith.get("numeric_update_accuracy"),
            "stale_claim_rate": faith.get("stale_claim_rate"),
            "decision_update_consistency": faith.get(
                "decision_update_consistency"
            ),
        })

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

    correctness_summary = {
        name: _safe_mean(
            report["metrics"].get(name)
            for report in correctness_reports
        )
        for name in metric_names
    }

    faithfulness_summary = {
        "mean_faithfulness_score": _safe_mean(
            row.get("faithfulness_score")
            for row in faithfulness_records
        ),
        "mean_changed_evidence_acknowledged": _safe_mean(
            row.get("changed_evidence_acknowledged")
            for row in faithfulness_records
        ),
        "mean_numeric_update_accuracy": _safe_mean(
            row.get("numeric_update_accuracy")
            for row in faithfulness_records
        ),
        "mean_stale_claim_rate": _safe_mean(
            row.get("stale_claim_rate")
            for row in faithfulness_records
        ),
        "mean_decision_update_consistency": _safe_mean(
            row.get("decision_update_consistency")
            for row in faithfulness_records
        ),
    }

    return {
        "schema_version": "phase15_llm_rescore_v3",
        "rescore_method": (
            "Replay saved original/counterfactual LLM text through the current "
            "deterministic verifier and current coherent intervention definitions. "
            "No LLM generation is performed during rescoring."
        ),
        "verifier_version": "phase15_final_verifier_v1",
        "source_run_sha256": hashlib.sha256(raw).hexdigest(),
        "source_run_path": str(path),
        "model": source.get("model"),
        "case_count": len(per_case),
        "per_case": per_case,
        "correctness_reports": correctness_reports,
        "faithfulness_records": faithfulness_records,
        "summary": {
            "correctness": correctness_summary,
            "faithfulness": faithfulness_summary,
            "completion": source.get("summary", {}).get("completion"),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rescore a saved Phase-15 LLM run with the current deterministic "
            "verifier. This does not call Ollama."
        )
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.run)
    if not path.exists():
        raise FileNotFoundError(path)

    result = rescore(path)

    if args.output:
        out = Path(args.output)
    else:
        out = path.with_name(
            path.stem + "__rescored_v3.json"
        )

    out.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("Phase 15 · deterministic saved-run rescore")
    print("=" * 96)
    print("Source:", path.resolve())
    print("Output:", out.resolve())
    print("Model:", result.get("model"))
    print("Cases:", result.get("case_count"))
    print()
    for row in result["per_case"]:
        print(
            "{:<18} correctness={:<8} faithfulness={:<8} stale={}"
            .format(
                row["case_id"],
                row.get("claim_precision"),
                row.get("faithfulness_score"),
                row.get("stale_claim_rate"),
            )
        )
    print()
    print(json.dumps(result["summary"], indent=2))
    print("=" * 96)


if __name__ == "__main__":
    main()
