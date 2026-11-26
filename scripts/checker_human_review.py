"""Compare AwareML's unchanged checker with exact-stimulus human judgments.

Run from PyCharm or: python scripts/checker_human_review.py
No model calls, training, network access, or changes to study files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_reviews(paths):
    records, seen = [], set()
    required = {"bank_sha256", "review_id", "assessor_id", "explanation_verdict",
                "recommendation_verdict", "claim_or_reason", "reference_location", "scope_clear"}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing review columns in {path}")
            for line, row in enumerate(reader, 2):
                row = {k: (v or "").strip() for k, v in row.items() if k}
                if any(not row[k] for k in required):
                    raise ValueError(f"Incomplete assessment: {path}, line {line}")
                if row["explanation_verdict"] not in {"Supported", "Incorrect", "Insufficient evidence"}:
                    raise ValueError(f"Unknown human verdict: {path}, line {line}")
                key = (row["review_id"], row["assessor_id"])
                if key in seen:
                    raise ValueError(f"Duplicate assessor/item: {key}. Supply each review once.")
                seen.add(key)
                records.append(row)
    if not records:
        raise ValueError("No assessments found.")
    return records


def align_reviews(records, bank, current_hash, identity, expected_items=40):
    if len(bank) != expected_items:
        raise ValueError(f"Expected {expected_items} exact bank items; found {len(bank)}.")
    hashes = {r["bank_sha256"] for r in records}
    if hashes != {current_hash}:
        raise ValueError(
            "STIMULUS VERSION MISMATCH. No checker scores were calculated.\n"
            f"Review bank hash(es): {sorted(hashes)}\nCurrent local bank hash: {current_hash}\n"
            "Open the project checkout and exact frozen bank used by the assessors. "
            "Also retain its matching task_context_revision.py and presentation files. "
            "Do not edit CSV hashes, rebuild the bank, or substitute review comments for explanations.")
    indexed = {}
    for item in bank:
        rid = identity(item)
        if rid in indexed:
            raise ValueError("Duplicate item identity in the bank.")
        indexed[rid] = item
    groups = {}
    for rec in records:
        if rec["review_id"] not in indexed:
            raise ValueError(f"Review ID does not match the local presentation: {rec['review_id']}")
        groups.setdefault(rec["review_id"], []).append(rec)
    if set(groups) != set(indexed):
        raise ValueError("Reviews do not cover every bank item. No subset is silently scored.")
    aligned = []
    for rid in sorted(indexed):
        reviews = groups[rid]
        if len({r["assessor_id"] for r in reviews}) < 2:
            raise ValueError(f"At least two assessors are required for {rid}.")
        labels = {r["explanation_verdict"] for r in reviews}
        if len(labels) != 1 or "Insufficient evidence" in labels:
            raise ValueError(f"Unresolved human disagreement/uncertainty for {rid}; adjudicate first.")
        aligned.append((rid, indexed[rid], reviews, next(iter(labels))))
    return aligned


def checker_verdict(report):
    claims = report.get("claims", [])
    counts = Counter(c.get("status") for c in claims)
    if set(counts) - {"supported", "contradicted", "unsupported"}:
        raise ValueError("Checker returned an unknown claim status.")
    # Predeclared explanation-level aggregation; not a new checker algorithm.
    if counts["contradicted"]:
        label = "Incorrect"
    elif not claims or counts["unsupported"]:
        label = "Insufficient evidence"
    else:
        label = "Supported"
    return label, counts


def ratio(a, b):
    return a / b if b else None


def summarize(rows):
    matrix = {h: {c: 0 for c in ["Supported", "Incorrect", "Insufficient evidence"]}
              for h in ["Supported", "Incorrect"]}
    for row in rows:
        matrix[row["human_verdict"]][row["checker_verdict"]] += 1
    supported, incorrect = matrix["Supported"], matrix["Incorrect"]
    ns, ni = sum(supported.values()), sum(incorrect.values())
    return {
        "items": len(rows), "human_supported": ns, "human_incorrect": ni,
        "confusion_matrix_human_rows_checker_columns": matrix,
        "exact_verdict_agreement": ratio(supported["Supported"] + incorrect["Incorrect"], len(rows)),
        "incorrect_explanations_explicitly_detected": incorrect["Incorrect"],
        "incorrect_explanations_wrongly_accepted": incorrect["Supported"],
        "incorrect_explanations_unresolved": incorrect["Insufficient evidence"],
        "explicit_error_detection_rate": ratio(incorrect["Incorrect"], ni),
        "wrong_acceptance_rate_on_incorrect": ratio(incorrect["Supported"], ni),
        "false_contradiction_rate_on_supported": ratio(supported["Incorrect"], ns),
        "nonacceptance_rate_on_supported": ratio(supported["Incorrect"] + supported["Insufficient evidence"], ns),
        "abstention_rate": ratio(sum(r["checker_verdict"] == "Insufficient evidence" for r in rows), len(rows)),
        "interpretation": "Explanation-level diagnostic agreement on reviewed study items. "
                          "An abstention is not credited as explicit error detection. "
                          "These are not claim-level human error rates or a faithfulness intervention evaluation.",
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reviews", type=Path, nargs="+", help="Two or more assessor CSV paths")
    parser.add_argument("--bank", type=Path, help="Exact reviewed bank directory (optional)")
    parser.add_argument("--output", type=Path, help="New output directory; never overwritten")
    parser.add_argument("--expected-items", type=int, default=40)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if not (root / "awareml").is_dir():
        raise ValueError("Extract this package into the AwareML project root, alongside awareml/.")
    reviews = args.reviews or [root / "checker_human_review_inputs" / name
                              for name in ["assessor_review.csv", "assessor_review_v2.csv"]]
    records = read_reviews(reviews)
    sys.path.insert(0, str(root))
    review_module = importlib.import_module("awareml.studies.morten_review_v3")
    context_module = importlib.import_module("awareml.studies.task_context_revision")
    bank_root = (args.bank or root / "data/journal/trust_stimulus_bank_v3/frozen").resolve()
    bank_path = bank_root / "stimuli_researcher.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_hash = context_module.review_bank_sha(bank_root)
    aligned = align_reviews(records, bank, bank_hash, review_module.review_identity, args.expected_items)

    from awareml.explanation_integrity.schemas import EvidenceCase
    from awareml.explanation_integrity.verifier import GeneralEvidenceVerifier
    verifier = GeneralEvidenceVerifier()
    items, reports, claim_rows = [], [], []
    for rid, item, human_reviews, human_label in aligned:
        participant = item.get("participant_version", {})
        text = participant.get("explanation")
        evidence = participant.get("reference_evidence")
        if not isinstance(text, str) or not text.strip() or not isinstance(evidence, dict) or not evidence:
            raise ValueError(f"Exact participant explanation/reference evidence is missing for {rid}.")
        if item.get("explanation") != text or item.get("evidence_summary") != evidence:
            raise ValueError(f"Participant and researcher copies differ for {rid}. Resolve before scoring.")
        case = EvidenceCase(case_id=item["stimulus_id"], dataset_id=item.get("dataset_id", ""),
                            source_stage=item.get("source_stage", ""), source_name=item.get("source_name", ""),
                            evidence=evidence, prompt=participant.get("prompt", ""), metadata={})
        report = verifier.verify(case, text).to_dict()
        verdict, counts = checker_verdict(report)
        record = {"review_id": rid, "stimulus_id": item["stimulus_id"],
                  "pair_id": item.get("pair_id", ""), "source_name": case.source_name,
                  "human_verdict": human_label, "checker_verdict": verdict,
                  "verdict_agrees": verdict == human_label, "claims_total": len(report["claims"]),
                  "claims_supported": counts["supported"], "claims_contradicted": counts["contradicted"],
                  "claims_unsupported": counts["unsupported"], "explanation_sha256": report["explanation_sha256"],
                  "explanation": text,
                  "human_reasons": json.dumps([{k: r[k] for k in ["assessor_id", "claim_or_reason", "reference_location"]}
                                                for r in human_reviews], ensure_ascii=False)}
        items.append(record)
        reports.append({"review_id": rid, "human_verdict": human_label, "checker_verdict": verdict,
                        "report": report})
        for number, claim in enumerate(report["claims"], 1):
            claim_rows.append({"review_id": rid, "stimulus_id": item["stimulus_id"], "claim_number": number,
                               "claim_text": claim["claim"].get("text"), "status": claim.get("status"),
                               "expected_value": json.dumps(claim.get("expected_value"), ensure_ascii=False),
                               "observed_value": json.dumps(claim.get("observed_value"), ensure_ascii=False),
                               "matched_evidence_key": claim.get("matched_evidence_key"), "reason": claim.get("reason")})
    summary = summarize(items)
    summary.update({"assessment_records": len(records), "assessors": sorted({r["assessor_id"] for r in records}),
                    "unanimous_reviewed_items": len(aligned), "bank_sha256": bank_hash,
                    "scope": "Post-hoc diagnostic validation on human-reviewed study explanations; "
                             "does not establish independent reviewer conduct or generalization to all explanations.",
                    "aggregation_rule": "Any contradicted claim => Incorrect; otherwise unsupported/no claims => "
                                        "Insufficient evidence; otherwise all extracted claims supported => Supported."})
    modules = ["awareml.explanation_integrity.verifier", "awareml.explanation_integrity.claims",
               "awareml.explanation_integrity.evidence", "awareml.explanation_integrity.schemas",
               "awareml.studies.morten_review_v3", "awareml.studies.task_context_revision"]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                  "bank_file": str(bank_path), "bank_file_sha256": file_sha(bank_path),
                  "reviews": [{"file": str(p.resolve()), "sha256": file_sha(p)} for p in reviews],
                  "checker_source_files": {name: {"path": inspect.getfile(importlib.import_module(name)),
                                                  "sha256": file_sha(inspect.getfile(importlib.import_module(name)))}
                                           for name in modules}, "script_sha256": file_sha(__file__)}
    try:
        provenance["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance["git_commit"] = None
    out = (args.output or root / "artifacts" / "checker_human_review" /
           datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "item_results.csv", items)
    write_csv(out / "claim_results.csv", claim_rows)
    mismatches = [r for r in items if not r["verdict_agrees"]]
    write_csv(out / "disagreements.csv", mismatches or [{"note": "No explanation-level disagreements."}])
    for name, value in [("summary.json", summary), ("checker_reports.json", reports), ("provenance.json", provenance)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    report_text = (
        "# Checker versus human assessment\n\n"
        f"{len(aligned)} explanations; {len(records)} assessments; {len(summary['assessors'])} assessor IDs.\n\n"
        "| Human verdict | Checker: Supported | Checker: Incorrect | Checker: Insufficient evidence |\n"
        "|---|---:|---:|---:|\n")
    for human, counts in summary["confusion_matrix_human_rows_checker_columns"].items():
        report_text += f"| {human} | {counts['Supported']} | {counts['Incorrect']} | {counts['Insufficient evidence']} |\n"
    report_text += (
        f"\nExact verdict agreement: {percent(summary['exact_verdict_agreement'])}.\n\n"
        f"Explicit error detection on human-incorrect explanations: {percent(summary['explicit_error_detection_rate'])}.\n\n"
        f"Wrong acceptance of human-incorrect explanations: {percent(summary['wrong_acceptance_rate_on_incorrect'])}.\n\n"
        f"False contradiction on human-supported explanations: {percent(summary['false_contradiction_rate_on_supported'])}.\n\n"
        f"Insufficient-evidence rate: {percent(summary['abstention_rate'])}.\n\n"
        "These are explanation-level diagnostics on the reviewed study bank. Unextracted errors can be missed. "
        "Human agreement is not proof that every reference label is correct. Abstentions are reported separately, "
        "not counted as successful explicit error detection. No evidence-change/faithfulness claims are tested here.\n\n"
        "Inspect disagreements.csv and claim_results.csv before reporting results. If checker rules are changed "
        "after inspecting these items, treat these items as development data and use new reviewed items for an "
        "independent evaluation.\n")
    (out / "REPORT.md").write_text(report_text, encoding="utf-8")
    print(f"Completed: {summary['items']} matched explanations, {summary['assessment_records']} assessments.")
    print(f"Exact verdict agreement: {percent(summary['exact_verdict_agreement'])}")
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"CHECK STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(1)
