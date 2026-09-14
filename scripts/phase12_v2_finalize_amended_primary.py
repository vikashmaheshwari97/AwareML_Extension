#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

SEED = "20260914"
TARGET_PER_K = 15
MIN_HUMAN = 24
AMENDMENT_ID = "human_enrichment_H041_H055"

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "journal" / "objective_selection_benchmark_v2"
WORK = ROOT / "phase12_v2_human_enrichment_H041_H055"

ORIG_GT = BASE / "results" / "ground_truth_all.csv"
ORIG_HARD = BASE / "results" / "hard_cases.csv"
DESIGN_MANIFEST = BASE / "frozen" / "design_manifest.json"
GT_MANIFEST = BASE / "frozen" / "ground_truth_manifest.json"

AMEND_GT = WORK / "results" / "amendment_ground_truth.csv"
AMEND_HARD = WORK / "results" / "amendment_hard_cases.csv"

FORMAL_AMEND_DIR = BASE / "amendments" / AMENDMENT_ID
FINAL_CSV = BASE / "results" / "primary_benchmark_amended.csv"
FINAL_MANIFEST = BASE / "frozen" / "amended_primary_manifest.json"
FINAL_MANIFEST_SHA = BASE / "frozen" / "amended_primary_manifest.json.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truthy(v: str) -> bool:
    return str(v).strip().lower() in {"yes", "true", "1", "y"}


def k_value(row: dict) -> int:
    for key in ("k_prime", "kprime", "k", "objective_count"):
        if key in row and str(row[key]).strip():
            return int(float(str(row[key]).strip()))
    raise RuntimeError(f"Missing k-prime column for {row.get('scenario_id')}")


def source_type(sid: str) -> str:
    if sid.startswith("H"):
        return "human"
    if sid.startswith("G"):
        return "generated"
    raise RuntimeError(f"Unknown scenario source prefix: {sid}")


def hard_ids_from_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["scenario_id"].strip() for r in read_csv(path) if r.get("scenario_id", "").strip()}


def is_hard(row: dict, hard_ids: set[str]) -> bool:
    sid = row["scenario_id"].strip()
    if sid in hard_ids:
        return True
    if "hard_case" in row and str(row["hard_case"]).strip():
        return truthy(row["hard_case"])
    return False


def stable_key(row: dict) -> str:
    # Deterministic tie-break only. No model output, private intent, or generator intent is read.
    sid = row["scenario_id"].strip()
    return hashlib.sha256(
        f"phase12-v2-amended-primary|{SEED}|{sid}".encode("utf-8")
    ).hexdigest()


def require_inputs():
    required = [
        ORIG_GT,
        DESIGN_MANIFEST,
        GT_MANIFEST,
        AMEND_GT,
        WORK / "collection" / "human_enrichment_candidates_H041_H055.csv",
        WORK / "provenance" / "amendment_protocol.json",
        WORK / "provenance" / "H041_H055_REALISM_AUDIT.json",
        WORK / "provenance" / "amendment_annotation_audit.json",
        WORK / "realism_review" / "human_enrichment_realism_filter_H041_H055.csv",
        WORK / "annotations" / "annotations_amendment_A.csv",
        WORK / "annotations" / "annotations_amendment_B.csv",
        WORK / "annotations" / "annotations_amendment_C.csv",
        WORK / "results" / "amendment_annotation_agreement.json",
        WORK / "results" / "amended_primary_feasibility.json",
        AMEND_HARD,
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required inputs:\n  " + "\n  ".join(missing))


def build_plan():
    require_inputs()

    original_rows = read_csv(ORIG_GT)
    amendment_rows = read_csv(AMEND_GT)
    orig_hard = hard_ids_from_file(ORIG_HARD)
    amend_hard = hard_ids_from_file(AMEND_HARD)

    seen = set()
    eligible = []

    for origin, rows, hard_ids in (
        ("original", original_rows, orig_hard),
        ("amendment", amendment_rows, amend_hard),
    ):
        for raw in rows:
            sid = raw.get("scenario_id", "").strip()
            if not sid:
                raise RuntimeError(f"{origin}: row without scenario_id")
            if sid in seen:
                raise RuntimeError(f"Duplicate scenario_id across combined pool: {sid}")
            seen.add(sid)

            k = k_value(raw)
            hard = is_hard(raw, hard_ids)

            if not hard and k in {1, 2, 3, 4}:
                row = dict(raw)
                row["_origin"] = origin
                row["_source_type"] = source_type(sid)
                row["_k_prime"] = k
                eligible.append(row)

    counts = {}
    chosen = []
    for k in (1, 2, 3, 4):
        pool = [r for r in eligible if r["_k_prime"] == k]
        humans = sorted(
            [r for r in pool if r["_source_type"] == "human"],
            key=stable_key,
        )
        generated = sorted(
            [r for r in pool if r["_source_type"] == "generated"],
            key=stable_key,
        )

        if len(pool) < TARGET_PER_K:
            raise RuntimeError(
                f"k'={k} has only {len(pool)} eligible cases; need {TARGET_PER_K}"
            )

        # Predeclared amendment selection rule:
        # maximize eligible genuine-human inclusion within each k' stratum,
        # capped by the fixed 15 slots; fill the rest deterministically from generated cases.
        take_h = humans[:TARGET_PER_K]
        remaining = TARGET_PER_K - len(take_h)
        take_g = generated[:remaining]
        selected_k = take_h + take_g
        if len(selected_k) != TARGET_PER_K:
            raise RuntimeError(f"Could not fill k'={k} to {TARGET_PER_K}")

        counts[str(k)] = {
            "eligible_total": len(pool),
            "eligible_human": len(humans),
            "eligible_generated": len(generated),
            "selected_human": len(take_h),
            "selected_generated": len(take_g),
        }
        chosen.extend(selected_k)

    human_n = sum(r["_source_type"] == "human" for r in chosen)
    generated_n = len(chosen) - human_n
    if human_n < MIN_HUMAN:
        raise RuntimeError(
            f"Human quota infeasible under fixed balancing: selected {human_n}, need >= {MIN_HUMAN}"
        )

    # Final deterministic order: k' then stable rank.
    final = []
    for k in (1, 2, 3, 4):
        selected_k = [r for r in chosen if r["_k_prime"] == k]
        selected_k = sorted(selected_k, key=stable_key)
        for rank, r in enumerate(selected_k, 1):
            out = {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
            out["k_prime"] = str(k)
            out["source_type"] = r["_source_type"]
            out["origin"] = r["_origin"]
            out["selection_rank_within_k"] = str(rank)
            out["selection_tiebreak_sha256"] = stable_key(r)
            final.append(out)

    summary = {
        "combined_eligible_cases": len(eligible),
        "original_eligible_cases": sum(r["_origin"] == "original" for r in eligible),
        "amendment_eligible_cases": sum(r["_origin"] == "amendment" for r in eligible),
        "selected_cases": len(final),
        "selected_human": human_n,
        "selected_generated": generated_n,
        "selected_original": sum(r["origin"] == "original" for r in final),
        "selected_amendment": sum(r["origin"] == "amendment" for r in final),
        "k_prime": counts,
    }
    return final, summary


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    preferred = [
        "scenario_id", "scenario", "accuracy", "runtime", "energy", "co2",
        "k_prime", "source_type", "origin", "selection_rank_within_k",
        "selection_tiebreak_sha256"
    ]
    fields = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def amendment_files():
    rels = [
        "collection/human_enrichment_candidates_H041_H055.csv",
        "provenance/amendment_protocol.json",
        "provenance/H041_H055_REALISM_AUDIT.json",
        "provenance/amendment_annotation_audit.json",
        "realism_review/human_enrichment_realism_filter_H041_H055.csv",
        "annotations/annotations_amendment_A.csv",
        "annotations/annotations_amendment_B.csv",
        "annotations/annotations_amendment_C.csv",
        "results/amendment_ground_truth.csv",
        "results/amendment_hard_cases.csv",
        "results/amendment_annotation_agreement.json",
        "results/amended_primary_feasibility.json",
    ]
    return [WORK / x for x in rels]


def validate():
    final, summary = build_plan()

    k_selected = {
        str(k): sum(int(r["k_prime"]) == k for r in final)
        for k in (1, 2, 3, 4)
    }
    print(json.dumps({
        "status": "PASS_READY_TO_FREEZE_AMENDED_PRIMARY",
        "selection_seed": SEED,
        "selection_policy": (
            "within each fixed k' stratum, include as many eligible human-written cases "
            "as possible up to 15; fill remaining slots from generated cases; deterministic "
            "SHA-256 ranking by scenario_id breaks ties"
        ),
        "selected_k_prime_counts": k_selected,
        **summary,
        "model_outputs_used_for_selection": False,
        "private_generator_intent_used": False,
        "existing_frozen_manifests_will_be_overwritten": False,
    }, indent=2))


def freeze():
    final, summary = build_plan()

    for p in (FINAL_CSV, FINAL_MANIFEST, FINAL_MANIFEST_SHA):
        if p.exists():
            raise RuntimeError(
                f"Refusing to overwrite frozen amended-primary artifact: {p.relative_to(ROOT)}"
            )
    if FORMAL_AMEND_DIR.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing formal amendment directory: {FORMAL_AMEND_DIR.relative_to(ROOT)}"
        )

    protected_before = {
        str(DESIGN_MANIFEST.relative_to(ROOT)): sha256(DESIGN_MANIFEST),
        str(GT_MANIFEST.relative_to(ROOT)): sha256(GT_MANIFEST),
        str(ORIG_GT.relative_to(ROOT)): sha256(ORIG_GT),
    }

    FORMAL_AMEND_DIR.mkdir(parents=True, exist_ok=False)
    copied = {}
    for src in amendment_files():
        rel = src.relative_to(WORK)
        dst = FORMAL_AMEND_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[str(dst.relative_to(ROOT)).replace("\\", "/")] = sha256(dst)

    write_csv(FINAL_CSV, final)

    selected_ids = [r["scenario_id"] for r in final]
    selected_k = {
        str(k): sum(int(r["k_prime"]) == k for r in final)
        for k in (1, 2, 3, 4)
    }
    human_by_k = {
        str(k): sum(int(r["k_prime"]) == k and r["source_type"] == "human" for r in final)
        for k in (1, 2, 3, 4)
    }

    manifest = {
        "artifact": "objective_selection_benchmark_v2_amended_primary",
        "release_status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_id": AMENDMENT_ID,
        "reason": (
            "The original frozen Phase-12-v2 annotation pool could not satisfy the "
            "predeclared >=24 human-written final-primary quota together with exact "
            "15/15/15/15 k-prime balancing after the frozen hard-case policy. "
            "H041-H055 were collected and independently reviewed/annotated before "
            "any confirmatory V2/V3.2 model evaluation."
        ),
        "parent_artifacts": {
            "design_manifest": {
                "path": str(DESIGN_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(DESIGN_MANIFEST),
            },
            "ground_truth_manifest": {
                "path": str(GT_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(GT_MANIFEST),
            },
            "ground_truth_all": {
                "path": str(ORIG_GT.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(ORIG_GT),
            },
        },
        "amendment_artifact_hashes": copied,
        "selection": {
            "primary_csv": str(FINAL_CSV.relative_to(ROOT)).replace("\\", "/"),
            "primary_csv_sha256": sha256(FINAL_CSV),
            "selection_seed": SEED,
            "selection_policy": (
                "Exactly 15 cases are selected for each human-confirmed k'=1,2,3,4. "
                "Within each k' stratum, all eligible human-written cases are prioritized "
                "up to the 15-slot cap; remaining slots are filled by eligible generated "
                "cases. Deterministic SHA-256 ranking by scenario_id is used only as a "
                "tie-breaker within source groups."
            ),
            "selection_inputs": [
                "human majority-vote ground truth",
                "frozen hard-case status",
                "source type (human/generated)",
                "human-confirmed k-prime",
                "deterministic scenario-id hash tie-break",
            ],
            "model_outputs_used": False,
            "private_generator_intent_used": False,
            "generator_intent_used_as_ground_truth": False,
            "selected_n": len(final),
            "selected_k_prime_counts": selected_k,
            "selected_human": summary["selected_human"],
            "selected_generated": summary["selected_generated"],
            "selected_human_by_k_prime": human_by_k,
            "selected_original": summary["selected_original"],
            "selected_amendment": summary["selected_amendment"],
            "selected_scenario_ids": selected_ids,
        },
        "feasibility": summary,
        "immutability": {
            "existing_design_manifest_overwritten": False,
            "existing_ground_truth_manifest_overwritten": False,
            "existing_ground_truth_all_overwritten": False,
            "existing_annotations_modified_by_this_script": False,
        },
    }

    FINAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256(FINAL_MANIFEST)
    FINAL_MANIFEST_SHA.write_text(
        f"{digest}  {FINAL_MANIFEST.name}\n",
        encoding="utf-8",
    )

    protected_after = {
        str(DESIGN_MANIFEST.relative_to(ROOT)): sha256(DESIGN_MANIFEST),
        str(GT_MANIFEST.relative_to(ROOT)): sha256(GT_MANIFEST),
        str(ORIG_GT.relative_to(ROOT)): sha256(ORIG_GT),
    }
    if protected_before != protected_after:
        raise RuntimeError("Protected original frozen artifacts changed unexpectedly")

    print(json.dumps({
        "status": "FROZEN",
        "artifact": manifest["artifact"],
        "primary_csv": str(FINAL_CSV.relative_to(ROOT)).replace("\\", "/"),
        "primary_csv_sha256": sha256(FINAL_CSV),
        "manifest": str(FINAL_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": digest,
        "selected_n": len(final),
        "selected_k_prime_counts": manifest["selection"]["selected_k_prime_counts"],
        "selected_human": manifest["selection"]["selected_human"],
        "selected_generated": manifest["selection"]["selected_generated"],
        "selected_human_by_k_prime": manifest["selection"]["selected_human_by_k_prime"],
        "selected_original": manifest["selection"]["selected_original"],
        "selected_amendment": manifest["selection"]["selected_amendment"],
        "existing_frozen_artifacts_unchanged": True,
        "model_outputs_used_for_selection": False,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["validate", "freeze"])
    args = ap.parse_args()
    if args.command == "validate":
        validate()
    else:
        freeze()


if __name__ == "__main__":
    main()
