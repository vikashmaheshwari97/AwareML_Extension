from __future__ import annotations

"""Phase-12-v2 confirmatory benchmark pipeline for AwareML.

The central safeguards are procedural as much as statistical:

1. The frozen Phase-12-v1 benchmark is never edited or reused as a final test.
2. Objective Selector V3.2 must be frozen before final v2 labels/outcomes exist.
3. Generator intent is private design metadata and is never ground truth.
4. The final 60-case benchmark is selected only from independent human labels.
5. No evaluated selector is run before the final primary benchmark is frozen.
6. Legacy V2 and V3.2 are evaluated on exactly the same untouched cases.

This module intentionally uses only the Python standard library so the benchmark
protocol does not introduce new scientific dependencies into AwareML.
"""

import csv
import hashlib
import itertools
import json
import math
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


OBJECTIVES: Tuple[str, ...] = ("Accuracy", "Runtime", "Energy", "CO2")
OBJECTIVE_COLUMNS: Tuple[str, ...] = ("accuracy", "runtime", "energy", "co2")
REJECTION_STATUSES = {"ambiguous", "contradictory", "out_of_scope"}
EXPLICIT_OBJECTIVE_RE = re.compile(r"\b(?:accuracy|runtime|energy|co2|co₂)\b", re.I)
PROTOCOL_REL = Path("configs/journal/phase12_v2_protocol.json")
BASE_REL = Path("data/journal/objective_selection_benchmark_v2")
V32_MANIFEST_REL = Path("data/journal/objective_selection_v32/manifest.json")
LEGACY_V1_MANIFEST_REL = Path("data/journal/objective_selection_benchmark_v1/frozen/manifest.json")
LEGACY_V2_SELECTOR_REL = Path("data/journal/objective_selection_v2/manifest.json")


class Phase12V2Error(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root(root: Optional[Path] = None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_bool(value: Any, *, allow_blank: bool = False) -> Optional[bool]:
    text = "" if value is None else str(value).strip().lower()
    if allow_blank and not text:
        return None
    if text in {"1", "true", "yes", "y", "keep", "include", "checked"}:
        return True
    if text in {"0", "false", "no", "n", "drop", "exclude", "unchecked"}:
        return False
    raise Phase12V2Error("Expected yes/no or 1/0, got {!r}.".format(value))


def objective_set_from_columns(row: Mapping[str, Any]) -> Set[str]:
    selected: Set[str] = set()
    for objective, column in zip(OBJECTIVES, OBJECTIVE_COLUMNS):
        if parse_bool(row.get(column), allow_blank=False):
            selected.add(objective)
    return selected


def labels_to_columns(labels: Iterable[str]) -> Dict[str, int]:
    label_set = set(labels)
    return {
        column: int(objective in label_set)
        for objective, column in zip(OBJECTIVES, OBJECTIVE_COLUMNS)
    }


def labels_from_pipe(value: str) -> List[str]:
    labels = [x.strip() for x in str(value or "").split("|") if x.strip()]
    unknown = sorted(set(labels) - set(OBJECTIVES))
    if unknown:
        raise Phase12V2Error("Unknown objective label(s): {}".format(", ".join(unknown)))
    return [x for x in OBJECTIVES if x in labels]


def labels_to_pipe(labels: Iterable[str]) -> str:
    label_set = set(labels)
    return "|".join(obj for obj in OBJECTIVES if obj in label_set)


def load_protocol(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    path = root / PROTOCOL_REL
    if not path.exists():
        raise Phase12V2Error("Missing Phase-12-v2 protocol: {}".format(path))
    payload = read_json(path)
    if payload.get("protocol_id") != "objective_selection_benchmark_v2":
        raise Phase12V2Error("Unexpected Phase-12-v2 protocol_id.")
    return payload


def _base(root: Optional[Path] = None) -> Path:
    return repo_root(root) / BASE_REL


def _ensure_no_model_outputs(root: Path) -> None:
    base = _base(root)
    forbidden = [
        base / "results" / "baseline_primary_outputs.jsonl",
        base / "results" / "v32_primary_outputs.jsonl",
        base / "results" / "baseline_paraphrase_outputs.jsonl",
        base / "results" / "v32_paraphrase_outputs.jsonl",
        base / "results" / "baseline_adversarial_outputs.jsonl",
        base / "results" / "v32_adversarial_outputs.jsonl",
    ]
    present = [p for p in forbidden if p.exists()]
    if present:
        raise Phase12V2Error(
            "Evaluated-model outputs already exist before this freeze step: {}".format(
                ", ".join(str(p.relative_to(root)) for p in present)
            )
        )


def verify_legacy_v1_immutable(root: Optional[Path] = None) -> Dict[str, Any]:
    """Verify the old Phase-12 artifact using its own adjacent checksum."""
    root = repo_root(root)
    manifest = root / LEGACY_V1_MANIFEST_REL
    if not manifest.exists():
        raise Phase12V2Error("Frozen Phase-12-v1 manifest is missing.")
    sha_file = Path(str(manifest) + ".sha256")
    if not sha_file.exists():
        raise Phase12V2Error("Frozen Phase-12-v1 manifest checksum is missing.")
    expected = sha_file.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(manifest)
    if actual != expected:
        raise Phase12V2Error(
            "Phase-12-v1 immutable baseline failed checksum validation."
        )
    payload = read_json(manifest)
    if payload.get("release_status") != "frozen":
        raise Phase12V2Error("Phase-12-v1 baseline is not marked frozen.")
    return {
        "path": str(LEGACY_V1_MANIFEST_REL).replace("\\", "/"),
        "sha256": actual,
        "release_status": payload.get("release_status"),
        "primary_benchmark_n": payload.get("primary_benchmark_n"),
    }


def verify_selector_freezes(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    v2 = root / LEGACY_V2_SELECTOR_REL
    v32 = root / V32_MANIFEST_REL
    if not v2.exists():
        raise Phase12V2Error("Frozen Phase-11 V2 selector manifest is missing.")
    if not v32.exists():
        raise Phase12V2Error(
            "V3.2 selector is not frozen. Run scripts/freeze_objective_selector_v32.py first."
        )
    v2_payload = read_json(v2)
    v32_payload = read_json(v32)
    if v2_payload.get("release_status") != "frozen":
        raise Phase12V2Error("Legacy V2 selector is not frozen.")
    if v32_payload.get("release_status") != "frozen":
        raise Phase12V2Error("V3.2 selector is not frozen.")
    return {
        "legacy_v2": {
            "path": str(LEGACY_V2_SELECTOR_REL).replace("\\", "/"),
            "sha256": sha256_file(v2),
        },
        "v32": {
            "path": str(V32_MANIFEST_REL).replace("\\", "/"),
            "sha256": sha256_file(v32),
            "selector_id": v32_payload.get("selector_id"),
        },
    }


def _validate_unique(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> None:
    values = [str(row.get(key, "")).strip() for row in rows]
    if any(not x for x in values):
        raise Phase12V2Error("{} contains blank {}.".format(label, key))
    if len(values) != len(set(values)):
        raise Phase12V2Error("{} contains duplicate {} values.".format(label, key))


def _normalized_scenario(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", str(text or "").lower())).strip()


def _token_jaccard_text(a: str, b: str) -> float:
    sa = set(_normalized_scenario(a).split())
    sb = set(_normalized_scenario(b).split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / float(len(sa | sb)) if (sa | sb) else 0.0


def validate_no_legacy_primary_reuse(root: Path, new_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Block exact or near-copy reuse of legacy Phase-12 primary/design scenarios."""
    legacy_files = [
        root / "data" / "journal" / "objective_selection_benchmark_v1" / "design" / "candidate_pool.generated.csv",
        root / "data" / "journal" / "objective_selection_benchmark_v1" / "human" / "human_written_scenarios.csv",
        root / "data" / "journal" / "objective_selection_benchmark_v1" / "human" / "final_annotation_pool.csv",
    ]
    legacy = []
    for path in legacy_files:
        if not path.exists():
            continue
        for row in read_csv(path):
            text = str(row.get("scenario", row.get("base_scenario", ""))).strip()
            if text:
                legacy.append((str(path.relative_to(root)), text))
    checked = 0
    for row in new_rows:
        text = str(row.get("scenario", "")).strip()
        nt = _normalized_scenario(text)
        for legacy_path, old in legacy:
            checked += 1
            no = _normalized_scenario(old)
            if nt == no:
                raise Phase12V2Error(
                    "Fresh benchmark reuses a legacy Phase-12 scenario exactly: {}".format(row.get("scenario_id"))
                )
            # High-overlap long cases are almost certainly copied/lightly edited.
            if min(len(nt.split()), len(no.split())) >= 6 and _token_jaccard_text(nt, no) >= 0.92:
                raise Phase12V2Error(
                    "Fresh benchmark case {} is a near-copy of legacy material in {}.".format(
                        row.get("scenario_id"), legacy_path
                    )
                )
    return {
        "legacy_files_checked": [str(p.relative_to(root)) for p in legacy_files if p.exists()],
        "legacy_scenarios_loaded": len(legacy),
        "pairwise_overlap_checks": checked,
        "reuse_detected": False,
    }


def validate_design_inputs(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    cfg = load_protocol(root)
    base = _base(root)
    generated_path = base / "design" / "generated_candidates.csv"
    intent_path = base / "design" / "private" / "generated_intent.PRIVATE.csv"
    provenance_path = base / "design" / "generation_provenance.json"
    human_path = base / "human" / "human_candidates.csv"
    filter_path = base / "human" / "realism_filter.csv"

    for path in (generated_path, intent_path, provenance_path, human_path, filter_path):
        if not path.exists():
            raise Phase12V2Error("Missing Phase-12-v2 design input: {}".format(path))

    generated = read_csv(generated_path)
    intents = read_csv(intent_path)
    humans = read_csv(human_path)
    filter_rows = read_csv(filter_path)
    provenance = read_json(provenance_path)

    expected_generated = int(cfg["design"]["generated_candidate_count"])
    if len(generated) != expected_generated:
        raise Phase12V2Error(
            "Expected {} generated candidates, found {}.".format(expected_generated, len(generated))
        )
    if len(humans) < int(cfg["design"]["minimum_human_raw"]):
        raise Phase12V2Error(
            "Need at least {} genuinely human-written raw scenarios; found {}.".format(
                cfg["design"]["minimum_human_raw"], len(humans)
            )
        )

    _validate_unique(generated, "scenario_id", "generated_candidates.csv")
    _validate_unique(humans, "scenario_id", "human_candidates.csv")
    all_ids = [row["scenario_id"].strip() for row in generated + humans]
    if len(all_ids) != len(set(all_ids)):
        raise Phase12V2Error("Generated and human scenario IDs overlap.")

    texts = [str(row.get("scenario", "")).strip() for row in generated + humans]
    if any(not text for text in texts):
        raise Phase12V2Error("Candidate scenario text may not be blank.")
    normalized_texts = [re.sub(r"\s+", " ", text.lower()) for text in texts]
    if len(normalized_texts) != len(set(normalized_texts)):
        raise Phase12V2Error("Duplicate scenario text detected across the candidate pool.")
    for row in generated + humans:
        match = EXPLICIT_OBJECTIVE_RE.search(str(row.get("scenario", "")))
        if match:
            raise Phase12V2Error(
                "{} explicitly names frozen objective {!r}.".format(
                    row.get("scenario_id"), match.group(0)
                )
            )

    legacy_overlap = validate_no_legacy_primary_reuse(root, generated + humans)

    intent_map = {row["scenario_id"].strip(): row for row in intents}
    generated_ids = {row["scenario_id"].strip() for row in generated}
    if set(intent_map) != generated_ids:
        raise Phase12V2Error("Private generation-intent IDs do not match generated candidate IDs.")

    intended_counts = Counter()
    for row in intents:
        try:
            k = int(row.get("intended_k_prime", ""))
        except ValueError as exc:
            raise Phase12V2Error("Invalid intended_k_prime.") from exc
        labels = labels_from_pipe(row.get("generation_intent", ""))
        if k not in {1, 2, 3, 4} or len(labels) != k:
            raise Phase12V2Error(
                "Private generator intent is inconsistent for {}.".format(row["scenario_id"])
            )
        intended_counts[k] += 1

    expected_counts = {
        int(k): int(v) for k, v in cfg["design"]["intended_k_prime_counts"].items()
    }
    if dict(intended_counts) != expected_counts:
        raise Phase12V2Error(
            "Generated intended k' distribution is {}, expected {}.".format(
                dict(intended_counts), expected_counts
            )
        )

    batches = Counter(str(row.get("generation_batch", "")).strip() for row in generated)
    if "" in batches:
        raise Phase12V2Error("Every generated candidate needs generation_batch provenance.")
    if len(batches) < int(cfg["design"]["minimum_generation_batches"]):
        raise Phase12V2Error("Candidate generation did not use enough independent batches.")

    if provenance.get("evaluated_model_used_for_generation") is not False:
        raise Phase12V2Error("The evaluated LLaMA model must not generate the v2 benchmark.")
    generator_model = str(provenance.get("generator_model", "")).strip()
    if not generator_model:
        raise Phase12V2Error("generation_provenance.json must identify the external generator model.")
    if generator_model.lower() == "llama3:8b":
        raise Phase12V2Error("The evaluated LLaMA 3 8B cannot generate its own benchmark.")

    filter_map = {row.get("scenario_id", "").strip(): row for row in filter_rows}
    if set(filter_map) != set(all_ids):
        missing = sorted(set(all_ids) - set(filter_map))
        extra = sorted(set(filter_map) - set(all_ids))
        raise Phase12V2Error(
            "Realism filter must contain every candidate exactly once. Missing={}, extra={}.".format(
                missing[:8], extra[:8]
            )
        )

    kept = []
    kept_human = 0
    registry = {
        row["scenario_id"].strip(): dict(row, source="generated") for row in generated
    }
    registry.update({row["scenario_id"].strip(): dict(row, source="human") for row in humans})
    filterer_ids = set()
    for scenario_id, row in filter_map.items():
        if str(row.get("keep", "")).strip() == "":
            raise Phase12V2Error("Realism filter is incomplete for {}.".format(scenario_id))
        keep = bool(parse_bool(row.get("keep")))
        unclear = parse_bool(row.get("ambiguous_or_unclear"), allow_blank=True)
        if keep and unclear is True:
            raise Phase12V2Error(
                "{} cannot be kept while marked ambiguous_or_unclear=yes.".format(scenario_id)
            )
        reviewer = str(row.get("reviewer_id", "")).strip()
        if not reviewer:
            raise Phase12V2Error("Every realism-filter row needs reviewer_id.")
        filterer_ids.add(reviewer)
        if keep:
            kept.append(scenario_id)
            if registry[scenario_id]["source"] == "human":
                kept_human += 1

    if len(kept) < int(cfg["design"]["minimum_annotation_pool"]):
        raise Phase12V2Error(
            "Realism filtering retained only {} cases; at least {} are required before annotation.".format(
                len(kept), cfg["design"]["minimum_annotation_pool"]
            )
        )

    return {
        "generated_candidates": len(generated),
        "human_candidates": len(humans),
        "total_candidates": len(generated) + len(humans),
        "kept_for_annotation": len(kept),
        "kept_human": kept_human,
        "generation_batches": dict(sorted(batches.items())),
        "intended_k_prime_counts_private": {str(k): intended_counts[k] for k in sorted(intended_counts)},
        "generator_model": generator_model,
        "realism_filterers": sorted(filterer_ids),
        "legacy_overlap_audit": legacy_overlap,
    }


def freeze_design(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    _ensure_no_model_outputs(root)
    legacy = verify_legacy_v1_immutable(root)
    selectors = verify_selector_freezes(root)
    summary = validate_design_inputs(root)
    base = _base(root)
    out = base / "frozen" / "design_manifest.json"
    sha_path = Path(str(out) + ".sha256")

    protected = [
        root / PROTOCOL_REL,
        root / "prompts" / "phase12_v2_external_scenario_generation.md",
        root / "docs" / "PHASE12_V2_HUMAN_INSTRUCTIONS.md",
        base / "design" / "generated_candidates.csv",
        base / "design" / "private" / "generated_intent.PRIVATE.csv",
        base / "design" / "generation_provenance.json",
        base / "human" / "human_candidates.csv",
        base / "human" / "realism_filter.csv",
    ]

    if out.exists() and sha_path.exists():
        existing = validate_manifest_with_assets(root, out)
        return existing

    file_hashes = {}
    for path in protected:
        if not path.exists():
            raise Phase12V2Error("Missing design asset: {}".format(path))
        file_hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)

    payload = {
        "artifact": "objective_selection_benchmark_v2_design",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "protocol_sha256": sha256_file(root / PROTOCOL_REL),
        "legacy_phase12_v1_immutable": legacy,
        "selector_freezes": selectors,
        "summary": summary,
        "generation_intent_is_ground_truth": False,
        "evaluated_model_outputs_existed_before_design_freeze": False,
        "heldout_dataset_contents_used": False,
        "file_sha256": file_hashes,
    }
    write_json(out, payload)
    digest = sha256_file(out)
    sha_path.write_text("{}  design_manifest.json\n".format(digest), encoding="utf-8")
    return payload


def validate_manifest_with_assets(root: Path, manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    sha_path = Path(str(manifest_path) + ".sha256")
    if not manifest_path.exists() or not sha_path.exists():
        raise Phase12V2Error("Frozen manifest/checksum missing: {}".format(manifest_path))
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(manifest_path)
    if expected != actual:
        raise Phase12V2Error("Manifest checksum mismatch: {}".format(manifest_path))
    payload = read_json(manifest_path)
    for rel, expected_sha in payload.get("file_sha256", {}).items():
        asset = root / rel
        if not asset.exists() or sha256_file(asset) != expected_sha:
            raise Phase12V2Error("Frozen asset changed or disappeared: {}".format(rel))
    return payload


def validate_design_freeze(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    return validate_manifest_with_assets(
        root, _base(root) / "frozen" / "design_manifest.json"
    )


def _candidate_registry(root: Path) -> Dict[str, Dict[str, str]]:
    base = _base(root)
    generated = read_csv(base / "design" / "generated_candidates.csv")
    humans = read_csv(base / "human" / "human_candidates.csv")
    registry: Dict[str, Dict[str, str]] = {}
    for row in generated:
        item = dict(row)
        item["source"] = "generated"
        registry[row["scenario_id"].strip()] = item
    for row in humans:
        item = dict(row)
        item["source"] = "human"
        registry[row["scenario_id"].strip()] = item
    return registry


def _kept_scenario_ids(root: Path) -> List[str]:
    filter_rows = read_csv(_base(root) / "human" / "realism_filter.csv")
    kept = []
    for row in filter_rows:
        if bool(parse_bool(row.get("keep"))):
            kept.append(row["scenario_id"].strip())
    return kept


def prepare_annotation_packets(
    root: Optional[Path] = None, *, force: bool = False
) -> Dict[str, Any]:
    root = repo_root(root)
    validate_design_freeze(root)
    _ensure_no_model_outputs(root)
    cfg = load_protocol(root)
    registry = _candidate_registry(root)
    kept_ids = _kept_scenario_ids(root)
    base = _base(root) / "annotations"
    base.mkdir(parents=True, exist_ok=True)

    packet_paths = []
    for idx, letter in enumerate(("A", "B", "C")):
        path = base / "annotations_{}.csv".format(letter)
        if path.exists() and not force and path.stat().st_size > 100:
            packet_paths.append(path)
            continue
        # Distinct deterministic order prevents annotators working down an
        # identical sequence while preserving reproducibility.
        ordered = sorted(
            kept_ids,
            key=lambda sid: stable_hash("phase12-v2-{}-{}".format(letter, sid)),
        )
        rows = [
            {
                "scenario_id": sid,
                "scenario": registry[sid]["scenario"],
                "accuracy": "",
                "runtime": "",
                "energy": "",
                "co2": "",
                "ambiguous": "",
                "annotator_id": "",
                "notes": "",
            }
            for sid in ordered
        ]
        write_csv(
            path,
            rows,
            [
                "scenario_id",
                "scenario",
                "accuracy",
                "runtime",
                "energy",
                "co2",
                "ambiguous",
                "annotator_id",
                "notes",
            ],
        )
        packet_paths.append(path)

    return {
        "scenario_count": len(kept_ids),
        "annotator_count": int(cfg["annotation"]["annotator_count"]),
        "packets": [str(p.relative_to(root)).replace("\\", "/") for p in packet_paths],
        "private_generator_intent_in_packets": False,
        "source_type_in_packets": False,
    }


def _fleiss_kappa_binary(votes: Sequence[Tuple[int, int]]) -> Optional[float]:
    """Fleiss' kappa for N subjects, 3 raters, 2 categories."""
    if not votes:
        return None
    n_raters = sum(votes[0])
    if n_raters < 2 or any(sum(v) != n_raters for v in votes):
        raise Phase12V2Error("Invalid vote matrix for Fleiss kappa.")
    n_subjects = len(votes)
    p_yes = sum(v[1] for v in votes) / float(n_subjects * n_raters)
    p_no = 1.0 - p_yes
    p_e = p_yes * p_yes + p_no * p_no
    p_i = [
        (v[0] * v[0] + v[1] * v[1] - n_raters)
        / float(n_raters * (n_raters - 1))
        for v in votes
    ]
    p_bar = sum(p_i) / float(n_subjects)
    if abs(1.0 - p_e) < 1e-12:
        return None
    return (p_bar - p_e) / (1.0 - p_e)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / float(len(union))


def _mean_pairwise_jaccard(label_sets: Sequence[Set[str]]) -> float:
    pairs = list(itertools.combinations(label_sets, 2))
    return sum(_jaccard(a, b) for a, b in pairs) / float(len(pairs)) if pairs else 1.0


def aggregate_annotations(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    validate_design_freeze(root)
    manifest_path = _base(root) / "frozen" / "ground_truth_manifest.json"
    if manifest_path.exists():
        return validate_manifest_with_assets(root, manifest_path)["agreement"]
    _ensure_no_model_outputs(root)
    cfg = load_protocol(root)
    registry = _candidate_registry(root)
    expected_ids = set(_kept_scenario_ids(root))
    ann_dir = _base(root) / "annotations"
    packet_paths = [ann_dir / "annotations_{}.csv".format(x) for x in ("A", "B", "C")]

    annotator_rows: List[Dict[str, Dict[str, str]]] = []
    annotator_ids: List[str] = []
    for path in packet_paths:
        if not path.exists():
            raise Phase12V2Error("Missing completed annotation packet: {}".format(path))
        rows = read_csv(path)
        _validate_unique(rows, "scenario_id", path.name)
        row_map = {r["scenario_id"].strip(): r for r in rows}
        if set(row_map) != expected_ids:
            raise Phase12V2Error("{} does not contain exactly the annotation pool.".format(path.name))
        ids = {str(r.get("annotator_id", "")).strip() for r in rows}
        if "" in ids or len(ids) != 1:
            raise Phase12V2Error(
                "{} must contain one nonblank annotator_id on every row.".format(path.name)
            )
        annotator_id = next(iter(ids))
        annotator_ids.append(annotator_id)
        # Validate completeness now, before any aggregation.
        for row in rows:
            objective_set_from_columns(row)
            parse_bool(row.get("ambiguous"), allow_blank=False)
        annotator_rows.append(row_map)

    if len(set(annotator_ids)) != len(annotator_ids):
        raise Phase12V2Error("The three annotation packets must come from distinct annotators.")

    per_objective_votes: Dict[str, List[Tuple[int, int]]] = {obj: [] for obj in OBJECTIVES}
    resolved_rows: List[Dict[str, Any]] = []
    hard_rows: List[Dict[str, Any]] = []
    all_pairwise = []
    unanimous_full = 0

    split_threshold = int(cfg["annotation"]["hard_case_policy"]["split_objectives_threshold"])
    jaccard_threshold = float(cfg["annotation"]["hard_case_policy"]["pairwise_jaccard_below"])
    ambiguous_vote_threshold = int(cfg["annotation"]["hard_case_policy"]["ambiguous_votes_at_least"])

    for sid in sorted(expected_ids):
        rows = [packet[sid] for packet in annotator_rows]
        sets = [objective_set_from_columns(row) for row in rows]
        ambiguity_votes = sum(bool(parse_bool(row.get("ambiguous"))) for row in rows)
        pairwise_j = _mean_pairwise_jaccard(sets)
        all_pairwise.append(pairwise_j)
        if sets[0] == sets[1] == sets[2]:
            unanimous_full += 1

        majority: Set[str] = set()
        split_objectives = 0
        vote_detail: Dict[str, int] = {}
        for objective in OBJECTIVES:
            yes = sum(objective in labels for labels in sets)
            no = len(sets) - yes
            per_objective_votes[objective].append((no, yes))
            vote_detail[objective] = yes
            if yes >= 2:
                majority.add(objective)
            if yes in {1, 2}:
                split_objectives += 1

        reasons = []
        if ambiguity_votes >= ambiguous_vote_threshold:
            reasons.append("human_ambiguity_majority")
        if pairwise_j < jaccard_threshold:
            reasons.append("low_pairwise_jaccard")
        if split_objectives >= split_threshold:
            reasons.append("multiple_2_to_1_objective_splits")
        if not majority:
            reasons.append("zero_label_majority")

        item = {
            "scenario_id": sid,
            "scenario": registry[sid]["scenario"],
            "source": registry[sid]["source"],
            "domain": registry[sid].get("domain", ""),
            "style": registry[sid].get("style", ""),
            "ground_truth": labels_to_pipe(majority),
            "k_prime": len(majority),
            "pairwise_jaccard": pairwise_j,
            "ambiguous_votes": ambiguity_votes,
            "split_objectives": split_objectives,
            "vote_accuracy": vote_detail["Accuracy"],
            "vote_runtime": vote_detail["Runtime"],
            "vote_energy": vote_detail["Energy"],
            "vote_co2": vote_detail["CO2"],
            "hard_case": bool(reasons),
            "hard_reasons": "|".join(reasons),
        }
        resolved_rows.append(item)
        if reasons:
            hard_rows.append(item)

    kappas = {
        objective: _fleiss_kappa_binary(per_objective_votes[objective])
        for objective in OBJECTIVES
    }
    defined = [x for x in kappas.values() if x is not None]
    agreement = {
        "annotator_count": 3,
        "annotator_ids": annotator_ids,
        "scenario_count": len(expected_ids),
        "per_objective_fleiss_kappa": kappas,
        "mean_defined_fleiss_kappa": sum(defined) / len(defined) if defined else None,
        "mean_pairwise_jaccard": sum(all_pairwise) / len(all_pairwise) if all_pairwise else None,
        "unanimous_full_set_agreement_rate": unanimous_full / float(len(expected_ids)),
        "hard_cases": len(hard_rows),
        "eligible_primary_cases": len(resolved_rows) - len(hard_rows),
        "hard_case_policy": cfg["annotation"]["hard_case_policy"],
        "ground_truth_source": "independent_human_majority_vote_only",
        "generation_intent_used_as_ground_truth": False,
    }

    results_dir = _base(root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    gt_path = results_dir / "ground_truth_all.csv"
    hard_path = results_dir / "hard_cases.csv"
    agreement_path = results_dir / "annotation_agreement.json"
    fields = [
        "scenario_id", "scenario", "source", "domain", "style", "ground_truth", "k_prime",
        "pairwise_jaccard", "ambiguous_votes", "split_objectives",
        "vote_accuracy", "vote_runtime", "vote_energy", "vote_co2", "hard_case", "hard_reasons",
    ]
    write_csv(gt_path, resolved_rows, fields)
    write_csv(hard_path, hard_rows, fields)
    write_json(agreement_path, agreement)

    protected = packet_paths + [gt_path, hard_path, agreement_path]
    file_hashes = {
        str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in protected
    }
    manifest = {
        "artifact": "objective_selection_benchmark_v2_ground_truth",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "design_manifest_sha256": sha256_file(_base(root) / "frozen" / "design_manifest.json"),
        "agreement": agreement,
        "file_sha256": file_hashes,
        "llm_outputs_existed_before_ground_truth_freeze": False,
        "generation_intent_used_as_ground_truth": False,
    }
    write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    Path(str(manifest_path) + ".sha256").write_text(
        "{}  ground_truth_manifest.json\n".format(digest), encoding="utf-8"
    )
    return agreement


def validate_ground_truth_freeze(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    validate_design_freeze(root)
    return validate_manifest_with_assets(
        root, _base(root) / "frozen" / "ground_truth_manifest.json"
    )


def _allocate_human_quotas(
    groups: Mapping[int, Sequence[Mapping[str, Any]]], minimum_total: int
) -> Dict[int, int]:
    # Start near-evenly, then redistribute deficits to strata with spare human cases.
    base = minimum_total // 4
    quotas = {
        k: min(base, sum(str(r.get("source")) == "human" for r in groups[k]))
        for k in (1, 2, 3, 4)
    }
    remaining = minimum_total - sum(quotas.values())
    while remaining > 0:
        candidates = []
        for k in (1, 2, 3, 4):
            available = sum(str(r.get("source")) == "human" for r in groups[k])
            spare = min(15, available) - quotas[k]
            if spare > 0:
                candidates.append((spare, -quotas[k], -k, k))
        if not candidates:
            raise Phase12V2Error(
                "Not enough human-written eligible cases to satisfy the final human quota."
            )
        candidates.sort(reverse=True)
        chosen_k = candidates[0][3]
        quotas[chosen_k] += 1
        remaining -= 1
    return quotas


def _diversity_pick(
    rows: Sequence[Mapping[str, Any]],
    n: int,
    *,
    human_quota: int,
    salt: str,
) -> List[Dict[str, Any]]:
    pool = [dict(r) for r in rows]
    selected: List[Dict[str, Any]] = []
    domain_freq = Counter(str(r.get("domain", "")).strip().lower() or "unknown" for r in pool)
    style_freq = Counter(str(r.get("style", "")).strip().lower() or "unknown" for r in pool)
    combo_freq = Counter(str(r.get("ground_truth", "")) for r in pool)

    def score(row: Mapping[str, Any]) -> Tuple[float, str]:
        domain = str(row.get("domain", "")).strip().lower() or "unknown"
        style = str(row.get("style", "")).strip().lower() or "unknown"
        combo = str(row.get("ground_truth", ""))
        used_domains = Counter(str(x.get("domain", "")).strip().lower() or "unknown" for x in selected)
        used_styles = Counter(str(x.get("style", "")).strip().lower() or "unknown" for x in selected)
        used_combos = Counter(str(x.get("ground_truth", "")) for x in selected)
        value = 0.0
        value += 4.0 if not used_combos[combo] else 1.0 / (1.0 + used_combos[combo])
        value += 3.0 if not used_domains[domain] else 0.5 / (1.0 + used_domains[domain])
        value += 2.0 if not used_styles[style] else 0.5 / (1.0 + used_styles[style])
        value += 1.0 / float(max(1, combo_freq[combo]))
        value += 0.75 / float(max(1, domain_freq[domain]))
        value += 0.50 / float(max(1, style_freq[style]))
        # Hash is only a deterministic tie breaker; it carries no model information.
        tie = stable_hash("{}:{}".format(salt, row.get("scenario_id", "")))
        return value, tie

    def choose_from(candidates: List[Dict[str, Any]]) -> None:
        if not candidates:
            raise Phase12V2Error("Diversity selection ran out of eligible cases.")
        ranked = sorted(candidates, key=lambda r: (score(r)[0], score(r)[1]), reverse=True)
        chosen = ranked[0]
        selected.append(chosen)
        pool.remove(chosen)

    for _ in range(human_quota):
        choose_from([r for r in pool if str(r.get("source")) == "human"])
    while len(selected) < n:
        choose_from(pool)
    return selected


def freeze_primary_benchmark(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    validate_ground_truth_freeze(root)
    manifest_path = _base(root) / "frozen" / "primary_manifest.json"
    if manifest_path.exists():
        return validate_manifest_with_assets(root, manifest_path)
    _ensure_no_model_outputs(root)
    cfg = load_protocol(root)
    base = _base(root)
    gt_rows = read_csv(base / "results" / "ground_truth_all.csv")
    eligible = [
        row
        for row in gt_rows
        if not bool(parse_bool(row.get("hard_case"))) and int(row.get("k_prime", "0")) in {1, 2, 3, 4}
    ]
    groups: Dict[int, List[Dict[str, str]]] = {
        k: [row for row in eligible if int(row["k_prime"]) == k] for k in (1, 2, 3, 4)
    }
    target_by_k = {
        int(k): int(v) for k, v in cfg["primary"]["target_k_prime_counts"].items()
    }
    for k, target in target_by_k.items():
        if len(groups[k]) < target:
            raise Phase12V2Error(
                "Only {} eligible human-ground-truth k'={} cases exist; {} are required. "
                "Do not substitute private generator intent. Collect/annotate more cases.".format(
                    len(groups[k]), k, target
                )
            )

    minimum_human = int(cfg["primary"]["minimum_human_written"])
    quotas = _allocate_human_quotas(groups, minimum_human)
    selected: List[Dict[str, Any]] = []
    for k in (1, 2, 3, 4):
        selected.extend(
            _diversity_pick(
                groups[k],
                target_by_k[k],
                human_quota=quotas[k],
                salt="phase12-v2-k{}".format(k),
            )
        )

    selected = sorted(selected, key=lambda r: (int(r["k_prime"]), stable_hash(r["scenario_id"])))
    human_count = sum(str(row.get("source")) == "human" for row in selected)
    if human_count < minimum_human:
        raise Phase12V2Error("Final benchmark failed minimum human-written requirement.")
    distinct_sets = len({str(row.get("ground_truth", "")) for row in selected})
    required_distinct = int(cfg["paraphrase"]["families"])
    if distinct_sets < required_distinct:
        raise Phase12V2Error(
            "Final 60 contain only {} distinct objective sets; at least {} are required for the pre-registered distinct-set paraphrase test. Collect/annotate more diverse cases rather than using model outputs to choose replacements.".format(
                distinct_sets, required_distinct
            )
        )

    out_rows = []
    for idx, row in enumerate(selected, 1):
        out_rows.append(
            {
                "benchmark_id": "V2-{:03d}".format(idx),
                "scenario_id": row["scenario_id"],
                "scenario": row["scenario"],
                "ground_truth": row["ground_truth"],
                "k_prime": int(row["k_prime"]),
                "source": row["source"],
                "domain": row.get("domain", ""),
                "style": row.get("style", ""),
                "pairwise_jaccard": row.get("pairwise_jaccard", ""),
                "vote_accuracy": row.get("vote_accuracy", ""),
                "vote_runtime": row.get("vote_runtime", ""),
                "vote_energy": row.get("vote_energy", ""),
                "vote_co2": row.get("vote_co2", ""),
            }
        )

    primary_path = base / "frozen" / "primary_benchmark.csv"
    write_csv(
        primary_path,
        out_rows,
        [
            "benchmark_id", "scenario_id", "scenario", "ground_truth", "k_prime", "source",
            "domain", "style", "pairwise_jaccard", "vote_accuracy", "vote_runtime", "vote_energy", "vote_co2",
        ],
    )

    summary = {
        "n": len(out_rows),
        "k_prime_counts": dict(Counter(str(r["k_prime"]) for r in out_rows)),
        "human_written": human_count,
        "human_written_fraction": human_count / float(len(out_rows)),
        "human_quota_by_k_prime": {str(k): quotas[k] for k in sorted(quotas)},
        "objective_set_counts": dict(Counter(r["ground_truth"] for r in out_rows)),
        "distinct_objective_sets": distinct_sets,
        "source_counts": dict(Counter(r["source"] for r in out_rows)),
        "domain_counts": dict(Counter(r["domain"] for r in out_rows)),
        "selection_used_model_outputs": False,
        "selection_used_generator_intent": False,
    }
    protected = [
        primary_path,
        base / "frozen" / "ground_truth_manifest.json",
        root / V32_MANIFEST_REL,
        root / LEGACY_V2_SELECTOR_REL,
    ]
    file_hashes = {
        str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in protected
    }
    manifest = {
        "artifact": "objective_selection_benchmark_v2_primary",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "summary": summary,
        "ground_truth_source": "independent_human_majority_vote_only",
        "generator_intent_used_as_ground_truth": False,
        "evaluated_selector_outputs_existed_before_primary_freeze": False,
        "selection_algorithm": "deterministic_diversity_greedy_without_model_outputs",
        "file_sha256": file_hashes,
    }
    write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    Path(str(manifest_path) + ".sha256").write_text(
        "{}  primary_manifest.json\n".format(digest), encoding="utf-8"
    )
    return manifest


def validate_primary_freeze(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    validate_ground_truth_freeze(root)
    return validate_manifest_with_assets(
        root, _base(root) / "frozen" / "primary_manifest.json"
    )


def prepare_paraphrase_bases(root: Optional[Path] = None, *, force: bool = False) -> Dict[str, Any]:
    root = repo_root(root)
    validate_primary_freeze(root)
    _ensure_no_model_outputs(root)
    cfg = load_protocol(root)
    base = _base(root)
    primary = read_csv(base / "frozen" / "primary_benchmark.csv")
    target = int(cfg["paraphrase"]["families"])
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in primary:
        groups[row["ground_truth"]].append(row)
    if len(groups) < target:
        raise Phase12V2Error(
            "Final primary benchmark has only {} distinct objective sets; {} are required for distinct paraphrase families.".format(
                len(groups), target
            )
        )

    # Prefer objective sets with fewer available examples first, then use a stable
    # hash. This makes the paraphrase subset diverse without looking at predictions.
    chosen_sets = sorted(
        groups,
        key=lambda gt: (len(groups[gt]), stable_hash("para-set:" + gt)),
    )[:target]
    bases = []
    for idx, gt in enumerate(chosen_sets, 1):
        rows = sorted(
            groups[gt],
            key=lambda r: stable_hash("para-base:" + r["scenario_id"]),
        )
        row = rows[0]
        bases.append(
            {
                "family_id": "P{:02d}".format(idx),
                "benchmark_id": row["benchmark_id"],
                "scenario_id": row["scenario_id"],
                "base_scenario": row["scenario"],
                "ground_truth": row["ground_truth"],
                "k_prime": row["k_prime"],
            }
        )

    para_dir = base / "paraphrases"
    base_path = para_dir / "base_scenarios.csv"
    candidates_path = para_dir / "paraphrase_candidates.csv"
    if base_path.exists() and not force:
        existing = read_csv(base_path)
        if existing:
            return {"families": len(existing), "path": str(base_path.relative_to(root))}

    write_csv(
        base_path,
        bases,
        ["family_id", "benchmark_id", "scenario_id", "base_scenario", "ground_truth", "k_prime"],
    )
    candidate_rows = []
    variants = int(cfg["paraphrase"]["paraphrases_per_family"])
    for row in bases:
        for j in range(1, variants + 1):
            candidate_rows.append(
                {
                    "family_id": row["family_id"],
                    "variant_id": "PARA{}".format(j),
                    "scenario": "",
                    "generator_model": "",
                    "generation_batch": "",
                }
            )
    write_csv(
        candidates_path,
        candidate_rows,
        ["family_id", "variant_id", "scenario", "generator_model", "generation_batch"],
    )
    return {
        "families": len(bases),
        "distinct_ground_truth_sets": len(set(r["ground_truth"] for r in bases)),
        "paraphrase_rows_to_generate": len(candidate_rows),
        "base_path": str(base_path.relative_to(root)).replace("\\", "/"),
        "candidate_path": str(candidates_path.relative_to(root)).replace("\\", "/"),
    }


def prepare_paraphrase_reviews(root: Optional[Path] = None, *, force: bool = False) -> Dict[str, Any]:
    root = repo_root(root)
    validate_primary_freeze(root)
    _ensure_no_model_outputs(root)
    cfg = load_protocol(root)
    base = _base(root)
    bases = read_csv(base / "paraphrases" / "base_scenarios.csv")
    candidates = read_csv(base / "paraphrases" / "paraphrase_candidates.csv")
    expected = int(cfg["paraphrase"]["families"]) * int(cfg["paraphrase"]["paraphrases_per_family"])
    if len(candidates) != expected:
        raise Phase12V2Error("Expected {} paraphrase candidates, found {}.".format(expected, len(candidates)))
    base_map = {row["family_id"]: row for row in bases}
    if len(base_map) != int(cfg["paraphrase"]["families"]):
        raise Phase12V2Error("Paraphrase base family count is incorrect.")

    keys = set()
    for row in candidates:
        key = (row["family_id"].strip(), row["variant_id"].strip())
        if key in keys:
            raise Phase12V2Error("Duplicate paraphrase variant {}.".format(key))
        keys.add(key)
        if key[0] not in base_map:
            raise Phase12V2Error("Unknown paraphrase family {}.".format(key[0]))
        scenario = str(row.get("scenario", "")).strip()
        if not scenario:
            raise Phase12V2Error("Paraphrase {} has not been generated yet.".format(key))
        match = EXPLICIT_OBJECTIVE_RE.search(scenario)
        if match:
            raise Phase12V2Error(
                "Paraphrase {} explicitly names frozen objective {!r}.".format(key, match.group(0))
            )
        model = str(row.get("generator_model", "")).strip()
        if not model or model.lower() == "llama3:8b":
            raise Phase12V2Error("Every paraphrase must record an outside generator model.")

    para_dir = base / "paraphrases"
    packet_paths = []
    for letter in ("A", "B"):
        path = para_dir / "review_{}.csv".format(letter)
        if path.exists() and not force and path.stat().st_size > 100:
            packet_paths.append(path)
            continue
        ordered = sorted(
            candidates,
            key=lambda r: stable_hash(
                "para-review-{}:{}:{}".format(letter, r["family_id"], r["variant_id"])
            ),
        )
        rows = [
            {
                "family_id": row["family_id"],
                "variant_id": row["variant_id"],
                "scenario": row["scenario"],
                "meaning_preserved": "",
                "accuracy": "",
                "runtime": "",
                "energy": "",
                "co2": "",
                "reviewer_id": "",
                "notes": "",
            }
            for row in ordered
        ]
        write_csv(
            path,
            rows,
            [
                "family_id", "variant_id", "scenario", "meaning_preserved",
                "accuracy", "runtime", "energy", "co2", "reviewer_id", "notes",
            ],
        )
        packet_paths.append(path)
    return {
        "paraphrase_evaluations": len(candidates),
        "review_packets": [str(p.relative_to(root)).replace("\\", "/") for p in packet_paths],
        "reviewers_required": 2,
    }


def freeze_paraphrase_set(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    validate_primary_freeze(root)
    manifest_path = _base(root) / "frozen" / "paraphrase_manifest.json"
    if manifest_path.exists():
        return validate_manifest_with_assets(root, manifest_path)
    _ensure_no_model_outputs(root)
    cfg = load_protocol(root)
    base = _base(root)
    para_dir = base / "paraphrases"
    bases = read_csv(para_dir / "base_scenarios.csv")
    candidates = read_csv(para_dir / "paraphrase_candidates.csv")
    base_map = {row["family_id"]: row for row in bases}
    candidate_map = {
        (row["family_id"], row["variant_id"]): row for row in candidates
    }

    reviews = []
    reviewer_ids = []
    for letter in ("A", "B"):
        path = para_dir / "review_{}.csv".format(letter)
        if not path.exists():
            raise Phase12V2Error("Missing paraphrase review packet {}.".format(path.name))
        rows = read_csv(path)
        row_map = {(r["family_id"], r["variant_id"]): r for r in rows}
        if set(row_map) != set(candidate_map):
            raise Phase12V2Error("{} does not match the paraphrase candidate set.".format(path.name))
        ids = {str(r.get("reviewer_id", "")).strip() for r in rows}
        if "" in ids or len(ids) != 1:
            raise Phase12V2Error("{} needs one nonblank reviewer_id.".format(path.name))
        reviewer_ids.append(next(iter(ids)))
        for r in rows:
            parse_bool(r.get("meaning_preserved"), allow_blank=False)
            objective_set_from_columns(r)
        reviews.append(row_map)
    if len(set(reviewer_ids)) != 2:
        raise Phase12V2Error("Paraphrase semantics must be reviewed by two distinct people.")

    accepted = []
    rejected = []
    for key in sorted(candidate_map):
        family_id, variant_id = key
        expected_labels = set(labels_from_pipe(base_map[family_id]["ground_truth"]))
        r1, r2 = reviews[0][key], reviews[1][key]
        preserved = [bool(parse_bool(r.get("meaning_preserved"))) for r in (r1, r2)]
        label_sets = [objective_set_from_columns(r) for r in (r1, r2)]
        reasons = []
        if not all(preserved):
            reasons.append("meaning_not_unanimously_preserved")
        if not all(labels == expected_labels for labels in label_sets):
            reasons.append("reviewed_objective_set_differs_from_base_ground_truth")
        row = {
            "family_id": family_id,
            "variant_id": variant_id,
            "scenario": candidate_map[key]["scenario"],
            "ground_truth": labels_to_pipe(expected_labels),
            "k_prime": len(expected_labels),
            "accepted": not reasons,
            "reasons": "|".join(reasons),
        }
        (accepted if not reasons else rejected).append(row)

    accepted_path = base / "frozen" / "paraphrase_set.csv"
    review_path = base / "results" / "paraphrase_review_summary.json"
    write_csv(
        accepted_path,
        accepted,
        ["family_id", "variant_id", "scenario", "ground_truth", "k_prime", "accepted", "reasons"],
    )
    summary = {
        "families": len(bases),
        "candidate_variants": len(candidates),
        "accepted_variants": len(accepted),
        "excluded_variants": len(rejected),
        "reviewer_ids": reviewer_ids,
        "acceptance_rule": "both reviewers preserve meaning AND both independently recover the base human ground-truth set",
        "excluded": rejected,
    }
    write_json(review_path, summary)
    protected = [
        para_dir / "base_scenarios.csv",
        para_dir / "paraphrase_candidates.csv",
        para_dir / "review_A.csv",
        para_dir / "review_B.csv",
        accepted_path,
        review_path,
    ]
    manifest = {
        "artifact": "objective_selection_benchmark_v2_paraphrases",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "summary": summary,
        "file_sha256": {
            str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in protected
        },
        "evaluated_selector_outputs_existed_before_freeze": False,
    }
    write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    Path(str(manifest_path) + ".sha256").write_text(
        "{}  paraphrase_manifest.json\n".format(digest), encoding="utf-8"
    )
    return manifest


def validate_paraphrase_freeze(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    return validate_manifest_with_assets(
        root, _base(root) / "frozen" / "paraphrase_manifest.json"
    )


def freeze_adversarial_set(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    validate_primary_freeze(root)
    manifest_path = _base(root) / "frozen" / "adversarial_manifest.json"
    if manifest_path.exists():
        return validate_manifest_with_assets(root, manifest_path)
    _ensure_no_model_outputs(root)
    cfg = load_protocol(root)
    base = _base(root)
    path = base / "adversarial" / "adversarial_cases.csv"
    if not path.exists():
        raise Phase12V2Error("Missing fresh adversarial_cases.csv.")
    rows = read_csv(path)
    min_n = int(cfg["adversarial"]["minimum_cases"])
    max_n = int(cfg["adversarial"]["maximum_cases"])
    if not min_n <= len(rows) <= max_n:
        raise Phase12V2Error("Adversarial set must contain {}-{} cases.".format(min_n, max_n))
    _validate_unique(rows, "case_id", "adversarial_cases.csv")
    normalized = set()
    for row in rows:
        scenario = str(row.get("scenario", "")).strip()
        expected = str(row.get("expected_status", "")).strip()
        category = str(row.get("category", "")).strip()
        if not scenario or not category:
            raise Phase12V2Error("Every adversarial row needs scenario and category.")
        if expected not in REJECTION_STATUSES:
            raise Phase12V2Error(
                "Adversarial expected_status must be ambiguous, contradictory, or out_of_scope."
            )
        key = re.sub(r"\s+", " ", scenario.lower())
        if key in normalized:
            raise Phase12V2Error("Duplicate adversarial scenario.")
        normalized.add(key)

    # Exact reuse of legacy v1 adversarial cases is forbidden.
    legacy_path = root / "data" / "journal" / "objective_selection_benchmark_v1" / "design" / "adversarial_set.csv"
    if legacy_path.exists():
        legacy_text = {
            re.sub(r"\s+", " ", str(r.get("scenario", "")).strip().lower())
            for r in read_csv(legacy_path)
        }
        overlap = sorted(normalized & legacy_text)
        if overlap:
            raise Phase12V2Error("Fresh adversarial set reuses legacy Phase-12-v1 cases.")

    frozen_path = base / "frozen" / "adversarial_set.csv"
    shutil.copyfile(path, frozen_path)
    summary = {
        "n": len(rows),
        "expected_status_counts": dict(Counter(r["expected_status"] for r in rows)),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "expected_status_frozen_before_evaluation": True,
    }
    manifest = {
        "artifact": "objective_selection_benchmark_v2_adversarial",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "summary": summary,
        "file_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path),
            str(frozen_path.relative_to(root)).replace("\\", "/"): sha256_file(frozen_path),
        },
        "evaluated_selector_outputs_existed_before_freeze": False,
    }
    write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    Path(str(manifest_path) + ".sha256").write_text(
        "{}  adversarial_manifest.json\n".format(digest), encoding="utf-8"
    )
    return manifest


def validate_adversarial_freeze(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    return validate_manifest_with_assets(
        root, _base(root) / "frozen" / "adversarial_manifest.json"
    )


def validate_all_evaluation_inputs_frozen(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    return {
        "primary": validate_primary_freeze(root),
        "paraphrase": validate_paraphrase_freeze(root),
        "adversarial": validate_adversarial_freeze(root),
        "legacy": verify_legacy_v1_immutable(root),
        "selectors": verify_selector_freezes(root),
    }


def _make_selector(root: Path, selector_name: str):
    from awareml.llm.confirmatory_runtime_v32 import ConfirmatoryOllamaClientV32
    from awareml.llm.objective_selection import JournalObjectiveSelector
    from awareml.llm.objective_selection_v32 import EvidenceGroundedObjectiveSelectorV32

    client = ConfirmatoryOllamaClientV32(root=root)
    runtime = client.verify_runtime()
    if selector_name == "baseline":
        return JournalObjectiveSelector(client=client, root=root), client, runtime
    if selector_name == "v32":
        return (
            EvidenceGroundedObjectiveSelectorV32(
                client=client,
                root=root,
                benchmark_mode=True,
                allow_semantic_recovery=False,
            ),
            client,
            runtime,
        )
    raise Phase12V2Error("selector_name must be baseline or v32")


def _run_inference_rows(
    root: Path,
    *,
    selector_name: str,
    rows: Sequence[Mapping[str, Any]],
    id_fields: Sequence[str],
    scenario_field: str,
    output_path: Path,
    metadata_path: Path,
) -> Dict[str, Any]:
    selector, client, runtime = _make_selector(root, selector_name)
    outputs = []
    for index, row in enumerate(rows, 1):
        scenario = str(row.get(scenario_field, "")).strip()
        result = selector.select(scenario)
        item: Dict[str, Any] = {
            field: row.get(field) for field in id_fields
        }
        item.update(
            {
                "scenario": scenario,
                "status": result.status,
                "selected_objectives": list(result.selected_objectives),
                "uncertainties": list(result.uncertainties),
                "source": result.source,
                "model": result.model,
                "fallback_used": bool(result.fallback_used),
                "row_index": index,
            }
        )
        if selector_name == "v32":
            item["audit"] = dict(getattr(selector, "last_audit", {}) or {})
        outputs.append(item)

    write_jsonl(output_path, outputs)
    metadata = {
        "artifact": "phase12_v2_{}_run".format(selector_name),
        "created_at_utc": utc_now(),
        "selector": selector_name,
        "selector_reporting_name": (
            "Phase-11 V2 method replay under Phase-11R confirmatory runtime"
            if selector_name == "baseline"
            else "Objective Selection V3.2"
        ),
        "rows": len(outputs),
        "runtime": runtime,
        "required_model": getattr(client, "model", None),
        "fallback_count": sum(bool(row["fallback_used"]) for row in outputs),
        "status_counts": dict(Counter(row["status"] for row in outputs)),
        "input_sha256": None,
        "output_sha256": sha256_file(output_path),
    }
    write_json(metadata_path, metadata)
    return metadata


def run_primary(root: Optional[Path] = None, *, selectors: Sequence[str] = ("baseline", "v32")) -> Dict[str, Any]:
    root = repo_root(root)
    validate_all_evaluation_inputs_frozen(root)
    base = _base(root)
    primary_path = base / "frozen" / "primary_benchmark.csv"
    rows = read_csv(primary_path)
    results = {}
    for selector_name in selectors:
        output = base / "results" / "{}_primary_outputs.jsonl".format(selector_name)
        metadata = base / "results" / "{}_primary_run_metadata.json".format(selector_name)
        if output.exists():
            raise Phase12V2Error(
                "{} already exists. Confirmatory outputs are immutable; do not silently rerun.".format(output)
            )
        info = _run_inference_rows(
            root,
            selector_name=selector_name,
            rows=rows,
            id_fields=("benchmark_id", "scenario_id"),
            scenario_field="scenario",
            output_path=output,
            metadata_path=metadata,
        )
        info["input_sha256"] = sha256_file(primary_path)
        write_json(metadata, info)
        results[selector_name] = info
    return results


def run_paraphrases(root: Optional[Path] = None, *, selectors: Sequence[str] = ("baseline", "v32")) -> Dict[str, Any]:
    root = repo_root(root)
    validate_all_evaluation_inputs_frozen(root)
    base = _base(root)
    input_path = base / "frozen" / "paraphrase_set.csv"
    rows = read_csv(input_path)
    results = {}
    for selector_name in selectors:
        output = base / "results" / "{}_paraphrase_outputs.jsonl".format(selector_name)
        metadata = base / "results" / "{}_paraphrase_run_metadata.json".format(selector_name)
        if output.exists():
            raise Phase12V2Error("{} already exists; confirmatory outputs are immutable.".format(output))
        info = _run_inference_rows(
            root,
            selector_name=selector_name,
            rows=rows,
            id_fields=("family_id", "variant_id"),
            scenario_field="scenario",
            output_path=output,
            metadata_path=metadata,
        )
        info["input_sha256"] = sha256_file(input_path)
        write_json(metadata, info)
        results[selector_name] = info
    return results


def run_adversarial(root: Optional[Path] = None, *, selectors: Sequence[str] = ("baseline", "v32")) -> Dict[str, Any]:
    root = repo_root(root)
    validate_all_evaluation_inputs_frozen(root)
    base = _base(root)
    input_path = base / "frozen" / "adversarial_set.csv"
    rows = read_csv(input_path)
    results = {}
    for selector_name in selectors:
        output = base / "results" / "{}_adversarial_outputs.jsonl".format(selector_name)
        metadata = base / "results" / "{}_adversarial_run_metadata.json".format(selector_name)
        if output.exists():
            raise Phase12V2Error("{} already exists; confirmatory outputs are immutable.".format(output))
        info = _run_inference_rows(
            root,
            selector_name=selector_name,
            rows=rows,
            id_fields=("case_id",),
            scenario_field="scenario",
            output_path=output,
            metadata_path=metadata,
        )
        info["input_sha256"] = sha256_file(input_path)
        write_json(metadata, info)
        results[selector_name] = info
    return results


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _set_metrics(truths: Sequence[Set[str]], preds: Sequence[Set[str]], statuses: Sequence[str]) -> Dict[str, Any]:
    if len(truths) != len(preds) or len(truths) != len(statuses):
        raise Phase12V2Error("Metric inputs have inconsistent lengths.")
    n = len(truths)
    if n == 0:
        raise Phase12V2Error("Cannot score an empty benchmark.")

    per_objective: Dict[str, Dict[str, Any]] = {}
    total_tp = total_fp = total_fn = total_tn = 0
    f1s = []
    for objective in OBJECTIVES:
        tp = sum(objective in t and objective in p for t, p in zip(truths, preds))
        fp = sum(objective not in t and objective in p for t, p in zip(truths, preds))
        fn = sum(objective in t and objective not in p for t, p in zip(truths, preds))
        tn = n - tp - fp - fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        per_objective[objective] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    exact = [t == p for t, p in zip(truths, preds)]
    jaccards = [_jaccard(t, p) for t, p in zip(truths, preds)]
    any_fp = [bool(p - t) for t, p in zip(truths, preds)]
    any_fn = [bool(t - p) for t, p in zip(truths, preds)]
    strict_superset = [bool(p > t) for t, p in zip(truths, preds)]
    strict_subset = [bool(p < t) for t, p in zip(truths, preds)]
    hamming_errors = sum(len(t ^ p) for t, p in zip(truths, preds))

    return {
        "n": n,
        "per_objective": per_objective,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": sum(f1s) / len(f1s),
        "exact_match_rate": sum(exact) / float(n),
        "mean_jaccard": sum(jaccards) / float(n),
        "hamming_loss": hamming_errors / float(n * len(OBJECTIVES)),
        "over_selection_any_fp_rate": sum(any_fp) / float(n),
        "under_selection_any_fn_rate": sum(any_fn) / float(n),
        "strict_superset_rate": sum(strict_superset) / float(n),
        "strict_subset_rate": sum(strict_subset) / float(n),
        "valid_status_rate": sum(s == "valid" for s in statuses) / float(n),
        "malformed_rate": sum(s == "malformed" for s in statuses) / float(n),
        "status_counts": dict(Counter(statuses)),
    }


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _bootstrap_cis(
    truths: Sequence[Set[str]],
    preds: Sequence[Set[str]],
    statuses: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    n = len(truths)
    keys = (
        "exact_match_rate", "micro_f1", "macro_f1", "mean_jaccard", "hamming_loss",
        "over_selection_any_fp_rate", "under_selection_any_fn_rate",
    )
    samples: Dict[str, List[float]] = {key: [] for key in keys}
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        metric = _set_metrics(
            [truths[i] for i in idx],
            [preds[i] for i in idx],
            [statuses[i] for i in idx],
        )
        for key in keys:
            samples[key].append(float(metric[key]))
    return {
        key: [_percentile(values, 0.025), _percentile(values, 0.975)]
        for key, values in samples.items()
    }


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def _load_primary_pairs(root: Path, selector_name: str):
    base = _base(root)
    primary = read_csv(base / "frozen" / "primary_benchmark.csv")
    outputs = read_jsonl(base / "results" / "{}_primary_outputs.jsonl".format(selector_name))
    out_map = {row["scenario_id"]: row for row in outputs}
    if set(out_map) != {row["scenario_id"] for row in primary}:
        raise Phase12V2Error("{} primary output IDs do not match frozen input.".format(selector_name))
    truths, preds, statuses, kprimes = [], [], [], []
    for row in primary:
        out = out_map[row["scenario_id"]]
        truths.append(set(labels_from_pipe(row["ground_truth"])))
        preds.append(set(out.get("selected_objectives") or []))
        statuses.append(str(out.get("status", "")))
        kprimes.append(int(row["k_prime"]))
    return primary, truths, preds, statuses, kprimes


def score_primary(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    final_path = _base(root) / "frozen" / "manifest.json"
    if final_path.exists():
        final = validate_manifest_with_assets(root, final_path)
        return {"baseline": final["baseline_primary_metrics"], "v32": final["v32_primary_metrics"], "comparison": final["paired_primary_comparison"]}
    validate_all_evaluation_inputs_frozen(root)
    cfg = load_protocol(root)
    base = _base(root)
    selector_metrics = {}
    exact_vectors = {}
    for selector_name in ("baseline", "v32"):
        primary, truths, preds, statuses, kprimes = _load_primary_pairs(root, selector_name)
        metrics = _set_metrics(truths, preds, statuses)
        metrics["selector"] = selector_name
        metrics["ground_truth_source"] = "independent_human_majority_vote_only"
        metrics["by_k_prime"] = {}
        for k in (1, 2, 3, 4):
            idx = [i for i, value in enumerate(kprimes) if value == k]
            part = _set_metrics(
                [truths[i] for i in idx], [preds[i] for i in idx], [statuses[i] for i in idx]
            )
            metrics["by_k_prime"][str(k)] = {
                key: part[key]
                for key in (
                    "n", "exact_match_rate", "micro_precision", "micro_recall", "micro_f1",
                    "macro_f1", "mean_jaccard", "hamming_loss",
                    "over_selection_any_fp_rate", "under_selection_any_fn_rate",
                )
            }
        metrics["bootstrap_95_ci"] = _bootstrap_cis(
            truths,
            preds,
            statuses,
            iterations=int(cfg["metrics"]["bootstrap_iterations"]),
            seed=int(cfg["metrics"]["bootstrap_seed"]),
        )
        metrics["fallback_used_count"] = sum(
            bool(row.get("fallback_used"))
            for row in read_jsonl(base / "results" / "{}_primary_outputs.jsonl".format(selector_name))
        )
        metrics_path = base / "results" / "{}_primary_metrics.json".format(selector_name)
        write_json(metrics_path, metrics)
        selector_metrics[selector_name] = metrics
        exact_vectors[selector_name] = [t == p for t, p in zip(truths, preds)]

    b = sum(a and not v for a, v in zip(exact_vectors["baseline"], exact_vectors["v32"]))
    c = sum((not a) and v for a, v in zip(exact_vectors["baseline"], exact_vectors["v32"]))
    comparison = {
        "artifact": "phase12_v2_paired_primary_comparison",
        "n": len(exact_vectors["baseline"]),
        "same_fresh_benchmark": True,
        "baseline_selector": "frozen Phase-11 V2 LLaMA selector",
        "candidate_selector": "frozen V3.2 evidence-grounded selector",
        "deltas_v32_minus_baseline": {
            key: selector_metrics["v32"][key] - selector_metrics["baseline"][key]
            for key in (
                "exact_match_rate", "micro_precision", "micro_recall", "micro_f1",
                "macro_f1", "mean_jaccard", "hamming_loss",
                "over_selection_any_fp_rate", "under_selection_any_fn_rate",
            )
        },
        "mcnemar_exact": {
            "baseline_correct_v32_wrong": b,
            "baseline_wrong_v32_correct": c,
            "two_sided_exact_p": _mcnemar_exact_p(b, c),
        },
    }
    write_json(base / "results" / "paired_primary_comparison.json", comparison)
    return {"baseline": selector_metrics["baseline"], "v32": selector_metrics["v32"], "comparison": comparison}


def score_paraphrases(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    final_path = _base(root) / "frozen" / "manifest.json"
    if final_path.exists():
        final = validate_manifest_with_assets(root, final_path)
        return {"baseline": final["baseline_paraphrase_metrics"], "v32": final["v32_paraphrase_metrics"]}
    validate_all_evaluation_inputs_frozen(root)
    base = _base(root)
    variants = read_csv(base / "frozen" / "paraphrase_set.csv")
    primary = read_csv(base / "frozen" / "primary_benchmark.csv")
    bases = read_csv(base / "paraphrases" / "base_scenarios.csv")
    family_base_sid = {row["family_id"]: row["scenario_id"] for row in bases}
    results = {}

    for selector_name in ("baseline", "v32"):
        variant_out = {
            (row["family_id"], row["variant_id"]): row
            for row in read_jsonl(base / "results" / "{}_paraphrase_outputs.jsonl".format(selector_name))
        }
        primary_out = {
            row["scenario_id"]: row
            for row in read_jsonl(base / "results" / "{}_primary_outputs.jsonl".format(selector_name))
        }
        exact = []
        consistent = []
        set_change_from_base = []
        family_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in variants:
            key = (row["family_id"], row["variant_id"])
            out = variant_out[key]
            truth = set(labels_from_pipe(row["ground_truth"]))
            pred = set(out.get("selected_objectives") or [])
            base_pred = set(primary_out[family_base_sid[row["family_id"]]].get("selected_objectives") or [])
            is_exact = pred == truth
            is_consistent = pred == base_pred
            exact.append(is_exact)
            consistent.append(is_consistent)
            set_change_from_base.append(not is_consistent)
            family_rows[row["family_id"]].append(
                {"exact": is_exact, "consistent": is_consistent}
            )
        n = len(variants)
        metrics = {
            "selector": selector_name,
            "families": len(family_rows),
            "paraphrase_evaluations": n,
            "paraphrase_exact_match_rate": sum(exact) / float(n),
            "paraphrase_error_rate_against_human_ground_truth": 1.0 - sum(exact) / float(n),
            "paraphrase_prediction_consistency_rate": sum(consistent) / float(n),
            "set_change_from_base_prediction_rate": sum(set_change_from_base) / float(n),
            "all_variants_human_semantics_reviewed": True,
            "by_family": {
                family: {
                    "n": len(items),
                    "exact_match_rate": sum(x["exact"] for x in items) / float(len(items)),
                    "prediction_consistency_rate": sum(x["consistent"] for x in items) / float(len(items)),
                }
                for family, items in sorted(family_rows.items())
            },
        }
        write_json(base / "results" / "{}_paraphrase_metrics.json".format(selector_name), metrics)
        results[selector_name] = metrics
    return results


def score_adversarial(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    final_path = _base(root) / "frozen" / "manifest.json"
    if final_path.exists():
        final = validate_manifest_with_assets(root, final_path)
        return {"baseline": final["baseline_adversarial_metrics"], "v32": final["v32_adversarial_metrics"]}
    validate_all_evaluation_inputs_frozen(root)
    base = _base(root)
    cases = read_csv(base / "frozen" / "adversarial_set.csv")
    case_map = {row["case_id"]: row for row in cases}
    results = {}
    for selector_name in ("baseline", "v32"):
        outputs = read_jsonl(base / "results" / "{}_adversarial_outputs.jsonl".format(selector_name))
        out_map = {row["case_id"]: row for row in outputs}
        if set(out_map) != set(case_map):
            raise Phase12V2Error("Adversarial output IDs do not match frozen cases.")
        taxonomy = Counter()
        correct = 0
        nonempty_rejection = 0
        details = []
        for case_id, case in case_map.items():
            out = out_map[case_id]
            expected = case["expected_status"]
            actual = out["status"]
            selected = set(out.get("selected_objectives") or [])
            if actual == expected:
                correct += 1
                if expected == "ambiguous":
                    category = "sensible_abstention"
                elif expected == "contradictory":
                    category = "contradiction_handling"
                else:
                    category = "out_of_scope_handling"
            elif actual == "malformed":
                category = "malformed_failure"
            elif actual == "valid" and selected == set(OBJECTIVES):
                category = "over_selection_all_objectives"
            elif actual == "valid" and selected:
                category = "unsupported_valid_selection"
            elif actual in REJECTION_STATUSES:
                category = "wrong_rejection_status"
            else:
                category = "silent_or_unclassified_failure"
            if expected in REJECTION_STATUSES and selected:
                nonempty_rejection += 1
            taxonomy[category] += 1
            details.append(
                {
                    "case_id": case_id,
                    "expected_status": expected,
                    "actual_status": actual,
                    "selected_objectives": sorted(selected),
                    "taxonomy": category,
                }
            )
        n = len(cases)
        metrics = {
            "selector": selector_name,
            "n": n,
            "status_accuracy": correct / float(n),
            "invalid_nonempty_selection_rate": nonempty_rejection / float(n),
            "malformed_rate": sum(x["actual_status"] == "malformed" for x in details) / float(n),
            "taxonomy_counts": dict(taxonomy),
            "details": details,
        }
        write_json(base / "results" / "{}_adversarial_metrics.json".format(selector_name), metrics)
        results[selector_name] = metrics
    return results


def finalize_benchmark(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    frozen = validate_all_evaluation_inputs_frozen(root)
    base = _base(root)
    required = [
        "baseline_primary_outputs.jsonl", "v32_primary_outputs.jsonl",
        "baseline_primary_metrics.json", "v32_primary_metrics.json", "paired_primary_comparison.json",
        "baseline_paraphrase_outputs.jsonl", "v32_paraphrase_outputs.jsonl",
        "baseline_paraphrase_metrics.json", "v32_paraphrase_metrics.json",
        "baseline_adversarial_outputs.jsonl", "v32_adversarial_outputs.jsonl",
        "baseline_adversarial_metrics.json", "v32_adversarial_metrics.json",
        "baseline_primary_run_metadata.json", "v32_primary_run_metadata.json",
        "baseline_paraphrase_run_metadata.json", "v32_paraphrase_run_metadata.json",
        "baseline_adversarial_run_metadata.json", "v32_adversarial_run_metadata.json",
    ]
    missing = [name for name in required if not (base / "results" / name).exists()]
    if missing:
        raise Phase12V2Error("Cannot finalize; missing result files: {}".format(", ".join(missing)))

    # Re-score from immutable raw outputs immediately before freezing final results.
    primary = score_primary(root)
    paraphrase = score_paraphrases(root)
    adversarial = score_adversarial(root)

    protected = [base / "results" / name for name in required]
    protected += [
        base / "frozen" / "design_manifest.json",
        base / "frozen" / "ground_truth_manifest.json",
        base / "frozen" / "primary_manifest.json",
        base / "frozen" / "paraphrase_manifest.json",
        base / "frozen" / "adversarial_manifest.json",
        root / V32_MANIFEST_REL,
        root / LEGACY_V2_SELECTOR_REL,
        root / LEGACY_V1_MANIFEST_REL,
    ]
    hashes = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in protected
    }
    final_manifest = {
        "artifact": "objective_selection_benchmark_v2",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "protocol_id": "objective_selection_benchmark_v2",
        "objective_vocabulary": list(OBJECTIVES),
        "primary_benchmark_n": primary["v32"]["n"],
        "ground_truth_source": "independent_human_majority_vote_only",
        "generation_intent_used_as_ground_truth": False,
        "legacy_phase12_v1_preserved": verify_legacy_v1_immutable(root),
        "selector_freezes": verify_selector_freezes(root),
        "paired_primary_comparison": primary["comparison"],
        "baseline_primary_metrics": primary["baseline"],
        "v32_primary_metrics": primary["v32"],
        "baseline_paraphrase_metrics": paraphrase["baseline"],
        "v32_paraphrase_metrics": paraphrase["v32"],
        "baseline_adversarial_metrics": adversarial["baseline"],
        "v32_adversarial_metrics": adversarial["v32"],
        "near_pareto_definition_unchanged": {
            "spec_id": "epsilon_pareto_v1",
            "epsilon": 0.05,
            "source": "awareml/engine/pareto_spec.py",
        },
        "weighting_policy_unchanged": {
            "policy_id": "equal_selected_v1",
            "rule": "Equal weight among selected objectives; all unselected objectives receive zero.",
        },
        "file_sha256": hashes,
    }
    manifest_path = base / "frozen" / "manifest.json"
    if manifest_path.exists():
        raise Phase12V2Error("Final v2 manifest already exists; do not overwrite a frozen benchmark.")
    write_json(manifest_path, final_manifest)
    digest = sha256_file(manifest_path)
    Path(str(manifest_path) + ".sha256").write_text(
        "{}  manifest.json\n".format(digest), encoding="utf-8"
    )
    # A new marker is used; the legacy v1 active marker is intentionally untouched.
    marker = root / "data" / "journal" / "active_objective_benchmark_v2.txt"
    marker.write_text(
        "objective_selection_benchmark_v2/frozen/manifest.json\n", encoding="utf-8"
    )
    return {"manifest": str(manifest_path.relative_to(root)), "sha256": digest, "payload": final_manifest}


def audit_benchmark(root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root(root)
    report: Dict[str, Any] = {
        "legacy_v1": verify_legacy_v1_immutable(root),
        "selectors": verify_selector_freezes(root),
        "stages": {},
    }
    for name in ("design", "ground_truth", "primary", "paraphrase", "adversarial"):
        path = _base(root) / "frozen" / "{}_manifest.json".format(name)
        if path.exists():
            report["stages"][name] = {
                "valid": True,
                "sha256": sha256_file(path),
                "artifact": validate_manifest_with_assets(root, path).get("artifact"),
            }
        else:
            report["stages"][name] = {"valid": False, "reason": "not frozen yet"}
    final = _base(root) / "frozen" / "manifest.json"
    if final.exists():
        report["final"] = {
            "valid": True,
            "sha256": sha256_file(final),
            "artifact": validate_manifest_with_assets(root, final).get("artifact"),
        }
    else:
        report["final"] = {"valid": False, "reason": "not finalized yet"}
    return report
