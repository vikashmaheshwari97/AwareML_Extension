from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

PHASE = 10
PROTOCOL_ID = "journal_experimental_protocol_v1"
PROTOCOL_NAME = "Journal Experimental Protocol v1"

CONFIG_DIR = Path("configs/journal")
PROMPT_PATH = Path("prompts/journal_objective_selection_v1.txt")
PROTOCOL_DIR = Path("data/journal/protocol_v1")
PROTOCOL_PATH = PROTOCOL_DIR / "journal_experimental_protocol_v1.json"
PROTOCOL_SHA_PATH = Path(str(PROTOCOL_PATH) + ".sha256")
ACTIVE_PROTOCOL_PATH = Path("data/journal/active_protocol.txt")

OBJECTIVES_CONFIG = CONFIG_DIR / "objectives_v1.json"
SCHEMA_CONFIG = CONFIG_DIR / "objective_selection_schema_v1.json"
LLM_CONFIG = CONFIG_DIR / "journal_llm_v1.json"
HELDOUT_POLICY_CONFIG = CONFIG_DIR / "heldout_policy_v1.json"
BASELINE_CONFIG = CONFIG_DIR / "engineering_baseline_v1.json"
BASELINE_ARTIFACTS_CONFIG = CONFIG_DIR / "baseline_artifacts_v1.json"
ROADMAP_RESOLUTION_CONFIG = CONFIG_DIR / "roadmap_resolution_v1.json"

CANONICAL_OBJECTIVE_LABELS = ("Accuracy", "Runtime", "Energy", "CO2")
CANONICAL_OBJECTIVE_KEYS = ("accuracy", "runtime", "energy", "co2")

PHASE10_ALLOWED_DIRTY_PREFIXES = (
    "awareml/journal/",
    "configs/journal/",
    "prompts/journal_objective_selection_v1.txt",
    "scripts/build_phase10_protocol.py",
    "scripts/validate_phase10_protocol.py",
    "scripts/freeze_phase10_protocol.py",
    "scripts/complete_phase10.py",
    "tests/test_phase10_protocol.py",
    "docs/PHASE10_JOURNAL_PROTOCOL_LOCK.md",
    "APPLY_PHASE10.txt",
    "PHASE10_CHANGELOG.md",
    "data/journal/",
)


class ProtocolError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    if not path.exists():
        raise ProtocolError("Missing JSON file: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ProtocolError("Missing YAML file: {}".format(path))
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("{} must contain a mapping.".format(path))
    return payload


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git"] + list(args),
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise ProtocolError("git executable is not available.")
    if proc.returncode != 0:
        raise ProtocolError(
            "git {} failed: {}".format(" ".join(args), proc.stderr.strip())
        )
    return proc.stdout.strip()


def git_head(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD")


def git_branch(root: Path) -> str:
    return _git(root, "branch", "--show-current")


def git_dirty_paths(root: Path) -> List[str]:
    output = _git(root, "status", "--porcelain", "--untracked-files=all")
    paths = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().strip('"').replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def phase10_only_dirty(paths: Sequence[str]) -> Tuple[bool, List[str]]:
    unexpected = []
    for path in paths:
        normalized = str(path).replace("\\", "/")
        if not any(
            normalized == prefix.rstrip("/") or normalized.startswith(prefix)
            for prefix in PHASE10_ALLOWED_DIRTY_PREFIXES
        ):
            unexpected.append(normalized)
    return len(unexpected) == 0, unexpected


def _assert_python_runtime() -> Dict[str, Any]:
    actual = tuple(sys.version_info[:3])
    expected = (3, 8, 10)
    if actual != expected:
        raise ProtocolError(
            "Phase 10 requires exact CPython 3.8.10; found {}.".format(
                platform.python_version()
            )
        )
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "required_version": "3.8.10",
        "match": True,
    }


def _objective_keys_from_current_code(root: Path) -> List[str]:
    path = root / "awareml" / "llm" / "schemas.py"
    if not path.exists():
        raise ProtocolError("Current Copilot schema not found: {}".format(path))
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "PrimaryObjectiveWeights":
            names = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    names.append(item.target.id)
            return names
    raise ProtocolError("PrimaryObjectiveWeights class not found in {}".format(path))


def _validate_objective_lock(root: Path) -> Dict[str, Any]:
    config = read_json(root / OBJECTIVES_CONFIG)
    labels = tuple(config.get("display_labels") or [])
    keys = tuple(config.get("internal_keys") or [])

    if labels != CANONICAL_OBJECTIVE_LABELS:
        raise ProtocolError(
            "Objective display labels must be exactly {}.".format(
                CANONICAL_OBJECTIVE_LABELS
            )
        )
    if keys != CANONICAL_OBJECTIVE_KEYS:
        raise ProtocolError(
            "Objective internal keys must be exactly {}.".format(
                CANONICAL_OBJECTIVE_KEYS
            )
        )

    code_keys = tuple(_objective_keys_from_current_code(root))
    if code_keys != CANONICAL_OBJECTIVE_KEYS:
        raise ProtocolError(
            "Current code objective keys {} do not match frozen journal vocabulary {}.".format(
                code_keys, CANONICAL_OBJECTIVE_KEYS
            )
        )

    schema = read_json(root / SCHEMA_CONFIG)
    selected = (
        schema.get("properties", {})
        .get("selected_objectives", {})
        .get("items", {})
        .get("enum", [])
    )
    if tuple(selected) != CANONICAL_OBJECTIVE_LABELS:
        raise ProtocolError(
            "Objective-selection schema enum must use literal journal labels."
        )

    prompt = (root / PROMPT_PATH).read_text(encoding="utf-8")
    for label in CANONICAL_OBJECTIVE_LABELS:
        if label not in prompt:
            raise ProtocolError(
                "Journal prompt is missing literal objective label: {}".format(label)
            )

    return {
        "display_labels": list(labels),
        "internal_keys": list(keys),
        "source_in_current_code": "awareml/llm/schemas.py::PrimaryObjectiveWeights",
        "current_code_match": True,
        "benchmark_task": "multi_label_objective_selection",
        "weights_part_of_primary_selection_output": False,
        "prompt_sha256": sha256_file(root / PROMPT_PATH),
        "schema_sha256": sha256_file(root / SCHEMA_CONFIG),
    }


def _validate_dataset_policy(root: Path) -> Dict[str, Any]:
    policy = read_json(root / HELDOUT_POLICY_CONFIG)
    train_path = root / policy["development_manifest"]
    heldout_path = root / policy["heldout_manifest"]
    train = read_yaml(train_path)
    heldout = read_yaml(heldout_path)

    train_ids = list(train.get("dataset_ids") or [])
    heldout_ids = list(heldout.get("dataset_ids") or [])

    if train.get("manifest_id") != "train_v1_47":
        raise ProtocolError("Expected train_v1_47 development manifest.")
    if train.get("purpose") != "meta_train":
        raise ProtocolError("train_v1_47 must have purpose=meta_train.")
    if int(train.get("expected_count", -1)) != 47 or len(train_ids) != 47:
        raise ProtocolError("Development manifest must contain exactly 47 dataset IDs.")
    if train.get("frozen") is not True:
        raise ProtocolError("The 47-dataset development split must remain frozen.")

    if heldout.get("manifest_id") != "test_v1_23":
        raise ProtocolError("Expected test_v1_23 held-out manifest.")
    if heldout.get("purpose") != "heldout_test":
        raise ProtocolError("test_v1_23 must have purpose=heldout_test.")
    if int(heldout.get("expected_count", -1)) != 23:
        raise ProtocolError("Held-out manifest must declare expected_count=23.")

    if heldout_ids:
        overlap = sorted(set(train_ids) & set(heldout_ids))
        if overlap:
            raise ProtocolError(
                "Development/held-out dataset overlap detected: {}".format(overlap)
            )
        if len(heldout_ids) != 23:
            raise ProtocolError(
                "If held-out identities are present, exactly 23 IDs are required."
            )

    identity_status = (
        "resolved_and_frozen"
        if len(heldout_ids) == 23 and heldout.get("frozen") is True
        else "unresolved_do_not_use"
    )

    return {
        "canonical_role_resolution": {
            "development_meta_training": {
                "manifest": policy["development_manifest"],
                "count": 47,
                "frozen": True,
            },
            "final_heldout_evaluation": {
                "manifest": policy["heldout_manifest"],
                "expected_count": 23,
                "identity_count_currently_present": len(heldout_ids),
                "manifest_frozen": bool(heldout.get("frozen")),
                "identity_status": identity_status,
            },
        },
        "roadmap_wording_resolution": policy["roadmap_wording_resolution"],
        "operational_policy": policy["operational_policy"],
        "heldout_access_allowed_before_phase18": False,
        "phase18_identity_gate": policy["phase18_identity_gate"],
        "heldout_dataset_contents_read_by_phase10": False,
        "train_manifest_sha256": sha256_file(train_path),
        "heldout_manifest_sha256": sha256_file(heldout_path),
    }


def _collect_artifacts(root: Path) -> List[Dict[str, Any]]:
    spec = read_json(root / BASELINE_ARTIFACTS_CONFIG)
    rows = []
    for entry in spec.get("artifacts", []):
        rel = Path(entry["path"])
        path = root / rel
        required = bool(entry.get("required", True))
        if not path.exists():
            if required:
                raise ProtocolError("Missing baseline artifact: {}".format(rel))
            rows.append(
                {
                    "path": rel.as_posix(),
                    "role": entry.get("role"),
                    "required": False,
                    "status": "missing_optional",
                    "sha256": None,
                    "size_bytes": None,
                }
            )
            continue

        actual_sha = sha256_file(path)
        expected_sha = entry.get("expected_sha256")
        if expected_sha and actual_sha != expected_sha:
            raise ProtocolError(
                "Baseline checksum mismatch for {}: expected {}, got {}.".format(
                    rel, expected_sha, actual_sha
                )
            )

        rows.append(
            {
                "path": rel.as_posix(),
                "role": entry.get("role"),
                "required": required,
                "status": "present",
                "sha256": actual_sha,
                "size_bytes": int(path.stat().st_size),
            }
        )
    return rows


def _active_markers(root: Path) -> Dict[str, Any]:
    marker_paths = [
        "data/meta/active_recommender_v2.txt",
        "data/llm/active_copilot.txt",
        "data/llm/active_faithfulness.txt",
        "data/ui/active_ui.txt",
    ]
    result = {}
    for rel in marker_paths:
        path = root / rel
        result[rel] = {
            "exists": path.exists(),
            "value": path.read_text(encoding="utf-8").strip() if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return result


def _validate_engineering_baseline(root: Path) -> Dict[str, Any]:
    config = read_json(root / BASELINE_CONFIG)
    expected_branch = config["branch"]
    expected_commit = config["reference_commit"]

    branch = git_branch(root)
    head = git_head(root)
    if branch != expected_branch:
        raise ProtocolError(
            "Phase 10 must be frozen from branch '{}'; current branch is '{}'.".format(
                expected_branch, branch
            )
        )
    if head != expected_commit:
        raise ProtocolError(
            "Engineering baseline mismatch.\n"
            "Expected current main commit: {}\n"
            "Current HEAD: {}\n"
            "Run `git pull --ff-only origin main` before continuing.".format(
                expected_commit, head
            )
        )

    dirty = git_dirty_paths(root)
    ok, unexpected = phase10_only_dirty(dirty)
    if not ok:
        raise ProtocolError(
            "Non-Phase-10 working-tree changes would contaminate the engineering "
            "baseline: {}. Commit/revert them before completing Phase 10.".format(
                ", ".join(unexpected)
            )
        )

    return {
        "repository": config["repository"],
        "branch": branch,
        "commit": head,
        "commit_role": "pre_phase10_engineering_baseline",
        "phase10_overlay_is_only_allowed_dirty_change": True,
        "dirty_paths_at_protocol_build": dirty,
        "artifact_checksums": _collect_artifacts(root),
        "active_markers": _active_markers(root),
    }


def _query_ollama_runtime(
    llm_config: Mapping[str, Any],
    base_url_override: Optional[str] = None,
) -> Dict[str, Any]:
    import requests

    base_url = (
        base_url_override
        or os.getenv("OLLAMA_BASE_URL")
        or llm_config.get("base_url")
        or "http://localhost:11434"
    ).rstrip("/")

    try:
        vr = requests.get(base_url + "/api/version", timeout=5.0)
        vr.raise_for_status()
        tr = requests.get(base_url + "/api/tags", timeout=5.0)
        tr.raise_for_status()
        version_payload = vr.json()
        tags_payload = tr.json()
    except Exception as exc:
        raise ProtocolError(
            "Ollama is required for the Phase-10 journal LLM lock but is not "
            "reachable at {}: {}: {}".format(
                base_url, type(exc).__name__, exc
            )
        )

    required_tag = str(llm_config["required_model_tag"])
    models = list(tags_payload.get("models") or [])
    exact = None
    for row in models:
        name = row.get("name") or row.get("model")
        if name == required_tag:
            exact = row
            break

    if exact is None:
        installed = [
            str(row.get("name") or row.get("model"))
            for row in models
            if row.get("name") or row.get("model")
        ]
        raise ProtocolError(
            "Required journal model '{}' is not installed exactly.\n"
            "Installed models: {}\n"
            "Phase 10 forbids silent model fallback.\n"
            "Run: ollama pull {}".format(required_tag, installed, required_tag)
        )

    details = exact.get("details") or {}
    parameter_size = details.get("parameter_size")
    if parameter_size and "8" not in str(parameter_size):
        raise ProtocolError(
            "Journal model tag '{}' resolved to unexpected parameter size '{}'.".format(
                required_tag, parameter_size
            )
        )

    digest = exact.get("digest")
    if not digest:
        raise ProtocolError(
            "Ollama did not expose a model digest for {}; cannot freeze exact model.".format(
                required_tag
            )
        )

    return {
        "verified": True,
        "base_url": base_url,
        "ollama_version": version_payload.get("version"),
        "model_tag": required_tag,
        "model_digest": digest,
        "model_modified_at": exact.get("modified_at"),
        "model_size_bytes": exact.get("size"),
        "model_details": details,
        "fallback_used": False,
    }


def _validate_llm_lock(
    root: Path,
    require_ollama: bool,
    base_url_override: Optional[str] = None,
) -> Dict[str, Any]:
    config = read_json(root / LLM_CONFIG)
    if config.get("strict_exact_model") is not True:
        raise ProtocolError("Journal LLM must use strict_exact_model=true.")
    if config.get("allow_fallback") is not False:
        raise ProtocolError("Journal LLM must use allow_fallback=false.")
    if config.get("required_model_tag") != "llama3:8b":
        raise ProtocolError(
            "Roadmap-aligned journal model tag must be exactly llama3:8b."
        )

    generation = config.get("generation") or {}
    required_generation = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stream": False,
        "format": "json",
    }
    for key, expected in required_generation.items():
        if generation.get(key) != expected:
            raise ProtocolError(
                "Journal LLM generation option {} must be {!r}.".format(key, expected)
            )

    runtime = (
        _query_ollama_runtime(config, base_url_override=base_url_override)
        if require_ollama
        else None
    )
    return {
        "provider": config["provider"],
        "model_family": config["model_family"],
        "parameter_size": config["parameter_size"],
        "required_model_tag": config["required_model_tag"],
        "strict_exact_model": True,
        "allow_fallback": False,
        "generation": generation,
        "prompt_file": config["prompt_file"],
        "prompt_sha256": sha256_file(root / config["prompt_file"]),
        "schema_file": config["schema_file"],
        "schema_sha256": sha256_file(root / config["schema_file"]),
        "runtime_lock": runtime,
        "runtime_verified": runtime is not None,
        "phase11_requirement": config["phase11_requirement"],
    }


def validate_static_inputs(
    root: Path,
    verify_git: bool = True,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    runtime = _assert_python_runtime()
    objectives = _validate_objective_lock(root)
    dataset_policy = _validate_dataset_policy(root)
    roadmap = read_json(root / ROADMAP_RESOLUTION_CONFIG)
    baseline = _validate_engineering_baseline(root) if verify_git else None
    return {
        "runtime": runtime,
        "objectives": objectives,
        "dataset_policy": dataset_policy,
        "roadmap_resolution": roadmap,
        "engineering_baseline": baseline,
    }


def build_protocol(
    root: Path,
    require_ollama: bool = True,
    base_url_override: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    static = validate_static_inputs(root, verify_git=True)
    llm_lock = _validate_llm_lock(
        root,
        require_ollama=require_ollama,
        base_url_override=base_url_override,
    )

    return {
        "schema_version": "1.0",
        "phase": PHASE,
        "protocol_id": PROTOCOL_ID,
        "protocol_name": PROTOCOL_NAME,
        "created_at_utc": utc_now(),
        "release_status": "frozen" if require_ollama else "preview",
        "purpose": (
            "Freeze the journal experimental contract before Phase 11 changes "
            "Copilot objective inference or any later journal evaluation."
        ),
        "engineering_baseline": static["engineering_baseline"],
        "roadmap_alignment": static["roadmap_resolution"],
        "dataset_split_policy": static["dataset_policy"],
        "objective_vocabulary": static["objectives"],
        "journal_llm": llm_lock,
        "downstream_weighting_policy": {
            "phase10_status": "explicitly_not_selected_yet",
            "primary_objective_selection_benchmark_uses_weights": False,
            "rule": (
                "Phase 11 must freeze the mapping from selected objective subset "
                "to recommender weights before Phase 12 benchmark execution. "
                "Objective-selection accuracy and downstream weighting are separate tasks."
            ),
        },
        "near_pareto_policy": {
            "phase10_status": "existing_implementation_recorded_not_redefined_here",
            "current_engineering_default": "epsilon_nondominance",
            "current_epsilon": 0.05,
            "journal_canonicalization_gate": "Phase 13",
        },
        "phase_gates": {
            "phase11": [
                "Implement selected-objective-set output against the frozen vocabulary.",
                "Use the exact frozen prompt/schema and exact journal LLM lock.",
                "Disable silent model fallback in journal evaluation mode.",
                "Freeze downstream weighting policy.",
            ],
            "phase12": [
                "Do not use the 705 recommender meta-runs as Copilot benchmark statements.",
                "Create a separate human-annotated NL objective-selection benchmark.",
            ],
            "phase15": [
                "Controlled explanation-correctness evidence must not silently consume the final held-out 23 datasets."
            ],
            "phase18": [
                "Recover the exact 23 held-out dataset identities from authoritative project history.",
                "Populate test_v1_23.yaml with exactly 23 IDs.",
                "Verify zero overlap with train_v1_47.",
                "Set held-out manifest frozen=true.",
                "Only then permit final held-out evaluation.",
            ],
        },
        "integrity_guards": {
            "heldout_dataset_contents_read_by_phase10": False,
            "heldout_dataset_results_used_by_phase10": False,
            "heldout_identity_names_invented": False,
            "705_meta_logs_modified": False,
            "recommender_v2_retrained": False,
            "copilot_behavior_modified": False,
            "faithfulness_behavior_modified": False,
            "ui_behavior_modified": False,
        },
        "runtime": static["runtime"],
    }


def freeze_protocol(
    root: Path,
    base_url_override: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    protocol = build_protocol(
        root,
        require_ollama=True,
        base_url_override=base_url_override,
    )
    path = root / PROTOCOL_PATH
    sha_path = root / PROTOCOL_SHA_PATH
    active_path = root / ACTIVE_PROTOCOL_PATH

    write_json(path, protocol)
    digest = sha256_file(path)
    sha_path.write_text(
        "{}  {}\n".format(digest, path.name),
        encoding="utf-8",
    )
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(
        "protocol_v1/{}\n".format(path.name),
        encoding="utf-8",
    )
    return {
        "protocol_path": str(path),
        "sha256_path": str(sha_path),
        "active_protocol_path": str(active_path),
        "sha256": digest,
        "protocol": protocol,
    }


def _validate_baseline_artifact_checksums(
    root: Path,
    artifact_rows: Iterable[Mapping[str, Any]],
) -> None:
    for row in artifact_rows:
        if row.get("status") != "present":
            continue
        path = root / str(row["path"])
        if not path.exists():
            raise ProtocolError(
                "Frozen baseline artifact disappeared: {}".format(row["path"])
            )
        if sha256_file(path) != row.get("sha256"):
            raise ProtocolError(
                "Frozen baseline artifact changed after protocol lock: {}.".format(
                    row["path"]
                )
            )


def validate_frozen_protocol(
    root: Path,
    require_ollama_match: bool = True,
    base_url_override: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    path = root / PROTOCOL_PATH
    sha_path = root / PROTOCOL_SHA_PATH
    active_path = root / ACTIVE_PROTOCOL_PATH

    for required in [path, sha_path, active_path]:
        if not required.exists():
            raise ProtocolError(
                "Missing frozen Phase-10 artifact: {}".format(required)
            )

    expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual_sha = sha256_file(path)
    if expected_sha != actual_sha:
        raise ProtocolError("Journal protocol checksum mismatch.")

    expected_active = "protocol_v1/{}".format(path.name)
    if active_path.read_text(encoding="utf-8").strip() != expected_active:
        raise ProtocolError("Active journal protocol marker is incorrect.")

    protocol = read_json(path)
    if protocol.get("phase") != 10 or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("Unexpected frozen Phase-10 protocol identity.")
    if protocol.get("release_status") != "frozen":
        raise ProtocolError("Journal protocol is not frozen.")

    labels = tuple(
        (protocol.get("objective_vocabulary") or {}).get("display_labels") or []
    )
    if labels != CANONICAL_OBJECTIVE_LABELS:
        raise ProtocolError("Frozen objective vocabulary changed.")

    split = protocol.get("dataset_split_policy") or {}
    if split.get("heldout_access_allowed_before_phase18") is not False:
        raise ProtocolError("Held-out protection policy was weakened.")

    baseline = protocol.get("engineering_baseline") or {}
    if baseline.get("commit") != read_json(root / BASELINE_CONFIG)["reference_commit"]:
        raise ProtocolError("Frozen engineering baseline commit is incorrect.")
    _validate_baseline_artifact_checksums(
        root,
        baseline.get("artifact_checksums") or [],
    )

    llm = protocol.get("journal_llm") or {}
    if llm.get("required_model_tag") != "llama3:8b":
        raise ProtocolError("Frozen journal LLM tag changed.")
    if llm.get("allow_fallback") is not False:
        raise ProtocolError("Frozen journal LLM unexpectedly permits fallback.")
    if not llm.get("runtime_verified"):
        raise ProtocolError("Frozen journal LLM runtime was never verified.")

    if require_ollama_match:
        live_runtime = _query_ollama_runtime(
            read_json(root / LLM_CONFIG),
            base_url_override=base_url_override,
        )
        frozen_runtime = llm.get("runtime_lock") or {}
        if live_runtime.get("model_digest") != frozen_runtime.get("model_digest"):
            raise ProtocolError(
                "The locally installed journal model digest differs from the frozen digest."
            )
        if live_runtime.get("ollama_version") != frozen_runtime.get("ollama_version"):
            raise ProtocolError(
                "The local Ollama version differs from the frozen journal runtime."
            )

    return {
        "status": "PASS",
        "protocol_path": str(path),
        "sha256": actual_sha,
        "engineering_baseline_commit": baseline.get("commit"),
        "objectives": list(labels),
        "heldout_expected_count": (
            split.get("canonical_role_resolution", {})
            .get("final_heldout_evaluation", {})
            .get("expected_count")
        ),
        "heldout_identity_status": (
            split.get("canonical_role_resolution", {})
            .get("final_heldout_evaluation", {})
            .get("identity_status")
        ),
        "journal_model": llm.get("required_model_tag"),
        "journal_model_digest": (llm.get("runtime_lock") or {}).get("model_digest"),
        "ollama_version": (llm.get("runtime_lock") or {}).get("ollama_version"),
        "heldout_dataset_contents_read_by_phase10": False,
    }
