from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase18_final_heldout_31.json"
INVENTORY_PATH = ROOT / "configs" / "phase18_dataset_inventory_31.csv"
DATASET_DIR = ROOT / "data" / "Testing Datasets"
WORK_DIR = ROOT / "data" / "journal" / "phase18_final_heldout_31_v1"
PREFLIGHT_DIR = WORK_DIR / "preflight"
FROZEN_DIR = WORK_DIR / "frozen"
RESULT_DIR = ROOT / "artifacts" / "phase18_final_heldout_31_v1"
RUNS_DIR = RESULT_DIR / "runs"
REDUCED_DIR = RESULT_DIR / "reduced"
EVAL_DIR = RESULT_DIR / "evaluation"

FRAMEWORKS = ("AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha")
SEEDS = (42, 43, 44)
OBJECTIVES = ("accuracy", "runtime", "energy", "co2")
MAX_SAMPLES = 30_000
WINDOW_SIZE = 1_000
TIME_BUDGET_SEC = 60.0
PREFERENCES_PER_DATASET = 100
PREFERENCE_GLOBAL_SEED = 42
EXPECTED_DATASETS = 31
EXPECTED_HISTORICAL = 23
EXPECTED_EXTENSION = 8
EXPECTED_RUNS = EXPECTED_DATASETS * len(FRAMEWORKS) * len(SEEDS)
EXPECTED_AGGREGATED = EXPECTED_DATASETS * len(FRAMEWORKS)
EXPECTED_PREFERENCE_CASES = EXPECTED_DATASETS * PREFERENCES_PER_DATASET
PROTOCOL_ID = "phase18_final_heldout_31_v1"
FREEZE_CONFIRMATION = "REVIEWED_31_DATASET_TASK_POLICY"
PREFLIGHT_SCHEMA_VERSION = "2.5"
PRIMARY_ELIGIBILITY_RULE = "explicit_task_policy_valid_and_no_development_overlap"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively convert values to strict JSON-compatible Python objects.

    Phase-18 audit DataFrames can legitimately contain pandas/NumPy missing values
    (for example when a dataset could not be fully profiled).  Python's JSON
    encoder rejects NaN/Infinity when ``allow_nan=False``.  For protocol files we
    deliberately preserve strict RFC-compatible JSON and represent every non-finite
    or missing scalar as JSON ``null`` rather than emitting the non-standard NaN
    token or crashing the preflight.
    """
    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # Covers pd.NA, pd.NaT and other pandas missing scalars.  Some container
    # objects return an array from pd.isna(), so only accept a scalar bool result.
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except Exception:
        pass

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing Phase-18 config: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_inventory() -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"Missing Phase-18 inventory: {INVENTORY_PATH}")
    df = pd.read_csv(INVENTORY_PATH, keep_default_na=False)
    required = {
        "dataset_id", "filename", "cohort", "target", "sensitive_attribute",
        "positive_label_json", "dataset_family", "source_type", "drift_type", "notes", "task_policy",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Inventory missing columns: {missing}")
    if len(df) != EXPECTED_DATASETS:
        raise ValueError(f"Expected {EXPECTED_DATASETS} inventory rows, found {len(df)}")
    if df["dataset_id"].duplicated().any():
        raise ValueError("Inventory dataset_id values are not unique.")
    if df["filename"].duplicated().any():
        raise ValueError("Inventory filename values are not unique.")
    counts = df["cohort"].value_counts().to_dict()
    if int(counts.get("historical_23", 0)) != EXPECTED_HISTORICAL:
        raise ValueError("Inventory must contain exactly 23 historical_23 datasets.")
    if int(counts.get("extension_8", 0)) != EXPECTED_EXTENSION:
        raise ValueError("Inventory must contain exactly 8 extension_8 datasets.")
    return df




def _resolve_python_candidate(env_name: str, windows_rel: str, posix_rel: str) -> Optional[Path]:
    override = os.getenv(env_name, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        return candidate if candidate.exists() else None
    for candidate in (ROOT / windows_rel, ROOT / posix_rel):
        if candidate.exists():
            return candidate
    return None


def _python_environment_record(label: str, executable: Optional[Path]) -> Dict[str, Any]:
    if executable is None:
        return {"label": label, "available": False, "executable": None}
    exe = str(executable)
    record: Dict[str, Any] = {"label": label, "available": True, "executable": exe}
    try:
        record["python_version"] = subprocess.check_output(
            [exe, "-c", "import sys; print(sys.version.replace(chr(10), ' '))"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True, timeout=30,
        ).strip()
    except Exception as exc:
        record["python_version_error"] = f"{type(exc).__name__}: {exc}"
    try:
        freeze = subprocess.check_output(
            [exe, "-m", "pip", "freeze", "--all"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True, timeout=120,
        )
        record["pip_freeze"] = sorted(
            line.strip() for line in freeze.splitlines() if line.strip()
        )
    except Exception as exc:
        record["pip_freeze"] = []
        record["pip_freeze_error"] = f"{type(exc).__name__}: {exc}"
    return record


def capture_environment_snapshot() -> Dict[str, Any]:
    main_exe = Path(sys.executable).resolve()
    evo_exe = _resolve_python_candidate(
        "AWAREML_EVO_PYTHON", ".venv-evo/Scripts/python.exe", ".venv-evo/bin/python"
    )
    oaml_exe = _resolve_python_candidate(
        "AWAREML_OAML_PYTHON", ".venv-oaml/Scripts/python.exe", ".venv-oaml/bin/python"
    )
    return {
        "created_utc": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "main": _python_environment_record("main", main_exe),
        "evo": _python_environment_record("evo", evo_exe),
        "oaml": _python_environment_record("oaml", oaml_exe),
        "environment_overrides": {
            "AWAREML_EVO_PYTHON": os.getenv("AWAREML_EVO_PYTHON"),
            "AWAREML_OAML_PYTHON": os.getenv("AWAREML_OAML_PYTHON"),
            "AWAREML_OAML_MODE": os.getenv("AWAREML_OAML_MODE", "online"),
        },
    }


def phase18_source_files() -> List[Path]:
    rels = [
        "awareml/engine/runner.py",
        "awareml/types.py",
        "awareml/analysis/fairness.py",
        "awareml/analysis/explainability.py",
        "awareml/analysis/sustainability.py",
        "awareml/frameworks/registry.py",
        "awareml/frameworks/autostreamml.py",
        "awareml/frameworks/autoclass.py",
        "awareml/frameworks/evoautoml.py",
        "awareml/frameworks/oaml.py",
        "awareml/frameworks/chacha.py",
        "awareml/workers/evo_worker.py",
        "awareml/workers/oaml_worker.py",
        "awareml/recommender/v2_service.py",
        "awareml/recommender/v2_profile.py",
        "awareml/recommender/v2_ranking.py",
        "awareml/recommender/v2_uncertainty.py",
        "awareml/recommender/v2_models.py",
        "awareml/recommender/v2_evaluation.py",
        "scripts/phase18_common.py",
        "scripts/phase18_prepare_heldout.py",
        "scripts/phase18_collect_results.py",
        "scripts/phase18_evaluate_recommender.py",
        "scripts/phase18_build_journal_tables.py",
        "configs/phase18_final_heldout_31.json",
        "configs/phase18_dataset_inventory_31.csv",
        "hpc/production/phase18/run_phase18_task.py",
        "hpc/production/phase18/resume_phase18_campaign.py",
        "hpc/production/phase18/phase18_array.sbatch",
    ]
    return [ROOT / rel for rel in rels if (ROOT / rel).exists()]


def current_git_head(root: Path = ROOT) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or None
    except Exception:
        return None


def current_git_status(root: Path = ROOT) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(root), stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return None


def normalize_dataset_id(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    if text.lower().endswith(".csv"):
        text = text[:-4]
    return "".join(ch.lower() for ch in text if ch.isalnum())


def resolve_target(inventory_row: Mapping[str, Any], df: pd.DataFrame) -> str:
    target = str(inventory_row.get("target") or "").strip()
    if target == "__LAST__":
        if df.shape[1] < 2:
            raise ValueError("Dataset must contain at least one feature and one target column.")
        return str(df.columns[-1])
    if target not in df.columns:
        raise ValueError(f"Configured target {target!r} not present in columns.")
    return target


def parse_positive_label(raw: Any) -> Any:
    text = str(raw or "").strip()
    if not text:
        return 1
    try:
        return json.loads(text)
    except Exception:
        return text


def target_task_assessment(series: pd.Series) -> Dict[str, Any]:
    n = int(len(series))
    unique = int(series.nunique(dropna=True))
    non_null = int(series.notna().sum())
    frac = float(unique / non_null) if non_null else 1.0
    dtype = str(series.dtype)
    numeric = bool(pd.api.types.is_numeric_dtype(series))

    if unique < 2:
        task = "invalid_single_class"
        status = "FAIL"
        reason = "target has fewer than two observed classes"
    elif not numeric:
        if unique <= 100 or frac <= 0.05:
            task = "classification"
            status = "PASS" if unique <= 20 else "REVIEW"
            reason = "categorical/string target"
        else:
            task = "suspicious_high_cardinality_categorical"
            status = "FAIL"
            reason = "high-cardinality target is not credible as a classification label without review"
    else:
        if unique <= 20:
            task = "classification"
            status = "PASS"
            reason = "low-cardinality numeric target"
        elif unique <= 100 and frac <= 0.05:
            task = "classification_multiclass_review"
            status = "REVIEW"
            reason = "numeric target has 21-100 classes; verify intended multiclass semantics"
        else:
            task = "likely_continuous_or_regression"
            status = "FAIL"
            reason = "numeric target has high cardinality and looks continuous"

    values = series.value_counts(dropna=False).head(20)
    dist = [
        {"value": str(idx), "count": int(count), "fraction": float(count / n) if n else 0.0}
        for idx, count in values.items()
    ]
    return {
        "target_dtype": dtype,
        "target_unique": unique,
        "target_unique_fraction": frac,
        "task_assessment": task,
        "audit_status": status,
        "audit_reason": reason,
        "target_distribution_top20_json": canonical_json(dist),
    }


def _linear_quantile(values: np.ndarray, q: Sequence[float]) -> np.ndarray:
    try:
        return np.quantile(values, q, method="linear")
    except TypeError:
        return np.quantile(values, q, interpolation="linear")


def build_ordinal_quantile_policy(series: pd.Series, max_classes: int = 5) -> Dict[str, Any]:
    """Define a deterministic derived ordinal classification task before outcomes.

    This is benchmark/task construction, not a learned model transformation. The
    original source target is never supplied as a feature. Parameters are frozen
    before any AutoML framework outcome is generated.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("ordinal_quantile_max5 requires a numeric target with no missing values")
    if int(numeric.nunique()) < 2:
        raise ValueError("ordinal_quantile_max5 requires at least two unique target values")
    q = [i / float(max_classes) for i in range(1, max_classes)]
    raw = [float(v) for v in _linear_quantile(numeric.to_numpy(dtype=float), q)]
    thresholds: List[float] = []
    for value in raw:
        if not thresholds or value > thresholds[-1]:
            thresholds.append(value)
    labels = np.digitize(numeric.to_numpy(dtype=float), np.asarray(thresholds, dtype=float), right=True)
    n_classes = int(len(np.unique(labels)))
    if n_classes < 2:
        raise ValueError("derived ordinal task collapsed to fewer than two classes")
    payload = {
        "transform_type": "ordinal_quantile_max5",
        "requested_max_classes": int(max_classes),
        "quantiles": q,
        "thresholds": thresholds,
        "right": True,
        "effective_classes": n_classes,
        "label_values": sorted(int(v) for v in np.unique(labels).tolist()),
        "definition_scope": "full_source_target_preoutcome_benchmark_construction",
        "warning": "Derived classification task; not the original regression/continuous prediction target.",
    }
    payload["transform_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return payload


def apply_task_policy(
    df: pd.DataFrame,
    source_target: str,
    task_policy: str,
    transform_payload: Optional[Mapping[str, Any]] = None,
    evaluation_target: str = "__phase18_target__",
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    policy = str(task_policy or "native_classification")
    if source_target not in df.columns:
        raise ValueError(f"Source target {source_target!r} is missing.")
    if policy == "native_classification":
        return df.copy(), source_target, {"transform_type": "none", "effective_classes": int(df[source_target].nunique(dropna=True))}
    if policy != "ordinal_quantile_max5":
        raise ValueError(f"Unknown Phase-18 task_policy: {policy!r}")
    payload = dict(transform_payload or build_ordinal_quantile_policy(df[source_target], max_classes=5))
    thresholds = np.asarray(payload.get("thresholds") or [], dtype=float)
    numeric = pd.to_numeric(df[source_target], errors="coerce")
    if numeric.isna().any():
        raise ValueError("Derived ordinal target contains missing/non-numeric values.")
    labels = np.digitize(numeric.to_numpy(dtype=float), thresholds, right=bool(payload.get("right", True)))
    out = df.drop(columns=[source_target]).copy()
    if evaluation_target in out.columns:
        raise ValueError(f"Reserved evaluation target column already exists: {evaluation_target}")
    out[evaluation_target] = labels.astype(int)
    effective = int(out[evaluation_target].nunique(dropna=True))
    expected = int(payload.get("effective_classes", effective))
    if effective != expected:
        raise ValueError(f"Derived target class count changed: actual={effective} frozen={expected}")
    return out, evaluation_target, payload


def dataframe_profile(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    features = df.drop(columns=[target])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    counts = df[target].value_counts(dropna=False)
    n = float(len(df))
    fractions = [float(v / n) for v in counts.tolist()] if n else []
    n_classes = int(df[target].nunique(dropna=False))
    if fractions:
        majority = float(max(fractions))
        minority = float(min(fractions))
        imbalance = float(majority / minority) if minority > 0 else math.nan
        entropy = -sum(p * math.log(p) for p in fractions if p > 0)
        entropy_norm = float(entropy / math.log(n_classes)) if n_classes > 1 else 0.0
    else:
        majority = minority = imbalance = entropy_norm = math.nan
    return {
        "n_samples_dataset": int(len(df)),
        "n_features": int(features.shape[1]),
        "n_numeric_features": int(len(numeric)),
        "n_categorical_features": int(len(categorical)),
        "numeric_feature_fraction": float(len(numeric) / features.shape[1]) if features.shape[1] else 0.0,
        "categorical_feature_fraction": float(len(categorical) / features.shape[1]) if features.shape[1] else 0.0,
        "missing_fraction": float(df.isna().mean().mean()),
        "n_classes": n_classes,
        "majority_class_fraction": majority,
        "minority_class_fraction": minority,
        "class_imbalance_ratio": imbalance,
        "class_entropy_normalized": entropy_norm,
    }


def _training_overlap_sources() -> Tuple[set[str], set[str], List[str], Dict[str, set[str]]]:
    """Load development IDs/hashes and retain hash->dataset evidence.

    The original Phase-18 preflight stopped on the first overlap.  v2.4 keeps
    the same strict separation rule, but gathers complete evidence so one run
    reports every problem instead of producing a sequence of stack traces.
    """
    ids: set[str] = set()
    hashes: set[str] = set()
    sources: List[str] = []
    hash_to_ids: Dict[str, set[str]] = {}
    candidates = [
        ROOT / "data" / "meta" / "snapshots" / "recommender_train_v2.parquet",
        ROOT / "data" / "meta" / "snapshots" / "meta_logs_v2.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        sources.append(str(path.relative_to(ROOT)))
        if "dataset_id" in frame.columns:
            ids.update(normalize_dataset_id(v) for v in frame["dataset_id"].dropna().tolist())

        hash_columns = [
            col for col in (
                "dataset_sha256", "source_sha256", "dataset_provenance__source_sha256"
            ) if col in frame.columns
        ]
        for col in hash_columns:
            for idx, raw_hash in frame[col].items():
                value = str(raw_hash).strip().lower()
                if not value or value == "nan":
                    continue
                hashes.add(value)
                if "dataset_id" in frame.columns:
                    try:
                        matched_id = str(frame.at[idx, "dataset_id"]).strip()
                    except Exception:
                        matched_id = ""
                    if matched_id:
                        hash_to_ids.setdefault(value, set()).add(matched_id)
    return ids, hashes, sources, hash_to_ids


def audit_datasets(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    inventory = load_inventory()
    train_ids, train_hashes, overlap_sources, hash_to_ids = _training_overlap_sources()
    rows: List[Dict[str, Any]] = []

    for _, item in inventory.iterrows():
        path = Path(dataset_dir) / str(item["filename"])
        base = dict(item)
        base.update({
            "dataset_path": str(path),
            "exists": path.exists(),
            "file_size_bytes": int(path.stat().st_size) if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
            "resolved_target": None,
            "source_target_audit_status": "FAIL",
            "overlap_with_training_id": False,
            "overlap_with_training_sha256": False,
            "training_overlap_matched_dataset_ids": "",
            "training_overlap_sources": ";".join(overlap_sources),
            "primary_heldout_eligible": False,
            "evaluation_role": "unresolved",
            "preflight_status": "FAIL",
            "preflight_reason": "missing file",
            "evaluation_target": None,
            "task_transform_payload_json": "",
            "task_transform_sha256": "",
        })
        if not path.exists():
            rows.append(base)
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
            source_target = resolve_target(item, df)
            source_assess = target_task_assessment(df[source_target])
            policy = str(item.get("task_policy") or "native_classification").strip()
            transformed, evaluation_target, transform_payload = apply_task_policy(
                df,
                source_target=source_target,
                task_policy=policy,
            )
            eval_assess = target_task_assessment(transformed[evaluation_target])
            profile = dataframe_profile(transformed, evaluation_target)
            normalized = normalize_dataset_id(item["dataset_id"])
            by_id = normalized in train_ids
            file_hash = str(base["sha256"]).lower() if base["sha256"] else ""
            by_hash = file_hash in train_hashes if file_hash else False
            matched_ids = sorted(hash_to_ids.get(file_hash, set())) if by_hash else []
            sensitive = str(item.get("sensitive_attribute") or "").strip()
            sensitive_ok = (not sensitive) or sensitive in transformed.columns

            reasons: List[str] = []
            infrastructure_ok = transformed.shape[1] >= 2 and sensitive_ok
            overlap = bool(by_id or by_hash)
            policy_valid = eval_assess["audit_status"] == "PASS" and profile["n_classes"] >= 2
            if policy == "ordinal_quantile_max5":
                reasons.append(
                    "explicit pre-outcome derived ordinal classification task; original numeric target removed from model features"
                )
            else:
                reasons.append(eval_assess["audit_reason"])
            if overlap:
                reasons.append("dataset overlaps development/training evidence")
            if matched_ids:
                reasons.append("matched training dataset id(s): " + ", ".join(matched_ids))
            if not sensitive_ok:
                reasons.append(f"configured sensitive attribute {sensitive!r} is missing")
            if transformed.shape[1] < 2:
                reasons.append("fewer than two columns after task construction")

            primary_eligible = bool(infrastructure_ok and policy_valid and not overlap)
            if not infrastructure_ok:
                status = "FAIL"
                role = "invalid_input"
            elif overlap:
                status = "EXCLUDE"
                role = "development_overlap_control"
            elif not policy_valid:
                status = "FAIL"
                role = "task_policy_invalid"
            elif policy == "ordinal_quantile_max5":
                status = "PASS_DERIVED"
                role = "primary_heldout_derived_classification"
            else:
                status = "PASS"
                role = "primary_heldout_native_classification"

            positive = 4 if policy == "ordinal_quantile_max5" else parse_positive_label(item.get("positive_label_json"))
            base.update(profile)
            base.update({
                "rows_in_file": int(len(df)),
                "columns_in_file": int(df.shape[1]),
                "resolved_target": source_target,
                "source_target_dtype": source_assess["target_dtype"],
                "source_target_unique": int(source_assess["target_unique"]),
                "source_task_assessment": source_assess["task_assessment"],
                "source_target_audit_status": source_assess["audit_status"],
                "evaluation_target": evaluation_target,
                "evaluation_target_unique": int(transformed[evaluation_target].nunique(dropna=True)),
                "task_assessment": eval_assess["task_assessment"],
                "target_dtype": eval_assess["target_dtype"],
                "target_unique": int(eval_assess["target_unique"]),
                "target_unique_fraction": float(eval_assess["target_unique_fraction"]),
                "audit_status": eval_assess["audit_status"],
                "audit_reason": eval_assess["audit_reason"],
                "target_distribution_top20_json": eval_assess["target_distribution_top20_json"],
                "target_is_last_column": bool(str(df.columns[-1]) == source_target),
                "sensitive_attribute_present": bool(sensitive_ok),
                "positive_label_resolved_json": canonical_json(positive),
                "overlap_with_training_id": bool(by_id),
                "overlap_with_training_sha256": bool(by_hash),
                "training_overlap_matched_dataset_ids": ",".join(matched_ids),
                "primary_heldout_eligible": primary_eligible,
                "evaluation_role": role,
                "preflight_status": status,
                "preflight_reason": "; ".join(reasons),
                "task_transform_payload_json": canonical_json(transform_payload),
                "task_transform_sha256": str(transform_payload.get("transform_sha256") or ""),
            })
        except Exception as exc:
            base.update({
                "preflight_status": "FAIL",
                "evaluation_role": "audit_error",
                "preflight_reason": f"{type(exc).__name__}: {exc}",
            })
        rows.append(base)

    return pd.DataFrame(rows)

def resolution_template(audit: pd.DataFrame) -> pd.DataFrame:
    """Record the explicit task policy used for every Phase-18 dataset."""
    cols = [
        "dataset_id", "filename", "cohort", "resolved_target", "source_target_unique",
        "source_task_assessment", "task_policy", "evaluation_target", "evaluation_target_unique",
        "task_transform_sha256", "preflight_status", "evaluation_role",
        "primary_heldout_eligible", "preflight_reason",
    ]
    available = [c for c in cols if c in audit.columns]
    out = audit[available].copy()
    out["final_decision"] = "use_as_predeclared"
    out["decision_rationale"] = out.apply(
        lambda r: (
            "Use source classification target unchanged."
            if str(r.get("task_policy")) == "native_classification"
            else "Use frozen ordinal quantile derived classification target; original source outcome is not a model feature."
        ), axis=1
    )
    return out

def validate_audit_for_freeze(audit: pd.DataFrame) -> None:
    if len(audit) != EXPECTED_DATASETS:
        raise RuntimeError(f"Audit contains {len(audit)} rows; expected {EXPECTED_DATASETS}.")
    overlap_sources = {
        str(value).strip()
        for value in audit.get("training_overlap_sources", pd.Series(dtype=str)).tolist()
        if str(value).strip()
    }
    if not overlap_sources:
        raise RuntimeError(
            "Cannot freeze Phase 18 because no local V2 development snapshot was available to verify separation."
        )
    blockers = audit[~audit["primary_heldout_eligible"].astype(bool)]
    if not blockers.empty:
        cols = ["dataset_id", "filename", "preflight_status", "evaluation_role", "preflight_reason"]
        raise RuntimeError(
            "Phase-18 31-dataset freeze still has blockers:\n" + blockers[cols].to_string(index=False)
        )
    if audit["sha256"].isna().any() or audit["sha256"].duplicated().any():
        raise RuntimeError("Every held-out dataset must have a unique non-null SHA256.")
    if (audit["overlap_with_training_id"].astype(bool) | audit["overlap_with_training_sha256"].astype(bool)).any():
        raise RuntimeError("Development overlap is not allowed in the 31-dataset primary protocol.")
    allowed = {"native_classification", "ordinal_quantile_max5"}
    if not set(audit["task_policy"].astype(str)).issubset(allowed):
        raise RuntimeError("Frozen audit contains an unknown task policy.")

def per_dataset_preference_seed(dataset_id: str, global_seed: int = PREFERENCE_GLOBAL_SEED) -> int:
    digest = hashlib.sha256(f"{global_seed}|{dataset_id}|phase18".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def build_preference_manifest(dataset_rows: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, ds in dataset_rows.iterrows():
        dataset_id = str(ds["dataset_id"])
        seed = per_dataset_preference_seed(dataset_id)
        rng = np.random.default_rng(seed)
        for pref_id in range(PREFERENCES_PER_DATASET):
            k_prime = int(rng.integers(1, 5))
            active = list(rng.choice(np.array(OBJECTIVES), size=k_prime, replace=False))
            weights = rng.dirichlet(np.ones(k_prime, dtype=float))
            w = {name: 0.0 for name in OBJECTIVES}
            for name, value in zip(active, weights):
                w[str(name)] = float(value)
            rows.append({
                "dataset_id": dataset_id,
                "cohort": str(ds["cohort"]),
                "preference_id": int(pref_id),
                "preference_seed": int(seed),
                "k_prime": int(k_prime),
                "active_objectives": ",".join(sorted(str(x) for x in active)),
                "w_accuracy": w["accuracy"],
                "w_runtime": w["runtime"],
                "w_energy": w["energy"],
                "w_co2": w["co2"],
            })
    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_PREFERENCE_CASES:
        raise RuntimeError(f"Preference manifest has {len(out)} rows; expected {EXPECTED_PREFERENCE_CASES}.")
    if out[["dataset_id", "preference_id"]].duplicated().any():
        raise RuntimeError("Preference manifest keys are not unique.")
    sums = out[["w_accuracy", "w_runtime", "w_energy", "w_co2"]].sum(axis=1)
    if not np.allclose(sums.to_numpy(float), 1.0, atol=1e-12):
        raise RuntimeError("Preference weights do not sum to one.")
    return out


def build_task_manifest(dataset_rows: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    task_id = 0
    for _, ds in dataset_rows.iterrows():
        positive = ds.get("positive_label_resolved_json") or ds.get("positive_label_json") or "1"
        for framework in FRAMEWORKS:
            for rep_index, seed in enumerate(SEEDS, start=1):
                rows.append({
                    "task_id": task_id,
                    "dataset_id": str(ds["dataset_id"]),
                    "filename": str(ds["filename"]),
                    "cohort": str(ds["cohort"]),
                    "dataset_sha256": str(ds["sha256"]),
                    "target": str(ds["evaluation_target"]),
                    "source_target": str(ds["resolved_target"]),
                    "task_policy": str(ds.get("task_policy") or "native_classification"),
                    "task_transform_payload_json": str(ds.get("task_transform_payload_json") or ""),
                    "task_transform_sha256": str(ds.get("task_transform_sha256") or ""),
                    "sensitive_attribute": str(ds.get("sensitive_attribute") or ""),
                    "positive_label_json": ("4" if str(ds.get("task_policy")) == "ordinal_quantile_max5" else str(positive)),
                    "dataset_family": str(ds.get("dataset_family") or "unknown"),
                    "source_type": str(ds.get("source_type") or "unknown"),
                    "drift_type": str(ds.get("drift_type") or "unknown"),
                    "framework": framework,
                    "seed": int(seed),
                    "sustainability_repetition_id": int(rep_index),
                    "sustainability_repetitions_planned": len(SEEDS),
                    "max_samples": MAX_SAMPLES,
                    "window_size": WINDOW_SIZE,
                    "time_budget_sec": TIME_BUDGET_SEC,
                    "track_sustainability": True,
                    "protocol_id": PROTOCOL_ID,
                })
                task_id += 1
    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_RUNS:
        raise RuntimeError(f"Task manifest has {len(out)} rows; expected {EXPECTED_RUNS}.")
    key_cols = ["dataset_id", "framework", "seed"]
    if out[key_cols].duplicated().any():
        raise RuntimeError("Task manifest dataset/framework/seed keys are not unique.")
    if out["task_id"].tolist() != list(range(EXPECTED_RUNS)):
        raise RuntimeError(f"Task IDs must be exactly 0..{EXPECTED_RUNS - 1}.")
    return out


def frozen_audit_path() -> Path:
    return FROZEN_DIR / "dataset_manifest_frozen.tsv"


def task_manifest_path() -> Path:
    return FROZEN_DIR / "hpc_task_manifest_465.tsv"


def preference_manifest_path() -> Path:
    return FROZEN_DIR / "preference_manifest_3100.csv"


def protocol_lock_path() -> Path:
    return FROZEN_DIR / "protocol_lock.json"


def load_frozen_dataset_manifest() -> pd.DataFrame:
    path = frozen_audit_path()
    if not path.exists():
        raise FileNotFoundError(f"Phase-18 protocol is not frozen yet: {path}")
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    if len(df) != EXPECTED_DATASETS:
        raise RuntimeError("Frozen dataset manifest does not contain 31 datasets.")
    return df


def load_task_manifest() -> pd.DataFrame:
    path = task_manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen task manifest: {path}")
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    if len(df) != EXPECTED_RUNS:
        raise RuntimeError("Frozen task manifest does not contain 465 tasks.")
    return df


def load_preference_manifest() -> pd.DataFrame:
    path = preference_manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen preference manifest: {path}")
    df = pd.read_csv(path)
    if len(df) != EXPECTED_PREFERENCE_CASES:
        raise RuntimeError("Frozen preference manifest does not contain 3,100 cases.")
    return df


def require_no_outcomes_before_freeze() -> None:
    if RUNS_DIR.exists() and any(RUNS_DIR.rglob("result.json")):
        raise RuntimeError(
            "Held-out result files already exist. Refusing to create/change the protocol freeze after outcomes. "
            "Archive/remove the experimental outputs only if this is genuinely a fresh pre-outcome protocol."
        )


def checksum_manifest(paths: Sequence[Path]) -> Dict[str, str]:
    return {str(Path(p).relative_to(ROOT)): sha256_file(Path(p)) for p in paths}


def active_recommender_manifest() -> Optional[Path]:
    marker = ROOT / "data" / "meta" / "active_recommender_v2.txt"
    if not marker.exists():
        return None
    rel = marker.read_text(encoding="utf-8").strip()
    if not rel:
        return None
    path = ROOT / "data" / "meta" / rel
    return path if path.exists() else None


def freeze_protocol(audit: pd.DataFrame, confirmation: str) -> Dict[str, Any]:
    if confirmation != FREEZE_CONFIRMATION:
        raise RuntimeError(
            f"Freeze confirmation must be exactly {FREEZE_CONFIRMATION!r}. Review dataset_audit.tsv first."
        )
    require_no_outcomes_before_freeze()
    validate_audit_for_freeze(audit)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    frozen = audit.copy()
    frozen.to_csv(frozen_audit_path(), sep="\t", index=False)

    prefs = build_preference_manifest(frozen)
    prefs.to_csv(preference_manifest_path(), index=False)

    tasks = build_task_manifest(frozen)
    tasks.to_csv(task_manifest_path(), sep="\t", index=False)

    recommender_manifest = active_recommender_manifest()
    environment_path = FROZEN_DIR / "environment_snapshot.json"
    write_json_atomic(environment_path, capture_environment_snapshot())

    files_to_hash = [
        CONFIG_PATH, INVENTORY_PATH, frozen_audit_path(), preference_manifest_path(),
        task_manifest_path(), environment_path, *phase18_source_files(),
    ]
    if recommender_manifest is not None:
        files_to_hash.append(recommender_manifest)

    lock = {
        "schema_version": "2.0",
        "phase": 18,
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "git_head": current_git_head(),
        "git_worktree_dirty": bool((current_git_status() or "").strip()),
        "dataset_directory": str(DATASET_DIR.relative_to(ROOT)),
        "expected_counts": {
            "datasets": EXPECTED_DATASETS,
            "historical_datasets": EXPECTED_HISTORICAL,
            "extension_datasets": EXPECTED_EXTENSION,
            "frameworks": len(FRAMEWORKS),
            "seeds": len(SEEDS),
            "framework_runs": EXPECTED_RUNS,
            "aggregated_dataset_framework_rows": EXPECTED_AGGREGATED,
            "preferences_per_dataset": PREFERENCES_PER_DATASET,
            "preference_cases": EXPECTED_PREFERENCE_CASES,
        },
        "execution": {
            "evaluation": "prequential_test_then_train",
            "max_samples": MAX_SAMPLES,
            "window_size": WINDOW_SIZE,
            "time_budget_sec_per_framework": TIME_BUDGET_SEC,
            "seeds": list(SEEDS),
            "frameworks": list(FRAMEWORKS),
            "track_sustainability": True,
            "sensitive_feature_policy": "audit_only",
            "xai_method": "auto",
        },
        "preference_protocol": {
            "objectives": list(OBJECTIVES),
            "k_prime": [1, 2, 3, 4],
            "preferences_per_dataset": PREFERENCES_PER_DATASET,
            "global_seed": PREFERENCE_GLOBAL_SEED,
            "per_dataset_seeded": True,
            "weights": "Dirichlet(1) over uniformly selected active-objective subset",
        },
        "environment_snapshot": str(environment_path.relative_to(ROOT)),
        "source_files_hashed": int(len(phase18_source_files())),
        "recommender_manifest": (
            str(recommender_manifest.relative_to(ROOT)) if recommender_manifest else None
        ),
        "recommender_manifest_sha256": (
            sha256_file(recommender_manifest) if recommender_manifest else None
        ),
        "checksums": checksum_manifest(files_to_hash),
        "scientific_guardrails": [
            "The frozen V2 recommender must not be retrained or tuned after held-out outcomes are observed.",
            "The historical 23 and extension 8 cohorts remain identifiable in every downstream table.",
            "The 3,100 preference cases are derived after the 465 framework runs; they are not additional framework executions.",
            "Primary preference evaluation ranks from frozen recommender predictions only; the historical CIKM-compatible real-min/max analysis is secondary.",
            "Sixteen source outcomes are explicitly converted to frozen ordinal quantile classification tasks before outcomes; they must be reported as derived tasks, not original regression targets.",
        ],
    }
    write_json_atomic(protocol_lock_path(), lock)
    write_text_atomic(
        FROZEN_DIR / "FROZEN_DO_NOT_EDIT.txt",
        "Phase-18 protocol frozen. Do not edit manifests after held-out outcomes are generated.\n"
        f"Protocol: {PROTOCOL_ID}\nCreated: {lock['created_utc']}\n",
    )
    return lock


def verify_frozen_checksums() -> None:
    lock_path = protocol_lock_path()
    if not lock_path.exists():
        raise FileNotFoundError("Protocol lock is missing. Run the reviewed freeze step first.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    bad = []
    for rel, expected in (lock.get("checksums") or {}).items():
        path = ROOT / rel
        if not path.exists():
            bad.append((rel, "missing", expected))
            continue
        actual = sha256_file(path)
        if actual != expected:
            bad.append((rel, actual, expected))
    if bad:
        text = "\n".join(f"{rel}: actual={actual} expected={expected}" for rel, actual, expected in bad)
        raise RuntimeError("Frozen Phase-18 checksum verification failed:\n" + text)


def audit_summary(audit: pd.DataFrame) -> Dict[str, Any]:
    status_counts = {
        str(k): int(v)
        for k, v in audit["preflight_status"].astype(str).value_counts().to_dict().items()
    }
    role_counts = {
        str(k): int(v)
        for k, v in audit["evaluation_role"].astype(str).value_counts().to_dict().items()
    }
    overlap = audit[
        audit["overlap_with_training_id"].astype(bool)
        | audit["overlap_with_training_sha256"].astype(bool)
    ]
    unresolved = audit[~audit["primary_heldout_eligible"].astype(bool)]
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "rows": int(len(audit)),
        "status_counts": status_counts,
        "role_counts": role_counts,
        "primary_heldout_eligible": int(audit["primary_heldout_eligible"].astype(bool).sum()),
        "primary_heldout_not_eligible": int((~audit["primary_heldout_eligible"].astype(bool)).sum()),
        "development_overlap_count": int(len(overlap)),
        "development_overlaps": overlap[
            ["dataset_id", "filename", "training_overlap_matched_dataset_ids"]
        ].to_dict(orient="records"),
        "cohorts": {str(k): int(v) for k, v in audit["cohort"].value_counts().to_dict().items()},
        "unresolved_dataset_ids": [str(v) for v in unresolved["dataset_id"].tolist()],
        "targets": audit[
            [
                "dataset_id", "resolved_target", "target_unique", "task_assessment",
                "preflight_status", "evaluation_role", "primary_heldout_eligible",
            ]
        ].to_dict(orient="records"),
    }

