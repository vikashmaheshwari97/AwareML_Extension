from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OBJECTIVES = ("accuracy", "runtime", "energy", "co2")
TARGETS = {"accuracy": "accuracy", "runtime": "runtime_sec", "energy": "energy_kwh", "co2": "co2_kg"}
FRAMEWORKS = ("AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha")
PRIMARY = FRAMEWORKS[:4]
LABELS = {"AutoStreamML": "AutoStreamML (AwareML port)", "AutoClass": "AutoClass (AwareML port)",
          "EvoAutoML": "EvoAutoML (isolated backend)", "OAML": "OAML online (River backend)",
          "ChaCha": "ChaCha-OVR (AwareML adaptation)"}
NUMERIC_FEATURES = ("profile_sample_size", "n_features", "n_numeric_features", "n_categorical_features",
                    "numeric_feature_fraction", "categorical_feature_fraction", "missing_fraction", "n_classes",
                    "majority_class_fraction", "minority_class_fraction", "class_imbalance_ratio", "class_entropy_normalized")
METADATA = ("dataset_family", "source_type", "drift_type")
SEEDS = (42, 43, 44, 45, 46)
ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(payload):
    return hashlib.sha256(json.dumps(clean(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_csv_sha(path):
    # Normalize line endings only; this is not a semantic row reorder or edit.
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for line in f:
            h.update(line.replace(b"\r\n", b"\n"))
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean(data), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_fingerprint(root=ROOT):
    root = Path(root)
    files = []
    for directory in ("awareml/protocol_v3", "awareml/frameworks", "awareml/workers"):
        for path in sorted((root / directory).rglob("*.py")):
            files.append((str(path.relative_to(root)).replace("\\", "/"), sha256(path)))
    for name in ("awareml/data.py", "scripts/awareml_v3.py"):
        path = root / name
        if path.exists():
            files.append((name, sha256(path)))
    return {"sha256": digest(files), "files": files}


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "river", "codecarbon", "psutil", "xgboost", "flaml", "vowpalwabbit"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_model": cpu or platform.machine(), "node": platform.node(), "packages": packages,
            "thread_limits": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def weights_vector(weights):
    if isinstance(weights, dict):
        a = np.asarray([weights.get(k, 0.0) for k in OBJECTIVES], dtype=float)
    else:
        a = np.asarray(weights, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
        raise ValueError("Use four finite nonnegative weights with positive total.")
    return a / a.sum()


def desirability(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
        raise ValueError("Expected finite candidate by four-objective outcomes.")
    spans = np.ptp(a, axis=0)
    out = np.full_like(a, 0.5)
    active = spans > 1e-12
    out[:, active] = (a[:, active] - a[:, active].min(axis=0)) / spans[active]
    out[:, 1:] = 1.0 - out[:, 1:]
    return out


def utilities(values, weights):
    return desirability(values).dot(weights_vector(weights))


def safe_path(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Path leaves its declared root: {}".format(relative))
    return result


def load_campaign(directory, verify_code=True, verify_streams=False):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "campaign.json")
    frozen = manifest["frozen"]
    if digest(frozen) != manifest["campaign_sha256"]:
        raise ValueError("Campaign manifest integrity check failed.")
    for name, expected in frozen["file_hashes"].items():
        if name.startswith("streams/") and not verify_streams:
            continue
        if sha256(safe_path(directory, name)) != expected:
            raise ValueError("Frozen file changed: {}".format(name))
    if verify_code and code_fingerprint()["sha256"] != frozen["code"]["sha256"]:
        raise ValueError("Execution source changed after campaign preparation; create a new campaign.")
    return manifest
