"""Current dashboard evidence. Frozen V2 study services remain independent."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBJECTIVES = ("accuracy", "runtime", "energy", "co2")
TARGETS = {"accuracy": "accuracy", "runtime": "runtime_sec", "energy": "energy_kwh", "co2": "co2_kg"}


def bundle_directory():
    return Path(os.environ.get("AWAREML_RECOMMENDER_V3_BUNDLE") or
                ROOT / "artifacts/recommender_v3/primary_prefix").expanduser().resolve()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _inside(directory, relative):
    path = (directory / relative).resolve()
    if directory not in path.parents:
        raise ValueError("Artifact path leaves the bundle directory.")
    return path


def bundle_status(directory=None):
    directory = Path(directory).resolve() if directory else bundle_directory()
    status = {"ready": False, "state": "training_pending", "directory": str(directory),
              "message": "Model training pending. The revised benchmark, collection and nested validation must finish before a trained bundle is available."}
    path = directory / "manifest.json"
    if not path.exists():
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = payload["frozen"]
        if _digest(manifest) != payload["bundle_sha256"]:
            raise ValueError("Bundle manifest integrity failure.")
        if manifest["schema"] != "awareml_meta_recommender_v3" or manifest["feature_set"] != "prefix":
            raise ValueError("The dashboard requires the prefix-only deployment bundle, not a metadata ablation model.")
        forbidden = {"n_samples_dataset", "dataset_family", "source_type", "drift_type"}
        if forbidden.intersection(manifest["features"]):
            raise ValueError("Deployment features contain full-stream or dataset metadata.")
        trained_version = manifest["environment"]["packages"]["scikit-learn"]
        if trained_version != importlib.metadata.version("scikit-learn"):
            raise ValueError("Use the training scikit-learn version to load the models.")
        files = manifest["file_hashes"]
        required = {"development_aggregates.csv", "training_collection.json"}
        required.update(entry["file"] for entry in manifest["models"].values())
        if not required.issubset(files) or set(manifest["models"]) != set(OBJECTIVES):
            raise ValueError("Training evidence or objective models are incomplete.")
        for name, expected in files.items():
            if _sha(_inside(directory, name)) != expected:
                raise ValueError("Artifact changed: " + name)
        collection = json.loads((directory / "training_collection.json").read_text(encoding="utf-8"))
        frozen = collection["frozen"]
        if _digest(frozen) != collection["collection_sha256"] or collection["collection_sha256"] != manifest["collection_sha256"]:
            raise ValueError("Training collection integrity failure.")
        if frozen["mode"] != "production" or frozen.get("hardware_issues"):
            raise ValueError("Training collection is not valid production evidence.")
        protocol = manifest["protocol"]
        if len(protocol["seeds"]) < 5 or len(set(protocol["seeds"])) != len(protocol["seeds"]):
            raise ValueError("The revised deployment bundle requires at least five distinct seeds.")
        if protocol["scientific_time_limit_sec"] is not None:
            raise ValueError("The bundle uses a scientific wall-clock cutoff.")
        if int(manifest["profile_size"]) != int(protocol["profile_size"]) or not 1 <= int(manifest["profile_size"]) <= 1000:
            raise ValueError("Invalid observed-prefix calibration size.")
        frameworks = list(manifest["frameworks"])
        if not frameworks or len(set(frameworks)) != len(frameworks):
            raise ValueError("Invalid framework portfolio.")
        import pandas as pd
        frame = pd.read_csv(directory / "development_aggregates.csv")
        ids = set(map(str, manifest["training_dataset_ids"]))
        if set(frame.cohort.astype(str)) != {"dev"} or set(frame.dataset_id.astype(str)) != ids:
            raise ValueError("The historical prior accepts the declared development datasets only.")
        if set(frame.framework) != set(frameworks) or frame.duplicated(["dataset_id", "framework"]).any():
            raise ValueError("Development framework rows are incomplete or duplicated.")
        if len(frame) != len(ids) * len(frameworks) or set(frame.seed_count.astype(int)) != {len(protocol["seeds"])}:
            raise ValueError("Development evidence does not contain all declared frameworks and seeds.")
        for key in TARGETS.values():
            values = pd.to_numeric(frame[key + "_mean"], errors="raise")
            if not all(math.isfinite(float(v)) and float(v) >= 0 for v in values):
                raise ValueError("Invalid development objective outcomes.")
        if (frame.accuracy_mean > 1).any() or (frame.energy_kwh_mean <= 0).any():
            raise ValueError("Invalid development accuracy or energy.")
        return {**status, "ready": True, "state": "ready", "message": "Validated prefix-only model bundle available.",
                "evidence_id": _sha(path), "manifest": manifest, "development_datasets": len(ids),
                "training_rows": len(frame), "underlying_runs": int(frame.seed_count.sum()),
                "framework_count": len(frameworks), "seeds": list(protocol["seeds"])}
    except Exception as exc:
        return {**status, "state": "invalid_bundle", "message": "Model bundle unavailable: " + str(exc)}


def require_bundle(directory=None):
    status = bundle_status(directory)
    if not status["ready"]:
        raise RuntimeError(status["message"])
    return status


def load_current_recommender(directory=None):
    status = require_bundle(directory)
    from awareml.protocol_v3.service import V3Recommender
    return V3Recommender(status["directory"])


def prefix_from_dataframe(frame, target, size=1000):
    import pandas as pd
    from awareml.protocol_v3.profile import prefix_profile
    if frame is None or not target or target not in frame.columns:
        raise ValueError("Load a dataset and select its classification target.")
    # Re-infer dtypes from the observed prefix, matching the frozen V3 service.
    prefix = pd.read_csv(StringIO(frame.iloc[:int(size)].to_csv(index=False)))
    return prefix_profile(prefix, target, int(size))


def normalize_weights(weights=None):
    values = {k: float((weights or {}).get(k, 0)) for k in OBJECTIVES}
    if any(not math.isfinite(v) or v < 0 for v in values.values()) or sum(values.values()) <= 0:
        raise ValueError("Select at least one positive objective weight.")
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def historical_prior(weights, directory=None):
    """Mean within-dataset utility of seed-aggregated DEVELOPMENT outcomes."""
    status = require_bundle(directory)
    import numpy as np
    import pandas as pd
    from awareml.protocol_v3.common import utilities
    weights = normalize_weights(weights)
    frame = pd.read_csv(Path(status["directory"]) / "development_aggregates.csv")
    frameworks = status["manifest"]["frameworks"]
    scores = {f: [] for f in frameworks}
    wins = dict.fromkeys(frameworks, 0)
    for _, part in frame.groupby("dataset_id", sort=True):
        part = part.set_index("framework").loc[frameworks]
        values = part[[TARGETS[k] + "_mean" for k in OBJECTIVES]].to_numpy(dtype=float)
        utility = utilities(values, weights)
        for index, framework in enumerate(frameworks):
            scores[framework].append(float(utility[index]))
            wins[framework] += int(abs(float(utility[index] - utility.max())) <= 1e-12)
    ranked = pd.DataFrame([{"framework": f, "historical_utility": float(np.mean(scores[f])),
                            "utility_std_across_datasets": float(np.std(scores[f], ddof=1)) if len(scores[f]) > 1 else None,
                            "win_count": wins[f], "support_datasets": len(scores[f])} for f in frameworks])
    ranked = ranked.sort_values("historical_utility", ascending=False, kind="stable").reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["framework_label"] = ranked.framework.replace({"ChaCha": "ChaCha-OVR (supplementary adaptation)"})
    return ranked, status


def evaluation_evidence(kind, directory=None):
    """Only display evaluations explicitly linked to this exact trained bundle."""
    if kind not in ("development", "external"):
        raise ValueError("Unknown evaluation role.")
    status = require_bundle(directory)
    bundle = Path(status["directory"])
    variable = "AWAREML_V3_" + kind.upper() + "_EVALUATION"
    default = bundle.parent / (bundle.name + "_" + kind + "_evaluation")
    path = Path(os.environ.get(variable) or default).resolve()
    if not (path / "evaluation.json").exists():
        return None
    data = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
    if data["bundle_manifest_sha256"] != status["evidence_id"] or data["portfolio"] != status["manifest"]["portfolio"]:
        raise ValueError("Evaluation does not match the active bundle.")
    expected_role = {"development": "nested_LODO_development_four_predeclared_profiles",
                     "external": "revised_external_evaluation_of_previously_evaluated_cohort"}[kind]
    if data["role"] != expected_role:
        raise ValueError("Evaluation role does not match its panel.")
    for name, expected in data["file_hashes"].items():
        if _sha(_inside(path, name)) != expected:
            raise ValueError("Evaluation artifact changed: " + name)
    return {"directory": str(path), "metadata": data}


def sync_copilot_evidence(state, status=None):
    status = status or bundle_status()
    frame, target = state.get("dataset"), state.get("target")
    size = int(status.get("manifest", {}).get("profile_size", 1000))
    prefix_id = "no_dataset"
    if frame is not None and target in frame.columns:
        prefix_id = hashlib.sha256((str(target) + frame.iloc[:size].to_csv(index=False)).encode()).hexdigest()
    token = str(status.get("evidence_id") or status["state"]) + ":" + prefix_id
    if state.get("copilot_current_evidence_id") != token:
        old = state.get("copilot_proposal")
        data = old.model_dump() if hasattr(old, "model_dump") else dict(old or {})
        if data.get("interpretation"):
            state["copilot_context_free_interpretation"] = data["interpretation"]
            state["copilot_context_free_meta"] = {"goal_parse": (state.get("copilot_meta") or {}).get("goal_parse", {})}
        for key in ("copilot_proposal", "copilot_ranked", "copilot_evidence", "copilot_meta", "copilot_review",
                    "copilot_auto_historical_result", "copilot_auto_historical_ranking", "copilot_prefix_profile"):
            state.pop(key, None)
        state["copilot_current_evidence_id"] = token
    return status


def propose_current(service, state, goal, use_llm):
    status = require_bundle()
    profile = prefix_from_dataframe(state["dataset"], state["target"], status["manifest"]["profile_size"])
    state["copilot_prefix_profile"] = profile
    return service.propose_from_profile(goal=goal, profile=profile,
        sensitive_attribute=state.get("sensitive"), current_config=None, use_llm=use_llm)


def apply_current_objective_override(state, corrected):
    from .pre14_usability import CANONICAL_OBJECTIVES, _set_interpretation_objectives, weights_for_objectives
    selected = [v for v in CANONICAL_OBJECTIVES if v in set(corrected or [])]
    if not selected:
        raise ValueError("Select at least one priority.")
    weights = weights_for_objectives(selected)
    state["copilot_human_corrected_objectives"] = selected
    state["copilot_human_corrected_weights"] = weights
    if state.get("copilot_context_free_interpretation") is not None:
        state["copilot_context_free_interpretation"] = _set_interpretation_objectives(
            state["copilot_context_free_interpretation"], selected)
    proposal = state.get("copilot_proposal")
    if proposal is None:
        return None
    profile = state.get("copilot_prefix_profile")
    if not profile:
        raise ValueError("Regenerate the proposal from the observed prefix before reranking.")
    ranked, meta = load_current_recommender().recommend_profile(profile, weights=weights)
    from awareml.llm.configuration import synthesize_configuration
    from awareml.llm.evidence import build_before_evidence
    new = proposal.model_copy(deep=True) if hasattr(proposal, "model_copy") else copy.deepcopy(proposal)
    new.interpretation = _set_interpretation_objectives(new.interpretation, selected)
    new.proposed_config, warnings = synthesize_configuration(new.interpretation, ranked, sensitive_attribute=state.get("sensitive"))
    new.ml_recommender_framework = str(ranked.iloc[0].framework)
    new.ml_recommender_rank = int(ranked.iloc[0]["rank"])
    new.ml_recommender_utility = float(ranked.iloc[0].utility)
    new.rationale = "The trained prefix-only models reranked the same observed prefix under your corrected priorities."
    new.warnings = list(new.warnings or []) + list(warnings) + list(meta.get("warnings", []))
    state.update(copilot_proposal=new, copilot_ranked=ranked, copilot_review=None,
                 copilot_evidence=build_before_evidence(new.interpretation, profile, ranked, meta))
    state["copilot_meta"] = {**(state.get("copilot_meta") or {}), "ranking": meta,
                              "human_objective_override": {"selected_objectives": selected, "weights": weights}}
    return new.ml_recommender_framework
