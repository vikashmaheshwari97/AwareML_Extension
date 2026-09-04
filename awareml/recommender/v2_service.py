from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from awareml.engine.pareto_spec import CANONICAL_EPSILON
from .v2_models import model_feature_columns, sanitize_predictions
from .v2_profile import candidate_rows_from_profile, profile_from_dataframe_v2
from .v2_ranking import rank_candidates
from .v2_uncertainty import interval_for_prediction


FRAMEWORKS = (
    "AutoStreamML",
    "AutoClass",
    "EvoAutoML",
    "OAML",
    "ChaCha",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_active_v2_manifest(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    marker = base / "data" / "meta" / "active_recommender_v2.txt"
    if not marker.exists():
        raise FileNotFoundError(
            "Phase-6 V2 recommender is not frozen/active yet: {}".format(marker)
        )

    rel = marker.read_text(encoding="utf-8").strip()
    if not rel:
        raise RuntimeError("Active V2 recommender marker is empty.")

    manifest = base / "data" / "meta" / rel
    if not manifest.exists():
        raise FileNotFoundError(
            "Active V2 model manifest does not exist: {}".format(manifest)
        )
    return manifest


class V2Recommender:
    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        root: Optional[Path] = None,
    ):
        self.root = (
            Path(root) if root is not None else Path(__file__).resolve().parents[2]
        )
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else locate_active_v2_manifest(self.root)
        )
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        self.model_dir = self.manifest_path.parent
        self.models = {}
        self._load_models()

    def _load_models(self):
        model_entries = self.manifest.get("models") or {}
        for target, entry in model_entries.items():
            path = self.model_dir / entry["file"]
            expected = entry.get("sha256")
            if expected and _sha256(path) != expected:
                raise RuntimeError("Model checksum mismatch: {}".format(path.name))
            self.models[target] = joblib.load(path)

        expected_targets = {"accuracy", "runtime", "energy", "co2"}
        if set(self.models) != expected_targets:
            raise RuntimeError("V2 model bundle targets are incomplete.")

    def predict_profile(
        self,
        profile: Mapping[str, Any],
        coverage: float = 0.90,
    ) -> pd.DataFrame:
        rows = candidate_rows_from_profile(dict(profile), FRAMEWORKS)
        features = model_feature_columns(True)

        missing = [column for column in features if column not in rows.columns]
        if missing:
            raise ValueError("Profile is missing V2 features: {}".format(missing))

        calibration = self.manifest.get("uncertainty") or {}

        for target, model in self.models.items():
            pred = sanitize_predictions(target, model.predict(rows[features]))
            rows[target] = pred

            target_cal = calibration.get(target)
            if target_cal is None:
                raise RuntimeError(
                    "Missing uncertainty calibration for {}.".format(target)
                )

            lower = []
            upper = []
            for value in pred:
                lo, hi = interval_for_prediction(
                    float(value),
                    target,
                    target_cal,
                    coverage=coverage,
                )
                lower.append(lo)
                upper.append(hi)

            rows[target + "_lower"] = lower
            rows[target + "_upper"] = upper

        keep = [
            "framework",
            "accuracy",
            "accuracy_lower",
            "accuracy_upper",
            "runtime",
            "runtime_lower",
            "runtime_upper",
            "energy",
            "energy_lower",
            "energy_upper",
            "co2",
            "co2_lower",
            "co2_upper",
        ]
        return rows[keep].copy()

    def recommend_profile(
        self,
        profile: Mapping[str, Any],
        weights: Optional[Mapping[str, float]] = None,
        ranking_mode: str = "point",
        coverage: float = 0.90,
        epsilon: float = CANONICAL_EPSILON,
    ):
        candidates = self.predict_profile(profile, coverage=coverage)
        return rank_candidates(
            candidates,
            weights=weights,
            mode=ranking_mode,
            objective_correlations=(
                self.manifest.get("objective_correlations") or {}
            ),
            epsilon=epsilon,
        )

    def recommend_dataframe(
        self,
        df: pd.DataFrame,
        target: str,
        weights: Optional[Mapping[str, float]] = None,
        window_size: int = 1000,
        time_budget_sec: float = 60.0,
        dataset_family: str = "unknown",
        source_type: str = "unknown",
        drift_type: str = "unknown",
        ranking_mode: str = "point",
        coverage: float = 0.90,
        epsilon: float = CANONICAL_EPSILON,
    ):
        profile = profile_from_dataframe_v2(
            df,
            target=target,
            window_size=window_size,
            time_budget_sec=time_budget_sec,
            dataset_family=dataset_family,
            source_type=source_type,
            drift_type=drift_type,
        )
        ranked, meta = self.recommend_profile(
            profile,
            weights=weights,
            ranking_mode=ranking_mode,
            coverage=coverage,
            epsilon=epsilon,
        )
        meta["profile"] = profile
        return ranked, meta
