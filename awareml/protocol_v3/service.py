from __future__ import annotations

from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

from .common import LABELS, OBJECTIVES, ROOT, utilities, weights_vector
from .models import load_bundle, predict
from .profile import prefix_profile


class V3Recommender:
    def __init__(self, bundle_directory=None):
        self.directory = Path(bundle_directory) if bundle_directory else ROOT / "artifacts/recommender_v3/primary_prefix"
        self.manifest, self.models = load_bundle(self.directory)
        if self.manifest["feature_set"] != "prefix":
            raise ValueError("The deployment service uses the prefix-only bundle; metadata models are ablation evidence.")

    def recommend_dataframe(self, frame, target, weights=None, **unused):
        # Re-infer types using only prefix values even if the UI supplied a frame
        # whose pandas dtypes were originally inferred from a larger CSV.
        prefix = frame.iloc[:int(self.manifest["profile_size"])]
        prefix = pd.read_csv(StringIO(prefix.to_csv(index=False)))
        profile = prefix_profile(prefix, target, int(self.manifest["profile_size"]))
        return self.recommend_profile(profile, weights)

    def recommend_profile(self, profile, weights=None, **unused):
        if "n_samples_dataset" in profile:
            raise ValueError("A full-stream V2 profile cannot be used as a V3 prefix profile.")
        expected = int(self.manifest["profile_size"])
        if int(profile.get("profile_sample_size", 0)) > expected:
            raise ValueError("Profile exceeds the frozen calibration prefix.")
        w = weights_vector(weights if weights is not None else [1, 1, 1, 1])
        candidates = pd.DataFrame([{**profile, "framework": fw} for fw in self.manifest["frameworks"]])
        for target, model in self.models.items():
            candidates[target] = predict(model, candidates, target, False)
            radius = self.manifest["uncertainty"][target]["absolute_radius"]
            candidates[target + "_lower"] = np.maximum(0, candidates[target] - radius)
            candidates[target + "_upper"] = candidates[target] + radius
            if target == "accuracy":
                candidates[target + "_upper"] = np.minimum(1, candidates[target + "_upper"])
        candidates["utility"] = utilities(candidates[list(OBJECTIVES)].to_numpy(), w)
        candidates["framework_label"] = candidates.framework.map(LABELS)
        candidates = candidates.sort_values("utility", ascending=False, kind="stable").reset_index(drop=True)
        candidates["rank"] = np.arange(1, len(candidates) + 1)
        meta = {"recommender_version": "v3", "profile": profile, "weights": dict(zip(OBJECTIVES, w.tolist())),
                "portfolio": self.manifest["portfolio"], "protocol": self.manifest["protocol"]["protocol_id"],
                "evidence_role": "predicted_complete_common_prefix_outcomes_after_labeled_calibration_prefix",
                "uncertainty": "empirical_nested_residual_intervals_not_a_winner_probability",
                "warnings": ["Energy and CO2 may represent the same signal under a fixed carbon intensity."]}
        return candidates, meta
