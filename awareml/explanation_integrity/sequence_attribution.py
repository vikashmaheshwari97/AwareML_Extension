"""Permutation Shapley estimator for two fixed sequence-score targets."""
from __future__ import annotations

import numpy as np


def paired_shapley(score, n_features, permutations=32, seed=42):
    """score(frozenset(indices)) -> (decision score, explanation score).

    Shared permutations and cached coalitions reduce variance and evaluation cost.
    Empty/full target values are retained for an additivity residual check.
    """
    if not 2 <= n_features <= 64 or permutations < 2:
        raise ValueError("Use 2–64 features and at least two permutations.")
    cache = {}

    def evaluate(indices):
        key = frozenset(indices)
        if key not in cache:
            values = np.asarray(score(key), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError("Scorer must return two finite sequence scores.")
            cache[key] = values
        return cache[key]

    rng = np.random.default_rng(seed)
    phi = np.zeros((n_features, 2))
    for _ in range(permutations):
        subset = set()
        before = evaluate(subset)
        for i in rng.permutation(n_features):
            subset.add(int(i))
            after = evaluate(subset)
            phi[i] += after - before
            before = after
    phi /= permutations
    base, full = evaluate(set()), evaluate(set(range(n_features)))
    return {"decision_attributions": phi[:, 0].tolist(), "explanation_attributions": phi[:, 1].tolist(),
            "base_scores": base.tolist(), "full_scores": full.tolist(),
            "additivity_residual": (full - base - phi.sum(axis=0)).tolist(),
            "coalitions_evaluated": len(cache), "permutations": permutations, "seed": seed}
