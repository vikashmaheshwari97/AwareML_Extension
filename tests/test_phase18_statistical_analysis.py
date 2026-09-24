from __future__ import print_function

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase18_statistical_analysis.py"
spec = importlib.util.spec_from_file_location("phase18_stats", str(SCRIPT))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_holm_is_monotone_in_sorted_p_order():
    p = [0.001, 0.03, 0.02, 0.5]
    adj = mod.holm_adjust(p)
    order = np.argsort(p)
    vals = adj[order]
    assert np.all(np.diff(vals) >= -1e-12)
    assert np.all((adj >= 0) & (adj <= 1))


def test_rank_biserial_orientation():
    assert mod.rank_biserial_paired([1, 2, 3]) == 1.0
    assert mod.rank_biserial_paired([-1, -2, -3]) == -1.0
    assert mod.rank_biserial_paired([0, 0, 0]) == 0.0


def test_bootstrap_ci_contains_constant():
    lo, hi = mod.bootstrap_ci([3.0] * 10, reps=1000)
    assert lo == 3.0
    assert hi == 3.0


def test_ground_truth_validation_31x5():
    rows = []
    for i in range(31):
        for fw in mod.EXPECTED_FRAMEWORKS:
            rows.append({"dataset_id": "d%02d" % i, "framework": fw})
    df = pd.DataFrame(rows)
    mod.validate_ground_truth(df)


def test_ground_truth_validation_rejects_duplicate():
    rows = []
    for i in range(31):
        for fw in mod.EXPECTED_FRAMEWORKS:
            rows.append({"dataset_id": "d%02d" % i, "framework": fw})
    df = pd.DataFrame(rows)
    df.iloc[-1] = df.iloc[-2]
    try:
        mod.validate_ground_truth(df)
    except RuntimeError:
        return
    raise AssertionError("duplicate dataset/framework row should fail")
