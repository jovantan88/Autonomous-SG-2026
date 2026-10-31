"""Unit tests for the tier rule, the probability/decision combination and feature causality."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import build_features, md_tier, pc_tier, risk_tier  # noqa: E402
from predict import RiskModel, combine  # noqa: E402

THR = {"p_critical": 0.2, "p_elevated": 0.3}


@pytest.mark.parametrize("log_pc, md, expected", [
    (-30, 30000, 0),      # far and negligible Pc
    (-6.0, 30000, 1),     # Pc exactly 1e-6 -> Elevated (inclusive)
    (-6.01, 30000, 0),
    (-4.0, 30000, 2),     # Pc exactly 1e-4 -> Critical (inclusive)
    (-30, 5000, 1),       # brief: Elevated is 1.0-5.0 km inclusive, Safe needs > 5.0 km
    (-30, 5001, 0),
    (-30, 1000, 1),       # 1.0 km -> Elevated
    (-30, 999, 2),        # < 1 km -> Critical
    (-5, 500, 2),         # worst of both rules
])
def test_risk_tier(log_pc, md, expected):
    assert risk_tier([log_pc], [md])[0] == expected


def test_component_rules():
    assert list(md_tier([500, 3000, 9000])) == [2, 1, 0]
    assert list(pc_tier([-3, -5, -7])) == [2, 1, 0]


def _combine(p_elev, p_crit, md, bound=None, lo=None, hi=None):
    n = len(md)
    med = np.full(n, -8.0)
    lo = np.full(n, -10.0) if lo is None else np.asarray(lo, float)
    hi = np.full(n, -7.0) if hi is None else np.asarray(hi, float)
    return combine(med, lo, hi, np.asarray(p_elev, float), np.asarray(p_crit, float),
                   np.asarray(md, float), THR, max_pc_bound=bound)


def test_probabilities_sum_to_one_and_respect_distance_rule():
    out = _combine([0.1, 0.6, 0.9], [0.05, 0.1, 0.5], [20000, 3000, 500])
    p = out[["p_safe", "p_elevated", "p_critical"]].to_numpy()
    assert np.allclose(p.sum(1), 1)
    assert p[1, 0] == 0                      # within 5 km can never be Safe
    assert list(p[2]) == [0, 0, 1]           # within 1 km is Critical with certainty
    assert list(out.tier) == [0, 1, 2]


def test_model_cannot_override_hard_distance_rule():
    out = _combine([0.0], [0.0], [800])
    assert out.tier[0] == 2 and "under 1 km" in out.reasons[0][0]


def test_thresholds_drive_pc_escalation():
    out = _combine([0.35, 0.1], [0.25, 0.1], [20000, 20000])
    assert list(out.tier) == [2, 0]


def test_max_pc_bound_forces_review_but_not_tier():
    out = _combine([0.05, 0.05], [0.01, 0.01], [20000, 20000], bound=[-3.5, -8.0])
    assert list(out.tier) == [0, 0]
    assert list(out.needs_review) == [True, False]
    assert any(r.startswith("Worst-case probability") for r in out.reasons[0])


def test_head_disagreement_is_flagged_not_hidden():
    # classifier says Critical but the whole Pc interval sits below 1e-4
    a = combine(np.array([-9.0]), np.array([-12.0]), np.array([-8.0]), np.array([0.6]), np.array([0.5]),
                np.array([20000.0]), THR)
    # regressor median above 1e-4 but classifier says P(Critical) is small
    b = combine(np.array([-3.8]), np.array([-6.0]), np.array([-3.0]), np.array([0.9]), np.array([0.02]),
                np.array([3000.0]), THR)
    for out in (a, b):
        assert out.heads_disagree[0] and out.needs_review[0]
        assert any(r.startswith("Model parts disagree") for r in out.reasons[0])
    assert a.tier[0] == 2  # decision unchanged, only flagged


def test_straddling_interval_flags_review():
    out = _combine([0.1], [0.1], [20000], lo=[-5.0], hi=[-3.0])
    assert out.needs_review[0]


def test_history_features_are_causal():
    """Changing a later CDM must not change features of an earlier CDM."""
    base = pd.DataFrame({
        "event_id": [1, 1, 1], "time_to_tca": [5.0, 4.0, 3.0], "miss_distance": [9000.0, 8000.0, 7000.0],
        "relative_speed": [12000.0] * 3, "relative_position_r": [10.0, 20.0, 30.0],
        "relative_position_t": [8000.0, 7000.0, 6000.0], "relative_position_n": [100.0] * 3,
        "c_object_type": ["DEBRIS"] * 3, "c_sigma_r": [50.0, 40.0, 30.0], "t_sigma_r": [5.0] * 3,
        "max_risk_estimate": [-6.0, -5.5, -5.0], "F10": [80.0] * 3, "AP": [5.0] * 3,
    })
    changed = base.copy()
    changed.loc[2, ["miss_distance", "max_risk_estimate", "c_sigma_r"]] = [100.0, -2.0, 1.0]
    a, b = build_features(base), build_features(changed)
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])


def test_stub_model_runs_without_trained_artifacts(tmp_path):
    m = RiskModel.load(tmp_path)
    assert not m.trained
    df = pd.DataFrame({"time_to_tca": [1.0], "miss_distance": [700.0], "relative_speed": [10000.0],
                       "max_risk_estimate": [-3.0], "c_object_type": ["DEBRIS"], "F10": [90.0], "AP": [8.0]})
    out = m.assess_screening(df)
    assert out.tier[0] == 2
