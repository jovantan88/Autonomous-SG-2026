"""Inference API used by the server, the dashboard export and the test-set scorer.

RiskModel.assess_cdms(df)       full allowed-column CDMs (with event history) -> assessments
RiskModel.assess_screening(df)  reduced inputs available from live screening feeds
                                (SOCRATES: no covariances / RTN geometry) -> assessments

The model only estimates log10 Pc. Turning that into an operational tier is done here with the
deterministic rule from the brief, so every decision can be traced to a rule and a probability.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import (MD_CRITICAL, MD_ELEVATED, PC_CRITICAL, PC_ELEVATED, TIERS, build_features,
                      md_tier, OBJECT_TYPES)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

SCREENING_COLUMNS = ["time_to_tca", "miss_distance", "relative_speed", "max_risk_estimate",
                     "c_object_type", "F10", "AP"]

# Default decision thresholds on P(Pc tier); overwritten by models/thresholds.json after training.
DEFAULT_THRESHOLDS = {"p_critical": 0.5, "p_elevated": 0.5}


def screening_features(df):
    """Features computable from a screening feed (no covariance / RTN geometry)."""
    md = df["miss_distance"].clip(lower=1.0)
    f = pd.DataFrame(index=df.index)
    f["time_to_tca"] = df["time_to_tca"]
    f["log_md"] = np.log10(md)
    f["relative_speed"] = df["relative_speed"]
    f["max_risk_estimate"] = df["max_risk_estimate"]
    f["log_hbr"] = (df["max_risk_estimate"] + 2 * np.log10(md) + np.log10(np.e)) / 2
    f["F10"] = df["F10"]
    f["AP"] = df["AP"]
    f["c_object_type"] = pd.Categorical(df["c_object_type"], categories=OBJECT_TYPES).codes
    return f


DISAGREE_PREFIX = "Model parts disagree"


def _fmt_pc(log_pc):
    if log_pc <= -29.9:
        return "~0"
    mant, exp = f"{10 ** log_pc:.1e}".split("e")
    return f"{mant}e{int(exp)}"  # 1.2e-5, not 1.2e-05


def _fmt_distance(metres):
    return f"{metres:,.0f} m" if metres < 1000 else f"{metres / 1000:.1f} km"


def combine(log_pc, lo, hi, p_elev, p_crit, miss_distance, thresholds, max_pc_bound=None):
    """Combine model Pc probabilities with the hard miss-distance rule.

    `max_pc_bound` (log10 of the worst-case Pc over covariance scaling) is a physical upper bound:
    when it reaches the Critical threshold the event can never be auto-cleared below Critical
    without an analyst looking at it."""
    m = md_tier(miss_distance)
    p_elev = np.maximum(p_elev, p_crit)
    q = np.stack([1 - p_elev, p_elev - p_crit, p_crit], axis=1)
    p_tier = q.copy()
    p_tier[m == 1] = np.stack([np.zeros((m == 1).sum()), q[m == 1, 0] + q[m == 1, 1], q[m == 1, 2]], 1)
    p_tier[m == 2] = [0.0, 0.0, 1.0]

    pc_dec = np.where(p_crit >= thresholds["p_critical"], 2,
                      np.where(p_elev >= thresholds["p_elevated"], 1, 0))
    tier = np.maximum(pc_dec, m)
    straddle_crit = (lo < PC_CRITICAL) & (hi >= PC_CRITICAL)
    straddle_elev = (lo < PC_ELEVATED) & (hi >= PC_ELEVATED)
    needs_review = (m < 2) & (straddle_crit | ((tier == 0) & straddle_elev))
    bound_alarm = np.zeros(len(tier), bool)
    if max_pc_bound is not None:
        bound_alarm = (tier < 2) & (np.asarray(max_pc_bound, float) >= PC_CRITICAL)
    needs_review = needs_review | bound_alarm
    # The quantile regressor and the threshold classifiers are separate heads; when they land on
    # different sides of the threshold that decides the tier, say so instead of hiding it.
    crit_contra = (m < 2) & (((pc_dec == 2) & (hi < PC_CRITICAL)) | ((pc_dec < 2) & (log_pc >= PC_CRITICAL)))
    elev_contra = (m == 0) & (((pc_dec >= 1) & (hi < PC_ELEVATED)) | ((pc_dec == 0) & (log_pc >= PC_ELEVATED)))
    heads_disagree = crit_contra | elev_contra
    needs_review = needs_review | heads_disagree

    reasons = []
    for i in range(len(tier)):
        r = []
        md = _fmt_distance(float(miss_distance[i]))
        if m[i] == 2:
            r.append(f"Miss distance {md} is under 1 km, so the event is Critical whatever the probability.")
        elif m[i] == 1:
            r.append(f"Miss distance {md} is between 1 and 5 km, so the event is at least Elevated.")
        else:
            r.append(f"Miss distance {md} is more than 5 km.")
        r.append(f"Estimated collision probability {_fmt_pc(log_pc[i])} "
                 f"(80% range {_fmt_pc(lo[i])} to {_fmt_pc(hi[i])}).")
        if pc_dec[i] == 2:
            r.append(f"{p_crit[i]:.0%} chance the probability reaches 1e-4, above the "
                     f"{thresholds['p_critical']:.0%} escalation threshold: Critical.")
        elif pc_dec[i] == 1:
            r.append(f"{p_elev[i]:.0%} chance the probability reaches 1e-6, above the "
                     f"{thresholds['p_elevated']:.0%} threshold: Elevated.")
        else:
            r.append(f"{p_elev[i]:.0%} chance the probability reaches 1e-6, below the "
                     f"{thresholds['p_elevated']:.0%} threshold.")
        if heads_disagree[i]:
            lim, p = ("1e-4", p_crit[i]) if crit_contra[i] else ("1e-6", p_elev[i])
            r.append(f"{DISAGREE_PREFIX}: the estimate is {_fmt_pc(log_pc[i])} but the classifier gives "
                     f"a {p:.0%} chance of {lim}. Review needed.")
        if bound_alarm[i]:
            r.append(f"Worst-case probability {_fmt_pc(float(max_pc_bound[i]))} is above 1e-4, "
                     "so Critical cannot be ruled out. Review needed.")
        elif needs_review[i] and not heads_disagree[i]:
            r.append("The 80% range crosses a tier limit. Review needed.")
        r.append(f"Result: {TIERS[tier[i]]}.")
        reasons.append(r)

    return pd.DataFrame({
        "log_pc": log_pc, "log_pc_lo": lo, "log_pc_hi": hi,
        "p_pc_elevated": p_elev, "p_pc_critical": p_crit,
        "p_safe": p_tier[:, 0], "p_elevated": p_tier[:, 1], "p_critical": p_tier[:, 2],
        "md_tier": m, "tier": tier, "needs_review": needs_review, "heads_disagree": heads_disagree,
        "reasons": reasons,
    })


class RiskModel:
    """Loads trained models from models/. Falls back to the isotropic-Gaussian physics estimate
    when no trained model is present (so the stack runs before training)."""

    def __init__(self, models=None, thresholds=None, meta=None):
        self.use_calibration = True
        self.models = models or {}
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.meta = meta or {}

    @property
    def trained(self):
        return bool(self.models)

    @classmethod
    def load(cls, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        models = {}
        try:
            import lightgbm as lgb
            for p in model_dir.glob("*.txt"):
                models[p.stem] = lgb.Booster(model_file=str(p))
        except ImportError:
            pass
        thr_path, meta_path = model_dir / "thresholds.json", model_dir / "meta.json"
        thresholds = json.loads(thr_path.read_text()) if thr_path.exists() else None
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        return cls(models, thresholds, meta)

    def _predict_set(self, prefix, X):
        """Returns (median, lo, hi, p_elev, p_crit) from the model set `prefix` (full|screen)."""
        m = self.models
        cols = self.meta.get(f"{prefix}_features", list(X.columns))
        X = X[cols]
        med = m[f"{prefix}_q50"].predict(X)
        lo = m[f"{prefix}_q10"].predict(X)
        hi = m[f"{prefix}_q90"].predict(X)
        p_elev = m[f"{prefix}_clf_elev"].predict(X)
        p_crit = m[f"{prefix}_clf_crit"].predict(X)
        cal = self.meta.get("prob_calibration", {}).get(prefix) if self.use_calibration else None
        if cal:  # isotonic maps fitted on out-of-fold predictions
            p_elev = np.interp(p_elev, cal["elev"]["x"], cal["elev"]["y"])
            p_crit = np.interp(p_crit, cal["crit"]["x"], cal["crit"]["y"])
        off = self.meta.get("conformal_offsets", {}).get(prefix, {})
        lo, hi = lo - off.get("lo", 0.0), hi + off.get("hi", 0.0)  # conformal widening (CQR)
        lo, hi = np.minimum(lo, med), np.maximum(hi, med)
        return np.clip(med, -30, 0), np.clip(lo, -30, 0), np.clip(hi, -30, 0), p_elev, p_crit

    def assess_cdms(self, df):
        df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
        X = build_features(df)
        if self.trained:
            med, lo, hi, pe, pc = self._predict_set("full", X)
            thr = self.thresholds.get("full", self.thresholds)
        else:
            med = X["log_pc_iso"].to_numpy()
            lo, hi = med - 2, np.minimum(med + 2, df["max_risk_estimate"].to_numpy())
            pe = (med >= PC_ELEVATED).astype(float)
            pc = (med >= PC_CRITICAL).astype(float)
            thr = DEFAULT_THRESHOLDS
        out = combine(med, lo, hi, pe, pc, df["miss_distance"].to_numpy(), thr,
                      max_pc_bound=df["max_risk_estimate"].to_numpy())
        if "event_forecast" in self.models and "event_features" in self.meta:
            out = pd.concat([out, self._forecast(X, out)], axis=1)
        return pd.concat([df, out], axis=1)

    def _forecast(self, X, assessed):
        """P(tier at the final CDM before TCA), from each CDM's point of view (causal features only)."""
        Xe = pd.concat([X.reset_index(drop=True),
                        assessed[["p_pc_elevated", "p_pc_critical", "log_pc"]].reset_index(drop=True)], axis=1)
        p = self.models["event_forecast"].predict(Xe[self.meta["event_features"]])
        t = self.meta.get("event_threshold", 0.3)
        tier = np.where(p[:, 2] >= t, 2, np.where(p[:, 1] + p[:, 2] >= t, 1, 0))
        return pd.DataFrame({"forecast_p_safe": p[:, 0], "forecast_p_elevated": p[:, 1],
                             "forecast_p_critical": p[:, 2], "forecast_tier": tier})

    def assess_screening(self, df):
        df = df.reset_index(drop=True)
        X = screening_features(df)
        if self.trained and "screen_q50" in self.models:
            med, lo, hi, pe, pc = self._predict_set("screen", X)
            thr = self.thresholds.get("screen", self.thresholds)
        else:
            # Without covariances the max-Pc bound is the conservative estimate.
            med = df["max_risk_estimate"].to_numpy(dtype=float)
            lo, hi = med - 3, med
            pe = (med >= PC_ELEVATED).astype(float)
            pc = (med >= PC_CRITICAL).astype(float)
            thr = DEFAULT_THRESHOLDS
        out = combine(med, lo, hi, pe, pc, df["miss_distance"].to_numpy(dtype=float), thr,
                      max_pc_bound=df["max_risk_estimate"].to_numpy(dtype=float))
        return pd.concat([df, out], axis=1)
