"""Feature engineering for the conjunction risk evaluator.

Only the columns permitted by the challenge brief are read (ALLOWED_COLUMNS).
`risk` (log10 Pc) is read separately and used only as a label.
"""
import numpy as np
import pandas as pd

ALLOWED_COLUMNS = [
    "event_id", "time_to_tca", "miss_distance", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP",
]
LABEL = "risk"

# Operational thresholds (NASA CA Handbook / ESA SDO practice, as given in the brief)
PC_CRITICAL = -4.0   # log10 Pc >= -4  -> maneuver required
PC_ELEVATED = -6.0   # log10 Pc >= -6  -> elevated
MD_CRITICAL = 1000.0  # metres
MD_ELEVATED = 5000.0  # metres
TIERS = ["Safe", "Elevated", "Critical"]

# CDM placeholder for "covariance unavailable" (~10x Earth radius in metres)
SIGMA_SENTINEL = 6.0e7
OBJECT_TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "TBA"]


def load(path, with_label=True):
    cols = ALLOWED_COLUMNS + ([LABEL] if with_label else [])
    df = pd.read_csv(path, usecols=cols)
    return df.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def md_tier(miss_distance):
    """Brief: Critical < 1.0 km; Elevated 1.0-5.0 km (inclusive); Safe > 5.0 km."""
    md = np.asarray(miss_distance)
    return np.select([md < MD_CRITICAL, md <= MD_ELEVATED], [2, 1], 0)


def pc_tier(log_pc):
    r = np.asarray(log_pc)
    return np.select([r >= PC_CRITICAL, r >= PC_ELEVATED], [2, 1], 0)


def risk_tier(log_pc, miss_distance):
    """Final operational tier = worst of the Pc rule and the miss-distance rule."""
    return np.maximum(pc_tier(log_pc), md_tier(miss_distance))


def isotropic_pc(md, sigma, hbr):
    """2D Pc for an isotropic Gaussian with combined std `sigma` and hard-body radius `hbr`
    (small-HBR approximation): Pc = HBR^2 / (2 sigma^2) * exp(-md^2 / (2 sigma^2))."""
    s2 = np.maximum(sigma, 1e-3) ** 2
    with np.errstate(divide="ignore"):
        log_pc = np.log10(hbr ** 2 / (2 * s2)) - (md ** 2 / (2 * s2)) / np.log(10)
    return np.clip(log_pc, -30, 0)


def physics_features(df):
    """Per-CDM features derived from encounter geometry (no event history)."""
    f = pd.DataFrame(index=df.index)
    md = df["miss_distance"].clip(lower=1.0)
    f["time_to_tca"] = df["time_to_tca"]
    f["log_md"] = np.log10(md)
    f["relative_speed"] = df["relative_speed"]
    for ax in "rtn":
        p = df[f"relative_position_{ax}"]
        f[f"pos_{ax}"] = p
        f[f"frac_{ax}"] = p.abs() / md
    f["max_risk_estimate"] = df["max_risk_estimate"]

    ts, cs = df["t_sigma_r"], df["c_sigma_r"]
    f["t_sigma_valid"] = (ts < SIGMA_SENTINEL).astype(int)
    f["c_sigma_valid"] = (cs < SIGMA_SENTINEL).astype(int) * cs.notna()
    f["log_t_sigma_r"] = np.log10(ts.clip(lower=1e-3))
    f["log_c_sigma_r"] = np.log10(cs.clip(lower=1e-3))
    sig = np.sqrt(ts ** 2 + cs.fillna(SIGMA_SENTINEL) ** 2)
    f["log_sigma_r_comb"] = np.log10(sig)
    f["log_md_over_sigma"] = f["log_md"] - f["log_sigma_r_comb"]
    # Radial-only Mahalanobis distance: how many sigmas apart in the radial axis
    f["radial_mahal"] = df["relative_position_r"].abs() / sig

    # Max-Pc bound implies an effective hard-body radius: Pc_max = HBR^2 / (e * md^2)
    log_hbr2 = df["max_risk_estimate"] + 2 * np.log10(md) + np.log10(np.e)
    f["log_hbr"] = log_hbr2 / 2
    hbr = 10 ** f["log_hbr"]
    f["log_pc_iso"] = isotropic_pc(md, sig, hbr)
    f["pc_gap_to_max"] = f["log_pc_iso"] - df["max_risk_estimate"]

    f["F10"] = df["F10"]
    f["AP"] = df["AP"]
    f["c_object_type"] = pd.Categorical(df["c_object_type"], categories=OBJECT_TYPES).codes
    return f


HISTORY_BASE = ["log_md", "max_risk_estimate", "log_c_sigma_r", "log_t_sigma_r", "log_pc_iso",
                "pos_r", "log_md_over_sigma"]


def history_features(df, f):
    """Causal event-history features: only CDMs issued *before* the current one
    (larger time_to_tca) are used, as an operator would have at that moment."""
    h = pd.DataFrame(index=df.index)
    g = f[HISTORY_BASE].groupby(df["event_id"])
    h["cdm_index"] = df.groupby("event_id").cumcount()
    t = df["time_to_tca"]
    prev_t = t.groupby(df["event_id"]).shift(1)
    h["days_since_prev_cdm"] = prev_t - t
    first_t = t.groupby(df["event_id"]).transform("first")
    h["days_since_first_cdm"] = first_t - t
    for c in HISTORY_BASE:
        prev = g[c].shift(1)
        first = g[c].transform("first")
        h[f"d_{c}_prev"] = f[c] - prev
        h[f"d_{c}_first"] = f[c] - first
        h[f"{c}_cummax"] = g[c].cummax()
        h[f"{c}_cummin"] = g[c].cummin()
        h[f"{c}_ewm"] = g[c].transform(lambda s: s.ewm(halflife=2).mean())
    h["slope_log_md_per_day"] = h["d_log_md_first"] / h["days_since_first_cdm"].replace(0, np.nan)
    return h


def build_features(df):
    f = physics_features(df)
    h = history_features(df, f)
    return pd.concat([f, h], axis=1)
