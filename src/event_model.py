"""Official task (updated brief, 31 Oct 12:45): predict each event's FINAL tier - the tier of the last
CDM before closest approach - using only CDMs issued >= 2 days before TCA.

    python src/event_model.py

Every CDM gets causal "as of this message" features (current risk and geometry, the trend so far, a
straight-line projection to TCA), so the same model gives a forecast that updates as closest approach
nears. Training rows are all CDMs >= 2 days out, weighted so each event counts once; evaluation is at
the latest CDM >= 2 days out, which is the moment a manoeuvre decision has to be made.

Evaluation follows the brief: a fixed 80/20 split of train events (stratified by final tier); model
choice and decision thresholds come from 5-fold CV inside the 80 %; the 20 % hold-out is scored once.
Columns beyond the 13 suggested ones are used and explained in COLUMN_NOTES.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import OBJECT_TYPES, SIGMA_SENTINEL, TIERS, md_tier, risk_tier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, OUT = ROOT / "data", ROOT / "models", ROOT / "outputs"
LABELS = ROOT / "docs" / "updates" / "challenge1_event_labels.xlsx"
SEED = 42
CUTOFF_DAYS = 2.0

COLUMNS = [
    "event_id", "time_to_tca", "risk", "miss_distance", "max_risk_estimate", "max_risk_scaling",
    "mahalanobis_distance", "relative_speed", "relative_position_r", "relative_position_t",
    "relative_position_n", "c_object_type", "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t",
    "c_sigma_n", "c_obs_used", "c_time_lastob_start", "c_recommended_od_span", "c_actual_od_span",
    "c_position_covariance_det", "t_position_covariance_det", "c_rcs_estimate", "t_span", "c_span",
    "F10", "AP", "SSN",
]
SUGGESTED = ["event_id", "time_to_tca", "miss_distance", "relative_speed", "relative_position_r",
             "relative_position_t", "relative_position_n", "c_object_type", "c_sigma_r", "t_sigma_r",
             "max_risk_estimate", "F10", "AP"]
COLUMN_NOTES = {
    "risk": "Pc of each early CDM. The brief notes the tier can be read from risk, so the task is to forecast "
            "the final tier from early CDMs; the latest early risk is the operational baseline (Uriot et al.).",
    "max_risk_scaling, mahalanobis_distance": "How far the covariance is from the worst case, and the miss "
                                               "distance in sigmas: they say how much Pc can still move.",
    "t/c_sigma_t, t/c_sigma_n, covariance determinants": "Along-track and cross-track uncertainty; the paper's "
                                                         "hard cases had large uncertainties (Sec. 5.4).",
    "c_obs_used, c_time_lastob_start, c_recommended/actual_od_span": "Quality of the chaser's orbit "
                                                                    "determination: sparse tracking makes "
                                                                    "late jumps more likely.",
    "c_rcs_estimate, t_span, c_span": "Object sizes, which set the hard-body radius in Pc.",
    "SSN": "Sunspot number, a drag proxy alongside F10.7 and Ap (ranked 18th in the paper's feature study).",
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_cdms(path):
    cols = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, usecols=[c for c in COLUMNS if c in cols])
    return df.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def load_labels():
    lab = pd.read_excel(LABELS, sheet_name="train_event_labels", header=2)
    lab["final_tier_code"] = lab.final_tier.map({"Safe": 0, "Elevated": 1, "Critical": 2})
    return lab.set_index("event_id")


def _log10(x, floor=1e-6):
    return np.log10(np.clip(np.asarray(x, float), floor, None))


def asof_features(df):
    """One row per CDM: features using only that CDM and earlier ones of the same event."""
    g = df.groupby("event_id", sort=False)
    t = df.time_to_tca.to_numpy(float)
    risk = df.risk.to_numpy(float)
    lmd = _log10(df.miss_distance, 1.0)
    f = pd.DataFrame(index=df.index)
    f["time_to_tca"] = t
    f["risk"] = risk
    f["log_md"] = lmd
    f["tier_now"] = risk_tier(risk, df.miss_distance)
    f["md_tier_now"] = md_tier(df.miss_distance)
    f["risk_margin_crit"] = risk + 4.0
    f["risk_margin_elev"] = risk + 6.0
    f["md_margin_1km"] = lmd - 3.0
    f["md_margin_5km"] = lmd - np.log10(5000)
    f["max_risk_estimate"] = df.max_risk_estimate
    f["gap_to_max"] = df.max_risk_estimate - risk
    f["log_max_risk_scaling"] = _log10(df.max_risk_scaling, 1e-12)
    f["log_mahal"] = _log10(df.mahalanobis_distance, 1e-9)
    f["relative_speed"] = df.relative_speed
    f["abs_pos_r"] = df.relative_position_r.abs()
    f["log_abs_pos_t"] = _log10(df.relative_position_t.abs(), 1.0)
    f["log_abs_pos_n"] = _log10(df.relative_position_n.abs(), 1.0)
    for c in ("t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"):
        f[f"log_{c}"] = _log10(df[c].where(df[c] < SIGMA_SENTINEL), 1e-3)
    f["log_c_cov_det"] = _log10(df.c_position_covariance_det, 1e-30)
    f["log_t_cov_det"] = _log10(df.t_position_covariance_det, 1e-30)
    for c in ("c_obs_used", "c_time_lastob_start", "c_recommended_od_span", "c_actual_od_span", "c_rcs_estimate",
              "t_span", "c_span", "F10", "AP", "SSN"):
        f[c] = df[c]
    f["c_object_type"] = pd.Categorical(df.c_object_type, categories=OBJECT_TYPES).codes

    # expanding (causal) history within the event
    n = g.cumcount().to_numpy() + 1
    f["n_cdms"] = n
    f["days_observed"] = g.time_to_tca.transform("first").to_numpy() - t
    for name, y in (("risk", risk), ("log_md", lmd)):
        s = pd.Series(y, index=df.index).groupby(df.event_id, sort=False)
        f[f"{name}_max"] = s.cummax().to_numpy()
        f[f"{name}_min"] = s.cummin().to_numpy()
        f[f"{name}_mean"] = s.cumsum().to_numpy() / n
        f[f"{name}_d_prev"] = y - s.shift(1).to_numpy()
        f[f"{name}_d_first"] = y - s.transform("first").to_numpy()
        # expanding OLS slope of y on time_to_tca, and straight-line projection to TCA (t = 0)
        sx = pd.Series(t, index=df.index).groupby(df.event_id, sort=False).cumsum().to_numpy()
        sy = s.cumsum().to_numpy()
        sxx = pd.Series(t * t, index=df.index).groupby(df.event_id, sort=False).cumsum().to_numpy()
        sxy = pd.Series(t * y, index=df.index).groupby(df.event_id, sort=False).cumsum().to_numpy()
        den = n * sxx - sx * sx
        slope = np.where((n >= 2) & (np.abs(den) > 1e-9), (n * sxy - sx * sy) / np.where(den == 0, 1, den), 0.0)
        f[f"{name}_slope"] = slope
        f[f"{name}_proj_tca"] = y - slope * t
    f["risk_proj_tca"] = np.clip(f["risk_proj_tca"], -30, 0)
    csig = _log10(df.c_sigma_t.where(df.c_sigma_t < SIGMA_SENTINEL), 1e-3)
    f["c_sigma_t_shrink"] = csig - pd.Series(csig, index=df.index).groupby(df.event_id, sort=False).transform("first").to_numpy()
    return f


SUGGESTED_FEATURES = [  # ablation: only the 13 suggested columns (no risk, no extra covariance terms)
    "time_to_tca", "log_md", "md_tier_now", "md_margin_1km", "md_margin_5km", "max_risk_estimate",
    "relative_speed", "abs_pos_r", "log_abs_pos_t", "log_abs_pos_n", "log_t_sigma_r", "log_c_sigma_r",
    "F10", "AP", "c_object_type", "n_cdms", "days_observed", "log_md_max", "log_md_min", "log_md_mean",
    "log_md_d_prev", "log_md_d_first", "log_md_slope", "log_md_proj_tca",
]
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, verbose=-1, random_state=SEED)
HEADS = {"crit": 2, "elev": 1}  # P(final tier >= k)


def event_table(df, f, labels=None):
    """Rows usable as inputs (>= 2 days out); the decision row is the latest one per event."""
    early = df.time_to_tca.to_numpy() >= CUTOFF_DAYS
    rows = f[early].copy()
    rows["event_id"] = df.event_id[early].to_numpy()
    rows["is_decision_row"] = ~rows.event_id.duplicated(keep="last")
    rows["w"] = 1.0 / rows.groupby("event_id").event_id.transform("size")
    if labels is not None:
        rows["y"] = rows.event_id.map(labels.final_tier_code).to_numpy()
    return rows


def fit_heads(X, y, w, n_estimators=None, Xv=None, yv=None, wv=None):
    models, best = {}, {}
    for h, k in HEADS.items():
        n_est = n_estimators.get(h, 3000) if isinstance(n_estimators, dict) else (n_estimators or 3000)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=n_est, **PARAMS)
        if Xv is not None:
            m.fit(X, (y >= k).astype(int), sample_weight=w, eval_set=[(Xv, (yv >= k).astype(int))],
                  eval_sample_weight=[wv], callbacks=[lgb.early_stopping(150, verbose=False)])
            best[h] = m.best_iteration_ or m.n_estimators
        else:
            m.fit(X, (y >= k).astype(int), sample_weight=w)
        models[h] = m
    return models, best


def predict_heads(models, X):
    pc = models["crit"].predict_proba(X)[:, 1]
    pe = np.maximum(models["elev"].predict_proba(X)[:, 1], pc)
    return pe, pc


def decide(pe, pc, thr):
    return np.where(pc >= thr["p_critical"], 2, np.where(pe >= thr["p_elevated"], 1, 0))


def tier_metrics(y, p):
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    return {"macro_f1": round(float(f1_score(y, p, average="macro")), 4),
            "f2_critical": round(float(fbeta_score(y == 2, p == 2, beta=2)), 4),
            "critical_recall": round(float(rec[2]), 4), "critical_precision": round(float(prec[2]), 4),
            "accuracy": round(float((y == p).mean()), 4),
            "recall": [round(float(v), 4) for v in rec], "precision": [round(float(v), 4) for v in prec],
            "missed_critical": int(((y == 2) & (p < 2)).sum()),
            "critical_called_safe": int(((y == 2) & (p == 0)).sum()),
            "false_alarms_critical": int(((y < 2) & (p == 2)).sum()),
            "confusion": cm.tolist(), "n": int(len(y))}


def tune(pe, pc, y):
    grid = np.round(np.r_[np.arange(0.02, 0.2, 0.01), np.arange(0.2, 0.9, 0.025)], 3)
    tc = max(grid, key=lambda t: fbeta_score(y == 2, pc >= t, beta=2))
    te = max(grid, key=lambda t: fbeta_score(y >= 1, (pc >= tc) | (pe >= t), beta=2))
    return {"p_critical": float(tc), "p_elevated": float(te)}


def cv_oof(rows, feats, n_folds=5):
    """Out-of-fold probabilities for decision rows; folds split by event, stratified by final tier."""
    ev = rows[rows.is_decision_row][["event_id", "y"]].reset_index(drop=True)
    pe = pd.Series(0.0, index=ev.event_id)
    pc = pd.Series(0.0, index=ev.event_id)
    best = {h: [] for h in HEADS}
    for a, b in StratifiedKFold(n_folds, shuffle=True, random_state=SEED).split(ev, ev.y):
        tr_ev, va_ev = set(ev.event_id[a]), set(ev.event_id[b])
        T, V = rows[rows.event_id.isin(tr_ev)], rows[rows.event_id.isin(va_ev) & rows.is_decision_row]
        Vall = rows[rows.event_id.isin(va_ev)]
        models, b_it = fit_heads(T[feats], T.y.to_numpy(), T.w.to_numpy(),
                                 Xv=Vall[feats], yv=Vall.y.to_numpy(), wv=Vall.w.to_numpy())
        e, c = predict_heads(models, V[feats])
        pe[V.event_id.to_numpy()], pc[V.event_id.to_numpy()] = e, c
        for h in HEADS:
            best[h].append(b_it[h])
    y = ev.set_index("event_id").y.loc[pe.index].to_numpy()
    return pe.to_numpy(), pc.to_numpy(), y, {h: int(np.mean(v) * 1.1) + 1 for h, v in best.items()}, pe.index


def explain(row, pe, pc, thr, tier):
    r = [f"Latest message is {row.time_to_tca:.1f} days before closest approach; its tier is "
         f"{TIERS[int(row.tier_now)]} (Pc {10 ** row.risk:.1e}, miss distance {10 ** row.log_md:,.0f} m)."
         if row.risk > -29.9 else
         f"Latest message is {row.time_to_tca:.1f} days before closest approach; its tier is "
         f"{TIERS[int(row.tier_now)]} (Pc ~0, miss distance {10 ** row.log_md:,.0f} m)."]
    if row.n_cdms >= 2:
        first = row.risk - row.risk_d_first
        if row.risk <= -29.9 and first <= -29.9:
            prob = "the probability stayed ~0"
        elif row.risk <= -29.9:
            prob = f"the probability fell from {10 ** first:.1e} to ~0"
        elif first <= -29.9:
            prob = f"the probability rose from ~0 to {10 ** row.risk:.1e}"
        else:
            prob = f"the probability changed by {row.risk_d_first:+.1f} orders of magnitude"
        r.append(f"Over the last {row.days_observed:.1f} days {prob} and the miss distance changed "
                 f"{100 * (10 ** row.log_md_d_first - 1):+.0f}%.")
    r.append(f"Chance the final tier is Critical: {pc:.0%} (escalate at {thr['p_critical']:.0%}). "
             f"At least Elevated: {pe:.0%} (threshold {thr['p_elevated']:.0%}).")
    if tier != int(row.tier_now):
        r.append(f"Forecast differs from the latest message: {TIERS[int(row.tier_now)]} now, "
                 f"{TIERS[tier]} expected at closest approach.")
    r.append(f"Forecast final tier: {TIERS[tier]}.")
    return r


def run():
    t0 = time.time()
    labels = load_labels()
    log("loading CDMs")
    tr = load_cdms(DATA / "train_data.csv")
    te = load_cdms(DATA / "test_data.csv")
    Ftr, Fte = asof_features(tr), asof_features(te)
    rows = event_table(tr, Ftr, labels)
    full_feats = [c for c in Ftr.columns]
    dec = rows[rows.is_decision_row]
    log(f"usable events {len(dec)}; final tiers {np.bincount(dec.y, minlength=3).tolist()}")

    # fixed 80/20 split of usable train events, stratified by final tier
    ev_tr, ev_ho = train_test_split(dec.event_id.to_numpy(), test_size=0.2, random_state=SEED, stratify=dec.y)
    split = pd.DataFrame({"event_id": np.r_[ev_tr, ev_ho], "split": ["train"] * len(ev_tr) + ["holdout"] * len(ev_ho)})
    split.sort_values("event_id").to_csv(OUT / "event_split.csv", index=False)
    R80, R20 = rows[rows.event_id.isin(set(ev_tr))], rows[rows.event_id.isin(set(ev_ho))]
    D20 = R20[R20.is_decision_row]
    y20 = D20.y.to_numpy()
    final_ttca = labels.final_time_to_tca_days
    late = (D20.event_id.map(final_ttca) < CUTOFF_DAYS).to_numpy()  # final CDM really is in the future

    results = {}
    feature_sets = {"all": full_feats, "suggested_13": SUGGESTED_FEATURES}
    for name, feats in feature_sets.items():
        log(f"CV ({name}, {len(feats)} features)")
        pe, pc, y, best, idx = cv_oof(R80, feats)
        thr = tune(pe, pc, y)
        cv = tier_metrics(y, decide(pe, pc, thr))
        # second operating point, also fixed on CV: best Critical recall with no more Critical false
        # alarms than the status quo (latest-message tier) has on the same CV events
        lrp_cv = R80[R80.is_decision_row].set_index("event_id").tier_now.loc[idx].to_numpy().astype(int)
        fa_lrp = int(((y < 2) & (lrp_cv == 2)).sum())
        grid = np.round(np.arange(0.05, 0.96, 0.005), 3)
        ok = [t for t in grid if ((y < 2) & (pc >= t)).sum() <= fa_lrp]
        thr_m = {"p_critical": float(min(ok)) if ok else 0.95, "p_elevated": thr["p_elevated"]}
        log(f"  thresholds {thr}; CV macro-F1 {cv['macro_f1']} F2c {cv['f2_critical']} recallC {cv['critical_recall']}")
        models, _ = fit_heads(R80[feats], R80.y.to_numpy(), R80.w.to_numpy(), n_estimators=best)
        e20, c20 = predict_heads(models, D20[feats])
        p20 = decide(e20, c20, thr)
        p20m = decide(e20, c20, thr_m)
        # hybrid: keep the latest-message tier and let the model only escalate (thresholds tuned on CV)
        g2 = np.round(np.r_[np.arange(0.05, 0.2, 0.01), np.arange(0.2, 0.96, 0.025)], 3)
        tc_h = max(g2, key=lambda t: fbeta_score(y == 2, (lrp_cv == 2) | (pc >= t), beta=2))
        te_h = max(g2, key=lambda t: fbeta_score(y >= 1, (lrp_cv >= 1) | (pc >= tc_h) | (pe >= t), beta=2))
        thr_h = {"p_critical": float(tc_h), "p_elevated": float(te_h)}
        lrp20 = D20.tier_now.to_numpy().astype(int)
        h_cv = np.maximum(lrp_cv, decide(pe, pc, thr_h))
        h20 = np.maximum(lrp20, decide(e20, c20, thr_h))
        log(f"  hybrid thresholds {thr_h}; CV F2c {tier_metrics(y, h_cv)['f2_critical']}")
        iso = IsotonicRegression(out_of_bounds="clip").fit(pc, (y == 2).astype(float))
        results[name] = {"features": feats, "thresholds": thr, "best_iterations": best, "cv": cv,
                         "hybrid": {"thresholds": thr_h, "cv": tier_metrics(y, h_cv),
                                    "holdout": tier_metrics(y20, h20),
                                    "holdout_forecast_needed": tier_metrics(y20[late], h20[late])},
                         "cv_status_quo": tier_metrics(y, lrp_cv),
                         "matched_false_alarm_point": {
                             "thresholds": thr_m, "cv": tier_metrics(y, decide(pe, pc, thr_m)),
                             "holdout": tier_metrics(y20, p20m),
                             "holdout_forecast_needed": tier_metrics(y20[late], p20m[late])},
                         "holdout": tier_metrics(y20, p20),
                         "holdout_forecast_needed": tier_metrics(y20[late], p20[late]),
                         "holdout_brier_critical": round(brier_score_loss(y20 == 2, c20), 5),
                         "holdout_brier_critical_calibrated": round(brier_score_loss(y20 == 2, iso.predict(c20)), 5),
                         "_iso": iso, "_pe20": e20, "_pc20": c20}

    # baselines on the same hold-out decision rows
    lrp = D20.tier_now.to_numpy().astype(int)
    lrp_margin = np.where((D20.risk >= -4.5) | (D20.log_md < 3.0), 2,
                          np.where((D20.risk >= -6.5) | (D20.log_md <= np.log10(5000)), 1, 0))
    dist = D20.md_tier_now.to_numpy().astype(int)
    baselines = {
        "Latest message tier (persistence, operational status quo)": lrp,
        "Latest tier with 0.5-decade safety margin": lrp_margin,
        "Latest miss distance only (no Pc)": dist,
    }
    base_out = {k: {"holdout": tier_metrics(y20, v), "holdout_forecast_needed": tier_metrics(y20[late], v[late])}
                for k, v in baselines.items()}

    # final model on all usable train events -> test predictions (per CDM and per event)
    # selection uses CV only (never the hold-out): best CV F2 on Critical over feature set x decision rule
    cands = [(v["hybrid"]["cv"]["f2_critical"], k, "hybrid") for k, v in results.items()] +             [(v["cv"]["f2_critical"], k, "model") for k, v in results.items()]
    _, chosen, mode = max(cands)
    log(f"CV selection: {sorted(cands, reverse=True)}")
    r = results[chosen]
    feats = r["features"]
    thr = r["hybrid"]["thresholds"] if mode == "hybrid" else r["thresholds"]
    log(f"decision rule: {mode} {thr}")

    def final_decision(pe_, pc_, now_):
        d = decide(pe_, pc_, thr)
        return np.maximum(now_.astype(int), d) if mode == "hybrid" else d
    log("final fit on all usable train events")
    final, _ = fit_heads(rows[feats], rows.y.to_numpy(), rows.w.to_numpy(), n_estimators=r["best_iterations"])
    MODELS.mkdir(exist_ok=True)
    for h, m in final.items():
        m.booster_.save_model(str(MODELS / f"final_tier_{h}.txt"))
    iso = r["_iso"]
    cal = {"x": [round(float(v), 6) for v in np.r_[0, iso.X_thresholds_, 1]],
           "y": [round(float(v), 6) for v in np.r_[iso.predict([0])[0], iso.y_thresholds_, iso.predict([1])[0]]]}
    (MODELS / "final_tier_meta.json").write_text(json.dumps({
        "features": feats, "thresholds": thr, "decision_rule": mode, "cutoff_days": CUTOFF_DAYS,
        "calibration_critical": cal,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_events": int(rows.event_id.nunique())}, indent=2))

    test_rows = event_table(te, Fte)
    e, c = predict_heads(final, test_rows[feats])
    tier = final_decision(e, c, test_rows.tier_now.to_numpy())
    per_cdm = pd.DataFrame({"event_id": test_rows.event_id.to_numpy(), "time_to_tca": test_rows.time_to_tca.to_numpy(),
                            "tier_now": test_rows.tier_now.to_numpy().astype(int),
                            "p_final_elevated_or_worse": e.round(4), "p_final_critical": c.round(4),
                            "forecast_final_tier": tier, "is_decision_row": test_rows.is_decision_row.to_numpy()})
    per_cdm.to_csv(OUT / "test_cdm_forecasts.csv", index=False)
    td = test_rows[test_rows.is_decision_row].reset_index(drop=True)
    ed, cd = e[test_rows.is_decision_row.to_numpy()], c[test_rows.is_decision_row.to_numpy()]
    tdec = final_decision(ed, cd, td.tier_now.to_numpy())
    ev_out = pd.DataFrame({
        "event_id": td.event_id, "n_cdms": td.n_cdms.astype(int), "latest_time_to_tca": td.time_to_tca.round(3),
        "latest_miss_distance_m": (10 ** td.log_md).round(0), "latest_log10_pc": td.risk.round(3),
        "tier_now": [TIERS[int(v)] for v in td.tier_now],
        "forecast_final_tier": [TIERS[int(v)] for v in tdec],
        "p_final_safe": (1 - ed).round(4), "p_final_elevated": (ed - cd).round(4), "p_final_critical": cd.round(4),
        "p_final_critical_calibrated": np.interp(cd, cal["x"], cal["y"]).round(4),
        "reasons": [" | ".join(explain(td.iloc[i], ed[i], cd[i], thr, int(tdec[i]))) for i in range(len(td))],
    })
    ev_out.to_csv(OUT / "test_event_predictions.csv", index=False)

    importance = pd.Series(final["crit"].booster_.feature_importance("gain"), index=feats)
    importance = (100 * importance / importance.sum()).sort_values(ascending=False).head(15)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "Predict each event's final tier (last CDM before TCA) from CDMs with time_to_tca >= 2 days.",
        "split": {"usable_events": int(len(dec)), "train_events": int(len(ev_tr)), "holdout_events": int(len(ev_ho)),
                  "holdout_final_tiers": np.bincount(y20, minlength=3).tolist(),
                  "holdout_forecast_needed_events": int(late.sum()),
                  "unusable_events": int((labels.usable_for_training == "no").sum()),
                  "file": "outputs/event_split.csv"},
        "chosen": chosen,
        "decision_rule": mode,
        "decision_thresholds": thr,
        "models": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()},
        "baselines": base_out,
        "feature_importance_critical": [{"feature": k, "gain_pct": round(float(v), 2)} for k, v in importance.items()],
        "column_notes": COLUMN_NOTES,
        "test_predictions": {"events": int(len(ev_out)),
                             "forecast_tier_counts": ev_out.forecast_final_tier.value_counts().to_dict(),
                             "tier_now_counts": ev_out.tier_now.value_counts().to_dict(),
                             "file": "outputs/test_event_predictions.csv"},
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "event_metrics.json").write_text(json.dumps(out, indent=2))
    rows_out = []
    for k, r_ in results.items():
        rows_out += [(f"model:{k} (F2-optimal)", r_), (f"model:{k} (status-quo false alarms)", r_["matched_false_alarm_point"]),
                     (f"hybrid:{k} (latest tier + model escalation)", r_["hybrid"])]
        print(f"CV {k}: model F2c {r_['cv']['f2_critical']} recC {r_['cv']['critical_recall']} FAc "
              f"{r_['cv']['false_alarms_critical']} | status quo F2c {r_['cv_status_quo']['f2_critical']} recC "
              f"{r_['cv_status_quo']['critical_recall']} FAc {r_['cv_status_quo']['false_alarms_critical']} | matched "
              f"recC {r_['matched_false_alarm_point']['cv']['critical_recall']}")
    for name, v in rows_out + list(base_out.items()):
        h = v["holdout"]
        f = v["holdout_forecast_needed"]
        print(f"{name[:60]:60s} HO F1 {h['macro_f1']:.4f} F2c {h['f2_critical']:.4f} recC {h['critical_recall']:.3f} "
              f"missC {h['missed_critical']:3d} FAc {h['false_alarms_critical']:3d} | forecast-needed F2c "
              f"{f['f2_critical']:.4f} recC {f['critical_recall']:.3f}")
    log("done")


if __name__ == "__main__":
    run()
