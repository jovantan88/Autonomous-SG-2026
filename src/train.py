"""Train, validate and score the conjunction risk evaluator.

    python src/train.py            # full run: CV on train, final fit, score test, write outputs

Outputs
  models/{full,screen}_{q10,q50,q90,clf_elev,clf_crit}.txt   LightGBM boosters
  models/thresholds.json, models/meta.json
  outputs/metrics.json            served by /api/metrics
  outputs/test_predictions.csv    one row per test CDM (tier + probabilities + reasons)
  outputs/failure_cases.csv

Validation follows Uriot et al. (2022): events are grouped (no CDM of one event in both train
and validation folds) and only CDMs issued >= 2 days before TCA are scored, matching test_data.
Thresholds are tuned for F2 (recall weighted 2x) because a missed Critical event is far more
costly than a false alarm.
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
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (LABEL, PC_CRITICAL, PC_ELEVATED, TIERS, build_features, load, md_tier,  # noqa: E402
                      physics_features, risk_tier)
from predict import SCREENING_COLUMNS, RiskModel, combine, screening_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, OUT = ROOT / "data", ROOT / "models", ROOT / "outputs"
SEED = 42
QUICK = "--quick" in sys.argv
N_FOLDS = 2 if QUICK else 5
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, verbose=-1, random_state=SEED)
HEADS = {
    "q10": dict(objective="quantile", alpha=0.10),
    "q50": dict(objective="quantile", alpha=0.50),
    "q90": dict(objective="quantile", alpha=0.90),
    "clf_elev": dict(objective="binary"),
    "clf_crit": dict(objective="binary"),
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def head_target(head, y):
    if head == "clf_elev":
        return (y >= PC_ELEVATED).astype(int)
    if head == "clf_crit":
        return (y >= PC_CRITICAL).astype(int)
    return y


def fit_head(head, X, y, Xv=None, yv=None, n_estimators=None):
    n_estimators = n_estimators or (200 if QUICK else 3000)
    p = {**PARAMS, **HEADS[head]}
    model = lgb.LGBMClassifier(**p, n_estimators=n_estimators) if head.startswith("clf") \
        else lgb.LGBMRegressor(**p, n_estimators=n_estimators)
    t = head_target(head, y)
    if Xv is not None:
        model.fit(X, t, eval_set=[(Xv, head_target(head, yv))],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X, t)
    return model


def predict_head(model, head, X):
    return model.predict_proba(X)[:, 1] if head.startswith("clf") else model.predict(X)


def cross_validate(X, y, groups):
    """Out-of-fold predictions for every head + best iteration counts."""
    oof = {h: np.zeros(len(y)) for h in HEADS}
    best = {h: [] for h in HEADS}
    for k, (a, b) in enumerate(GroupKFold(N_FOLDS).split(X, groups=groups)):
        for h in HEADS:
            m = fit_head(h, X.iloc[a], y[a], X.iloc[b], y[b])
            oof[h][b] = predict_head(m, h, X.iloc[b])
            best[h].append(m.best_iteration_ or m.n_estimators)
        log(f"  fold {k + 1}/{N_FOLDS} done")
    return oof, {h: int(np.mean(v) * 1.1) + 1 for h, v in best.items()}


def tier_metrics(true_tier, pred_tier):
    cm = confusion_matrix(true_tier, pred_tier, labels=[0, 1, 2])
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    return {
        "macro_f1": round(float(f1_score(true_tier, pred_tier, average="macro")), 4),
        "f2_critical": round(float(fbeta_score(true_tier == 2, pred_tier == 2, beta=2)), 4),
        "f2_elevated_or_worse": round(float(fbeta_score(true_tier >= 1, pred_tier >= 1, beta=2)), 4),
        "accuracy": round(float((true_tier == pred_tier).mean()), 4),
        "recall": [round(float(v), 4) for v in rec],
        "precision": [round(float(v), 4) for v in prec],
        "missed_critical": int(((true_tier == 2) & (pred_tier < 2)).sum()),
        "critical_called_safe": int(((true_tier == 2) & (pred_tier == 0)).sum()),
        "false_alarms_critical": int(((true_tier < 2) & (pred_tier == 2)).sum()),
        "confusion": cm.tolist(),
        "n": int(len(true_tier)),
    }


def tune_thresholds(p_elev, p_crit, md, true_tier):
    """Pick decision thresholds on OOF predictions: maximise F2 for Critical, then for >=Elevated."""
    m = md_tier(md)
    grid = np.round(np.concatenate([np.arange(0.02, 0.2, 0.01), np.arange(0.2, 0.95, 0.025)]), 3)
    best_c = max(grid, key=lambda t: fbeta_score(true_tier == 2, np.maximum(m, 2 * (p_crit >= t)) == 2, beta=2))
    crit = np.maximum(m == 2, p_crit >= best_c)
    best_e = max(grid, key=lambda t: fbeta_score(
        true_tier >= 1, crit | (m >= 1) | (np.maximum(p_elev, p_crit) >= t), beta=2))
    return {"p_critical": float(best_c), "p_elevated": float(best_e)}


def decide(p_elev, p_crit, md, thr):
    m = md_tier(md)
    pc = np.where(p_crit >= thr["p_critical"], 2, np.where(np.maximum(p_elev, p_crit) >= thr["p_elevated"], 1, 0))
    return np.maximum(pc, m)


def calibration(p, y):
    """Reliability table on fixed probability bins (most CDMs sit near 0, so quantile bins collapse)."""
    edges = np.array([0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0000001])
    idx = np.digitize(p, edges[1:-1])
    return [{"p_pred": round(float(p[idx == i].mean()), 4), "p_obs": round(float(y[idx == i].mean()), 4),
             "n": int((idx == i).sum())} for i in range(len(edges) - 1) if (idx == i).any()]


def explain_failure(row):
    why = []
    for thr, name in ((PC_CRITICAL, "1e-4"), (PC_ELEVATED, "1e-6")):
        if abs(row.true_log_pc - thr) <= 0.1:
            why.append(f"borderline: true Pc {10 ** row.true_log_pc:.2e} is within 0.1 decades of the {name} threshold")
    if getattr(row, "needs_review", False):
        why.append("flagged for analyst review (not auto-cleared)")
    if row.true_tier == 2 and row.miss_distance >= 1000:
        why.append(f"Critical only because true Pc {10 ** row.true_log_pc:.1e} >= 1e-4 while miss distance "
                   f"{row.miss_distance / 1000:.1f} km looks benign")
    if row.c_sigma_r > 1000 or row.t_sigma_r > 1000:
        why.append(f"very large radial uncertainty (chaser sigma_r {row.c_sigma_r:,.0f} m)")
    if row.max_risk_estimate < -6:
        why.append(f"max-Pc bound {row.max_risk_estimate:.1f} gives little warning")
    if row.p_critical > 0.02 and row.pred_tier < 2:
        why.append(f"model saw some risk (P(crit)={row.p_critical:.0%}) but below threshold")
    if not why:
        why.append("Pc depends on along-track / normal covariance terms that are outside the allowed "
                   "columns (only sigma_r is available)")
    return "; ".join(why)


def score_test(te, yte, tier_te):
    """Score test_data through the same RiskModel inference path the server uses; writes
    outputs/test_predictions.csv and outputs/failure_cases.csv; returns the metrics.json sections."""
    log("scoring test via RiskModel")
    model = RiskModel.load(MODELS)
    a = model.assess_cdms(te.drop(columns=[LABEL]))
    assert (a.event_id.to_numpy() == te.event_id.to_numpy()).all()  # stable sort keeps test order
    a["true_log_pc"], a["true_tier"] = yte, tier_te
    a["pred_tier"] = a["tier"]
    lo, hi = a.log_pc_lo.to_numpy(), a.log_pc_hi.to_numpy()
    yt = a.true_log_pc.to_numpy()
    nonfloor = yt > -30
    regression = {
        "test_mae_log_pc": round(float(np.abs(a.log_pc - yt).mean()), 3),
        "test_mae_log_pc_high_risk": round(float(np.abs(a.log_pc - yt)[yt >= PC_ELEVATED].mean()), 3),
        "interval_coverage_80": round(float(((yt >= lo) & (yt <= hi)).mean()), 4),
        "interval_coverage_80_nonfloor": round(float(((yt >= lo) & (yt <= hi))[nonfloor].mean()), 4),
        "note": "risk = -30 is the dataset floor for 'Pc effectively zero'; MAE is dominated by that mass",
    }
    wrong = (a.pred_tier != a.true_tier).to_numpy()
    rev = a.needs_review.to_numpy()
    missed = ((a.true_tier == 2) & (a.pred_tier < 2)).to_numpy()
    review_queue = {
        "needs_review_rate": round(float(rev.mean()), 4),
        "n_needs_review": int(rev.sum()),
        "error_rate_in_review_queue": round(float(wrong[rev].mean()), 4) if rev.any() else 0.0,
        "error_rate_auto_decided": round(float(wrong[~rev].mean()), 4),
        "missed_critical": int(missed.sum()),
        "missed_critical_flagged_for_review": int((missed & rev).sum()),
        "missed_critical_silent": int((missed & ~rev).sum()),
        "n_heads_disagree": int(a["heads_disagree"].sum()) if "heads_disagree" in a.columns else 0,
    }
    hidden = a.true_tier.to_numpy() > md_tier(a.miss_distance)
    pc_driven = {"n_rows_where_pc_raises_tier": int(hidden.sum()),
                 "caught_by_model": int((a.pred_tier.to_numpy() >= a.true_tier.to_numpy())[hidden].sum()),
                 "caught_by_distance_rule": 0}

    fails = a[(a.true_tier == 2) & (a.pred_tier < 2)].sort_values("true_log_pc", ascending=False)
    fails = pd.concat([fails, a[(a.true_tier == 0) & (a.pred_tier == 2)].sort_values(
        "p_critical", ascending=False).head(10)])
    fails["why"] = [explain_failure(r) for r in fails.itertuples()]
    fail_cols = ["event_id", "time_to_tca", "miss_distance", "true_tier", "pred_tier", "needs_review", "log_pc",
                 "true_log_pc", "p_critical", "max_risk_estimate", "c_sigma_r", "t_sigma_r", "c_object_type", "why"]
    fails[fail_cols].to_csv(OUT / "failure_cases.csv", index=False)

    keep = ["event_id", "time_to_tca", "miss_distance", "c_object_type", "log_pc", "log_pc_lo", "log_pc_hi",
            "p_pc_elevated", "p_pc_critical", "p_safe", "p_elevated", "p_critical", "md_tier", "tier",
            "needs_review", "true_log_pc", "true_tier"]
    keep += [c for c in a.columns if c.startswith("forecast_")]
    out = a[keep].copy()
    out["tier_name"] = out.tier.map(dict(enumerate(TIERS)))
    out["reasons"] = a.reasons.map(" | ".join)
    out.to_csv(OUT / "test_predictions.csv", index=False)
    return {
        "regression": regression,
        "review_queue": review_queue,
        "pc_driven_escalations": pc_driven,
        "served_model_test": tier_metrics(a.true_tier.to_numpy(), a.pred_tier.to_numpy()),
        "calibration": calibration(a.p_pc_critical.to_numpy(), (yt >= PC_CRITICAL).astype(float)),
        "calibration_elevated": calibration(a.p_pc_elevated.to_numpy(), (yt >= PC_ELEVATED).astype(float)),
        "failure_cases": json.loads(fails[fail_cols].head(40).to_json(orient="records")),
    }


def score_only():
    """Re-score test with the saved models and refresh those sections of outputs/metrics.json."""
    te = load(DATA / "test_data.csv")
    yte = te[LABEL].to_numpy()
    scores = score_test(te, yte, risk_tier(yte, te.miss_distance))
    path = OUT / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics.update(scores)
    # keep the methods table in sync with what the server actually serves (calibration / thresholds)
    screen = RiskModel.load(MODELS).assess_screening(te[SCREENING_COLUMNS])
    served = {"LightGBM, per-CDM physics features": scores["served_model_test"],
              "LightGBM, screening": tier_metrics(risk_tier(yte, te.miss_distance), screen.tier.to_numpy())}
    for mth in metrics.get("methods", []):
        for prefix, test_metrics in served.items():
            if mth["name"].startswith(prefix):
                mth["test"] = test_metrics
    metrics["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: scores[k] for k in ["regression", "review_queue", "pc_driven_escalations"]}, indent=1))


def run():
    t0 = time.time()
    MODELS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    log("loading data")
    tr = load(DATA / "train_data.csv")
    te = load(DATA / "test_data.csv")
    if QUICK:  # smoke-test the pipeline on a slice
        tr = tr[tr.event_id < 2500].reset_index(drop=True)
        te = te[te.event_id < 400].reset_index(drop=True)
    ytr, yte = tr[LABEL].to_numpy(), te[LABEL].to_numpy()
    tier_tr, tier_te = risk_tier(ytr, tr.miss_distance), risk_tier(yte, te.miss_distance)
    mask = (tr.time_to_tca >= 2).to_numpy()  # score CV like the test set
    log("building features")
    Ftr_all, Fte_all = build_features(tr), build_features(te)
    full_cols = list(physics_features(tr.head(2)).columns)
    hist_cols = list(Ftr_all.columns)
    Str, Ste = screening_features(tr), screening_features(te)
    screen_cols = list(Str.columns)

    methods, thresholds, best_iters, oofs = [], {}, {}, {}

    def baseline(name, desc, log_pc_tr, log_pc_te):
        methods.append({"name": name, "kind": "baseline", "description": desc,
                        "cv": tier_metrics(tier_tr[mask], risk_tier(log_pc_tr, tr.miss_distance)[mask]),
                        "test": tier_metrics(tier_te, risk_tier(log_pc_te, te.miss_distance))})

    baseline("Miss-distance rule only", "Applies the 1 km / 5 km rule and assumes Pc is always safe (status quo "
             "triage on distance)", np.full(len(tr), -30.0), np.full(len(te), -30.0))
    baseline("Max-Pc bound as Pc", "Treats max_risk_estimate (worst-case Pc over covariance scaling) as the Pc",
             tr.max_risk_estimate.to_numpy(), te.max_risk_estimate.to_numpy())
    baseline("Isotropic Gaussian Pc (physics)", "2D Pc with combined radial sigma and hard-body radius implied by the "
             "max-Pc bound", Ftr_all.log_pc_iso.to_numpy(), Fte_all.log_pc_iso.to_numpy())

    sets = {"full": (Ftr_all[full_cols], Fte_all[full_cols], "LightGBM, per-CDM physics features (chosen)"),
            "history": (Ftr_all[hist_cols], Fte_all[hist_cols], "LightGBM, + causal event-history features (ablation)"),
            "screen": (Str[screen_cols], Ste[screen_cols], "LightGBM, screening inputs only (live SOCRATES mode)")}
    test_preds = {}
    for name, (Xtr, Xte, desc) in sets.items():
        log(f"CV for {name} ({Xtr.shape[1]} features)")
        oof, best = cross_validate(Xtr, ytr, tr.event_id)
        thr = tune_thresholds(oof["clf_elev"][mask], oof["clf_crit"][mask], tr.miss_distance[mask], tier_tr[mask])
        log(f"  thresholds {thr}")
        log(f"final fit {name}")
        preds = {}
        for h in HEADS:
            m = fit_head(h, Xtr, ytr, n_estimators=best[h])
            preds[h] = predict_head(m, h, Xte)
            if name != "history":
                m.booster_.save_model(str(MODELS / f"{name}_{h}.txt"))
        cv_tier = decide(oof["clf_elev"], oof["clf_crit"], tr.miss_distance, thr)
        te_tier = decide(preds["clf_elev"], preds["clf_crit"], te.miss_distance, thr)
        methods.append({"name": desc, "kind": "model" if name != "history" else "ablation",
                        "description": f"quantile regression + binary classifiers, thresholds {thr}",
                        "cv": tier_metrics(tier_tr[mask], cv_tier[mask]), "test": tier_metrics(tier_te, te_tier)})
        thresholds[name], best_iters[name], oofs[name], test_preds[name] = thr, best, oof, preds
        if name == "full":
            fi = pd.Series(lgb.Booster(model_file=str(MODELS / "full_clf_crit.txt")).feature_importance("gain"),
                           index=full_cols)
            feature_importance = [{"feature": k, "gain_pct": round(float(v), 2)}
                                  for k, v in (100 * fi / fi.sum()).sort_values(ascending=False).head(15).items()]

    thr_out = {"full": thresholds["full"], "screen": thresholds["screen"]}
    (MODELS / "thresholds.json").write_text(json.dumps(thr_out, indent=2))
    (MODELS / "meta.json").write_text(json.dumps({
        "full_features": full_cols, "screen_features": screen_cols, "screen_inputs": SCREENING_COLUMNS,
        "best_iterations": best_iters, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(tr)), "train_events": int(tr.event_id.nunique()),
        "conformal_offsets": {n: conformal_offsets(oofs[n], ytr, mask) for n in ("full", "screen")}}, indent=2))

    test_scores = score_test(te, yte, tier_te)

    # ---- event-level forecast: tier at the final CDM (< 1 day to TCA) from CDMs >= 2 days out
    log("event-level forecast")
    event_forecast = event_level(tr, Ftr_all, ytr, oofs["full"], hist_cols, thresholds["full"])

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_set": {"name": "test_data.csv", "n_rows": int(len(te)), "n_events": int(te.event_id.nunique()),
                     "true_tier_counts": np.bincount(tier_te, minlength=3).tolist()},
        "cv_set": {"name": "train_data.csv, 5-fold GroupKFold by event, scored on CDMs >= 2 days to TCA",
                   "n_rows": int(mask.sum()), "n_events": int(tr.event_id[mask].nunique()),
                   "true_tier_counts": np.bincount(tier_tr[mask], minlength=3).tolist()},
        "chosen_method": "LightGBM, per-CDM physics features (chosen)",
        "thresholds": thr_out,
        "methods": methods,
        **test_scores,
        "feature_importance": feature_importance,
        "event_forecast": event_forecast,
        "screening_model": {"description": "Trained on the CDM data restricted to inputs a TLE screening feed "
                                           "(CelesTrak SOCRATES) provides: miss distance, relative speed, max Pc, "
                                           "time to TCA, object type, F10.7, Ap.",
                            "test": next(m["test"] for m in methods if m["name"].startswith("LightGBM, screening"))},
        "notes": [
            "Ground truth tier = worst of Pc rule and miss-distance rule, computed from the CDM's own risk column.",
            "The model never sees `risk`; it uses only the 13 allowed columns from the brief.",
            "Thresholds are tuned on cross-validation (never on test) to maximise F2, i.e. recall-weighted.",
            "Train/test differ by mission mix (Uriot et al. 2022, Sec. 4.5); CV-to-test gaps are expected.",
            "Live SOCRATES conjunctions are TLE-based screening products without covariance: the live tier is "
            "screening-grade and must be confirmed with CSpOC CDMs before any manoeuvre decision.",
        ],
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log("done in", round(time.time() - t0), "s")
    for mth in methods:
        print(f"{mth['name'][:52]:52s} CV F1 {mth['cv']['macro_f1']:.4f} F2c {mth['cv']['f2_critical']:.4f} | "
              f"TEST F1 {mth['test']['macro_f1']:.4f} F2c {mth['test']['f2_critical']:.4f} "
              f"missedC {mth['test']['missed_critical']} falseC {mth['test']['false_alarms_critical']}")
    print(json.dumps({k: metrics[k] for k in ["regression", "review_queue", "pc_driven_escalations"]}, indent=1))
    print(json.dumps(event_forecast, indent=1)[:3000])


EVENT_THRESHOLD = 0.3  # fixed a priori: escalate when P(final tier >= k) >= 30%


def event_level(tr, F, y, oof_full, hist_cols, thr):
    """Forecast the tier of an event's final CDM (< 1 day to TCA) from its latest CDM >= 2 days out,
    on train events that satisfy the Kelvins test-set constraints. Grouped 5-fold CV, then a final
    fit on all eligible events saved to models/event_forecast.txt."""
    df = tr.assign(_y=y)
    last = df.groupby("event_id").tail(1)
    eligible = last[last.time_to_tca < 1].event_id
    early = df[(df.time_to_tca >= 2) & df.event_id.isin(eligible)]
    idx = early.groupby("event_id").tail(1).index  # latest CDM >= 2 days (sorted by time desc)
    final = last.set_index("event_id").loc[df.loc[idx, "event_id"]]
    y_final = risk_tier(final._y.to_numpy(), final.miss_distance.to_numpy())
    X = F.loc[idx, hist_cols].copy()
    # same quantities RiskModel exposes at inference: p_pc_elevated, p_pc_critical, log_pc
    X["p_pc_elevated"] = np.maximum(oof_full["clf_elev"][idx], oof_full["clf_crit"][idx])
    X["p_pc_critical"] = oof_full["clf_crit"][idx]
    X["log_pc"] = np.clip(oof_full["q50"][idx], -30, 0)
    groups = df.loc[idx, "event_id"].to_numpy()
    cur_true = risk_tier(df.loc[idx, "_y"].to_numpy(), df.loc[idx, "miss_distance"].to_numpy())
    cur_model = decide(oof_full["clf_elev"][idx], oof_full["clf_crit"][idx], df.loc[idx, "miss_distance"], thr)
    proba = np.zeros((len(idx), 3))

    def make():
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=300,
                                  **{**PARAMS, "num_leaves": 15, "min_child_samples": 20})
    for a, b in GroupKFold(N_FOLDS).split(X, groups=groups):
        c = make()
        c.fit(X.iloc[a], y_final[a], sample_weight=np.where(y_final[a] > 0, 3.0, 1.0))
        proba[b] = c.predict_proba(X.iloc[b])
    final_model = make()
    final_model.fit(X, y_final, sample_weight=np.where(y_final > 0, 3.0, 1.0))
    final_model.booster_.save_model(str(MODELS / "event_forecast.txt"))
    meta_path = MODELS / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"event_features": list(X.columns), "event_threshold": EVENT_THRESHOLD})
    meta_path.write_text(json.dumps(meta, indent=2))
    p_ge1, p_ge2 = proba[:, 1] + proba[:, 2], proba[:, 2]
    fc = np.where(p_ge2 >= EVENT_THRESHOLD, 2, np.where(p_ge1 >= EVENT_THRESHOLD, 1, 0))
    return {
        "description": "Predict the tier at the final CDM (< 1 day to TCA) from the latest CDM issued >= 2 days "
                       "before TCA (the moment a manoeuvre must be planned). Train events meeting the Kelvins "
                       "test-set constraints; grouped 5-fold CV. test_data has no final CDMs so this cannot be "
                       "scored on test.",
        "n_events": int(len(idx)), "final_tier_counts": np.bincount(y_final, minlength=3).tolist(),
        "persistence_true_current_tier": tier_metrics(y_final, cur_true),
        "persistence_model_current_tier": tier_metrics(y_final, cur_model),
        "forecast_model": tier_metrics(y_final, fc),
        "note": "persistence_true_current_tier uses the CDM's own Pc (the Latest Risk Prediction baseline of "
                "Uriot et al.) and is shown as a reference only: it needs the `risk` column the brief excludes.",
    }


def conformal_offsets(oof, y, mask, alpha=0.1):
    """Split-conformal correction for the 10/90 % quantiles (CQR, Romano et al. 2019), per side,
    from out-of-fold predictions on CDMs >= 2 days out. Returns amounts to widen lo / hi by."""
    lo, hi, yy = oof["q10"][mask], oof["q90"][mask], y[mask]
    n = len(yy)
    level = min(1.0, (1 - alpha) * (1 + 1 / n))
    return {"lo": round(float(max(0.0, np.quantile(lo - yy, level))), 4),
            "hi": round(float(max(0.0, np.quantile(yy - hi, level))), 4)}


def calibrate_only():
    """Out-of-fold 10/90 % quantiles for the saved model sets -> conformal offsets in models/meta.json."""
    tr = load(DATA / "train_data.csv")
    ytr = tr[LABEL].to_numpy()
    mask = (tr.time_to_tca >= 2).to_numpy()
    meta = json.loads((MODELS / "meta.json").read_text())
    F = build_features(tr)
    S = screening_features(tr)
    offsets = {}
    for name, X in [("full", F[meta["full_features"]]), ("screen", S[meta["screen_features"]])]:
        oof = {h: np.zeros(len(ytr)) for h in ("q10", "q90")}
        for k, (a, b) in enumerate(GroupKFold(N_FOLDS).split(X, groups=tr.event_id)):
            for h in oof:
                m = fit_head(h, X.iloc[a], ytr[a], n_estimators=meta["best_iterations"][name][h])
                oof[h][b] = m.predict(X.iloc[b])
        raw = float(((ytr >= oof["q10"]) & (ytr <= oof["q90"]))[mask].mean())
        offsets[name] = {**conformal_offsets(oof, ytr, mask), "raw_cv_coverage": round(raw, 4)}
        log(f"{name}: raw CV coverage {raw:.3f}, offsets {offsets[name]}")
    meta["conformal_offsets"] = offsets
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))


def ece(p, y, bins=(0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0000001)):
    """Expected calibration error on fixed bins (count-weighted |mean p - observed rate|)."""
    idx = np.digitize(p, np.asarray(bins)[1:-1])
    return float(sum(abs(p[idx == i].mean() - y[idx == i].mean()) * (idx == i).sum()
                     for i in np.unique(idx)) / len(p))


def oof_classifiers(X, y, groups, best, n_folds):
    oof = {h: np.zeros(len(y)) for h in ("clf_elev", "clf_crit")}
    for a, b in GroupKFold(n_folds).split(X, groups=groups):
        for h in oof:
            m = fit_head(h, X.iloc[a], y[a], n_estimators=best[h])
            oof[h][b] = predict_head(m, h, X.iloc[b])
    return oof


def robustness():
    """(1) Isotonic probability calibration fitted on event-grouped OOF predictions, thresholds re-tuned
    on the calibrated OOF; (2) leave-mission-out CV (mission_id used only to split, never as a feature)."""
    tr, te = load(DATA / "train_data.csv"), load(DATA / "test_data.csv")
    ytr, yte = tr[LABEL].to_numpy(), te[LABEL].to_numpy()
    tier_tr = risk_tier(ytr, tr.miss_distance)
    mask = (tr.time_to_tca >= 2).to_numpy()
    mission = pd.read_csv(DATA / "train_data.csv", usecols=["event_id", "mission_id"]).drop_duplicates("event_id")
    tr_mission = tr.event_id.map(mission.set_index("event_id").mission_id).to_numpy()
    meta = json.loads((MODELS / "meta.json").read_text())
    thresholds = json.loads((MODELS / "thresholds.json").read_text())
    metrics = json.loads((OUT / "metrics.json").read_text())
    F, S = build_features(tr), screening_features(tr)
    Fte, Ste = build_features(te), screening_features(te)
    sets = {"full": (F[meta["full_features"]], Fte[meta["full_features"]]),
            "screen": (S[meta["screen_features"]], Ste[meta["screen_features"]])}
    raw_model = RiskModel.load(MODELS)
    calib, quality, cv_cal, leave_mission_out = {}, {}, {}, None
    for name, (X, Xte) in sets.items():
        log(f"OOF classifiers for {name}")
        oof = oof_classifiers(X, ytr, tr.event_id, meta["best_iterations"][name], N_FOLDS)
        calib[name], quality[name], cal_oof = {}, {}, {}
        for h, key, thr in (("clf_elev", "elev", PC_ELEVATED), ("clf_crit", "crit", PC_CRITICAL)):
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", increasing=True)
            iso.fit(oof[h][mask], (ytr[mask] >= thr).astype(float))
            xs = np.r_[0.0, iso.X_thresholds_, 1.0]
            ys = np.r_[iso.predict([0.0])[0], iso.y_thresholds_, iso.predict([1.0])[0]]
            calib[name][key] = {"x": [round(float(v), 6) for v in xs], "y": [round(float(v), 6) for v in ys]}
            cal_oof[h] = np.interp(oof[h], xs, ys)
            p_raw = raw_model.models[f"{name}_{h}"].predict(Xte)
            p_cal = np.interp(p_raw, xs, ys)
            yt = (yte >= thr).astype(float)
            quality[name][key] = {"brier_raw": round(brier_score_loss(yt, p_raw), 5),
                                  "brier_cal": round(brier_score_loss(yt, p_cal), 5),
                                  "ece_raw": round(ece(p_raw, yt), 5), "ece_cal": round(ece(p_cal, yt), 5)}
        thr_new = tune_thresholds(cal_oof["clf_elev"][mask], cal_oof["clf_crit"][mask],
                                  tr.miss_distance[mask], tier_tr[mask])
        thresholds[name] = thr_new
        cv_cal[name] = tier_metrics(tier_tr[mask], decide(cal_oof["clf_elev"], cal_oof["clf_crit"],
                                                          tr.miss_distance, thr_new)[mask])
        log(f"  {name}: calibrated thresholds {thr_new}; quality {quality[name]}")
        if name != "full":
            continue
        log("leave-mission-out CV")
        lmo = {h: np.zeros(len(ytr)) for h in ("clf_elev", "clf_crit")}
        folds = []
        for a, b in GroupKFold(6).split(X, groups=tr_mission):
            for h in lmo:
                key = "elev" if h == "clf_elev" else "crit"
                m = fit_head(h, X.iloc[a], ytr[a], n_estimators=meta["best_iterations"]["full"][h])
                lmo[h][b] = np.interp(predict_head(m, h, X.iloc[b]), calib["full"][key]["x"],
                                      calib["full"][key]["y"])
            bm = np.zeros(len(ytr), bool)
            bm[b] = True
            bm &= mask
            pt = decide(lmo["clf_elev"], lmo["clf_crit"], tr.miss_distance, thr_new)
            fm = tier_metrics(tier_tr[bm], pt[bm])
            fm.pop("confusion")
            folds.append({"held_out_missions": sorted(int(v) for v in np.unique(tr_mission[b])), **fm})
            log(f"  fold missions {folds[-1]['held_out_missions']}: F1 {fm['macro_f1']} missedC {fm['missed_critical']}")
        pred_lmo = decide(lmo["clf_elev"], lmo["clf_crit"], tr.miss_distance, thr_new)
        leave_mission_out = {
            "description": "6-fold GroupKFold by mission_id: every satellite in a validation fold is unseen in "
                           "training. mission_id is used only for splitting, never as a model input. Scored on "
                           "CDMs >= 2 days to TCA with calibrated probabilities and thresholds.",
            "overall": tier_metrics(tier_tr[mask], pred_lmo[mask]),
            "event_grouped_cv": cv_cal["full"],
            "distance_rule_same_rows": tier_metrics(tier_tr[mask], md_tier(tr.miss_distance)[mask]),
            "folds": folds,
        }
    meta["prob_calibration"] = calib
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    (MODELS / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    metrics["thresholds"] = thresholds
    metrics["calibration_quality"] = quality
    metrics["leave_mission_out"] = leave_mission_out
    for mth in metrics["methods"]:
        for name, label in (("full", "LightGBM, per-CDM physics features"), ("screen", "LightGBM, screening")):
            if mth["name"].startswith(label):
                mth["cv"] = cv_cal[name]
                mth["description"] = ("quantile regression + isotonic-calibrated classifiers, thresholds "
                                      f"{thresholds[name]}")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pick = ("macro_f1", "f2_critical", "missed_critical", "false_alarms_critical")
    print(json.dumps({"quality": quality, "thresholds": thresholds,
                      "lmo_overall": {k: leave_mission_out["overall"][k] for k in pick},
                      "event_cv": {k: cv_cal["full"][k] for k in pick},
                      "distance_rule": {k: leave_mission_out["distance_rule_same_rows"][k] for k in pick}},
                     indent=1))


def event_only():
    """Re-fit only the event-level forecaster (needs out-of-fold Pc predictions of the full model)."""
    tr = load(DATA / "train_data.csv")
    ytr = tr[LABEL].to_numpy()
    F = build_features(tr)
    full_cols = json.loads((MODELS / "meta.json").read_text())["full_features"]
    thr = json.loads((MODELS / "thresholds.json").read_text())["full"]
    log("out-of-fold predictions for the full model")
    oof, _ = cross_validate(F[full_cols], ytr, tr.event_id)
    ef = event_level(tr, F, ytr, oof, list(F.columns), thr)
    path = OUT / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics["event_forecast"] = ef
    path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: {m: ef[k][m] for m in ["macro_f1", "f2_critical", "missed_critical",
                                                 "false_alarms_critical"]}
                      for k in ef if isinstance(ef[k], dict)}, indent=1))


if __name__ == "__main__":
    if "--score-only" in sys.argv:
        score_only()
    elif "--event-only" in sys.argv:
        event_only()
    elif "--calibrate-only" in sys.argv:
        calibrate_only()
    elif "--robustness" in sys.argv:
        robustness()
    else:
        run()
