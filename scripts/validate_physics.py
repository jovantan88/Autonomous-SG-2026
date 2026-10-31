"""Validate src/physics.py against ESA's own columns (mahalanobis_distance, max_risk_estimate,
max_risk_scaling, risk) on data/train_data.csv and data/test_data.csv.

Writes outputs/physics_validation.json and prints a summary.
    python scripts/validate_physics.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import physics as P  # noqa: E402
from features import TIERS, risk_tier  # noqa: E402

OUT = ROOT / "outputs" / "physics_validation.json"
CORR_COLS = [f"{p}_{k}" for p in "tc" for k in ("ct_r", "cn_r", "cn_t")]


def _f(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def load():
    return {s: pd.read_csv(ROOT / "data" / f"{s}_data.csv") for s in ("train", "test")}


def valid_cov_mask(df):
    ph, mi = P.covariance_status(df)
    return ~ph & ~mi


# ------------------------------------------------------------------ 1. frame convention
def frame_check(df):
    ok = valid_cov_mask(df)
    d = df[ok]
    p, v = d[P.POS_COLS].to_numpy(float), d[P.VEL_COLS].to_numpy(float)
    ref = d["mahalanobis_distance"].to_numpy(float)
    res = {}
    for frame in ("target_rtn", "chaser_rtn_circular", "chaser_rtn_no_radius_term", "chaser_rtn"):
        C = P.combined_covariance(d, frame=frame)
        x2, C2, _ = P.encounter_plane(p, v, C)
        with np.errstate(invalid="ignore"):
            m3 = P.mahalanobis_3d(p, C)
        m2 = P.mahalanobis_2d(x2, C2)
        for kind, m in (("3d", m3), ("2d", m2)):
            lr = np.abs(np.log10(m / ref))
            res[f"{frame}/{kind}"] = {
                "median_abs_log10_ratio": _f(np.nanmedian(lr), 6),
                "p90": _f(np.nanpercentile(lr, 90), 6),
                "p99": _f(np.nanpercentile(lr, 99), 6),
                "frac_within_0.1pct": _f(np.nanmean(lr < np.log10(1.001))),
                "frac_within_1pct": _f(np.nanmean(lr < np.log10(1.01))),
            }
    res["n_rows"] = int(ok.sum())
    return res


# ------------------------------------------------------------------ 2. HBR candidates
HBR_CANDIDATES = {
    "(t_span + c_span) / 2": lambda d: (d.t_span + d.c_span) / 2,
    "t_span + c_span": lambda d: d.t_span + d.c_span,
    "t_span / 2": lambda d: d.t_span / 2,
    "t_span": lambda d: d.t_span,
    "sqrt(t_span^2 + c_span^2) / 2": lambda d: np.hypot(d.t_span, d.c_span) / 2,
    "sqrt(t_span^2 + c_span^2)": lambda d: np.hypot(d.t_span, d.c_span),
}


IMPLIED_LABEL = "implied by max_risk_estimate (per CDM; CALIBRATED on ESA output)"
HBR_MODES = {
    "auto": "radius solved per CDM from ESA's max_risk_estimate: calibrated on an ESA output, so its "
            "agreement with ESA risk is partly circular. Label results 'radius calibrated on ESA'.",
    "spans": "(t_span + c_span) / 2 from the CDM inputs only: the independent reproduction of ESA risk.",
}


def pc_err_stats(lp, risk):
    e = lp - risk
    s30 = risk > P.LOG10_PC_FLOOR
    s6 = risk >= -6
    out = {}
    for name, s in (("risk>-30", s30), ("risk>=-6", s6)):
        ee = e[s]
        out[name] = {
            "n": int(s.sum()),
            "MAE": _f(np.nanmean(np.abs(ee))),
            "median_abs_err": _f(np.nanmedian(np.abs(ee)), 5),
            "bias_median": _f(np.nanmedian(ee), 5),
            "frac_within_0.1": _f(np.nanmean(np.abs(ee) < 0.1)),
            "frac_within_0.5": _f(np.nanmean(np.abs(ee) < 0.5)),
            "spearman": _f(spearmanr(lp[s], risk[s], nan_policy="omit").statistic),
        }
    return out


def hbr_check(df, out_auto, x2, C2):
    """Agreement of each HBR candidate with ESA risk; implied-HBR agreement for each candidate."""
    ok = out_auto["pc_valid"].to_numpy()
    d = df[ok]
    risk = d["risk"].to_numpy(float)
    R_impl = out_auto["hbr_m"].to_numpy()[ok]
    res = {}
    for name, fn in HBR_CANDIDATES.items():
        R = fn(d).to_numpy(float)
        lp = P.log10_pc_2d(x2[ok], C2[ok], R)
        lr = np.log10(R_impl / R)
        res[name] = {"pc_vs_risk": pc_err_stats(lp, risk)["risk>-30"],
                     "implied_hbr_ratio_median_log10": _f(np.median(lr), 5),
                     "frac_implied_hbr_within_1pct": _f(np.mean(np.abs(lr) < np.log10(1.01)))}
    res[IMPLIED_LABEL] = {
        "pc_vs_risk": pc_err_stats(out_auto["log_pc_phys"].to_numpy()[ok], risk)["risk>-30"],
        "implied_hbr_ratio_median_log10": 0.0, "frac_implied_hbr_within_1pct": 1.0}
    return res


# ------------------------------------------------------------------ 4. validation table
def tier_stats(lp, risk, md):
    tt = risk_tier(risk, md)
    tp = risk_tier(lp, md)
    cm = confusion_matrix(tt, tp, labels=[0, 1, 2])
    return {
        "confusion_true_rows_pred_cols": {TIERS[i]: {TIERS[j]: int(cm[i, j]) for j in range(3)} for i in range(3)},
        "macro_f1": _f(f1_score(tt, tp, labels=[0, 1, 2], average="macro"), 5),
        "accuracy": _f(np.mean(tt == tp), 5),
        "missed_critical": int(((tt == 2) & (tp < 2)).sum()),
        "critical_false_alarms": int(((tp == 2) & (tt < 2)).sum()),
        "n_true_critical": int((tt == 2).sum()),
        # Pc rule alone (miss-distance rule removed) - harder test of the Pc value
        "pc_rule_only": {
            "missed_critical": int(((risk >= -4) & (lp < -4)).sum()),
            "false_critical": int(((lp >= -4) & (risk < -4)).sum()),
            "missed_elevated_or_worse": int(((risk >= -6) & (lp < -6)).sum()),
            "false_elevated_or_worse": int(((lp >= -6) & (risk < -6)).sum()),
            "n_risk_ge_-4": int((risk >= -4).sum()),
            "n_risk_ge_-6": int((risk >= -6).sum()),
        },
    }


def validation(df, out, out_spans):
    ok = out["pc_valid"].to_numpy()
    risk = df["risk"].to_numpy(float)[ok]
    md = df["miss_distance"].to_numpy(float)[ok]
    lp = out["log_pc_phys"].to_numpy()[ok]
    res = {"n_rows": int(len(df)), "n_pc_valid": int(ok.sum()),
           "pc_error": pc_err_stats(lp, risk),
           "floor_agreement": _f(np.mean((risk <= -30) == (lp <= -30)), 5),
           "tiers": tier_stats(lp, risk, md)}
    lp_sp = out_spans["log_pc_phys"].to_numpy()[ok]
    res["spans_hbr"] = {"pc_error": pc_err_stats(lp_sp, risk), "tiers": tier_stats(lp_sp, risk, md)}
    # independent check: max-Pc with the span HBR vs max_risk_estimate / max_risk_scaling, for ESA's
    # definition (k0 = m_2D^2 / 2, Pc there) and for the exact maximum over k
    mre = df["max_risk_estimate"].to_numpy(float)[ok]
    mrs = df["max_risk_scaling"].to_numpy(float)[ok]
    k0 = out_spans["max_scale_esa"].to_numpy()[ok]
    ratio = out_spans["hbr_m"].to_numpy()[ok] ** 2 / (k0 * out_spans["sigma_minor"].to_numpy()[ok] ** 2)
    big = ratio > 0.05                                    # HBR comparable to the scaled sigma
    # rows where the span HBR is ESA's radius (Pc reproduces risk) and the two definitions differ
    lp_sp = out_spans["log_pc_phys"].to_numpy()[ok]
    gap = out_spans["log_pc_max_phys"].to_numpy()[ok] - out_spans["log_pc_max_esa"].to_numpy()[ok]
    disc = big & (np.abs(lp_sp - df["risk"].to_numpy(float)[ok]) < 0.003) & (gap > 0.002)
    mp = {"n_rows": int(ok.sum()), "n_hbr2_over_k0_sigma_minor2_gt_0.05": int(big.sum()),
          "n_discriminating_rows": int(disc.sum()),
          "note": "max_risk_scaling = m_2D^2/2 (analytic small-HBR optimum) and max_risk_estimate = Pc "
                  "at that k, not the exact maximum over k"}
    for name, lpm, km in (("esa_definition(k=m2d^2/2)", out_spans["log_pc_max_esa"], out_spans["max_scale_esa"]),
                          ("exact_max_over_k", out_spans["log_pc_max_phys"], out_spans["max_scale"])):
        em = lpm.to_numpy()[ok] - mre
        ek = np.log10(km.to_numpy()[ok] / mrs)
        mp[name] = {
            "pc_MAE": _f(np.nanmean(np.abs(em))), "pc_median_abs_err": _f(np.nanmedian(np.abs(em)), 5),
            "pc_frac_within_0.01": _f(np.nanmean(np.abs(em) < 0.01)),
            "pc_frac_within_0.1": _f(np.nanmean(np.abs(em) < 0.1)),
            "scale_vs_max_risk_scaling_median_abs_log10": _f(np.nanmedian(np.abs(ek)), 6),
            "scale_frac_within_1pct": _f(np.nanmean(np.abs(ek) < np.log10(1.01))),
            "large_hbr_rows_scale_median_abs_log10": _f(np.nanmedian(np.abs(ek[big])), 5) if big.any() else None,
            "discriminating_rows_pc_median_abs_err": _f(np.nanmedian(np.abs(em[disc])), 5) if disc.any() else None,
        }
    mp["exact_minus_esa_max_pc_max_decades"] = _f(np.nanmax(gap), 4)
    res["max_pc_vs_esa_columns_spans_hbr"] = mp
    em_auto = out["log_pc_max_esa"].to_numpy()[ok] - mre
    impl = out["hbr_source"].to_numpy()[ok] == "implied"
    res["max_pc_vs_max_risk_estimate_implied_hbr"] = {
        "MAE_implied_rows": _f(np.nanmean(np.abs(em_auto[impl])), 7),
        "note": "HBR solved from max_risk_estimate: consistency check (zero by construction)"}
    m3 = np.abs(np.log10(out["mahal_3d"].to_numpy()[ok] / df["mahalanobis_distance"].to_numpy(float)[ok]))
    res["mahal_3d_vs_esa_median_abs_log10"] = _f(np.nanmedian(m3), 6)
    # rows not valid: what does ESA report there?
    ph, mi = P.covariance_status(df)
    inv = {}
    for name, mask in (("placeholder_sigma", ph & ~mi), ("nan_covariance", mi),
                       ("non_pd_covariance", ~ph & ~mi & ~ok)):
        r = df["risk"].to_numpy(float)[mask]
        lpm = out["log_pc_phys"].to_numpy()[mask]
        inv[name] = {"n": int(mask.sum()),
                     "esa_risk_median": _f(np.median(r)) if len(r) else None,
                     "esa_risk_range": [_f(r.min()), _f(r.max())] if len(r) else None,
                     "frac_esa_floor": _f(np.mean(r <= -30)) if len(r) else None,
                     "ours_minus_esa_median_abs": _f(np.nanmedian(np.abs(lpm - r))) if np.isfinite(lpm).any() else None}
    res["invalid_rows"] = inv
    return res


# ------------------------------------------------------------------ worst cases
def worst_cases(dfs, outs, k=10):
    frames = []
    for split in ("train", "test"):
        df, out = dfs[split], outs[split]
        d = pd.concat([df, out], axis=1)
        d = d[d.pc_valid].copy()
        d["split"] = split
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["err"] = d.log_pc_phys - d.risk
    d["mahal_err_pct"] = 100 * (d.mahal_3d / d.mahalanobis_distance - 1)
    d["max_abs_corr"] = d[CORR_COLS].abs().max(axis=1)
    d["md_over_hbr"] = d.miss_distance / d.hbr_m
    d["hbr_over_sigma_minor"] = d.hbr_m / d.sigma_minor
    flags = {
        "degenerate_correlation(|rho|>0.999)": d.max_abs_corr > 0.999,
        "low_relative_speed(<500 m/s)": d.relative_speed < 500,
        "miss_within_50_hbr(max_risk_estimate inconsistent)": d.md_over_hbr < 50,
        "frame_mismatch(|mahal err|>0.2%)": d.mahal_err_pct.abs() > 0.2,
        "far_tail(mahal_2d>8)": d.mahal_2d > 8,
        "hbr_comparable_to_sigma(>0.5)": d.hbr_over_sigma_minor > 0.5,
    }

    def explain(i):
        reasons = [k for k, m in flags.items() if m.iloc[i]]
        return "; ".join(reasons) if reasons else "unexplained"

    bad = d.err.abs() > 0.1
    summary = {"n_valid": int(len(d)), "n_abs_err_gt_0.1": int(bad.sum()), "n_abs_err_gt_0.5": int((d.err.abs() > 0.5).sum()),
               "n_abs_err_gt_1": int((d.err.abs() > 1).sum()),
               "flag_rates_in_bad_vs_all": {k: {"bad": _f(m[bad].mean()), "all": _f(m.mean())} for k, m in flags.items()},
               "unexplained_frac_of_bad": _f(np.mean([explain(i) == "unexplained" for i in np.flatnonzero(bad)]))}
    top = d.reindex(d.err.abs().sort_values(ascending=False).index).head(k)
    cols = ["split", "event_id", "time_to_tca", "risk", "log_pc_phys", "err", "max_risk_estimate", "miss_distance",
            "relative_speed", "mahalanobis_distance", "mahal_3d", "mahal_2d", "sigma_major", "sigma_minor", "hbr_m",
            "hbr_spans_m", "max_abs_corr", "md_over_hbr"]
    rows = []
    for idx, r in top.iterrows():
        rec = {c: (r[c] if isinstance(r[c], str) else _f(r[c], 4)) for c in cols}
        rec["event_id"] = int(r.event_id)
        rec["explanation"] = explain(d.index.get_loc(idx))
        rows.append(rec)
    return summary, rows


# ------------------------------------------------------------------ manoeuvre example
def manoeuvre_example(df, out):
    d = pd.concat([df, out], axis=1)
    cand = d[d.pc_valid & (d.log_pc_phys >= -4) & (d.time_to_tca >= 1.0)]
    if cand.empty:
        cand = d[d.pc_valid].nlargest(1, "log_pc_phys")
    row = cand.sort_values("log_pc_phys", ascending=False).iloc[0]
    src = df.loc[row.name]
    t0 = time.perf_counter()
    opts = P.manoeuvre_options(src, target_log_pc=-5.0, burn_lead_orbits=(0.5, 1, 2, 3), max_dv=0.5)
    dt = time.perf_counter() - t0
    t0 = time.perf_counter()
    opts_nom = P.manoeuvre_options(src, target_log_pc=-5.0, burn_lead_orbits=(0.5, 1, 2, 3), max_dv=0.5,
                                   robust=False)
    dt_nom = time.perf_counter() - t0
    info = {k: (_f(src[k], 4) if not isinstance(src[k], str) else src[k]) for k in
            ("event_id", "time_to_tca", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
             "relative_position_r", "relative_position_t", "relative_position_n",
             "relative_velocity_r", "relative_velocity_t", "relative_velocity_n", "t_j2k_sma")}
    info["test_row_index"] = int(row.name)
    info["log_pc_phys"] = _f(row.log_pc_phys)
    info["hbr_m"] = _f(row.hbr_m)
    info["sigma_major"] = _f(row.sigma_major, 2)
    info["sigma_minor"] = _f(row.sigma_minor, 2)
    best = P.best_manoeuvre(opts)
    return {"cdm": info, "options": opts, "best": best, "options_nominal_non_robust": opts_nom,
            "runtime_s": _f(dt, 3), "runtime_nominal_s": _f(dt_nom, 3),
            "assumptions": list(P.MANOEUVRE_ASSUMPTIONS)}


def manoeuvre_robustness(df, out, max_events=40, target=-5.0):
    """Robust vs nominal burns on the latest high-risk CDM of up to `max_events` test events."""
    d = pd.concat([df, out], axis=1)
    cand = d[d.pc_valid & (d.log_pc_phys >= -4) & (d.time_to_tca >= 1.0)]
    rows = cand.sort_values("time_to_tca").groupby("event_id", sort=False).head(1)
    rows = rows.sort_values("log_pc_phys", ascending=False).head(max_events)
    rec = {"nominal": [], "robust": []}
    t0 = time.perf_counter()
    for idx in rows.index:
        for mode, rb in (("nominal", False), ("robust", True)):
            for o in P.manoeuvre_options(df.loc[idx], target_log_pc=target, robust=rb):
                if o is None or not o["feasible"]:
                    continue
                for dname, od in o["options"].items():
                    if od is None:
                        continue
                    rec[mode].append({"row": int(idx), "lead": o["lead_orbits"], "dir": dname, "dv": od["abs_dv_mps"],
                                      "dv_nom": od["abs_dv_nominal_mps"], "short": od["log_pc_after_short"],
                                      "ecc": od["log_pc_after_ecc_worst"], "worst": od["log_pc_worst"],
                                      "sens": od["eccentricity_sensitive"], "best": od["abs_dv_mps"] == o["abs_dv_mps"]})
    dt = time.perf_counter() - t0
    res = {"n_cdms": int(len(rows)), "target_log_pc": target, "execution_error": P.EXECUTION_ERROR,
           "model_margin_decades": P.MODEL_MARGIN_DECADES, "runtime_s_per_call_avg": _f(dt / max(2 * len(rows), 1), 3)}
    for mode, L in rec.items():
        t = pd.DataFrame(L)
        if t.empty:
            res[mode] = {"n_options": 0}
            continue
        b = t[t.best]
        res[mode] = {
            "n_options": int(len(t)), "n_chosen": int(len(b)),
            "frac_options_meeting_target_at_0.9x_burn": _f(np.mean(t.short <= target + 1e-9)),
            "frac_options_meeting_target_at_worst_eccentricity": _f(np.mean(t.ecc <= target + 1e-9)),
            "frac_options_meeting_target_over_whole_uncertainty_set": _f(np.mean(t.worst <= target + 1e-9)),
            "max_log_pc_at_0.9x_burn_minus_target": _f(t.short.max() - target),
            "max_log_pc_worst_minus_target": _f(t.worst.max() - target),
            "chosen_median_log_pc_at_0.9x_burn_minus_target": _f(b.short.median() - target),
            "n_eccentricity_sensitive": int(t.sens.sum()),
            "median_dv_factor_vs_nominal": _f((t.dv / t.dv_nom).median()),
            "max_dv_factor_vs_nominal": _f((t.dv / t.dv_nom).max()),
        }
    return res


# ------------------------------------------------------------------ main
def main():
    t_load = time.perf_counter()
    dfs = load()
    print(f"loaded train {len(dfs['train'])} / test {len(dfs['test'])} CDMs in {time.perf_counter() - t_load:.1f}s")

    # runtime of the public entry point on all rows
    allrows = pd.concat([dfs["train"], dfs["test"]], ignore_index=True)
    t0 = time.perf_counter()
    P.full_pc(allrows)
    rt_auto = time.perf_counter() - t0
    t0 = time.perf_counter()
    P.full_pc(allrows, hbr="spans")
    rt_spans = time.perf_counter() - t0
    del allrows
    print(f"full_pc on {len(dfs['train']) + len(dfs['test'])} rows: {rt_auto:.1f}s (hbr=auto), {rt_spans:.1f}s (hbr=spans)")

    report = {"runtime_s": {"full_pc_all_rows_hbr_auto": _f(rt_auto, 2), "full_pc_all_rows_hbr_spans": _f(rt_spans, 2),
                            "n_rows": int(len(dfs["train"]) + len(dfs["test"]))},
              "settings": {"HBR_FORMULA": P.HBR_FORMULA, "SIGMA_PLACEHOLDER": P.SIGMA_PLACEHOLDER,
                           "LOG10_PC_FLOOR": P.LOG10_PC_FLOOR, "N_GL": P.N_GL, "N_GL_EASY": P.N_GL_EASY,
                           "WINDOW_NATS": P.WINDOW_NATS}}
    outs, outs_sp = {}, {}
    report["frame"], report["hbr_candidates"], report["validation"] = {}, {}, {}
    for split, df in dfs.items():
        outs[split] = P.full_pc(df)
        outs_sp[split] = P.full_pc(df, hbr="spans")
        report["frame"][split] = frame_check(df)
        C = P.combined_covariance(df)
        x2 = outs[split][["miss_plane_x", "miss_plane_y"]].to_numpy()
        _, C2, _ = P.encounter_plane(df[P.POS_COLS].to_numpy(float), df[P.VEL_COLS].to_numpy(float), C)
        report["hbr_candidates"][split] = hbr_check(df, outs[split], x2, C2)
        report["validation"][split] = validation(df, outs[split], outs_sp[split])
        src = outs[split]["hbr_source"].value_counts().to_dict()
        report["validation"][split]["hbr_source_counts"] = {k: int(v) for k, v in src.items()}

    summary, rows = worst_cases(dfs, outs)
    report["worst_cases"] = {"summary": summary, "top10": rows}
    report["manoeuvre_example"] = manoeuvre_example(dfs["test"], outs["test"])
    report["manoeuvre_robustness_test"] = manoeuvre_robustness(dfs["test"], outs["test"])
    report["hbr_modes"] = HBR_MODES
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    # ------------------------------------------------------------ printed summary
    print("\n== 1. Covariance frame (median |log10(ours / ESA mahalanobis_distance)|) ==")
    for split in dfs:
        fr = report["frame"][split]
        print(f"  {split}: " + ", ".join(f"{k}={v['median_abs_log10_ratio']:.2e}" for k, v in fr.items() if k != "n_rows"))
    print("\n== 2. HBR candidates (Pc vs ESA risk, rows risk > -30, pc_valid) ==")
    for split in dfs:
        print(f"  {split}")
        for k, v in report["hbr_candidates"][split].items():
            s = v["pc_vs_risk"]
            print(f"    {k:40s} MAE {s['MAE']:.4f}  median {s['median_abs_err']:.4f}  within0.1 {s['frac_within_0.1']:.4f}"
                  f"  | implied/candidate median log10 {v['implied_hbr_ratio_median_log10']:+.4f}"
                  f" within1% {v['frac_implied_hbr_within_1pct']:.3f}")
    print("\n== 3/4. Validation vs ESA risk (pc_valid rows; 'implied' = radius CALIBRATED on ESA "
          "max_risk_estimate, 'spans' = CDM inputs only) ==")
    hdr = (f"  {'split':6s} {'HBR':7s} {'subset':9s} {'n':>7s} {'MAE':>7s} {'medAE':>7s} {'<0.1':>6s} {'<0.5':>6s}"
           f" {'rho':>6s} | {'macroF1':>7s} {'missCrit':>8s} {'critFA':>6s}")
    print(hdr)
    for split in dfs:
        V = report["validation"][split]
        for hb, pe, ti in (("implied", V["pc_error"], V["tiers"]), ("spans", V["spans_hbr"]["pc_error"], V["spans_hbr"]["tiers"])):
            for sub in ("risk>-30", "risk>=-6"):
                s = pe[sub]
                print(f"  {split:6s} {hb:7s} {sub:9s} {s['n']:7d} {s['MAE']:7.4f} {s['median_abs_err']:7.4f} {s['frac_within_0.1']:6.3f}"
                      f" {s['frac_within_0.5']:6.3f} {s['spearman']:6.3f} | {ti['macro_f1']:7.4f} {ti['missed_critical']:8d}"
                      f" {ti['critical_false_alarms']:6d}")
        mx = V["max_pc_vs_esa_columns_spans_hbr"]
        for nm in ("esa_definition(k=m2d^2/2)", "exact_max_over_k"):
            q = mx[nm]
            print(f"  {split}: {nm:26s} (span HBR) vs max_risk_estimate MAE {q['pc_MAE']:.4f} median "
                  f"{q['pc_median_abs_err']:.5f} within0.01 {q['pc_frac_within_0.01']:.3f} | discriminating rows "
                  f"(n={mx['n_discriminating_rows']}) median {q['discriminating_rows_pc_median_abs_err']} | k vs "
                  f"max_risk_scaling median |log10| {q['scale_vs_max_risk_scaling_median_abs_log10']:.2e} within1% "
                  f"{q['scale_frac_within_1pct']:.4f} (large-HBR rows {q['large_hbr_rows_scale_median_abs_log10']})")
        print(f"  {split}: Pc-rule-only (implied HBR): {V['tiers']['pc_rule_only']}")
        print(f"  {split}: invalid rows {json.dumps(V['invalid_rows'])}")
    print("\n== Worst disagreements (pc_valid rows, implied HBR) ==")
    print("  " + json.dumps(summary))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(rows).drop(columns=["explanation"]).to_string(index=False))
    for r in rows:
        print(f"  {r['split']} event {r['event_id']} t-{r['time_to_tca']}d err {r['err']:+.3f}: {r['explanation']}")
    print("\n== Manoeuvre example (test CDM) ==")
    ex = report["manoeuvre_example"]
    print("  " + json.dumps(ex["cdm"]))
    for o, on, L in zip(ex["options"], ex["options_nominal_non_robust"], (0.5, 1, 2, 3)):
        if o is None:
            print(f"  lead {L} orbits: no along-track burn <= 0.5 m/s reaches Pc < 1e-5 robustly")
        elif not o["feasible"]:
            print(f"  lead {L} orbits: infeasible - {o['reason']}")
        else:
            nom = "" if on is None else (f" (nominal minimum {on['abs_dv_mps'] * 100:.2f} cm/s: 0.9x burn -> "
                                         f"{on['log_pc_after_short']:.2f}, worst {on['log_pc_worst']:.2f})")
            print(f"  lead {o['lead_orbits']} orbits ({o['lead_time_h']:.2f} h): {o['direction']} {o['abs_dv_mps'] * 100:.2f} cm/s"
                  f" -> log10 Pc {o['log_pc_before']:.2f} -> {o['log_pc_after']:.2f} (0.9x {o['log_pc_after_short']:.2f},"
                  f" 1.1x {o['log_pc_after_long']:.2f}, ecc worst {o['log_pc_after_ecc_worst']:.2f}, worst {o['log_pc_worst']:.2f}),"
                  f" miss {o['miss_distance_before_m']:.0f} -> {o['miss_distance_after_m']:.0f} m; {o['effect']};"
                  f" e|dT| {o['ecc_radial_uncertainty_m']:.1f} m{' [ECC-SENSITIVE]' if o['eccentricity_sensitive'] else ''}{nom}")
    b = ex["best"]
    if b is not None:
        print(f"  recommended: {b['lead_orbits']} orbit(s) before TCA, {b['direction']} {b['abs_dv_mps'] * 100:.2f} cm/s ({b['effect']})")
    print("\n== Manoeuvre robustness, test CDMs (latest high-risk CDM per event) ==")
    print("  " + json.dumps(report["manoeuvre_robustness_test"]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
