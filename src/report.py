"""Render outputs/metrics.json as outputs/eval_report.md (python src/report.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = ["Safe", "Elevated", "Critical"]


def pct(v):
    return f"{100 * v:.1f}%"


def methods_table(methods, split):
    rows = ["| Method | Kind | Macro-F1 | F2 (Critical) | Recall S / E / C | Missed Critical | Critical false alarms |",
            "|---|---|---|---|---|---|---|"]
    for m in methods:
        r = m[split]
        rec = " / ".join(f"{v:.3f}" for v in r["recall"])
        rows.append(f"| {m['name']} | {m['kind']} | {r['macro_f1']:.4f} | {r['f2_critical']:.4f} | {rec} | "
                    f"{r['missed_critical']} | {r['false_alarms_critical']} |")
    return "\n".join(rows)


def confusion(cm):
    rows = ["| true \\ predicted | Safe | Elevated | Critical |", "|---|---|---|---|"]
    rows += [f"| **{TIERS[i]}** | " + " | ".join(f"{v:,}" for v in row) + " |" for i, row in enumerate(cm)]
    return "\n".join(rows)


def main():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    chosen = next(x for x in m["methods"] if x["name"] == m["chosen_method"])
    base = m["methods"][0]
    ef = m["event_forecast"]
    rq, reg, pcd = m["review_queue"], m["regression"], m["pc_driven_escalations"]
    lines = [
        "# Evaluation report — conjunction risk evaluator",
        "",
        f"Generated {m['generated_at']} by `src/train.py`.",
        "",
        f"* **Test set:** `{m['eval_set']['name']}`, {m['eval_set']['n_rows']:,} CDMs / "
        f"{m['eval_set']['n_events']:,} events, true tiers S/E/C = "
        f"{' / '.join(f'{v:,}' for v in m['eval_set']['true_tier_counts'])}",
        f"* **Validation:** {m['cv_set']['name']} ({m['cv_set']['n_rows']:,} CDMs)",
        f"* **Decision thresholds (tuned on CV for F2):** {m['thresholds']['full']}",
        "",
        "## Tier classification — held-out test set",
        "",
        methods_table(m["methods"], "test"),
        "",
        "## Tier classification — grouped cross-validation on train",
        "",
        methods_table(m["methods"], "cv"),
        "",
        f"## Confusion matrix — {chosen['name']} (test)",
        "",
        confusion(chosen["test"]["confusion"]),
        "",
        f"## Confusion matrix — {base['name']} (test)",
        "",
        confusion(base["test"]["confusion"]),
        "",
        "## Where the model matters",
        "",
        f"* {pcd['n_rows_where_pc_raises_tier']:,} test CDMs have a true tier *higher* than the miss-distance rule "
        f"gives (risk hidden in Pc). The model catches {pcd['caught_by_model']:,} of them "
        f"({pct(pcd['caught_by_model'] / max(pcd['n_rows_where_pc_raises_tier'], 1))}); the distance rule catches 0.",
        f"* Review queue: {pct(rq['needs_review_rate'])} of CDMs are flagged *needs review* (80% Pc interval straddles "
        f"a tier boundary). Error rate inside the queue {pct(rq['error_rate_in_review_queue'])} vs "
        f"{pct(rq['error_rate_auto_decided'])} for auto-decided CDMs — the flag concentrates the mistakes where an "
        "analyst looks.",
        f"* Safety net: of {rq.get('missed_critical', 0)} Critical CDMs the model under-classifies, "
        f"{rq.get('missed_critical_flagged_for_review', 0)} are still put in the review queue (max-Pc bound >= 1e-4 "
        f"or straddling interval); **{rq.get('missed_critical_silent', 0)} are missed silently**.",
        f"* log10 Pc regression: MAE {reg['test_mae_log_pc']} (all), {reg['test_mae_log_pc_high_risk']} on true "
        f"Pc >= 1e-6. 80% interval coverage {pct(reg['interval_coverage_80'])} "
        f"({pct(reg['interval_coverage_80_nonfloor'])} excluding the -30 floor).",
        "",
        "## Event-level forecast (tier at TCA from CDMs >= 2 days out)",
        "",
        ef["description"],
        "",
        "| Method | Macro-F1 | F2 (Critical) | Missed Critical | Critical false alarms |",
        "|---|---|---|---|---|",
    ]
    for k, name in [("persistence_true_current_tier", "Latest Risk Prediction (uses `risk`, reference only)"),
                    ("persistence_model_current_tier", "Persistence of model's current tier"),
                    ("forecast_model", "Forecast model")]:
        r = ef[k]
        lines.append(f"| {name} | {r['macro_f1']:.4f} | {r['f2_critical']:.4f} | {r['missed_critical']} | "
                     f"{r['false_alarms_critical']} |")
    lmo = m.get("leave_mission_out")
    if lmo:
        pick = [("Leave-mission-out CV (unseen satellites)", lmo["overall"]),
                ("Event-grouped CV (same rows)", lmo["event_grouped_cv"]),
                ("Miss-distance rule (same rows)", lmo["distance_rule_same_rows"])]
        lines += ["", "## Generalisation to unseen missions", "", lmo["description"], "",
                  "| Validation | Macro-F1 | F2 (Critical) | Missed Critical | Critical false alarms |",
                  "|---|---|---|---|---|"]
        lines += [f"| {n} | {r['macro_f1']:.4f} | {r['f2_critical']:.4f} | {r['missed_critical']} | "
                  f"{r['false_alarms_critical']} |" for n, r in pick]
        lines += ["", "| Held-out missions | CDMs | Macro-F1 | Missed Critical | Critical false alarms |",
                  "|---|---|---|---|---|"]
        lines += [f"| {', '.join(map(str, f['held_out_missions']))} | {f['n']:,} | {f['macro_f1']:.4f} | "
                  f"{f['missed_critical']} | {f['false_alarms_critical']} |" for f in lmo["folds"]]
    cq = m.get("calibration_quality")
    if cq:
        lines += ["", "## Probability calibration (test)", "",
                  "Isotonic maps fitted on event-grouped out-of-fold predictions (train only).", "",
                  "| Model | Head | Brier raw | Brier calibrated | ECE raw | ECE calibrated |", "|---|---|---|---|---|---|"]
        for name, heads in cq.items():
            for h, q in heads.items():
                lines.append(f"| {name} | P(Pc >= {'1e-4' if h == 'crit' else '1e-6'}) | {q['brier_raw']} | "
                             f"{q['brier_cal']} | {q['ece_raw']} | {q['ece_cal']} |")
    lines += ["", "## Failure cases (test)", "",
              "| event | days to TCA | miss dist (m) | true | predicted | est. log Pc | true log Pc | P(crit) | why |",
              "|---|---|---|---|---|---|---|---|---|"]
    for f in m["failure_cases"][:20]:
        lines.append(f"| {f['event_id']} | {f['time_to_tca']:.2f} | {f['miss_distance']:,.0f} | {TIERS[f['true_tier']]} "
                     f"| {TIERS[f['pred_tier']]} | {f['log_pc']:.2f} | {f['true_log_pc']:.2f} | {f['p_critical']:.2f} "
                     f"| {f['why']} |")
    lines += ["", "## Top features (gain, Critical classifier)", "",
              "| feature | gain % |", "|---|---|"]
    lines += [f"| {x['feature']} | {x['gain_pct']} |" for x in m["feature_importance"]]
    lines += ["", "## Notes", ""] + [f"* {n}" for n in m["notes"]]
    (ROOT / "outputs" / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote outputs/eval_report.md")


if __name__ == "__main__":
    main()
