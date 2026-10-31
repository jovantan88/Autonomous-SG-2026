# Evaluation report — conjunction risk evaluator

Generated 2026-10-31T03:49:37+00:00 by `src/train.py`.

* **Test set:** `test_data.csv`, 24,484 CDMs / 2,167 events, true tiers S/E/C = 15,756 / 6,501 / 2,227
* **Validation:** train_data.csv, 5-fold GroupKFold by event, scored on CDMs >= 2 days to TCA (111,939 CDMs)
* **Decision thresholds (tuned on CV for F2):** {'p_critical': 0.1, 'p_elevated': 0.19}

## Tier classification — held-out test set

| Method | Kind | Macro-F1 | F2 (Critical) | Recall S / E / C | Missed Critical | Critical false alarms |
|---|---|---|---|---|---|---|
| Miss-distance rule only | baseline | 0.9518 | 0.9670 | 1.000 / 0.842 / 0.959 | 91 | 0 |
| Max-Pc bound as Pc | baseline | 0.8415 | 0.9847 | 0.712 / 0.975 / 1.000 | 0 | 173 |
| Isotropic Gaussian Pc (physics) | baseline | 0.9517 | 0.9670 | 1.000 / 0.843 / 0.959 | 91 | 1 |
| LightGBM, per-CDM physics features (chosen) | model | 0.9625 | 0.9904 | 0.954 / 0.973 / 0.996 | 8 | 76 |
| LightGBM, + causal event-history features (ablation) | ablation | 0.9658 | 0.9891 | 0.953 / 0.984 / 0.991 | 21 | 38 |
| LightGBM, screening inputs only (live SOCRATES mode) | model | 0.9400 | 0.9844 | 0.926 / 0.945 / 0.988 | 26 | 70 |

## Tier classification — grouped cross-validation on train

| Method | Kind | Macro-F1 | F2 (Critical) | Recall S / E / C | Missed Critical | Critical false alarms |
|---|---|---|---|---|---|---|
| Miss-distance rule only | baseline | 0.9567 | 0.9701 | 1.000 / 0.848 / 0.963 | 304 | 0 |
| Max-Pc bound as Pc | baseline | 0.8348 | 0.9841 | 0.737 / 0.974 / 1.000 | 0 | 661 |
| Isotropic Gaussian Pc (physics) | baseline | 0.9566 | 0.9699 | 1.000 / 0.849 / 0.963 | 304 | 6 |
| LightGBM, per-CDM physics features (chosen) | model | 0.9645 | 0.9927 | 0.964 / 0.970 / 0.999 | 7 | 274 |
| LightGBM, + causal event-history features (ablation) | ablation | 0.9677 | 0.9939 | 0.964 / 0.980 / 0.998 | 20 | 170 |
| LightGBM, screening inputs only (live SOCRATES mode) | model | 0.9371 | 0.9863 | 0.934 / 0.943 / 0.992 | 64 | 310 |

## Confusion matrix — LightGBM, per-CDM physics features (chosen) (test)

| true \ predicted | Safe | Elevated | Critical |
|---|---|---|---|
| **Safe** | 15,029 | 724 | 3 |
| **Elevated** | 103 | 6,325 | 73 |
| **Critical** | 0 | 8 | 2,219 |

## Confusion matrix — Miss-distance rule only (test)

| true \ predicted | Safe | Elevated | Critical |
|---|---|---|---|
| **Safe** | 15,756 | 0 | 0 |
| **Elevated** | 1,025 | 5,476 | 0 |
| **Critical** | 8 | 83 | 2,136 |

## Where the model matters

* 1,116 test CDMs have a true tier *higher* than the miss-distance rule gives (risk hidden in Pc). The model catches 1,005 of them (90.1%); the distance rule catches 0.
* Review queue: 4.2% of CDMs are flagged *needs review* (80% Pc interval straddles a tier boundary). Error rate inside the queue 26.6% vs 2.7% for auto-decided CDMs — the flag concentrates the mistakes where an analyst looks.
* Safety net: of 8 Critical CDMs the model under-classifies, 8 are still put in the review queue (max-Pc bound >= 1e-4 or straddling interval); **0 are missed silently**.
* log10 Pc regression: MAE 3.244 (all), 1.121 on true Pc >= 1e-6. 80% interval coverage 81.8% (82.6% excluding the -30 floor).

## Event-level forecast (tier at TCA from CDMs >= 2 days out)

Predict the tier at the final CDM (< 1 day to TCA) from the latest CDM issued >= 2 days before TCA (the moment a manoeuvre must be planned). Train events meeting the Kelvins test-set constraints; grouped 5-fold CV. test_data has no final CDMs so this cannot be scored on test.

| Method | Macro-F1 | F2 (Critical) | Missed Critical | Critical false alarms |
|---|---|---|---|---|
| Latest Risk Prediction (uses `risk`, reference only) | 0.8684 | 0.8162 | 99 | 113 |
| Persistence of model's current tier | 0.8591 | 0.8156 | 98 | 120 |
| Forecast model | 0.8679 | 0.8195 | 97 | 112 |

## Generalisation to unseen missions

6-fold GroupKFold by mission_id: every satellite in a validation fold is unseen in training. mission_id is used only for splitting, never as a model input. Scored on CDMs >= 2 days to TCA with calibrated probabilities and thresholds.

| Validation | Macro-F1 | F2 (Critical) | Missed Critical | Critical false alarms |
|---|---|---|---|---|
| Leave-mission-out CV (unseen satellites) | 0.9585 | 0.9907 | 16 | 320 |
| Event-grouped CV (same rows) | 0.9645 | 0.9927 | 7 | 274 |
| Miss-distance rule (same rows) | 0.9567 | 0.9701 | 304 | 0 |

| Held-out missions | CDMs | Macro-F1 | Missed Critical | Critical false alarms |
|---|---|---|---|---|
| 5 | 20,608 | 0.9939 | 0 | 8 |
| 15, 16, 22, 24 | 18,661 | 0.9833 | 0 | 6 |
| 1, 18, 20, 23 | 16,343 | 0.8837 | 8 | 220 |
| 2, 4 | 19,640 | 0.9556 | 0 | 47 |
| 7, 9, 13 | 18,038 | 0.9579 | 3 | 21 |
| 3, 6, 10, 14, 19 | 18,649 | 0.9742 | 5 | 18 |

## Probability calibration (test)

Isotonic maps fitted on event-grouped out-of-fold predictions (train only).

| Model | Head | Brier raw | Brier calibrated | ECE raw | ECE calibrated |
|---|---|---|---|---|---|
| full | P(Pc >= 1e-6) | 0.0349 | 0.03491 | 0.0147 | 0.01369 |
| full | P(Pc >= 1e-4) | 0.00377 | 0.00364 | 0.00189 | 0.0005 |
| screen | P(Pc >= 1e-6) | 0.07396 | 0.07429 | 0.02935 | 0.02865 |
| screen | P(Pc >= 1e-4) | 0.00565 | 0.00556 | 0.00309 | 0.00157 |

## Failure cases (test)

| event | days to TCA | miss dist (m) | true | predicted | est. log Pc | true log Pc | P(crit) | why |
|---|---|---|---|---|---|---|---|---|
| 1930 | 4.80 | 3,183 | Critical | Elevated | -8.13 | -3.91 | 0.00 | borderline: true Pc 1.22e-04 is within 0.1 decades of the 1e-4 threshold; flagged for analyst review (not auto-cleared); Critical only because true Pc 1.2e-04 >= 1e-4 while miss distance 3.2 km looks benign |
| 107 | 3.33 | 1,215 | Critical | Elevated | -3.83 | -3.93 | 0.08 | borderline: true Pc 1.16e-04 is within 0.1 decades of the 1e-4 threshold; flagged for analyst review (not auto-cleared); Critical only because true Pc 1.2e-04 >= 1e-4 while miss distance 1.2 km looks benign; model saw some risk (P(crit)=8%) but below threshold |
| 107 | 2.75 | 1,220 | Critical | Elevated | -4.16 | -3.94 | 0.08 | borderline: true Pc 1.16e-04 is within 0.1 decades of the 1e-4 threshold; flagged for analyst review (not auto-cleared); Critical only because true Pc 1.2e-04 >= 1e-4 while miss distance 1.2 km looks benign; model saw some risk (P(crit)=8%) but below threshold |
| 107 | 3.73 | 1,219 | Critical | Elevated | -3.90 | -3.94 | 0.08 | borderline: true Pc 1.16e-04 is within 0.1 decades of the 1e-4 threshold; flagged for analyst review (not auto-cleared); Critical only because true Pc 1.2e-04 >= 1e-4 while miss distance 1.2 km looks benign; model saw some risk (P(crit)=8%) but below threshold |
| 107 | 2.33 | 1,226 | Critical | Elevated | -5.58 | -3.94 | 0.08 | borderline: true Pc 1.15e-04 is within 0.1 decades of the 1e-4 threshold; flagged for analyst review (not auto-cleared); Critical only because true Pc 1.1e-04 >= 1e-4 while miss distance 1.2 km looks benign; model saw some risk (P(crit)=8%) but below threshold |
| 107 | 3.01 | 1,233 | Critical | Elevated | -3.61 | -3.94 | 0.08 | borderline: true Pc 1.14e-04 is within 0.1 decades of the 1e-4 threshold; flagged for analyst review (not auto-cleared); Critical only because true Pc 1.1e-04 >= 1e-4 while miss distance 1.2 km looks benign; model saw some risk (P(crit)=8%) but below threshold |
| 1930 | 4.22 | 3,265 | Critical | Elevated | -8.20 | -3.95 | 0.00 | borderline: true Pc 1.13e-04 is within 0.1 decades of the 1e-4 threshold; flagged for analyst review (not auto-cleared); Critical only because true Pc 1.1e-04 >= 1e-4 while miss distance 3.3 km looks benign |
| 1733 | 6.98 | 5,691 | Critical | Elevated | -4.36 | -4.00 | 0.08 | borderline: true Pc 1.00e-04 is within 0.1 decades of the 1e-4 threshold; flagged for analyst review (not auto-cleared); Critical only because true Pc 1.0e-04 >= 1e-4 while miss distance 5.7 km looks benign; model saw some risk (P(crit)=8%) but below threshold |
| 712 | 5.34 | 5,569 | Safe | Critical | -5.57 | -6.41 | 0.64 | flagged for analyst review (not auto-cleared) |
| 712 | 4.24 | 5,702 | Safe | Critical | -7.48 | -9.58 | 0.64 | flagged for analyst review (not auto-cleared) |
| 220 | 6.66 | 5,148 | Safe | Critical | -3.13 | -7.64 | 0.46 | flagged for analyst review (not auto-cleared) |

## Top features (gain, Critical classifier)

| feature | gain % |
|---|---|
| max_risk_estimate | 59.99 |
| radial_mahal | 8.34 |
| log_sigma_r_comb | 8.3 |
| log_md | 3.92 |
| pc_gap_to_max | 2.64 |
| frac_r | 2.24 |
| log_c_sigma_r | 1.78 |
| time_to_tca | 1.72 |
| log_hbr | 1.48 |
| log_pc_iso | 1.04 |
| relative_speed | 1.02 |
| log_t_sigma_r | 0.9 |
| pos_r | 0.9 |
| pos_t | 0.89 |
| frac_n | 0.87 |

## Notes

* Ground truth tier = worst of Pc rule and miss-distance rule, computed from the CDM's own risk column.
* The model never sees `risk`; it uses only the 13 allowed columns from the brief.
* Thresholds are tuned on cross-validation (never on test) to maximise F2, i.e. recall-weighted.
* Train/test differ by mission mix (Uriot et al. 2022, Sec. 4.5); CV-to-test gaps are expected.
* Live SOCRATES conjunctions are TLE-based screening products without covariance: the live tier is screening-grade and must be confirmed with CSpOC CDMs before any manoeuvre decision.
