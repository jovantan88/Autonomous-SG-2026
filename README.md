# Conjunction Watch — AI conjunction risk evaluator (Autonomous SG 2026, Airbus Challenge #1)

> *"How could we leverage AI-driven technologies for satellite collision avoidance?"*

A decision-support tool for the **conjunction-assessment analyst on shift**. Today that person receives
hundreds of conjunction data messages (CDMs) a day, opens each one, reads Pc and miss distance, and decides
by hand whether to escalate to flight dynamics. Conjunction Watch reads the CDM telemetry, estimates the collision
probability (Pc) **with an uncertainty interval**, applies the NASA / ESA tier rule, explains every decision,
and puts the uncertain ones in front of the analyst — who makes the call.

| Tier | Rule (from the brief, NASA CA Handbook / ESA SDO) | Action |
|---|---|---|
| **Critical** | Pc ≥ 1e-4 **or** miss distance < 1 km | Manoeuvre planning |
| **Elevated** | 1e-6 ≤ Pc < 1e-4 **or** miss distance 1–5 km | Monitor, re-screen on next CDM |
| **Safe** | Pc < 1e-6 **and** miss distance > 5 km | No action |

## What is in the box

| Part | What it does | Where |
|---|---|---|
| Pc estimator | LightGBM quantile regression (10/50/90 %) of log10 Pc + calibrated classifiers for P(Pc ≥ 1e-6) and P(Pc ≥ 1e-4), using **only the 13 allowed columns** | `src/features.py`, `src/train.py` |
| Tier decision | Deterministic rule = worst of (Pc rule, miss-distance rule). Thresholds on the probabilities tuned on cross-validation for **F2** (missing a Critical costs more than a false alarm) | `src/predict.py` |
| Explanations | Every assessment carries the rule that fired, the Pc estimate + interval and the probability vs threshold; a *needs review* flag when the interval straddles a tier boundary | `src/predict.py` |
| Event forecast | Predicts the tier at TCA from the latest CDM ≥ 2 days out (when a manoeuvre must be planned) | `src/train.py::event_level` |
| Live screening | Pulls live conjunctions from **CelesTrak SOCRATES Plus**, object types from **SATCAT**, TLEs from **CelesTrak GP** (SGP4 trajectories), **NOAA SWPC** F10.7 / Ap — the public sources listed at orbitalradar.com/data-sources | `src/live.py` |
| Dashboard | Cesium globe with both objects' trajectories around TCA, 24 h warning window, risk queue, CDM timelines, evaluation tab, manual CDM assessment, analyst decision log | `dashboard/`, `src/server.py` |

## Quick start

```bash
pip install -r requirements.txt
# put train_data.csv and test_data.csv (challenge Drive, Challenge #1/archive(1).zip) in data/
python src/train.py          # ~25 min on 8 cores: CV, thresholds, conformal intervals, final models,
                             # event forecaster, test scoring -> models/, outputs/
python src/report.py         # outputs/eval_report.md
python -m uvicorn src.server:app --port 8000
# open http://localhost:8000   (http://localhost:8000/?mock=1 runs the dashboard on fixtures only)
python -m pytest tests/      # offline: model + backend/API tests, no network
```

Live sources are cached in `live_cache/` (do not delete it: CelesTrak throttles clients that re-download).
`CARA_OFFLINE=1` serves only that cache (`CARA_OFFLINE=celestrak.org` blocks just one host), marked stale
once a refresh is due.

Test-set predictions (one row per CDM, tier + probabilities + reasons): `outputs/test_predictions.csv`.

## Results (held-out `test_data.csv`)

Test set: 24,484 CDMs / 2,167 events never used for training or tuning (true tiers Safe / Elevated /
Critical = 15,756 / 6,501 / 2,227). Decision thresholds were tuned on grouped cross-validation only.

| Method | Macro-F1 | F2 (Critical) | Missed Critical | Critical false alarms |
|---|---|---|---|---|
| Miss-distance rule only (status quo triage) | 0.952 | 0.967 | **91** | 0 |
| Max-Pc bound treated as Pc (worst case) | 0.842 | 0.985 | 0 | **173** |
| Isotropic Gaussian Pc (physics, radial sigma only) | 0.952 | 0.967 | 91 | 1 |
| **Conjunction Watch (LightGBM, physics features)** | **0.963** | **0.990** | **8** | **76** |
| ablation: + event-history features | 0.966 | 0.989 | 21 | 38 |
| live mode: screening inputs only (no covariance) | 0.940 | 0.984 | 26 | 70 |

* **Missed Critical CDMs drop from 91 to 8** versus distance-only triage, and the model finds 1,005 of the
  1,116 CDMs (90.1 %) whose risk is hidden in Pc rather than in miss distance.
* **None of the 8 is missed silently**: all 8 are in the *needs review* queue (4.1 % of CDMs). The error rate
  inside that queue is 26.6 % vs 2.7 % for auto-decided CDMs, so the flag puts the analyst where the mistakes are.
* The 80 % Pc interval covers **81.8 %** of test CDMs after split-conformal calibration on CV (70 % before).
* **Unseen satellites:** leave-mission-out CV (every satellite in a fold unseen in training) gives
  macro-F1 0.959, 16 missed Critical and 320 false alarms on 111,939 train CDMs, against 0.965 / 7 / 274 for
  event-grouped CV and 0.957 / **304** / 0 for the distance rule. The weakest fold (missions 1, 18, 20, 23) drops to
  macro-F1 0.884; generalisation to a new mission is the main residual risk.
* Probabilities are isotonic-calibrated on out-of-fold predictions: on test, the expected calibration error of
  P(Pc >= 1e-4) falls from 0.0019 to 0.0005 (Brier 0.00377 -> 0.00364).
* Event-level forecast of the tier at TCA (from the last CDM >= 2 days out, CV on 8,293 train events):
  macro-F1 0.868 vs 0.859 for persisting the model's current tier and 0.868 for the *Latest Risk
  Prediction* baseline that uses the excluded `risk` column — consistent with Uriot et al.: the naive forecast
  is very hard to beat, and we do not claim otherwise.

Full report with confusion matrices, calibration, failure cases and feature importance:
[`outputs/eval_report.md`](outputs/eval_report.md).

## Honest evaluation — where it fails

* **All 8 missed Critical CDMs are borderline**: true Pc between 1.0e-4 and 1.2e-4, i.e. within 0.1 decades
  of the manoeuvre threshold (events 107, 1930, 1733). Distance said 1.2–5.7 km. They are flagged for review
  because the 80 % interval straddles 1e-4 or the worst-case Pc bound is >= 1e-4.
* **The two model heads can disagree** (event 107: median Pc estimate 1.5e-4 but classifier P(Critical) 2 %).
  209 test CDMs show such a contradiction; they are not hidden: the reasons say "Model components disagree"
  and the CDM goes to the review queue. A single calibrated predictive distribution would remove the
  contradiction at its source.
* **False alarms (76 Critical calls on non-Critical CDMs)** come mostly from events like 712 and 220, where
  the true Pc depends on along-track / normal covariance terms outside the 13 allowed columns (only the radial
  sigma is available). With the full CDM covariance a direct Pc computation would remove most of them.
* **Calibration is good for Critical, imperfect for Elevated under shift**: after isotonic calibration, test
  P(Pc >= 1e-4) bins match observed rates within a few points (e.g. 0.26 -> 0.29, 0.61 -> 0.60), apart from
  the top bin (0.94 predicted, 0.87 observed). P(Pc >= 1e-6) is under-confident in the 0.4–0.8 range
  (0.60 -> 0.71, 0.77 -> 0.86) because the test set has more Elevated events than train (mission mix, paper
  Sec. 4.5). Probabilities are shown as evidence for the analyst, not as a final answer.
* **Live mode is screening-grade**: SOCRATES uses public TLEs and no covariance; the screening model loses
  about 2 points of macro-F1 and misses 26 Critical CDMs on test. Live tiers say so on screen.

## Design choices grounded in the literature

Uriot et al., *Spacecraft collision avoidance challenge: design and results of a machine learning
competition*, Astrodynamics 6(2), 2022 ([docs/reference](docs/reference/)) — the paper describing the ESA
Kelvins competition this dataset comes from:

* **Strong naive baselines first.** The paper shows the *latest risk* forecast beats most ML entries, so we
  report the status-quo distance rule, the max-Pc bound and a physics Pc as baselines, and the
  latest-risk baseline for the event forecast.
* **Recall-weighted metric.** False negatives are far more costly (paper Sec. 4.3, F2 score); decision
  thresholds are tuned for F2 on cross-validation, never on the test set.
* **Grouped validation matching the test set.** Events are never split across folds and only CDMs ≥ 2 days
  before TCA are scored, mirroring how `test_data` was built (paper Sec. 4.2).
* **Uncertainty, not point estimates.** The paper's consistently-misclassified events had high covariance
  and big Pc jumps (Sec. 5.4, Fig. 13) and it recommends outputting a risk *distribution*; we output a
  quantile interval and route straddling cases to an analyst.
* **Model selection fixed before looking at test.** The per-CDM model (23 features) was chosen over the
  event-history variant (62 features) after the first grouped-CV run, where they tied (macro-F1 0.9751 vs
  0.9748). In the final CV the history variant is marginally ahead (F2 0.994 vs 0.993, within fold noise), so
  the simpler, more explainable model stays. On test the history variant would have missed 21 Critical CDMs
  instead of 8, with 38 false alarms instead of 76. Both are reported, and neither choice was revisited after
  seeing test.
* **Mission / distribution shift.** Train and test differ by mission mix (Sec. 4.5), so CV-to-test gaps are
  reported, not hidden.

## Trust and traceability

* The model **never sees** the `risk` column; it is used only as ground truth.
* Every tier shows *which rule fired* and *the probability vs the threshold*; the miss-distance rule is a hard
  rule the model cannot override.
* Live data shows its source, `Last-Modified` time and a **STALE** badge when a feed is down (last cached copy).
* Live SOCRATES conjunctions are TLE/SGP4 screening products **without covariance**: labelled
  *screening-grade — confirm with CSpOC CDMs before any manoeuvre*.
* The analyst records *Acknowledge / Escalate / Dismiss* with a note; the decision log exports to CSV.

## Repository layout

```
src/features.py    allowed columns, tier rule, physics + history features
src/train.py       CV, threshold tuning, final fit, test scoring, metrics.json
src/predict.py     RiskModel inference + tier combination + explanations
src/live.py        live feeds (SOCRATES, SATCAT, GP/SGP4, NOAA SWPC, optional Space-Track)
src/server.py      FastAPI backend (docs/api_contract.md)
dashboard/         operator console (Cesium + Plotly)
outputs/           metrics.json, eval_report.md, test_predictions.csv, failure_cases.csv
docs/              challenge brief, API contract, reference paper
```
