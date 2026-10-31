# TIP Watcher: Tier and Impact Probability for satellite conjunctions

Autonomous SG 2026, Airbus Challenge #1: *"How could we leverage AI-driven technologies for satellite
collision avoidance?"*

**User.** The conjunction-assessment analyst on shift at a satellite operator. Every close approach produces a
stream of conjunction data messages (CDMs) over about a week. Today the analyst reads each one and, when a 
manoeuvre has to be planned (about 2 days out), goes by the latest message.

**What TIP Watcher does.** From the messages issued 2 or more days before closest approach it forecasts each
event's **final tier** (Safe / Elevated / Critical, NASA CA Handbook and ESA rules) **with a probability**,
explains each forecast in plain language, and shows the analyst what needs attention first. It never decides on
its own: the analyst acknowledges, escalates or dismisses, and every decision is logged.

| Tier | Rule (Pc = 10^risk, miss distance in metres) |
|---|---| 
| Critical (manoeuvre required) | Pc ≥ 1e-4 **or** miss distance < 1,000 m |
| Elevated | not Critical, and 1e-6 ≤ Pc < 1e-4 **or** miss distance 1,000–5,000 m |
| Safe | Pc < 1e-6 **and** miss distance > 5,000 m |

## How we meet the updated brief

| Brief item (update of 31 Oct, 12:45) | Where |
|---|---|
| Predict each event's final tier from CDMs with time_to_tca ≥ 2 days, with a probability | `src/event_model.py`, `outputs/test_event_predictions.csv` (all 2,167 test events) |
| Report metrics on a held-out split of train_data by event_id; recall on Critical and a confusion matrix | `outputs/event_metrics.json`, fixed 80/20 split in `outputs/event_split.csv` |
| Dashboard: risk scores and miss distances as closest approach gets nearer | Events view: reported Pc, physics Pc and forecast probability, miss distance and relative position r/t/n against time_to_tca |
| Warnings view using time_to_tca windows | Events view sidebar: events reaching closest approach within 3 / 4 / 5 / 7 days (`/api/test/warnings`) |
| Event labels file | used as the target; our tiers match all 13,154 labelled events exactly |

## Results (hold-out: 2,389 train events never used for fitting or tuning; 167 finally Critical)

Model choice and decision thresholds come from 5-fold cross-validation inside the 80 % training events only.
The candidate with the best CV F2 on Critical (13 suggested columns, hybrid rule) was then scored once on the
hold-out.

| Method | Critical recall | Missed Critical | Critical false alarms | F2 (Critical) | Macro-F1 |
|---|---|---|---|---|---|
| Latest message tier (status quo) | 0.892 | 18 | 25 | 0.885 | **0.904** |
| Latest tier + 0.5-decade safety margin | 0.898 | 17 | 30 | 0.884 | 0.882 |
| **TIP Watcher (latest tier + model escalation)** | **0.940** | **10** | 60 | **0.887** | 0.858 |
| Model alone, F2-optimal threshold | 0.952 | 8 | 83 | 0.874 | 0.828 |

On the 1,781 hold-out events whose final message really is in the future (in the rest the last message is
already ≥ 2 days out), Critical recall is **0.914 vs 0.845** for the status quo (10 vs 18 missed).

Confusion matrix of the served rule on the hold-out (rows = true final tier):

| | Safe | Elevated | Critical |
|---|---|---|---|
| **Safe** | 1,695 | 91 | 4 |
| **Elevated** | 27 | 349 | 56 |
| **Critical** | 1 | 9 | 157 |

## Honest evaluation: where it falls short

* **It trades false alarms for catches.** 8 more Critical events caught costs 35 more false alarms (60 vs 25)
  and lowers macro-F1 (0.858 vs 0.904). For collision avoidance a missed Critical is far worse than a false
  alarm, which is why we optimise F2, but the analyst should know the trade.
* **At equal false alarms the model does not beat the status quo.** Constraining the model to 25–33 false alarms
  gives Critical recall 0.838, below 0.892 for "keep the latest tier". Its value is as an escalation layer on
  top of the latest message, not as a replacement. This matches the ESA competition paper (Uriot et al. 2022): the
  naive latest-risk forecast is very hard to beat.
* **The final tier is mostly decided by miss distance.** Miss-distance features carry about 88 % of the
  Critical classifier's gain; Pc-driven Critical events are rare in this data.
* **No final outcomes for test_data.** Test predictions are demo predictions only; all metrics come from the
  train hold-out.
* **Data shift.** Train and test differ by mission mix (paper, Sec. 4.5). Leave-mission-out CV for the
  supporting per-message model is in `outputs/eval_report.md`.

## Inputs and why

The chosen model uses the 13 suggested columns. The hybrid rule also uses the latest message's tier, which needs
its `risk`: the brief notes the tier can be read straight from `risk`, and the latest-message tier is the
operator's status quo, so we start from it and only let the model escalate. Other columns (full covariance,
object sizes, orbit sizes) are used only by the physics module, explained below, and in the ablation
`all` feature set (`COLUMN_NOTES` in `src/event_model.py`).

## Supporting components

* **Per-message Pc estimator** (`src/train.py`, `src/predict.py`): estimates each message's Pc from the 13
  suggested columns with an 80 % interval (81.8 % test coverage after conformal calibration), flags
  uncertain messages for review, and powers the live view. Report: `outputs/eval_report.md`.
* **Physics Pc and avoidance burns** (`src/physics.py`): the 2D Pc from the full covariance of both objects
  reproduces ESA's `risk` column to a median 0.0014 decades (mean 0.004) on test; an independent Monte Carlo check
  agrees to better than 0.001 decades. `manoeuvre_options` sizes an along-track burn (with margin for ±10 %
  burn error and orbit eccentricity) that brings Pc below a target; checked against numerical propagation.
* **Live sky** (`src/live.py`): today's conjunctions from CelesTrak SOCRATES Plus, object types from SATCAT,
  orbits from CelesTrak GP (SGP4 flight paths and collision points on a Cesium globe), space weather from NOAA
  SWPC. Public data, cited, fetched by script and cached; labelled as a screening estimate without covariance.

## Run it

```bash
pip install -r requirements.txt
# put train_data.csv and test_data.csv in data/ (challenge Drive, Challenge 1 folder)
python src/event_model.py        # official task: CV, hold-out, test predictions (~5 min)
python src/train.py              # supporting per-message model (~25 min)
python -m uvicorn src.server:app --port 8000
# open http://localhost:8000   (?mock=1 runs the UI on bundled sample data)
python -m pytest tests/          # 90+ tests
```

On Vercel, live feeds and derived assessments are cached in temporary storage because the application
directory is read-only. This cache is local to each instance and may be cleared on a cold start. Local
runs use `live_cache/`; set `CARA_CACHE_DIR` to override either default with a writable directory.

## Trust and traceability

* Every forecast lists its reasons: the latest message's tier, how Pc and miss distance moved, the probability
  against the threshold, and why the forecast differs from the latest message.
* Thresholds are tuned on cross-validation only; the hold-out is scored once.
* Live data shows its source and age, and turns "Stale" when a feed is down.
* The analyst makes the call: Acknowledge / Escalate / Dismiss with a note, exported as CSV.

## Literature

Uriot et al., *Spacecraft collision avoidance challenge: design and results of a machine learning competition*,
Astrodynamics 6(2), 2022 ([docs/reference](docs/reference/)): source of the dataset, the latest-risk baseline,
the recall-weighted F2 metric and the observation that hard cases are high-uncertainty events.

## Repository layout

```
src/event_model.py   official task: final-tier forecast, CV + hold-out, test predictions
src/features.py      tier rule, per-message features
src/train.py         supporting per-message Pc estimator, calibration, robustness
src/predict.py       per-message inference and plain-language reasons
src/physics.py       full-covariance 2D Pc, avoidance-burn options
src/live.py          CelesTrak / NOAA live feeds, SGP4
src/server.py        FastAPI backend (docs/api_contract.md)
dashboard/           TIP Watcher operator console
outputs/             event_metrics.json, test_event_predictions.csv, eval_report.md, physics_validation.json
docs/updates/        updated problem statements and challenge1_event_labels.xlsx
```
