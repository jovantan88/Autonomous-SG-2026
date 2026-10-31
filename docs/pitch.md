# 3-minute pitch — Conjunction Watch

Timings are cumulative. Every number below is in `outputs/eval_report.md` and reproduces with
`python src/train.py`.

## 0:00 – 0:30 · The user and the pain (problem realism)

* Our user is the **conjunction-assessment analyst on shift** at a satellite operator, the role ESA's Space
  Debris Office plays for its own fleet.
* They receive CDMs from CSpOC about three times a day for every close approach in a 7-day window. Each one is
  read by hand: Pc, miss distance, uncertainty. Then they decide whether to wake flight dynamics.
* **Status quo shortcut:** triage on miss distance. On the 24k-CDM test set that misses **91 Critical CDMs**,
  because the risk sits in Pc, not distance.

## 0:30 – 1:30 · Live demo (technical soundness)

1. **Live map:** today's SOCRATES screening (CelesTrak) for the next 24 h. Point at the header KPIs: how many
   conjunctions are within 1 km and 5 km in the window, which is the brief's 24-hour warning.
2. Click the top card in the **Critical events** sidebar. The globe flies to the **collision point** and draws
   both objects' **flight paths** from SGP4 around TCA.
3. Detail drawer: tier, P(Critical), Pc with an 80 % interval, and the **exact rules that fired**, read out loud.
   Press **Escalate**, which shows that the analyst decides and the decision log records it.
4. **CDM evaluator:** one test event's timeline (estimated Pc and band vs the real Pc as TCA approaches).

## 1:30 – 2:20 · Evidence (honest evaluation)

* Held-out test with thresholds tuned only on grouped CV, following the method of the ESA competition paper
  (Uriot et al. 2022).
* **Missed Critical: 91 → 8** vs distance triage; false alarms 76, against 173 for the worst-case Pc bound.
* **0 silent misses:** all 8 misses are in the review queue (4.1 % of CDMs, about 10x the error rate of
  auto-decided CDMs), so the analyst's attention goes where the model is unsure.
* **Unseen satellites** (leave-mission-out): 16 missed vs 304 for distance triage.
* **Where it fails:** all 8 misses are borderline (true Pc 1.0–1.2e-4). False alarms come from covariance
  terms outside the allowed 13 columns. Forecasting the tier at TCA does **not** beat the naive latest-risk
  baseline, the same finding as the paper. We show this rather than hide it.

## 2:20 – 2:50 · Trust and impact

* The model never sees `risk`. The tier is a published rule (NASA CA Handbook / ESA). The ML only estimates Pc,
  with an interval that covers 81.8 % of cases for a nominal 80 %.
* Live tiers are labelled **screening-grade** (TLE, no covariance): confirm with CSpOC CDMs before any manoeuvre.
* Impact: the analyst reads about 4 % of CDMs closely instead of all of them, and critical risk hidden behind
  "safe-looking" distances drops by roughly 10x (91 → 8, all of those 8 still reviewed).

## 2:50 – 3:00 · Close

"Conjunction Watch reads the CDMs, explains every flag, and leaves the call to the analyst."
