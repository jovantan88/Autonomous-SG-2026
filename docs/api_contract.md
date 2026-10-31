# CARA dashboard — API contract

Backend: `src/server.py` (FastAPI, `uvicorn src.server:app --port 8000`). Serves the dashboard
from `dashboard/` at `/` and JSON under `/api`. All times are UTC ISO-8601. Distances in metres,
speeds in m/s, `log_pc` = log10(Pc). Tier codes: `0 = Safe`, `1 = Elevated`, `2 = Critical`.

## Tier rule (deterministic, from the challenge brief)

```
Critical  : Pc >= 1e-4  OR miss_distance < 1 km
Elevated  : 1e-6 <= Pc < 1e-4  OR 1 km <= miss_distance <= 5 km
Safe      : Pc < 1e-6 AND miss_distance > 5 km
```

The model only estimates Pc (it never sees `risk`); the miss-distance half of the rule is applied
as a hard rule and is always reported in `reasons`.

## Common "assessment" object

Returned for every CDM / conjunction that is scored:

```json
{
  "log_pc": -5.2,              // model median estimate of log10 Pc
  "log_pc_lo": -7.1,           // 10th percentile
  "log_pc_hi": -4.3,           // 90th percentile
  "p_pc_elevated": 0.62,       // P(log Pc >= -6)
  "p_pc_critical": 0.08,       // P(log Pc >= -4)
  "p_tier": [0.30, 0.62, 0.08],// P(Safe), P(Elevated), P(Critical) after combining with miss-distance rule
  "md_tier": 1,                // tier from miss distance alone
  "tier": 1,                   // operational decision (safety-biased thresholds, see metrics)
  "needs_review": true,        // uncertainty straddles a tier boundary -> human must look
  "reasons": ["Miss distance 3.2 km is within 1-5 km -> Elevated", "..."]
}
```

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{status, model_loaded, live_sources: {socrates, swpc, satcat, spacetrack, gp}, live_sources_detail, model, test_set, live_assessment}`; each `live_sources` value is a status string: `ok`, `stale` (last refresh failed, or the copy is overdue by > 15 min past its cache time), `error`, `not fetched` or `disabled` |
| GET | `/api/metrics` | contents of `outputs/metrics.json` (baselines, model results, confusion matrices, failure cases); 503 `{detail}` while the file does not exist yet |
| GET | `/api/test/events` | `{events: [...]}` latest-CDM summary per test event: `event_id, n_cdms, time_to_tca, miss_distance, c_object_type, assessment, true_tier, true_log_pc, forecast` |
| GET | `/api/test/events/{event_id}` | `{event_id, cdms: [{time_to_tca, miss_distance, relative_speed, relative_position_r/t/n, relative_velocity_r/t/n (display only), c_sigma_r, t_sigma_r, max_risk_estimate, true_log_pc, assessment}]}` |
| GET | `/api/live/conjunctions?window_h=24&limit=300&sort=risk` (+ optional `points=1`, `tier=1` or `tier=1,2`, `needs_review=0/1`) | `{source, fetched_at, data_as_of, stale, n_total, window_h, n_in_window, filter, n_matching, n_returned, warnings: {...}, conjunctions: [...]}`; `window_h` <= 168, `limit` <= 2000, `sort` = `risk`, `tca` or `range` (else 422) |
| GET | `/api/live/spaceweather` | `{fetched_at, f107: {latest, series[]}, ap: {latest, series[]}, kp: {latest, series[]}}` |
| GET | `/api/live/track?a=<norad>&b=<norad>&tca=<iso>&span_min=50&step_s=20` | `{a: {norad, name, tle_epoch, ecef: [[t_iso,x,y,z],...], lla: [[t_iso,lat,lon,alt_km],...]}, b: {...}, relative_rtn: [[dt_s, r, t, n], ...], min_range_m, tca_model, min_at_edge, relative_speed_ms, rtn_at_tca_m, tca_point, propagator}`; 422 for a bad `tca`, 404/503 `{detail}` when an element set is unavailable |
| POST | `/api/admin/reload-model` | header `X-CARA-Admin: 1` (or the value of `CARA_ADMIN_TOKEN`) required, else 403. Reloads models/ only when every file is complete, settled for 2 s and passes a one-row prediction; otherwise keeps the previous model and reports `model.error` |
| POST | `/api/assess` | body = one CDM with the allowed columns (+ optional history list) → `assessment` + `forecast, input_warnings, ignored_fields, model`; 422 `{detail: [...]}` on invalid input |

### Collision points (live)

`GET /api/live/conjunctions?...&points=1` adds to every returned row:

```json
"tca_point": {"lat": 51.2, "lon": -12.7, "alt_km": 548.3, "range_km": 0.41, "tca_offset_s": -0.8},
        // midpoint of the two objects at *our* SGP4 closest approach within +/-120 s of the SOCRATES TCA,
        // or null (element sets pending/missing, or our closest approach is > 100 km: no meaningful point)
        // range_km = our SGP4 min range with *our* element sets; tca_offset_s = its time minus the SOCRATES TCA
"tle_status": "ok" | "inconsistent" | "pending" | "missing",
        // ok = our SGP4 reproduces the screening; inconsistent = it does not (see sgp4_check)
"sgp4_check": {"min_range_m": 410.0, "miss_distance_m": 380.0, "agrees": true, "distance_tier": 2,
               "tca_offset_s": -0.8, "min_at_edge": false, "epoch_shift_h": [21.9, 28.0], "note": "..."}   // or null
        // agrees = same distance-rule tier AND within 1 km of SOCRATES' miss distance;
        // epoch_shift_h = our element-set epoch minus the one SOCRATES used (TCA - DSE), hours, A then B
```

SOCRATES screens with older element sets than ours most of the time, so many rows come back
`inconsistent`: the UI must show those points hollow / flagged and never as confirmed collision points.

TLEs come from one bulk `gp.php?GROUP=active&FORMAT=csv` (OMM) download (covers nearly every primary; cached
6 h; CSV because the TLE format cannot carry the 6-digit catalog numbers most new payloads have; a
still-fresh legacy `FORMAT=tle` cache is used until it expires, it is never downloaded again) plus throttled per-object `gp.php?CATNR=` fetches for the remaining (mostly debris) secondaries
(<= 2 requests/s, cached 24 h on disk, at most 300 new objects per 2 h). Rows whose TLEs are still being
fetched come back `pending`; the UI re-polls every 60 s and points appear as they resolve.

`/api/live/track` default `span_min` is 50 (about +/- half an orbit, the "flight path" either side of TCA).

### Filters (live)

Sorted by risk, the top few hundred rows of a 24 h window are all Critical. `tier=` and `needs_review=`
select rows *before* sorting and `limit`, so a client can also load the top Elevated and needs-review
conjunctions (the dashboard merges `limit=500` + `tier=1&limit=150` + `needs_review=1&limit=150`).
`warnings` always describes the whole window, unfiltered; `n_matching` counts the rows that pass the filter.
The feed (and the top-level `n_in_window`) also keeps conjunctions whose TCA passed less than 1 h ago;
`warnings` counts only TCAs in `[now, now + window_h]` and reports the recently passed ones as `n_recent_past`.

### `warnings` (live, 24 h window)

```json
{
  "window_h": 24,
  "n_in_window": 812,           // TCA in [now, now + window_h]
  "n_recent_past": 31,          // TCA passed < 1 h ago (in the feed, in no other count)
  "n_within_1km": 41,
  "n_within_5km": 812,
  "n_critical": 57,
  "n_elevated": 700,
  "by_hour": [{"hour": 0, "critical": 3, "elevated": 30}, ...],
  "top_primaries": [{"name": "STARLINK-1234", "count": 9}, ...]
}
```

### live conjunction row

```json
{
  "id": "59770-100042-2026-11-04T13:32:10Z",
  "sat1": {"norad": 59770, "name": "STARLINK-31829", "status": "+", "type": "PAYLOAD", "tle_age_d": 6.1},
  "sat2": {"norad": 100042, "name": "STARLINK-38002", "status": "+", "type": "PAYLOAD", "tle_age_d": 6.2},
  "tca": "2026-11-04T13:32:10.051Z",
  "hours_to_tca": 94.3,
  "miss_distance": 11.0,
  "relative_speed": 11976.0,
  "max_prob": 1.0,
  "dilution_km": 0.0,
  "assessment": { ... },       // reduced-feature model (no covariances in SOCRATES), plus "heads_disagree":
                               // true when the classifier alone raised the tier while the Pc quantiles stay below
                               // that tier's threshold (then needs_review is true and a reason says so)
  "provenance": "CelesTrak SOCRATES Plus sort-minRange.csv, Last-Modified 2026-10-31T02:42:29Z"
}
```

## Live data sources (all public; cached server-side to respect provider rate limits)

| Source | URL | Used for | Cache |
|---|---|---|---|
| CelesTrak SOCRATES Plus | `https://celestrak.org/SOCRATES/sort-minRange.csv`, `sort-maxProb.csv` | live conjunction screening (TLE/SGP4-based) | 2 h (file updates ~3x/day) |
| CelesTrak SATCAT | `https://celestrak.org/pub/satcat.csv` | object type (PAY/R/B/DEB/UNK) | 24 h |
| CelesTrak GP | `https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv` (bulk), `...?CATNR=<id>&FORMAT=TLE` (`FORMAT=CSV` for 6-digit ids) | element sets for SGP4 trajectories and collision points | 6 h bulk, 24 h per object |
| NOAA SWPC | `https://services.swpc.noaa.gov/products/10cm-flux-30-day.json`, `.../json/predicted_fredericksburg_a_index.json`, `.../products/noaa-planetary-k-index.json` | F10.7 and Ap model inputs | 1 h |
| Space-Track `cdm_public` (optional) | `https://www.space-track.org` | real CDMs if `SPACETRACK_USER`/`SPACETRACK_PASS` env vars are set by the user | 1 h |

No CORS by default (the dashboard is same-origin); `CARA_CORS_ORIGINS=<origin>,...` allows read-only GETs from
a dashboard hosted elsewhere. A cached SOCRATES / SATCAT payload that cannot be parsed is reported as `stale`
with `parse_error` in `/api/health` while the previous parse keeps being served; malformed lines are skipped and
counted in `files.<label>.skipped_rows`.

If a source is unreachable the server serves the last cached snapshot with `stale: true` and its
original timestamp; the UI must show this. With nothing cached the endpoint answers 503 `{detail}`.
`CARA_OFFLINE=1` (all hosts) or `CARA_OFFLINE=celestrak.org` (a comma list of hosts) makes the server
use only its disk cache for those hosts, without recording failures or back-off (demo without network,
or to keep a throttled provider untouched).

## `outputs/metrics.json` (written by `src/train.py`, served verbatim by `/api/metrics`)

```json
{
  "generated_at": "2026-10-31T12:00:00Z",
  "eval_set": {"name": "test_data.csv", "n_rows": 24484, "n_events": 2167, "true_tier_counts": [15756, 6501, 2227]},
  "chosen_method": "LightGBM quantile + classifiers (physics + history features)",
  "thresholds": {"full": {"p_critical": 0.2, "p_elevated": 0.35}, "screen": {...}},
  "methods": [
    {
      "name": "Miss-distance rule only", "kind": "baseline",
      "description": "Applies only the 1 km / 5 km rule; assumes Pc is safe",
      "cv":   {"macro_f1": 0.0, "f2_critical": 0.0, "recall": [s, e, c], "precision": [s, e, c], "confusion": [[...],[...],[...]]},
      "test": {"macro_f1": 0.0, "f2_critical": 0.0, "recall": [s, e, c], "precision": [s, e, c], "confusion": [[...],[...],[...]]}
    }
  ],
  "regression": {"test_mae_log_pc": 0.0, "test_mae_log_pc_high_risk": 0.0, "interval_coverage_80": 0.0},
  "calibration": [{"p_pred": 0.05, "p_obs": 0.04, "n": 1200}],
  "failure_cases": [{"event_id": 12, "time_to_tca": 2.3, "miss_distance": 7000, "true_tier": 2, "pred_tier": 0,
                      "log_pc": -7.1, "true_log_pc": -3.8, "p_critical": 0.03, "why": "..."}],
  "event_forecast": {"description": "...", "cv": {...}, "baseline_latest": {...}},
  "screening_model": {"description": "...", "test": {...}},
  "feature_importance": [{"feature": "max_risk_estimate", "gain_pct": 31.2}],
  "notes": ["..."]
}
```

`confusion[i][j]` = number of rows with true tier `i` predicted as tier `j`.
Frontends must render unknown/missing keys gracefully (the file is extended over time).
