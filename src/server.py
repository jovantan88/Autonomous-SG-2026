"""FastAPI backend for the CARA dashboard (see docs/api_contract.md).

    python -m uvicorn src.server:app --port 8000        (from the repo root)

Serves the dashboard from dashboard/ at "/" and JSON under /api. All times UTC ISO-8601, distances
in metres, speeds in m/s, log_pc = log10(Pc), tiers 0 Safe / 1 Elevated / 2 Critical.

Environment overrides: CARA_MODEL_DIR, CARA_TEST_DATA, CARA_METRICS, CARA_DASHBOARD, CARA_CACHE_DIR,
CARA_BACKGROUND=0 (no background refresh threads; used by the offline tests), CARA_CORS_ORIGINS
(read-only CORS for a dashboard hosted elsewhere; default none), CARA_ADMIN_TOKEN (required value of
the X-CARA-Admin header on POST /api/admin/*; without it any non-empty value is accepted):

    curl -X POST -H "X-CARA-Admin: 1" http://localhost:8000/api/admin/reload-model
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import shutil
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import features as F  # noqa: E402
import live  # noqa: E402
from predict import DISAGREE_PREFIX, SCREENING_COLUMNS, RiskModel, _fmt_pc  # noqa: E402

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, field_validator  # noqa: E402

log = logging.getLogger("cara.server")


class Settings:
    def __init__(self):
        self.model_dir = Path(os.environ.get("CARA_MODEL_DIR", ROOT / "models"))
        self.test_data = Path(os.environ.get("CARA_TEST_DATA", ROOT / "data" / "test_data.csv"))
        self.metrics = Path(os.environ.get("CARA_METRICS", ROOT / "outputs" / "metrics.json"))
        self.dashboard = Path(os.environ.get("CARA_DASHBOARD", ROOT / "dashboard"))
        self.background = os.environ.get("CARA_BACKGROUND", "1") != "0"
        self.live_poll_s = 60            # background: check source TTLs this often (fetch() enforces TTLs)
        self.live_reassess_s = 1800      # re-run the screening model at least every 30 min (time_to_tca drifts)
        self.max_limit = 2000
        self.max_window_h = 168.0        # SOCRATES covers 7 days
        self.prefetch_rows = 300         # background: queue TLEs of the top-N risk rows for collision points


SETTINGS = Settings()
TIER_NAMES = F.TIERS


# ---------------------------------------------------------------------------------------------
# JSON: never emit NaN / inf (-> null), accept numpy / pandas scalars
# ---------------------------------------------------------------------------------------------

def clean(o: Any) -> Any:
    if isinstance(o, bool) or o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (datetime, pd.Timestamp)):
        return live.iso(o)
    if isinstance(o, Path):
        return str(o)
    return o


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(clean(content), ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")


def fnum(x, nd: Optional[int] = None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


# ---------------------------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------------------------

@dataclass
class TestSet:
    df: pd.DataFrame                 # every test CDM: inputs + assessment + display/label columns
    bounds: dict                     # event_id -> (start, stop) row positions
    events: list                     # pre-built /api/test/events rows (sorted)
    key: dict
    computed_at: str
    seconds: float
    from_cache: bool


@dataclass
class LiveAssessed:
    df: pd.DataFrame                 # screening rows + assessment columns, sorted by TCA
    tca_ns: np.ndarray
    snap: live.SocratesSnapshot
    key: tuple
    computed_at: datetime
    seconds: float
    f10: Optional[float]
    ap: Optional[float]
    sw_status: dict
    horizon_end_ns: int = 0          # rows with TCA up to here are assessed (int64 max = all)
    horizon_h: Optional[float] = None


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = RiskModel()
        self.model_version: Optional[tuple] = None
        self.model_loaded_at: Optional[str] = None
        self.model_error: Optional[str] = None
        self.test: Optional[TestSet] = None
        self.test_error: Optional[str] = None
        self.test_lock = threading.Lock()
        self.live: Optional[LiveAssessed] = None
        self.live_error: Optional[str] = None
        self.live_lock = threading.Lock()
        self.point_memo: dict = {}
        self.stop = threading.Event()
        self.threads: list = []


STATE = State()
HEADS = ["q10", "q50", "q90", "clf_elev", "clf_crit"]


def model_files_version(model_dir: Path) -> tuple:
    if not model_dir.exists():
        return ()
    out = []
    for p in sorted(model_dir.iterdir()):
        if p.is_file() and p.suffix in (".txt", ".json"):
            st = p.stat()
            out.append((p.name, st.st_mtime_ns, st.st_size))
    return tuple(out)


def model_info() -> dict:
    m = STATE.model
    meta = m.meta or {}
    return {"trained": m.trained, "n_models": len(m.models), "models": sorted(m.models),
            "screening_model": "screen_q50" in m.models, "event_forecast": "event_forecast" in m.models,
            "thresholds": m.thresholds, "trained_at": meta.get("trained_at"),
            "train_rows": meta.get("train_rows"), "loaded_at": STATE.model_loaded_at,
            "fallback": None if m.trained else "physics fallback (no trained models in models/)",
            "error": STATE.model_error, "model_dir": str(SETTINGS.model_dir)}


MODEL_SETTLE_S = 2.0      # model files modified more recently than this may still be being written


def _model_file_problem(p: Path) -> Optional[str]:
    """LightGBM crashes the process (segfault, no exception) on a truncated model file, so every
    *.txt must end the way Booster.save_model ends it before it is handed to LightGBM."""
    try:
        with open(p, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(fh.tell() - 65536, 0))
            tail = fh.read().decode("utf-8", "replace")
    except OSError as exc:
        return f"{p.name}: {exc}"
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if "end of parameters" not in tail or not lines or \
            not (lines[-1].startswith("pandas_categorical:") or lines[-1] == "end of parameters"):
        return f"{p.name} is incomplete"
    return None


class ModelsNotSettled(ValueError):
    pass


def _snapshot_models(src: Path, dst: Path) -> tuple:
    """Copy models/ to `dst` and check the copies, so files that change while training writes them can
    neither crash LightGBM nor mix two training runs. Returns the files' version (as
    model_files_version); raises ModelsNotSettled when the set is still being written."""
    files = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix in (".txt", ".json")] if src.exists() else []
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in files}
    young = [n for n, (mt, _) in before.items() if time.time() - mt / 1e9 < MODEL_SETTLE_S]
    if young:
        raise ModelsNotSettled(f"model files still changing ({', '.join(young)}); training may still be writing")
    for p in files:
        shutil.copy2(p, dst / p.name)
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists()}
    if after != before:
        raise ModelsNotSettled("model files changed while being read; training may still be writing")
    bad = [msg for p in files if p.suffix == ".txt" for msg in [_model_file_problem(dst / p.name)] if msg]
    if bad:
        raise ModelsNotSettled("; ".join(bad) + " (training may still be writing)")
    return tuple((n, mt, sz) for n, (mt, sz) in before.items())


SMOKE_CDM = {"event_id": -1, "time_to_tca": 2.0, "miss_distance": 3000.0, "relative_speed": 10000.0,
             "relative_position_r": 100.0, "relative_position_t": 2900.0, "relative_position_n": 700.0,
             "c_object_type": "DEBRIS", "c_sigma_r": 50.0, "t_sigma_r": 10.0, "max_risk_estimate": -4.0,
             "F10": 150.0, "AP": 10.0}


def _smoke_test(m: RiskModel) -> None:
    """Predict once with both model sets, so a meta.json that does not match the boosters (e.g. old
    boosters with a new meta.json mid-training) is rejected before it replaces a working model."""
    df = pd.DataFrame([{c: dict(SMOKE_CDM, time_to_tca=t).get(c) for c in F.ALLOWED_COLUMNS} for t in (3.0, 2.0)])
    out = m.assess_cdms(df)
    scr = m.assess_screening(pd.DataFrame([{c: SMOKE_CDM[c] for c in SCREENING_COLUMNS}]))
    for name, o in (("assess_cdms", out), ("assess_screening", scr)):
        if not np.isfinite(o["log_pc"].to_numpy(dtype=float)).all():
            raise ValueError(f"smoke test: {name} returned a non-finite log_pc")


def load_model() -> dict:
    """(Re)load models/ ; keeps the previous model if the new set is unreadable, incomplete, still being
    written or fails a one-row prediction."""
    version = model_files_version(SETTINGS.model_dir)
    try:
        for attempt in range(3):                    # a save that is just finishing settles within seconds
            try:
                with tempfile.TemporaryDirectory(prefix="cara-models-", ignore_cleanup_errors=True) as tmp:
                    version = _snapshot_models(SETTINGS.model_dir, Path(tmp))
                    m = RiskModel.load(Path(tmp))   # boosters are read into memory; the copy can go
                break
            except ModelsNotSettled:
                if attempt == 2:
                    raise
                time.sleep(MODEL_SETTLE_S)
        if m.trained:
            missing = [f"full_{h}" for h in HEADS if f"full_{h}" not in m.models]
            if any(k.startswith("screen_") for k in m.models):
                missing += [f"screen_{h}" for h in HEADS if f"screen_{h}" not in m.models]
            if missing:
                raise ValueError(f"incomplete model set in {SETTINGS.model_dir} (missing {missing}); "
                                 "training may still be writing files")
        _smoke_test(m)
        err = None
    except Exception as exc:  # noqa: BLE001
        log.exception("model load failed")
        err = f"{type(exc).__name__}: {exc}"[:400]
        m = None
    with STATE.lock:
        if m is not None:
            STATE.model = m
            STATE.model_version = version
            STATE.model_loaded_at = live.iso(live.utcnow())
        elif STATE.model_version is None:
            STATE.model_version = ("fallback",)
            STATE.model_loaded_at = live.iso(live.utcnow())
        STATE.model_error = err
        STATE.point_memo.clear()
    return model_info()


def ensure_model():
    if STATE.model_version is None:
        load_model()


# ---------------------------------------------------------------------------------------------
# Assessment serialisation
# ---------------------------------------------------------------------------------------------

def assessment_records(df: pd.DataFrame) -> list[dict]:
    if len(df) == 0:
        return []
    c = {k: df[k].to_numpy() for k in ["log_pc", "log_pc_lo", "log_pc_hi", "p_pc_elevated", "p_pc_critical",
                                       "p_safe", "p_elevated", "p_critical", "md_tier", "tier",
                                       "needs_review", "reasons"]}
    out = []
    for i in range(len(df)):
        out.append({
            "log_pc": fnum(c["log_pc"][i], 6), "log_pc_lo": fnum(c["log_pc_lo"][i], 6),
            "log_pc_hi": fnum(c["log_pc_hi"][i], 6),
            "p_pc_elevated": fnum(c["p_pc_elevated"][i], 6), "p_pc_critical": fnum(c["p_pc_critical"][i], 6),
            "p_tier": [fnum(c["p_safe"][i], 6), fnum(c["p_elevated"][i], 6), fnum(c["p_critical"][i], 6)],
            "md_tier": int(c["md_tier"][i]), "tier": int(c["tier"][i]),
            "needs_review": bool(c["needs_review"][i]), "reasons": list(c["reasons"][i]),
        })
    return out


FORECAST_DESC = "Event-level forecast: P(tier at the final CDM before TCA) from this CDM's point of view"


def forecast_records(df: pd.DataFrame) -> list:
    if "forecast_tier" not in df.columns:
        return [None] * len(df)
    ps, pe, pc = (df[k].to_numpy() for k in ["forecast_p_safe", "forecast_p_elevated", "forecast_p_critical"])
    t = df["forecast_tier"].to_numpy()
    return [{"description": FORECAST_DESC, "tier": int(t[i]),
             "p_tier": [fnum(ps[i], 6), fnum(pe[i], 6), fnum(pc[i], 6)]} for i in range(len(df))]


# ---------------------------------------------------------------------------------------------
# Test set (data/test_data.csv) assessments, cached on disk keyed by the model files
# ---------------------------------------------------------------------------------------------

DISPLAY_COLS = ["relative_velocity_r", "relative_velocity_t", "relative_velocity_n"]


def _file_key(p: Path) -> list:
    try:
        st = p.stat()
        return [str(p), st.st_mtime_ns, st.st_size]
    except OSError:
        return [str(p), None, None]


def compute_test_set(force: bool = False) -> TestSet:
    ensure_model()
    model, version = STATE.model, STATE.model_version
    path = SETTINGS.test_data
    if not path.exists():
        raise FileNotFoundError(f"test data not found: {path}")
    key = {"models": [list(v) for v in version], "trained": model.trained, "data": _file_key(path),
           "code": [_file_key(SRC / "predict.py"), _file_key(SRC / "features.py")], "v": 2}
    pkl = live.CACHE_DIR / "test_assessments.pkl"
    t0 = time.perf_counter()
    if not force and pkl.exists():
        try:
            with open(pkl, "rb") as fh:
                obj = pickle.load(fh)
            if obj.get("key") == key:
                df = obj["df"]
                return _build_test_set(df, key, obj.get("computed_at"), time.perf_counter() - t0, True)
        except Exception:  # noqa: BLE001 - stale / corrupt cache -> recompute
            log.warning("ignoring unreadable %s", pkl)

    df = F.load(path)                                   # ALLOWED_COLUMNS + risk (label), sorted
    a = model.assess_cdms(df[F.ALLOWED_COLUMNS])        # the model only ever sees allowed columns
    if not (np.array_equal(a["event_id"].to_numpy(), df["event_id"].to_numpy())
            and np.allclose(a["time_to_tca"].to_numpy(), df["time_to_tca"].to_numpy())):
        raise RuntimeError("assess_cdms changed the row order; cannot join labels")
    header = pd.read_csv(path, nrows=0).columns
    disp_cols = [c for c in DISPLAY_COLS if c in header]
    if disp_cols:                                       # display-only columns, never model inputs
        disp = pd.read_csv(path, usecols=["event_id", "time_to_tca"] + disp_cols)
        disp = disp.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")
        disp = disp.reset_index(drop=True)
        if np.array_equal(disp["event_id"].to_numpy(), a["event_id"].to_numpy()):
            for c in disp_cols:
                a[c] = disp[c].to_numpy()
    for c in DISPLAY_COLS:
        if c not in a.columns:
            a[c] = np.nan
    a["true_log_pc"] = df[F.LABEL].to_numpy()           # evaluation display only
    a["true_tier"] = F.risk_tier(df[F.LABEL].to_numpy(), df["miss_distance"].to_numpy())
    computed_at = live.iso(live.utcnow())
    try:
        pkl.parent.mkdir(parents=True, exist_ok=True)
        tmp = pkl.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"key": key, "df": a, "computed_at": computed_at}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, pkl)
    except OSError:
        log.warning("could not write %s", pkl)
    return _build_test_set(a, key, computed_at, time.perf_counter() - t0, False)


def _build_test_set(df: pd.DataFrame, key, computed_at, seconds, from_cache) -> TestSet:
    df = df.reset_index(drop=True)
    ev = df["event_id"].to_numpy()
    starts = np.r_[0, np.nonzero(ev[1:] != ev[:-1])[0] + 1]
    stops = np.r_[starts[1:], len(df)]
    bounds = {int(ev[s]): (int(s), int(e)) for s, e in zip(starts, stops)}
    last = df.iloc[stops - 1].copy()                    # smallest time_to_tca of each event
    last["n_cdms"] = stops - starts
    order = np.lexsort((-last["log_pc"].to_numpy(), -last["p_pc_critical"].to_numpy(),
                        -last["tier"].to_numpy(), -last["p_critical"].to_numpy()))
    last = last.iloc[order]
    ass, fc = assessment_records(last), forecast_records(last)
    events = []
    for i, (_, r) in enumerate(last.iterrows()):
        events.append({"event_id": int(r["event_id"]), "n_cdms": int(r["n_cdms"]),
                       "time_to_tca": fnum(r["time_to_tca"], 6), "miss_distance": fnum(r["miss_distance"], 1),
                       "relative_speed": fnum(r["relative_speed"], 1), "c_object_type": r["c_object_type"],
                       "max_risk_estimate": fnum(r["max_risk_estimate"], 6),
                       "assessment": ass[i], "true_tier": int(r["true_tier"]),
                       "true_log_pc": fnum(r["true_log_pc"], 6), "forecast": fc[i]})
    return TestSet(df, bounds, events, key, computed_at, round(seconds, 2), from_cache)


def ensure_test_set(wait_s: float = 0.0) -> TestSet:
    if STATE.test is not None:
        return STATE.test
    if not STATE.test_lock.acquire(timeout=max(wait_s, 0.01) if wait_s else -1):
        raise HTTPException(503, "test-set assessments are still being computed; retry shortly")
    try:
        if STATE.test is None:
            try:
                STATE.test = compute_test_set()
                STATE.test_error = None
            except Exception as exc:  # noqa: BLE001
                STATE.test_error = f"{type(exc).__name__}: {exc}"[:400]
                log.exception("test set failed")
                raise HTTPException(503, f"test-set assessments unavailable: {STATE.test_error}")
        return STATE.test
    finally:
        STATE.test_lock.release()


# ---------------------------------------------------------------------------------------------
# Live screening (SOCRATES) assessments
# ---------------------------------------------------------------------------------------------

ASSESS_OUT = ["log_pc", "log_pc_lo", "log_pc_hi", "p_pc_elevated", "p_pc_critical", "p_safe", "p_elevated",
              "p_critical", "md_tier", "tier", "needs_review", "reasons"]


FULL = None   # horizon_h=None -> every conjunction in the SOCRATES snapshot (7 days)
NS_PER_H = 3_600_000_000_000
FAR_FUTURE_NS = np.iinfo("int64").max


def _usable(cur: Optional[LiveAssessed], key, horizon_h: Optional[float], force: bool) -> bool:
    if cur is None or force or cur.key != key:
        return False
    if (live.utcnow() - cur.computed_at).total_seconds() >= SETTINGS.live_reassess_s:
        return False
    return _covers(cur, horizon_h)


def _covers(cur: LiveAssessed, horizon_h: Optional[float]) -> bool:
    if horizon_h is None:
        return cur.horizon_end_ns == FAR_FUTURE_NS
    return cur.horizon_end_ns >= pd.Timestamp(live.utcnow()).value + horizon_h * NS_PER_H


def flag_head_disagreement(a: pd.DataFrame, model: RiskModel) -> pd.DataFrame:
    """Live SOCRATES inputs are outside the training distribution (no covariance, ~80% payload
    secondaries, every row <= 5 km), and there the screening model's classifier and quantile heads can
    contradict each other. When the Pc part of the decision (not the distance rule) raised the tier on
    the classifier alone while the 90th-percentile Pc is below that tier's threshold, the row goes to
    an analyst instead of being auto-decided. predict.combine flags the same contradiction; this adds
    the live-data context to the reason."""
    a = a.copy()
    thr = model.thresholds.get("screen", model.thresholds) if model.trained else {}
    if not thr or not len(a):
        if "heads_disagree" not in a.columns:
            a["heads_disagree"] = np.zeros(len(a), bool)
        return a
    md, tier, hi = a["md_tier"].to_numpy(int), a["tier"].to_numpy(int), a["log_pc_hi"].to_numpy(float)
    pc, pe = a["p_pc_critical"].to_numpy(float), a["p_pc_elevated"].to_numpy(float)
    crit = (tier == 2) & (md < 2) & (pc >= thr.get("p_critical", 1.1)) & (hi < F.PC_CRITICAL)
    elev = (tier == 1) & (md < 1) & (pe >= thr.get("p_elevated", 1.1)) & (hi < F.PC_ELEVATED)
    bad = crit | elev
    reasons = a["reasons"].to_numpy(object).copy()
    for i in np.nonzero(bad)[0]:
        p, lim = (pc[i], "1e-4") if crit[i] else (pe[i], "1e-6")
        msg = (f"{DISAGREE_PREFIX}: the classifier gives a {p:.0%} chance of {lim}, but even the high end of "
               f"the estimate is {_fmt_pc(hi[i])}. Live screening inputs are outside the training data. "
               "Review needed.")
        r = [x for x in reasons[i] if not x.startswith(DISAGREE_PREFIX)]
        r.insert(len(r) - 1 if r and r[-1].startswith("Result:") else len(r), msg)
        reasons[i] = r
    a["reasons"] = reasons
    prior = a["heads_disagree"].to_numpy(bool) if "heads_disagree" in a.columns else np.zeros(len(a), bool)
    a["needs_review"] = a["needs_review"].to_numpy(bool) | bad
    a["heads_disagree"] = bad | prior
    return a


def compute_live(network: bool = True, force_reassess: bool = False,
                 horizon_h: Optional[float] = FULL) -> LiveAssessed:
    """Assess the SOCRATES conjunctions from now-1h up to `horizon_h` ahead (None = all) with the
    screening model (vectorised). Trained models make the full ~180k-row set take ~25 s, so the
    request path assesses a 24 h+ horizon (a few seconds) and the background loop the full set.
    Re-used until the SOCRATES/SATCAT payload, the model or F10/AP change, or it is 30 min old."""
    ensure_model()
    try:
        sw = live.space_weather(history=False)
    except Exception as exc:  # noqa: BLE001
        sw = {"errors": {"space_weather": str(exc)}}
    f10 = (sw.get("f107") or {}).get("latest")
    ap = (sw.get("ap") or {}).get("latest")
    snap = live.socrates_snapshot(network=network)
    key = (snap.version, STATE.model_version, f10, ap)
    cur = STATE.live
    if _usable(cur, key, horizon_h, force_reassess):
        cur.snap = snap
        return cur
    with STATE.live_lock:
        cur = STATE.live
        if _usable(cur, key, horizon_h, force_reassess):
            return cur
        t0 = time.perf_counter()
        now = live.utcnow()
        rows = live.screening_rows(snap.rows, now=now, f10=f10, ap=ap)
        if horizon_h is not None:
            rows = rows.loc[rows["hours_to_tca"].to_numpy() <= horizon_h].reset_index(drop=True)
        model = STATE.model
        if len(rows):
            a = flag_head_disagreement(model.assess_screening(rows[SCREENING_COLUMNS]), model)
            for c in ASSESS_OUT + ["heads_disagree"]:
                rows[c] = a[c].to_numpy()
        else:
            for c in ASSESS_OUT + ["heads_disagree"]:
                rows[c] = pd.Series(dtype=object if c == "reasons" else float)
        tca_ns = live.to_ns(rows["tca"]) if len(rows) else np.zeros(0, "int64")
        sw_status = {"F10": f10, "AP": ap, "f107_time": (sw.get("f107") or {}).get("time"),
                     "ap_time": (sw.get("ap") or {}).get("time"),
                     "source": (sw.get("f107") or {}).get("source"), "stale": sw.get("stale"),
                     "errors": sw.get("errors")}
        la = LiveAssessed(rows, tca_ns, snap, key, now, round(time.perf_counter() - t0, 2), f10, ap, sw_status)
        la.horizon_end_ns = (FAR_FUTURE_NS if horizon_h is None
                             else pd.Timestamp(now).value + int(horizon_h * NS_PER_H))
        la.horizon_h = horizon_h
        STATE.live = la
        STATE.live_error = None
        log.info("live assessment: %d rows (horizon %s h) in %.2fs", len(rows),
                 "all" if horizon_h is None else f"{horizon_h:g}", la.seconds)
        return la


FAST_HORIZON_H = 26.0   # covers the default 24 h window, with 2 h of slack as time moves on


def get_live(window_h: float) -> LiveAssessed:
    cur = STATE.live
    if SETTINGS.background and cur is not None and _covers(cur, window_h) and \
            (live.utcnow() - cur.computed_at).total_seconds() < 2 * SETTINGS.live_reassess_s:
        return cur                                       # the background loop keeps it fresh
    horizon = FULL if not SETTINGS.background else max(window_h + 2.0, FAST_HORIZON_H)
    try:
        return compute_live(network=True, horizon_h=horizon)
    except Exception as exc:  # noqa: BLE001 - source, parse or model failure: serve the last good assessment
        if not isinstance(exc, live.LiveSourceError):
            log.exception("live assessment failed")
        STATE.live_error = f"{type(exc).__name__}: {exc}"[:400]
        if cur is not None:
            return cur
        raise HTTPException(503, f"live screening unavailable: {STATE.live_error}")


def _hours(la: LiveAssessed, now: datetime) -> np.ndarray:
    return (la.tca_ns - pd.Timestamp(now).value) / 3.6e12


def _risk_order(df: pd.DataFrame) -> np.ndarray:
    return np.lexsort((df["miss_distance"].to_numpy(), -df["max_risk_estimate"].to_numpy(),
                       -df["p_pc_critical"].to_numpy(), -df["p_critical"].to_numpy(),
                       -df["tier"].to_numpy()))


POINT_MAX_RANGE_KM = 100.0   # beyond this the "midpoint" of the two objects is physically meaningless
POINT_AGREE_M = 1000.0       # our SGP4 min range must be within 1 km of SOCRATES' miss distance ...
                             # ... and give the same distance-rule tier, or the point is "inconsistent"


def sgp4_check(p: dict, miss_m: float, ta, tb, tca_ns: int, dse: tuple) -> dict:
    """Compare our own closest approach (current element sets) with SOCRATES' screening."""
    min_m = float(p["range_km"]) * 1000.0
    t_tier, s_tier = int(F.md_tier([min_m])[0]), int(F.md_tier([miss_m])[0])
    agrees = t_tier == s_tier and abs(min_m - miss_m) <= POINT_AGREE_M
    shifts = []                  # our epoch minus the epoch SOCRATES used (TCA - DSE), hours; > 0 = ours newer
    for t, d in ((ta, dse[0]), (tb, dse[1])):
        ok = d is not None and math.isfinite(d)
        shifts.append(round(((t.epoch_jd - 2440587.5) * 86400.0 - (tca_ns / 1e9 - d * 86400.0)) / 3600.0, 2)
                      if ok else None)
    if agrees:
        note = "SGP4 with our element sets reproduces the screening"
    else:
        note = (f"SGP4 with our element sets gives {min_m:,.0f} m (distance rule: {TIER_NAMES[t_tier]}) vs SOCRATES "
                f"{miss_m:,.0f} m ({TIER_NAMES[s_tier]}): the element sets differ (newer TLE or manoeuvre); "
                "re-screen before acting")
        if p["range_km"] > POINT_MAX_RANGE_KM:
            note += "; no collision point is plotted"
    return {"min_range_m": round(min_m, 1), "miss_distance_m": round(miss_m, 1), "agrees": bool(agrees),
            "distance_tier": t_tier, "tca_offset_s": p.get("tca_offset_s"), "min_at_edge": p.get("min_at_edge"),
            "epoch_shift_h": shifts, "note": note}


def collision_point(n1: int, n2: int, tca, tca_ns: int, miss_m: float = float("nan"),
                    dse: tuple = (None, None)) -> tuple[Optional[dict], str, Optional[dict]]:
    """(tca_point, tle_status, sgp4_check). The point is the midpoint of both objects at *our* SGP4
    closest approach near the SOCRATES TCA; tle_status is 'inconsistent' when that closest approach
    disagrees with SOCRATES' miss distance, and no point is returned beyond POINT_MAX_RANGE_KM."""
    store = live.STORE
    try:
        ta, tb = store.lookup(n1), store.lookup(n2)
        if ta is None or tb is None:
            missing = (ta is None and store.is_missing(n1)) or (tb is None and store.is_missing(n2))
            return None, "missing" if missing else "pending", None
        k = (n1, n2, tca_ns, id(ta), ta.epoch_jd, id(tb), tb.epoch_jd)
        memo = STATE.point_memo
        if k in memo:
            return memo[k]
        p = live.closest_point(ta, tb, tca)
        if p is None:
            res = (None, "missing", None)
        else:
            chk = sgp4_check(p, float(miss_m), ta, tb, tca_ns, dse) if math.isfinite(miss_m) else None
            point = {key: p[key] for key in ("lat", "lon", "alt_km", "range_km", "tca_offset_s")}
            if p["range_km"] > POINT_MAX_RANGE_KM:
                point = None
            res = (point, "ok" if chk is None or chk["agrees"] else "inconsistent", chk)
    except Exception:  # noqa: BLE001 - a malformed element set must not fail the whole feed
        log.warning("collision point %s x %s failed", n1, n2, exc_info=True)
        return None, "missing", None
    if len(memo) > 50000:
        memo.clear()
    memo[k] = res
    return res


def conjunction_rows(sub: pd.DataFrame, provenance: str, points: bool) -> list[dict]:
    if len(sub) == 0:
        return []
    tca = pd.to_datetime(sub["tca"], utc=True)
    tca_s = tca.dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    tca_ms = (tca.dt.strftime("%Y-%m-%dT%H:%M:%S.") + (tca.dt.microsecond // 1000).astype(str).str.zfill(3)
              + "Z").to_numpy()
    tca_ns = live.to_ns(tca)
    ass = assessment_records(sub)
    cols = {c: sub[c].to_numpy() for c in ["norad_1", "name_1", "status_1", "type_1", "dse_1", "norad_2",
                                           "name_2", "status_2", "type_2", "dse_2", "hours_to_tca",
                                           "miss_distance", "relative_speed", "max_prob", "dilution_km",
                                           "max_risk_estimate", "type_src_2"]}
    disagree = sub["heads_disagree"].to_numpy(bool) if "heads_disagree" in sub.columns else np.zeros(len(sub), bool)
    for i, a in enumerate(ass):
        a["heads_disagree"] = bool(disagree[i])
    if points:
        order = []
        for i in range(len(sub)):
            order += [int(cols["norad_1"][i]), int(cols["norad_2"][i])]
        live.STORE.request(order)
    out = []
    for i in range(len(sub)):
        n1, n2 = int(cols["norad_1"][i]), int(cols["norad_2"][i])
        row = {
            "id": f"{n1}-{n2}-{tca_s[i]}",
            "sat1": {"norad": n1, "name": cols["name_1"][i], "status": cols["status_1"][i],
                     "type": cols["type_1"][i], "tle_age_d": fnum(cols["dse_1"][i], 3)},
            "sat2": {"norad": n2, "name": cols["name_2"][i], "status": cols["status_2"][i],
                     "type": cols["type_2"][i], "tle_age_d": fnum(cols["dse_2"][i], 3),
                     "type_source": cols["type_src_2"][i]},
            "tca": tca_ms[i], "hours_to_tca": fnum(cols["hours_to_tca"][i], 3),
            "miss_distance": fnum(cols["miss_distance"][i], 1),
            "relative_speed": fnum(cols["relative_speed"][i], 1),
            "max_prob": fnum(cols["max_prob"][i]), "max_risk_estimate": fnum(cols["max_risk_estimate"][i], 4),
            "dilution_km": fnum(cols["dilution_km"][i], 3),
            "assessment": ass[i], "provenance": provenance,
        }
        if points:
            p, st, chk = collision_point(n1, n2, tca_ms[i], int(tca_ns[i]), float(cols["miss_distance"][i]),
                                         (fnum(cols["dse_1"][i]), fnum(cols["dse_2"][i])))
            row["tca_point"], row["tle_status"], row["sgp4_check"] = p, st, chk
        out.append(row)
    return out


def prefetch_tles(la: LiveAssessed):
    """Queue element sets for the rows the dashboard shows first (top risk in 24 h, soonest)."""
    now = live.utcnow()
    h = _hours(la, now)
    df = la.df.assign(hours_to_tca=h)
    w = df[(h >= -live.PAST_TCA_ALLOWANCE_H) & (h <= 24)]
    top = w.iloc[_risk_order(w)[:SETTINGS.prefetch_rows]] if len(w) else w
    ids = []
    for a, b in zip(top["norad_1"].to_numpy(), top["norad_2"].to_numpy()):
        ids += [int(a), int(b)]
    live.STORE.request(ids)


# ---------------------------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------------------------

def _background():
    try:
        ensure_test_set()
    except Exception:  # noqa: BLE001
        log.exception("test set warm-up failed")
    while not STATE.stop.is_set():
        try:
            if STATE.live is None:
                prefetch_tles(compute_live(network=True, horizon_h=FAST_HORIZON_H))
            la = compute_live(network=True, horizon_h=FULL)
            prefetch_tles(la)
            live.space_weather(history=True)           # keeps the 30-day Ap/Kp history cache warm
        except Exception as exc:  # noqa: BLE001
            STATE.live_error = f"{type(exc).__name__}: {exc}"[:400]
            log.warning("live refresh failed: %s", STATE.live_error)
        STATE.stop.wait(SETTINGS.live_poll_s)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_model()
    if SETTINGS.background:
        STATE.stop.clear()
        live.STORE.start()
        t = threading.Thread(target=_background, name="cara-background", daemon=True)
        t.start()
        STATE.threads.append(t)
    yield
    STATE.stop.set()
    live.STORE.stop()


app = FastAPI(title="CARA-Lite conjunction risk API", version="0.2", lifespan=lifespan,
              default_response_class=SafeJSONResponse)
# The dashboard is served from this same origin, so no CORS is needed. A dashboard hosted elsewhere
# (?api=http://localhost:8000) can be allowed read-only with CARA_CORS_ORIGINS=<origin>[,<origin>...].
CORS_ORIGINS = [o.strip() for o in os.environ.get("CARA_CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET"], allow_headers=[])
ADMIN_HEADER = "X-CARA-Admin"


def require_admin(request: Request):
    """POST /api/admin/* needs the X-CARA-Admin header (any web page the operator visits can send a
    plain cross-origin POST, but not a custom header without a CORS preflight, which is refused).
    With CARA_ADMIN_TOKEN set, the header must carry that token."""
    got = request.headers.get(ADMIN_HEADER)
    token = os.environ.get("CARA_ADMIN_TOKEN")
    if not got or (token and got != token):
        raise HTTPException(403, f"admin endpoint: send header '{ADMIN_HEADER}: "
                                 f"{'<CARA_ADMIN_TOKEN>' if token else '1'}'")


# ---------------------------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------------------------

def _src_str(d: dict) -> str:
    return str(d.get("status") or "unknown")


@app.get("/api/health")
def health():
    ensure_model()
    src = live.sources_status()
    live_sources = {"socrates": _src_str(src["socrates"]), "swpc": _src_str(src["swpc"]),
                    "satcat": _src_str(src["satcat"]), "spacetrack": _src_str(src["spacetrack"]),
                    "gp": _src_str(src["gp"])}
    la = STATE.live
    t = STATE.test
    status = "ok"
    if STATE.model_error or not STATE.model.trained or live_sources["socrates"] not in ("ok",):
        status = "degraded"
    return {
        "status": status, "model_loaded": bool(STATE.model.trained), "live_sources": live_sources,
        "live_sources_detail": src, "model": model_info(),
        "test_set": {"ready": t is not None, "n_cdms": len(t.df) if t else None,
                     "n_events": len(t.events) if t else None, "computed_at": t.computed_at if t else None,
                     "from_cache": t.from_cache if t else None, "error": STATE.test_error,
                     "path": str(SETTINGS.test_data)},
        "live_assessment": {"ready": la is not None, "n_rows": len(la.df) if la else None,
                            "assessed_at": live.iso(la.computed_at) if la else None,
                            "seconds": la.seconds if la else None,
                            "horizon_h": la.horizon_h if la else None, "error": STATE.live_error},
        "background": SETTINGS.background, "server_time": live.iso(live.utcnow()),
    }


@app.get("/api/metrics")
def metrics():
    p = SETTINGS.metrics
    if not p.exists():
        raise HTTPException(503, f"{p.name} not generated yet (run python src/train.py)")
    try:
        return json.loads(p.read_text(encoding="utf-8"))   # json accepts NaN; clean() maps it to null
    except ValueError as exc:
        raise HTTPException(503, f"{p.name} is being rewritten or invalid: {exc}")


@app.get("/api/test/events")
def test_events():
    t = ensure_test_set(wait_s=120)
    return {"events": t.events, "n_events": len(t.events), "n_cdms": len(t.df),
            "sorted_by": "assessment.p_tier[2] (P(Critical)) desc", "computed_at": t.computed_at,
            "model": {"trained": STATE.model.trained, "trained_at": (STATE.model.meta or {}).get("trained_at")}}


CDM_FIELDS = ["time_to_tca", "miss_distance", "relative_speed", "relative_position_r", "relative_position_t",
              "relative_position_n", "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
              "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP", "true_log_pc"]


@app.get("/api/test/events/{event_id}")
def test_event(event_id: int):
    t = ensure_test_set(wait_s=120)
    if event_id not in t.bounds:
        raise HTTPException(404, f"event {event_id} not in the test set")
    s, e = t.bounds[event_id]
    sub = t.df.iloc[s:e]
    ass, fc = assessment_records(sub), forecast_records(sub)
    cdms = []
    for i, (_, r) in enumerate(sub.iterrows()):
        d = {}
        for c in CDM_FIELDS:
            v = r[c]
            d[c] = v if isinstance(v, str) else fnum(v, 6)
        d["true_tier"] = int(r["true_tier"])
        d["assessment"] = ass[i]
        d["forecast"] = fc[i]
        cdms.append(d)
    return {"event_id": event_id, "n_cdms": len(cdms), "cdms": cdms,
            "display_only": DISPLAY_COLS + ["true_log_pc", "true_tier"]}


@app.get("/api/live/conjunctions")
def live_conjunctions(window_h: float = Query(24.0, gt=0), limit: int = Query(300, ge=1),
                      sort: Literal["risk", "tca", "range"] = "risk", points: int = Query(0, ge=0, le=1),
                      tier: Optional[str] = Query(None, pattern=r"^[0-2](,[0-2])*$",
                                                  description="only these tier codes, e.g. 1 or 1,2"),
                      needs_review: Optional[int] = Query(None, ge=0, le=1)):
    window_h = min(float(window_h), SETTINGS.max_window_h)
    limit = min(int(limit), SETTINGS.max_limit)
    la = get_live(window_h)
    now = live.utcnow()
    h = _hours(la, now)
    inw = (h >= -live.PAST_TCA_ALLOWANCE_H) & (h <= window_h)
    snap_h = (live.to_ns(la.snap.rows["tca"]) - pd.Timestamp(now).value) / NS_PER_H
    n_total = int((snap_h >= -live.PAST_TCA_ALLOWANCE_H).sum())
    sub = la.df.loc[inw].copy()
    sub["hours_to_tca"] = h[inw]
    warnings = live.summarize_warnings(sub, window_h)      # always the whole window, before any filter
    sub0 = sub
    tiers = sorted({int(x) for x in tier.split(",")}) if tier else None
    if tiers is not None:
        sub = sub.loc[sub["tier"].isin(tiers).to_numpy()]
    if needs_review is not None:
        sub = sub.loc[sub["needs_review"].astype(bool).to_numpy() == bool(needs_review)]
    if sort == "risk":
        order = _risk_order(sub)
    elif sort == "tca":
        order = np.argsort(sub["hours_to_tca"].to_numpy(), kind="mergesort")
    else:
        order = np.argsort(sub["miss_distance"].to_numpy(), kind="mergesort")
    top = sub.iloc[order[:limit]]
    snap = la.snap
    rows = conjunction_rows(top, snap.provenance, bool(points))
    resp = {
        "source": "CelesTrak SOCRATES Plus (sort-minRange.csv + sort-maxProb.csv)",
        "fetched_at": live.iso(snap.fetched_at), "data_as_of": live.iso(snap.data_as_of), "stale": snap.stale,
        "n_total": n_total, "window_h": window_h, "n_in_window": int(inw.sum()),
        "sort": sort, "limit": limit, "filter": {"tier": tiers, "needs_review": needs_review},
        "n_matching": int(len(sub)), "n_returned": len(rows), "warnings": warnings, "conjunctions": rows,
        "assessed_at": live.iso(la.computed_at), "assessment_seconds": la.seconds,
        "assessed_horizon_h": la.horizon_h,
        "model": {"kind": "screening model (reduced features: no covariance / RTN geometry in SOCRATES)",
                  "trained": "screen_q50" in STATE.model.models,
                  "trained_at": (STATE.model.meta or {}).get("trained_at"),
                  "thresholds": STATE.model.thresholds.get("screen", STATE.model.thresholds),
                  "n_heads_disagree": int(np.asarray(sub0["heads_disagree"], bool).sum())
                  if "heads_disagree" in sub0.columns else 0,
                  "ood_note": "Live SOCRATES rows are out of the training distribution (no covariance, mostly "
                              "payload secondaries, all <= 5 km): the estimated Pc and its interval are indicative "
                              "only; SOCRATES' max Pc is the screening reference."},
        "space_weather_used": la.sw_status, "files": snap.files, "satcat": snap.satcat_status,
        "note": "Screening-grade (TLE/SGP4, no covariance) - confirm with CSpOC CDMs before any manoeuvre.",
    }
    if points:
        resp["tle"] = live.STORE.stats()
        resp["points_summary"] = {s: sum(1 for r in rows if r["tle_status"] == s)
                                  for s in ("ok", "inconsistent", "pending", "missing")}
        resp["points_summary"]["plotted"] = sum(1 for r in rows if r["tca_point"] is not None)
    if snap.parse_error:
        resp["parse_error"] = snap.parse_error
    if STATE.live_error:
        resp["refresh_error"] = STATE.live_error
    return resp


@app.get("/api/live/spaceweather")
def live_spaceweather():
    sw = live.space_weather(history=True, history_network=not SETTINGS.background)
    if sw["f107"]["latest"] is None and sw["ap"]["latest"] is None and sw["kp"]["latest"] is None:
        raise HTTPException(503, f"space weather unavailable: {sw.get('errors')}")
    return sw


@app.get("/api/live/track")
def live_track(a: int = Query(..., gt=0), b: int = Query(..., gt=0), tca: str = Query(...),
               span_min: float = Query(50.0, gt=0, le=1440), step_s: float = Query(20.0, gt=0, le=600)):
    try:
        t0 = live.parse_time(tca)
    except (ValueError, TypeError):
        raise HTTPException(422, f"tca is not an ISO-8601 time: {tca!r}")
    try:
        return live.propagate(a, b, t0, span_min=span_min, step_s=step_s, store=live.STORE)
    except live.BudgetExhausted as exc:
        raise HTTPException(503, str(exc))
    except live.LiveSourceError as exc:
        raise HTTPException(404, f"element set unavailable: {exc}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


class CDMIn(BaseModel):
    """One CDM with the allowed columns (docs/api_contract.md). Extra keys are ignored and
    reported back; the label `risk` is never used."""
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    event_id: Optional[int] = None
    time_to_tca: float = Field(ge=0, le=30, description="days")
    miss_distance: float = Field(ge=0, description="m")
    relative_speed: float = Field(ge=0, description="m/s")
    relative_position_r: float
    relative_position_t: float
    relative_position_n: float
    c_object_type: str
    c_sigma_r: Optional[float] = Field(default=None, gt=0)
    t_sigma_r: float = Field(gt=0)
    max_risk_estimate: float = Field(ge=-30, le=0, description="log10 of the max-Pc bound")
    F10: float = Field(gt=0, le=1000)
    AP: float = Field(ge=0, le=1000)

    @field_validator("c_object_type")
    @classmethod
    def _type(cls, v: str) -> str:
        u = str(v).strip().upper()
        if u not in F.OBJECT_TYPES:
            raise ValueError(f"c_object_type must be one of {F.OBJECT_TYPES}")
        return u


class AssessIn(CDMIn):
    history: list[CDMIn] = Field(default_factory=list, max_length=200,
                                 description="earlier CDMs of the same event (time_to_tca >= this CDM's)")


@app.post("/api/assess")
def assess(body: AssessIn = Body(...)):
    ensure_model()
    cur = body.model_dump(exclude={"history"})
    ignored = sorted(set((body.model_extra or {}).keys()))
    hist = [h.model_dump() for h in body.history]
    for i, h in enumerate(hist):
        if h["time_to_tca"] < cur["time_to_tca"]:
            raise HTTPException(422, f"history[{i}].time_to_tca ({h['time_to_tca']}) is closer to TCA than the "
                                     f"assessed CDM ({cur['time_to_tca']}); the assessed CDM must be the latest")
    ev = cur.get("event_id") if cur.get("event_id") is not None else -1
    recs = sorted(hist, key=lambda r: -r["time_to_tca"]) + [cur]
    df = pd.DataFrame([{c: r.get(c) for c in F.ALLOWED_COLUMNS} for r in recs])
    df["event_id"] = ev
    df["c_sigma_r"] = pd.to_numeric(df["c_sigma_r"], errors="coerce")
    out = STATE.model.assess_cdms(df)
    row = out.iloc[[len(out) - 1]]
    res = assessment_records(row)[0]
    res["forecast"] = forecast_records(row)[0]
    md = cur["miss_distance"]
    pos = math.sqrt(cur["relative_position_r"] ** 2 + cur["relative_position_t"] ** 2 + cur["relative_position_n"] ** 2)
    warnings = []
    if md > 0 and abs(pos - md) / md > 0.05:
        warnings.append(f"|relative_position| = {pos:,.0f} m differs from miss_distance = {md:,.0f} m by more than 5%")
    res.update({"n_history": len(hist), "input_warnings": warnings,
                "ignored_fields": [k for k in ignored if k != "history"],
                "model": {"trained": STATE.model.trained, "set": "full" if STATE.model.trained else "physics fallback"},
                "tier_name": TIER_NAMES[res["tier"]]})
    return res


@app.post("/api/admin/reload-model", dependencies=[Depends(require_admin)])
def reload_model():
    t0 = time.perf_counter()
    info = load_model()
    test = None
    with STATE.test_lock:
        try:
            STATE.test = compute_test_set()
            STATE.test_error = None
            test = {"n_events": len(STATE.test.events), "n_cdms": len(STATE.test.df),
                    "seconds": STATE.test.seconds, "from_cache": STATE.test.from_cache}
        except Exception as exc:  # noqa: BLE001
            STATE.test_error = f"{type(exc).__name__}: {exc}"[:400]
            test = {"error": STATE.test_error}
    live_res = None
    if STATE.live is not None:
        try:   # same horizon as before, so a full (7-day) assessment is not replaced by a 26 h one
            la = compute_live(network=False, force_reassess=True, horizon_h=STATE.live.horizon_h)
            live_res = {"n_rows": len(la.df), "seconds": la.seconds, "horizon_h": la.horizon_h}
        except Exception as exc:  # noqa: BLE001
            live_res = {"error": str(exc)[:300]}
    return {"reloaded": STATE.model_error is None, "model": info, "test_set": test, "live": live_res,
            "seconds": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------------------------------------
# Dashboard (static) at "/"
# ---------------------------------------------------------------------------------------------

if SETTINGS.dashboard.is_dir():
    app.mount("/", StaticFiles(directory=str(SETTINGS.dashboard), html=True), name="dashboard")
else:
    @app.get("/")
    def _no_dashboard():
        return {"detail": f"dashboard directory not found: {SETTINGS.dashboard}", "api": "/docs"}
