"""Offline tests for the live data layer (src/live.py) and the API (src/server.py).

No network: `live._http_get` is replaced by a fake that serves tests/fixtures/, the cache lives in
a temp dir and "now" is frozen at 2026-11-05T00:00:00Z.
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ["CARA_BACKGROUND"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import live  # noqa: E402

FX = Path(__file__).resolve().parent / "fixtures"
NOW = datetime(2026, 11, 5, 0, 0, 0, tzinfo=timezone.utc)
LAST_MOD = "Thu, 05 Nov 2026 00:00:00 GMT"

URL_FILES = {
    live.SOCRATES_URLS["minRange"]: "socrates_minRange.csv",
    live.SOCRATES_URLS["maxProb"]: "socrates_maxProb.csv",
    live.SATCAT_URL: "satcat.csv",
    live.SWPC_URLS["f107"]: "swpc_f107.json",
    live.SWPC_URLS["kp"]: "swpc_kp.json",
    live.SWPC_URLS["afred"]: "swpc_afred.json",
    live.CELESTRAK_SW_URL: "SW-Last5Years.csv",
    live.GP_ACTIVE_CSV_URL: "gp_active.csv",
    live.GP_ACTIVE_URL: "gp_active.tle",
    live.GP_TLE_URL.format(norad=30520): "gp_30520.tle",
    live.GP_CSV_URL.format(norad=101127): "gp_101127.csv",
}


class FakeNet:
    def __init__(self):
        self.calls = []
        self.down = False
        self.status = {}

    def __call__(self, url, headers, timeout=None):
        self.calls.append(url)
        assert "CARA" in headers["User-Agent"]
        if self.down:
            raise ConnectionError("network down (test)")
        if url in self.status:
            return self.status[url], b"refused", {}
        name = URL_FILES.get(url)
        if name is None:
            if "gp.php?CATNR=" in url:                       # CelesTrak's answer for unknown objects
                return 200, b"No GP data found", {}
            return 404, b"not found", {}
        return 200, (FX / name).read_bytes(), {"Last-Modified": LAST_MOD}


@pytest.fixture
def net(tmp_path, monkeypatch):
    fake = FakeNet()
    monkeypatch.setattr(live, "_http_get", fake)
    monkeypatch.setattr(live, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(live, "utcnow", lambda: NOW)
    monkeypatch.setattr(live, "BULK_MIN_SETS", 1)
    monkeypatch.setattr(live, "SOCRATES_MIN_ROWS", 1)       # fixtures are a handful of rows
    monkeypatch.setattr(live, "SATCAT_MIN_ROWS", 1)
    monkeypatch.setattr(live, "GP_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(live, "STORE", live.TLEStore())
    live._memo.clear()
    return fake


# ------------------------------------------------------------------------------------------
# fetch layer
# ------------------------------------------------------------------------------------------

def test_fetch_caches_then_serves_stale_copy(net, monkeypatch):
    url = live.SATCAT_URL
    r1 = live.fetch("satcat", url, ttl_s=3600)
    assert not r1.from_cache and not r1.stale and r1.last_modified.year == 2026
    r2 = live.fetch("satcat", url, ttl_s=3600)
    assert r2.from_cache and len(net.calls) == 1               # TTL respected: no second request
    # cache expired + provider down -> last copy, flagged stale, error recorded
    monkeypatch.setattr(live, "utcnow", lambda: NOW + timedelta(hours=2))
    net.down = True
    r3 = live.fetch("satcat", url, ttl_s=3600)
    assert r3.stale and r3.content == r1.content and "network down" in r3.error
    assert len(net.calls) == 2
    r4 = live.fetch("satcat", url, ttl_s=3600)                  # in back-off: no new request
    assert r4.stale and len(net.calls) == 2
    assert live.cache_status("satcat")["status"] == "stale"


def test_fetch_without_cache_raises_clear_error_and_backs_off(net):
    net.down = True
    with pytest.raises(live.LiveSourceError, match="no cached copy"):
        live.fetch("swpc_kp", live.SWPC_URLS["kp"], ttl_s=60)
    with pytest.raises(live.LiveSourceError, match="back-off"):
        live.fetch("swpc_kp", live.SWPC_URLS["kp"], ttl_s=60)
    assert len(net.calls) == 1


def test_backoff_doubles_on_consecutive_failures(net, monkeypatch):
    net.down = True
    url = live.SWPC_URLS["f107"]
    for minutes, expect_calls in [(0, 1), (4, 1), (6, 2), (15, 2), (17, 3), (35, 3), (38, 4)]:
        monkeypatch.setattr(live, "utcnow", lambda m=minutes: NOW + timedelta(minutes=m))
        with pytest.raises(live.LiveSourceError):
            live.fetch("swpc_f107", url, ttl_s=60)
        assert len(net.calls) == expect_calls, minutes      # retries after 5, then 10, then 20 min
    net.down = False
    monkeypatch.setattr(live, "utcnow", lambda: NOW + timedelta(minutes=120))
    assert not live.fetch("swpc_f107", url, ttl_s=60).stale  # recovers; failure record cleared
    assert live._read_meta("swpc_f107.fail") is None


def test_http_403_backs_off_for_two_hours(net, monkeypatch):
    url = live.SOCRATES_URLS["minRange"]
    net.status[url] = 403
    with pytest.raises(live.LiveSourceError):
        live.fetch("socrates_minRange", url, ttl_s=60)
    monkeypatch.setattr(live, "utcnow", lambda: NOW + timedelta(minutes=90))
    with pytest.raises(live.LiveSourceError, match="back-off"):
        live.fetch("socrates_minRange", url, ttl_s=60)
    assert len(net.calls) == 1


# ------------------------------------------------------------------------------------------
# SOCRATES parsing, units, object types
# ------------------------------------------------------------------------------------------

def test_split_status_and_name_fallback():
    assert live.split_status("FENGYUN 1C DEB [-]") == ("FENGYUN 1C DEB", "-")
    assert live.split_status("ISS (ZARYA) [+]") == ("ISS (ZARYA)", "+")
    assert live.split_status("NO MARKER") == ("NO MARKER", "")
    assert live.name_type("FENGYUN 1C DEB", "-") == "DEBRIS"
    assert live.name_type("SL-16 R/B", "-") == "ROCKET BODY"
    assert live.name_type("SOME SAT", "+") == "PAYLOAD"
    assert live.name_type("SOME SAT", "B") == "PAYLOAD"
    assert live.name_type("DEBUT", "?") == "UNKNOWN"          # 'DEB' must be a word


def test_load_socrates_union_dedupe_and_units(net):
    rows, snap = live.load_socrates(now=NOW)
    # 6 minRange rows + 3 maxProb rows -> 7 unique (two duplicates incl. a swapped pair), 1 too old
    assert len(snap.rows) == 7
    assert len(rows) == 6
    assert (rows["hours_to_tca"] >= -1).all()
    r = rows.set_index(["norad_1", "norad_2"])
    s = r.loc[(65282, 30520)]
    assert s["name_1"] == "STARLINK-35021" and s["status_1"] == "+"
    assert s["name_2"] == "FENGYUN 1C DEB" and s["status_2"] == "-"
    assert s["miss_distance"] == pytest.approx(13.0)                     # km -> m
    assert s["relative_speed"] == pytest.approx(15124.0)                 # km/s -> m/s
    assert s["hours_to_tca"] == pytest.approx(13 + 9.5 / 60 + 0.991 / 3600, abs=1e-6)
    assert s["time_to_tca"] == pytest.approx(s["hours_to_tca"] / 24)    # days
    assert s["max_risk_estimate"] == pytest.approx(math.log10(0.2696))
    assert s["in_files"] == "sort-minRange.csv+sort-maxProb.csv"
    assert r.loc[(65282, 25544), "max_risk_estimate"] == -30.0           # MAX_PROB 0 -> floor
    assert r.loc[(65282, 25544), "hours_to_tca"] == pytest.approx(-0.5)  # recent past kept
    assert r.loc[(65282, 40000), "in_files"] == "sort-maxProb.csv"
    # swapped duplicate (22285, 25544) collapsed onto the minRange row
    assert ((rows["norad_1"] == 22285) | (rows["norad_2"] == 22285)).sum() == 1
    # space weather from the SWPC fixtures
    f107 = json.loads((FX / "swpc_f107.json").read_text())[-1]["flux"]
    assert (rows["F10"] == f107).all() and rows["AP"].notna().all()
    assert snap.data_as_of == datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert "sort-minRange.csv + sort-maxProb.csv" in snap.provenance


def test_object_types_from_satcat_with_name_fallback(net):
    rows, _ = live.load_socrates(now=NOW)
    t = dict(zip(rows["norad_2"], rows["c_object_type"]))
    src = dict(zip(rows["norad_2"], rows["type_src_2"]))
    assert t[30520] == "DEBRIS" and src[30520] == "SATCAT"
    assert t[22285] == "ROCKET BODY"
    assert t[40000] == "UNKNOWN" and src[40000] == "SATCAT"              # UNK -> UNKNOWN
    assert t[25544] == "PAYLOAD"                                         # PAY -> PAYLOAD
    assert t[99001] == "DEBRIS" and src[99001] == "name"                 # not in SATCAT
    assert t[101127] == "UNKNOWN" and src[101127] == "name"              # [?] and no DEB/R/B


def test_screening_rows_feed_the_model(net):
    sys.path.insert(0, str(ROOT / "src"))
    from predict import SCREENING_COLUMNS, RiskModel
    rows, _ = live.load_socrates(now=NOW)
    out = RiskModel().assess_screening(rows[SCREENING_COLUMNS])
    assert len(out) == len(rows)
    md_crit = rows["miss_distance"].to_numpy() < 1000
    assert (out["tier"].to_numpy()[md_crit] == 2).all()                  # hard distance rule
    assert (out["tier"] >= 1).all()                                      # every SOCRATES row <= 5 km


# ------------------------------------------------------------------------------------------
# warnings block
# ------------------------------------------------------------------------------------------

def test_warnings_counts():
    df = pd.DataFrame({
        "hours_to_tca": [-0.5, 0.2, 0.9, 5.5, 5.6, 23.9],
        "miss_distance": [800, 4000, 999.9, 1000, 5000, 300],
        "tier": [2, 1, 2, 1, 2, 2],
        "needs_review": [False, True, False, False, True, False],
        "name_1": ["A", "B", "A", "C", "A", "B"], "name_2": ["X"] * 6,
        "type_1": ["PAYLOAD"] * 6, "type_2": ["DEBRIS", "PAYLOAD", "ROCKET BODY", "PAYLOAD", "PAYLOAD", "DEBRIS"],
        "max_risk_estimate": [-3.0] * 6,
    })
    w = live.summarize_warnings(df, 24)
    # the -0.5 h row (TCA already passed) is only counted in n_recent_past, never as "next 24 h"
    assert w["window_h"] == 24 and w["n_in_window"] == 5 and w["n_recent_past"] == 1
    assert w["n_within_1km"] == 2 and w["n_within_5km"] == 5
    assert w["n_critical"] == 3 and w["n_elevated"] == 2 and w["n_needs_review"] == 2
    assert w["n_critical_by_pc"] == 1                                    # tier 2 at 5000 m
    assert w["n_debris_or_rb"] == 2
    assert len(w["by_hour"]) == 24
    h0 = w["by_hour"][0]
    assert (h0["critical"], h0["elevated"], h0["total"]) == (1, 1, 2)    # 0.2 h and 0.9 h only
    assert w["by_hour"][5] == {"hour": 5, "critical": 1, "elevated": 1, "total": 2}
    assert w["by_hour"][23]["critical"] == 1
    assert sum(b["total"] for b in w["by_hour"]) == 5
    assert w["top_primaries"][0]["count"] == 2 and w["top_primaries"][0]["name"] in ("A", "B")
    assert any("Closest approach" in s and "300 m" in s for s in w["insights"])
    assert any("last 1 h" in s for s in w["insights"])
    empty = live.summarize_warnings(df.iloc[:0], 6)
    assert empty["n_in_window"] == 0 and len(empty["by_hour"]) == 6 and empty["insights"] == []


# ------------------------------------------------------------------------------------------
# frames and SGP4
# ------------------------------------------------------------------------------------------

def test_gmst_vallado_example_3_5():
    from sgp4.api import jday
    jd, fr = jday(1992, 8, 20, 12, 14, 0.0)
    assert np.degrees(live.gmst_iau82(np.array([jd + fr])))[0] == pytest.approx(152.578787810, abs=1e-7)


def test_teme_to_ecef_vallado_2006():
    """Vallado et al. 2006 'Revisiting Spacetrack Report #3', TEME -> PEF example."""
    from sgp4.api import jday
    jd, fr = jday(2004, 4, 6, 7, 51, 28.386009 - 0.4399619)          # UT1 = UTC + dUT1
    r = live.teme_to_ecef(np.array([[5094.18016210, 6127.64465950, 6380.34453270]]), np.array([jd + fr]))[0]
    assert r == pytest.approx([-1033.47503130, 7901.30558560, 6380.34453270], abs=1e-4)   # km (0.1 m)


def test_rtn_frame():
    r = np.array([[7000.0, 0.0, 0.0]])
    v = np.array([[0.0, 7.5, 0.0]])
    basis = live.rtn_basis(r, v)[0]
    assert basis == pytest.approx(np.eye(3))                        # R=x, T=y, N=z for this orbit
    assert live.to_rtn(r, v, r + [[1.0, 2.0, 3.0]])[0] == pytest.approx([1.0, 2.0, 3.0])
    # rotated/inclined orbit: frame stays orthonormal and right-handed, R along r, N along h
    r2, v2 = np.array([[3000.0, 5000.0, 4000.0]]), np.array([[-5.0, 1.0, 3.0]])
    b = live.rtn_basis(r2, v2)[0]
    assert b @ b.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(b) == pytest.approx(1.0)
    assert b[0] == pytest.approx(r2[0] / np.linalg.norm(r2[0]))
    d = live.to_rtn(r2, v2, r2 + 10 * b[1][None, :])[0]            # 10 km along-track
    assert d == pytest.approx([0, 10, 0], abs=1e-9)


def test_geodetic_round_trip_and_poles():
    lat = np.array([0.0, 45.0, -89.9, 60.0])
    lon = np.array([0.0, 30.0, 120.0, -150.0])
    alt = np.array([0.0, 500.0, 35786.0, 1200.0])
    out = live.ecef_to_geodetic(live.geodetic_to_ecef(lat, lon, alt))
    assert out[:, 0] == pytest.approx(lat, abs=1e-7)
    assert out[:, 1] == pytest.approx(lon, abs=1e-7)
    assert out[:, 2] == pytest.approx(alt, abs=1e-5)                # km (1 cm)
    pole = live.ecef_to_geodetic(np.array([[0.0, 0.0, 6356752.314245 + 1000.0]]))[0]
    assert pole[0] == pytest.approx(90.0) and pole[2] == pytest.approx(1.0, abs=1e-6)


def test_catalog_numbers_and_tle_parsing():
    assert live.catalog_number("25544") == 25544
    assert live.catalog_number("A1127") == 101127                  # Alpha-5
    assert live.catalog_number("Z9999") == 339999
    cat = live.parse_tle_catalog((FX / "gp_active.tle").read_text())
    assert set(cat) == {25544, 65282} and cat[65282][0] == "STARLINK-35021"
    name, l1, l2 = live.parse_tle((FX / "gp_30520.tle").read_text(), 30520)
    assert name == "FENGYUN 1C DEB" and l1.startswith("1 30520")
    with pytest.raises(live.NoDataError):
        live.parse_tle("No GP data found", 1)
    with pytest.raises(ValueError):
        live.parse_tle((FX / "gp_30520.tle").read_text(), 12345)


def test_tle_store_bulk_per_object_and_status(net):
    st = live.STORE
    assert st.lookup(65282) is None                       # nothing cached, and lookup never fetches
    assert net.calls == []
    o = st.get(101127, network=True)                      # 6-digit id, no bulk file yet -> OMM CSV query
    assert net.calls == [live.GP_CSV_URL.format(norad=101127)]
    assert o.omm is not None and o.satrec.satnum_str.strip() != "" and st.status(101127) == "ok"
    assert abs((o.epoch - datetime(2026, 10, 30, 13, 9, 13, tzinfo=timezone.utc)).total_seconds()) < 1
    st.load_bulk(network=True)                            # one bulk OMM CSV download, never the TLE format
    assert net.calls[-1] == live.GP_ACTIVE_CSV_URL and live.GP_ACTIVE_URL not in net.calls
    assert st.status(65282) == "ok" and st.status(25544) == "ok"
    b = live.TLEStore()
    b.load_bulk(network=False)                            # bulk alone (disk cache) resolves 6-digit ids
    assert b._from_bulk(101127).omm is not None and b._from_bulk(101127).epoch == o.epoch
    assert abs(b._from_bulk(65282).epoch_jd - live.TLE(65282, "", *live.parse_tle_catalog(
        (FX / "gp_active.tle").read_text())[65282][1:], None).epoch_jd) < 1e-8
    assert st.status(30520) == "pending"                  # not in the bulk file
    assert st.request([30520, 65282, 30520]) == 1         # only the unresolved object is queued
    t = st.get(30520, network=True)                       # CATNR query (what the worker does)
    assert t.name == "FENGYUN 1C DEB" and st.status(30520) == "ok"
    with pytest.raises(live.LiveSourceError):
        st.get(77777, network=True)                       # "No GP data found"
    assert st.status(77777) == "missing"
    assert st.budget_used() == 3
    n = len(net.calls)
    st.get(30520, network=True)                           # cached: no new request, no budget
    assert len(net.calls) == n and st.budget_used() == 3


def test_tle_store_budget_and_circuit_breaker(net, monkeypatch):
    monkeypatch.setattr(live, "GP_BUDGET", 2)
    monkeypatch.setattr(live, "GP_INTERACTIVE_RESERVE", 1)
    st = live.STORE
    st.get(30520, network=True, interactive=False)          # uses the only background slot
    with pytest.raises(live.BudgetExhausted):
        st.get(101127, network=True, interactive=False)
    st.get(101127, network=True, interactive=True)          # reserve kept for user-triggered tracks
    monkeypatch.setattr(live, "GP_BUDGET", 100)
    net.down = True
    for nid in (11, 12, 13):
        with pytest.raises(live.LiveSourceError):
            st.get(nid, network=True)
    with pytest.raises(live.ProviderPaused):                # breaker open: no request is made
        st.get(14, network=True)
    assert st.stats()["paused_until"] is not None
    assert st.status(11) == "pending"                       # transient failures stay pending
    fresh = live.TLEStore()                                 # a restarted server stays paused
    with pytest.raises(live.ProviderPaused):
        fresh.get(15, network=True)
    calls, used = len(net.calls), fresh.budget_used()
    with pytest.raises(live.LiveSourceError):               # object still in its own fetch back-off:
        fresh.get(11, network=True)                         # no request, no budget, breaker untouched
    assert len(net.calls) == calls and fresh.budget_used() == used and fresh._consecutive_fail == 0


def test_bulk_uses_fresh_legacy_tle_cache_then_switches_to_csv(net, monkeypatch):
    live.fetch("gp_active", live.GP_ACTIVE_URL, live.TTL["gp_bulk"])      # a cache written by an older version
    n = len(net.calls)
    st = live.TLEStore()
    res = st.load_bulk(network=True)
    assert res.key == "gp_active" and len(net.calls) == n                 # still fresh: no request at all
    assert st.status(65282) == "ok" and st.lookup(101127) is None          # TLE format lacks 6-digit ids
    monkeypatch.setattr(live, "utcnow", lambda: NOW + timedelta(hours=7))
    st2 = live.TLEStore()
    res = st2.load_bulk(network=True)                                      # expired -> OMM CSV, never TLE
    assert res.key == "gp_active_csv" and net.calls[n:] == [live.GP_ACTIVE_CSV_URL]
    assert st2._from_bulk(101127) is not None


def test_offline_switch_serves_cache_without_requests_or_backoff(net, monkeypatch):
    url = live.SOCRATES_URLS["minRange"]
    live.fetch("socrates_minRange", url, live.TTL["socrates"])
    meta_before = live._read_meta("socrates_minRange")
    monkeypatch.setattr(live, "utcnow", lambda: NOW + timedelta(hours=3))  # cache expired
    monkeypatch.setenv("CARA_OFFLINE", "celestrak.org")
    n = len(net.calls)
    res = live.fetch("socrates_minRange", url, live.TTL["socrates"])
    assert res.stale and "CARA_OFFLINE" in res.error and len(net.calls) == n
    assert live._read_meta("socrates_minRange") == meta_before            # no back-off recorded
    hs = live.cache_status("socrates_minRange", live.TTL["socrates"])      # /api/health still says stale
    assert hs["status"] == "stale" and "overdue" in hs["error"]
    assert live.cache_status("socrates_minRange")["status"] == "ok"        # (TTL-less view: no failure)
    with pytest.raises(live.LiveSourceError, match="CARA_OFFLINE"):
        live.fetch("satcat", live.SATCAT_URL, live.TTL["satcat"])          # nothing cached
    assert live._read_meta("satcat.fail") is None
    sw = live.fetch("swpc_kp", live.SWPC_URLS["kp"], live.TTL["swpc"])     # other hosts still allowed
    assert not sw.stale and net.calls[-1] == live.SWPC_URLS["kp"]
    st = live.TLEStore()
    with pytest.raises(live.ProviderPaused):                                # per-object: no budget, no breaker
        st.get(30520, network=True)
    assert st.budget_used() == 0 and st._consecutive_fail == 0 and st._trips == 0
    monkeypatch.setenv("CARA_OFFLINE", "1")
    with pytest.raises(live.LiveSourceError):
        live.fetch("swpc_f107", live.SWPC_URLS["f107"], live.TTL["swpc"])
    monkeypatch.setenv("CARA_OFFLINE", "0")
    res = live.fetch("socrates_minRange", url, live.TTL["socrates"])      # back online: refreshes at once
    assert not res.stale and net.calls[-1] == url


def test_propagate_and_tca_point_with_fixture_tles(net):
    st = live.STORE
    st.load_bulk(network=True)
    tca = "2026-11-05 13:09:30.991"
    r = live.propagate(65282, 30520, tca, store=st)
    assert r["requested"]["span_min"] == 50 and r["requested"]["n_samples"] == 151
    assert len(r["a"]["ecef"]) == 151 and len(r["b"]["lla"]) == 151 and len(r["relative_rtn"]) == 151
    # newer TLEs than SOCRATES used (1 day), propagated 6 days: same geometry, tens of km apart
    assert r["relative_speed_ms"] == pytest.approx(15124, abs=50)
    assert 0 < r["min_range_m"] < 100e3 and abs(r["tca_offset_s"]) < 60 and not r["min_at_edge"]
    # min range from the refined search <= every sampled range
    sampled = [math.sqrt(x[1] ** 2 + x[2] ** 2 + x[3] ** 2) for x in r["relative_rtn"]]
    assert r["min_range_m"] <= min(sampled) + 1.0
    # altitude ~ 450-560 km, |ECEF| consistent with lat/lon/alt
    for (_, x, y, z), (_, lat, lon, alt) in zip(r["a"]["ecef"][::25], r["a"]["lla"][::25]):
        assert 400 < alt < 600 and -90 <= lat <= 90 and -180 <= lon <= 180
        assert live.geodetic_to_ecef(lat, lon, alt) == pytest.approx([x, y, z], abs=5.0)
    # tca_point = midpoint of both ECEF positions at the requested TCA (sample 75)
    p = live.tca_point(65282, 30520, tca, store=st)
    a75, b75 = np.array(r["a"]["ecef"][75][1:]), np.array(r["b"]["ecef"][75][1:])
    mid = live.ecef_to_geodetic(((a75 + b75) / 2)[None, :])[0]
    assert p["lat"] == pytest.approx(mid[0], abs=1e-3) and p["lon"] == pytest.approx(mid[1], abs=1e-3)
    assert p["alt_km"] == pytest.approx(mid[2], abs=0.01)
    assert p["range_km"] == pytest.approx(np.linalg.norm(a75 - b75) / 1000, abs=0.01)
    assert r["tca_point"] == p
    assert live.tca_point(65282, 99999999, tca, store=st) is None


# ------------------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------------------

@pytest.fixture
def client(net, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import server
    empty_models = tmp_path / "models"
    empty_models.mkdir()
    monkeypatch.setattr(server.SETTINGS, "model_dir", empty_models)      # physics fallback model
    monkeypatch.setattr(server.SETTINGS, "test_data", FX / "test_data_small.csv")
    monkeypatch.setattr(server.SETTINGS, "metrics", FX / "metrics.json")
    monkeypatch.setattr(server.SETTINGS, "background", False)
    monkeypatch.setattr(server, "STATE", server.State())
    live.STORE.load_bulk(network=True)
    with TestClient(server.app) as c:
        yield c


def _no_nan(o):
    if isinstance(o, float):
        assert math.isfinite(o)
    elif isinstance(o, dict):
        for v in o.values():
            _no_nan(v)
    elif isinstance(o, list):
        for v in o:
            _no_nan(v)


ASSESS_KEYS = {"log_pc", "log_pc_lo", "log_pc_hi", "p_pc_elevated", "p_pc_critical", "p_tier", "md_tier",
               "tier", "needs_review", "reasons"}


def test_api_health_and_metrics(client):
    h = client.get("/api/health").json()
    assert set(h["live_sources"]) >= {"socrates", "swpc", "satcat", "spacetrack"}
    assert h["live_sources"]["spacetrack"] == "disabled"
    assert h["model_loaded"] is False and h["model"]["fallback"]
    m = client.get("/api/metrics")
    assert m.status_code == 200 and m.json()["nan_value"] is None        # NaN -> null
    assert "NaN" not in m.text


def test_api_test_events(client):
    r = client.get("/api/test/events")
    assert r.status_code == 200
    ev = r.json()["events"]
    raw = pd.read_csv(FX / "test_data_small.csv")
    assert len(ev) == raw["event_id"].nunique()
    pc = [e["assessment"]["p_tier"][2] for e in ev]
    assert pc == sorted(pc, reverse=True)
    for e in ev:
        assert ASSESS_KEYS <= set(e["assessment"])
        g = raw[raw["event_id"] == e["event_id"]]
        assert e["n_cdms"] == len(g)
        last = g.loc[g["time_to_tca"].idxmin()]
        assert e["time_to_tca"] == pytest.approx(last["time_to_tca"])     # latest CDM
        assert e["true_log_pc"] == pytest.approx(last["risk"])
        import features
        assert e["true_tier"] == int(features.risk_tier([last["risk"]], [last["miss_distance"]])[0])
        assert "forecast" in e
    _no_nan(r.json())
    eid = ev[0]["event_id"]
    d = client.get(f"/api/test/events/{eid}").json()
    assert d["event_id"] == eid and len(d["cdms"]) == ev[0]["n_cdms"]
    t = [c["time_to_tca"] for c in d["cdms"]]
    assert t == sorted(t, reverse=True)
    c0 = d["cdms"][0]
    for k in ["miss_distance", "relative_speed", "relative_position_r", "relative_velocity_n", "c_sigma_r",
              "t_sigma_r", "max_risk_estimate", "true_log_pc", "assessment"]:
        assert k in c0
    assert client.get("/api/test/events/987654321").status_code == 404
    assert (live.CACHE_DIR / "test_assessments.pkl").exists()             # cached, in the temp cache dir


def test_api_live_conjunctions(client):
    r = client.get("/api/live/conjunctions?window_h=24&limit=300&sort=risk")
    assert r.status_code == 200
    j = r.json()
    for k in ["source", "fetched_at", "data_as_of", "stale", "n_total", "window_h", "warnings", "conjunctions"]:
        assert k in j
    assert j["n_total"] == 6 and j["window_h"] == 24
    w = j["warnings"]
    assert j["n_in_window"] == 5                             # feed: 36 h row excluded, -0.5 h row included
    assert w["n_in_window"] == 4 and w["n_recent_past"] == 1  # warnings: only TCAs still ahead
    assert w["n_within_1km"] == sum(1 for x in j["conjunctions"] if x["hours_to_tca"] >= 0 and x["miss_distance"] < 1000) and w["n_within_5km"] == 4
    assert w["n_critical"] + w["n_elevated"] == 4 and len(w["by_hour"]) == 24
    rows = j["conjunctions"]
    assert len(rows) == 5
    tiers = [x["assessment"]["tier"] for x in rows]
    assert tiers == sorted(tiers, reverse=True)
    s = next(x for x in rows if x["sat1"]["norad"] == 65282 and x["sat2"]["norad"] == 30520)
    assert s["id"] == "65282-30520-2026-11-05T13:09:30Z"
    assert s["tca"] == "2026-11-05T13:09:30.991Z"
    assert s["miss_distance"] == 13.0 and s["relative_speed"] == 15124.0 and s["max_prob"] == 0.2696
    assert s["sat2"] == {"norad": 30520, "name": "FENGYUN 1C DEB", "status": "-", "type": "DEBRIS",
                         "tle_age_d": 7.206, "type_source": "SATCAT"}
    assert s["assessment"]["tier"] == 2 and "Last-Modified 2026-11-05T00:00:00Z" in s["provenance"]
    assert "tca_point" not in s
    _no_nan(j)
    by_tca = client.get("/api/live/conjunctions?sort=tca&window_h=48").json()["conjunctions"]
    h = [x["hours_to_tca"] for x in by_tca]
    assert h == sorted(h) and len(h) == 6
    by_rng = client.get("/api/live/conjunctions?sort=range&limit=2").json()["conjunctions"]
    assert [x["miss_distance"] for x in by_rng] == [13.0, 600.0]
    assert client.get("/api/live/conjunctions?sort=bogus").status_code == 422
    assert client.get("/api/live/conjunctions?limit=999999").json()["limit"] == 2000
    # server-side filters (the risk-sorted top N can be all Critical): warnings still cover the window
    for t in ("2", "1", "0,1"):
        f = client.get(f"/api/live/conjunctions?window_h=24&tier={t}").json()
        want = {int(x) for x in t.split(",")}
        assert f["warnings"] == w and f["filter"]["tier"] == sorted(want)
        assert all(x["assessment"]["tier"] in want for x in f["conjunctions"])
        assert f["n_matching"] == sum(1 for x in rows if x["assessment"]["tier"] in want)
    nr = client.get("/api/live/conjunctions?window_h=24&needs_review=1").json()
    assert nr["n_matching"] == sum(1 for x in rows if x["assessment"]["needs_review"])
    assert all(x["assessment"]["needs_review"] for x in nr["conjunctions"])
    assert client.get("/api/live/conjunctions?tier=3").status_code == 422
    assert client.get("/api/live/conjunctions?needs_review=2").status_code == 422


def test_live_assessment_horizon(client):
    import server
    part = server.compute_live(horizon_h=4.0)                 # what the request path does first
    h = part.df["hours_to_tca"].to_numpy()
    assert len(part.df) == 2 and (h <= 4.0).all()
    assert server._covers(part, 3.0) and not server._covers(part, 10.0) and not server._covers(part, None)
    full = server.compute_live(horizon_h=None)                # what the background loop does
    assert len(full.df) == 6 and server._covers(full, 1000.0) and server._covers(full, None)
    assert server.compute_live(horizon_h=4.0) is full          # a full set also serves short windows


def test_api_live_conjunctions_points(client):
    j = client.get("/api/live/conjunctions?points=1").json()
    rows = {(x["sat1"]["norad"], x["sat2"]["norad"]): x for x in j["conjunctions"]}
    s = rows[(65282, 30520)]
    assert s["tle_status"] == "pending" and s["tca_point"] is None       # 30520 not in bulk -> queued
    iss = rows[(65282, 25544)]                                            # both in bulk -> checked
    assert iss["tle_status"] in ("ok", "inconsistent") and iss["sgp4_check"] is not None
    if iss["tca_point"] is not None:
        assert set(iss["tca_point"]) >= {"lat", "lon", "alt_km", "range_km"} and -90 <= iss["tca_point"]["lat"] <= 90
    assert 30520 in live.STORE._queue                                     # queued for the worker
    live.STORE.get(30520, network=True)                                   # what the worker would do
    j2 = client.get("/api/live/conjunctions?points=1").json()
    s2 = next(x for x in j2["conjunctions"] if x["sat2"]["norad"] == 30520)
    # fixture TLEs are newer than SOCRATES' and propagated 6 days: tens of km vs SOCRATES' 13 m
    chk = s2["sgp4_check"]
    assert s2["tle_status"] == "inconsistent" and not chk["agrees"] and chk["miss_distance_m"] == 13.0
    assert 1000 < chk["min_range_m"] < 100e3 and chk["distance_tier"] in (0, 1) and "re-screen" in chk["note"]
    assert 400 < s2["tca_point"]["alt_km"] < 600                          # < 100 km apart: still plotted
    assert s2["tca_point"]["range_km"] == pytest.approx(chk["min_range_m"] / 1000, abs=1e-3)
    tr = client.get("/api/live/track", params={"a": 65282, "b": 30520, "tca": s2["tca"]}).json()
    assert chk["min_range_m"] == pytest.approx(tr["min_range_m"], rel=0.01)   # same closest approach as the track
    assert all(isinstance(x, float) for x in chk["epoch_shift_h"])
    assert j2["points_summary"]["ok"] + j2["points_summary"]["inconsistent"] >= 2


def test_api_spaceweather(client):
    j = client.get("/api/live/spaceweather").json()
    f = json.loads((FX / "swpc_f107.json").read_text())
    assert j["f107"]["latest"] == f[-1]["flux"] and len(j["f107"]["series"]) == len(f)
    assert {"time_tag", "flux", "time", "value"} <= set(j["f107"]["series"][0])
    assert j["kp"]["latest"] is not None and {"time_tag", "Kp"} <= set(j["kp"]["series"][-1])
    assert {"date", "ap"} <= set(j["ap"]["series"][0])
    assert len(j["ap"]["series"]) >= 5 + 2                  # SWPC days + CelesTrak history before them
    assert j["fetched_at"] is not None


def test_api_track(client):
    r = client.get("/api/live/track", params={"a": 65282, "b": 30520, "tca": "2026-11-05T13:09:30.991Z"})
    assert r.status_code == 200
    j = r.json()
    assert j["a"]["name"] == "STARLINK-35021" and j["b"]["name"] == "FENGYUN 1C DEB"
    assert len(j["a"]["ecef"]) == 151 and len(j["a"]["ecef"][0]) == 4 and j["a"]["tle_epoch"].endswith("Z")
    assert len(j["relative_rtn"][0]) == 4 and j["min_range_m"] > 0 and j["tca_model"].endswith("Z")
    assert j["tca_point"]["alt_km"] > 300
    r2 = client.get("/api/live/track", params={"a": 65282, "b": 30520, "tca": "2026-11-05T13:09:30Z",
                                              "span_min": 20, "step_s": 10})
    assert r2.json()["requested"]["n_samples"] == 121
    assert client.get("/api/live/track", params={"a": 65282, "b": 30520, "tca": "not-a-time"}).status_code == 422
    assert client.get("/api/live/track", params={"a": -1, "b": 30520, "tca": "2026-11-05"}).status_code == 422
    assert client.get("/api/live/track", params={"a": 65282, "b": 77777, "tca": "2026-11-05"}).status_code == 404


GOOD_CDM = {"time_to_tca": 2.3, "miss_distance": 3180.0, "relative_speed": 11200.0,
            "relative_position_r": 120.0, "relative_position_t": -3000.0, "relative_position_n": 1050.0,
            "c_object_type": "debris", "c_sigma_r": 150.0, "t_sigma_r": 12.0, "max_risk_estimate": -4.6,
            "F10": 90.0, "AP": 11.0}


def test_api_assess_and_validation(client):
    r = client.post("/api/assess", json=GOOD_CDM)
    assert r.status_code == 200, r.text
    a = r.json()
    assert ASSESS_KEYS <= set(a) and a["md_tier"] == 1 and a["tier"] >= 1
    assert sum(a["p_tier"]) == pytest.approx(1.0, abs=1e-6)
    hist = [dict(GOOD_CDM, time_to_tca=5.0, miss_distance=4100.0), dict(GOOD_CDM, time_to_tca=3.1)]
    r = client.post("/api/assess", json=dict(GOOD_CDM, history=hist, risk=-2.0))
    assert r.status_code == 200 and r.json()["n_history"] == 2 and r.json()["ignored_fields"] == ["risk"]
    bad = [
        {k: v for k, v in GOOD_CDM.items() if k != "miss_distance"},   # missing field
        dict(GOOD_CDM, miss_distance=-5),                              # negative distance
        dict(GOOD_CDM, c_object_type="SPACESHIP"),                     # unknown type
        dict(GOOD_CDM, max_risk_estimate=1.5),                         # log10 Pc must be <= 0
        dict(GOOD_CDM, t_sigma_r="abc"),                               # not a number
        dict(GOOD_CDM, history=[dict(GOOD_CDM, time_to_tca=1.0)]),     # history closer to TCA
    ]
    for b in bad:
        assert client.post("/api/assess", json=b).status_code == 422, b
    assert client.post("/api/assess", content="{not json", headers={"content-type": "application/json"}).status_code == 422
    crit = client.post("/api/assess", json=dict(GOOD_CDM, miss_distance=400.0, relative_position_t=-300.0,
                                               relative_position_n=200.0)).json()
    assert crit["tier"] == 2 and "hard rule" in crit["reasons"][0]
    assert a["input_warnings"] == []
    odd = client.post("/api/assess", json=dict(GOOD_CDM, miss_distance=10000.0)).json()
    assert any("differs from miss_distance" in w for w in odd["input_warnings"])


def test_api_admin_reload_and_dashboard(client):
    assert client.post("/api/admin/reload-model").status_code == 403     # plain cross-site POST refused
    r = client.post("/api/admin/reload-model", headers={"X-CARA-Admin": "1"})
    assert r.status_code == 200 and r.json()["test_set"]["n_events"] == 4
    assert r.json()["model"]["trained"] is False
    root = client.get("/")
    assert root.status_code == 200 and "<html" in root.text.lower()


# ------------------------------------------------------------------------------------------
# robustness: corrupt payloads, half-written models, admin endpoint
# ------------------------------------------------------------------------------------------

def _with_extra_field(lines, i):
    f = lines[i].split(",")
    f[1] += ", EXTRA"                                                  # unquoted comma in a name
    return lines[:i] + [",".join(f)] + lines[i + 1:]


def test_socrates_malformed_lines_are_dropped_not_fatal(net):
    lines = (FX / "socrates_minRange.csv").read_text().splitlines()
    good = live.parse_socrates({"minRange": "\n".join(lines).encode()})
    for i in (1, 3):                                   # first data row (pandas would infer an index) and later
        df = live.parse_socrates({"minRange": "\n".join(_with_extra_field(lines, i)).encode()})
        assert len(df) == len(good) - 1 and df.attrs["skipped_rows"] == {"minRange": 1}
        assert set(df["norad_1"]) <= set(good["norad_1"]) and df["range_km"].notna().all()
    with pytest.raises(ValueError, match="rows parsed"):               # a truncated download is rejected
        live._validate_table(("NORAD_CAT_ID_1",), "SOCRATES_MIN_ROWS")(lines[0].encode() + b"\n")


def test_unparseable_socrates_cache_serves_previous_parse(net):
    first = live.socrates_snapshot()
    data_path, _ = live._paths("socrates_minRange")
    data_path.write_bytes(b"NORAD_CAT_ID_1,TCA_RANGE,MAX_PROB\n1,2,3\n")     # columns missing
    meta = json.loads(live._paths("socrates_minRange")[1].read_text())
    meta["sha_len"] = -1                                                   # new payload version
    live._paths("socrates_minRange")[1].write_text(json.dumps(meta))
    snap = live.socrates_snapshot()
    assert snap.parse_error and snap.stale and len(snap.rows) == len(first.rows)
    assert live.sources_status()["socrates"]["status"] == "stale"
    assert live.sources_status()["socrates"]["parse_error"]


def test_malformed_bulk_omm_record_is_dropped(net):
    rows = (FX / "gp_active.csv").read_text().splitlines()
    head = rows[0].split(",")
    bad = rows[1].split(",")
    bad[head.index("MEAN_MOTION")] = ""
    (live.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    st = live.TLEStore()
    res = live.FetchResult("gp_active_csv", live.GP_ACTIVE_CSV_URL, "\n".join([rows[0], ",".join(bad)] + rows[2:]).encode(),
                           NOW, None, meta={"sha_len": 1, "downloaded_at": "x"})
    import unittest.mock as um
    with um.patch.object(live, "fetch_cached", lambda key, url: res if key == "gp_active_csv" else None):
        st.load_bulk(network=False)
    assert int(bad[head.index("NORAD_CAT_ID")]) not in st._bulk and len(st._bulk) == len(rows) - 2
    assert st.lookup(int(bad[head.index("NORAD_CAT_ID")])) is None


def _tiny_booster(path):
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    b = lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 4},
                  lgb.Dataset(X, X[:, 0]), num_boost_round=3)
    b.save_model(str(path))


def test_model_reload_rejects_mismatched_and_half_written_sets(client, tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "MODEL_SETTLE_S", 0.0)
    md = tmp_path / "models_new"
    md.mkdir()
    monkeypatch.setattr(server.SETTINGS, "model_dir", md)
    for h in server.HEADS:
        _tiny_booster(md / f"full_{h}.txt")
    # boosters that do not match the features (no / wrong meta.json): the smoke test rejects them
    info = server.load_model()
    assert info["trained"] is False and "LightGBMError" in info["error"]
    # a half-written booster never reaches LightGBM (which segfaults on it): rejected, server alive
    p = md / "full_q90.txt"
    p.write_bytes(p.read_bytes()[:len(p.read_bytes()) // 2])
    info = server.load_model()
    assert info["trained"] is False and "incomplete" in info["error"]
    assert client.get("/api/health").json()["model"]["error"] == info["error"]
    assert client.post("/api/assess", json=GOOD_CDM).status_code == 200      # previous model still serves


def test_collision_point_survives_malformed_element_set(client, monkeypatch):
    import server
    t = live.STORE.lookup(65282)
    broken = live.TLE(25544, "BROKEN", None, None, NOW, omm={"MEAN_MOTION": ""})
    monkeypatch.setattr(live.STORE, "lookup", lambda n: broken if n == 25544 else t)
    p, st, chk = server.collision_point(65282, 25544, "2026-11-05T13:09:30Z", 0, 13.0)
    assert (p, st, chk) == (None, "missing", None)
