"""Live data layer for the CARA dashboard.

Sources (all public, see docs/api_contract.md):
  * CelesTrak SOCRATES Plus  sort-minRange.csv + sort-maxProb.csv   (live conjunction screening)
  * CelesTrak SATCAT         satcat.csv                             (object type of each NORAD id)
  * CelesTrak GP             gp.php?GROUP=active&FORMAT=csv         (bulk OMM, every active object incl.
                                                                     6-digit catalog numbers)
                             gp.php?CATNR=<id>&FORMAT=TLE|CSV       (one object per query, throttled)
  CARA_OFFLINE=1 (or a comma list of hosts) disables network access: the disk cache is served.
  * NOAA SWPC                F10.7 flux, planetary K/ap index        (model inputs F10 / AP)
  * CelesTrak SpaceData      SW-Last5Years.csv                      (space-weather fallback / history)
  * Space-Track cdm_public   only when SPACETRACK_USER / SPACETRACK_PASS are set in the env

Every download goes through `fetch()`, which caches the payload on disk (CACHE_DIR) with a
per-source TTL, sends a descriptive User-Agent, uses conditional GETs, backs off after failures
(2 h after an HTTP 403/429, which is how CelesTrak signals "you already have this data") and serves
the last cached copy with stale=True when the network or the provider is unavailable.
SOCRATES is never re-downloaded within its 2 h TTL unless forced (and even a forced refresh is
limited to one request per FORCE_MIN_INTERVAL_S).

Per-object GP queries are made only by `TLEStore`: <= 2 requests/s, <= GP_BUDGET new objects per
2 h, 24 h disk cache, done by a background thread so API calls never wait on them.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("cara.live")

ROOT = Path(__file__).resolve().parents[1]


def _cache_dir() -> Path:
    if override := os.environ.get("CARA_CACHE_DIR"):
        return Path(override)
    # Vercel's application bundle is read-only; only temporary storage is writable.
    # Keep the persistent project-local cache for ordinary server/local runs.
    if os.environ.get("VERCEL") == "1":
        return Path(tempfile.gettempdir()) / "cara-live-cache"
    return ROOT / "live_cache"


CACHE_DIR = _cache_dir()

USER_AGENT = ("CARA-collision-risk-dashboard/0.2 (Autonomous SG 2026 hackathon, Airbus track; "
              "non-commercial research demo; disk-cached, polls SOCRATES at most every 2 h)")
HTTP_TIMEOUT = (10, 90)          # (connect, read) seconds; SOCRATES files are ~22 MB
FAILURE_BACKOFF_S = 300          # after a failed request, do not retry the same URL for 5 min
REFUSED_BACKOFF_S = 2 * 3600     # after HTTP 403 / 429 (provider refusing us) wait 2 h
FORCE_MIN_INTERVAL_S = 600       # even forced refreshes hit the network at most every 10 min
MAX_BACKOFF_S = 6 * 3600         # consecutive failures double the back-off up to 6 h

HOUR = 3600
TTL = {                          # seconds
    "socrates": 2 * HOUR,        # files update ~3x/day
    "satcat": 24 * HOUR,
    "gp": 24 * HOUR,             # one object (CATNR query)
    "gp_bulk": 6 * HOUR,         # GROUP=active bulk OMM CSV
    "swpc": 1 * HOUR,
    "celestrak_sw": 6 * HOUR,
    "spacetrack": 1 * HOUR,
}

SOCRATES_URLS = {
    "minRange": "https://celestrak.org/SOCRATES/sort-minRange.csv",
    "maxProb": "https://celestrak.org/SOCRATES/sort-maxProb.csv",
}
SATCAT_URL = "https://celestrak.org/pub/satcat.csv"
# Bulk element sets of every active object. The TLE format cannot carry 6-digit catalog numbers
# (CelesTrak leaves those objects out of TLE output), and most new payloads now have one, so the
# bulk file is downloaded as OMM CSV. The TLE copy is only read from an existing disk cache (it is
# never downloaded again: fetching the same GP data twice is what CelesTrak answers with HTTP 403).
GP_ACTIVE_CSV_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv"
GP_ACTIVE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
GP_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE"
# The TLE format cannot carry 6-digit catalog numbers (CelesTrak omits them from TLE output), so
# objects >= 100000 are fetched as OMM CSV, which python-sgp4 initialises directly.
GP_CSV_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=CSV"
SWPC_URLS = {
    "f107": "https://services.swpc.noaa.gov/products/10cm-flux-30-day.json",
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "afred": "https://services.swpc.noaa.gov/json/predicted_fredericksburg_a_index.json",
}
CELESTRAK_SW_URL = "https://celestrak.org/SpaceData/SW-Last5Years.csv"
SPACETRACK_LOGIN = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_CDM_URL = ("https://www.space-track.org/basicspacedata/query/class/cdm_public/"
                      "TCA/%3Enow/orderby/TCA%20asc/limit/500/format/json")

LOG_PC_FLOOR = -30.0
PAST_TCA_ALLOWANCE_H = 1.0       # keep conjunctions whose TCA passed less than 1 h ago
SATCAT_TYPE_MAP = {"PAY": "PAYLOAD", "R/B": "ROCKET BODY", "DEB": "DEBRIS", "UNK": "UNKNOWN"}
ACTIVE_MARKERS = {"+", "P", "B", "X"}   # operational / partially operational / backup / extended

# Per-object GP throttling (CelesTrak blocks abusive clients)
GP_MIN_INTERVAL_S = 0.5          # <= 2 requests / s
GP_BUDGET = 300                  # <= 300 per-object network requests ...
GP_BUDGET_WINDOW_S = 2 * HOUR    # ... per rolling 2 h
GP_INTERACTIVE_RESERVE = 30      # background worker leaves this many for user-triggered tracks
GP_MISSING_TTL_S = 24 * HOUR     # "no GP data" answers are remembered for a day
BULK_MIN_SETS = 100              # sanity check on the bulk GROUP=active file
GP_RETRY_S = 600                 # transient per-object failure: retry after 10 min
GP_BREAKER_FAILS = 3             # consecutive network failures before pausing all per-object queries
GP_BREAKER_PAUSE_S = 900         # ... for 15 min (2 h after an HTTP 403 / 429)


class LiveSourceError(RuntimeError):
    """A live source could not be fetched and no cached copy exists."""


class HTTPStatusError(LiveSourceError):
    def __init__(self, status: int, url: str, body: str = ""):
        self.status = status
        snippet = re.sub(r"\s+", " ", body or "").strip()[:160]
        super().__init__(f"HTTP {status} from {url}" + (f": {snippet}" if snippet else ""))


class NoDataError(ValueError):
    """The provider answered, but holds no data for the request (e.g. 'No GP data found')."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt, ms: bool = False) -> Optional[str]:
    if dt is None or (isinstance(dt, float) and math.isnan(dt)):
        return None
    if isinstance(dt, pd.Timestamp):
        if pd.isna(dt):
            return None
        dt = dt.to_pydatetime()
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if ms:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_ns(ts: pd.Series) -> np.ndarray:
    """tz-aware datetime Series -> int64 ns since epoch, whatever the pandas resolution."""
    return pd.to_datetime(ts, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def parse_time(value) -> datetime:
    """Parse an ISO-8601 / SOCRATES-style timestamp into an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    ts = pd.Timestamp(str(value).strip())
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


# ---------------------------------------------------------------------------------------------
# Disk-cached HTTP fetch
# ---------------------------------------------------------------------------------------------

@dataclass
class FetchResult:
    key: str
    url: str
    content: bytes
    fetched_at: datetime                  # when this payload was last confirmed with the provider
    last_modified: Optional[datetime]     # provider's Last-Modified header (data_as_of)
    stale: bool = False                   # True = refresh failed, serving an older cached copy
    from_cache: bool = False              # True = no network request was made for this call
    error: Optional[str] = None           # last error when stale
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    @property
    def version(self) -> tuple:
        """Identifies the payload; changes whenever new content is downloaded."""
        return (self.key, self.meta.get("sha_len"), self.meta.get("downloaded_at"))

    def status(self) -> dict:
        return {"status": "stale" if self.stale else "ok", "url": self.url,
                "fetched_at": iso(self.fetched_at), "data_as_of": iso(self.last_modified),
                "stale": self.stale, "error": self.error}


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def _paths(key: str) -> tuple[Path, Path]:
    k = _safe_key(key)
    return CACHE_DIR / f"{k}.data", CACHE_DIR / f"{k}.meta.json"


def _read_meta(key: str) -> Optional[dict]:
    _, mp = _paths(key)
    try:
        return json.loads(mp.read_text())
    except (OSError, ValueError):
        return None


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}_{threading.get_ident()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_meta(key: str, meta: dict) -> None:
    _write_atomic(_paths(key)[1], json.dumps(meta, indent=1).encode())


def _parse_http_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


class OfflineError(ConnectionError):
    """Network access to this host is disabled by CARA_OFFLINE (no request was made)."""


def offline_hosts() -> Optional[set]:
    """CARA_OFFLINE=1 (or 'all') blocks every live source; CARA_OFFLINE=celestrak.org,example.com
    blocks just those hosts (and their subdomains). The server then serves the disk cache, marked
    stale once a refresh was due. Used for demos without network, to keep a script off a provider
    that is throttling us, and to simulate an outage. None = no restriction."""
    v = os.environ.get("CARA_OFFLINE", "").strip().lower()
    if not v or v in ("0", "false", "no"):
        return None
    if v in ("1", "true", "yes", "all", "*"):
        return {"*"}
    return {h.strip() for h in v.split(",") if h.strip()}


def _check_offline(url: str) -> None:
    hosts = offline_hosts()
    if not hosts:
        return
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if "*" in hosts or any(host == h or host.endswith("." + h) for h in hosts):
        raise OfflineError(f"network access to {host} disabled by CARA_OFFLINE")


def _http_get(url: str, headers: dict, timeout=HTTP_TIMEOUT):
    """The single network entry point (monkeypatched in the offline tests).
    Returns (status_code, content_bytes, response_headers_dict)."""
    _check_offline(url)
    import requests
    r = requests.get(url, headers=headers, timeout=timeout)
    return r.status_code, r.content, dict(r.headers)


def _backoff_for(exc: Exception, n_fail: int = 1) -> int:
    """Exponential back-off: 5, 10, 20 ... min (2, 4, 6 h after 403/429), capped at 6 h."""
    base = REFUSED_BACKOFF_S if isinstance(exc, HTTPStatusError) and exc.status in (403, 429) else FAILURE_BACKOFF_S
    return int(min(base * 2 ** max(n_fail - 1, 0), max(MAX_BACKOFF_S, base)))


def fetch(key: str, url: str, ttl_s: float, force: bool = False,
          getter: Optional[Callable[[str, dict], tuple]] = None,
          validate: Optional[Callable[[bytes], None]] = None) -> FetchResult:
    """Return the payload of `url`, using the disk cache under CACHE_DIR/<key>.

    * cache younger than ttl_s            -> served from disk, no request
    * cache expired (or force)            -> conditional GET (If-Modified-Since); 304 refreshes TTL
    * network/provider failure            -> last cached copy with stale=True (error recorded)
    * failure and nothing cached          -> LiveSourceError
    A failed URL is not retried for FAILURE_BACKOFF_S (REFUSED_BACKOFF_S after 403/429), doubling
    with every consecutive failure up to MAX_BACKOFF_S, and a forced refresh is rate limited to
    FORCE_MIN_INTERVAL_S, so we never hammer a provider that is down or refusing us.
    """
    data_path, _ = _paths(key)
    with _lock_for(key):
        now = utcnow()
        meta = _read_meta(key) if data_path.exists() else None
        have_cache = meta is not None and meta.get("url") == url
        if have_cache:
            fetched_at = parse_time(meta["fetched_at"])
            age = (now - fetched_at).total_seconds()
            last_attempt = parse_time(meta["last_attempt"]) if meta.get("last_attempt") else fetched_at
            since_attempt = (now - last_attempt).total_seconds()
            in_backoff = bool(meta.get("last_error")) and since_attempt < meta.get("backoff_s", FAILURE_BACKOFF_S)
            fresh = age < ttl_s and not force
            forced_too_soon = force and since_attempt < FORCE_MIN_INTERVAL_S
            if fresh or in_backoff or forced_too_soon:
                return _result_from_cache(key, url, meta, from_cache=True)
        else:
            # remember failed attempts even without a cache, to back off
            fail_meta = _read_meta(key + ".fail")
            if fail_meta and fail_meta.get("url") == url:
                attempt_ref = parse_time(fail_meta["last_attempt"])
                wait = fail_meta.get("backoff_s", FAILURE_BACKOFF_S)
                if (now - attempt_ref).total_seconds() < wait and not force:
                    raise LiveSourceError(
                        f"{key}: source unavailable ({fail_meta.get('last_error')}); no cached copy yet; "
                        f"retrying after a {wait // 60} min back-off")

        try:
            _check_offline(url)
        except OfflineError as exc:
            # No request is made and nothing is recorded (no back-off / breaker state), so turning
            # CARA_OFFLINE off later refreshes immediately. A due refresh that cannot happen = stale.
            if have_cache:
                res = _result_from_cache(key, url, meta, from_cache=True)
                res.stale, res.error = True, f"OfflineError: {exc}"
                return res
            raise LiveSourceError(f"{key}: {exc} and no cached copy exists") from exc

        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        if have_cache and meta.get("last_modified_raw"):
            headers["If-Modified-Since"] = meta["last_modified_raw"]
        try:
            status, content, resp_headers = (getter or _http_get)(url, headers)
            if status == 304 and have_cache:
                meta.update(fetched_at=iso(now), last_attempt=iso(now), last_error=None, n_fail=0)
                _write_meta(key, meta)
                return _result_from_cache(key, url, meta, from_cache=False)
            if status != 200:
                raise HTTPStatusError(status, url, (content or b"")[:400].decode("utf-8", "replace"))
            if not content:
                raise LiveSourceError(f"empty response from {url}")
            if validate is not None:
                validate(content)
        except Exception as exc:  # network error, HTTP error, invalid payload
            err = f"{type(exc).__name__}: {exc}"[:300]
            prev = meta if have_cache else (_read_meta(key + ".fail") or {})
            n_fail = int(prev.get("n_fail") or 0) + 1 if prev.get("url", url) == url else 1
            backoff = _backoff_for(exc, n_fail)
            log.warning("fetch %s failed (%d in a row, next try in %d min): %s", key, n_fail, backoff // 60, err)
            if have_cache:
                meta.update(last_attempt=iso(now), last_error=err, backoff_s=backoff, n_fail=n_fail)
                _write_meta(key, meta)
                res = _result_from_cache(key, url, meta, from_cache=True)
                res.stale, res.error = True, err
                return res
            _write_meta(key + ".fail", {"url": url, "last_attempt": iso(now), "last_error": err,
                                        "backoff_s": backoff, "n_fail": n_fail})
            raise LiveSourceError(f"{key}: could not fetch {url} and no cached copy exists ({err})") from exc

        lm_raw = {k.lower(): v for k, v in resp_headers.items()}.get("last-modified")
        new_meta = {"url": url, "fetched_at": iso(now), "downloaded_at": iso(now),
                    "last_attempt": iso(now), "last_error": None,
                    "last_modified_raw": lm_raw, "last_modified": iso(_parse_http_date(lm_raw)),
                    "sha_len": len(content)}
        _write_atomic(data_path, content)
        _write_meta(key, new_meta)
        _paths(key + ".fail")[1].unlink(missing_ok=True)
        log.info("fetched %s (%d bytes)", key, len(content))
        return FetchResult(key, url, content, now, _parse_http_date(lm_raw), stale=False,
                           from_cache=False, meta=new_meta)


def _result_from_cache(key, url, meta, from_cache) -> FetchResult:
    data_path, _ = _paths(key)
    content = data_path.read_bytes()
    lm = parse_time(meta["last_modified"]) if meta.get("last_modified") else None
    # A cached copy is stale when the last attempt to refresh it failed.
    stale = bool(meta.get("last_error"))
    return FetchResult(key, url, content, parse_time(meta["fetched_at"]), lm, stale=stale,
                       from_cache=from_cache, error=meta.get("last_error"), meta=meta)


def fetch_cached(key: str, url: str) -> Optional[FetchResult]:
    """The cached payload of `url` regardless of age, without any network access (or None)."""
    data_path, _ = _paths(key)
    if not data_path.exists():
        return None
    meta = _read_meta(key)
    if not meta or meta.get("url") != url:
        return None
    try:
        return _result_from_cache(key, url, meta, from_cache=True)
    except OSError:
        return None


def backoff_remaining(key: str, url: str) -> float:
    """Seconds until fetch() would make a request for `url` after a failure (0 = no back-off)."""
    has_data = _paths(key)[0].exists()
    meta = _read_meta(key) if has_data else None
    if not (meta and meta.get("url") == url):
        meta = _read_meta(key + ".fail")
        if not (meta and meta.get("url") == url):
            return 0.0
    if not meta.get("last_error") or not meta.get("last_attempt"):
        return 0.0
    since = (utcnow() - parse_time(meta["last_attempt"])).total_seconds()
    return max(float(meta.get("backoff_s", FAILURE_BACKOFF_S)) - since, 0.0)


def cache_is_fresh(key: str, url: str, ttl_s: float) -> bool:
    meta = _read_meta(key)
    if not meta or meta.get("url") != url or not _paths(key)[0].exists():
        return False
    return (utcnow() - parse_time(meta["fetched_at"])).total_seconds() < ttl_s


STALE_GRACE_S = 900              # a cache this much past its TTL is reported stale (refresh overdue)


def cache_status(key: str, ttl_s: Optional[float] = None) -> dict:
    """Status of a cached source without touching the network (used by /api/health). With ttl_s, a
    copy whose refresh is overdue by more than STALE_GRACE_S is also reported stale (e.g. offline)."""
    meta = _read_meta(key)
    if meta is None:
        fail = _read_meta(key + ".fail")
        if fail:
            return {"status": "error", "fetched_at": None, "data_as_of": None, "stale": None,
                    "error": fail.get("last_error"), "last_attempt": fail.get("last_attempt")}
        return {"status": "not fetched", "fetched_at": None, "data_as_of": None, "stale": None,
                "error": None}
    stale, err = bool(meta.get("last_error")), meta.get("last_error")
    if not stale and ttl_s is not None and meta.get("fetched_at"):
        age = (utcnow() - parse_time(meta["fetched_at"])).total_seconds()
        if age > ttl_s + STALE_GRACE_S:
            stale, err = True, f"refresh overdue: cached copy is {age / 3600:.1f} h old (refreshed every {ttl_s / 3600:g} h)"
    return {"status": "stale" if stale else "ok", "url": meta.get("url"),
            "fetched_at": meta.get("fetched_at"), "data_as_of": meta.get("last_modified"),
            "stale": stale, "error": err}


# ---------------------------------------------------------------------------------------------
# SATCAT: object types
# ---------------------------------------------------------------------------------------------

_memo: dict[str, tuple] = {}
_memo_lock = threading.Lock()
PARSE_ERRORS: dict[str, str] = {}   # source -> last parse error of a cached payload (shown by /api/health)
SOCRATES_MIN_ROWS = 1000            # a real SOCRATES file has ~200k rows; fewer = truncated / changed format
SATCAT_MIN_ROWS = 1000              # SATCAT has ~65k rows
MAX_BAD_LINE_FRAC = 0.01            # more malformed lines than this = the format changed: reject the payload


def _validate_csv_header(*required):
    def check(content: bytes):
        head = content[:4096].decode("utf-8", errors="replace").splitlines()[0] if content else ""
        missing = [c for c in required if c not in head]
        if missing:
            raise ValueError(f"unexpected CSV header (missing {missing}): {head[:120]!r}")
    return check


def read_csv_tolerant(content: bytes, **kw) -> tuple[pd.DataFrame, int]:
    """pd.read_csv that drops lines whose field count differs from the header's (e.g. an unquoted
    comma in an object name) instead of failing the whole file. pandas alone either raises, or, when
    the bad line is the first data row, infers an index column / truncates and shifts every field.
    Returns (frame, n_dropped_lines)."""
    lines = content.splitlines()
    if not lines:
        raise ValueError("empty CSV")
    n_fields = len(next(csv.reader([lines[0].decode("utf-8", "replace")])))
    keep, bad = [lines[0]], 0
    for ln in lines[1:]:
        if not ln.strip():
            continue
        if ln.count(b",") + 1 == n_fields and b'"' not in ln:
            keep.append(ln)
        elif b'"' in ln and len(next(csv.reader([ln.decode("utf-8", "replace")]))) == n_fields:
            keep.append(ln)
        else:
            bad += 1
    return pd.read_csv(io.BytesIO(b"\n".join(keep) + b"\n"), index_col=False, **kw), bad


def _validate_table(required: tuple, min_rows_attr: str, **kw):
    """fetch() validator: the payload must fully parse, have enough rows and few malformed lines,
    so a bad download never replaces the last good cached copy."""
    header = _validate_csv_header(*required)

    def check(content: bytes):
        header(content)
        df, skipped = read_csv_tolerant(content, **kw)
        min_rows = globals()[min_rows_attr]
        if len(df) < min_rows:
            raise ValueError(f"only {len(df)} rows parsed (expected >= {min_rows})")
        if skipped > MAX_BAD_LINE_FRAC * (len(df) + skipped):
            raise ValueError(f"{skipped} malformed lines of {len(df) + skipped}")
    return check


def load_satcat(force: bool = False) -> tuple[dict, FetchResult]:
    """NORAD id -> dataset object type (PAYLOAD / ROCKET BODY / DEBRIS / UNKNOWN)."""
    res = fetch("satcat", SATCAT_URL, TTL["satcat"], force=force,
                validate=_validate_table(("NORAD_CAT_ID", "OBJECT_TYPE"), "SATCAT_MIN_ROWS",
                                         usecols=["NORAD_CAT_ID", "OBJECT_TYPE"]))
    return load_satcat_from(res)


def load_satcat_from(res: FetchResult) -> tuple[dict, FetchResult]:
    """Parse a SATCAT payload (memoised by payload version). Raises LiveSourceError if unparseable."""
    with _memo_lock:
        hit = _memo.get("satcat")
        if hit and hit[0] == res.version:
            return hit[1], res
    try:
        df, skipped = read_csv_tolerant(res.content, usecols=["NORAD_CAT_ID", "OBJECT_TYPE"],
                                        dtype={"OBJECT_TYPE": "string"})
    except Exception as exc:  # noqa: BLE001 - corrupt cached payload
        PARSE_ERRORS["satcat"] = f"{type(exc).__name__}: {exc}"[:300]
        raise LiveSourceError(f"satcat: cached payload unreadable ({PARSE_ERRORS['satcat']})") from exc
    PARSE_ERRORS.pop("satcat", None)
    if skipped:
        log.warning("SATCAT: skipped %d malformed lines", skipped)
    df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"], errors="coerce")
    df = df.dropna(subset=["NORAD_CAT_ID"])
    mapping = dict(zip(df["NORAD_CAT_ID"].astype("int64").to_numpy(),
                       df["OBJECT_TYPE"].str.strip().map(SATCAT_TYPE_MAP).fillna("UNKNOWN").to_numpy()))
    with _memo_lock:
        _memo["satcat"] = (res.version, mapping)
    return mapping, res


_DEB_RX = re.compile(r"(^|[\s/(-])DEB($|[\s)\]-])")


def name_type(name: str, status: str) -> str:
    """Object type from a CelesTrak name + status marker (fallback when SATCAT has no entry)."""
    u = (name or "").upper()
    if _DEB_RX.search(u):
        return "DEBRIS"
    if "R/B" in u:
        return "ROCKET BODY"
    if (status or "") in ACTIVE_MARKERS:
        return "PAYLOAD"
    return "UNKNOWN"


def type_from_name(name: pd.Series, status: pd.Series) -> pd.Series:
    """Vectorised name_type(); string work is done once per unique (name, status) pair."""
    pairs = pd.MultiIndex.from_arrays([name.fillna("").astype(str), status.fillna("").astype(str)])
    codes, uniq = pd.factorize(pairs)
    vals = np.array([name_type(n, st) for n, st in uniq], dtype=object)
    return pd.Series(vals[codes], index=name.index)


def object_types(norad: pd.Series, name: pd.Series, status: pd.Series,
                 satcat: Optional[dict]) -> tuple[pd.Series, pd.Series]:
    """Returns (type, type_source) per row: SATCAT where known, else name/status fallback."""
    fallback = type_from_name(name, status)
    if satcat:
        known = norad.map(satcat)
        typ = known.fillna(fallback)
        src = np.where(known.notna(), "SATCAT", "name")
    else:
        typ, src = fallback, np.full(len(norad), "name")
    return typ.astype(str), pd.Series(src, index=norad.index)


# ---------------------------------------------------------------------------------------------
# Space weather (NOAA SWPC, CelesTrak fallback / history)
# ---------------------------------------------------------------------------------------------

def _records(payload) -> list[dict]:
    """SWPC products come either as a list of dicts or as a list of lists with a header row."""
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        header = [str(h) for h in payload[0]]
        return [dict(zip(header, row)) for row in payload[1:]]
    return list(payload) if isinstance(payload, list) else []


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _series(points, kind: str) -> list[dict]:
    """Series items carry generic keys (time, value) plus the SWPC product's own key names
    (time_tag/flux, date/ap, time_tag/Kp) so either convention can be read."""
    out = []
    for t, v in points:
        if v is None:
            continue
        item = {"time": iso(t), "value": v}
        if kind == "f107":
            item.update(time_tag=iso(t), flux=v)
        elif kind == "ap":
            item.update(date=t.strftime("%Y-%m-%d"), ap=v)
        elif kind == "kp":
            item.update(time_tag=iso(t), Kp=v)
        out.append(item)
    return out


def parse_swpc_f107(content: bytes) -> list[tuple[datetime, float]]:
    out = []
    for r in _records(json.loads(content)):
        v = _num(r.get("flux", r.get("f10.7")))
        if r.get("time_tag") and v is not None:
            out.append((parse_time(r["time_tag"]), v))
    return sorted(out)


def parse_swpc_kp(content: bytes) -> list[tuple[datetime, Optional[float], Optional[float]]]:
    """-> [(time, Kp, ap)] 3-hourly."""
    out = []
    for r in _records(json.loads(content)):
        if not r.get("time_tag"):
            continue
        out.append((parse_time(r["time_tag"]), _num(r.get("Kp", r.get("kp_index"))),
                    _num(r.get("a_running", r.get("ap")))))
    return sorted(out, key=lambda x: x[0])


def parse_swpc_afred(content: bytes) -> list[dict]:
    """Predicted Fredericksburg A index (1-3 day outlook) -> [{time, value}]."""
    recs = _records(json.loads(content))
    if not recs:
        return []
    r = sorted(recs, key=lambda x: str(x.get("date", "")))[-1]
    try:
        base = parse_time(r["date"])
    except (KeyError, ValueError):
        return []
    out = []
    for i, k in enumerate(["afred_1_day", "afred_2_day", "afred_3_day"], start=1):
        v = _num(r.get(k))
        if v is not None:
            out.append({"time": iso(base + timedelta(days=i)), "value": v})
    return out


def daily_ap(kp_rows) -> list[tuple[datetime, float]]:
    """Daily Ap = mean of the eight 3-hourly ap values of each UTC day (partial days allowed)."""
    if not kp_rows:
        return []
    df = pd.DataFrame([(t, ap) for t, _, ap in kp_rows if ap is not None], columns=["t", "ap"])
    if df.empty:
        return []
    df["day"] = pd.to_datetime(df["t"], utc=True).dt.floor("D")
    g = df.groupby("day")["ap"].mean()
    return [(d.to_pydatetime(), round(float(v), 2)) for d, v in g.items()]


def parse_celestrak_sw(content: bytes, days: int = 30):
    """CelesTrak SW-Last5Years.csv -> (f107 series, daily Ap series, 3-hourly Kp series)."""
    df = pd.read_csv(io.BytesIO(content))
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True, errors="coerce")
    df = df.dropna(subset=["DATE"])
    obs = df[df["F10.7_OBS"].notna() & df["AP_AVG"].notna()]
    obs = obs[obs["DATE"] <= pd.Timestamp(utcnow())].tail(days)
    f107 = [(d.to_pydatetime(), _num(v)) for d, v in zip(obs["DATE"], obs["F10.7_OBS"])]
    ap = [(d.to_pydatetime(), _num(v)) for d, v in zip(obs["DATE"], obs["AP_AVG"])]
    kp = []
    for _, row in obs.iterrows():
        for i in range(8):
            v = _num(row.get(f"KP{i + 1}"))
            if v is not None:
                kp.append((row["DATE"].to_pydatetime() + timedelta(hours=3 * i), v / 10.0))
    return f107, ap, kp


def _require(parser, msg):
    """Validator for fetch(): the payload must parse to a non-empty result."""
    def check(content: bytes):
        if not parser(content):
            raise ValueError(msg)
    return check


def space_weather(force: bool = False, history: bool = True, history_network: bool = True) -> dict:
    """Latest F10.7 and daily Ap (model inputs) plus ~30-day series for display.

    Latest values come from NOAA SWPC. SWPC's K-index product only spans ~7 days, so with
    history=True the Ap/Kp series are extended back to 30 days from CelesTrak SW-Last5Years.csv
    (cached 6 h); CelesTrak also replaces any SWPC product that is unavailable.
    history_network=False uses only a cached CelesTrak copy for the history extension (the server's
    background loop refreshes it), so a slow CelesTrak never delays the response."""
    out = {"fetched_at": None, "stale": False, "sources": {}, "errors": {},
           "f107": {"latest": None, "time": None, "series": [], "source": None},
           "ap": {"latest": None, "time": None, "series": [], "forecast": [], "source": None},
           "kp": {"latest": None, "time": None, "series": [], "source": None}}
    fetched = []

    def note(name, res):
        out["sources"][name] = res.status()
        fetched.append(res)

    try:
        res = fetch("swpc_f107", SWPC_URLS["f107"], TTL["swpc"], force=force,
                    validate=_require(parse_swpc_f107, "no F10.7 values"))
        f = parse_swpc_f107(res.content)
        out["f107"].update(latest=f[-1][1], time=iso(f[-1][0]), series=_series(f, "f107"),
                           source="NOAA SWPC 10cm-flux-30-day")
        note("swpc_f107", res)
    except Exception as exc:  # noqa: BLE001
        out["errors"]["swpc_f107"] = str(exc)[:300]
    kp_rows = []
    try:
        res = fetch("swpc_kp", SWPC_URLS["kp"], TTL["swpc"], force=force,
                    validate=_require(parse_swpc_kp, "no Kp values"))
        kp_rows = parse_swpc_kp(res.content)
        kp_pts = [(t, kp) for t, kp, _ in kp_rows if kp is not None]
        ap_pts = [(t, ap) for t, _, ap in kp_rows if ap is not None]
        if ap_pts:
            last8 = [v for _, v in ap_pts[-8:]]
            out["ap"].update(latest=round(float(np.mean(last8)), 2), time=iso(ap_pts[-1][0]),
                             series=_series(daily_ap(kp_rows), "ap"),
                             source="NOAA SWPC planetary K-index (daily Ap = mean of the last 8 3-hourly ap)")
        if kp_pts:
            out["kp"].update(latest=kp_pts[-1][1], time=iso(kp_pts[-1][0]), series=_series(kp_pts, "kp"),
                             source="NOAA SWPC planetary K-index")
        note("swpc_kp", res)
    except Exception as exc:  # noqa: BLE001
        out["errors"]["swpc_kp"] = str(exc)[:300]
    try:
        res = fetch("swpc_afred", SWPC_URLS["afred"], TTL["swpc"], force=force)
        out["ap"]["forecast"] = parse_swpc_afred(res.content)
        note("swpc_afred", res)
    except Exception as exc:  # noqa: BLE001
        out["errors"]["swpc_afred"] = str(exc)[:300]

    need_fallback = (out["f107"]["latest"] is None or out["ap"]["latest"] is None
                     or out["kp"]["latest"] is None or any(r.stale for r in fetched))
    if need_fallback or history:
        try:
            if need_fallback or history_network:
                res = fetch("celestrak_sw", CELESTRAK_SW_URL, TTL["celestrak_sw"], force=force,
                            validate=_validate_csv_header("DATE", "F10.7_OBS", "AP_AVG"))
            else:
                res = fetch_cached("celestrak_sw", CELESTRAK_SW_URL)
                if res is None:
                    raise LiveSourceError("CelesTrak SW history not cached yet (refreshed in the background)")
            f107, ap, kp = parse_celestrak_sw(res.content)
            src = "CelesTrak SW-Last5Years.csv"
            for name, ser in (("f107", f107), ("ap", ap), ("kp", kp)):
                if not ser:
                    continue
                cur = out[name]
                if cur["latest"] is None:            # SWPC product unavailable -> fallback
                    cur.update(latest=ser[-1][1], time=iso(ser[-1][0]), series=_series(ser, name),
                               source=src + " (fallback)")
                elif name in ("ap", "kp") and cur["series"]:   # extend SWPC's 7 days to 30
                    first = parse_time(cur["series"][0]["time"])
                    older = [(t, v) for t, v in ser if t < first]
                    if older:
                        cur["series"] = _series(older, name) + cur["series"]
                        cur["source"] += f"; history before {iso(first)} from {src}"
            note("celestrak_sw", res)
        except Exception as exc:  # noqa: BLE001
            out["errors"]["celestrak_sw"] = str(exc)[:300]

    if fetched:
        out["fetched_at"] = iso(min(r.fetched_at for r in fetched))
        out["stale"] = any(r.stale for r in fetched)
    return out


# ---------------------------------------------------------------------------------------------
# SOCRATES Plus conjunctions
# ---------------------------------------------------------------------------------------------

SOCRATES_COLUMNS = ["NORAD_CAT_ID_1", "OBJECT_NAME_1", "DSE_1", "NORAD_CAT_ID_2", "OBJECT_NAME_2",
                    "DSE_2", "TCA", "TCA_RANGE", "TCA_RELATIVE_SPEED", "MAX_PROB", "DILUTION"]
_STATUS_RX = re.compile(r"\s*\[([^\]]*)\]\s*$")


def split_status(raw: str) -> tuple[str, str]:
    """'FENGYUN 1C DEB [-]' -> ('FENGYUN 1C DEB', '-')"""
    raw = (raw or "").strip()
    m = _STATUS_RX.search(raw)
    if m:
        return raw[:m.start()].strip(), m.group(1).strip()
    return raw, ""


def _split_status(names: pd.Series) -> tuple[pd.Series, pd.Series]:
    # ~200k rows but only a few tens of thousands of distinct names: work on the uniques
    codes, uniq = pd.factorize(names.fillna("").astype(str))
    parts = [split_status(u) for u in uniq]
    clean = np.array([p[0] for p in parts], dtype=object)
    status = np.array([p[1] for p in parts], dtype=object)
    return pd.Series(clean[codes], index=names.index), pd.Series(status[codes], index=names.index)


def parse_socrates(files: dict[str, bytes]) -> pd.DataFrame:
    """Parse + union + dedupe the SOCRATES CSVs (time-independent part, memoised by caller).

    Output columns: norad_1, name_1, status_1, dse_1, norad_2, name_2, status_2, dse_2, tca (UTC),
    range_km, speed_kms, max_prob, dilution_km, in_files.
    """
    frames = []
    labels = list(files)
    skipped_rows = {}
    for bit, (label, content) in enumerate(files.items()):
        df, skipped_rows[label] = read_csv_tolerant(
            content, dtype={"OBJECT_NAME_1": "string", "OBJECT_NAME_2": "string", "TCA": "string"})
        if skipped_rows[label]:
            log.warning("SOCRATES %s: skipped %d malformed lines", label, skipped_rows[label])
        df.columns = [c.strip() for c in df.columns]
        missing = [c for c in SOCRATES_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"SOCRATES {label}: missing columns {missing}")
        df = df[SOCRATES_COLUMNS].copy()
        df["_bit"] = 1 << bit
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in ["NORAD_CAT_ID_1", "NORAD_CAT_ID_2", "DSE_1", "DSE_2", "TCA_RANGE", "TCA_RELATIVE_SPEED",
              "MAX_PROB", "DILUTION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    tca_str = df["TCA"].astype("string").str.strip()
    df["tca"] = pd.to_datetime(tca_str, utc=True, errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")
    bad = df["tca"].isna() & tca_str.notna()
    if bad.any():  # tolerate other ISO variants (e.g. without fractional seconds)
        df.loc[bad, "tca"] = pd.to_datetime(tca_str[bad], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["NORAD_CAT_ID_1", "NORAD_CAT_ID_2", "tca", "TCA_RANGE"])
    df["NORAD_CAT_ID_1"] = df["NORAD_CAT_ID_1"].astype("int64")
    df["NORAD_CAT_ID_2"] = df["NORAD_CAT_ID_2"].astype("int64")

    # Union of both files: one row per (unordered pair, TCA to the second); remember the files.
    df["_lo"] = np.minimum(df["NORAD_CAT_ID_1"], df["NORAD_CAT_ID_2"])
    df["_hi"] = np.maximum(df["NORAD_CAT_ID_1"], df["NORAD_CAT_ID_2"])
    df["_ts"] = to_ns(df["tca"].dt.floor("s")) // 10 ** 9      # unit-safe (pandas 3 parses to us)
    keys = ["_lo", "_hi", "_ts"]
    df = df.drop_duplicates(keys + ["_bit"]).copy()
    df["_mask"] = df.groupby(keys, sort=False)["_bit"].transform("sum")
    df = df.drop_duplicates(keys, keep="first")
    names = {m: "+".join(f"sort-{labels[b]}.csv" for b in range(len(labels)) if m & (1 << b))
             for m in df["_mask"].unique()}
    df = df.assign(in_files=df["_mask"].map(names))

    n1, s1 = _split_status(df["OBJECT_NAME_1"])
    n2, s2 = _split_status(df["OBJECT_NAME_2"])
    out = pd.DataFrame({
        "norad_1": df["NORAD_CAT_ID_1"].to_numpy(), "name_1": n1.to_numpy(), "status_1": s1.to_numpy(),
        "dse_1": df["DSE_1"].to_numpy(dtype=float),
        "norad_2": df["NORAD_CAT_ID_2"].to_numpy(), "name_2": n2.to_numpy(), "status_2": s2.to_numpy(),
        "dse_2": df["DSE_2"].to_numpy(dtype=float),
        "tca": df["tca"].to_numpy(), "range_km": df["TCA_RANGE"].to_numpy(dtype=float),
        "speed_kms": df["TCA_RELATIVE_SPEED"].to_numpy(dtype=float),
        "max_prob": df["MAX_PROB"].to_numpy(dtype=float), "dilution_km": df["DILUTION"].to_numpy(dtype=float),
        "in_files": df["in_files"].to_numpy(),
    })
    out["tca"] = pd.to_datetime(out["tca"], utc=True)
    out = out.sort_values("tca", kind="mergesort").reset_index(drop=True)
    out.attrs["skipped_rows"] = skipped_rows
    return out


def log10_prob(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log10(np.where(p > 0, p, np.nan))
    return np.clip(np.nan_to_num(lp, nan=LOG_PC_FLOOR), LOG_PC_FLOOR, 0.0)


@dataclass
class SocratesSnapshot:
    rows: pd.DataFrame          # parsed + typed conjunctions (time independent)
    fetched_at: datetime
    data_as_of: Optional[datetime]
    stale: bool
    files: dict                 # label -> FetchResult.status()
    version: tuple
    satcat_status: dict
    parse_error: Optional[str] = None   # the newest payload could not be parsed; rows are the previous parse

    @property
    def provenance(self) -> str:
        names = " + ".join(f"sort-{k}.csv" for k, v in self.files.items() if v.get("status") != "error")
        lm = iso(self.data_as_of) or "unknown"
        return f"CelesTrak SOCRATES Plus {names}, Last-Modified {lm}"


def socrates_snapshot(force: bool = False, network: bool = True) -> SocratesSnapshot:
    """Download (cached) both SOCRATES files + SATCAT, parse once per cache refresh (memoised).
    network=False uses whatever is on disk (any age) and never makes a request."""
    results, errors = {}, {}
    for label, url in SOCRATES_URLS.items():
        key = f"socrates_{label}"
        try:
            if network:
                results[label] = fetch(key, url, TTL["socrates"], force=force,
                                       validate=_validate_table(("NORAD_CAT_ID_1", "TCA_RANGE", "MAX_PROB"),
                                                                "SOCRATES_MIN_ROWS"))
            else:
                r = fetch_cached(key, url)
                if r is None:
                    raise LiveSourceError(f"{key}: nothing cached yet")
                results[label] = r
        except LiveSourceError as exc:
            errors[label] = str(exc)
    if not results:
        raise LiveSourceError("SOCRATES unavailable: " + "; ".join(errors.values()))
    try:
        if network:
            satcat, satcat_res = load_satcat()
        else:
            r = fetch_cached("satcat", SATCAT_URL)
            if r is None:
                raise LiveSourceError("satcat: nothing cached yet")
            satcat, satcat_res = load_satcat_from(r)
        satcat_status = satcat_res.status()
        satcat_version = satcat_res.version
    except LiveSourceError as exc:
        satcat, satcat_status, satcat_version = None, {"status": "error", "error": str(exc)}, None

    version = tuple(r.version for r in results.values()) + (satcat_version,)
    with _memo_lock:
        hit = _memo.get("socrates")
        failed = _memo.get("socrates_failed")
    lms = [r.last_modified for r in results.values() if r.last_modified]
    fetched_at, data_as_of = min(r.fetched_at for r in results.values()), (min(lms) if lms else None)
    parse_error = None
    if hit and hit[0] == version:
        rows, skipped = hit[1], hit[2]
    elif failed and failed[0] == version:          # this payload already failed: do not re-parse it
        parse_error = failed[1]
    else:
        t0 = time.perf_counter()
        try:
            rows = parse_socrates({k: r.content for k, r in results.items()})
            skipped = rows.attrs.get("skipped_rows", {})
            for i in (1, 2):
                typ, src = object_types(rows[f"norad_{i}"], rows[f"name_{i}"], rows[f"status_{i}"], satcat)
                rows[f"type_{i}"] = typ.to_numpy()
                rows[f"type_src_{i}"] = src.to_numpy()
            log.info("parsed SOCRATES: %d rows in %.2fs", len(rows), time.perf_counter() - t0)
            with _memo_lock:
                _memo["socrates"] = (version, rows, skipped, fetched_at, data_as_of)
                _memo.pop("socrates_failed", None)
        except Exception as exc:  # noqa: BLE001 - corrupt payload: keep serving the previous parse
            parse_error = f"{type(exc).__name__}: {exc}"[:300]
            log.error("SOCRATES payload unparseable: %s", parse_error)
            with _memo_lock:
                _memo["socrates_failed"] = (version, parse_error)
    if parse_error is not None:
        PARSE_ERRORS["socrates"] = parse_error
        if not hit:
            raise LiveSourceError(f"SOCRATES payload unparseable and no earlier parse to fall back on: {parse_error}")
        version, rows, skipped, fetched_at, data_as_of = hit   # the previous payload, with its own timestamps
    else:
        PARSE_ERRORS.pop("socrates", None)

    files = {k: r.status() for k, r in results.items()}
    for k, n in skipped.items():
        if k in files and n:
            files[k]["skipped_rows"] = int(n)
    for k, e in errors.items():
        files[k] = {"status": "error", "error": e}
    if parse_error is not None:
        for k in results:
            files[k].update(status="stale", stale=True, fetched_at=iso(fetched_at), data_as_of=iso(data_as_of),
                            error=f"newest payload unparseable ({parse_error}); serving the previous parse")
    return SocratesSnapshot(rows=rows, fetched_at=fetched_at, data_as_of=data_as_of,
                            stale=any(r.stale for r in results.values()) or bool(errors) or parse_error is not None,
                            files=files, version=version, satcat_status=satcat_status, parse_error=parse_error)


def screening_rows(rows: pd.DataFrame, now: Optional[datetime] = None,
                   f10: Optional[float] = None, ap: Optional[float] = None) -> pd.DataFrame:
    """Map SOCRATES rows to RiskModel.assess_screening inputs (SI units, dataset vocabulary).

    Keeps only TCAs from now - PAST_TCA_ALLOWANCE_H onward. Adds `hours_to_tca`.
    The secondary object (sat2) plays the role of the dataset's `c_` (chaser) object.
    """
    now = now or utcnow()
    hours = (rows["tca"] - pd.Timestamp(now)).dt.total_seconds().to_numpy() / 3600.0
    keep = hours >= -PAST_TCA_ALLOWANCE_H
    out = rows.loc[keep].copy()
    out["hours_to_tca"] = hours[keep]
    out["time_to_tca"] = np.clip(out["hours_to_tca"].to_numpy() / 24.0, 0.0, None)   # days
    out["miss_distance"] = out["range_km"].to_numpy() * 1000.0                          # m
    out["relative_speed"] = out["speed_kms"].to_numpy() * 1000.0                        # m/s
    out["max_risk_estimate"] = log10_prob(out["max_prob"].to_numpy())                    # log10, floor -30
    out["c_object_type"] = out["type_2"].to_numpy()
    out["F10"] = np.nan if f10 is None else float(f10)
    out["AP"] = np.nan if ap is None else float(ap)
    return out.reset_index(drop=True)


def load_socrates(now: Optional[datetime] = None, force: bool = False, sw: Optional[dict] = None,
                  network: bool = True):
    """Live screening rows ready for RiskModel.assess_screening, plus snapshot metadata."""
    snap = socrates_snapshot(force=force, network=network)
    if sw is None:
        try:
            sw = space_weather(history=False)
        except Exception:  # noqa: BLE001 - space weather is optional for screening
            sw = {}
    f10 = (sw.get("f107") or {}).get("latest")
    ap = (sw.get("ap") or {}).get("latest")
    return screening_rows(snap.rows, now=now, f10=f10, ap=ap), snap


def summarize_warnings(df: pd.DataFrame, window_h: float, top_n: int = 5) -> dict:
    """The `warnings` block of /api/live/conjunctions.

    `df` holds the assessed conjunctions inside the window (columns hours_to_tca, miss_distance,
    tier, needs_review, name_1, name_2, type_1, type_2, max_risk_estimate). Rows whose TCA has
    already passed (hours_to_tca < 0, kept in the feed for PAST_TCA_ALLOWANCE_H) are only counted in
    `n_recent_past`: every other figure describes TCAs in [now, now + window_h]."""
    n_hours = max(int(math.ceil(window_h)), 1)
    past = df["hours_to_tca"].to_numpy(dtype=float) < 0 if len(df) else np.zeros(0, bool)
    n_past = int(past.sum())
    df = df.loc[~past]
    n = int(len(df))
    tier = df["tier"].to_numpy(dtype=int) if n else np.zeros(0, int)
    md = df["miss_distance"].to_numpy(dtype=float) if n else np.zeros(0)
    hours = df["hours_to_tca"].to_numpy(dtype=float) if n else np.zeros(0)
    bucket = np.clip(np.floor(hours), 0, n_hours - 1).astype(int)
    by_hour = []
    for h in range(n_hours):
        sel = bucket == h
        by_hour.append({"hour": h, "critical": int((sel & (tier == 2)).sum()),
                        "elevated": int((sel & (tier == 1)).sum()), "total": int(sel.sum())})
    top = []
    if n:   # by NORAD id: many new objects share generic names such as "OBJECT A"
        key = df["norad_1"] if "norad_1" in df.columns else df["name_1"]
        vc = key.value_counts().head(top_n)
        names = dict(zip(key, df["name_1"]))
        top = [{"name": str(names[k]), "count": int(v), **({"norad": int(k)} if "norad_1" in df.columns else {})}
               for k, v in vc.items()]
    within_1 = int((md < 1000.0).sum())
    within_5 = int((md <= 5000.0).sum())
    n_crit, n_elev = int((tier == 2).sum()), int((tier == 1).sum())
    n_review = int(df["needs_review"].astype(bool).sum()) if n else 0
    crit_by_pc = int(((tier == 2) & (md >= 1000.0)).sum())
    debris = 0
    if n:
        debris = int((df["type_1"].isin(["DEBRIS", "ROCKET BODY"]) | df["type_2"].isin(["DEBRIS", "ROCKET BODY"])).sum())

    insights = []
    if n:
        w = f"{window_h:g} h"
        insights.append(f"{n:,} screened conjunctions with TCA in the next {w}; {within_1:,} pass within 1 km "
                        f"(Critical by the distance rule).")
        if crit_by_pc:
            insights.append(f"{crit_by_pc:,} more are Critical on estimated Pc although the miss distance is >= 1 km.")
        busiest = max(by_hour, key=lambda b: b["total"])
        insights.append(f"Busiest hour: +{busiest['hour']} h with {busiest['total']:,} conjunctions "
                        f"({busiest['critical']:,} Critical).")
        insights.append(f"{debris / n:.0%} involve debris or a rocket body.")
        i = int(np.argmin(md))
        r = df.iloc[i]
        insights.append(f"Closest approach: {r['name_1']} vs {r['name_2']}, {md[i]:,.0f} m, "
                        f"{r['hours_to_tca']:.1f} h from now.")
        if n_review:
            insights.append(f"{n_review:,} need analyst review (worst-case Pc >= 1e-4 or the Pc interval "
                            "straddles a tier boundary).")
        if top:
            insights.append(f"{top[0]['name']} has the most conjunctions in the window ({top[0]['count']}).")
        insights.append("Screening-grade (TLE/SGP4, no covariance): confirm with CSpOC CDMs before any manoeuvre.")

    if n_past:
        insights.append(f"{n_past:,} more conjunctions had their TCA in the last {PAST_TCA_ALLOWANCE_H:g} h "
                        "(listed in the feed, not counted above).")

    return {"window_h": window_h, "n_in_window": n, "n_recent_past": n_past,
            "n_within_1km": within_1, "n_within_5km": within_5,
            "n_critical": n_crit, "n_elevated": n_elev, "n_needs_review": n_review,
            "n_critical_by_pc": crit_by_pc, "n_debris_or_rb": debris,
            "by_hour": by_hour, "top_primaries": top, "insights": insights}


# ---------------------------------------------------------------------------------------------
# GP element sets (TLE / OMM)
# ---------------------------------------------------------------------------------------------

_ALPHA5 = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # Alpha-5: A=10 ... Z=33, skipping I and O


def catalog_number(field5: str) -> int:
    s = field5.strip()
    if s and s[0].isalpha():
        return (_ALPHA5.index(s[0].upper()) + 10) * 10000 + int(s[1:])
    return int(s)


def jd_to_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=(jd - 2440587.5) * 86400.0)


@dataclass(eq=False)
class TLE:
    """One GP element set: either a TLE (line1/line2) or OMM fields (6-digit catalog numbers)."""
    norad: int
    name: str
    line1: Optional[str]
    line2: Optional[str]
    fetched_at: Optional[datetime]
    stale: bool = False
    source: str = ""
    omm: Optional[dict] = None
    _sat: object = field(default=None, repr=False)

    @property
    def satrec(self):
        if self._sat is None:
            from sgp4.api import Satrec
            if self.line1:
                self._sat = Satrec.twoline2rv(self.line1, self.line2)
            else:
                from sgp4 import omm
                s = Satrec()
                omm.initialize(s, self.omm)
                self._sat = s
        return self._sat

    @property
    def epoch(self) -> datetime:
        s = self.satrec
        return jd_to_datetime(s.jdsatepoch + s.jdsatepochF)

    @property
    def epoch_jd(self) -> float:
        s = self.satrec
        return s.jdsatepoch + s.jdsatepochF


def parse_tle(text: str, norad: Optional[int] = None) -> tuple[str, str, str]:
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    l1 = next((ln for ln in lines if ln.startswith("1 ")), None)
    l2 = next((ln for ln in lines if ln.startswith("2 ")), None)
    if not l1 or not l2:
        raise NoDataError(f"no TLE in response: {text.strip()[:80]!r}")
    i1 = lines.index(l1)
    name = lines[i1 - 1].strip() if i1 > 0 and not lines[i1 - 1].startswith(("1 ", "2 ")) else ""
    if name.startswith("0 "):
        name = name[2:].strip()
    if norad is not None and catalog_number(l1[2:7]) != int(norad):
        raise ValueError(f"TLE catalog number {l1[2:7].strip()} does not match requested {norad}")
    return name, l1, l2


def parse_tle_catalog(text: str) -> dict[int, tuple[str, str, str]]:
    """Bulk 3-line TLE file -> {norad: (name, line1, line2)}."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out = {}
    for i in range(len(lines) - 1):
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith("1 ") and l2.startswith("2 ") and len(l1) >= 64 and len(l2) >= 64:
            name = lines[i - 1].strip() if i > 0 and not lines[i - 1].startswith(("1 ", "2 ")) else ""
            if name.startswith("0 "):
                name = name[2:].strip()
            try:
                out[catalog_number(l1[2:7])] = (name, l1, l2)
            except ValueError:
                continue
    return out


def parse_omm_csv(content: bytes, norad: Optional[int] = None) -> list[dict]:
    from sgp4 import omm
    text = content.decode("utf-8", "replace")
    if "NORAD_CAT_ID" not in text[:2000]:
        raise NoDataError(f"no GP data in response: {text.strip()[:80]!r}")
    recs = list(omm.parse_csv(io.StringIO(text)))
    if norad is not None:
        recs = [r for r in recs if int(float(r.get("NORAD_CAT_ID", -1))) == int(norad)]
    if not recs:
        raise NoDataError(f"no GP record for {norad}")
    return recs


def gp_url(norad: int) -> str:
    return (GP_CSV_URL if int(norad) > 99999 else GP_TLE_URL).format(norad=int(norad))


def satrec_from(rec):
    """Build the SGP4 record of an OMM dict or a (name, line1, line2) tuple; raises if malformed."""
    from sgp4.api import Satrec
    if isinstance(rec, dict):
        from sgp4 import omm
        s = Satrec()
        omm.initialize(s, rec)
        return s
    return Satrec.twoline2rv(rec[1], rec[2])


def _validate_gp(norad: int):
    def check(content: bytes):
        if int(norad) > 99999:
            satrec_from(parse_omm_csv(content, norad)[0])
        else:
            satrec_from(parse_tle(content.decode("utf-8", "replace"), norad))
    return check


def elements_from_result(norad: int, res: FetchResult, source: str) -> TLE:
    if int(norad) > 99999:
        rec = parse_omm_csv(res.content, norad)[0]
        return TLE(int(norad), str(rec.get("OBJECT_NAME", "")).strip(), None, None, res.fetched_at,
                   res.stale, source, omm=rec)
    name, l1, l2 = parse_tle(res.text, norad)
    return TLE(int(norad), name, l1, l2, res.fetched_at, res.stale, source)


class BudgetExhausted(LiveSourceError):
    pass


class ProviderPaused(BudgetExhausted):
    """Per-object queries are paused after repeated provider failures (circuit breaker)."""


class TLEStore:
    """Element sets for SGP4: one bulk GROUP=active download (6 h cache) plus throttled
    per-object CATNR queries (<= 2 req/s, <= GP_BUDGET per 2 h, 24 h disk cache) made by a
    background worker. API code calls `lookup()` / `status()` (never touch the network) and
    `request()` (queue objects); only `get(network=True)` (a user-triggered track) fetches inline."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self._lock = threading.RLock()
        self._mem: dict[int, TLE] = {}
        self._disk_checked: set[int] = set()
        self._bulk: dict[int, tuple] = {}
        self._bulk_version = None
        self._bulk_res: Optional[FetchResult] = None
        self._bulk_error: Optional[str] = None
        self._bulk_tles: dict[int, TLE] = {}
        self._bulk_checked = 0.0
        self._missing: Optional[dict] = None
        self._retry_at: dict[int, float] = {}
        self._queue: "OrderedDict[int, float]" = OrderedDict()
        self._inflight: set[int] = set()
        self._budget: Optional[deque] = None
        self._rate_lock = threading.Lock()
        self._last_req = 0.0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.n_fetched = 0
        self.n_failed = 0
        self._consecutive_fail = 0
        self._pause_until = 0.0
        self._pause_reason: Optional[str] = None
        self._trips = 0

    # -- persistence -------------------------------------------------------------------------
    def _dir(self) -> Path:
        return self.cache_dir or CACHE_DIR

    def _missing_map(self) -> dict:
        if self._missing is None:
            try:
                self._missing = json.loads((self._dir() / "gp_missing.json").read_text())
            except (OSError, ValueError):
                self._missing = {}
        return self._missing

    def _save_missing(self):
        _write_atomic(self._dir() / "gp_missing.json", json.dumps(self._missing_map()).encode())

    def _budget_q(self) -> deque:
        if self._budget is None:
            try:
                state = json.loads((self._dir() / "gp_budget.json").read_text())
            except (OSError, ValueError):
                state = {}
            self._budget = deque(sorted(float(t) for t in state.get("requests", [])))
            # the circuit breaker survives restarts, so restarting never re-hammers a provider
            self._pause_until = max(self._pause_until, float(state.get("pause_until") or 0.0))
            self._trips = max(self._trips, int(state.get("trips") or 0))
            self._pause_reason = self._pause_reason or state.get("pause_reason")
        cutoff = time.time() - GP_BUDGET_WINDOW_S
        while self._budget and self._budget[0] < cutoff:
            self._budget.popleft()
        return self._budget

    def budget_used(self) -> int:
        with self._lock:
            return len(self._budget_q())

    def _save_state(self):
        q = self._budget_q()
        _write_atomic(self._dir() / "gp_budget.json", json.dumps(
            {"requests": list(q), "pause_until": self._pause_until, "trips": self._trips,
             "pause_reason": self._pause_reason}).encode())

    def _spend(self):
        self._budget_q().append(time.time())
        self._save_state()

    # -- bulk ----------------------------------------------------------------------------------
    def load_bulk(self, network: bool = False) -> Optional[FetchResult]:
        """Parse the bulk active-objects file (memoised). network=True refreshes the OMM CSV when
        older than 6 h (background thread / startup); network=False only reads the disk cache.
        A legacy bulk TLE file still fresh in the cache is used until it expires (no request)."""
        res = None
        if network:
            csv_cached = fetch_cached("gp_active_csv", GP_ACTIVE_CSV_URL)
            legacy_fresh = cache_is_fresh("gp_active", GP_ACTIVE_URL, TTL["gp_bulk"])
            if csv_cached is not None or not legacy_fresh:
                try:
                    res = fetch("gp_active_csv", GP_ACTIVE_CSV_URL, TTL["gp_bulk"],
                                validate=_require(lambda c: len(parse_omm_csv(c)) >= BULK_MIN_SETS,
                                                  "bulk GP file has too few element sets"))
                    self._bulk_error = res.error
                except LiveSourceError as exc:
                    self._bulk_error = str(exc)
        if res is None:
            res = fetch_cached("gp_active_csv", GP_ACTIVE_CSV_URL) or fetch_cached("gp_active", GP_ACTIVE_URL)
        if res is None:
            return None
        with self._lock:
            if res.version != self._bulk_version:
                if res.key == "gp_active_csv":
                    bulk = {}
                    for rec in parse_omm_csv(res.content):
                        try:
                            bulk[int(float(rec["NORAD_CAT_ID"]))] = rec
                        except (KeyError, TypeError, ValueError):
                            continue
                else:
                    bulk = parse_tle_catalog(res.text)
                n_bad = 0
                for k in list(bulk):            # one malformed record must never reach SGP4 callers
                    try:
                        satrec_from(bulk[k])
                    except Exception:  # noqa: BLE001
                        del bulk[k]
                        n_bad += 1
                if n_bad:
                    log.warning("bulk GP catalog (%s): dropped %d malformed element sets", res.key, n_bad)
                self._bulk = bulk
                self._bulk_tles = {}
                self._bulk_version = res.version
                log.info("bulk GP catalog (%s): %d objects", res.key, len(self._bulk))
            self._bulk_res = res
        return res

    def _from_bulk(self, norad: int) -> Optional[TLE]:
        t = self._bulk_tles.get(norad)
        if t is None and norad in self._bulk:
            rec = self._bulk[norad]
            r = self._bulk_res
            fetched, stale = (r.fetched_at if r else None), bool(r and r.stale)
            if isinstance(rec, dict):           # OMM CSV record
                t = TLE(norad, str(rec.get("OBJECT_NAME", "")).strip(), None, None, fetched, stale,
                        "CelesTrak GP GROUP=active (bulk)", omm=rec)
            else:
                name, l1, l2 = rec
                t = TLE(norad, name, l1, l2, fetched, stale, "CelesTrak GP GROUP=active (bulk TLE)")
            self._bulk_tles[norad] = t
        return t

    # -- lookup ------------------------------------------------------------------------------
    def lookup(self, norad: int) -> Optional[TLE]:
        """Best element set known locally (memory, bulk file, per-object disk cache); no network."""
        norad = int(norad)
        with self._lock:
            if self._bulk_version is None and time.time() - self._bulk_checked > 30:
                self._bulk_checked = time.time()
                self.load_bulk(network=False)
            cand = [t for t in (self._mem.get(norad), self._from_bulk(norad)) if t is not None]
            if norad not in self._mem and norad not in self._disk_checked:
                self._disk_checked.add(norad)
                res = fetch_cached(f"gp_{norad}", gp_url(norad))
                if res is not None:
                    try:
                        t = elements_from_result(norad, res, "CelesTrak GP CATNR query")
                        self._mem[norad] = t
                        cand.append(t)
                    except (ValueError, KeyError):
                        pass
        good = []
        for t in cand:
            try:
                good.append((t.epoch_jd, t))
            except Exception:  # noqa: BLE001 - malformed element set: never hand it out
                continue
        return max(good, key=lambda x: x[0])[1] if good else None

    def is_missing(self, norad: int) -> bool:
        with self._lock:
            m = self._missing_map().get(str(int(norad)))
            if not m:
                return False
            return time.time() - float(m.get("at", 0)) < GP_MISSING_TTL_S

    def status(self, norad: int) -> str:
        if self.lookup(norad) is not None:
            return "ok"
        return "missing" if self.is_missing(norad) else "pending"

    def request(self, norads: Iterable[int]) -> int:
        """Queue objects for the background worker (in priority order). Returns #queued."""
        n = 0
        with self._lock:
            for x in norads:
                x = int(x)
                if x <= 0 or x in self._queue or x in self._inflight:
                    continue
                if self.lookup(x) is not None or self.is_missing(x):
                    continue
                self._queue[x] = time.time()
                n += 1
        if n:
            self._wake.set()
        return n

    def get(self, norad: int, network: bool = False, interactive: bool = True) -> TLE:
        """Element set for `norad`; with network=True fetches it now if nothing is cached."""
        t = self.lookup(norad)
        if t is not None:
            return t
        if not network:
            raise LiveSourceError(f"no element set cached for {norad} (queued for download)")
        return self._fetch_one(int(norad), interactive=interactive)

    def _fetch_one(self, norad: int, interactive: bool) -> TLE:
        key, url = f"gp_{norad}", gp_url(norad)
        needs_net = not cache_is_fresh(key, url, TTL["gp"])
        wait_s = backoff_remaining(key, url) if needs_net else 0.0
        if wait_s > 0:              # fetch() will not make a request (stale copy or back-off error)
            needs_net = False
        if needs_net:
            try:
                _check_offline(url)
            except OfflineError as exc:     # not a provider failure: no budget, no breaker trip
                raise ProviderPaused(f"{exc}; {norad} stays queued") from exc
            with self._lock:
                self._budget_q()    # loads the persisted breaker state
                if time.time() < self._pause_until:
                    raise ProviderPaused(f"per-object GP queries paused until "
                                         f"{iso(datetime.fromtimestamp(self._pause_until, timezone.utc))} "
                                         f"({self._pause_reason}); {norad} stays queued")
                allowed = GP_BUDGET if interactive else GP_BUDGET - GP_INTERACTIVE_RESERVE
                if len(self._budget_q()) >= allowed:
                    raise BudgetExhausted(f"per-object GP budget used ({allowed} requests / "
                                          f"{GP_BUDGET_WINDOW_S // 3600} h); {norad} stays queued")
            with self._rate_lock:
                wait = self._last_req + GP_MIN_INTERVAL_S - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last_req = time.monotonic()
                with self._lock:
                    self._spend()
        try:
            res = fetch(key, url, TTL["gp"], validate=_validate_gp(norad))
            t = elements_from_result(norad, res, "CelesTrak GP CATNR query")
        except LiveSourceError as exc:
            cause = exc.__cause__
            with self._lock:
                self.n_failed += 1
                if isinstance(cause, NoDataError) or (isinstance(cause, HTTPStatusError) and cause.status == 404):
                    self._missing_map()[str(norad)] = {"at": time.time(), "reason": str(exc)[:200]}
                    self._save_missing()
                    self._consecutive_fail = 0          # the provider answered
                else:
                    self._retry_at[norad] = time.time() + max(GP_RETRY_S, wait_s)
                    if needs_net:
                        self._consecutive_fail += 1
                        refused = isinstance(cause, HTTPStatusError) and cause.status in (403, 429)
                        if refused or self._consecutive_fail >= GP_BREAKER_FAILS:
                            base = REFUSED_BACKOFF_S if refused else GP_BREAKER_PAUSE_S
                            self._pause_until = time.time() + min(base * 2 ** min(self._trips, 4), MAX_BACKOFF_S)
                            self._trips += 1
                            self._pause_reason = str(cause or exc)[:160]
                            self._save_state()
                            log.warning("pausing per-object GP queries until %s: %s",
                                        iso(datetime.fromtimestamp(self._pause_until, timezone.utc)),
                                        self._pause_reason)
            raise
        with self._lock:
            self._mem[norad] = t
            self._disk_checked.add(norad)
            if needs_net and not res.from_cache:
                self.n_fetched += 1
                self._consecutive_fail = 0
                if self._trips:
                    self._trips = 0
                    self._save_state()
        return t

    # -- background worker -------------------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tle-store", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def _next(self) -> Optional[int]:
        with self._lock:
            now = time.time()
            for x in list(self._queue):
                if self._retry_at.get(x, 0) > now:
                    continue
                del self._queue[x]
                if self.lookup(x) is not None or self.is_missing(x):
                    continue
                self._inflight.add(x)
                return x
        return None

    def _run(self):
        last_bulk = 0.0
        while not self._stop.is_set():
            try:
                if time.time() - last_bulk > 600:      # fetch() itself enforces the 6 h TTL
                    self.load_bulk(network=True)
                    last_bulk = time.time()
                x = self._next()
                if x is None:
                    self._wake.wait(30)
                    self._wake.clear()
                    continue
                sleep_for = 0.0
                try:
                    self._fetch_one(x, interactive=False)
                except BudgetExhausted as exc:            # budget used up or provider paused
                    with self._lock:
                        self._queue[x] = time.time()
                        self._queue.move_to_end(x, last=False)
                        if isinstance(exc, ProviderPaused):
                            until = self._pause_until
                        else:
                            until = (self._budget_q()[0] if self._budget_q() else time.time()) + GP_BUDGET_WINDOW_S
                    sleep_for = min(max(until - time.time(), 30), 600)
                except LiveSourceError as exc:
                    log.info("GP %s: %s", x, exc)
                    with self._lock:
                        if x in self._retry_at:
                            self._queue[x] = time.time()
                finally:
                    with self._lock:
                        self._inflight.discard(x)
                if sleep_for:
                    self._stop.wait(sleep_for)
            except Exception:  # noqa: BLE001 - keep the worker alive
                log.exception("TLE worker error")
                self._stop.wait(30)

    def stats(self) -> dict:
        with self._lock:
            r = self._bulk_res
            return {
                "bulk": {"url": r.url if r else GP_ACTIVE_CSV_URL, "n_objects": len(self._bulk),
                         "fetched_at": iso(r.fetched_at) if r else None,
                         "stale": bool(r.stale) if r else None, "error": self._bulk_error},
                "n_per_object_cached": len(self._mem),
                "n_queued": len(self._queue), "n_inflight": len(self._inflight),
                "n_missing": sum(1 for v in self._missing_map().values()
                                 if time.time() - float(v.get("at", 0)) < GP_MISSING_TTL_S),
                "budget_used": len(self._budget_q()), "budget": GP_BUDGET,
                "budget_window_h": GP_BUDGET_WINDOW_S / 3600, "min_interval_s": GP_MIN_INTERVAL_S,
                "worker_running": bool(self._thread and self._thread.is_alive()),
                "fetched_this_session": self.n_fetched, "failed_this_session": self.n_failed,
                "paused_until": iso(datetime.fromtimestamp(self._pause_until, timezone.utc))
                if time.time() < self._pause_until else None,
                "pause_reason": self._pause_reason if time.time() < self._pause_until else None,
            }


STORE = TLEStore()


def tle(norad: int, force: bool = False) -> TLE:
    """Element set for one object (cached locally or fetched now, throttled)."""
    return STORE.get(norad, network=True)


# ---------------------------------------------------------------------------------------------
# SGP4 propagation, frames
# ---------------------------------------------------------------------------------------------

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def jd_arrays(t0: datetime, offsets_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sgp4.api import jday
    t0 = t0.astimezone(timezone.utc)
    jd0, fr0 = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + t0.microsecond * 1e-6)
    fr = fr0 + np.asarray(offsets_s, dtype=float) / 86400.0
    whole = np.floor(fr)
    return jd0 + whole, fr - whole


def gmst_iau82(jd_ut1: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal time (IAU-82, Vallado eq. 3-47) in radians. UTC is used as UT1
    (|UT1-UTC| < 0.9 s -> < 0.5 km at the equator, invisible at dashboard scale)."""
    t = (np.asarray(jd_ut1, dtype=float) - 2451545.0) / 36525.0
    sec = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t + 0.093104 * t ** 2 - 6.2e-6 * t ** 3)
    return np.mod(sec, 86400.0) / 86400.0 * 2.0 * np.pi


def teme_to_ecef(r_teme: np.ndarray, jd_ut1: np.ndarray) -> np.ndarray:
    """Rotate TEME positions (N,3) to ECEF/PEF by GMST about z (polar motion ignored)."""
    th = gmst_iau82(jd_ut1)
    c, s = np.cos(th), np.sin(th)
    r = np.atleast_2d(np.asarray(r_teme, dtype=float))
    x = c * r[:, 0] + s * r[:, 1]
    y = -s * r[:, 0] + c * r[:, 1]
    return np.stack([x, y, r[:, 2]], axis=1)


def ecef_to_geodetic(xyz_m: np.ndarray) -> np.ndarray:
    """ECEF (N,3) metres -> (N,3) [lat deg, lon deg, alt km] on WGS84 (Bowring, 2 iterations;
    sub-metre for LEO-GEO altitudes)."""
    p_ = np.atleast_2d(np.asarray(xyz_m, dtype=float))
    x, y, z = p_[:, 0], p_[:, 1], p_[:, 2]
    a, e2 = WGS84_A, WGS84_E2
    b = a * (1 - WGS84_F)
    ep2 = (a * a - b * b) / (b * b)
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    beta = np.arctan2(z * a, p * b)
    for _ in range(2):
        lat = np.arctan2(z + ep2 * b * np.sin(beta) ** 3, p - e2 * a * np.cos(beta) ** 3)
        beta = np.arctan2((1 - WGS84_F) * np.sin(lat), np.cos(lat))
    sl, cl = np.sin(lat), np.cos(lat)
    n = a / np.sqrt(1 - e2 * sl ** 2)
    alt = p * cl + z * sl - a * a / n          # stable at all latitudes
    return np.stack([np.degrees(lat), np.degrees(lon), alt / 1000.0], axis=1)


def geodetic_to_ecef(lat_deg, lon_deg, alt_km) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    h = np.asarray(alt_km, dtype=float) * 1000.0
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - WGS84_E2) + h) * np.sin(lat)], axis=-1)


def rtn_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(N,3,3) rows = R, T, N unit vectors of the primary's RTN frame:
    R = r/|r|, N = (r x v)/|r x v|, T = N x R."""
    r, v = np.atleast_2d(r), np.atleast_2d(v)
    R = r / np.linalg.norm(r, axis=1, keepdims=True)
    h = np.cross(r, v)
    N = h / np.linalg.norm(h, axis=1, keepdims=True)
    T = np.cross(N, R)
    return np.stack([R, T, N], axis=1)


def to_rtn(r_a: np.ndarray, v_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Position of B relative to A expressed in A's RTN frame (same units as input)."""
    basis = rtn_basis(r_a, v_a)
    return np.einsum("nij,nj->ni", basis, np.atleast_2d(r_b) - np.atleast_2d(r_a))


def _sgp4(sat, jd, fr):
    e, r, v = sat.sgp4_array(np.atleast_1d(jd), np.atleast_1d(fr))
    return e, r, v


def _range_km(sa, sb, t0, dt_s: float) -> float:
    jd, fr = jd_arrays(t0, np.array([dt_s]))
    _, ra, _ = _sgp4(sa, jd, fr)
    _, rb, _ = _sgp4(sb, jd, fr)
    return float(np.linalg.norm(rb[0] - ra[0]))


def golden_min(f, a: float, b: float, tol: float = 1e-3, max_iter: int = 100) -> tuple[float, float]:
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


MAX_TRACK_SAMPLES = 4001


def _point(ra_km, rb_km, jd_full) -> dict:
    """Midpoint of A and B (TEME, km) at one instant -> {lat, lon, alt_km, range_km}."""
    mid = (np.asarray(ra_km) + np.asarray(rb_km)) / 2.0
    ecef = teme_to_ecef(mid[None, :] * 1000.0, np.array([jd_full]))
    lat, lon, alt = ecef_to_geodetic(ecef)[0]
    return {"lat": round(float(lat), 4), "lon": round(float(lon), 4), "alt_km": round(float(alt), 2),
            "range_km": round(float(np.linalg.norm(np.asarray(rb_km) - np.asarray(ra_km))), 3)}


def tca_point(norad_a: int, norad_b: int, tca, tle_a: Optional[TLE] = None, tle_b: Optional[TLE] = None,
              store: Optional[TLEStore] = None) -> Optional[dict]:
    """Midpoint of the two objects at `tca` (SGP4 with locally available element sets):
    {lat, lon, alt_km, range_km}; `range_km` is the SGP4 separation at that instant with *our*
    element sets (they can be newer than the ones SOCRATES used). None if an element set is missing
    or SGP4 fails."""
    store = store or STORE
    ta = tle_a or store.lookup(norad_a)
    tb = tle_b or store.lookup(norad_b)
    if ta is None or tb is None:
        return None
    t0 = parse_time(tca)
    from sgp4.api import jday
    jd, fr = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + t0.microsecond * 1e-6)
    try:
        ea, ra, _ = ta.satrec.sgp4(jd, fr)
        eb, rb, _ = tb.satrec.sgp4(jd, fr)
    except Exception:  # noqa: BLE001 - malformed element set
        return None
    if ea or eb:
        return None
    return _point(ra, rb, jd + fr)


CHECK_WINDOW_S = 120.0     # our own closest approach is searched within +/- this of the SOCRATES TCA
CHECK_STEP_S = 2.0


def closest_point(tle_a: TLE, tle_b: TLE, tca, window_s: float = CHECK_WINDOW_S) -> Optional[dict]:
    """Closest approach of A and B near `tca` with *our* element sets (2 s samples over +/- window_s,
    refined by golden-section search), i.e. the SGP4 counterpart of SOCRATES' miss distance.

    Returns the midpoint there, {lat, lon, alt_km, range_km (= our min range), tca_offset_s (vs `tca`),
    min_at_edge}, or None if SGP4 fails. Raises if an element set is malformed."""
    t0 = parse_time(tca)
    sa, sb = tle_a.satrec, tle_b.satrec
    offsets = np.arange(-window_s, window_s + CHECK_STEP_S / 2, CHECK_STEP_S)
    jd, fr = jd_arrays(t0, offsets)
    ea, ra, _ = _sgp4(sa, jd, fr)
    eb, rb, _ = _sgp4(sb, jd, fr)
    ok = (ea == 0) & (eb == 0)
    if not ok.any():
        return None
    rel = np.where(ok, np.linalg.norm(rb - ra, axis=1), np.inf)
    i0 = int(np.argmin(rel))
    lo_b, hi_b = offsets[max(i0 - 1, 0)], offsets[min(i0 + 1, len(offsets) - 1)]
    dt, _ = golden_min(lambda x: _range_km(sa, sb, t0, x), lo_b, hi_b, tol=1e-3)
    jd_m, fr_m = jd_arrays(t0, np.array([dt]))
    e1, ra_m, _ = _sgp4(sa, jd_m, fr_m)
    e2, rb_m, _ = _sgp4(sb, jd_m, fr_m)
    if e1[0] or e2[0]:
        return None
    p = _point(ra_m[0], rb_m[0], float(jd_m[0] + fr_m[0]))
    p.update(tca_offset_s=round(float(dt), 3), min_at_edge=bool(i0 in (0, len(offsets) - 1)))
    return p


def propagate(norad_a: int, norad_b: int, tca, span_min: float = 50.0, step_s: float = 20.0,
              tle_a: Optional[TLE] = None, tle_b: Optional[TLE] = None,
              store: Optional[TLEStore] = None) -> dict:
    """SGP4-propagate both objects over [tca - span/2, tca + span/2].

    Returns ECEF samples (metres) and geodetic lat/lon/alt for both, B relative to A in A's RTN
    frame (metres), and the minimum range found by golden-section refinement around the sampled
    minimum (with the element sets available now, which may be newer than SOCRATES's)."""
    t0 = parse_time(tca)
    if not (0 < step_s <= 600):
        raise ValueError("step_s must be in (0, 600]")
    if not (0 < span_min <= 24 * 60):
        raise ValueError("span_min must be in (0, 1440]")
    half = span_min * 30.0
    n = int(round(2 * half / step_s)) + 1
    if n > MAX_TRACK_SAMPLES:
        raise ValueError(f"too many samples ({n}); increase step_s or reduce span_min")
    offsets = np.linspace(-half, half, n)
    store = store or STORE
    ta = tle_a or store.get(norad_a, network=True)
    tb = tle_b or store.get(norad_b, network=True)
    sa, sb = ta.satrec, tb.satrec
    jd, fr = jd_arrays(t0, offsets)
    ea, ra, va = _sgp4(sa, jd, fr)
    eb, rb, vb = _sgp4(sb, jd, fr)
    from sgp4.api import SGP4_ERRORS
    for label, e, t in (("a", ea, ta), ("b", eb, tb)):
        bad = np.nonzero(e)[0]
        if len(bad) == len(e):
            raise ValueError(f"SGP4 failed for {label} ({t.norad}): {SGP4_ERRORS.get(int(e[bad[0]]), e[bad[0]])}")
    ok = (ea == 0) & (eb == 0)
    if not ok.any():
        raise ValueError("SGP4 failed for both objects at every sample")

    rel = np.linalg.norm(rb - ra, axis=1)
    i0 = int(np.argmin(np.where(ok, rel, np.inf)))
    lo_b = offsets[max(i0 - 1, 0)]
    hi_b = offsets[min(i0 + 1, n - 1)]
    dt_min, rmin_km = golden_min(lambda x: _range_km(sa, sb, t0, x), lo_b, hi_b, tol=1e-3)
    jd_m, fr_m = jd_arrays(t0, np.array([dt_min]))
    _, ra_m, va_m = _sgp4(sa, jd_m, fr_m)
    _, rb_m, vb_m = _sgp4(sb, jd_m, fr_m)
    rel_speed = float(np.linalg.norm(vb_m[0] - va_m[0]))
    rtn_tca = to_rtn(ra_m, va_m, rb_m)[0]

    jd_full = jd + fr
    ecef_a = teme_to_ecef(ra, jd_full) * 1000.0
    ecef_b = teme_to_ecef(rb, jd_full) * 1000.0
    lla_a, lla_b = ecef_to_geodetic(ecef_a), ecef_to_geodetic(ecef_b)
    rtn = to_rtn(ra, va, rb) * 1000.0
    t_iso = [iso(t0 + timedelta(seconds=float(o)), ms=True) for o in offsets]

    def obj(t: TLE, ecef, lla, e):
        epoch = t.epoch
        good = e == 0
        return {"norad": t.norad, "name": t.name, "tle_epoch": iso(epoch, ms=True),
                "tle_age_d": round((t0 - epoch).total_seconds() / 86400.0, 3),
                "tle": [t.line1, t.line2] if t.line1 else None,
                "elements_source": t.source + (" (OMM CSV)" if t.omm else ""),
                "tle_fetched_at": iso(t.fetched_at), "tle_stale": t.stale,
                "ecef": [[ts, round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)]
                         for ts, p, g in zip(t_iso, ecef, good) if g],
                "lla": [[ts, round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 4)]
                        for ts, p, g in zip(t_iso, lla, good) if g]}

    point = _point(ra_m[0], rb_m[0], float(jd_m[0] + fr_m[0]))
    point_req = tca_point(ta.norad, tb.norad, t0, tle_a=ta, tle_b=tb)
    return {
        "a": obj(ta, ecef_a, lla_a, ea), "b": obj(tb, ecef_b, lla_b, eb),
        "frame": {"ecef": "ECEF (TEME rotated by IAU-82 GMST, polar motion ignored), metres",
                  "lla": "WGS84 geodetic latitude/longitude (deg), altitude (km)",
                  "relative_rtn": "B minus A in A's RTN frame (R radial, T along-track, N cross-track), "
                                  "[dt_s from requested TCA, r, t, n] metres"},
        "relative_rtn": [[round(float(o), 3), round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)]
                         for o, p, k in zip(offsets, rtn, ok) if k],
        "min_range_m": round(rmin_km * 1000.0, 1),
        "tca_model": iso(t0 + timedelta(seconds=dt_min), ms=True),
        "tca_offset_s": round(dt_min, 3),
        "min_at_edge": bool(i0 in (0, n - 1)),
        "relative_speed_ms": round(rel_speed * 1000.0, 1),
        "rtn_at_tca_m": [round(float(x) * 1000.0, 1) for x in rtn_tca],
        "tca_point": point_req,          # midpoint at the requested (SOCRATES) TCA
        "tca_point_model": point,        # midpoint at our refined closest approach
        "requested": {"tca": iso(t0, ms=True), "span_min": span_min, "step_s": step_s, "n_samples": n},
        "propagator": "SGP4 (python-sgp4, WGS72) from CelesTrak GP element sets",
    }


# ---------------------------------------------------------------------------------------------
# Space-Track cdm_public (optional)
# ---------------------------------------------------------------------------------------------

def spacetrack_configured() -> bool:
    return bool(os.environ.get("SPACETRACK_USER")) and bool(os.environ.get("SPACETRACK_PASS"))


def _spacetrack_getter(url: str, headers: dict):
    """Login + query in one session. Credentials come only from the environment and are never
    logged, cached or returned."""
    _check_offline(url)
    import requests
    with requests.Session() as s:
        s.headers.update({"User-Agent": headers.get("User-Agent", USER_AGENT)})
        r = s.post(SPACETRACK_LOGIN, data={"identity": os.environ["SPACETRACK_USER"],
                                           "password": os.environ["SPACETRACK_PASS"]}, timeout=HTTP_TIMEOUT)
        if r.status_code != 200 or "Failed" in r.text[:200]:
            return 401, b"login failed", {}
        q = s.get(url, timeout=HTTP_TIMEOUT)
        return q.status_code, q.content, dict(q.headers)


def spacetrack_cdms(force: bool = False) -> dict:
    if not spacetrack_configured():
        return {"status": "disabled", "cdms": [],
                "note": "set SPACETRACK_USER and SPACETRACK_PASS environment variables to enable"}
    res = fetch("spacetrack_cdm_public", SPACETRACK_CDM_URL, TTL["spacetrack"], force=force,
                getter=_spacetrack_getter, validate=json.loads)
    data = json.loads(res.content)
    return {**res.status(), "cdms": data if isinstance(data, list) else []}


def spacetrack_status() -> dict:
    if not spacetrack_configured():
        return {"status": "disabled", "fetched_at": None, "data_as_of": None, "stale": None, "error": None,
                "note": "set SPACETRACK_USER / SPACETRACK_PASS to enable"}
    return cache_status("spacetrack_cdm_public")


def sources_status(store: Optional[TLEStore] = None) -> dict:
    """Per-source status from the cache metadata only (never touches the network)."""
    soc = {k: cache_status(f"socrates_{k}", TTL["socrates"]) for k in SOCRATES_URLS}
    soc_overall = dict(soc["minRange"])
    if soc_overall.get("status") != "ok" and soc["maxProb"].get("status") == "ok":
        soc_overall = dict(soc["maxProb"])
    soc_overall["files"] = soc
    sw = {k: cache_status(f"swpc_{k}", TTL["swpc"]) for k in SWPC_URLS}
    sw_overall = dict(sw["f107"])
    sw_overall["products"] = sw
    sw_overall["fallback_celestrak"] = cache_status("celestrak_sw", TTL["celestrak_sw"])
    st = store or STORE
    gp = st.stats()
    if st._bulk_res is not None:                    # the bulk file in use; stale once its refresh is overdue
        bulk_cache = cache_status(st._bulk_res.key, TTL["gp_bulk"])
    else:
        bulk_cache = cache_status("gp_active_csv", TTL["gp_bulk"])
        if bulk_cache["status"] == "not fetched" and _read_meta("gp_active") is not None:
            bulk_cache = cache_status("gp_active", TTL["gp_bulk"])
    gp["status"] = bulk_cache["status"]
    gp["bulk"]["cache"] = bulk_cache
    satcat = cache_status("satcat", TTL["satcat"])
    for name, d in (("socrates", soc_overall), ("satcat", satcat)):
        if PARSE_ERRORS.get(name):                  # cached payload unreadable: previous parse served
            d.update(status="stale" if d.get("status") == "ok" else d.get("status"), stale=True,
                     parse_error=PARSE_ERRORS[name])
    return {"socrates": soc_overall, "swpc": sw_overall, "satcat": satcat,
            "spacetrack": spacetrack_status(), "gp": gp}
