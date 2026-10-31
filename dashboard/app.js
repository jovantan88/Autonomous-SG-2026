/* Conjunction Watch - operator dashboard (vanilla JS, no build step).
 * Contract: docs/api_contract.md. All distances in metres, speeds in m/s, log_pc = log10(Pc),
 * tiers 0 Safe / 1 Elevated / 2 Critical. ?mock=1 reads dashboard/mock/*.json instead of /api.
 * Tier rule: Critical = Pc >= 1e-4 or miss < 1 km; Elevated = 1e-6 <= Pc < 1e-4 or miss 1-5 km.
 * The tool recommends; the analyst decides and every decision is logged.
 */
(function () {
  'use strict';

  // ================================================================ config / state
  const QS = new URLSearchParams(location.search);
  const MOCK = QS.get('mock') === '1';
  const API_BASE = (QS.get('api') || '').replace(/\/+$/, '');
  const POLL_LIVE_MS = 60000;
  const LIVE_LIMIT = 500;     // highest-risk conjunctions per poll
  const EXTRA_LIMIT = 150;    // plus the top Elevated and the top needs-review ones
  const CLOSEST_LIMIT = 10;   // plus the closest approaches (for the insights)
  const POLL_SW_MS = 10 * 60000;
  const POLL_HEALTH_MS = 60000;
  const TIERS = ['Safe', 'Elevated', 'Critical'];
  const TIER_CSS = ['t0', 't1', 't2'];
  // mirrors the :root tokens in styles.css
  const COLORS = {
    crit: '#FF5C5C', elev: '#F2B33D', safe: '#4CC38A', review: '#A98BFF',
    a: '#3DD6F5', b: '#F2913D', text: '#E8EEF4', muted: '#8FA1B3', focus: '#7CC4FF',
    grid: 'rgba(51,65,79,.75)', line: '#3B4A59', panel: '#1F2A36', bg: '#18212B', outline: '#0E141A',
  };
  const FONT = '"IBM Plex Sans", system-ui, sans-serif';
  const TIER_COLOR = [COLORS.safe, COLORS.elev, COLORS.crit];
  const REDUCED_MOTION = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  // Keyless dark basemaps. CARTO dark_all now answers every tile with an "API KEY REQUIRED"
  // image (HTTP 200), so Esri's World Dark Gray canvas is the default; ?basemap=carto|natural overrides.
  const ESRI = 'https://services.arcgisonline.com/arcgis/rest/services/Canvas';
  const BASEMAPS = {
    esri: { url: `${ESRI}/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, probe: `${ESRI}/World_Dark_Gray_Base/MapServer/tile/0/0/0`,
      ref: `${ESRI}/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`,
      credit: 'Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community' },
    carto: { url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png', subdomains: 'abcd', probe: 'https://a.basemaps.cartocdn.com/dark_all/0/0/0.png',
      credit: '© OpenStreetMap contributors © CARTO' },
  };
  const BASEMAP = QS.get('basemap') || 'esri';
  const STATUS_TEXT = {
    '+': 'Active', '-': 'Inactive', '?': 'Status unknown', P: 'Partially operational', B: 'Backup / standby',
    S: 'Spare', X: 'Extended mission', D: 'Decayed', '': 'Status not given',
  };
  const ALLOWED = ['event_id', 'time_to_tca', 'miss_distance', 'relative_speed', 'relative_position_r',
    'relative_position_t', 'relative_position_n', 'c_object_type', 'c_sigma_r', 't_sigma_r',
    'max_risk_estimate', 'F10', 'AP'];
  const OBJECT_TYPES = ['DEBRIS', 'PAYLOAD', 'ROCKET BODY', 'UNKNOWN', 'TBA'];

  const S = {
    view: 'live',
    windowH: 24,
    live: null, liveErr: null, liveExtraErr: [], liveLoading: false, liveFetchedAt: null, liveSeq: 0,
    sw: null, swErr: null,
    health: null, healthErr: null,
    feedFilter: 'critical', feedSort: 'risk', hourFilter: null,
    search: '',
    selectedId: null, selectedRow: null, track: null, trackErr: null, trackLoading: false, trackSeq: 0,
    layers: { elev: true, labels: true },
    test: { events: null, err: null, loading: false, pred: 'all', review: false, dis: false, under: false, sort: 'risk',
      selected: null, detail: null, detailErr: null, detailLoading: false, cdmIdx: null, seq: 0 },
    metrics: null, metricsErr: null, metricsLoading: false,
    log: [], logPersist: true,
    analyst: '',
  };

  // ================================================================ tiny DOM helpers
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const esc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const num = (v) => { const n = typeof v === 'string' && v.trim() !== '' ? Number(v) : v; return isNum(n) ? n : null; };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const get = (o, path, d) => { let c = o; for (const k of path.split('.')) { if (c == null) return d; c = c[k]; } return c == null ? d : c; };

  function lsGet(key, fallback) {
    try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch (e) { return fallback; }
  }
  function lsSet(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; }
  }

  let toastTimer = null;
  function toast(msg, bad) {
    const t = $('#toast');
    if (!t) return;
    t.textContent = msg;
    t.classList.toggle('bad', !!bad);
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, bad ? 6500 : 3500);
  }

  // ================================================================ formatting (units on every number)
  function fmtDist(m) {
    m = num(m);
    if (m == null) return '–';
    if (Math.abs(m) < 1000) return `${Math.round(m)}\u00a0m`;
    if (Math.abs(m) < 10000) return `${(m / 1000).toFixed(2)}\u00a0km`;
    if (Math.abs(m) < 1e6) return `${(m / 1000).toFixed(1)}\u00a0km`;
    return `${Math.round(m / 1000).toLocaleString('en-US')}\u00a0km`;
  }
  function fmtSpeed(ms) { ms = num(ms); return ms == null ? '–' : `${(ms / 1000).toFixed(2)}\u00a0km/s`; }
  function pcText(lp) {
    lp = num(lp);
    if (lp == null) return '–';
    if (lp <= -29.9) return '<1e-30';
    const v = Math.pow(10, lp);
    const s = v.toExponential(1).replace('e+', 'e');
    return s.replace(/e-0(\d)/, 'e-$1');
  }
  function pcHtml(lp, extra) {
    const t = num(lp) == null ? 'not available' : `log10 Pc = ${num(lp).toFixed(2)}${lp <= -29.9 ? ' (dataset floor: Pc effectively 0)' : ''}`;
    return `<span title="${esc(t + (extra ? '; ' + extra : ''))}">${esc(pcText(lp))}</span>`;
  }
  function pct(p, digits) {
    p = num(p);
    if (p == null) return '–';
    const v = p * 100;
    if (v > 0 && v < 1 && !digits) return '<1%';
    if (v < 100 && v > 99 && !digits) return '>99%';
    return `${v.toFixed(digits || 0)}%`;
  }
  function fmtNum(v, d) { v = num(v); return v == null ? '–' : v.toLocaleString('en-US', { maximumFractionDigits: d == null ? 2 : d, minimumFractionDigits: 0 }); }
  function fixed(v, d) { v = num(v); return v == null ? '–' : v.toFixed(d); }
  function parseT(iso) {
    if (!iso) return null;
    if (iso instanceof Date) return iso.getTime();
    let s = String(iso).trim();
    if (/^\d{4}-\d{2}-\d{2} \d/.test(s)) s = s.replace(' ', 'T');
    if (/T\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(s)) s += 'Z';
    const t = Date.parse(s);
    return Number.isFinite(t) ? t : null;
  }
  const pad = (n) => String(n).padStart(2, '0');
  // "10:53:32 UTC" (withDate false), "31 Oct 10:53:32 UTC" (default), "10:53 UTC" (withDate 'hm')
  function fmtUTC(iso, withDate) {
    const t = parseT(iso);
    if (t == null) return '–';
    const d = new Date(t);
    const hm = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    if (withDate === 'hm') return `${hm} UTC`;
    const hms = `${hm}:${pad(d.getUTCSeconds())} UTC`;
    if (withDate === false) return hms;
    return `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${hms}`;
  }
  const sameUtcDay = (a, b) => new Date(a).toISOString().slice(0, 10) === new Date(b).toISOString().slice(0, 10);
  // time of day, with the date only when it is not today (UTC)
  function fmtWhen(iso) {
    const t = parseT(iso);
    if (t == null) return '–';
    return sameUtcDay(t, Date.now()) ? fmtUTC(iso, false) : fmtUTC(iso);
  }
  function fmtDur(ms) {
    let s = Math.abs(ms) / 1000;
    const d = Math.floor(s / 86400); s -= d * 86400;
    const h = Math.floor(s / 3600); s -= h * 3600;
    const m = Math.floor(s / 60);
    if (d > 0) return `${d} d ${h} h`;
    if (h > 0) return `${h} h ${m} m`;
    if (m > 0) return `${m} m`;
    return 'under 1 m';
  }
  function countdown(iso) {
    const t = parseT(iso);
    if (t == null) return '–';
    const dt = t - Date.now();
    return dt >= 0 ? `in ${fmtDur(dt)}` : `${fmtDur(-dt)} ago`;
  }
  function ago(iso) {
    const t = parseT(iso);
    if (t == null) return 'unknown';
    const dt = Date.now() - t;
    if (dt < 0) return `in ${fmtDur(-dt)}`;
    if (dt < 60000) return 'just now';
    return `${fmtDur(dt)} ago`;
  }
  // tier pill: coloured dot + the word (colour always paired with text)
  function tierPill(t) {
    if (!isNum(t) || t < 0 || t > 2) return '<span class="tier nt"><i class="dot"></i>No tier</span>';
    return `<span class="tier ${TIER_CSS[t]}"><i class="dot ${TIER_CSS[t]}"></i>${TIERS[t]}</span>`;
  }
  const REVIEW_TIP = 'Needs review: the uncertainty spans a tier boundary, or the worst-case Pc reaches 1e-4.';
  const reviewPill = () => `<span class="tag review" title="${esc(REVIEW_TIP)}">Review</span>`;
  const lv = (label, html) => `<span class="lv"><span class="l">${esc(label)}</span>${html}</span>`;
  function pairHtml(a, b) {
    return `${esc(a || '?')} <span class="x">×</span> ${esc(b || '?')}`;
  }
  const pairText = (r) => `${get(r, 'sat1.name', '?')} × ${get(r, 'sat2.name', '?')}`;
  const tierOf = (row) => num(get(row, 'assessment.tier'));
  const ptier = (row, i) => num(get(row, `assessment.p_tier.${i}`));

  // ================================================================ API layer (real or mock)
  class ApiError extends Error {
    constructor(msg, status) { super(msg); this.status = status; }
  }
  async function fetchJSON(url, opts) {
    opts = opts || {};
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), opts.timeout || 45000);
    let res;
    try {
      res = await fetch(url, {
        method: opts.method || 'GET',
        headers: opts.body ? { 'Content-Type': 'application/json' } : undefined,
        body: opts.body ? JSON.stringify(opts.body) : undefined,
        signal: ctl.signal,
        cache: 'no-store',
      });
    } catch (e) {
      clearTimeout(timer);
      throw new ApiError(e.name === 'AbortError' ? 'request timed out' : 'API unreachable (is the backend running?)', 0);
    }
    clearTimeout(timer);
    let body = null;
    const text = await res.text();
    try { body = text ? JSON.parse(text) : null; } catch (e) { body = null; }
    if (!res.ok) {
      let detail = body && (body.detail || body.error || body.message);
      if (Array.isArray(detail)) detail = detail.map((d) => (d && d.msg ? `${Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : ''}: ${d.msg}` : JSON.stringify(d))).join('; ');
      if (detail && typeof detail !== 'string') detail = JSON.stringify(detail);
      throw new ApiError(`HTTP ${res.status}${detail ? ': ' + String(detail).slice(0, 300) : ''}`, res.status);
    }
    if (body == null) throw new ApiError('empty or invalid JSON response', res.status);
    return body;
  }

  let mockShiftMs = null;
  const mockCache = {};
  async function mockFile(name) {
    if (!mockCache[name]) {
      mockCache[name] = fetchJSON(`mock/${name}`).catch((e) => { delete mockCache[name]; throw e; });
    }
    const v = await mockCache[name];
    return JSON.parse(JSON.stringify(v));
  }
  function shiftIso(iso, ms) { const t = parseT(iso); return t == null ? iso : new Date(t + ms).toISOString(); }

  async function mockApi(path, params, opts) {
    await new Promise((r) => setTimeout(r, 120 + Math.random() * 180));
    if (path === '/api/health') return mockFile('health.json');
    if (path === '/api/metrics') return mockFile('metrics.json');
    if (path === '/api/test/events') return mockFile('test_events.json');
    let m = path.match(/^\/api\/test\/events\/(\d+)$/);
    if (m) {
      try { return await mockFile(`test_event_${m[1]}.json`); } catch (e) {
        throw new ApiError(`no mock fixture for event ${m[1]} (fixtures exist for events 1, 48, 82, 107, 109)`, 404);
      }
    }
    if (path === '/api/live/spaceweather') return mockFile('spaceweather.json');
    if (path === '/api/live/conjunctions') {
      const d = await mockFile('live_conjunctions.json');
      if (mockShiftMs == null) mockShiftMs = Date.now() - (parseT(d.fetched_at) || Date.now());
      const sh = mockShiftMs;
      d.fetched_at = shiftIso(d.fetched_at, sh);
      d.data_as_of = shiftIso(d.data_as_of, sh);
      const now = Date.now();
      const w = num(params.window_h) || 24;
      let rows = (d.conjunctions || []).map((r) => {
        r.tca = shiftIso(r.tca, sh);
        r.hours_to_tca = (parseT(r.tca) - now) / 3.6e6;
        return r;
      }).filter((r) => r.hours_to_tca >= -0.5 && r.hours_to_tca <= w);
      d.window_h = w;
      d.warnings = computeWarnings(rows, w);
      if (params.tier != null) { const want = String(params.tier).split(',').map(Number); rows = rows.filter((r) => want.includes(tierOf(r))); }
      if (params.needs_review != null) rows = rows.filter((r) => !!get(r, 'assessment.needs_review') === (String(params.needs_review) === '1'));
      rows = sortRows(rows, ({ range: 'md', tca: 'tca' })[params.sort] || 'risk').slice(0, num(params.limit) || 500);
      if (String(params.points) !== '1') rows.forEach((r) => { delete r.tca_point; delete r.tle_status; });
      d.conjunctions = rows;
      return d;
    }
    if (path === '/api/live/track') {
      const t = await mockFile('track.json');
      return rebaseMockTrack(t, params);
    }
    if (path === '/api/assess') {
      const a = await mockFile('assess.json');
      a._mock_note = 'Mock response (fixture) - not computed from the submitted CDM.';
      return a;
    }
    throw new ApiError(`no mock for ${path}`, 404);
  }

  function api(path, params, opts) {
    params = params || {};
    if (MOCK) return mockApi(path, params, opts || {});
    const qs = new URLSearchParams();
    Object.keys(params).forEach((k) => { if (params[k] != null && params[k] !== '') qs.set(k, params[k]); });
    const q = qs.toString();
    return fetchJSON(`${API_BASE}${path}${q ? '?' + q : ''}`, opts);
  }

  // Mock only: move the single track fixture so that it passes through the selected event's
  // collision point at its TCA (rotation about Earth's centre + time shift). Labelled in the UI.
  function rebaseMockTrack(t, params) {
    const row = S.selectedRow;
    const tcaTarget = parseT(params.tca);
    const tcaModel = parseT(t.tca_model);
    const isFixturePair = num(params.a) === num(get(t, 'a.norad')) && num(params.b) === num(get(t, 'b.norad'));
    const dtShift = tcaTarget != null && tcaModel != null ? tcaTarget - tcaModel : 0;
    const shiftSamples = (arr) => arr.map((p) => [shiftIso(p[0], dtShift), p[1], p[2], p[3]]);
    t.a.ecef = shiftSamples(t.a.ecef || []);
    t.b.ecef = shiftSamples(t.b.ecef || []);
    t.tca_model = shiftIso(t.tca_model, dtShift);
    if (!isFixturePair && row) {
      t.a.name = get(row, 'sat1.name', t.a.name); t.a.norad = get(row, 'sat1.norad', t.a.norad);
      t.b.name = get(row, 'sat2.name', t.b.name); t.b.norad = get(row, 'sat2.norad', t.b.norad);
      const pt = row.tca_point;
      if (pt && isNum(pt.lat) && isNum(pt.lon)) {
        const i0 = nearestIdx(t.a.ecef, parseT(t.tca_model));
        const mid = [0, 1, 2].map((k) => (t.a.ecef[i0][k + 1] + t.b.ecef[i0][k + 1]) / 2);
        const target = geodeticToEcef(pt.lat, pt.lon, (pt.alt_km || 550) * 1000);
        const R = rotationBetween(mid, target);
        const scale = norm(target) / norm(mid);
        const tf = (p) => { const v = matVec(R, [p[1], p[2], p[3]]); return [p[0], v[0] * scale, v[1] * scale, v[2] * scale]; };
        t.a.ecef = t.a.ecef.map(tf);
        t.b.ecef = t.b.ecef.map(tf);
      }
      const md = num(row.miss_distance);
      if (md && num(t.min_range_m)) {
        const k = md / t.min_range_m;
        t.relative_rtn = (t.relative_rtn || []).map((p) => [p[0], p[1] * k, p[2] * k, p[3] * k]);
        if (Array.isArray(t.rtn_at_tca_m)) t.rtn_at_tca_m = t.rtn_at_tca_m.map((x) => x * k);
        t.min_range_m = md;
      }
    }
    t.mock_note = isFixturePair ? 'Mock fixture track' : 'Mock: fixture track re-projected through this collision point';
    return t;
  }
  function nearestIdx(samples, tMs) {
    let best = 0; let bd = Infinity;
    samples.forEach((p, i) => { const d = Math.abs(parseT(p[0]) - tMs); if (d < bd) { bd = d; best = i; } });
    return best;
  }
  const norm = (v) => Math.hypot(v[0], v[1], v[2]);
  function geodeticToEcef(lat, lon, h) {
    const a = 6378137.0; const f = 1 / 298.257223563; const e2 = f * (2 - f);
    const la = lat * Math.PI / 180; const lo = lon * Math.PI / 180;
    const n = a / Math.sqrt(1 - e2 * Math.sin(la) ** 2);
    return [(n + h) * Math.cos(la) * Math.cos(lo), (n + h) * Math.cos(la) * Math.sin(lo), (n * (1 - e2) + h) * Math.sin(la)];
  }
  function rotationBetween(u, v) {
    const a = u.map((x) => x / norm(u)); const b = v.map((x) => x / norm(v));
    const k = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const s = norm(k); const c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    if (s < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const K = [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]];
    const K2 = matMul(K, K);
    const f = (1 - c) / (s * s);
    return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + K[i][j] + K2[i][j] * f));
  }
  function matMul(A, B) { return [0, 1, 2].map((i) => [0, 1, 2].map((j) => A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j])); }
  function matVec(A, v) { return [0, 1, 2].map((i) => A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2]); }

  // ================================================================ live data helpers
  // Same order as the server's _risk_order: tier, P(Critical), P(Pc >= 1e-4), SOCRATES max Pc, miss distance.
  // (The live median est. Pc is not used: on SOCRATES inputs it is out of distribution, see pcInfo.)
  function riskKey(r) {
    const t = tierOf(r); const pc = ptier(r, 2); const pcc = num(get(r, 'assessment.p_pc_critical'));
    const mx = num(r.max_risk_estimate) != null ? num(r.max_risk_estimate) : num(get(r, 'assessment.log_pc'));
    return [t == null ? -1 : t, pc == null ? 0 : pc, pcc == null ? 0 : pcc, mx == null ? -31 : mx, -(num(r.miss_distance) || 1e9)];
  }
  const mdTier = (m) => (m < 1000 ? 2 : (m <= 5000 ? 1 : 0));
  const firstNum = (...vs) => { for (const v of vs) { if (num(v) != null) return num(v); } return null; };
  // Our SGP4 closest approach (current element sets) vs SOCRATES' screening miss distance. They "agree"
  // when the distance rule gives the same tier and they are within 1 km of each other.
  function pointCheck(r) {
    const chk = r.sgp4_check || null;
    const t = S.track && S.selectedId === r.id ? S.track : null;
    const sgp = firstNum(chk && chk.min_range_m, t && t.min_range_m, r.tca_point && num(r.tca_point.range_km) != null ? r.tca_point.range_km * 1000 : null);
    const miss = num(r.miss_distance);
    if (sgp == null || miss == null) return { known: false };
    const agrees = chk && typeof chk.agrees === 'boolean' ? chk.agrees : (mdTier(sgp) === mdTier(miss) && Math.abs(sgp - miss) <= 1000);
    return { known: true, agrees, sgp, miss, chk, text: `Our orbit data puts them ${fmtDist(sgp)} apart; the screening says ${fmtDist(miss)}` };
  }
  const MISMATCH_TIP = 'Orbit mismatch: propagating our newest element sets does not reproduce the screening (different distance tier, or more than 1 km apart). One object may have manoeuvred or have a newer element set. Re-screen before acting.';
  // Live rows: the screening model's Pc estimate on SOCRATES inputs (out of distribution) is only
  // informative when its 80% interval is narrow and the classifier and quantile heads agree.
  function pcInfo(r) {
    const a = r.assessment || {};
    const lo = num(a.log_pc_lo); const hi = num(a.log_pc_hi); const pcc = num(a.p_pc_critical);
    const thr = firstNum(get(S.live, 'model.thresholds.p_critical'), get(S.health, 'model.thresholds.screen.p_critical'));
    const width = lo != null && hi != null ? hi - lo : null;
    const disagree = a.heads_disagree === true || (hi != null && pcc != null && thr != null && hi < -4 && pcc >= thr);
    const wide = width != null && width > 6;
    const why = [wide ? `its 80% interval spans ${width.toFixed(0)} orders of magnitude` : null,
      disagree ? `the classifier gives P(Pc ≥ 1e-4) = ${pct(pcc)} while the 90th-percentile Pc is ${pcText(hi)}` : null].filter(Boolean).join('; ');
    return { informative: !(wide || disagree), wide, disagree, width,
      tip: why ? `The model estimate is hidden because ${why}. Live screening data has no covariance, so the screening maximum Pc is the reference.` : '' };
  }
  const maxPcText = (r) => (num(r.max_risk_estimate) != null ? pcText(r.max_risk_estimate) : (isNum(r.max_prob) ? (r.max_prob > 0 ? r.max_prob.toExponential(1) : '0') : '–'));
  function cmpKeys(a, b) { for (let i = 0; i < a.length; i++) { if (a[i] !== b[i]) return b[i] - a[i]; } return 0; }
  function sortRows(rows, how) {
    const out = rows.slice();
    if (how === 'tca') out.sort((a, b) => (parseT(a.tca) || 0) - (parseT(b.tca) || 0));
    else if (how === 'md') out.sort((a, b) => (num(a.miss_distance) || 1e12) - (num(b.miss_distance) || 1e12));
    else out.sort((a, b) => cmpKeys(riskKey(a), riskKey(b)));
    return out;
  }
  function computeWarnings(rows, w) {
    const byHour = Array.from({ length: Math.max(1, Math.ceil(w)) }, (_, h) => ({ hour: h, critical: 0, elevated: 0 }));
    const prim = {};
    let n1 = 0; let n5 = 0; let nc = 0; let ne = 0; let past = 0;
    rows = rows.filter((r) => { const p = num(r.hours_to_tca) < 0; if (p) past++; return !p; });   // TCA passed: not "next N h"
    rows.forEach((r) => {
      const md = num(r.miss_distance); const t = tierOf(r);
      if (md != null && md < 1000) n1++;
      if (md != null && md <= 5000) n5++;
      if (t === 2) nc++;
      if (t === 1) ne++;
      const h = Math.floor(num(r.hours_to_tca));
      if (h >= 0 && h < byHour.length) { if (t === 2) byHour[h].critical++; else if (t === 1) byHour[h].elevated++; }
      const nm = get(r, 'sat1.name');
      if (nm) prim[nm] = (prim[nm] || 0) + 1;
    });
    const top = Object.keys(prim).map((k) => ({ name: k, count: prim[k] })).sort((a, b) => b.count - a.count).slice(0, 5);
    return { window_h: w, n_in_window: rows.length, n_recent_past: past, n_within_1km: n1, n_within_5km: n5, n_critical: nc, n_elevated: ne, by_hour: byHour, top_primaries: top, computed_client_side: true };
  }
  const liveRows = () => (S.live && Array.isArray(S.live.conjunctions) ? S.live.conjunctions : []);
  const rowById = (id) => liveRows().find((r) => r.id === id) || (S.selectedRow && S.selectedRow.id === id ? S.selectedRow : null);
  function matchesSearch(row, q) {
    if (!q) return true;
    const a = row.sat1 || {}; const b = row.sat2 || {};
    const hay = `${a.name || ''} ${a.norad || ''} ${b.name || ''} ${b.norad || ''}`.toLowerCase();
    return q.toLowerCase().split(/\s+/).filter(Boolean).every((tok) => hay.includes(tok));
  }
  function rowKey(r) {
    return r.id || `${get(r, 'sat1.norad')}-${get(r, 'sat2.norad')}-${r.tca}`;
  }

  // ================================================================ header: clock, data status
  function tickClock() {
    const d = new Date();
    const el = $('#clock-t');
    if (el) el.textContent = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  }
  function refreshCountdowns() {
    $$('[data-tca]').forEach((el) => {
      const t = parseT(el.getAttribute('data-tca'));
      el.textContent = countdown(el.getAttribute('data-tca'));
      if (el.classList.contains('it-time')) el.classList.toggle('soon', t != null && t - Date.now() < 6 * 3.6e6);
    });
  }

  // One header item: Live / Stale / Offline plus the data time. Source details, space weather and
  // input problems go in its popover.
  function renderStatus() {
    const el = $('#status');
    if (!el) return;
    const d = S.live;
    const rows = [];
    const warn = [];
    const bad = [];
    if (d) {
      rows.push(['Source', d.source || 'unknown']);
      rows.push(['Data from', `${fmtUTC(d.data_as_of)} (${ago(d.data_as_of)})`]);
      rows.push(['Fetched', `${fmtUTC(d.fetched_at)} (${ago(d.fetched_at)})`]);
      if (isNum(d.n_total)) rows.push(['Rows in feed', d.n_total.toLocaleString('en-US')]);
    }
    if (S.liveFetchedAt) rows.push(['Last update', fmtUTC(new Date(S.liveFetchedAt).toISOString())]);
    const sw = S.sw;
    if (sw) {
      rows.push(['Space weather', `F10.7 ${fmtNum(get(sw, 'f107.latest'), 0)} sfu, Ap ${fmtNum(get(sw, 'ap.latest'), 0)}, Kp ${fixed(get(sw, 'kp.latest'), 1)}${sw.stale ? ' (cached copy)' : ''}`]);
    } else if (S.swErr) rows.push(['Space weather', 'Unavailable']);
    const ml = get(S.health, 'model_loaded');
    if (ml != null) rows.push(['Model', ml ? 'Trained model loaded' : 'Not loaded: fallback estimates']);
    if (get(S.health, 'model.error')) bad.push(`Model reload failed, previous model kept: ${String(S.health.model.error).slice(0, 160)}`);
    if (S.liveErr) bad.push(`Last error: ${S.liveErr}`);
    if (d && d.refresh_error) bad.push(`Server refresh error: ${d.refresh_error}`);
    if (d && d.parse_error) bad.push(`Newest screening file unreadable, previous one served: ${String(d.parse_error).slice(0, 160)}`);
    if (d && d.files && typeof d.files === 'object') {
      Object.keys(d.files).forEach((k) => { const f = d.files[k]; if (f && f.error) bad.push(`sort-${k}.csv: ${String(f.error).slice(0, 160)}`); });
    }
    (S.liveExtraErr || []).forEach((x) => bad.push(`Could not load ${x}`));
    const SRC_NAMES = { gp: 'Orbit data (CelesTrak)', satcat: 'Object types (SATCAT)', swpc: 'Space weather (NOAA SWPC)' };
    const ls = get(S.health, 'live_sources') || {};
    let srcBad = 0;
    Object.keys(SRC_NAMES).forEach((k) => {
      const v = ls[k];
      if (v && v !== 'ok' && v !== 'disabled') {
        srcBad++;
        const det = get(S.health, `live_sources_detail.${k}`) || {};
        const err = det.error || get(det, 'bulk.cache.error') || get(det, 'bulk.error') || '';
        bad.push(`${SRC_NAMES[k]}: ${v}${err ? '. ' + String(err).slice(0, 140) : ''}`);
      }
    });
    if (S.healthErr) { srcBad++; bad.push(`Server health check failed: ${S.healthErr}`); }

    let cls = 'is-wait'; let label = 'Connecting';
    if (MOCK) { cls = 'is-mock'; label = 'Mock data'; }
    else if (!d && S.liveErr) { cls = 'is-off'; label = 'Offline'; warn.push('Live data unavailable. Check that the backend is running, then retry.'); }
    else if (d && (d.stale || S.liveErr)) {
      cls = 'is-stale'; label = 'Stale';
      warn.push(S.liveErr ? 'The API is unreachable. Showing the last data received.' : 'A source was unreachable. The server is serving its last cached copy.');
    } else if (d && (srcBad || (S.liveExtraErr || []).length)) { cls = 'is-stale'; label = 'Live'; warn.push('Screening data is live, but another input failed (details below).'); }
    else if (d) { cls = ''; label = 'Live'; }
    el.className = `status ${cls}`.trim();
    $('.status-label', el).textContent = label;
    $('.status-time', el).textContent = d ? `Data from ${fmtUTC(d.data_as_of, 'hm')}` : '';
    $('#status-pop').innerHTML = (warn.length ? `<p class="pop-warn">${warn.map(esc).join('<br>')}</p>` : '') +
      (rows.length ? `<dl>${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>` : '<p class="muted">Waiting for the first update.</p>') +
      (bad.length ? `<p class="pop-bad">${bad.map(esc).join('<br>')}</p>` : '');
    el.setAttribute('aria-label', `${label}${d ? `, data from ${fmtUTC(d.data_as_of, 'hm')}` : ''}`);
  }

  // ================================================================ router / views
  const VIEWS = ['live', 'evaluator', 'model', 'assess'];
  function route() {
    const h = (location.hash || '#live').slice(1).split('/');
    let v = h[0] || 'live';
    if (!VIEWS.includes(v)) v = 'live';
    setView(v);
    if (v === 'evaluator' && h[1] && num(Number(h[1])) != null && String(S.test.selected) !== h[1]) selectTestEvent(Number(h[1]));
    if (v === 'live' && h[1]) {
      const id = decodeURIComponent(h.slice(1).join('/'));
      if (id !== S.selectedId) { if (rowById(id)) selectEvent(id, { fly: true }); else S.pendingSelect = id; }
    }
  }
  function setView(v) {
    S.view = v;
    $('#app').setAttribute('data-view', v);
    $$('.nav a').forEach((a) => { if (a.dataset.view === v) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    $$('#stage > .view').forEach((el) => { el.hidden = el.dataset.view !== v; });
    $('#side-live').hidden = v !== 'live';
    $('#side-test').hidden = v !== 'evaluator';
    if (v === 'evaluator') ensureTestEvents();
    if (v === 'model') ensureMetrics();
    if (v === 'assess') ensureAssessForm();
    if (v === 'live' && G.viewer) { G.viewer.resize(); requestRender(); }
    setTimeout(resizePlots, 30);
  }
  function resizePlots() {
    if (!window.Plotly) return;
    $$('.js-plotly-plot').forEach((el) => { if (el.offsetParent) { try { Plotly.Plots.resize(el); } catch (e) { /* ignore */ } } });
  }

  // ================================================================ live polling
  async function loadLive(opts) {
    const seq = ++S.liveSeq;
    S.liveLoading = true;
    if (!S.live) renderFeed();
    try {
      // The risk-sorted top 500 is usually all Critical, so the Elevated and Needs-review queues are
      // loaded with two small server-side filtered requests, plus the closest approaches for the
      // insights, and merged (optional: older servers ignore the params and duplicates are dropped).
      const base = { window_h: S.windowH, sort: 'risk', points: 1 };
      const [main, ...extra] = await Promise.allSettled([
        api('/api/live/conjunctions', Object.assign({ limit: LIVE_LIMIT }, base)),
        api('/api/live/conjunctions', Object.assign({ limit: EXTRA_LIMIT, tier: 1 }, base)),
        api('/api/live/conjunctions', Object.assign({ limit: EXTRA_LIMIT, needs_review: 1 }, base)),
        api('/api/live/conjunctions', Object.assign({}, base, { limit: CLOSEST_LIMIT, sort: 'range' })),
      ]);
      if (main.status === 'rejected') throw main.reason;
      const d = main.value;
      if (seq !== S.liveSeq) return;
      if (!Array.isArray(d.conjunctions)) throw new ApiError('response has no conjunctions list', 200);
      d.conjunctions.forEach((r) => { if (!r.id) r.id = rowKey(r); });
      const seen = new Set(d.conjunctions.map((r) => r.id));
      d.n_main = d.conjunctions.length;
      const extraErr = [];
      extra.forEach((x, i) => {
        if (x.status !== 'fulfilled' || !x.value || !Array.isArray(x.value.conjunctions)) {
          extraErr.push(`${['the top Elevated', 'the needs-review', 'the closest'][i]} conjunctions (${x.status === 'rejected' ? (x.reason && x.reason.message) || x.reason : 'no conjunctions list'})`);
          return;
        }
        x.value.conjunctions.forEach((r) => {
          if (!r.id) r.id = rowKey(r);
          if (!seen.has(r.id)) { seen.add(r.id); d.conjunctions.push(r); }
        });
      });
      if (!d.warnings || !isNum(d.warnings.n_in_window)) d.warnings = computeWarnings(d.conjunctions, S.windowH);
      S.live = d;
      S.liveErr = null;
      S.liveExtraErr = extraErr;
      S.liveFetchedAt = Date.now();
      if (S.selectedId) {
        const r = d.conjunctions.find((x) => x.id === S.selectedId);
        if (r) S.selectedRow = r;
      }
    } catch (e) {
      if (seq !== S.liveSeq) return;
      S.liveErr = e.message || String(e);
      if (!opts || !opts.quiet) toast(S.live ? `Live data unavailable. Showing data from ${fmtUTC(S.live.data_as_of, 'hm')}.` : `Live data unavailable: ${S.liveErr}.`, true);
    } finally {
      if (seq === S.liveSeq) S.liveLoading = false;
    }
    renderLiveAll();
    if (S.pendingSelect && S.live) {
      const id = S.pendingSelect;
      S.pendingSelect = null;
      if (rowById(id)) selectEvent(id, { fly: true });
      else toast('That conjunction is not in the current window. Try a longer window.', true);
    }
  }
  function renderLiveAll() {
    renderStatus();
    renderWarnings();
    renderInsights();
    renderFeed();
    renderLegend();
    updatePoints();
    if (S.selectedId && $('#detail').classList.contains('open')) renderDetail();
  }
  async function loadSpaceWeather() {
    try { S.sw = await api('/api/live/spaceweather'); S.swErr = null; } catch (e) { S.swErr = e.message; }
    renderStatus();
  }
  async function loadHealth() {
    try { S.health = await api('/api/health'); S.healthErr = null; } catch (e) { S.healthErr = e.message; }
    renderStatus();
  }

  // ================================================================ sidebar 1: warning for the selected window
  const winText = () => `the next ${S.windowH}\u00a0hours`;
  const fmtInt = (v) => (isNum(num(v)) ? num(v).toLocaleString('en-US') : '–');
  const hourText = (h) => `in ${h}–${h + 1} h`;
  function renderLiveErr() {
    const el = $('#live-err');
    if (!el) return;
    if (!S.liveErr) { el.hidden = true; el.innerHTML = ''; return; }
    el.hidden = false;
    el.innerHTML = S.live
      ? `<b>Live data unavailable.</b> Showing data from ${esc(fmtUTC(S.live.data_as_of, 'hm'))}. <button type="button" class="btn sm" data-act="retry-live">Retry</button>`
      : `<b>Live data unavailable.</b> ${esc(S.liveErr)}. <button type="button" class="btn sm" data-act="retry-live">Retry</button>`;
  }
  const STAT_TIPS = {
    'Within 5 km': 'Miss distance up to 5 km: Elevated or worse on distance alone',
    Critical: 'Pc of 1e-4 or more, or miss distance under 1 km',
    Elevated: 'Pc from 1e-6 to 1e-4, or miss distance from 1 to 5 km',
  };
  function renderWarnings() {
    const w = get(S.live, 'warnings') || null;
    const head = $('#warn-head');
    if (!w) {
      head.textContent = S.liveErr && !S.live ? 'No conjunction data yet.' : 'Loading conjunctions…';
      $('#warn-stats').innerHTML = '';
    } else {
      const n1 = num(w.n_within_1km);
      head.innerHTML = n1 === 0 ? `No conjunctions pass within 1 km in ${esc(winText())}.`
        : `<span class="n">${fmtInt(n1)}</span> conjunction${n1 === 1 ? ' passes' : 's pass'} within 1&nbsp;km in ${esc(winText())}.`;
      $('#warn-stats').innerHTML = [['Within 5 km', w.n_within_5km, ''], ['Critical', w.n_critical, 'c2'], ['Elevated', w.n_elevated, 'c1']]
        .map(([l, v, c]) => `<div title="${esc(STAT_TIPS[l])}"><dt>${l}</dt><dd class="${c}">${fmtInt(v)}</dd></div>`).join('');
    }
    renderHours(w);
  }
  // Critical conjunctions per hour of the window; a bar filters the risk list to that hour.
  function renderHours(w) {
    const el = $('#hours');
    const axis = $('#hours-axis');
    const peak = $('#hours-peak');
    if (!w || !Array.isArray(w.by_hour) || !w.by_hour.length) {
      el.innerHTML = `<span class="state">${S.live ? 'No hourly breakdown returned.' : ''}</span>`;
      axis.innerHTML = ''; peak.textContent = '';
      return;
    }
    const bins = w.by_hour.map((b) => ({ h: num(b.hour), c: num(b.critical) || 0, e: num(b.elevated) || 0 })).filter((b) => b.h != null);
    const max = Math.max(1, ...bins.map((b) => b.c));
    const focusH = S.hourFilter != null ? S.hourFilter : bins[0].h;
    peak.textContent = `Peak ${max.toLocaleString('en-US')}`;
    el.classList.toggle('filtered', S.hourFilter != null);
    el.innerHTML = bins.map((b) => {
      const tip = `TCA ${hourText(b.h)}: ${b.c.toLocaleString('en-US')} Critical, ${b.e.toLocaleString('en-US')} Elevated`;
      return `<button type="button" data-hour="${b.h}" class="${S.hourFilter === b.h ? 'on' : ''}" tabindex="${b.h === focusH ? 0 : -1}" aria-pressed="${S.hourFilter === b.h}" aria-label="${esc(tip)}" title="${esc(tip)}"><i style="height:${Math.max(4, 100 * b.c / max).toFixed(1)}%"></i></button>`;
    }).join('');
    const W = bins.length;
    const step = W <= 12 ? W / 2 : (W <= 24 ? 6 : (W <= 48 ? 12 : 24));
    const ticks = [];
    for (let h = 0; h <= W + 1e-9; h += step) ticks.push(h === 0 ? 'Now' : `+${Math.round(h)} h`);
    axis.innerHTML = ticks.map((t) => `<span>${t}</span>`).join('');
  }
  function setHourFilter(h) {
    S.hourFilter = h == null || S.hourFilter === h ? null : h;
    renderHours(get(S.live, 'warnings'));
    renderFeed();
  }

  // ================================================================ sidebar 2: insights (plain sentences)
  function renderInsights() {
    const el = $('#insights');
    if (!el) return;
    if (!S.live) { el.innerHTML = `<li class="empty">${S.liveErr ? 'Nothing to summarise until live data loads.' : 'Waiting for data…'}</li>`; return; }
    const w = get(S.live, 'warnings') || {};
    const rows = liveRows().filter((r) => num(r.hours_to_tca) >= 0);
    if (!rows.length) { el.innerHTML = `<li class="empty">No conjunctions in ${esc(winText())}. Try a longer window.</li>`; return; }
    const total = (num(w.n_in_window) || 0) + (num(w.n_recent_past) || 0);
    const truncated = total > liveRows().length;
    const link = (r) => `<a href="#live/${esc(encodeURIComponent(r.id))}" data-sel="${esc(r.id)}">${esc(get(r, 'sat1.name', '?'))} and ${esc(get(r, 'sat2.name', '?'))}</a>`;
    const items = [];
    const closest = sortRows(rows, 'md')[0];
    if (closest && num(closest.miss_distance) != null) {
      items.push(`Closest approach: ${link(closest)} pass ${esc(fmtDist(closest.miss_distance))} apart at ${esc(fmtWhen(closest.tca))}, ${esc(countdown(closest.tca))}.`);
    }
    const mx = (r) => firstNum(r.max_risk_estimate, isNum(r.max_prob) && r.max_prob > 0 ? Math.log10(r.max_prob) : null, -31);
    const byPc = rows.slice().sort((a, b) => mx(b) - mx(a))[0];
    if (byPc && mx(byPc) > -29.9) items.push(`Highest ${truncated ? 'listed ' : ''}collision probability: ${link(byPc)}, with Pc up to ${esc(maxPcText(byPc))}.`);
    if (isNum(w.n_critical_by_pc) && w.n_critical_by_pc > 0) {
      items.push(`${fmtInt(w.n_critical_by_pc)} conjunction${w.n_critical_by_pc === 1 ? ' is' : 's are'} Critical on collision probability alone, with a miss distance over 1\u00a0km.`);
    }
    if (isNum(w.n_debris_or_rb) && num(w.n_in_window) > 0) {
      items.push(`${esc(pct(w.n_debris_or_rb / w.n_in_window))} of conjunctions involve debris or a rocket body.`);
    } else {
      const n1 = rows.filter((r) => num(r.miss_distance) < 1000);
      const deb = n1.filter((r) => /DEBRIS|ROCKET/.test(`${get(r, 'sat2.type', '')} ${get(r, 'sat1.type', '')}`)).length;
      if (n1.length) items.push(`${deb} of ${n1.length} listed approaches under 1 km involve debris or a rocket body.`);
    }
    const top = Array.isArray(w.top_primaries) ? w.top_primaries[0] : null;
    if (top && top.name) {
      items.push(`<a href="#" data-q="${esc(top.norad != null ? top.norad : top.name)}" title="Show only conjunctions involving ${esc(top.name)}">${esc(top.name)}</a> is involved in the most conjunctions (${fmtInt(top.count)}).`);
    }
    const bins = Array.isArray(w.by_hour) ? w.by_hour.filter((b) => num(b.hour) != null) : [];
    if (bins.length > 1) {
      const busy = bins.slice().sort((a, b) => (num(b.critical) || 0) - (num(a.critical) || 0))[0];
      if (num(busy.critical) > 0) items.push(`Busiest hour: <a href="#" data-hour="${num(busy.hour)}">TCA ${hourText(num(busy.hour))}</a>, with ${fmtInt(busy.critical)} Critical.`);
    }
    el.innerHTML = items.slice(0, 5).map((i) => `<li>${i}</li>`).join('');
  }

  // ================================================================ sidebar 3: risk list
  const isReview = (r) => get(r, 'assessment.needs_review') === true;
  function tierFilter(rows, f) {
    if (f === 'critical') return rows.filter((r) => tierOf(r) === 2);
    if (f === 'elevated') return rows.filter((r) => tierOf(r) === 1);
    if (f === 'review') return rows.filter(isReview);
    return rows;
  }
  function feedRows() {
    let rows = liveRows();
    if (S.search) rows = rows.filter((r) => matchesSearch(r, S.search));
    if (S.hourFilter != null) rows = rows.filter((r) => Math.floor(num(r.hours_to_tca)) === S.hourFilter);
    return { rows: sortRows(tierFilter(rows, S.feedFilter), S.feedSort) };
  }
  const FILTER_NAME = { critical: 'Critical', elevated: 'Elevated', review: 'needs-review', all: '' };
  const chipHtml = (text, act, label) => `<span class="chip"><span>${esc(text)}</span><button type="button" data-act="${act}" aria-label="${esc(label)}">×</button></span>`;
  // "Showing the 500 highest-risk of 1,226": only when the API limit cut the list
  function truncText() {
    const W = get(S.live, 'warnings') || {};
    const totals = { critical: W.n_critical, elevated: W.n_elevated, review: W.n_needs_review, all: (num(W.n_in_window) || 0) + (num(W.n_recent_past) || 0) };
    const total = num(totals[S.feedFilter]);
    const loaded = tierFilter(liveRows(), S.feedFilter).length;
    return total != null && total > loaded ? `Showing the ${loaded.toLocaleString('en-US')} highest-risk of ${total.toLocaleString('en-US')}` : '';
  }
  function renderFeed() {
    const list = $('#feed');
    if (!list) return;
    renderLiveErr();
    const { rows } = feedRows();
    $('#feed-count').textContent = S.live ? rows.length.toLocaleString('en-US') : '';
    const meta = [];
    if (S.hourFilter != null) meta.push(chipHtml(`TCA ${hourText(S.hourFilter)}`, 'clear-hour', 'Clear hour filter'));
    if (S.search) meta.push(chipHtml(`Matching “${S.search}”`, 'clear-search', 'Clear search'));
    if (S.live) { const t = truncText(); if (t) meta.push(`<span>${esc(t)}</span>`); }
    const extraErr = S.liveExtraErr || [];
    if (extraErr.length) meta.push(`<span class="warn-line" title="${esc('Could not load ' + extraErr.join('; '))}">Some rows did not load. <button type="button" class="btn sm" data-act="retry-live">Retry</button></span>`);
    $('#feed-meta').innerHTML = meta.join('');

    if (!S.live) {
      list.innerHTML = S.liveErr
        ? `<li class="state err-state"><b>Live data unavailable.</b> ${esc(S.liveErr)}. <button type="button" class="btn sm" data-act="retry-live">Retry</button></li>`
        : '<li class="skel"></li><li class="skel"></li><li class="skel"></li><li class="skel"></li>';
      return;
    }
    if (!rows.length) {
      const W = get(S.live, 'warnings') || {};
      const inWin = { critical: W.n_critical, elevated: W.n_elevated, review: W.n_needs_review, all: W.n_in_window }[S.feedFilter];
      const missing = !S.search && S.hourFilter == null && num(inWin) > 0;
      const fname = FILTER_NAME[S.feedFilter];
      let why;
      if (S.search) why = `Nothing in the list matches “${esc(S.search)}”. <button type="button" class="btn sm" data-act="clear-search">Clear search</button>`;
      else if (S.hourFilter != null) why = `No ${fname ? fname + ' ' : ''}conjunctions loaded for TCA ${hourText(S.hourFilter)}. <button type="button" class="btn sm" data-act="clear-hour">Show all hours</button>`;
      else if (missing) why = `${fmtInt(inWin)} in the window, but none loaded. <button type="button" class="btn sm" data-act="retry-live">Retry</button>`;
      else if (S.feedFilter === 'critical') why = `No Critical conjunctions in ${esc(winText())}. <button type="button" class="btn sm" data-act="feed-all">Show all</button>`;
      else if (S.feedFilter === 'review') why = 'Nothing needs review. <button type="button" class="btn sm" data-act="feed-all">Show all</button>';
      else if (S.feedFilter === 'elevated') why = `No Elevated conjunctions in ${esc(winText())}. <button type="button" class="btn sm" data-act="feed-all">Show all</button>`;
      else why = 'No conjunctions in this window. Choose a longer window above.';
      list.innerHTML = `<li class="state${missing ? ' err-state' : ''}">${why}</li>`;
      return;
    }
    const decs = latestDecisions();
    list.innerHTML = rows.map((r) => rowHtml(r, decs[r.id])).join('');
  }
  function rowHtml(r, dec) {
    const a = r.assessment || {};
    const t = tierOf(r);
    const tc = TIER_CSS[t] || '';
    const tcaMs = parseT(r.tca);
    const soon = tcaMs != null && tcaMs - Date.now() < 6 * 3.6e6;
    const chk = pointCheck(r);
    const mismatch = chk.known && !chk.agrees;
    const tags = [];
    if (a.needs_review && S.feedFilter !== 'review') tags.push(reviewPill());
    if (mismatch) tags.push(`<span class="tag warn" title="${esc(`${chk.text}. ${MISMATCH_TIP}`)}">Orbit mismatch</span>`);
    else if (r.tle_status === 'missing') tags.push('<span class="tag" title="No element set found for one of the objects, so there is no collision point or flight path.">No orbit data</span>');
    if (dec) tags.push(decTag(dec.action));
    const mx = num(r.max_risk_estimate) != null || isNum(r.max_prob);
    const label = `${TIERS[t] || 'No tier'}: ${pairText(r)}, miss ${fmtDist(r.miss_distance)}, closest approach ${countdown(r.tca)}${a.needs_review ? ', needs review' : ''}${mismatch ? ', orbit mismatch' : ''}`;
    return `<li><div class="item ${tc}${S.selectedId === r.id ? ' sel' : ''}" role="button" tabindex="0" data-sel="${esc(r.id)}" title="${esc(pairText(r))}" aria-label="${esc(label)}">
      <div class="it1"><i class="dot ${tc}"></i><span class="it-name">${pairHtml(get(r, 'sat1.name'), get(r, 'sat2.name'))}</span><span class="it-time${soon ? ' soon' : ''}" data-tca="${esc(r.tca)}">${esc(countdown(r.tca))}</span></div>
      <div class="it2"><span>Miss <b>${esc(fmtDist(r.miss_distance))}</b></span>${mx ? `<span title="Screening maximum collision probability">Pc ≤ <b>${esc(maxPcText(r))}</b></span>` : ''}${tags.join('')}</div>
    </div></li>`;
  }

  // ================================================================ map legend
  function renderLegend() {
    const rows = liveRows();
    $('#lg-safe').hidden = !rows.some((r) => tierOf(r) === 0 && r.tca_point);
    const el = $('#lg-count');
    if (!el) return;
    el.title = '';
    if (!S.live) { el.textContent = ''; return; }
    if (!rows.some((r) => 'tle_status' in r || 'tca_point' in r)) { el.textContent = 'This server does not send collision points.'; return; }
    const ok = rows.filter((r) => r.tca_point && isNum(r.tca_point.lat)).length;
    const pend = rows.filter((r) => r.tle_status === 'pending').length;
    const miss = rows.filter((r) => r.tle_status === 'missing').length;
    const far = rows.filter((r) => !r.tca_point && pointCheck(r).known && !pointCheck(r).agrees).length;
    el.textContent = `${ok.toLocaleString('en-US')} of ${rows.length.toLocaleString('en-US')} listed conjunctions are on the map.`;
    el.title = [pend ? `${pend}: no orbit data yet (fetched in the background; the map updates every minute)` : '',
      miss ? `${miss}: no orbit data` : '',
      far ? `${far}: not drawn, because our orbit data puts the objects more than 100 km apart` : ''].filter(Boolean).join('\n');
  }

  // ================================================================ Plotly helpers
  function plotlyBase(extra) {
    return Object.assign({
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: FONT, size: 12, color: COLORS.muted },
      margin: { l: 48, r: 12, t: 10, b: 36 },
      hoverlabel: { bgcolor: COLORS.panel, bordercolor: COLORS.line, font: { color: COLORS.text, family: FONT, size: 12 } },
      xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.line, linecolor: COLORS.line, tickcolor: COLORS.line },
      yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.line, linecolor: COLORS.line, tickcolor: COLORS.line },
      legend: { orientation: 'h', x: 0, y: 1.14, font: { size: 12, color: COLORS.text } },
    }, extra || {});
  }
  const PCFG = { displayModeBar: false, responsive: true };
  function plotReact(el, data, layout, cfg) {
    if (!el) return;
    if (!el.classList.contains('js-plotly-plot')) el.innerHTML = '';
    Array.from(el.children).forEach((n) => { if (n.classList && n.classList.contains('state')) n.remove(); });
    Plotly.react(el, data, layout, cfg || PCFG);
  }
  function noPlotly(el) { el.innerHTML = '<div class="state">Chart unavailable: Plotly did not load.</div>'; }

  // ================================================================ Cesium globe
  const G = { viewer: null, ptsDS: null, selDS: null, ptEntities: new Map(), cartoLayer: null, tileErrors: 0, fellBack: false,
    hoverId: null, sel: null, lastTick: 0 };

  function requestRender() { if (G.viewer) G.viewer.scene.requestRender(); }

  function globeFallback(msg) {
    const fb = $('#globe-fallback');
    fb.hidden = false;
    fb.innerHTML = `<div><b>Globe unavailable</b>${esc(msg)} The list, details and decisions still work.</div>`;
  }

  function initGlobe() {
    if (!window.Cesium) { globeFallback('CesiumJS did not load from the CDN (offline or blocked).'); return; }
    const C = window.Cesium;
    try {
      const bm = BASEMAPS[BASEMAP] || BASEMAPS.esri;
      const carto = new C.UrlTemplateImageryProvider({
        url: bm.url, subdomains: bm.subdomains, credit: bm.credit, maximumLevel: 8,
      });
      G.cartoLayer = new C.ImageryLayer(carto);
      const viewer = new C.Viewer('globe', {
        baseLayer: G.cartoLayer,
        baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: true,
        navigationHelpButton: false, animation: false, timeline: false, fullscreenButton: false,
        infoBox: false, selectionIndicator: false, vrButton: false, projectionPicker: false,
        skyBox: false, shouldAnimate: false,
        creditContainer: $('#credits'),
        requestRenderMode: true, maximumRenderTimeChange: Infinity,
        contextOptions: { webgl: { alpha: false } },
      });
      G.viewer = viewer;
      const sc = viewer.scene;
      sc.backgroundColor = C.Color.fromCssColorString(COLORS.bg);
      sc.globe.baseColor = C.Color.fromCssColorString(COLORS.panel);
      sc.globe.showGroundAtmosphere = false;
      sc.globe.enableLighting = false;
      sc.fog.enabled = false;
      if (sc.sun) sc.sun.show = false;
      if (sc.moon) sc.moon.show = false;
      if (sc.skyAtmosphere) { sc.skyAtmosphere.show = true; sc.skyAtmosphere.brightnessShift = -0.35; }
      sc.screenSpaceCameraController.minimumZoomDistance = 2.0e5;
      sc.screenSpaceCameraController.maximumZoomDistance = 1.2e8;
      viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(C.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
      viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(100, 12, 1.75e7) });

      if (bm.ref && QS.get('placenames') === '1') {   // place-name overlay is off by default (clutter)
        try {
          const ref = viewer.imageryLayers.addImageryProvider(new C.UrlTemplateImageryProvider({ url: bm.ref, maximumLevel: 8, credit: bm.credit }));
          ref.alpha = 0.55;
          G.refLayer = ref;
        } catch (e) { /* labels are optional */ }
      }
      if (BASEMAP === 'natural') fallbackImagery('requested');
      else {
        carto.errorEvent.addEventListener(() => { G.tileErrors++; if (G.tileErrors > 12) fallbackImagery('tile errors'); });
        const probe = new Image();
        const probeTimer = setTimeout(() => fallbackImagery('tile timeout'), 9000);
        probe.onload = () => clearTimeout(probeTimer);
        probe.onerror = () => { clearTimeout(probeTimer); fallbackImagery('tile probe failed'); };
        probe.crossOrigin = 'anonymous';
        probe.src = bm.probe;
      }

      G.ptsDS = new C.CustomDataSource('collision-points');
      G.selDS = new C.CustomDataSource('selection');
      viewer.dataSources.add(G.ptsDS);
      viewer.dataSources.add(G.selDS);

      const handler = new C.ScreenSpaceEventHandler(sc.canvas);
      let pending = null; let raf = 0;
      handler.setInputAction((m) => {
        pending = m.endPosition;
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = 0; hoverAt(pending); });
      }, C.ScreenSpaceEventType.MOUSE_MOVE);
      handler.setInputAction((m) => {
        const id = pickId(m.position);
        if (id) selectEvent(id, { fly: true });
      }, C.ScreenSpaceEventType.LEFT_CLICK);
      sc.canvas.addEventListener('mouseleave', () => hideTip());

      viewer.clock.onTick.addEventListener(onClockTick);
      viewer.camera.moveEnd.addEventListener(() => { if (S.layers.labels && !S.selectedId) updatePoints(); });
      sc.morphComplete.addEventListener(() => { requestRender(); });
      updatePoints();
    } catch (e) {
      console.error(e);
      G.viewer = null;
      globeFallback(`The browser could not start WebGL (${e.message || e}).`);
    }
  }
  function fallbackImagery(why) {
    if (G.fellBack || !G.viewer) return;
    G.fellBack = true;
    const C = window.Cesium;
    try {
      const layer = C.ImageryLayer.fromProviderAsync(
        C.TileMapServiceImageryProvider.fromUrl(C.buildModuleUrl('Assets/Textures/NaturalEarthII')), { brightness: 0.35 });
      G.viewer.imageryLayers.remove(G.cartoLayer, true);
      if (G.refLayer) { G.viewer.imageryLayers.remove(G.refLayer, true); G.refLayer = null; }
      G.viewer.imageryLayers.add(layer, 0);
      console.warn('CARTO tiles unavailable, using bundled Natural Earth II:', why);
      requestRender();
    } catch (e) { console.warn('imagery fallback failed', e); }
  }
  function pickId(pos) {
    if (!G.viewer || !pos) return null;
    let picked;
    try { picked = G.viewer.scene.pick(pos); } catch (e) { return null; }
    const ent = picked && picked.id;
    if (!ent || !ent.id || typeof ent.id !== 'string') return null;
    if (ent.id.startsWith('pt:')) return ent.id.slice(3);
    if (ent.id.startsWith('sel:') && S.selectedId) return S.selectedId;
    return null;
  }
  function hoverAt(pos) {
    const id = pickId(pos);
    const canvas = G.viewer && G.viewer.scene.canvas;
    if (canvas) canvas.style.cursor = id ? 'pointer' : '';
    if (!id) { hideTip(); return; }
    const r = rowById(id);
    if (!r) { hideTip(); return; }
    const tip = $('#map-tip');
    const a = r.assessment || {};
    const chk = pointCheck(r);
    tip.innerHTML = `<div>${tierPill(tierOf(r))}${a.needs_review ? ' ' + reviewPill() : ''}</div>
      <div class="tt">${pairHtml(get(r, 'sat1.name'), get(r, 'sat2.name'))}</div>
      <div class="tm">Miss <b>${esc(fmtDist(r.miss_distance))}</b>&emsp;Pc ≤ <b>${esc(maxPcText(r))}</b></div>
      <div class="tm">TCA <b>${esc(fmtWhen(r.tca))}</b>, ${esc(countdown(r.tca))}</div>
      ${chk.known && !chk.agrees ? `<div class="tw">Orbit mismatch: ${esc(chk.text)}.</div>` : ''}`;
    tip.hidden = false;
    const box = $('#view-live').getBoundingClientRect();
    const tw = tip.offsetWidth; const th = tip.offsetHeight;
    let x = pos.x + 14; let y = pos.y + 14;
    if (x + tw > box.width - 8) x = pos.x - tw - 14;
    if (y + th > box.height - 8) y = pos.y - th - 14;
    tip.style.left = `${Math.max(4, x)}px`;
    tip.style.top = `${Math.max(4, y)}px`;
  }
  function hideTip() { const t = $('#map-tip'); if (t) t.hidden = true; }

  function pointVisible(r) {
    const t = tierOf(r);
    if (t == null) return false;
    if (t === 1 && !S.layers.elev) return false;
    return true;
  }
  // 7-11 px by the screening maximum Pc (1e-10 -> 7 px, 1e-2 and above -> 11 px)
  function pointSize(r) {
    const lp = firstNum(r.max_risk_estimate, get(r, 'assessment.log_pc'));
    return Math.round(7 + 4 * clamp(((lp == null ? -12 : lp) + 10) / 8, 0, 1));
  }
  const labelText = (r) => pairText(r);
  // compact label on the selected collision point: "TCA 10:53:32 UTC, 60 m" (our propagation when known)
  function selLabel(r) {
    const t = S.track && S.selectedId === r.id ? S.track : null;
    const md = firstNum(t && t.min_range_m, get(r, 'sgp4_check.min_range_m'),
      r.tca_point && num(r.tca_point.range_km) != null ? r.tca_point.range_km * 1000 : null, r.miss_distance);
    return `TCA ${fmtUTC((t && t.tca_model) || r.tca, false)}, ${fmtDist(md)}`;
  }
  function labelStyle(C, selected) {
    return {
      font: `${selected ? 600 : 500} 12px ${FONT}`, fillColor: C.Color.fromCssColorString(COLORS.text),
      showBackground: true, backgroundColor: C.Color.fromCssColorString(COLORS.panel).withAlpha(0.92),
      backgroundPadding: new C.Cartesian2(6, 3), pixelOffset: new C.Cartesian2(0, selected ? -14 : -11),
      horizontalOrigin: C.HorizontalOrigin.CENTER, verticalOrigin: C.VerticalOrigin.BOTTOM,
    };
  }

  // Collision points are filled dots: colour = tier, size = risk, 1 px dark outline. A faded dot
  // (35% alpha) means our orbit data does not reproduce the screening. While an event is selected
  // (or a search is active) the other dots dim to 40%.
  function updatePoints() {
    if (!G.viewer || !G.ptsDS) return;
    const C = window.Cesium;
    const rows = liveRows().filter((r) => r.tca_point && isNum(r.tca_point.lat) && isNum(r.tca_point.lon));
    const keep = new Set();
    const anySel = !!S.selectedId;
    const q = S.search;
    const hits = q ? rows.filter((r) => matchesSearch(r, q)) : [];
    const labelled = new Set(q ? (hits.length <= 8 ? hits.map((r) => r.id) : [])
      : sortRows(rows.filter((r) => tierOf(r) === 2), 'risk').slice(0, 5).map((r) => r.id));
    const outline = C.Color.fromCssColorString(COLORS.outline);
    const sc = G.viewer.scene;
    const occluder = sc.mode === C.SceneMode.SCENE3D ? new C.EllipsoidalOccluder(C.Ellipsoid.WGS84, G.viewer.camera.positionWC) : null;
    const ents = G.ptsDS.entities;
    ents.suspendEvents();
    rows.forEach((r) => {
      const key = `pt:${r.id}`;
      keep.add(key);
      const t = tierOf(r) || 0;
      const color = C.Color.fromCssColorString(TIER_COLOR[t]);
      const isSel = S.selectedId === r.id;
      const chk = pointCheck(r);
      const faded = chk.known && !chk.agrees;
      let alpha = faded ? 0.35 : 1;
      if (!isSel && (anySel || (q && !matchesSearch(r, q)))) alpha *= 0.4;
      const pos = C.Cartesian3.fromDegrees(r.tca_point.lon, r.tca_point.lat, Math.max(num(r.tca_point.alt_km) || 0, 0) * 1000);
      const point = {
        pixelSize: isSel ? 14 : pointSize(r),
        color: color.withAlpha(alpha),
        outlineColor: isSel ? C.Color.WHITE : outline.withAlpha(alpha),
        outlineWidth: isSel ? 2 : 1,
        disableDepthTestDistance: isSel ? Number.POSITIVE_INFINITY : 0,
      };
      const showLabel = isSel || (S.layers.labels && !anySel && labelled.has(r.id) && (!occluder || occluder.isPointVisible(pos)));
      const text = isSel ? selLabel(r) : labelText(r);
      let e = G.ptEntities.get(key);
      if (!e) {
        e = ents.add({ id: key, position: pos, point,
          label: Object.assign(labelStyle(C, isSel), { text, show: showLabel, disableDepthTestDistance: Number.POSITIVE_INFINITY }) });
        G.ptEntities.set(key, e);
      } else {
        e.position = pos;
        Object.keys(point).forEach((k) => { e.point[k] = point[k]; });
        const ls = labelStyle(C, isSel);
        e.label.font = ls.font;
        e.label.pixelOffset = ls.pixelOffset;
        e.label.text = text;
        e.label.show = showLabel;
      }
      e.show = pointVisible(r) || isSel;
    });
    for (const [key, e] of G.ptEntities) {
      if (!keep.has(key)) { ents.remove(e); G.ptEntities.delete(key); }
    }
    ents.resumeEvents();
    requestRender();
  }

  // ---------------------------------------------------------------- selection: flight paths + collision marker
  function clearSelectionGraphics() {
    if (!G.viewer) return;
    G.selDS.entities.removeAll();
    G.sel = null;
    G.viewer.clock.shouldAnimate = false;
    G.viewer.scene.requestRenderMode = true;
    $('#playbar').hidden = true;
    requestRender();
  }
  function flyToRow(r) {
    if (!G.viewer || !r || !r.tca_point || !isNum(r.tca_point.lat) || !isNum(r.tca_point.lon)) return;
    const C = window.Cesium;
    const p = r.tca_point;
    const H = 3.2e6;
    const altM = clamp(num(p.alt_km) == null ? 500 : num(p.alt_km), 0, 50000) * 1000;   // never inside the Earth
    let dest = C.Cartesian3.fromDegrees(p.lon, p.lat, altM + H);
    const det = $('#detail');
    const drawerW = det.classList.contains('open') && window.innerWidth > 900 ? det.offsetWidth : 0;
    const canvasW = G.viewer.scene.canvas.clientWidth || 1;
    if (drawerW && G.viewer.scene.mode === C.SceneMode.SCENE3D) {
      // shift the camera east so the collision point sits in the middle of the area the drawer leaves visible
      const target = C.Cartesian3.fromDegrees(p.lon, p.lat, altM);
      const fov = G.viewer.camera.frustum.fov || Math.PI / 3;
      const shift = 2 * H * Math.tan(fov / 2) * (drawerW / 2) / canvasW;
      const enu = C.Transforms.eastNorthUpToFixedFrame(target);
      const east = C.Matrix4.multiplyByPointAsVector(enu, new C.Cartesian3(1, 0, 0), new C.Cartesian3());
      const up = C.Matrix4.multiplyByPointAsVector(enu, new C.Cartesian3(0, 0, 1), new C.Cartesian3());
      dest = C.Cartesian3.add(target, C.Cartesian3.add(C.Cartesian3.multiplyByScalar(up, H, new C.Cartesian3()),
        C.Cartesian3.multiplyByScalar(east, shift, new C.Cartesian3()), new C.Cartesian3()), new C.Cartesian3());
    }
    G.viewer.camera.flyTo({ destination: dest, orientation: { heading: 0, pitch: -Math.PI / 2, roll: 0 }, duration: REDUCED_MOTION ? 0 : 1.4 });
  }
  function drawTrack(r, t) {
    if (!G.viewer) return;
    const C = window.Cesium;
    clearSelectionGraphics();
    const mk = (obj) => {
      const prop = new C.SampledPositionProperty();
      prop.setInterpolationOptions({ interpolationDegree: 5, interpolationAlgorithm: C.LagrangePolynomialApproximation });
      const positions = [];
      const times = [];
      (obj.ecef || []).forEach((p) => {
        if (!Array.isArray(p) || p.length < 4) return;
        const jd = C.JulianDate.fromIso8601(String(p[0]));
        const c3 = new C.Cartesian3(p[1], p[2], p[3]);
        prop.addSample(jd, c3);
        positions.push(c3);
        times.push(jd);
      });
      return { prop, positions, times };
    };
    const A = mk(t.a || {}); const B = mk(t.b || {});
    if (A.positions.length < 2 || B.positions.length < 2) throw new Error('track has too few samples');
    const colA = C.Color.fromCssColorString(COLORS.a); const colB = C.Color.fromCssColorString(COLORS.b);
    // flight paths keep a subtle glow; small moving markers show where each object is now
    const path = (id, pos, col) => G.selDS.entities.add({
      id,
      polyline: { positions: pos, width: 4, arcType: C.ArcType.NONE,
        material: new C.PolylineGlowMaterialProperty({ glowPower: 0.2, taperPower: 1.0, color: col }) },
    });
    path('sel:path-a', A.positions, colA);
    path('sel:path-b', B.positions, colB);
    const marker = (id, prop, col, name) => G.selDS.entities.add({
      id, position: prop,
      point: { pixelSize: 8, color: col, outlineColor: C.Color.fromCssColorString(COLORS.outline), outlineWidth: 1 },
      label: { text: name || '', font: `500 12px ${FONT}`, fillColor: col, showBackground: true,
        backgroundColor: C.Color.fromCssColorString(COLORS.panel).withAlpha(0.92), backgroundPadding: new C.Cartesian2(6, 3),
        pixelOffset: new C.Cartesian2(10, 0), horizontalOrigin: C.HorizontalOrigin.LEFT, verticalOrigin: C.VerticalOrigin.CENTER, show: S.layers.labels },
    });
    marker('sel:obj-a', A.prop, colA, get(t, 'a.name') || get(r, 'sat1.name'));
    marker('sel:obj-b', B.prop, colB, get(t, 'b.name') || get(r, 'sat2.name'));

    // The selected collision point is the row's own dot (restyled in updatePoints). Only when the row
    // has no point do we add one at our closest approach (midpoint of both objects).
    const tcaIso = t.tca_model || r.tca;
    const tcaJd = C.JulianDate.fromIso8601(String(tcaIso));
    const hasDot = r.tca_point && isNum(r.tca_point.lat) && isNum(r.tca_point.lon);
    if (!hasDot) {
      const pa = A.prop.getValue(tcaJd); const pb = B.prop.getValue(tcaJd);
      if (pa && pb) {
        const C2 = C.Color.fromCssColorString(TIER_COLOR[tierOf(r) || 0]);
        G.selDS.entities.add({
          id: 'sel:tca', position: C.Cartesian3.midpoint(pa, pb, new C.Cartesian3()),
          point: { pixelSize: 14, color: C2, outlineColor: C.Color.WHITE, outlineWidth: 2, disableDepthTestDistance: Number.POSITIVE_INFINITY },
          label: Object.assign(labelStyle(C, true), { text: selLabel(r), show: true, disableDepthTestDistance: Number.POSITIVE_INFINITY }),
        });
      }
    }
    updatePoints();   // refresh the selected dot's label with our closest approach

    const start = C.JulianDate.compare(A.times[0], B.times[0]) > 0 ? A.times[0] : B.times[0];
    const stop = C.JulianDate.compare(A.times[A.times.length - 1], B.times[B.times.length - 1]) < 0 ? A.times[A.times.length - 1] : B.times[B.times.length - 1];
    const clk = G.viewer.clock;
    clk.startTime = start.clone();
    clk.stopTime = stop.clone();
    let cur = C.JulianDate.addSeconds(tcaJd, -600, new C.JulianDate());
    if (C.JulianDate.compare(cur, start) < 0) cur = start.clone();
    clk.currentTime = cur;
    clk.clockRange = C.ClockRange.LOOP_STOP;
    clk.clockStep = C.ClockStep.SYSTEM_CLOCK_MULTIPLIER;
    clk.multiplier = G.speed || 60;
    clk.shouldAnimate = !REDUCED_MOTION;
    G.viewer.scene.requestRenderMode = false;
    G.sel = { start, stop, tcaJd, positions: A.positions.concat(B.positions), span: C.JulianDate.secondsDifference(stop, start) };
    $('#playbar').hidden = false;
    updatePlaybar(true);
  }
  function onClockTick(clock) {
    if (!G.sel) return;
    const now = performance.now();
    if (now - G.lastTick < 150) return;
    G.lastTick = now;
    updatePlaybar();
  }
  function updatePlaybar(force) {
    if (!G.sel || !G.viewer) return;
    const C = window.Cesium;
    const clk = G.viewer.clock;
    const dt = C.JulianDate.secondsDifference(clk.currentTime, G.sel.tcaJd);
    const sign = dt < 0 ? '−' : '+';
    const a = Math.abs(dt);
    const iso = C.JulianDate.toIso8601(clk.currentTime, 0);
    const pbt = $('#pb-time');
    pbt.textContent = `T${sign}${pad(Math.floor(a / 60))}:${pad(Math.floor(a % 60))}`;
    pbt.title = `${fmtUTC(iso, false)}; T is the time from closest approach`;
    const scrub = $('#pb-scrub');
    if (!G.scrubbing) scrub.value = String(Math.round(1000 * C.JulianDate.secondsDifference(clk.currentTime, G.sel.start) / Math.max(1, G.sel.span)));
    const play = $('#pb-play');
    play.innerHTML = clk.shouldAnimate ? '&#10074;&#10074;' : '&#9654;';
    play.setAttribute('aria-label', clk.shouldAnimate ? 'Pause' : 'Play');
    if (force) requestRender();
  }
  function fitPaths() {
    if (!G.viewer || !G.sel) return;
    const C = window.Cesium;
    const bs = C.BoundingSphere.fromPoints(G.sel.positions);
    G.viewer.camera.flyToBoundingSphere(bs, { duration: REDUCED_MOTION ? 0 : 1.2, offset: new C.HeadingPitchRange(0, -1.2, bs.radius * 3.2) });
  }
  function applyLayerVisibility() {
    if (!G.viewer) return;
    G.selDS.entities.values.forEach((e) => {
      if (e.id.startsWith('sel:obj') && e.label) e.label.show = S.layers.labels;
    });
    updatePoints();
  }

  // ================================================================ select / deselect a live event
  function selectEvent(id, opts) {
    opts = opts || {};
    const r = rowById(id);
    if (!r) return;
    try { history.replaceState(null, '', `#live/${encodeURIComponent(id)}`); } catch (e) { /* ignore */ }
    if (S.view !== 'live') setView('live');
    const same = S.selectedId === id;
    S.selectedId = id;
    S.selectedRow = r;
    $$('#feed .item.sel').forEach((c) => c.classList.remove('sel'));
    const card = $(`#feed .item[data-sel="${CSS.escape(id)}"]`);
    if (card) {
      card.classList.add('sel');
      if (opts.scroll !== false && window.innerWidth > 900) card.scrollIntoView({ block: 'nearest', behavior: REDUCED_MOTION ? 'auto' : 'smooth' });
    }
    // phones: the drawer is a bottom sheet, so bring the map into view above it
    if (window.innerWidth <= 900) $('#view-live').scrollIntoView({ block: 'start', behavior: REDUCED_MOTION ? 'auto' : 'smooth' });
    openDetail();
    updatePoints();
    if (opts.fly !== false) flyToRow(r);
    if (!same || !S.track) loadTrack(r);
    else renderDetail();
  }
  function deselect() {
    if (!S.selectedId) return;
    if (location.hash.startsWith('#live/')) { try { history.replaceState(null, '', '#live'); } catch (e) { /* ignore */ } }
    S.selectedId = null;
    S.selectedRow = null;
    S.track = null; S.trackErr = null; S.trackLoading = false; S.trackSeq++;
    $$('#feed .item.sel').forEach((c) => c.classList.remove('sel'));
    closeDetail();
    clearSelectionGraphics();
    updatePoints();
  }
  async function loadTrack(r) {
    const seq = ++S.trackSeq;
    S.track = null; S.trackErr = null; S.trackLoading = true;
    clearSelectionGraphics();
    renderDetail();
    const a = get(r, 'sat1.norad'); const b = get(r, 'sat2.norad');
    if (a == null || b == null || !r.tca) {
      S.trackLoading = false; S.trackErr = 'NORAD ids or closest-approach time missing'; renderDetail(); return;
    }
    try {
      const t = await api('/api/live/track', { a, b, tca: r.tca, span_min: 50 });
      if (seq !== S.trackSeq) return;
      S.track = t;
      try { drawTrack(r, t); if (!(r.tca_point && isNum(r.tca_point.lat))) fitPaths(); } catch (e) { console.warn(e); S.trackErr = `could not draw the paths (${e.message})`; }
    } catch (e) {
      if (seq !== S.trackSeq) return;
      S.trackErr = e.message || String(e);
    }
    S.trackLoading = false;
    renderDetail();
  }
  function openDetail() {
    const d = $('#detail');
    d.classList.add('open');
    d.setAttribute('aria-hidden', 'false');
    $('#view-live').classList.add('drawer-open');
  }
  function closeDetail() {
    const d = $('#detail');
    d.classList.remove('open');
    d.setAttribute('aria-hidden', 'true');
    $('#view-live').classList.remove('drawer-open');
    if (window.Plotly) { const c = $('#rtn-chart'); if (c) { try { Plotly.purge(c); } catch (e) { /* ignore */ } } }
  }

  // ================================================================ detail drawer
  const sentenceCase = (s) => { s = String(s || '').trim(); return s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : ''; };
  function objRow(o, cls, trackObj, tca) {
    o = o || {};
    const st = o.status == null ? '' : String(o.status);
    const tcaMs = parseT(tca);
    const ourMs = trackObj ? parseT(trackObj.tle_epoch) : null;
    const age = ourMs != null && tcaMs != null ? (tcaMs - ourMs) / 864e5 : num(o.tle_age_d);
    return `<div class="obj">
      <div class="obj-name"><i class="ln ${cls}" aria-hidden="true"></i>${esc(o.name || '?')}</div>
      <div class="obj-meta"><span>NORAD <b>${esc(o.norad != null ? o.norad : '?')}</b></span><span>${esc(o.type ? sentenceCase(o.type) : 'Type unknown')}</span><span>${esc(STATUS_TEXT[st] || 'Status unknown')}</span>${age != null ? `<span title="Days from the element set epoch to the closest approach">Orbit data <b>${esc(age.toFixed(1))} days</b> old</span>` : ''}</div>
      ${trackObj && trackObj.tle_stale ? `<div class="warn-line">Orbit data refresh failed; using the copy fetched ${esc(fmtUTC(trackObj.tle_fetched_at))}.</div>` : ''}
    </div>`;
  }
  function stackBar(p) {
    p = Array.isArray(p) ? p.map((x) => num(x) || 0) : [0, 0, 0];
    const seg = (i) => (p[i] > 0.001 ? `<div class="s${i}" style="flex:${p[i]}" title="P(${TIERS[i]}) = ${pct(p[i], 1)}"></div>` : '');
    return `<div class="stack" role="img" aria-label="P(Safe) ${pct(p[0])}, P(Elevated) ${pct(p[1])}, P(Critical) ${pct(p[2])}">${seg(0)}${seg(1)}${seg(2)}</div>
      <div class="stack-legend"><span><i class="dot t0"></i>Safe <b>${pct(p[0], 1)}</b></span><span><i class="dot t1"></i>Elevated <b>${pct(p[1], 1)}</b></span><span><i class="dot t2"></i>Critical <b>${pct(p[2], 1)}</b></span></div>`;
  }
  // reasons are API text, shown verbatim as a plain list
  function reasonsHtml(reasons) {
    if (!Array.isArray(reasons) || !reasons.length) return '<p class="muted">The model returned no reasons.</p>';
    return `<ul class="reasons">${reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`;
  }
  // one sentence, in the tier colour, built from the tier, the distance rule and the numbers
  function verdictText(r) {
    const a = r.assessment || {};
    const t = tierOf(r);
    const md = num(r.miss_distance);
    const mdt = num(a.md_tier);
    const mx = num(r.max_risk_estimate);
    if (t === 2) {
      if (mdt === 2 && md != null) return `Critical: the objects pass ${fmtDist(md)} apart, under the 1\u00a0km limit.`;
      if (mx != null && mx >= -4) return `Critical: collision probability up to ${pcText(mx)}, above the 1e-4 limit.`;
      return `Critical: the model gives a ${pct(a.p_pc_critical)} chance that the collision probability is 1e-4 or more.`;
    }
    if (t === 1) {
      if (mdt === 1 && md != null) return `Elevated: the objects pass ${fmtDist(md)} apart, inside the 5\u00a0km warning distance.`;
      if (mx != null && mx >= -6) return `Elevated: collision probability up to ${pcText(mx)}, above the 1e-6 limit.`;
      return `Elevated: the model gives a ${pct(a.p_pc_elevated)} chance that the collision probability is 1e-6 or more.`;
    }
    if (t === 0) return `Safe: the objects pass ${fmtDist(md)} apart and the collision probability stays under 1e-6.`;
    return 'No tier: the model returned no assessment for this conjunction.';
  }
  function renderDetail() {
    const el = $('#detail');
    const r = S.selectedRow;
    if (!el || !r) return;
    const a = r.assessment || {};
    const t = S.track;
    const tier = tierOf(r);
    const s1 = r.sat1 || {}; const s2 = r.sat2 || {};
    const keepScroll = el.scrollTop;
    const oldNote = $('.act-note', el);
    const prevNote = oldNote ? oldNote.value : '';
    const noteFocused = oldNote && document.activeElement === oldNote;
    const chk = pointCheck(r);
    const pi = pcInfo(r);
    const ours = firstNum(t && t.min_range_m, get(r, 'sgp4_check.min_range_m'));
    const tcaMs = parseT(r.tca);

    let trackHtml;
    if (S.trackLoading) trackHtml = '<div class="loading">Computing flight paths…</div>';
    else if (S.trackErr) trackHtml = `<p class="err-box"><b>Flight paths unavailable.</b> ${esc(S.trackErr)}. <button type="button" class="btn sm" data-act="retry-track">Retry</button></p>`;
    else if (t) trackHtml = `<div id="rtn-chart" class="rtn-chart" aria-label="Path of object B relative to object A, along-track against cross-track"></div>${t.mock_note ? `<p class="chart-note">${esc(t.mock_note)}.</p>` : ''}`;
    else trackHtml = '';
    const edgeWarn = t && t.min_at_edge ? '<p class="warn-line">The closest approach is at the edge of the computed window; the true minimum may lie outside it.</p>' : '';

    // our orbit data vs the screening: one line, only when they disagree
    let mismatch = '';
    if (chk.known && !chk.agrees) {
      const shifts = get(r, 'sgp4_check.epoch_shift_h') || [];
      const ourNewer = shifts.length ? shifts.every((x) => num(x) == null || x >= 0) : (t && parseT(get(t, 'a.tle_epoch')) != null);
      mismatch = `<p class="mismatch" title="${esc(chk.text + '. ' + MISMATCH_TIP)}">Our ${ourNewer ? 'newer ' : ''}orbit data puts them ${esc(fmtDist(chk.sgp))} apart; re-screen before acting.</p>`;
    }

    const M = get(S.live, 'model') || {};
    const swu = get(S.live, 'space_weather_used') || {};
    const prov = [r.provenance,
      S.live && `Assessed by the ${M.kind || 'screening model'}${M.trained === false ? ' (not trained: fallback estimates)' : ''}${M.trained_at ? `, trained ${fmtUTC(M.trained_at)}` : ''}, at ${fmtUTC(S.live.assessed_at)}`,
      S.live && `Space weather used: F10.7 ${fmtNum(swu.F10, 1)} sfu, Ap ${fmtNum(swu.AP, 1)}${swu.stale ? ' (cached copy)' : ''}`,
      t && t.propagator,
      t && get(t, 'a.elements_source') && `Orbit data, object A: ${get(t, 'a.elements_source')}`,
      t && get(t, 'b.elements_source') && `Orbit data, object B: ${get(t, 'b.elements_source')}`,
      r.tca_point && `Collision point: ${fixed(r.tca_point.lat, 2)}°, ${fixed(r.tca_point.lon, 2)}°, ${fmtNum(r.tca_point.alt_km, 0)} km altitude`,
      S.live && `Feed fetched ${fmtUTC(S.live.fetched_at)}${S.live.stale ? ' (cached copy)' : ''}`].filter(Boolean);

    el.innerHTML = `
      <div class="dt-head">
        <h2 class="dt-title">${pairHtml(s1.name, s2.name)}</h2>
        <button type="button" class="icon-btn" data-act="deselect" aria-label="Close (Esc)" title="Close (Esc)">&times;</button>
      </div>
      <p class="verdict ${TIER_CSS[tier] || 'nt'}">${esc(verdictText(r))}${a.needs_review ? '<span class="verdict-sub">Needs review: the uncertainty spans a tier boundary.</span>' : ''}</p>
      <div class="keys">
        <div><div class="k-l">Miss distance</div><div class="k-v">${esc(fmtDist(r.miss_distance))}</div><div class="k-s">${ours != null ? `${esc(fmtDist(ours))} with our orbit data` : 'From the screening'}</div></div>
        <div title="Screening maximum Pc: the worst case over covariance scaling"><div class="k-l">Collision risk</div><div class="k-v">${maxPcText(r) !== '–' ? `Pc ≤ ${esc(maxPcText(r))}` : '–'}</div><div class="k-s">${pi.informative && num(a.p_pc_critical) != null ? `Model: ${esc(pct(a.p_pc_critical))} chance of Pc ≥ 1e-4` : 'Screening maximum'}</div></div>
        <div><div class="k-l">Closest approach</div><div class="k-v">${esc(fmtUTC(r.tca, false))}</div><div class="k-s">${tcaMs != null && !sameUtcDay(tcaMs, Date.now()) ? `${esc(fmtUTC(r.tca).split(' ').slice(0, 2).join(' '))}, ` : ''}<span data-tca="${esc(r.tca)}">${esc(countdown(r.tca))}</span></div></div>
        <div><div class="k-l">Relative speed</div><div class="k-v">${esc(fmtSpeed(r.relative_speed))}</div><div class="k-s">&nbsp;</div></div>
      </div>
      ${pi.informative ? '' : `<p class="keys-note" title="${esc(pi.tip)}">Model estimate hidden: these inputs are outside the training data.</p>`}
      <div class="dt-sec"><h3>Why</h3>${reasonsHtml(a.reasons)}</div>
      <div class="dt-sec"><h3>Relative path</h3>${edgeWarn}${trackHtml}</div>
      <div class="dt-sec"><h3>Objects</h3><div class="objs">${objRow(s1, 'a', t && t.a, r.tca)}${objRow(s2, 'b', t && t.b, r.tca)}</div>${mismatch}</div>
      <p class="caveat">Based on public TLE screening without covariance. Confirm with CSpOC CDMs before manoeuvring.</p>
      <details class="prov"><summary>Data sources</summary><div>${prov.map(esc).join('<br>')}</div></details>
      <div class="dt-sec" style="margin-top:12px"><h3>Decision</h3>${actionsHtml('live', r.id, decisionsFor('live', r.id), true)}</div>`;
    el.scrollTop = keepScroll;
    const newNote = $('.act-note', el);
    if (newNote && prevNote) newNote.value = prevNote;
    if (newNote && noteFocused) newNote.focus({ preventScroll: true });
    if (t && !S.trackLoading && !S.trackErr) renderRtnChart($('#rtn-chart'), t, r);
  }

  function renderRtnChart(el, t, r) {
    if (!el) return;
    if (!window.Plotly) { noPlotly(el); return; }
    const rel = (t.relative_rtn || []).filter((p) => Array.isArray(p) && p.length >= 4);
    if (!rel.length) { el.innerHTML = '<div class="state">No relative path samples.</div>'; return; }
    let tca = Array.isArray(t.rtn_at_tca_m) ? t.rtn_at_tca_m : null;
    if (!tca) {
      let best = rel[0]; let bd = Infinity;
      rel.forEach((p) => { const d = Math.hypot(p[1], p[2], p[3]); if (d < bd) { bd = d; best = p; } });
      tca = [best[1], best[2], best[3]];
    }
    const km = (v) => v / 1000;
    const half = Math.max(7, 1.6 * Math.max(Math.abs(km(tca[1])), Math.abs(km(tca[2]))));
    const circ = (rad, dash, col) => ({ type: 'circle', xref: 'x', yref: 'y', x0: -rad, y0: -rad, x1: rad, y1: rad, line: { color: col, width: 1.2, dash } });
    const data = [
      { type: 'scatter', mode: 'lines', name: 'Object B', x: rel.map((p) => km(p[2])), y: rel.map((p) => km(p[3])),
        customdata: rel.map((p) => [p[0], Math.hypot(p[1], p[2], p[3])]), line: { color: COLORS.b, width: 2 },
        hovertemplate: '%{customdata[0]:.0f} s from TCA<br>Along-track %{x:.2f} km, cross-track %{y:.2f} km<br>Range %{customdata[1]:,.0f} m<extra></extra>' },
      { type: 'scatter', mode: 'markers', name: 'Object A', x: [0], y: [0], marker: { color: COLORS.a, size: 9, line: { color: COLORS.outline, width: 1 } }, hovertemplate: 'Object A<extra></extra>' },
      { type: 'scatter', mode: 'markers+text', name: 'Closest approach', x: [km(tca[1])], y: [km(tca[2])], text: [fmtDist(num(t.min_range_m) != null ? t.min_range_m : Math.hypot(tca[0], tca[1], tca[2]))],
        textposition: 'top right', textfont: { color: COLORS.text, size: 12, family: FONT },
        marker: { color: COLORS.text, size: 7, line: { color: COLORS.outline, width: 1 } },
        hovertemplate: `Closest approach<br>Along-track %{x:.3f} km, cross-track %{y:.3f} km<br>Radial ${esc(fmtDist(tca[0]))}<extra></extra>` },
    ];
    const layout = plotlyBase({
      margin: { l: 44, r: 8, t: 8, b: 38 }, showlegend: false,
      xaxis: { title: { text: 'Along-track (km)', font: { size: 12 } }, range: [-half, half], gridcolor: COLORS.grid, zerolinecolor: COLORS.line },
      yaxis: { title: { text: 'Cross-track (km)', font: { size: 12 } }, range: [-half, half], scaleanchor: 'x', scaleratio: 1, gridcolor: COLORS.grid, zerolinecolor: COLORS.line },
      // the 1 km and 5 km circles only mean something at a readable scale
      shapes: half <= 60 ? [circ(1, 'solid', COLORS.crit), circ(5, 'dash', COLORS.elev)] : [],
      annotations: half <= 60 ? [
        { x: -0.75, y: -0.75, text: '1 km', showarrow: false, font: { color: COLORS.crit, size: 11 }, xanchor: 'right', yanchor: 'top' },
        { x: 3.6, y: 3.6, text: '5 km', showarrow: false, font: { color: COLORS.elev, size: 11 }, xanchor: 'left' },
      ] : [],
    });
    plotReact(el, data, layout, Object.assign({}, PCFG, { scrollZoom: true }));
  }

  // ================================================================ analyst actions + decision log
  const ACTIONS = [
    { key: 'acknowledge', label: 'Acknowledge', cls: '' },
    { key: 'escalate', label: 'Escalate to flight dynamics', cls: 'esc' },
    { key: 'dismiss', label: 'Dismiss', cls: 'ghost' },
  ];
  const DEC_WORD = { acknowledge: 'Acknowledged', escalate: 'Escalated', dismiss: 'Dismissed' };
  const decTag = (action) => `<span class="dec ${esc(action)}" title="Latest analyst decision">${esc(DEC_WORD[action] || sentenceCase(action))}</span>`;
  function actionsHtml(view, id, decs, withNote) {
    return `<div class="actions" data-view="${esc(view)}" data-id="${esc(id)}">
      <textarea class="act-note" placeholder="Note: rationale, CDMs checked, who you told" aria-label="Decision note"></textarea>
      <label class="analyst">Your initials <input class="analyst-inline" maxlength="24" value="${esc(S.analyst)}" placeholder="e.g. GW" aria-label="Your initials"></label>
      <div class="act-row">${ACTIONS.map((a) => `<button type="button" class="btn sm ${a.cls}" data-decide="${a.key}">${esc(a.label)}</button>`).join('')}</div>
      ${withNote ? '<p class="act-note-line">You make the call; this tool only recommends.</p>' : ''}
      ${decs.length ? `<ul class="act-hist">${decs.slice(0, 5).map((d) => `<li><b>${esc(DEC_WORD[d.action] || d.action)}</b> by ${esc(d.analyst)}, ${esc(fmtUTC(d.ts))}${d.note ? `: ${esc(d.note)}` : ''}</li>`).join('')}</ul>` : ''}
    </div>`;
  }
  function loadLog() {
    let v = null;
    try { v = JSON.parse(localStorage.getItem('cw.decisions.v1') || '[]'); S.logPersist = true; } catch (e) { v = []; S.logPersist = false; }
    S.log = Array.isArray(v) ? v : [];
    S.analyst = lsGet('cw.analyst', '') || '';
  }
  function saveLog() {
    S.logPersist = lsSet('cw.decisions.v1', S.log);
    renderLogCount();
  }
  function renderLogCount() { const c = $('#log-count'); if (c) c.textContent = S.log.length; }
  function decisionsFor(view, id) { return S.log.filter((d) => d.view === view && String(d.event_id) === String(id)).sort((a, b) => (a.ts < b.ts ? 1 : -1)); }
  function latestDecisions() {
    const out = {};
    S.log.forEach((d) => { if (d.view === 'live' && (!out[d.event_id] || out[d.event_id].ts < d.ts)) out[d.event_id] = d; });
    return out;
  }
  function setAnalyst(v) {
    S.analyst = (v || '').trim();
    lsSet('cw.analyst', S.analyst);
    $$('.analyst-inline').forEach((i) => { if (i.value !== S.analyst) i.value = S.analyst; i.closest('.analyst').classList.toggle('need', !S.analyst); });
    const main = $('#analyst');
    if (main && main.value !== S.analyst) main.value = S.analyst;
  }
  function decide(box, action) {
    const view = box.dataset.view; const id = box.dataset.id;
    const initials = ($('.analyst-inline', box).value || '').trim();
    if (!initials) {
      const lab = $('.analyst', box); lab.classList.add('need'); $('.analyst-inline', box).focus();
      toast('Add your initials first. Every decision is logged with a name.', true);
      return;
    }
    setAnalyst(initials);
    const noteEl = $('.act-note', box);
    const note = (noteEl.value || '').trim();
    noteEl.value = '';
    const entry = { ts: new Date().toISOString(), analyst: initials, view, event_id: id, action, note };
    if (view === 'live') {
      const r = rowById(id) || {};
      const a = r.assessment || {};
      Object.assign(entry, {
        pair: `${get(r, 'sat1.name', '?')} x ${get(r, 'sat2.name', '?')}`, norad_a: get(r, 'sat1.norad'), norad_b: get(r, 'sat2.norad'),
        tca: r.tca, miss_distance_m: r.miss_distance, model_tier: TIERS[num(a.tier)], needs_review: !!a.needs_review,
        log_pc: a.log_pc, p_critical: ptier(r, 2), model_min_range_m: S.track ? S.track.min_range_m : null,
        data_as_of: get(S.live, 'data_as_of'), source: r.provenance || get(S.live, 'source'),
      });
    } else if (view === 'evaluator') {
      const d = S.test.detail; const ev = (S.test.events || []).find((e) => String(e.event_id) === String(id)) || {};
      const c = d && d.cdms ? d.cdms[S.test.cdmIdx != null ? S.test.cdmIdx : d.cdms.length - 1] : null;
      const a = (c && c.assessment) || ev.assessment || {};
      Object.assign(entry, { pair: `test event ${id}`, time_to_tca_d: c ? c.time_to_tca : ev.time_to_tca, miss_distance_m: c ? c.miss_distance : ev.miss_distance,
        model_tier: TIERS[num(a.tier)], needs_review: !!a.needs_review, log_pc: a.log_pc, true_tier: TIERS[num(ev.true_tier)], source: 'test_data.csv' });
    }
    S.log.push(entry);
    saveLog();
    toast(`${DEC_WORD[action]}: ${entry.pair || id}, logged by ${initials}.${S.logPersist ? '' : ' Browser storage is off, so export the log before closing this tab.'}`);
    if (view === 'live') { renderDetail(); renderFeed(); } else if (view === 'evaluator') renderTestDetail();
    renderLogDrawer();
  }
  function openLog() {
    renderLogDrawer();
    const d = $('#logdrawer');
    d.classList.add('open'); d.setAttribute('aria-hidden', 'false');
    $('#scrim').hidden = false;
    $('#log-btn').setAttribute('aria-expanded', 'true');
    setTimeout(() => d.focus(), 50);
  }
  function closeLog() {
    const d = $('#logdrawer');
    d.classList.remove('open'); d.setAttribute('aria-hidden', 'true');
    $('#scrim').hidden = true;
    $('#log-btn').setAttribute('aria-expanded', 'false');
  }
  function renderLogDrawer() {
    const list = $('#ld-list');
    if (!list) return;
    $('#ld-storage').textContent = S.logPersist ? `${S.log.length} decision${S.log.length === 1 ? '' : 's'} saved in this browser. Export a CSV to hand over a shift.` :
      'Browser storage is off: decisions last only until this tab closes. Export a CSV to keep them.';
    const f = $('#ld-filter').value;
    const rows = S.log.filter((d) => f === 'all' || d.view === f).slice().sort((a, b) => (a.ts < b.ts ? 1 : -1));
    if (!rows.length) { list.innerHTML = '<li class="empty">No decisions yet. Open a conjunction and choose Acknowledge, Escalate or Dismiss.</li>'; return; }
    const VIEW_NAME = { live: 'Map', evaluator: 'Evaluator', assess: 'Assess' };
    list.innerHTML = rows.map((d) => `<li>
      <div class="r1">${decTag(d.action)}<b>${esc(d.pair || d.event_id)}</b><span class="t">${esc(fmtUTC(d.ts))}</span></div>
      ${d.note ? `<div class="r2">${esc(d.note)}</div>` : ''}
      <div class="r3"><span>By ${esc(d.analyst)}</span><span>${esc(VIEW_NAME[d.view] || d.view)}</span>${d.model_tier ? `<span>Model said ${esc(d.model_tier)}${d.needs_review ? ', review' : ''}</span>` : ''}${isNum(d.miss_distance_m) ? `<span>Miss ${esc(fmtDist(d.miss_distance_m))}</span>` : ''}${d.tca ? `<span>TCA ${esc(fmtUTC(d.tca))}</span>` : ''}</div>
    </li>`).join('');
  }
  function exportLog() {
    const cols = ['ts', 'analyst', 'view', 'event_id', 'pair', 'action', 'note', 'model_tier', 'needs_review', 'log_pc', 'p_critical',
      'miss_distance_m', 'model_min_range_m', 'tca', 'time_to_tca_d', 'true_tier', 'norad_a', 'norad_b', 'data_as_of', 'source'];
    const q = (v) => { if (v == null) return ''; const s = String(v); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const csv = [cols.join(',')].concat(S.log.map((d) => cols.map((c) => q(d[c])).join(','))).join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `conjunction-watch-decisions-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '')}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ================================================================ CDM evaluator
  const fcTier = (e) => {
    const f = e && e.forecast;
    if (f && isNum(num(f.tier))) return num(f.tier);
    const t = num(get(e, 'assessment.forecast_tier'));
    return t;
  };
  async function ensureTestEvents(force) {
    if ((S.test.events || S.test.loading) && !force) { renderTestQueue(); return; }
    S.test.loading = true; S.test.err = null;
    renderTestQueue();
    try {
      const d = await api('/api/test/events');
      const ev = Array.isArray(d) ? d : d.events;
      if (!Array.isArray(ev)) throw new ApiError('response has no events list', 200);
      S.test.events = ev;
    } catch (e) { S.test.err = e.message; }
    S.test.loading = false;
    renderTestQueue();
    if (S.test.selected == null && S.test.events && S.test.events.length) {
      const first = testRows()[0];
      if (first) selectTestEvent(first.event_id, { noHash: true });
    } else if (S.test.selected == null) renderTestDetail();
  }
  function testRows() {
    let rows = S.test.events || [];
    const T = S.test;
    if (T.pred !== 'all') rows = rows.filter((e) => num(get(e, 'assessment.tier')) === Number(T.pred));
    if (T.review) rows = rows.filter((e) => get(e, 'assessment.needs_review') === true);
    if (T.dis) rows = rows.filter((e) => isNum(num(e.true_tier)) && num(get(e, 'assessment.tier')) !== num(e.true_tier));
    if (T.under) rows = rows.filter((e) => isNum(num(e.true_tier)) && num(get(e, 'assessment.tier')) < num(e.true_tier));
    if (S.search && /^\d+$/.test(S.search.trim())) rows = rows.filter((e) => String(e.event_id).includes(S.search.trim()));
    rows = rows.slice();
    const pt = (e) => num(get(e, 'assessment.tier')) || 0;
    if (T.sort === 'id') rows.sort((a, b) => a.event_id - b.event_id);
    else if (T.sort === 'ttca') rows.sort((a, b) => (num(a.time_to_tca) || 0) - (num(b.time_to_tca) || 0));
    else if (T.sort === 'error') {
      const err = (e) => { const d = (num(e.true_tier) == null ? 0 : num(e.true_tier)) - pt(e); return d > 0 ? 10 + d : (d < 0 ? 5 - d : 0); };
      rows.sort((a, b) => err(b) - err(a) || pt(b) - pt(a));
    } else rows.sort((a, b) => pt(b) - pt(a) || (num(get(b, 'assessment.p_tier.2')) || 0) - (num(get(a, 'assessment.p_tier.2')) || 0) || (num(get(b, 'assessment.log_pc')) || -31) - (num(get(a, 'assessment.log_pc')) || -31));
    return rows;
  }
  function renderTestQueue() {
    const list = $('#tq');
    if (!list) return;
    if (S.test.loading && !S.test.events) { list.innerHTML = '<li class="skel"></li><li class="skel"></li><li class="skel"></li>'; return; }
    if (S.test.err && !S.test.events) {
      list.innerHTML = `<li class="state err-state"><b>Test events unavailable.</b> ${esc(S.test.err)}. <button type="button" class="btn sm" data-act="retry-test">Retry</button></li>`;
      $('#tq-count').textContent = '';
      return;
    }
    const rows = testRows();
    $('#tq-count').textContent = rows.length.toLocaleString('en-US');
    const all = S.test.events || [];
    const nDis = all.filter((e) => isNum(num(e.true_tier)) && num(get(e, 'assessment.tier')) !== num(e.true_tier)).length;
    const nUnder = all.filter((e) => isNum(num(e.true_tier)) && num(get(e, 'assessment.tier')) < num(e.true_tier)).length;
    $('#tq-foot').innerHTML = `<span title="Latest CDM per event, 2 or more days before TCA. Ground truth comes from the CDM's risk column, which the model never sees.">${all.length.toLocaleString('en-US')} events: ${nDis} with the wrong tier, ${nUnder} under-called</span>`;
    if (!rows.length) { list.innerHTML = '<li class="state">No events match these filters.</li>'; return; }
    list.innerHTML = rows.slice(0, 600).map((e) => {
      const a = e.assessment || {};
      const pt = num(a.tier); const tt = num(e.true_tier);
      const flag = tt == null || pt == null ? '' : (pt < tt ? `<span class="tag bad" title="Predicted ${TIERS[pt]}, ground truth ${TIERS[tt]}">Under-called</span>` : (pt > tt ? `<span class="tag" title="Predicted ${TIERS[pt]}, ground truth ${TIERS[tt]}">Over-called</span>` : ''));
      return `<li><div class="item ${TIER_CSS[pt] || ''}${String(S.test.selected) === String(e.event_id) ? ' sel' : ''}" role="button" tabindex="0" data-test="${esc(e.event_id)}" aria-label="${esc(`Event ${e.event_id}, predicted ${TIERS[pt] || 'no tier'}`)}">
        <div class="it1"><i class="dot ${TIER_CSS[pt] || ''}"></i><span class="it-name">Event ${esc(e.event_id)} <span class="x">${esc(e.c_object_type ? sentenceCase(e.c_object_type) : '')}</span></span><span class="it-time">${esc(fixed(e.time_to_tca, 1))} d to TCA</span></div>
        <div class="it2"><span>Miss <b>${esc(fmtDist(e.miss_distance))}</b></span><span>Pc <b>${pcHtml(a.log_pc)}</b></span>${a.needs_review ? reviewPill() : ''}${flag}</div>
      </div></li>`;
    }).join('');
  }
  async function selectTestEvent(id, opts) {
    opts = opts || {};
    S.test.selected = id;
    S.test.cdmIdx = null;
    if (!opts.noHash && location.hash !== `#evaluator/${id}`) history.replaceState(null, '', `#evaluator/${id}`);
    $$('#tq .item.sel').forEach((c) => c.classList.remove('sel'));
    const card = $(`#tq .item[data-test="${CSS.escape(String(id))}"]`);
    if (card) { card.classList.add('sel'); card.scrollIntoView({ block: 'nearest' }); }
    const seq = ++S.test.seq;
    S.test.detail = null; S.test.detailErr = null; S.test.detailLoading = true;
    renderTestDetail();
    try {
      const d = await api(`/api/test/events/${encodeURIComponent(id)}`);
      if (seq !== S.test.seq) return;
      if (!Array.isArray(d.cdms)) throw new ApiError('response has no cdms list', 200);
      d.cdms.sort((a, b) => (num(b.time_to_tca) || 0) - (num(a.time_to_tca) || 0));
      S.test.detail = d;
    } catch (e) {
      if (seq !== S.test.seq) return;
      S.test.detailErr = e.message;
    }
    S.test.detailLoading = false;
    renderTestDetail();
  }
  function renderTestDetail() {
    const el = $('#ev-main');
    if (!el) return;
    const id = S.test.selected;
    if (id == null) {
      el.innerHTML = S.test.err ? `<div class="state err-state"><b>Test events unavailable.</b> ${esc(S.test.err)}. <button type="button" class="btn sm" data-act="retry-test">Retry</button></div>` :
        '<div class="state">Pick an event from the list to see its CDM history, the model estimate against ground truth, and the reasons for its tier.</div>';
      return;
    }
    const ev = (S.test.events || []).find((e) => String(e.event_id) === String(id)) || { event_id: id };
    const pT = num(get(ev, 'assessment.tier')); const tT = num(ev.true_tier);
    const errTag = pT == null || tT == null || pT === tT ? '' : (pT < tT ? '<span class="tag bad" title="Predicted tier is below ground truth: a missed escalation">Under-called</span>' : '<span class="tag" title="Predicted tier is above ground truth: a false alarm">Over-called</span>');
    const head = `<div class="page-head"><h2>Event ${esc(id)}</h2>
      ${lv('Predicted', tierPill(pT))}${lv('Ground truth', tierPill(tT))}${fcTier(ev) != null ? lv('Forecast at TCA', tierPill(fcTier(ev))) : ''}
      ${get(ev, 'assessment.needs_review') ? reviewPill() : ''}${errTag}${ev.c_object_type ? lv('Secondary', esc(sentenceCase(ev.c_object_type))) : ''}</div>`;
    if (S.test.detailLoading) { el.innerHTML = head + '<div class="loading">Loading CDM history…</div>'; return; }
    if (S.test.detailErr) { el.innerHTML = head + `<p class="err-box"><b>Event detail unavailable.</b> ${esc(S.test.detailErr)}. <button type="button" class="btn sm" data-act="retry-test-detail">Retry</button></p>`; return; }
    const d = S.test.detail;
    if (!d || !d.cdms.length) { el.innerHTML = head + '<div class="state">This event has no CDMs.</div>'; return; }
    const cdms = d.cdms;
    const idx = S.test.cdmIdx != null ? S.test.cdmIdx : cdms.length - 1;
    const c = cdms[idx];
    const a = c.assessment || {};
    const fc = c.forecast || ev.forecast;
    const prevNote = ($('.act-note', el) || {}).value || '';
    el.innerHTML = head + `
      <div class="tiles">
        <div class="tile"><div class="l">CDMs</div><div class="v">${esc(cdms.length)}</div></div>
        <div class="tile"><div class="l">Time to TCA</div><div class="v">${esc(fixed(c.time_to_tca, 2))} <small>days</small></div></div>
        <div class="tile"><div class="l">Miss distance</div><div class="v">${esc(fmtDist(c.miss_distance))}</div></div>
        <div class="tile"><div class="l">Estimated Pc</div><div class="v">${pcHtml(a.log_pc)}</div><div class="d">80%: ${esc(pcText(a.log_pc_lo))} to ${esc(pcText(a.log_pc_hi))}</div></div>
        <div class="tile" title="From the CDM's risk column; used for evaluation only"><div class="l">True Pc</div><div class="v">${pcHtml(c.true_log_pc)}</div></div>
        <div class="tile"><div class="l">Relative speed</div><div class="v">${esc(fmtSpeed(c.relative_speed))}</div></div>
      </div>
      <div class="ev-grid">
        <div class="panel wide"><div class="panel-head"><h3>Collision probability over time</h3><span class="hint">Select a point to inspect that CDM</span></div><div id="ev-pc" class="chart"></div></div>
        <div class="panel"><div class="panel-head"><h3>Miss distance</h3></div><div id="ev-md" class="chart sm"></div></div>
        <div class="panel"><div class="panel-head"><h3>Miss vector (RTN, m)</h3></div><div id="ev-rtn" class="chart sm"></div></div>
        <div class="panel"><div class="panel-head"><h3>Why</h3><span class="hint">CDM ${idx + 1} of ${cdms.length}, ${esc(fixed(c.time_to_tca, 2))} days to TCA</span></div>
          <div class="page-head" style="margin-bottom:12px">${lv('Decision', tierPill(num(a.tier)))}${lv('Distance rule', tierPill(num(a.md_tier)))}${a.needs_review ? reviewPill() : ''}</div>
          ${stackBar(a.p_tier)}
          <div style="margin-top:12px">${reasonsHtml(a.reasons)}</div>
          ${fc ? `<p class="hint" style="margin-top:10px">Forecast tier at the final CDM: <b>${esc(TIERS[num(fc.tier)] || '?')}</b>${Array.isArray(fc.p_tier) ? `, P(Critical) ${esc(pct(fc.p_tier[2]))}` : ''}.</p>` : ''}
        </div>
        <div class="panel"><div class="panel-head"><h3>Decision</h3></div>${actionsHtml('evaluator', id, decisionsFor('evaluator', id), false)}
          <p class="hint" style="margin-top:10px" title="Only the 13 allowed columns are model inputs">Inputs: c_sigma_r ${esc(fmtDist(c.c_sigma_r))}${num(c.c_sigma_r) >= 6e7 ? ' (not available)' : ''}, t_sigma_r ${esc(fmtDist(c.t_sigma_r))}, max Pc bound ${esc(pcText(c.max_risk_estimate))}, F10.7 ${esc(fmtNum(c.F10, 0))}, Ap ${esc(fmtNum(c.AP, 0))}.</p></div>
        <!-- Reserved: "Full-covariance Pc and manoeuvre options" panel (physics engine). Leave empty until wired. -->
        <div id="physics-slot" class="physics-slot wide" data-panel="Full-covariance Pc and manoeuvre options"></div>
        <div class="panel wide"><div class="panel-head"><h3>CDM history</h3><span class="hint">First to latest CDM</span></div>
          <div class="tbl-wrap"><table class="tbl"><thead><tr><th>CDM</th><th>Days to TCA</th><th>Miss distance</th><th>Relative speed</th><th>Estimated Pc</th><th>80% interval</th><th>Max Pc bound</th><th>True Pc</th><th class="l" style="min-width:0">Decision</th><th></th></tr></thead><tbody>
          ${cdms.map((x, i) => { const xa = x.assessment || {}; return `<tr class="click${i === idx ? ' chosen' : ''}" data-cdm="${i}"><td>${i + 1}</td><td>${esc(fixed(x.time_to_tca, 2))}</td><td>${esc(fmtDist(x.miss_distance))}</td><td>${esc(fmtSpeed(x.relative_speed))}</td><td>${esc(pcText(xa.log_pc))}</td><td>${esc(pcText(xa.log_pc_lo))} to ${esc(pcText(xa.log_pc_hi))}</td><td>${esc(pcText(x.max_risk_estimate))}</td><td>${esc(pcText(x.true_log_pc))}</td><td class="l" style="min-width:0">${tierPill(num(xa.tier))}</td><td>${xa.needs_review ? reviewPill() : ''}</td></tr>`; }).join('')}
          </tbody></table></div></div>
      </div>`;
    const note = $('.act-note', el); if (note && prevNote) note.value = prevNote;
    renderEvCharts(cdms, idx);
  }

  function renderEvCharts(cdms, idx) {
    const pcEl = $('#ev-pc'); const mdEl = $('#ev-md'); const rtnEl = $('#ev-rtn');
    if (!window.Plotly) { [pcEl, mdEl, rtnEl].forEach((e) => e && noPlotly(e)); return; }
    const x = cdms.map((c) => num(c.time_to_tca));
    const A = (k) => cdms.map((c) => num(get(c, `assessment.${k}`)));
    const med = A('log_pc'); const lo = A('log_pc_lo'); const hi = A('log_pc_hi');
    const tru = cdms.map((c) => num(c.true_log_pc));
    const bound = cdms.map((c) => num(c.max_risk_estimate));
    const finite = [].concat(med, lo, hi, tru, bound).filter((v) => v != null);
    const ymin = Math.max(-31, Math.min(-8, Math.floor(Math.min.apply(null, finite.length ? finite : [-10])) - 1));
    const hl = (y, col, txt) => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: y, y1: y, line: { color: col, width: 1, dash: 'dash' } });
    const ann = (y, col, txt) => ({ xref: 'paper', x: 0, y, text: txt, showarrow: false, xanchor: 'left', yanchor: 'bottom', font: { color: col, size: 11 } });
    const sel = [x[idx]];
    const pcData = [
      { x, y: hi, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
      { x, y: lo, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(124,196,255,.16)', name: '80% interval', hoverinfo: 'skip' },
      { x, y: med, mode: 'lines+markers', name: 'Model estimate', line: { color: COLORS.focus, width: 2 }, marker: { size: 7, color: COLORS.focus, line: { color: COLORS.panel, width: 1 } },
        customdata: cdms.map((c) => [pcText(get(c, 'assessment.log_pc')), pcText(get(c, 'assessment.log_pc_lo')), pcText(get(c, 'assessment.log_pc_hi'))]),
        hovertemplate: '%{x:.2f} days to TCA<br>Estimated Pc %{customdata[0]} (80%: %{customdata[1]} to %{customdata[2]})<extra></extra>' },
      { x, y: tru, mode: 'lines+markers', name: 'Ground truth', line: { color: COLORS.text, width: 1.5, dash: 'dot' }, marker: { size: 7, symbol: 'diamond', color: COLORS.text, line: { color: COLORS.panel, width: 1 } },
        hovertemplate: '%{x:.2f} days to TCA<br>True log10 Pc %{y:.2f}<extra></extra>' },
      { x, y: bound, mode: 'lines', name: 'Max Pc bound', line: { color: COLORS.muted, width: 1, dash: 'dashdot' }, hovertemplate: '%{x:.2f} days to TCA<br>Max Pc bound (log10) %{y:.2f}<extra></extra>' },
      { x: sel, y: [med[idx]], mode: 'markers', name: 'Selected CDM', showlegend: false, marker: { size: 15, color: 'rgba(0,0,0,0)', line: { color: COLORS.text, width: 2 } }, hoverinfo: 'skip' },
    ];
    plotReact(pcEl, pcData, plotlyBase({
      margin: { l: 50, r: 12, t: 30, b: 40 },
      xaxis: { title: { text: 'Days to TCA', font: { size: 12 } }, autorange: 'reversed', gridcolor: COLORS.grid, zeroline: false },
      yaxis: { title: { text: 'log10 Pc', font: { size: 12 } }, range: [ymin, 0.3], gridcolor: COLORS.grid, zeroline: false },
      shapes: [hl(-6, COLORS.elev), hl(-4, COLORS.crit)],
      annotations: [ann(-6, COLORS.elev, 'Elevated from 1e-6'), ann(-4, COLORS.crit, 'Critical from 1e-4')],
    }), PCFG);
    if (!pcEl._cw) { pcEl._cw = true; pcEl.on('plotly_click', (ev) => { const p = ev.points && ev.points[0]; if (p && isNum(p.pointIndex) && p.curveNumber >= 2 && p.curveNumber <= 4) { S.test.cdmIdx = p.pointIndex; renderTestDetail(); } }); }

    const md = cdms.map((c) => num(c.miss_distance));
    plotReact(mdEl, [
      { x, y: md.map((m) => (m == null ? null : m / 1000)), mode: 'lines+markers', name: 'Miss distance', line: { color: COLORS.focus, width: 2 }, marker: { size: 7, color: COLORS.focus, line: { color: COLORS.panel, width: 1 } }, hovertemplate: '%{x:.2f} days to TCA<br>%{y:.3f} km<extra></extra>' },
      { x: sel, y: [md[idx] / 1000], mode: 'markers', marker: { size: 15, color: 'rgba(0,0,0,0)', line: { color: COLORS.text, width: 2 } }, hoverinfo: 'skip', showlegend: false },
    ], plotlyBase({
      showlegend: false, margin: { l: 50, r: 12, t: 10, b: 40 },
      xaxis: { title: { text: 'Days to TCA', font: { size: 12 } }, autorange: 'reversed', gridcolor: COLORS.grid, zeroline: false },
      yaxis: { title: { text: 'Miss distance (km)', font: { size: 12 } }, type: 'log', dtick: 1, tickformat: '~g', gridcolor: COLORS.grid },
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: COLORS.crit, dash: 'dash', width: 1 } }, { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 5, y1: 5, line: { color: COLORS.elev, dash: 'dash', width: 1 } }],
      annotations: [{ xref: 'paper', x: 1, y: 0, yref: 'y', text: '1 km', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: COLORS.crit, size: 11 } },
        { xref: 'paper', x: 1, y: Math.log10(5), yref: 'y', text: '5 km', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: COLORS.elev, size: 11 } }],
    }), PCFG);

    const R = cdms.map((c) => num(c.relative_position_r)); const T = cdms.map((c) => num(c.relative_position_t)); const N = cdms.map((c) => num(c.relative_position_n));
    const data3 = [
      { type: 'scatter3d', mode: 'lines+markers', name: 'Miss vector per CDM', x: T, y: N, z: R,
        line: { color: COLORS.focus, width: 3 }, marker: { size: 4, color: x, colorscale: [[0, COLORS.focus], [1, '#3B4A59']], reversescale: true },
        customdata: x, hovertemplate: '%{customdata:.2f} days to TCA<br>T %{x:.0f} m, N %{y:.0f} m, R %{z:.0f} m<extra></extra>' },
      { type: 'scatter3d', mode: 'markers', name: 'Primary (origin)', x: [0], y: [0], z: [0], marker: { size: 5, color: COLORS.a }, hovertemplate: 'Primary at the origin<extra></extra>' },
    ];
    const c = cdms[idx];
    const vr = num(c.relative_velocity_r); const vt = num(c.relative_velocity_t); const vn = num(c.relative_velocity_n);
    if (vr != null && vt != null && vn != null && R[idx] != null) {
      const vmag = Math.hypot(vr, vt, vn) || 1;
      const dt = 2.5 * Math.max(200, num(c.miss_distance) || 1000) / vmag;
      const pts = [-1, 0, 1].map((s) => [T[idx] + vt * dt * s, N[idx] + vn * dt * s, R[idx] + vr * dt * s]);
      data3.push({ type: 'scatter3d', mode: 'lines', name: 'Flyby line', x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), z: pts.map((p) => p[2]),
        line: { color: COLORS.b, width: 4, dash: 'dash' }, hovertemplate: 'Straight-line flyby from the relative velocity (illustration)<extra></extra>' });
    }
    data3.push({ type: 'scatter3d', mode: 'markers', name: 'Selected CDM', x: [T[idx]], y: [N[idx]], z: [R[idx]], marker: { size: 7, color: COLORS.text, symbol: 'diamond' }, hoverinfo: 'skip' });
    const ax = (t) => ({ title: { text: t, font: { size: 11 } }, gridcolor: COLORS.grid, zerolinecolor: COLORS.line, backgroundcolor: 'rgba(0,0,0,0)', color: COLORS.muted });
    plotReact(rtnEl, data3, plotlyBase({
      margin: { l: 0, r: 0, t: 0, b: 0 }, legend: { orientation: 'h', x: 0, y: 1.02, font: { size: 12, color: COLORS.text } },
      scene: { xaxis: ax('T (m)'), yaxis: ax('N (m)'), zaxis: ax('R (m)'), aspectmode: 'cube', bgcolor: 'rgba(0,0,0,0)', camera: { eye: { x: 1.5, y: 1.4, z: 0.9 } } },
    }), PCFG);
  }

  // ================================================================ Model (metrics)
  async function ensureMetrics(force) {
    if ((S.metrics || S.metricsLoading) && !force) { if (S.metrics) renderMetrics(); return; }
    S.metricsLoading = true; S.metricsErr = null;
    $('#model-main').innerHTML = '<div class="loading" style="padding:20px">Loading metrics…</div>';
    try { S.metrics = await api('/api/metrics'); S.metricsErrStatus = null; } catch (e) { S.metricsErr = e.message; S.metricsErrStatus = e.status; S.metrics = null; }
    S.metricsLoading = false;
    renderMetrics();
  }
  const f3 = (v) => (isNum(num(v)) ? num(v).toFixed(3) : '–');
  const isMetricBlock = (o) => o && typeof o === 'object' && !Array.isArray(o) && (isNum(o.macro_f1) || Array.isArray(o.recall) || Array.isArray(o.confusion));
  function renderMetrics() {
    const el = $('#model-main');
    const m = S.metrics;
    if (!m || typeof m !== 'object' || (!m.methods && !m.generated_at)) {
      const notGen = !S.metricsErr || S.metricsErrStatus === 404 || S.metricsErrStatus === 503;
      el.innerHTML = `<div class="state ${S.metricsErr ? 'err-state' : ''}"><b>${notGen ? 'No metrics yet.' : 'Metrics unavailable.'}</b> ${S.metricsErr ? esc(S.metricsErr) + '. ' : ''}${notGen ? 'Run <code>python src/train.py</code> to write <code>outputs/metrics.json</code>, then reload.' : ''} <button type="button" class="btn sm" data-act="retry-metrics">Reload</button></div>`;
      return;
    }
    const methods = Array.isArray(m.methods) ? m.methods : [];
    const mName = (n) => String(n || '').replace(/\s*\(chosen\)\s*$/i, '');   // the tag already says it
    const chosen = methods.find((x) => x.name === m.chosen_method) || methods.find((x) => x.kind === 'model') || methods[0];
    const base = methods.find((x) => /miss-distance rule/i.test(x.name || '')) || methods.find((x) => x.kind === 'baseline');
    const split = S.metricSplit || 'test';
    const ct = get(chosen, split) || {}; const bt = get(base, split) || {};
    const delta = (a, b, lowerBetter) => {
      if (!isNum(num(a)) || !isNum(num(b))) return '';
      const d = num(a) - num(b); const good = lowerBetter ? d < 0 : d > 0;
      return `<div class="d ${d === 0 ? '' : (good ? 'good' : 'bad')}">${d > 0 ? '+' : ''}${Number.isInteger(d) ? d : d.toFixed(3)} vs distance rule</div>`;
    };
    const ev = m.eval_set || {};
    const known = ['generated_at', 'eval_set', 'cv_set', 'chosen_method', 'thresholds', 'methods', 'regression', 'review_queue', 'pc_driven_escalations', 'calibration', 'calibration_elevated', 'feature_importance', 'failure_cases', 'event_forecast', 'screening_model', 'notes', 'runtime_s'];
    const extra = Object.keys(m).filter((k) => !known.includes(k));
    const rq = m.review_queue || {}; const pe = m.pc_driven_escalations || {}; const rg = m.regression || {};
    const recC = get(ct, 'recall.2');
    const splitName = split === 'test' ? 'held-out test set' : 'cross-validation folds';

    let html = `<div class="page-head"><h2>Model</h2>
      ${lv('Chosen', `<span>${esc(mName(m.chosen_method) || 'not named')}</span>`)}
      <span class="hint" title="${esc(Array.isArray(ev.true_tier_counts) ? `True tiers (Safe, Elevated, Critical): ${ev.true_tier_counts.join(', ')}` : '')}">Evaluated on ${esc(ev.name || '?')}: ${esc(fmtNum(ev.n_rows, 0))} CDMs, ${esc(fmtNum(ev.n_events, 0))} events. Generated ${esc(fmtUTC(m.generated_at))}.</span>
      <div class="seg" id="metric-split" role="group" aria-label="Evaluation split" style="margin-left:auto">
        <button type="button" data-split="test" class="${split === 'test' ? 'on' : ''}">Held-out test</button><button type="button" data-split="cv" class="${split === 'cv' ? 'on' : ''}">Cross-validation</button></div></div>`;
    html += `<div class="tiles">
      <div class="tile"><div class="l">Macro F1</div><div class="v">${f3(ct.macro_f1)}</div>${delta(ct.macro_f1, bt.macro_f1)}</div>
      <div class="tile" title="Recall-weighted F score for the Critical tier"><div class="l">F2 Critical</div><div class="v">${f3(ct.f2_critical)}</div>${delta(ct.f2_critical, bt.f2_critical)}</div>
      <div class="tile"><div class="l">Critical recall</div><div class="v">${isNum(num(recC)) ? pct(recC, 1) : '–'}</div>${delta(recC, get(bt, 'recall.2'))}</div>
      <div class="tile"><div class="l">Missed Critical</div><div class="v">${esc(fmtNum(ct.missed_critical, 0))}</div>${delta(ct.missed_critical, bt.missed_critical, true)}</div>
      <div class="tile"><div class="l">False Critical alarms</div><div class="v">${esc(fmtNum(ct.false_alarms_critical, 0))}</div>${delta(ct.false_alarms_critical, bt.false_alarms_critical, true)}</div>
      <div class="tile"><div class="l">80% interval coverage</div><div class="v">${isNum(num(rg.interval_coverage_80)) ? pct(rg.interval_coverage_80, 1) : '–'}</div><div class="d">Target 80%</div></div>
    </div>`;
    html += `<div class="panel mb"><div class="panel-head"><h3>Baselines and model</h3><span class="hint">Recall and precision per tier on the ${splitName}. S, E, C = Safe, Elevated, Critical.</span></div><div class="tbl-wrap"><table class="tbl">
      <thead><tr><th rowspan="2">Method</th><th rowspan="2">Kind</th><th class="grp" colspan="2">Cross-validation</th><th class="grp" colspan="2">Test</th><th class="grp" colspan="3">Recall</th><th class="grp" colspan="3">Precision</th><th rowspan="2">Missed Critical</th><th rowspan="2">Critical called Safe</th><th rowspan="2">False Critical alarms</th></tr>
      <tr><th>Macro F1</th><th>F2 Critical</th><th>Macro F1</th><th>F2 Critical</th><th>S</th><th>E</th><th>C</th><th>S</th><th>E</th><th>C</th></tr></thead><tbody>`;
    const best = (k, sp) => Math.max.apply(null, methods.map((x) => num(get(x, `${sp}.${k}`))).filter((v) => v != null).concat([-1]));
    methods.forEach((x) => {
      const s = get(x, split) || {};
      const cell = (sp, k) => { const v = num(get(x, `${sp}.${k}`)); return `<td class="${v != null && v === best(k, sp) ? 'best' : ''}">${f3(v)}</td>`; };
      const arr = (k, i) => `<td>${isNum(num(get(s, `${k}.${i}`))) ? pct(get(s, `${k}.${i}`), 1) : '–'}</td>`;
      html += `<tr class="${x === chosen ? 'chosen' : ''}" title="${esc(x.description || '')}"><td class="l">${esc(mName(x.name))}${x === chosen ? ' <span class="tag">Chosen</span>' : ''}</td><td>${esc(sentenceCase(x.kind || ''))}</td>
        ${cell('cv', 'macro_f1')}${cell('cv', 'f2_critical')}${cell('test', 'macro_f1')}${cell('test', 'f2_critical')}
        ${arr('recall', 0)}${arr('recall', 1)}${arr('recall', 2)}${arr('precision', 0)}${arr('precision', 1)}${arr('precision', 2)}
        <td>${esc(fmtNum(s.missed_critical, 0))}</td><td>${esc(fmtNum(s.critical_called_safe, 0))}</td><td>${esc(fmtNum(s.false_alarms_critical, 0))}</td></tr>`;
    });
    if (!methods.length) html += '<tr><td colspan="15" class="l hint">No methods in metrics.json.</td></tr>';
    html += '</tbody></table></div></div>';
    const opts = methods.map((x, i) => `<option value="${i}" ${x === chosen ? 'selected' : ''}>${esc(mName(x.name))}</option>`).join('');
    html += `<div class="grid3">
      <div class="panel"><div class="panel-head"><h3>Confusion</h3><select id="cm-method" aria-label="Method for the confusion matrix" style="max-width:220px;margin-left:auto">${opts}</select></div><div id="cm-a" class="chart sm"></div></div>
      <div class="panel"><div class="panel-head"><h3>Confusion: ${esc(base ? mName(base.name) : 'baseline')}</h3></div><div id="cm-b" class="chart sm"></div></div>
      <div class="panel"><div class="panel-head"><h3>Calibration</h3><span class="hint">Predicted against observed</span></div><div id="calib" class="chart sm"></div></div>
    </div>`;
    html += `<div class="grid3">
      <div class="panel"><div class="panel-head"><h3 title="Review flags fire when the 80% Pc interval spans a tier boundary or the max Pc bound reaches 1e-4">Review queue</h3></div><dl class="kv">
        <dt>CDMs sent to review</dt><dd>${isNum(num(rq.needs_review_rate)) ? pct(rq.needs_review_rate, 2) : '–'}</dd>
        <dt>Error rate in review</dt><dd>${isNum(num(rq.error_rate_in_review_queue)) ? pct(rq.error_rate_in_review_queue, 1) : '–'}</dd>
        <dt>Error rate, decided automatically</dt><dd>${isNum(num(rq.error_rate_auto_decided)) ? pct(rq.error_rate_auto_decided, 1) : '–'}</dd></dl></div>
      <div class="panel"><div class="panel-head"><h3 title="Cases the 1 km / 5 km distance rule cannot see">Escalations driven by Pc</h3></div><dl class="kv">
        <dt>CDMs where Pc raises the tier</dt><dd>${esc(fmtNum(pe.n_rows_where_pc_raises_tier, 0))}</dd>
        <dt>Caught by the model</dt><dd>${esc(fmtNum(pe.caught_by_model, 0))}</dd>
        <dt>Caught by the distance rule</dt><dd>${esc(fmtNum(pe.caught_by_distance_rule, 0))}</dd></dl></div>
      <div class="panel"><div class="panel-head"><h3 title="${esc(rg.note || '')}">Pc estimate and thresholds</h3></div><dl class="kv">
        <dt>Mean error, log10 Pc</dt><dd>${f3(rg.test_mae_log_pc)}</dd>
        <dt>Mean error, high risk</dt><dd>${f3(rg.test_mae_log_pc_high_risk)}</dd>
        <dt>80% coverage, above floor</dt><dd>${isNum(num(rg.interval_coverage_80_nonfloor)) ? pct(rg.interval_coverage_80_nonfloor, 1) : '–'}</dd>
        ${Object.keys(m.thresholds || {}).map((k) => { const t = m.thresholds[k]; return t && typeof t === 'object' ? `<dt>Thresholds (${esc(k)})</dt><dd>Critical ${esc(pct(t.p_critical))}, Elevated ${esc(pct(t.p_elevated))}</dd>` : ''; }).join('')}</dl></div>
    </div>`;
    const ef = m.event_forecast;
    const sm = m.screening_model;
    const blockRows = (obj) => Object.keys(obj || {}).filter((k) => isMetricBlock(obj[k])).map((k) => { const b = obj[k]; return `<tr><td class="l">${esc(sentenceCase(k.replace(/_/g, ' ')))}</td><td>${f3(b.macro_f1)}</td><td>${f3(b.f2_critical)}</td><td>${isNum(num(get(b, 'recall.2'))) ? pct(get(b, 'recall.2'), 1) : '–'}</td><td>${esc(fmtNum(b.missed_critical, 0))}</td><td>${esc(fmtNum(b.false_alarms_critical, 0))}</td><td>${esc(fmtNum(b.n, 0))}</td></tr>`; }).join('');
    const blockHead = (first) => `<thead><tr><th>${first}</th><th>Macro F1</th><th>F2 Critical</th><th>Critical recall</th><th>Missed Critical</th><th>False Critical</th><th>N</th></tr></thead>`;
    html += `<div class="grid2">
      <div class="panel"><div class="panel-head"><h3 title="${esc(ef ? [ef.description, ef.note].filter(Boolean).join(' ') : '')}">Event forecast: tier at TCA</h3></div>${ef ? `
        <div class="tbl-wrap"><table class="tbl">${blockHead('Method')}<tbody>${blockRows(ef) || '<tr><td colspan="7" class="l hint">No scored blocks.</td></tr>'}</tbody></table></div>` : '<div class="state">Not in metrics.json yet.</div>'}</div>
      <div class="panel"><div class="panel-head"><h3 title="${esc(sm ? sm.description || '' : '')}">Screening model for live data</h3></div>${sm ? `
        <div class="tbl-wrap"><table class="tbl">${blockHead('Split')}<tbody>${blockRows(sm) || '<tr><td colspan="7" class="l hint">No scored blocks.</td></tr>'}</tbody></table></div>` : '<div class="state">Not in metrics.json yet.</div>'}</div>
    </div>`;
    const fcases = Array.isArray(m.failure_cases) ? m.failure_cases : [];
    html += `<div class="grid2">
      <div class="panel"><div class="panel-head"><h3>Feature importance</h3><span class="hint">Share of total split gain</span></div><div id="fi" class="chart"></div></div>
      <div class="panel"><div class="panel-head"><h3>Where it fails</h3><span class="hint">Select a row to open it in the evaluator</span></div>
        ${fcases.length ? `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Event</th><th>Days</th><th>Miss</th><th>Truth</th><th>Predicted</th><th>Est. Pc</th><th>True Pc</th><th class="l">Why</th></tr></thead><tbody>
        ${fcases.map((f) => `<tr class="click" data-goto="${esc(f.event_id)}" tabindex="0"><td><a href="#evaluator/${esc(f.event_id)}">${esc(f.event_id)}</a></td><td>${esc(fixed(f.time_to_tca, 2))}</td><td>${esc(fmtDist(f.miss_distance))}</td><td>${tierPill(num(f.true_tier))}</td><td>${tierPill(num(f.pred_tier))}</td><td>${esc(pcText(f.log_pc))}</td><td>${esc(pcText(f.true_log_pc))}</td><td class="l">${esc(f.why || '')}</td></tr>`).join('')}
        </tbody></table></div>` : '<div class="state">No failure cases listed.</div>'}</div>
    </div>`;
    if (Array.isArray(m.notes) && m.notes.length) html += `<div class="panel mb"><div class="panel-head"><h3>Notes</h3></div><ul class="notes">${m.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul></div>`;
    if (extra.length) html += `<div class="panel"><details class="raw"><summary>Other keys in metrics.json (${extra.map(esc).join(', ')})</summary><pre>${esc(JSON.stringify(extra.reduce((o, k) => { o[k] = m[k]; return o; }, {}), null, 2))}</pre></details></div>`;
    el.innerHTML = html;
    renderMetricCharts(methods, chosen, base, split);
  }
  function confusionChart(el, block) {
    const cm = block && Array.isArray(block.confusion) ? block.confusion : null;
    if (!cm) { el.innerHTML = '<div class="state">No confusion matrix for this method and split.</div>'; return; }
    const rowsum = cm.map((r) => r.reduce((a, b) => a + (num(b) || 0), 0) || 1);
    const z = cm.map((r, i) => r.map((v) => (num(v) || 0) / rowsum[i]));
    const text = cm.map((r, i) => r.map((v) => `${(num(v) || 0).toLocaleString('en-US')}<br>${pct((num(v) || 0) / rowsum[i])}`));
    plotReact(el, [{ type: 'heatmap', z, x: TIERS, y: TIERS, text, texttemplate: '%{text}', textfont: { size: 12, color: COLORS.text },
      colorscale: [[0, '#222E3B'], [1, '#2F6A9A']], zmin: 0, zmax: 1, showscale: false, xgap: 2, ygap: 2,
      hovertemplate: 'True %{y}, predicted %{x}<br>%{text}<extra></extra>' }],
    plotlyBase({ margin: { l: 78, r: 8, t: 8, b: 44 }, yaxis: { autorange: 'reversed', gridcolor: 'rgba(0,0,0,0)', title: { text: 'True', font: { size: 12 } } }, xaxis: { side: 'bottom', gridcolor: 'rgba(0,0,0,0)', title: { text: 'Predicted', font: { size: 12 } } } }), PCFG);
  }
  function renderMetricCharts(methods, chosen, base, split) {
    if (!window.Plotly) { $$('#cm-a,#cm-b,#calib,#fi').forEach(noPlotly); return; }
    const sel = $('#cm-method');
    const draw = () => { const mth = methods[Number(sel.value)] || chosen; confusionChart($('#cm-a'), get(mth, split)); };
    if (sel) { sel.onchange = draw; draw(); }
    confusionChart($('#cm-b'), get(base, split));
    const cal = Array.isArray(S.metrics.calibration) ? S.metrics.calibration : [];
    const cal2 = Array.isArray(S.metrics.calibration_elevated) ? S.metrics.calibration_elevated : [];
    const calEl = $('#calib');
    if (!cal.length && !cal2.length) calEl.innerHTML = '<div class="state">No calibration data.</div>';
    else {
      const tr = (arr, name, col) => ({ x: arr.map((c) => num(c.p_pred)), y: arr.map((c) => num(c.p_obs)), mode: 'lines+markers', name, line: { color: col, width: 2 },
        marker: { color: col, size: arr.map((c) => clamp(6 + Math.log10((num(c.n) || 1) + 1) * 2.5, 6, 16)), line: { color: COLORS.panel, width: 1 } },
        customdata: arr.map((c) => c.n), hovertemplate: `${name}<br>Predicted %{x:.3f}, observed %{y:.3f}<br>N = %{customdata}<extra></extra>` });
      const data = [{ x: [0, 1], y: [0, 1], mode: 'lines', name: 'Perfect', line: { color: COLORS.muted, dash: 'dot', width: 1 }, hoverinfo: 'skip' }];
      if (cal.length) data.push(tr(cal, 'P(Pc ≥ 1e-4)', COLORS.crit));
      if (cal2.length) data.push(tr(cal2, 'P(Pc ≥ 1e-6)', COLORS.elev));
      plotReact(calEl, data, plotlyBase({ margin: { l: 46, r: 10, t: 28, b: 40 }, xaxis: { title: { text: 'Predicted probability', font: { size: 12 } }, range: [0, 1], gridcolor: COLORS.grid }, yaxis: { title: { text: 'Observed frequency', font: { size: 12 } }, range: [0, 1], gridcolor: COLORS.grid } }), PCFG);
    }
    const fi = (Array.isArray(S.metrics.feature_importance) ? S.metrics.feature_importance : []).slice(0, 15).reverse();
    const fiEl = $('#fi');
    if (!fi.length) fiEl.innerHTML = '<div class="state">No feature importance.</div>';
    else {
      plotReact(fiEl, [{ type: 'bar', orientation: 'h', y: fi.map((f) => f.feature), x: fi.map((f) => num(f.gain_pct)), marker: { color: COLORS.focus },
        hovertemplate: '%{y}: %{x:.1f}% of gain<extra></extra>', text: fi.map((f) => `${fixed(f.gain_pct, 1)}%`), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: COLORS.muted } }],
      plotlyBase({ margin: { l: 150, r: 44, t: 6, b: 30 }, xaxis: { ticksuffix: '%', gridcolor: COLORS.grid }, yaxis: { gridcolor: 'rgba(0,0,0,0)', tickfont: { size: 12, color: COLORS.text } } }), PCFG);
    }
  }

  // ================================================================ Assess one CDM
  // label = what the analyst reads; k = the API column (shown as the field tooltip)
  const FIELDS = [
    { k: 'event_id', label: 'Event id', u: '', step: 1, help: 'Any whole number; groups the CDM history' },
    { k: 'time_to_tca', label: 'Time to TCA', u: 'days', step: 'any', min: 0 },
    { k: 'miss_distance', label: 'Miss distance', u: 'm', step: 'any', min: 0 },
    { k: 'relative_speed', label: 'Relative speed', u: 'm/s', step: 'any', min: 0 },
    { k: 'relative_position_r', label: 'Radial offset', u: 'm', step: 'any' },
    { k: 'relative_position_t', label: 'Along-track offset', u: 'm', step: 'any' },
    { k: 'relative_position_n', label: 'Cross-track offset', u: 'm', step: 'any' },
    { k: 'c_object_type', label: 'Secondary object type', u: '', type: 'select' },
    { k: 'c_sigma_r', label: 'Secondary radial sigma', u: 'm', step: 'any', min: 0, help: '6e7 means no covariance' },
    { k: 't_sigma_r', label: 'Primary radial sigma', u: 'm', step: 'any', min: 0 },
    { k: 'max_risk_estimate', label: 'Max Pc bound', u: 'log10', step: 'any', max: 0 },
    { k: 'F10', label: 'F10.7 solar flux', u: 'sfu', step: 'any', min: 0 },
    { k: 'AP', label: 'Ap index', u: '', step: 'any', min: 0 },
  ];
  // Real CDMs from data/test_data.csv (the true tier comes from `risk` and is shown only as a note).
  const PRESETS = {
    safe: { note: 'Test-set event 0: miss 32.9 km, true Pc 4.6e-8 (Safe).', v: { event_id: 0, time_to_tca: 5.922, miss_distance: 32878, relative_speed: 7929, relative_position_r: -376.3, relative_position_t: 27867.5, relative_position_n: 17443.8, c_object_type: 'UNKNOWN', c_sigma_r: 212.79, t_sigma_r: 7.354, max_risk_estimate: -7.2287, F10: 75, AP: 9 } },
    elevated: { note: 'Test-set event 2: miss 21.2 km, so distance alone says Safe, but the true Pc is 2.3e-5 (Elevated).', v: { event_id: 2, time_to_tca: 6.852, miss_distance: 21175, relative_speed: 9279, relative_position_r: 14.7, relative_position_t: -16644.3, relative_position_n: -13090.9, c_object_type: 'UNKNOWN', c_sigma_r: 84.015, t_sigma_r: 2.385, max_risk_estimate: -4.4988, F10: 66, AP: 3 } },
    critical: { note: 'Test-set event 87: miss 1.03 km, so distance alone says Elevated, but the true Pc is 8.4e-4 (Critical).', v: { event_id: 87, time_to_tca: 3.376, miss_distance: 1032, relative_speed: 14861, relative_position_r: 63.0, relative_position_t: -123.5, relative_position_n: 1023.5, c_object_type: 'DEBRIS', c_sigma_r: 33.218, t_sigma_r: 4.373, max_risk_estimate: -2.9172, F10: 78, AP: 3 } },
  };
  let assessBuilt = false;
  function ensureAssessForm() {
    if (assessBuilt) return;
    assessBuilt = true;
    $('#assess-fields').innerHTML = FIELDS.map((f) => `<div class="fld" data-k="${f.k}" title="API field: ${f.k}">
      <label for="f-${f.k}">${esc(f.label)}${f.u ? ` <span class="u">(${esc(f.u)})</span>` : ''}</label>
      ${f.type === 'select' ? `<select id="f-${f.k}" name="${f.k}">${OBJECT_TYPES.map((o) => `<option value="${o}">${o === 'TBA' ? 'To be assigned' : sentenceCase(o)}</option>`).join('')}</select>` :
        `<input id="f-${f.k}" name="${f.k}" type="number" inputmode="decimal" step="${f.step}"${f.min != null ? ` min="${f.min}"` : ''}${f.max != null ? ` max="${f.max}"` : ''} required>`}
      ${f.help ? `<span class="fh">${esc(f.help)}</span>` : ''}</div>`).join('');
    $('#assess-fields').addEventListener('input', checkAssessGeometry);
    const draft = lsGet('cw.assess.draft', null);
    applyPreset(draft && typeof draft === 'object' ? { v: draft, note: 'Your last input, restored.' } : PRESETS.elevated);
  }
  function applyPreset(p) {
    FIELDS.forEach((f) => { const el = $(`#f-${f.k}`); if (el && p.v[f.k] != null) el.value = p.v[f.k]; });
    $('#preset-note').textContent = p.note || '';
    $$('.fld.bad').forEach((x) => x.classList.remove('bad'));
    $('#assess-err').textContent = '';
    checkAssessGeometry();
  }
  function readAssessForm() {
    const body = {}; const bad = [];
    FIELDS.forEach((f) => {
      const el = $(`#f-${f.k}`);
      const fld = el.closest('.fld');
      fld.classList.remove('bad');
      if (f.type === 'select') { body[f.k] = el.value; return; }
      const v = el.value.trim() === '' ? null : Number(el.value);
      if (v == null || !Number.isFinite(v) || (f.min != null && v < f.min) || (f.max != null && v > f.max)) { bad.push(f.k); fld.classList.add('bad'); return; }
      body[f.k] = f.k === 'event_id' ? Math.round(v) : v;
    });
    return { body, bad };
  }
  function checkAssessGeometry() {
    const g = (k) => Number(($(`#f-${k}`) || {}).value);
    const md = g('miss_distance'); const r = Math.hypot(g('relative_position_r'), g('relative_position_t'), g('relative_position_n'));
    const out = $('#assess-check');
    if (!out) return;
    if (Number.isFinite(md) && md > 0 && Number.isFinite(r) && r > 0) {
      const rel = Math.abs(r - md) / md;
      out.innerHTML = rel > 0.05 ? `<span class="warn">The offsets add up to ${esc(fmtDist(r))}, not the ${esc(fmtDist(md))} miss distance (${esc(pct(rel))} apart). Check the inputs.</span>` :
        `The offsets add up to the miss distance (${esc(fmtDist(r))}).`;
    } else out.textContent = '';
  }
  async function submitAssess(ev) {
    ev.preventDefault();
    const { body, bad } = readAssessForm();
    const err = $('#assess-err');
    if (bad.length) { err.textContent = `Check ${bad.map((k) => (FIELDS.find((f) => f.k === k) || {}).label || k).join(', ').toLowerCase()}.`; return; }
    err.textContent = '';
    lsSet('cw.assess.draft', body);
    const out = $('#assess-result');
    const btn = $('#assess-go');
    btn.disabled = true;
    out.innerHTML = '<div class="loading">Assessing…</div>';
    try {
      const a = await api('/api/assess', {}, { method: 'POST', body });
      renderAssessResult(a, body);
    } catch (e) {
      out.innerHTML = `<p class="err-box"><b>Assessment failed.</b> ${esc(e.message)}. Check the inputs and try again.</p>`;
    } finally { btn.disabled = false; }
  }
  function pcScale(a) {
    const pos = (lp) => clamp(((num(lp) == null ? -30 : num(lp)) + 30) / 30 * 100, 0, 100);
    const lo = pos(a.log_pc_lo); const hi = pos(a.log_pc_hi); const md = pos(a.log_pc);
    return `<div class="pc-scale" role="img" aria-label="Estimated Pc ${esc(pcText(a.log_pc))}, 80% interval ${esc(pcText(a.log_pc_lo))} to ${esc(pcText(a.log_pc_hi))}">
      <div class="pc-axis"></div><div class="pc-zone z1"></div><div class="pc-zone z2"></div>
      <div class="pc-band" style="left:${lo}%;width:${Math.max(0.6, hi - lo)}%"></div>
      <div class="pc-med" style="left:${md}%"></div>
      <div class="pc-top" style="left:${clamp(md, 6, 94)}%">${esc(pcText(a.log_pc))}</div>
      <div class="pc-lbl" style="left:0%;transform:none">1e-30</div><div class="pc-lbl" style="left:50%">1e-15</div>
      <div class="pc-lbl" style="left:80%">1e-6</div><div class="pc-lbl" style="left:100%;transform:translateX(-100%)">1</div>
    </div>`;
  }
  function assessVerdict(a, body) {
    const t = num(a.tier); const md = num(body.miss_distance); const mdt = num(a.md_tier);
    if (t === 2) return mdt === 2 ? `Critical: the objects pass ${fmtDist(md)} apart, under the 1\u00a0km limit.` : `Critical: the estimated collision probability is likely 1e-4 or more (P = ${pct(a.p_pc_critical)}).`;
    if (t === 1) return mdt === 1 ? `Elevated: the objects pass ${fmtDist(md)} apart, inside the 5\u00a0km warning distance.` : `Elevated: the estimated collision probability is likely 1e-6 or more (P = ${pct(a.p_pc_elevated)}).`;
    if (t === 0) return `Safe: the objects pass ${fmtDist(md)} apart and the collision probability is likely under 1e-6.`;
    return 'No tier returned.';
  }
  function renderAssessResult(a, body) {
    const out = $('#assess-result');
    if (!a || typeof a !== 'object') { out.innerHTML = '<p class="err-box"><b>Unexpected response from the API.</b></p>'; return; }
    const t = num(a.tier);
    out.innerHTML = `
      <div class="panel-head"><h3>Result</h3><span class="hint">${esc(fmtUTC(new Date().toISOString()))}</span></div>
      ${a._mock_note ? `<p class="warn-line">${esc(a._mock_note)}</p>` : ''}
      <p class="verdict ${TIER_CSS[t] || 'nt'}" style="padding:0 0 12px">${esc(assessVerdict(a, body))}${a.needs_review ? '<span class="verdict-sub">Needs review: the uncertainty spans a tier boundary.</span>' : ''}</p>
      <div class="page-head" style="margin-bottom:8px">${lv('Recommended', tierPill(t))}${lv('Distance rule', tierPill(num(a.md_tier)))}</div>
      <h4 class="sub-h">Tier probabilities</h4>${stackBar(a.p_tier)}
      <h4 class="sub-h">Estimated Pc with 80% interval</h4>${pcScale(a)}
      <dl class="kv" style="margin-top:10px">
        <dt>Estimated Pc</dt><dd>${pcHtml(a.log_pc)}</dd>
        <dt>80% interval</dt><dd>${esc(pcText(a.log_pc_lo))} to ${esc(pcText(a.log_pc_hi))}</dd>
        <dt>Chance Pc ≥ 1e-6</dt><dd>${esc(pct(a.p_pc_elevated, 1))}</dd>
        <dt>Chance Pc ≥ 1e-4</dt><dd>${esc(pct(a.p_pc_critical, 1))}</dd>
        ${a.forecast_tier != null ? `<dt>Forecast tier at TCA</dt><dd>${tierPill(num(a.forecast_tier))}</dd>` : ''}
        ${a.forecast && a.forecast.tier != null ? `<dt>Forecast at the final CDM</dt><dd>${tierPill(num(a.forecast.tier))}${Array.isArray(a.forecast.p_tier) ? ` <small>P(Critical) ${esc(pct(a.forecast.p_tier[2]))}</small>` : ''}</dd>` : ''}
        ${a.model ? `<dt>Model</dt><dd><small>${esc(a.model.set || (a.model.trained ? 'trained' : 'fallback'))}</small></dd>` : ''}
      </dl>
      ${Array.isArray(a.input_warnings) && a.input_warnings.length ? `<p class="warn-line" style="margin-top:10px">${a.input_warnings.map(esc).join('<br>')}</p>` : ''}
      ${Array.isArray(a.ignored_fields) && a.ignored_fields.length ? `<p class="hint">Ignored fields (not model inputs): ${a.ignored_fields.map(esc).join(', ')}</p>` : ''}
      <h4 class="sub-h">Why</h4>${reasonsHtml(a.reasons)}
      <div class="form-actions"><button type="button" class="btn sm" data-act="log-assess">Log this assessment</button></div>
      <details class="raw" style="margin-top:12px"><summary>Request sent</summary><pre>${esc(JSON.stringify(body, null, 2))}</pre></details>`;
    S.lastAssess = { a, body };
  }

  // ================================================================ wiring
  function setFeedFilter(f) {
    S.feedFilter = f;
    $$('#feed-filter button').forEach((b) => { const on = b.dataset.f === f; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on ? 'true' : 'false'); });
  }
  function setSeg(container, attr, value) {
    $$(`${container} button`).forEach((b) => { const on = b.dataset[attr] === String(value); b.classList.toggle('on', on); b.setAttribute('aria-pressed', on ? 'true' : 'false'); });
  }
  function moveSelection(dir) {
    if (S.view === 'evaluator') {
      const rows = testRows(); if (!rows.length) return;
      let i = rows.findIndex((e) => String(e.event_id) === String(S.test.selected));
      i = clamp(i + dir, 0, rows.length - 1);
      selectTestEvent(rows[i].event_id);
      return;
    }
    if (S.view !== 'live') return;
    const { rows } = feedRows(); if (!rows.length) return;
    let i = rows.findIndex((r) => r.id === S.selectedId);
    i = i < 0 ? 0 : clamp(i + dir, 0, rows.length - 1);
    selectEvent(rows[i].id, { fly: true });
  }
  function clearSearch() {
    $('#q').value = ''; S.search = '';
    renderFeed(); updatePoints();
    if (S.view === 'evaluator') renderTestQueue();
  }
  const isTyping = (el) => el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);

  function wire() {
    window.addEventListener('hashchange', route);
    // search
    const q = $('#q');
    let qt = 0;
    q.addEventListener('input', () => {
      clearTimeout(qt);
      qt = setTimeout(() => {
        S.search = q.value.trim();
        renderFeed(); updatePoints();
        if (S.view === 'evaluator') renderTestQueue();
      }, 120);
    });
    q.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { const { rows } = feedRows(); if (S.view === 'live' && rows.length) selectEvent(rows[0].id, { fly: true }); }
      if (e.key === 'Escape') { clearSearch(); q.blur(); }
    });
    // window chips
    $('#win-chips').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-w]'); if (!b) return;
      S.windowH = Number(b.dataset.w);
      lsSet('cw.window', S.windowH);
      $$('#win-chips button').forEach((x) => { const on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-pressed', on ? 'true' : 'false'); });
      S.hourFilter = null;
      if (S.view !== 'live') location.hash = '#live';
      loadLive();
      renderWarnings();
    });
    // sidebar
    $('#feed-filter').addEventListener('click', (e) => { const b = e.target.closest('button[data-f]'); if (!b) return; setFeedFilter(b.dataset.f); renderFeed(); });
    $('#feed-sort').addEventListener('change', (e) => { S.feedSort = e.target.value; renderFeed(); });
    $('#tq-pred').addEventListener('click', (e) => { const b = e.target.closest('button[data-t]'); if (!b) return; S.test.pred = b.dataset.t; setSeg('#tq-pred', 't', b.dataset.t); renderTestQueue(); });
    $('#tq-review').addEventListener('change', (e) => { S.test.review = e.target.checked; renderTestQueue(); });
    $('#tq-dis').addEventListener('change', (e) => { S.test.dis = e.target.checked; renderTestQueue(); });
    $('#tq-under').addEventListener('change', (e) => { S.test.under = e.target.checked; renderTestQueue(); });
    $('#tq-sort').addEventListener('change', (e) => { S.test.sort = e.target.value; renderTestQueue(); });
    // hour strip: arrow keys move between bars (one tab stop for the whole strip)
    $('#hours').addEventListener('keydown', (e) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      const bars = $$('#hours button');
      const i = bars.indexOf(document.activeElement);
      if (i < 0) return;
      e.preventDefault();
      const j = e.key === 'Home' ? 0 : (e.key === 'End' ? bars.length - 1 : clamp(i + (e.key === 'ArrowRight' ? 1 : -1), 0, bars.length - 1));
      bars[i].tabIndex = -1; bars[j].tabIndex = 0; bars[j].focus();
    });
    // legend options
    $('#lg-elev').addEventListener('change', (e) => { S.layers.elev = e.target.checked; lsSet('cw.layers', S.layers); updatePoints(); });
    $('#lg-labels').addEventListener('change', (e) => { S.layers.labels = e.target.checked; lsSet('cw.layers', S.layers); applyLayerVisibility(); });
    // play bar
    $('#pb-play').addEventListener('click', () => { if (!G.viewer) return; G.viewer.clock.shouldAnimate = !G.viewer.clock.shouldAnimate; updatePlaybar(true); });
    $('#pb-speed').addEventListener('click', () => {
      const speeds = [10, 30, 60, 120, 300]; const cur = G.speed || 60;
      G.speed = speeds[(speeds.indexOf(cur) + 1) % speeds.length];
      if (G.viewer) G.viewer.clock.multiplier = G.speed;
      $('#pb-speed').innerHTML = `${G.speed}&times;`;
    });
    $('#pb-tca').addEventListener('click', () => { if (!G.sel) return; G.viewer.clock.currentTime = G.sel.tcaJd.clone(); G.viewer.clock.shouldAnimate = false; updatePlaybar(true); });
    $('#pb-fit').addEventListener('click', fitPaths);
    const scrub = $('#pb-scrub');
    scrub.addEventListener('input', () => {
      if (!G.sel) return;
      G.scrubbing = true;
      G.viewer.clock.shouldAnimate = false;
      G.viewer.clock.currentTime = window.Cesium.JulianDate.addSeconds(G.sel.start, G.sel.span * Number(scrub.value) / 1000, new window.Cesium.JulianDate());
      updatePlaybar(true);
    });
    scrub.addEventListener('change', () => { G.scrubbing = false; });
    // decision log
    $('#log-btn').addEventListener('click', openLog);
    $('#ld-close').addEventListener('click', closeLog);
    $('#scrim').addEventListener('click', closeLog);
    $('#ld-filter').addEventListener('change', renderLogDrawer);
    $('#ld-export').addEventListener('click', exportLog);
    $('#ld-clear').addEventListener('click', () => {
      if (!S.log.length) return;
      if (!window.confirm(`Clear all ${S.log.length} logged decisions from this browser? Export the CSV first if you need them.`)) return;
      S.log = []; saveLog(); renderLogDrawer(); renderFeed(); if (S.selectedId) renderDetail();
    });
    $('#analyst').addEventListener('input', (e) => setAnalyst(e.target.value));
    // assess
    $('#assess-form').addEventListener('submit', submitAssess);
    $$('[data-preset]').forEach((b) => b.addEventListener('click', () => applyPreset(PRESETS[b.dataset.preset])));
    $('#assess-reset').addEventListener('click', () => { FIELDS.forEach((f) => { const el = $(`#f-${f.k}`); if (el && f.type !== 'select') el.value = ''; }); $('#preset-note').textContent = ''; checkAssessGeometry(); });
    $('#assess-sw').addEventListener('click', async () => {
      if (!S.sw) await loadSpaceWeather();
      const f = get(S.sw, 'f107.latest'); const a = get(S.sw, 'ap.latest');
      if (f == null && a == null) { toast('Space weather is unavailable right now. Enter F10.7 and Ap by hand.', true); return; }
      if (f != null) $('#f-F10').value = f;
      if (a != null) $('#f-AP').value = a;
      $('#preset-note').textContent = `F10.7 and Ap set from ${get(S.sw, 'f107.source', 'NOAA SWPC')}, fetched ${fmtUTC(S.sw.fetched_at)}.`;
    });

    // delegated clicks
    document.addEventListener('click', (e) => {
      const selEl = e.target.closest('[data-sel]');
      if (selEl) { e.preventDefault(); selectEvent(selEl.getAttribute('data-sel'), { fly: true }); return; }
      const testEl = e.target.closest('[data-test]');
      if (testEl) { selectTestEvent(Number(testEl.getAttribute('data-test'))); return; }
      const dec = e.target.closest('[data-decide]');
      if (dec) {
        const box = dec.closest('.actions');
        if (box) decide(box, dec.dataset.decide);
        return;
      }
      const qBtn = e.target.closest('[data-q]');
      if (qBtn) { e.preventDefault(); $('#q').value = qBtn.dataset.q; S.search = qBtn.dataset.q; setFeedFilter('all'); renderFeed(); updatePoints(); return; }
      const hourEl = e.target.closest('[data-hour]');
      if (hourEl) { e.preventDefault(); setHourFilter(Number(hourEl.dataset.hour)); return; }
      const cdmRow = e.target.closest('[data-cdm]');
      if (cdmRow) { S.test.cdmIdx = Number(cdmRow.dataset.cdm); renderTestDetail(); return; }
      const go = e.target.closest('[data-goto]');
      if (go) { e.preventDefault(); location.hash = `#evaluator/${go.dataset.goto}`; return; }
      const sp = e.target.closest('#metric-split button[data-split]');
      if (sp) { S.metricSplit = sp.dataset.split; renderMetrics(); return; }
      const act = e.target.closest('[data-act]');
      if (!act) return;
      const a = act.dataset.act;
      if (a === 'deselect') deselect();
      else if (a === 'retry-live') loadLive();
      else if (a === 'retry-track' && S.selectedRow) loadTrack(S.selectedRow);
      else if (a === 'retry-test') ensureTestEvents(true);
      else if (a === 'retry-test-detail' && S.test.selected != null) selectTestEvent(S.test.selected);
      else if (a === 'retry-metrics') ensureMetrics(true);
      else if (a === 'clear-hour') setHourFilter(null);
      else if (a === 'clear-search') clearSearch();
      else if (a === 'feed-all') { setFeedFilter('all'); renderFeed(); }
      else if (a === 'log-assess' && S.lastAssess) {
        if (!S.analyst) { openLog(); $('#analyst').focus(); toast('Add your initials in the decision log first.', true); return; }
        const { a: as, body } = S.lastAssess;
        S.log.push({ ts: new Date().toISOString(), analyst: S.analyst, view: 'assess', event_id: body.event_id, pair: `Manual CDM, event ${body.event_id}`, action: 'acknowledge',
          note: 'Manual assessment logged', model_tier: TIERS[num(as.tier)], needs_review: !!as.needs_review, log_pc: as.log_pc, miss_distance_m: body.miss_distance, time_to_tca_d: body.time_to_tca, source: 'Assess form' });
        saveLog(); toast('Assessment logged.');
      }
    });
    document.addEventListener('input', (e) => { if (e.target.classList && e.target.classList.contains('analyst-inline')) setAnalyst(e.target.value); });
    document.addEventListener('keydown', (e) => {
      const card = e.target.closest && e.target.closest('.item[role="button"]');
      if (card && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); card.click(); return; }
      if (e.key === 'Escape') {
        if ($('#logdrawer').classList.contains('open')) { closeLog(); return; }
        if (S.view === 'live' && S.selectedId) { deselect(); return; }
      }
      if (isTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === '/') { e.preventDefault(); $('#q').focus(); }
      else if (e.key === 'j' || (e.key === 'ArrowDown' && card)) { e.preventDefault(); moveSelection(1); }
      else if (e.key === 'k' || (e.key === 'ArrowUp' && card)) { e.preventDefault(); moveSelection(-1); }
    });
    window.addEventListener('resize', () => { clearTimeout(wire._rt); wire._rt = setTimeout(resizePlots, 150); });
  }

  // ================================================================ boot
  function boot() {
    // read-only debugging handle (state + globe), no side effects
    window.ConjunctionWatch = { state: S, globe: G, mock: MOCK };
    loadLog();
    renderLogCount();
    $('#analyst').value = S.analyst;
    if (MOCK) $('#mock-banner').hidden = false;
    const w = lsGet('cw.window', 24);
    if ([6, 12, 24, 48, 72].includes(w)) {
      S.windowH = w;
      $$('#win-chips button').forEach((x) => { const on = Number(x.dataset.w) === w; x.classList.toggle('on', on); x.setAttribute('aria-pressed', on ? 'true' : 'false'); });
    }
    const layers = lsGet('cw.layers', null);
    if (window.innerWidth <= 900) S.layers.labels = false;   // labels crowd a phone-sized globe
    if (layers && typeof layers === 'object') {
      if (typeof layers.elev === 'boolean') S.layers.elev = layers.elev;
      if (typeof layers.labels === 'boolean') S.layers.labels = layers.labels;
    }
    $('#lg-elev').checked = S.layers.elev;
    $('#lg-labels').checked = S.layers.labels;
    wire();
    tickClock();
    setInterval(tickClock, 1000);
    setInterval(refreshCountdowns, 15000);
    renderFeed(); renderWarnings(); renderInsights(); renderStatus();
    route();
    initGlobe();
    loadLive();
    loadSpaceWeather();
    loadHealth();
    setInterval(() => loadLive({ quiet: !!S.live }), POLL_LIVE_MS);
    setInterval(loadSpaceWeather, POLL_SW_MS);
    setInterval(loadHealth, POLL_HEALTH_MS);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
