/* Conjunction Watch - operator dashboard (vanilla JS, no build step).
 * Contract: docs/api_contract.md. All distances in metres, speeds in m/s, log_pc = log10(Pc),
 * tiers 0 Safe / 1 Elevated / 2 Critical. ?mock=1 reads dashboard/mock/*.json instead of /api.
 * Decision support only: every recommendation is shown with its evidence and the analyst decides.
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
  const POLL_SW_MS = 10 * 60000;
  const POLL_HEALTH_MS = 60000;
  const TIERS = ['Safe', 'Elevated', 'Critical'];
  const TIER_CSS = ['t0', 't1', 't2'];
  const COLORS = {
    crit: '#FF5D6C', elev: '#F6B23C', safe: '#3ED598', review: '#B78BFF',
    a: '#00BCD4', b: '#E67E22', text: '#F8FBFF', text2: '#C3D2E2', muted: '#94A6B9',
    grid: 'rgba(82,100,119,.35)', line: 'rgba(82,100,119,.7)', panel: '#222D39', bg: '#18212B',
    primary: '#1A73E8', primaryText: '#8AB4F8', live: '#49EFA4',
  };
  const TIER_COLOR = [COLORS.safe, COLORS.elev, COLORS.crit];
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
    live: null, liveErr: null, liveExtraErr: [], liveLoading: false, liveFetchedAt: null, liveSeq: 0, onlyPoints: false,
    sw: null, swErr: null,
    health: null, healthErr: null,
    feedFilter: 'critical', feedSort: 'risk', hourFilter: null,
    search: '',
    selectedId: null, selectedRow: null, track: null, trackErr: null, trackLoading: false, trackSeq: 0,
    layers: { crit: true, elev: true, safe: true, review: true, points: true, paths: true, labels: true },
    minSev: 0,
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
    if (Math.abs(m) < 1000) return `${Math.round(m)} m`;
    if (Math.abs(m) < 10000) return `${(m / 1000).toFixed(2)} km`;
    if (Math.abs(m) < 1e6) return `${(m / 1000).toFixed(1)} km`;
    return `${Math.round(m / 1000).toLocaleString('en-US')} km`;
  }
  function fmtSpeed(ms) { ms = num(ms); return ms == null ? '–' : `${(ms / 1000).toFixed(2)} km/s`; }
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
    return `<span title="${esc(t + (extra ? ' · ' + extra : ''))}">${esc(pcText(lp))}</span>`;
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
  function fmtUTC(iso, withDate) {
    const t = parseT(iso);
    if (t == null) return '–';
    const d = new Date(t);
    const hms = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}Z`;
    if (withDate === false) return hms;
    return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${hms}`;
  }
  function fmtDur(ms, long) {
    const neg = ms < 0;
    let s = Math.abs(ms) / 1000;
    const d = Math.floor(s / 86400); s -= d * 86400;
    const h = Math.floor(s / 3600); s -= h * 3600;
    const m = Math.floor(s / 60); s -= m * 60;
    let out;
    if (d > 0) out = `${d}d ${pad(h)}h`;
    else if (h > 0) out = `${h}h ${pad(m)}m`;
    else if (m > 0 || !long) out = `${m}m ${pad(Math.floor(s))}s`;
    else out = `${Math.floor(s)}s`;
    return neg ? `-${out}` : out;
  }
  function countdown(iso) {
    const t = parseT(iso);
    if (t == null) return '–';
    const dt = t - Date.now();
    return dt >= 0 ? `TCA in ${fmtDur(dt)}` : `TCA ${fmtDur(-dt)} ago`;
  }
  function ago(iso) {
    const t = parseT(iso);
    if (t == null) return 'unknown';
    const dt = Date.now() - t;
    if (dt < 0) return `in ${fmtDur(-dt)}`;
    if (dt < 60000) return 'just now';
    return `${fmtDur(dt)} ago`;
  }
  function tierPill(t, prefix) {
    if (!isNum(t) || t < 0 || t > 2) return '<span class="pill nt">No tier</span>';
    return `<span class="pill ${TIER_CSS[t]}">${prefix ? `<span class="pfx">${esc(prefix)}</span> ` : ''}${TIERS[t]}</span>`;
  }
  const reviewPill = () => '<span class="pill rv" title="Uncertainty interval straddles a tier boundary, or the worst-case Pc bound reaches 1e-4: analyst review required">Review</span>';
  function pairHtml(a, b) {
    return `${esc(a || '?')}<span class="x">×</span>${esc(b || '?')}`;
  }
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
      rows = sortRows(rows, 'risk').slice(0, num(params.limit) || 500);
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
    return { known: true, agrees, sgp, miss, chk, text: `SGP4 with our element sets ${fmtDist(sgp)} vs SOCRATES ${fmtDist(miss)}` };
  }
  const SGP4_DISAGREE_TIP = 'Our SGP4 propagation with the current CelesTrak element sets does not reproduce the SOCRATES screening (different distance-rule tier or more than 1 km apart). The element sets differ (newer TLE, or a manoeuvre since screening): re-screen before acting.';
  // Live rows: the screening model's Pc estimate on SOCRATES inputs (out of distribution) is only
  // informative when its 80% interval is narrow and the classifier and quantile heads agree.
  function pcInfo(r) {
    const a = r.assessment || {};
    const lo = num(a.log_pc_lo); const hi = num(a.log_pc_hi); const pcc = num(a.p_pc_critical);
    const thr = firstNum(get(S.live, 'model.thresholds.p_critical'), get(S.health, 'model.thresholds.screen.p_critical'));
    const width = lo != null && hi != null ? hi - lo : null;
    const disagree = a.heads_disagree === true || (hi != null && pcc != null && thr != null && hi < -4 && pcc >= thr);
    const wide = width != null && width > 6;
    const why = [wide ? `the 80% interval spans ${width.toFixed(0)} decades` : null,
      disagree ? `the classifier gives P(Pc ≥ 1e-4) = ${pct(pcc)} while the 90th-percentile Pc is ${pcText(hi)}` : null].filter(Boolean).join('; ');
    return { informative: !(wide || disagree), wide, disagree, width,
      tip: why ? `Uninformative (out of distribution): ${why}. Live SOCRATES inputs have no covariance, so use SOCRATES' max Pc as the screening reference.` : '' };
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

  // ================================================================ header: clock, pills, ticker, space weather
  function tickClock() {
    const d = new Date();
    const el = $('#clock-t');
    if (el) el.textContent = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  }
  function refreshCountdowns() {
    $$('[data-tca]').forEach((el) => {
      const t = parseT(el.getAttribute('data-tca'));
      el.textContent = countdown(el.getAttribute('data-tca'));
      if (el.classList.contains('cd')) el.classList.toggle('soon', t != null && t - Date.now() < 6 * 3.6e6);
    });
  }

  function renderLivePill() {
    const el = $('#live-pill');
    if (!el) return;
    const txt = $('.lp-text', el);
    el.className = 'live-pill';
    const d = S.live;
    const lines = [];
    if (d) {
      lines.push(`Source: ${d.source || 'unknown'}`);
      lines.push(`Data as of: ${fmtUTC(d.data_as_of)} (${ago(d.data_as_of)})`);
      lines.push(`Fetched by server: ${fmtUTC(d.fetched_at)} (${ago(d.fetched_at)})`);
      if (isNum(d.n_total)) lines.push(`Rows in feed: ${d.n_total.toLocaleString('en-US')}`);
    }
    if (S.liveFetchedAt) lines.push(`Last dashboard poll: ${fmtUTC(new Date(S.liveFetchedAt).toISOString())}`);
    const ml = get(S.health, 'model_loaded');
    if (ml != null) lines.push(`Model: ${ml ? 'trained models loaded' : 'NOT loaded - physics / max-Pc fallback estimates'}`);
    if (get(S.health, 'model.error')) lines.push(`Model reload problem (previous model kept): ${String(S.health.model.error).slice(0, 180)}`);
    if (S.liveErr) lines.push(`Last error: ${S.liveErr}`);
    if (d && d.refresh_error) lines.push(`Server refresh error: ${d.refresh_error}`);
    if (d && d.parse_error) lines.push(`Newest SOCRATES file unreadable (previous one served): ${String(d.parse_error).slice(0, 180)}`);
    if (d && d.files && typeof d.files === 'object') {
      Object.keys(d.files).forEach((k) => { const f = d.files[k]; if (f && f.error) lines.push(`sort-${k}.csv: ${String(f.error).slice(0, 180)}`); });
    }
    (S.liveExtraErr || []).forEach((x) => lines.push(`Could not load ${x}`));
    // the other live inputs: element sets (flight paths, collision points), SATCAT (object types), SWPC (F10/AP)
    const SRC_NAMES = { gp: 'Element sets (CelesTrak GP)', satcat: 'SATCAT object types', swpc: 'Space weather (NOAA SWPC)' };
    const srcBad = [];
    const ls = get(S.health, 'live_sources') || {};
    Object.keys(SRC_NAMES).forEach((k) => {
      const v = ls[k];
      if (v && v !== 'ok' && v !== 'disabled') {
        const det = get(S.health, `live_sources_detail.${k}`) || {};
        const err = det.error || get(det, 'bulk.cache.error') || get(det, 'bulk.error') || '';
        srcBad.push(`${SRC_NAMES[k]}: ${v.toUpperCase()}${err ? ' - ' + String(err).slice(0, 160) : ''}`);
      }
    });
    if (S.healthErr) srcBad.push(`Server health check failed: ${S.healthErr}`);
    lines.push(...srcBad);
    let cls = 'is-wait'; let label = 'CONNECTING';
    if (MOCK) { cls = 'is-mock'; label = 'MOCK'; }
    else if (!d && S.liveErr) { cls = 'is-off'; label = 'OFFLINE'; }
    else if (d && (d.stale || S.liveErr)) { cls = 'is-stale'; label = 'STALE'; if (S.liveErr) lines.unshift('API unreachable - showing the last data received'); else lines.unshift('A source was unreachable - server is serving its last cached snapshot'); }
    else if (d && (srcBad.length || (S.liveExtraErr || []).length)) { cls = 'is-stale'; label = 'LIVE*'; lines.unshift('SOCRATES screening is live, but another input is stale or failed (details below)'); }
    else if (d) { cls = ''; label = 'LIVE'; }
    if (cls) el.classList.add(cls);
    txt.textContent = label;
    el.title = lines.join('\n') || 'Connecting to the live feed';
    el.setAttribute('aria-label', `${label}. ${lines.join('. ')}`);
  }

  function renderSpaceWeather() {
    const el = $('#sw-chip');
    if (!el) return;
    const d = S.sw;
    if (!d) {
      el.innerHTML = `F10.7 – <i>|</i> Ap – <i>|</i> Kp –`;
      el.title = S.swErr ? `Space weather unavailable: ${S.swErr}` : 'Loading space weather (NOAA SWPC)';
      return;
    }
    const f = get(d, 'f107.latest'); const a = get(d, 'ap.latest'); const k = get(d, 'kp.latest');
    el.innerHTML = `F10.7 ${esc(fmtNum(f, 0))} <i>|</i> Ap ${esc(fmtNum(a, 0))} <i>|</i> Kp ${esc(fixed(k, 1))}`;
    el.classList.toggle('is-stale', !!d.stale);
    const lines = [
      `F10.7 solar flux: ${fmtNum(f, 1)} sfu (${get(d, 'f107.source', 'NOAA SWPC')}${get(d, 'f107.time') ? ', ' + fmtUTC(get(d, 'f107.time')) : ''})`,
      `Ap geomagnetic index: ${fmtNum(a, 1)} (${get(d, 'ap.source', 'NOAA SWPC')})`,
      `Kp: ${fixed(k, 2)}${get(d, 'kp.time') ? ' at ' + fmtUTC(get(d, 'kp.time')) : ''}`,
      `Fetched: ${fmtUTC(d.fetched_at)} (${ago(d.fetched_at)})${d.stale ? ' - STALE (served from cache)' : ''}`,
      'F10 and AP are model inputs for live screening.',
    ];
    el.title = lines.join('\n');
  }

  function renderTicker() {
    const track = $('#ticker-track');
    if (!track) return;
    const rows = sortRows(liveRows().filter((r) => (tierOf(r) || 0) >= 1 && num(r.hours_to_tca) >= 0), 'risk').slice(0, 14);
    if (!rows.length) {
      track.classList.remove('run');
      track.innerHTML = `<span class="tick-empty">${S.live ? 'No Elevated or Critical conjunctions in this window' : (S.liveErr ? 'Live feed unavailable' : 'Waiting for live conjunctions…')}</span>`;
      return;
    }
    const item = (r) => {
      const t = tierOf(r);
      return `<button type="button" class="tick-item" data-sel="${esc(r.id)}" title="Select this conjunction">` +
        `${tierPill(t)} <b>${pairHtml(get(r, 'sat1.name'), get(r, 'sat2.name'))}</b>` +
        `<span class="sep">·</span>${esc(fmtDist(r.miss_distance))}` +
        `<span class="sep">·</span>max Pc ${esc(maxPcText(r))}` +
        `<span class="sep">·</span><span data-tca="${esc(r.tca)}">${esc(countdown(r.tca))}</span></button>`;
    };
    const html = rows.map(item).join('');
    track.innerHTML = html + html.replace(/<button /g, '<button tabindex="-1" aria-hidden="true" ');
    track.style.setProperty('--tick-dur', `${Math.max(30, rows.length * 7)}s`);
    track.classList.add('run');
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
      // loaded with two small server-side filtered requests and merged (optional: older servers ignore
      // the filter params and the duplicates are dropped).
      const base = { window_h: S.windowH, sort: 'risk', points: 1 };
      const [main, ...extra] = await Promise.allSettled([
        api('/api/live/conjunctions', Object.assign({ limit: LIVE_LIMIT }, base)),
        api('/api/live/conjunctions', Object.assign({ limit: EXTRA_LIMIT, tier: 1 }, base)),
        api('/api/live/conjunctions', Object.assign({ limit: EXTRA_LIMIT, needs_review: 1 }, base)),
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
          extraErr.push(`${['the top Elevated', 'the needs-review'][i]} conjunctions (${x.status === 'rejected' ? (x.reason && x.reason.message) || x.reason : 'no conjunctions list'})`);
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
      if (!opts || !opts.quiet) toast(`Live update failed: ${S.liveErr}${S.live ? ' - showing the last data received' : ''}`, true);
    } finally {
      if (seq === S.liveSeq) S.liveLoading = false;
    }
    renderLiveAll();
    if (S.pendingSelect && S.live) {
      const id = S.pendingSelect;
      S.pendingSelect = null;
      if (rowById(id)) selectEvent(id, { fly: true });
      else toast('The linked conjunction is not in the current window.', true);
    }
  }
  function renderLiveAll() {
    renderLivePill();
    renderTicker();
    renderFeed();
    renderKpiCard();
    renderOverlayCounts();
    updatePoints();
    if (S.selectedId && $('#detail').classList.contains('open')) renderDetail();
  }
  async function loadSpaceWeather() {
    try { S.sw = await api('/api/live/spaceweather'); S.swErr = null; } catch (e) { S.swErr = e.message; }
    renderSpaceWeather();
  }
  async function loadHealth() {
    try { S.health = await api('/api/health'); S.healthErr = null; } catch (e) { S.healthErr = e.message; }
    renderLivePill();
    renderSideFoot();
  }

  // ================================================================ sidebar: live feed
  function feedRows() {
    let rows = liveRows();
    if (S.search) rows = rows.filter((r) => matchesSearch(r, S.search));
    if (S.hourFilter != null) rows = rows.filter((r) => Math.floor(num(r.hours_to_tca)) === S.hourFilter);
    if (S.onlyPoints) rows = rows.filter((r) => r.tca_point && isNum(r.tca_point.lat));
    const counts = {
      critical: rows.filter((r) => tierOf(r) === 2).length,
      elevated: rows.filter((r) => tierOf(r) === 1).length,
      review: rows.filter((r) => get(r, 'assessment.needs_review') === true).length,
      all: rows.length,
    };
    const f = S.feedFilter;
    if (f === 'critical') rows = rows.filter((r) => tierOf(r) === 2);
    else if (f === 'elevated') rows = rows.filter((r) => tierOf(r) === 1);
    else if (f === 'review') rows = rows.filter((r) => get(r, 'assessment.needs_review') === true);
    return { rows: sortRows(rows, S.feedSort), counts };
  }
  const FEED_TITLES = { critical: 'Critical events', elevated: 'Elevated events', review: 'Needs review', all: 'All events' };
  function renderFeed() {
    const list = $('#feed');
    if (!list) return;
    const { rows, counts } = feedRows();
    $$('#feed-filter button').forEach((b) => { $('span', b).textContent = counts[b.dataset.f]; });
    $('#feed-title').textContent = FEED_TITLES[S.feedFilter];
    const fc = $('#feed-count');
    fc.textContent = rows.length;
    fc.className = `count ${S.feedFilter === 'critical' ? 'count-crit' : ''}`;
    $('#rail-count').textContent = counts.critical;
    const hourChip = $('#feed-hour');
    if (S.hourFilter != null) {
      hourChip.hidden = false;
      hourChip.innerHTML = `TCA +${S.hourFilter}–${S.hourFilter + 1} h <button type="button" aria-label="Clear hour filter" data-act="clear-hour">×</button>`;
    } else hourChip.hidden = true;
    const total = get(S.live, 'warnings.n_in_window');
    const past = num(get(S.live, 'warnings.n_recent_past')) || 0;
    $('#feed-meta').textContent = S.live ? `${liveRows().length}${isNum(total) && total + past > liveRows().length ? ' of ' + (total + past).toLocaleString('en-US') : ''} in ${S.windowH} h` : '';
    const note = $('#feed-note');
    const W = get(S.live, 'warnings') || {};
    const notes = [];
    const extraErr = S.liveExtraErr || [];
    if (extraErr.length) notes.push(`<span class="warn-txt"><b>Could not load</b> ${esc(extraErr.join('; '))}: the Elevated / Needs-review tabs may be incomplete. <button type="button" class="btn sm" data-act="retry-live">Retry</button></span>`);
    if (S.live && isNum(total) && total + past > liveRows().length) {
      const nMain = isNum(S.live.n_main) ? S.live.n_main : liveRows().length;
      const nExtra = liveRows().length - nMain;
      notes.push(`Loaded the <b>${nMain}</b> highest-risk${nExtra > 0 ? ` plus <b>${nExtra}</b> top Elevated / needs-review` : ''} of <b>${(total + past).toLocaleString('en-US')}</b> conjunctions in the feed (API limit). Next ${S.windowH} h totals: ` +
        `<b>${fmtNum(W.n_critical, 0)}</b> Critical · <b>${fmtNum(W.n_elevated, 0)}</b> Elevated${isNum(W.n_needs_review) ? ` · <b>${fmtNum(W.n_needs_review, 0)}</b> need review` : ''}${past ? ` (plus ${fmtNum(past, 0)} whose TCA passed in the last hour)` : ''}. Narrow the window to see more of the queue.`);
    }
    if (S.live && rows.length) {
      const withPt = rows.filter((r) => r.tca_point && isNum(r.tca_point.lat)).length;
      const top = rows.slice(0, 20);
      const topPt = top.filter((r) => r.tca_point && isNum(r.tca_point.lat)).length;
      const nDis = rows.filter((r) => { const c = pointCheck(r); return c.known && !c.agrees; }).length;
      notes.push(`Collision points: <b>${withPt}</b> of ${rows.length} listed (${topPt} of the top ${top.length})${nDis ? ` · <span class="warn-txt">${nDis} where SGP4 disagrees with SOCRATES</span>` : ''}${S.onlyPoints ? ' · showing only rows with a point' : ''}.`);
    }
    note.hidden = !notes.length;
    note.innerHTML = notes.join('<br>');

    if (!S.live) {
      if (S.liveErr) {
        list.innerHTML = `<li class="state err-state"><b>Live feed unavailable.</b><br>${esc(S.liveErr)}<br><br><button type="button" class="btn sm" data-act="retry-live">Retry</button></li>`;
      } else {
        list.innerHTML = '<li class="skel"></li><li class="skel"></li><li class="skel"></li><li class="skel"></li>';
      }
      renderSideFoot();
      return;
    }
    if (!rows.length) {
      // never claim "nothing" when the window totals say otherwise (e.g. a filtered request failed)
      const inWin = { critical: W.n_critical, elevated: W.n_elevated, review: W.n_needs_review, all: W.n_in_window }[S.feedFilter];
      const missing = !S.search && S.hourFilter == null && !S.onlyPoints && num(inWin) > 0;
      const why = S.search ? `No loaded conjunction matches “${esc(S.search)}”.` :
        (S.onlyPoints ? 'No listed conjunction has a collision point yet (element sets pending).' :
          (missing ? `<b>${fmtNum(inWin, 0)}</b> in the window, but none could be loaded${extraErr.length ? ` (${esc(extraErr.join('; '))})` : ''}. <button type="button" class="btn sm" data-act="retry-live">Retry</button>` :
            (S.feedFilter === 'critical' ? `No Critical conjunctions in the next ${S.windowH} h.` :
              S.feedFilter === 'review' ? 'Nothing is flagged for review.' : 'No conjunctions in this window.')));
      list.innerHTML = `<li class="state${missing ? ' err-state' : ''}">${why}${S.feedFilter !== 'all' && !missing ? ' <button type="button" class="btn sm ghost" data-act="feed-all">Show all</button>' : ''}</li>`;
      renderSideFoot();
      return;
    }
    const decs = latestDecisions();
    list.innerHTML = rows.map((r) => cardHtml(r, decs[r.id])).join('');
    renderSideFoot();
  }
  function cardHtml(r, dec) {
    const a = r.assessment || {};
    const t = tierOf(r);
    const tcaMs = parseT(r.tca);
    const soon = tcaMs != null && tcaMs - Date.now() < 6 * 3.6e6;
    const s1 = r.sat1 || {}; const s2 = r.sat2 || {};
    const tle = [num(s1.tle_age_d), num(s2.tle_age_d)];
    const hit = S.search && matchesSearch(r, S.search);
    const chk = pointCheck(r);
    const ptTxt = r.tle_status === 'pending' ? ' · point pending TLE' : (r.tle_status === 'missing' ? ' · no TLE' :
      (chk.known && !r.tca_point ? ' · not plotted' : ''));
    const pi = pcInfo(r);
    const ruleCrit = num(a.md_tier) === 2;
    const pcc = num(a.p_pc_critical);
    const pcTitle = `P(Pc ≥ 1e-4) = ${pct(pcc)} from the screening model (Pc part of the rule only)${pi.informative ? '' : '. ' + pi.tip}`;
    const sgChip = chk.known && !chk.agrees ? `<span class="pill sg" title="${esc(chk.text + '. ' + SGP4_DISAGREE_TIP)}">SGP4 disagrees</span>` : '';
    const estPc = pi.informative
      ? `<span>est. Pc <b>${pcHtml(a.log_pc)}</b> <small>[${esc(pcText(a.log_pc_lo))}–${esc(pcText(a.log_pc_hi))}]</small></span>`
      : `<span class="pc-un" title="${esc(pi.tip)}">est. Pc uninformative</span>`;
    const bar = ruleCrit
      ? `<div class="card-bar rule" title="SOCRATES miss distances come from TLEs without covariance, so no probability is attached to the distance rule">Critical by distance rule (screening miss ${esc(fmtDist(r.miss_distance))} &lt; 1 km)</div>`
      : `<div class="card-bar" title="${esc(pcTitle)}"><span>P(Pc≥1e-4)</span><span class="pbar${pi.informative ? '' : ' un'}"><i style="width:${clamp((pcc || 0) * 100, 0, 100).toFixed(1)}%"></i></span><span>${esc(pct(pcc))}</span></div>`;
    return `<li><div class="card${S.selectedId === r.id ? ' sel' : ''}${hit ? ' hit' : ''}" role="button" tabindex="0" data-sel="${esc(r.id)}" aria-label="${esc(`${TIERS[t] || 'Unknown'}: ${s1.name} and ${s2.name}, miss distance ${fmtDist(r.miss_distance)}${chk.known && !chk.agrees ? ', SGP4 disagrees' : ''}`)}">
      <div class="card-top">${tierPill(t)}${a.needs_review ? reviewPill() : ''}${sgChip}<span class="cd${soon ? ' soon' : ''}" data-tca="${esc(r.tca)}">${esc(countdown(r.tca))}</span></div>
      <div class="card-title">${pairHtml(s1.name, s2.name)}</div>
      <div class="card-meta"><span>miss <b>${esc(fmtDist(r.miss_distance))}</b></span><span><b>${esc(fmtSpeed(r.relative_speed))}</b></span><span title="SOCRATES worst-case Pc over covariance scaling (screening reference)">max Pc <b>${esc(maxPcText(r))}</b></span>${estPc}</div>
      ${bar}
      <div class="card-foot"><span>${esc(s1.type || '?')} × ${esc(s2.type || '?')}</span><span>·</span><span title="TLE age (days since epoch) for primary / secondary">TLE ${tle[0] == null ? '?' : tle[0].toFixed(1)} / ${tle[1] == null ? '?' : tle[1].toFixed(1)} d${esc(ptTxt)}</span>${dec ? `<span class="dec">${decBadge(dec.action)}</span>` : ''}</div>
    </div></li>`;
  }
  function renderSideFoot() {
    const el = $('#side-foot');
    if (!el) return;
    const d = S.live;
    const ml = get(S.health, 'model_loaded');
    const parts = [];
    if (d) parts.push(`<b>${esc(d.source || 'Live screening')}</b> · data as of ${esc(fmtUTC(d.data_as_of))} (${esc(ago(d.data_as_of))})${d.stale ? ' · <span class="pill st">STALE</span>' : ''}`);
    parts.push('Screening-grade (TLE/SGP4, no covariance) – confirm with CSpOC CDMs before any manoeuvre.');
    if (ml === false) parts.push('<span style="color:var(--stale)">Model not loaded on the server: estimates use the max-Pc / physics fallback.</span>');
    el.innerHTML = parts.join('<br>');
  }

  // ================================================================ bottom-left KPI card ("Next 24 h")
  function renderKpiCard() {
    const w = get(S.live, 'warnings') || null;
    $('#kpi-title').textContent = `Next ${S.windowH} h`;
    $('#kpi-sub').textContent = S.live ? `as of ${fmtUTC(S.live.data_as_of, false)}` : '';
    const v = (k) => (w && isNum(w[k]) ? w[k].toLocaleString('en-US') : '–');
    $('#kpi-tiles').innerHTML = [
      ['n_in_window', 'In window', ''], ['n_within_1km', '< 1 km', 'k2'], ['n_within_5km', '≤ 5 km', 'k1'],
      ['n_critical', 'Critical', 'k2'], ['n_elevated', 'Elevated', 'k1'],
    ].map(([k, l, c]) => `<div class="kpi ${c}" title="${esc(kpiTip(k))}"><div class="v">${v(k)}</div><div class="l">${l}</div></div>`).join('');
    renderHoursChart(w);
    renderInsights(w);
  }
  function kpiTip(k) {
    return {
      n_in_window: `Conjunctions with TCA in the next ${S.windowH} h (SOCRATES screens to 5 km)${num(get(S.live, 'warnings.n_recent_past')) ? `; ${fmtNum(S.live.warnings.n_recent_past, 0)} whose TCA passed in the last hour are listed in the feed but not counted here` : ''}`,
      n_within_1km: 'Miss distance below 1 km: Critical by the hard distance rule',
      n_within_5km: 'Miss distance up to 5 km: at least Elevated by the distance rule',
      n_critical: 'Operational decision Critical (Pc ≥ 1e-4 or < 1 km)',
      n_elevated: 'Operational decision Elevated',
    }[k];
  }
  function plotlyBase(extra) {
    return Object.assign({
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, system-ui, sans-serif', size: 11, color: COLORS.text2 },
      margin: { l: 44, r: 12, t: 10, b: 34 },
      hoverlabel: { bgcolor: '#18212B', bordercolor: '#526477', font: { color: COLORS.text, family: 'Inter, sans-serif', size: 12 } },
      xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.line, linecolor: COLORS.line, tickcolor: COLORS.line },
      yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.line, linecolor: COLORS.line, tickcolor: COLORS.line },
      legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 11 } },
    }, extra || {});
  }
  const PCFG = { displayModeBar: false, responsive: true };
  function plotReact(el, data, layout, cfg) {
    if (!el) return;
    if (!el.classList.contains('js-plotly-plot')) el.innerHTML = '';
    Array.from(el.children).forEach((n) => { if (n.classList && n.classList.contains('state')) n.remove(); });
    Plotly.react(el, data, layout, cfg || PCFG);
  }
  function noPlotly(el) { el.innerHTML = '<div class="state">Chart unavailable (Plotly failed to load).</div>'; }
  function renderHoursChart(w) {
    const el = $('#kpi-hours');
    if (!el) return;
    if (!w || !Array.isArray(w.by_hour) || !w.by_hour.length) { if (window.Plotly) { try { Plotly.purge(el); } catch (e) { /* ignore */ } el._cwClick = false; } el.innerHTML = `<div class="state">${S.live ? 'No hourly breakdown returned' : (S.liveErr ? 'No data' : 'Loading…')}</div>`; return; }
    if (!window.Plotly) { noPlotly(el); return; }
    const hours = w.by_hour.map((b) => num(b.hour));
    const now = Date.now();
    const hourLabel = (h) => `+${h}–${h + 1} h (${fmtUTC(new Date(now + h * 3.6e6).toISOString(), false).slice(0, 5)}Z)`;
    const mk = (key, name, color) => ({
      type: 'bar', name, x: hours, y: w.by_hour.map((b) => num(b[key]) || 0),
      marker: { color, line: { color: COLORS.panel, width: 1 } },
      customdata: hours.map(hourLabel),
      hovertemplate: `%{customdata}<br>${name}: <b>%{y}</b><extra></extra>`,
    });
    const layout = plotlyBase({
      barmode: 'stack', bargap: 0.18, margin: { l: 28, r: 6, t: 18, b: 22 },
      legend: { orientation: 'h', x: 0, y: 1.25, font: { size: 10.5 } },
      xaxis: { gridcolor: 'rgba(0,0,0,0)', linecolor: COLORS.line, tickfont: { size: 10 }, ticksuffix: 'h', dtick: hours.length > 36 ? 12 : (hours.length > 12 ? 6 : 1), fixedrange: true },
      yaxis: { gridcolor: COLORS.grid, tickfont: { size: 10 }, fixedrange: true, rangemode: 'tozero' },
    });
    plotReact(el, [mk('critical', 'Critical', COLORS.crit), mk('elevated', 'Elevated', COLORS.elev)], layout, PCFG);
    if (!el._cwClick) {
      el._cwClick = true;
      el.on('plotly_click', (ev) => {
        const p = ev && ev.points && ev.points[0];
        if (!p) return;
        S.hourFilter = S.hourFilter === p.x ? null : p.x;
        if (S.feedFilter !== 'all') setFeedFilter('all');
        renderFeed();
      });
    }
  }
  function renderInsights(w) {
    const el = $('#kpi-insights');
    if (!el) return;
    const rows = liveRows();
    if (!rows.length) { el.innerHTML = ''; return; }
    const items = [];
    const closest = sortRows(rows.filter((r) => num(r.hours_to_tca) >= 0), 'md')[0];
    if (closest) items.push(`Closest approach <b>${esc(fmtDist(closest.miss_distance))}</b>: <a href="#" data-sel="${esc(closest.id)}">${pairHtml(get(closest, 'sat1.name'), get(closest, 'sat2.name'))}</a>, ${esc(countdown(closest.tca))}`);
    const mxOf = (r) => firstNum(r.max_risk_estimate, get(r, 'assessment.log_pc'), -31);
    const byPc = rows.filter((r) => num(r.hours_to_tca) >= 0).sort((a, b) => mxOf(b) - mxOf(a))[0];
    if (byPc && mxOf(byPc) > -29.9) items.push(`Highest SOCRATES max Pc <b>${esc(maxPcText(byPc))}</b>: <a href="#" data-sel="${esc(byPc.id)}">${pairHtml(get(byPc, 'sat1.name'), get(byPc, 'sat2.name'))}</a>`);
    if (isNum(get(w, 'n_debris_or_rb')) && isNum(get(w, 'n_in_window'))) items.push(`<b>${w.n_debris_or_rb.toLocaleString('en-US')}</b> of ${w.n_in_window.toLocaleString('en-US')} involve debris or a rocket body`);
    else {
      const n1 = rows.filter((r) => num(r.miss_distance) < 1000);
      const deb = n1.filter((r) => /DEBRIS|ROCKET/.test(`${get(r, 'sat2.type', '')} ${get(r, 'sat1.type', '')}`)).length;
      if (n1.length) items.push(`<b>${deb}</b> of ${n1.length} loaded sub-km approaches involve debris or a rocket body`);
    }
    const rv = isNum(get(w, 'n_needs_review')) ? w.n_needs_review : rows.filter((r) => get(r, 'assessment.needs_review') === true).length;
    if (rv) items.push(`<b>${rv.toLocaleString('en-US')}</b> flagged for analyst review (uncertainty straddles a tier boundary)`);
    if (isNum(get(w, 'n_critical_by_pc')) && w.n_critical_by_pc > 0) items.splice(1, 0, `<b>${w.n_critical_by_pc.toLocaleString('en-US')}</b> Critical only because of Pc (miss ≥ 1 km: a distance-only rule would miss them)`);
    const top = (w && Array.isArray(w.top_primaries)) ? w.top_primaries.slice(0, 5) : [];
    items.splice(4);
    el.innerHTML = `<ul>${items.map((i) => `<li><span>${i}</span></li>`).join('')}</ul>` +
      (top.length ? `<div class="prim-chips" aria-label="Primaries with most conjunctions">${top.map((p) => `<button type="button" data-q="${esc(p.norad != null ? p.norad : p.name)}" title="Filter the feed by ${esc(p.name)}${p.norad != null ? ' (NORAD ' + esc(p.norad) + ')' : ''}">${esc(p.name)}<b>${esc(p.count)}</b></button>`).join('')}</div>` : '');
  }

  // ================================================================ overlay panel (layers)
  function renderOverlayCounts() {
    const rows = liveRows();
    const c = { 0: 0, 1: 0, 2: 0, r: 0 };
    rows.forEach((r) => { const t = tierOf(r); if (t != null) c[t]++; if (get(r, 'assessment.needs_review')) c.r++; });
    $$('#layer-grid [data-c]').forEach((el) => { el.textContent = c[el.dataset.c]; });
    const st = $('#pts-status');
    if (!st) return;
    if (!S.live) { st.textContent = S.liveErr ? 'Collision points: no data' : 'Collision points: waiting for data'; return; }
    const hasStatus = rows.some((r) => 'tle_status' in r || 'tca_point' in r);
    if (!hasStatus) { st.innerHTML = 'Collision points not provided by the server (<code>points=1</code> unsupported).'; return; }
    const ok = rows.filter((r) => r.tca_point && isNum(r.tca_point.lat)).length;
    const pend = rows.filter((r) => r.tle_status === 'pending').length;
    const miss = rows.filter((r) => r.tle_status === 'missing').length;
    const dis = rows.filter((r) => { const c = pointCheck(r); return c.known && !c.agrees; });
    const hidden = dis.filter((r) => !r.tca_point).length;
    st.innerHTML = `<b>${ok}</b> points drawn` +
      (dis.length ? ` · <span class="pend" title="${esc(SGP4_DISAGREE_TIP)}">${dis.length} SGP4 ≠ SOCRATES (hollow${hidden ? `; ${hidden} > 100 km apart, not drawn` : ''})</span>` : '') +
      (pend ? ` · <span class="pend">${pend} pending TLE</span>` : '') +
      (miss ? ` · ${miss} without TLE` : '');
    st.title = 'Collision point = midpoint of both objects at our SGP4 closest approach near the SOCRATES TCA. Hollow = our element sets do not reproduce the SOCRATES miss distance. Pending points appear as the server resolves TLEs (re-polled every 60 s).';
  }

  // ================================================================ Cesium globe
  const G = { viewer: null, ptsDS: null, selDS: null, ptEntities: new Map(), cartoLayer: null, tileErrors: 0, fellBack: false,
    hoverId: null, sel: null, lastTick: 0, ringImg: null };

  function requestRender() { if (G.viewer) G.viewer.scene.requestRender(); }

  function globeFallback(msg) {
    const fb = $('#globe-fallback');
    fb.hidden = false;
    fb.innerHTML = `<div><b>3D globe unavailable</b>${esc(msg)}<br><br>Every conjunction is still listed in the sidebar, and the detail drawer, reasons and analyst actions work without the globe.</div>`;
  }

  function initGlobe() {
    if (!window.Cesium) { globeFallback('CesiumJS failed to load from the CDN (offline or blocked).'); return; }
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
      sc.backgroundColor = C.Color.fromCssColorString('#0B1118');
      sc.globe.baseColor = C.Color.fromCssColorString('#111A24');
      sc.globe.showGroundAtmosphere = false;
      sc.globe.enableLighting = false;
      sc.fog.enabled = false;
      if (sc.sun) sc.sun.show = false;
      if (sc.moon) sc.moon.show = false;
      if (sc.skyAtmosphere) { sc.skyAtmosphere.show = true; sc.skyAtmosphere.brightnessShift = -0.35; }
      sc.screenSpaceCameraController.minimumZoomDistance = 2.0e5;
      sc.screenSpaceCameraController.maximumZoomDistance = 1.2e8;
      viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(C.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
      viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(100, 12, 2.4e7) });

      if (bm.ref) {
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
      G.ringImg = makeRing();

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
  function makeRing() {
    const c = document.createElement('canvas');
    c.width = c.height = 96;
    const g = c.getContext('2d');
    g.strokeStyle = '#FF5D6C'; g.lineWidth = 6;
    g.beginPath(); g.arc(48, 48, 40, 0, Math.PI * 2); g.stroke();
    g.strokeStyle = 'rgba(255,93,108,.45)'; g.lineWidth = 3;
    g.beginPath(); g.arc(48, 48, 26, 0, Math.PI * 2); g.stroke();
    return c;
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
    tip.innerHTML = `<div>${tierPill(tierOf(r))} ${a.needs_review ? reviewPill() : ''}</div>
      <div class="tt">${pairHtml(get(r, 'sat1.name'), get(r, 'sat2.name'))}</div>
      <div class="tm">Miss distance <b>${esc(fmtDist(r.miss_distance))}</b> · max Pc <b>${esc(maxPcText(r))}</b></div>
      <div class="tm">TCA ${esc(fmtUTC(r.tca))} · ${esc(countdown(r.tca))}</div>
      ${chk.known ? `<div class="tm ${chk.agrees ? 'hint' : 'warn-txt'}">${chk.agrees ? 'SGP4 reproduces the screening' : '<b>SGP4 disagrees</b>'}: ${esc(chk.text)}</div>` : ''}
      ${r.tca_point ? `<div class="tm hint">${esc(fixed(r.tca_point.lat, 2))}°, ${esc(fixed(r.tca_point.lon, 2))}°, ${esc(fmtNum(r.tca_point.alt_km, 0))} km alt</div>` : ''}`;
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
    if (t < S.minSev) return false;
    if (t === 2 && !S.layers.crit) return false;
    if (t === 1 && !S.layers.elev) return false;
    if (t === 0 && !S.layers.safe) return false;
    return true;
  }
  function pointSize(r) {
    const lp = firstNum(r.max_risk_estimate, get(r, 'assessment.log_pc'));
    const base = 6 + 10 * clamp(((lp == null ? -30 : lp) + 12) / 12, 0, 1);
    return Math.round(base);
  }
  function labelText(r) { return `${get(r, 'sat1.name', '?')} × ${get(r, 'sat2.name', '?')}`; }

  function updatePoints() {
    if (!G.viewer || !G.ptsDS) return;
    const C = window.Cesium;
    const rows = liveRows().filter((r) => r.tca_point && isNum(r.tca_point.lat) && isNum(r.tca_point.lon));
    const keep = new Set();
    const labelled = new Set(sortRows(rows.filter((r) => tierOf(r) === 2), 'risk').slice(0, 12).map((r) => r.id));
    const q = S.search;
    const ents = G.ptsDS.entities;
    ents.suspendEvents();
    rows.forEach((r) => {
      const key = `pt:${r.id}`;
      keep.add(key);
      const t = tierOf(r) || 0;
      const color = C.Color.fromCssColorString(TIER_COLOR[t]);
      const review = !!get(r, 'assessment.needs_review') && S.layers.review;
      const isSel = S.selectedId === r.id;
      const isHit = q && matchesSearch(r, q);
      const dim = q && !isHit;
      const chk = pointCheck(r);
      const hollow = chk.known && !chk.agrees;          // our SGP4 does not reproduce the screening
      const size = pointSize(r) + (isSel ? 6 : 0) + (isHit ? 3 : 0);
      const pos = C.Cartesian3.fromDegrees(r.tca_point.lon, r.tca_point.lat, Math.max(num(r.tca_point.alt_km) || 0, 0) * 1000);
      let e = G.ptEntities.get(key);
      const outlineColor = isSel || isHit ? C.Color.WHITE : (hollow ? color : (review ? C.Color.fromCssColorString(COLORS.review) : C.Color.fromCssColorString('#0B1118').withAlpha(0.8)));
      const outlineWidth = isSel ? 3 : (review || isHit || hollow ? 2.5 : 1);
      const fill = color.withAlpha(hollow ? 0.08 : (dim ? 0.25 : 0.92));
      const showLabel = S.layers.labels && (labelled.has(r.id) || isSel || isHit);
      if (!e) {
        e = ents.add({
          id: key, position: pos,
          point: { pixelSize: size, color: fill, outlineColor, outlineWidth },
          label: {
            text: labelText(r), font: '600 11px Inter, sans-serif', fillColor: C.Color.fromCssColorString(COLORS.text),
            showBackground: true, backgroundColor: C.Color.fromCssColorString('#18212B').withAlpha(0.85),
            backgroundPadding: new C.Cartesian2(6, 3), pixelOffset: new C.Cartesian2(0, -16),
            horizontalOrigin: C.HorizontalOrigin.CENTER, verticalOrigin: C.VerticalOrigin.BOTTOM,
            show: showLabel, scaleByDistance: new C.NearFarScalar(2e6, 1.0, 3e7, 0.75),
          },
        });
        G.ptEntities.set(key, e);
      } else {
        e.position = pos;
        e.point.pixelSize = size;
        e.point.color = fill;
        e.point.outlineColor = outlineColor;
        e.point.outlineWidth = outlineWidth;
        e.label.text = labelText(r);
        e.label.show = showLabel;
      }
      e.show = pointVisible(r) || isSel;
    });
    for (const [key, e] of G.ptEntities) {
      if (!keep.has(key)) { ents.remove(e); G.ptEntities.delete(key); }
    }
    ents.resumeEvents();
    G.ptsDS.show = S.layers.points;
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
    G.viewer.camera.flyTo({ destination: dest, orientation: { heading: 0, pitch: -Math.PI / 2, roll: 0 }, duration: 1.4 });
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
    const showLbl = S.layers.labels;
    const path = (id, pos, col) => G.selDS.entities.add({
      id, show: S.layers.paths,
      polyline: { positions: pos, width: 5, arcType: C.ArcType.NONE,
        material: new C.PolylineGlowMaterialProperty({ glowPower: 0.22, taperPower: 1.0, color: col }) },
    });
    path('sel:path-a', A.positions, colA);
    path('sel:path-b', B.positions, colB);
    const marker = (id, prop, col, name) => G.selDS.entities.add({
      id, position: prop,
      point: { pixelSize: 10, color: col, outlineColor: C.Color.WHITE, outlineWidth: 2 },
      label: { text: name || '', font: '600 12px Inter, sans-serif', fillColor: col, showBackground: true,
        backgroundColor: C.Color.fromCssColorString('#18212B').withAlpha(0.85), backgroundPadding: new C.Cartesian2(6, 3),
        pixelOffset: new C.Cartesian2(12, 0), horizontalOrigin: C.HorizontalOrigin.LEFT, verticalOrigin: C.VerticalOrigin.CENTER, show: showLbl },
    });
    marker('sel:obj-a', A.prop, colA, get(t, 'a.name') || get(r, 'sat1.name'));
    marker('sel:obj-b', B.prop, colB, get(t, 'b.name') || get(r, 'sat2.name'));

    // collision point at the model's closest approach (midpoint of both objects), fallback SOCRATES point
    const tcaIso = t.tca_model || r.tca;
    const tcaJd = C.JulianDate.fromIso8601(String(tcaIso));
    let tcaPos = null;
    const pa = A.prop.getValue(tcaJd); const pb = B.prop.getValue(tcaJd);
    if (pa && pb) tcaPos = C.Cartesian3.midpoint(pa, pb, new C.Cartesian3());
    else if (r.tca_point) tcaPos = C.Cartesian3.fromDegrees(r.tca_point.lon, r.tca_point.lat, (num(r.tca_point.alt_km) || 0) * 1000);
    const md = num(t.min_range_m) != null ? t.min_range_m : r.miss_distance;
    if (tcaPos) {
      const t0 = performance.now();
      const phase = () => ((performance.now() - t0) % 1600) / 1600;
      G.selDS.entities.add({
        id: 'sel:tca', position: tcaPos,
        billboard: {
          image: G.ringImg,
          scale: new C.CallbackProperty(() => 0.35 + 0.55 * phase(), false),
          color: new C.CallbackProperty(() => C.Color.WHITE.withAlpha(1 - 0.85 * phase()), false),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        point: { pixelSize: 9, color: C.Color.fromCssColorString(COLORS.crit), outlineColor: C.Color.WHITE, outlineWidth: 2, disableDepthTestDistance: Number.POSITIVE_INFINITY },
        label: { text: `TCA ${fmtUTC(tcaIso, false)} · SGP4 ${fmtDist(md)}${pointCheck(r).agrees === false ? ` (SOCRATES ${fmtDist(r.miss_distance)})` : ''}`, font: '700 12px Inter, sans-serif',
          fillColor: C.Color.fromCssColorString('#FFD6DA'), showBackground: true,
          backgroundColor: C.Color.fromCssColorString('#3A1D25').withAlpha(0.92), backgroundPadding: new C.Cartesian2(7, 4),
          pixelOffset: new C.Cartesian2(0, -26), verticalOrigin: C.VerticalOrigin.BOTTOM, disableDepthTestDistance: Number.POSITIVE_INFINITY, show: true },
      });
    }

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
    clk.shouldAnimate = true;
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
    $('#pb-time').textContent = `T${sign}${pad(Math.floor(a / 60))}:${pad(Math.floor(a % 60))} · ${fmtUTC(iso, false)}`;
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
    G.viewer.camera.flyToBoundingSphere(bs, { duration: 1.2, offset: new C.HeadingPitchRange(0, -1.2, bs.radius * 3.2) });
  }
  function applyLayerVisibility() {
    if (!G.viewer) return;
    G.ptsDS.show = S.layers.points;
    G.selDS.entities.values.forEach((e) => {
      if (e.id.startsWith('sel:path')) e.show = S.layers.paths;
      if (e.id.startsWith('sel:obj') && e.label) e.label.show = S.layers.labels;
    });
    updatePoints();
  }

  // ================================================================ select / deselect a live event
  // Map overlays (Layers, Next-N-h) cover the encounter; fold them away while an event is open and
  // restore whatever the analyst had before.
  const OVERLAYS = [['#ovl-toggle', '#overlay'], ['#kpi-toggle', '#kpi-card']];
  function setOverlayCollapsed(btnSel, boxSel, c) {
    const box = $(boxSel), btn = $(btnSel);
    if (!box || !btn) return;
    box.classList.toggle('collapsed', c);
    btn.innerHTML = c ? '+' : '&minus;';
    btn.setAttribute('aria-expanded', c ? 'false' : 'true');
  }
  function foldOverlays() {
    if (S.overlaysBefore) return;
    S.overlaysBefore = OVERLAYS.map(([, box]) => !!($(box) && $(box).classList.contains('collapsed')));
    OVERLAYS.forEach(([btn, box]) => setOverlayCollapsed(btn, box, true));
  }
  function restoreOverlays() {
    if (!S.overlaysBefore) return;
    OVERLAYS.forEach(([btn, box], i) => setOverlayCollapsed(btn, box, S.overlaysBefore[i]));
    S.overlaysBefore = null;
    setTimeout(resizePlots, 30);
  }
  function selectEvent(id, opts) {
    opts = opts || {};
    const r = rowById(id);
    if (!r) return;
    foldOverlays();
    try { history.replaceState(null, '', `#live/${encodeURIComponent(id)}`); } catch (e) { /* ignore */ }
    if (S.view !== 'live') setView('live');
    const same = S.selectedId === id;
    S.selectedId = id;
    S.selectedRow = r;
    $$('#feed .card.sel').forEach((c) => c.classList.remove('sel'));
    const card = $(`#feed .card[data-sel="${CSS.escape(id)}"]`);
    if (card) {
      card.classList.add('sel');
      if (opts.scroll !== false) card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
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
    $$('#feed .card.sel').forEach((c) => c.classList.remove('sel'));
    closeDetail();
    clearSelectionGraphics();
    restoreOverlays();
    updatePoints();
  }
  async function loadTrack(r) {
    const seq = ++S.trackSeq;
    S.track = null; S.trackErr = null; S.trackLoading = true;
    clearSelectionGraphics();
    renderDetail();
    const a = get(r, 'sat1.norad'); const b = get(r, 'sat2.norad');
    if (a == null || b == null || !r.tca) {
      S.trackLoading = false; S.trackErr = 'missing NORAD ids or TCA'; renderDetail(); return;
    }
    try {
      const t = await api('/api/live/track', { a, b, tca: r.tca, span_min: 50 });
      if (seq !== S.trackSeq) return;
      S.track = t;
      try { drawTrack(r, t); if (!(r.tca_point && isNum(r.tca_point.lat))) fitPaths(); } catch (e) { console.warn(e); S.trackErr = `could not draw the track (${e.message})`; }
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
  // SOCRATES' element set (epoch = TCA - DSE) and ours (bulk / per-object GP) are labelled separately:
  // when they differ, our SGP4 and the SOCRATES screening can disagree.
  function objCard(o, cls, label, trackObj, tca, shiftH) {
    o = o || {};
    const st = o.status == null ? '' : String(o.status);
    const sAge = num(o.tle_age_d);
    const tcaMs = parseT(tca);
    const sEpochMs = sAge != null && tcaMs != null ? tcaMs - sAge * 864e5 : null;
    let ourMs = trackObj ? parseT(trackObj.tle_epoch) : null;
    if (ourMs == null && sEpochMs != null && num(shiftH) != null) ourMs = sEpochMs + shiftH * 3.6e6;
    const ourAge = ourMs != null && tcaMs != null ? (tcaMs - ourMs) / 864e5 : null;
    const dh = ourMs != null && sEpochMs != null ? (ourMs - sEpochMs) / 3.6e6 : null;
    const iso = (ms) => new Date(ms).toISOString();
    return `<div class="obj ${cls}">
      <div class="hint">${label}</div>
      <div class="on">${esc(o.name || '?')}</div>
      <div class="om">NORAD <b>${esc(o.norad != null ? o.norad : '?')}</b> · <b>${esc(o.type || 'type ?')}</b></div>
      <div class="om">${esc(STATUS_TEXT[st] || `Status “${st}”`)}${st ? ` <span class="muted">[${esc(st)}]</span>` : ''}</div>
      <div class="om" title="SOCRATES DSE: days from the epoch of the element set SOCRATES screened with to TCA">SOCRATES TLE age at TCA <b>${esc(sAge != null ? sAge.toFixed(1) + ' d' : '?')}</b>${sEpochMs != null ? ` <small>(epoch ≈ ${esc(fmtUTC(iso(sEpochMs)))})</small>` : ''}</div>
      ${ourMs != null ? `<div class="om" title="Element set used for our SGP4 flight paths and collision point">Our TLE epoch <b>${esc(fmtUTC(iso(ourMs)))}</b>${ourAge != null ? ` <small>(age ${esc(ourAge.toFixed(1))} d at TCA)</small>` : ''}</div>` : ''}
      ${dh != null ? `<div class="om">Epoch difference <b>${dh >= 0 ? '+' : '−'}${esc(Math.abs(dh).toFixed(1))} h</b> <small>(${dh >= 0 ? 'ours newer' : 'ours older'})</small></div>` : ''}
      ${trackObj && trackObj.elements_source ? `<div class="om"><small>${esc(trackObj.elements_source)}</small></div>` : ''}
      ${trackObj && trackObj.tle_stale ? `<div class="om warn-txt">Element set STALE: refresh failed, cached copy fetched ${esc(fmtUTC(trackObj.tle_fetched_at))}</div>` : ''}
    </div>`;
  }
  function stackBar(p) {
    p = Array.isArray(p) ? p.map((x) => num(x) || 0) : [0, 0, 0];
    const seg = (i) => (p[i] > 0.001 ? `<div class="s${i}" style="flex:${p[i]}" title="P(${TIERS[i]}) = ${pct(p[i], 1)}">${p[i] >= 0.12 ? pct(p[i]) : ''}</div>` : '');
    return `<div class="stack" role="img" aria-label="P(Safe) ${pct(p[0])}, P(Elevated) ${pct(p[1])}, P(Critical) ${pct(p[2])}">${seg(0)}${seg(1)}${seg(2)}</div>
      <div class="stack-legend"><span><i class="dot t0"></i>Safe ${pct(p[0], 1)}</span><span><i class="dot t1"></i>Elevated ${pct(p[1], 1)}</span><span><i class="dot t2"></i>Critical ${pct(p[2], 1)}</span></div>`;
  }
  function reasonsHtml(reasons) {
    if (!Array.isArray(reasons) || !reasons.length) return '<p class="hint">No reasons returned by the model.</p>';
    return `<ol class="reasons">${reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ol>`;
  }
  function renderDetail() {
    const el = $('#detail');
    const r = S.selectedRow;
    if (!el || !r) return;
    const a = r.assessment || {};
    const t = S.track;
    const s1 = r.sat1 || {}; const s2 = r.sat2 || {};
    const keepScroll = el.scrollTop;
    const oldNote = $('.act-note', el);
    const prevNote = oldNote ? oldNote.value : '';
    const noteFocused = oldNote && document.activeElement === oldNote;
    const ptxt = r.tca_point ? `${fixed(r.tca_point.lat, 2)}°, ${fixed(r.tca_point.lon, 2)}°, ${fmtNum(r.tca_point.alt_km, 0)} km` :
      (r.tle_status === 'pending' ? 'pending TLE download' : (r.tle_status === 'missing' ? 'no TLE available' :
        (r.tle_status === 'inconsistent' ? 'not drawn: with our element sets the objects stay > 100 km apart' : 'not provided')));
    let trackHtml;
    if (S.trackLoading) trackHtml = '<div class="loading">Propagating both objects with SGP4…</div>';
    else if (S.trackErr) trackHtml = `<div class="err-box"><b>Trajectory unavailable.</b> ${esc(S.trackErr)}<br><button type="button" class="btn sm" data-act="retry-track" style="margin-top:6px">Retry</button></div>`;
    else if (t) trackHtml = `<div id="rtn-chart" class="rtn-chart" aria-label="Relative trajectory of the secondary in the primary's RTN frame"></div>
      <div class="chart-note">B relative to A in A's RTN frame, projected on the T (along-track) / N (cross-track) plane.${Array.isArray(t.rtn_at_tca_m) ? ` Radial offset at TCA: ${esc(fmtDist(t.rtn_at_tca_m[0]))}.` : ''}${t.mock_note ? ` <b>${esc(t.mock_note)}.</b>` : ''}</div>`;
    else trackHtml = '';
    const modelMin = t && num(t.min_range_m) != null ? `${fmtDist(t.min_range_m)}${t.tca_model ? ` <small>at ${esc(fmtUTC(t.tca_model, false))}</small>` : ''}` : (S.trackLoading ? '<small>computing…</small>' : '–');
    const decs = decisionsFor('live', r.id);
    const chk = pointCheck(r);
    const pi = pcInfo(r);
    const shifts = get(r, 'sgp4_check.epoch_shift_h') || [];
    const sh = (x) => (num(x) != null ? `${x >= 0 ? '+' : ''}${x.toFixed(1)} h` : '?');
    const shiftTxt = shifts.some((x) => num(x) != null) ? ` Element-set epochs (ours minus SOCRATES'): A ${sh(shifts[0])}, B ${sh(shifts[1])}.` : '';
    const consistency = chk.known && !chk.agrees ? `<div class="screen-warn bad-warn"><b>SGP4 disagrees with the screening.</b> ${esc(chk.text)} (distance rule: ${esc(TIERS[mdTier(chk.sgp)])} vs ${esc(TIERS[mdTier(chk.miss)])}).${esc(shiftTxt)} The element sets differ (newer TLE, or a manoeuvre since screening): re-screen before acting.${r.tca_point ? '' : ' No collision point is drawn: with our element sets the objects stay more than 100 km apart.'}</div>` : '';
    const edgeWarn = t && t.min_at_edge ? '<div class="hint" style="color:var(--stale)">Closest approach is at the edge of the propagated window: the true minimum lies outside ±25 min.</div>' : '';
    const M = get(S.live, 'model') || {};
    const swu = get(S.live, 'space_weather_used') || {};
    const prov = [r.provenance,
      S.live && `Assessed by the ${M.kind || 'screening model'}${M.trained === false ? ' (NOT trained: fallback estimates)' : ''}${M.trained_at ? `, trained ${fmtUTC(M.trained_at)}` : ''}, at ${fmtUTC(S.live.assessed_at)}, with F10.7 ${fmtNum(swu.F10, 1)} / Ap ${fmtNum(swu.AP, 1)}${swu.stale ? ' (space weather STALE)' : ''}`,
      t && t.propagator, t && get(t, 'a.tle_epoch') && `Our TLE epochs A ${fmtUTC(get(t, 'a.tle_epoch'))}, B ${fmtUTC(get(t, 'b.tle_epoch'))}`,
      S.live && `Feed fetched ${fmtUTC(S.live.fetched_at)}${S.live.stale ? ' (STALE)' : ''}`].filter(Boolean);
    const ruleCrit = num(a.md_tier) === 2;
    const probHtml = ruleCrit
      ? `<h4>Tier</h4><div class="rule-box"><b>Critical by the distance rule</b>: screening miss ${esc(fmtDist(r.miss_distance))} &lt; 1 km. No probability is attached: SOCRATES miss distances come from TLEs without covariance, and the rule treats them as exact.</div>
        <div class="hint" style="margin-top:6px">Pc part of the rule only: P(Pc ≥ 1e-4) = ${esc(pct(a.p_pc_critical, 1))}${pi.informative ? '' : ' (uninformative, see below)'}</div>`
      : `<h4>Tier probabilities <small class="muted">(given the screening miss distance is exact)</small></h4>${stackBar(a.p_tier)}`;
    const ourMin = t && num(t.min_range_m) != null ? modelMin : (chk.known && get(r, 'sgp4_check.min_range_m') != null ? `${esc(fmtDist(chk.sgp))} <small>(±2 min search)</small>` : modelMin);

    el.innerHTML = `
      <div class="dt-head">
        <div class="row"><div class="dt-title">${pairHtml(s1.name, s2.name)}</div>
          <button type="button" class="mini-btn" data-act="deselect" aria-label="Close detail (Esc)" title="Close (Esc)">&times;</button></div>
        <div class="dt-badges">${tierPill(tierOf(r), 'Decision')}${a.needs_review ? reviewPill() : ''}
          <span class="pill nt" title="Tier from the miss-distance rule alone">Distance rule: ${esc(TIERS[num(a.md_tier)] || '?')}</span>
          ${chk.known ? (chk.agrees ? `<span class="pill nt" title="${esc(chk.text)}">SGP4 reproduces</span>` : `<span class="pill sg" title="${esc(chk.text + '. ' + SGP4_DISAGREE_TIP)}">SGP4 disagrees</span>`) : ''}
          ${a.heads_disagree ? '<span class="pill rv" title="The classifier alone raised the tier while the Pc quantiles stay below its threshold">Model heads disagree</span>' : ''}
          ${S.live && S.live.stale ? '<span class="pill st">Stale data</span>' : ''}</div>
        ${consistency}
      </div>
      <div class="dt-sec"><div class="objs">${objCard(s1, 'a', 'Primary (A)', t && t.a, r.tca, shifts[0])}${objCard(s2, 'b', 'Secondary (B)', t && t.b, r.tca, shifts[1])}</div></div>
      <div class="dt-sec">
        ${probHtml}
        <dl class="kv" style="margin-top:10px">
          <dt>SOCRATES max Pc</dt><dd><b>${esc(isNum(r.max_prob) ? (r.max_prob > 0 ? r.max_prob.toExponential(1) : '0') : '–')}</b> <small>worst case over covariance scaling · dilution ${esc(isNum(r.dilution_km) ? fmtDist(r.dilution_km * 1000) : '–')}</small></dd>
          <dt>Est. Pc (median)</dt><dd>${pi.informative ? `${pcHtml(a.log_pc)} <small>log10 ${esc(fixed(a.log_pc, 2))}</small>` : `<span class="pc-un">uninformative</span> <small class="muted">${esc(pcText(a.log_pc))}</small>`}</dd>
          <dt>80% interval</dt><dd>${esc(pcText(a.log_pc_lo))} – ${esc(pcText(a.log_pc_hi))}${pi.width != null ? ` <small>(${esc(pi.width.toFixed(1))} decades)</small>` : ''}</dd>
          <dt>P(Pc ≥ 1e-6) / P(Pc ≥ 1e-4)</dt><dd>${esc(pct(a.p_pc_elevated, 1))} / ${esc(pct(a.p_pc_critical, 1))}</dd>
          ${pi.informative ? '' : `<dt></dt><dd><small class="warn-txt">${esc(pi.tip)}</small></dd>`}
          <dt>Miss distance (SOCRATES)</dt><dd>${esc(fmtDist(r.miss_distance))}</dd>
          <dt>Our min range (SGP4)</dt><dd>${ourMin}</dd>
          <dt>Relative speed</dt><dd>${esc(fmtSpeed(r.relative_speed))}${t && num(t.relative_speed_ms) ? ` <small>SGP4 ${esc(fmtSpeed(t.relative_speed_ms))}</small>` : ''}</dd>
          <dt>TCA (UTC)</dt><dd>${esc(fmtUTC(r.tca))} <small data-tca="${esc(r.tca)}">${esc(countdown(r.tca))}</small></dd>
          <dt>Collision point</dt><dd><small>${esc(ptxt)}</small></dd>
          ${t && (get(t, 'a.elements_source') || get(t, 'b.elements_source')) ? `<dt>Element sets</dt><dd><small>A: ${esc(get(t, 'a.elements_source') || '?')}<br>B: ${esc(get(t, 'b.elements_source') || '?')}</small></dd>` : ''}
        </dl>
      </div>
      <div class="dt-sec"><h4>Why this tier (model reasons, verbatim)</h4>${reasonsHtml(a.reasons)}</div>
      <div class="dt-sec"><h4>Relative trajectory around TCA</h4>${edgeWarn}${trackHtml}</div>
      <div class="dt-sec">
        <h4>Provenance</h4>
        <div class="prov">${prov.map(esc).join('<br>')}</div>
        <div class="screen-warn">Screening-grade (TLE/SGP4, no covariance) – confirm with CSpOC CDMs before any manoeuvre.</div>
      </div>
      <div class="dt-sec">
        <h4>Analyst action</h4>
        ${actionsHtml('live', r.id, decs)}
      </div>`;
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
    if (!rel.length) { el.innerHTML = '<div class="state">No relative trajectory samples.</div>'; return; }
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
      { type: 'scatter', mode: 'lines', name: 'B relative to A', x: rel.map((p) => km(p[2])), y: rel.map((p) => km(p[3])),
        customdata: rel.map((p) => [p[0], Math.hypot(p[1], p[2], p[3])]), line: { color: COLORS.b, width: 2 },
        hovertemplate: 't = %{customdata[0]:.0f} s from TCA<br>T %{x:.2f} km · N %{y:.2f} km<br>range %{customdata[1]:,.0f} m<extra></extra>' },
      { type: 'scatter', mode: 'markers', name: 'A (origin)', x: [0], y: [0], marker: { color: COLORS.a, size: 9, line: { color: COLORS.text, width: 1 } }, hovertemplate: 'Primary A at origin<extra></extra>' },
      { type: 'scatter', mode: 'markers+text', name: 'TCA', x: [km(tca[1])], y: [km(tca[2])], text: [`TCA ${fmtDist(num(t.min_range_m) != null ? t.min_range_m : Math.hypot(tca[0], tca[1], tca[2]))}`],
        textposition: 'top right', textfont: { color: COLORS.text, size: 11 },
        marker: { color: COLORS.crit, size: 10, symbol: 'diamond', line: { color: COLORS.text, width: 1 } },
        hovertemplate: `Closest approach<br>T %{x:.3f} km · N %{y:.3f} km<br>R ${esc(fmtDist(tca[0]))}<extra></extra>` },
    ];
    const layout = plotlyBase({
      margin: { l: 42, r: 10, t: 26, b: 36 }, showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.14, font: { size: 10.5 } },
      xaxis: { title: { text: 'T along-track (km)', font: { size: 10.5 } }, range: [-half, half], gridcolor: COLORS.grid, zerolinecolor: COLORS.line },
      yaxis: { title: { text: 'N cross-track (km)', font: { size: 10.5 } }, range: [-half, half], scaleanchor: 'x', scaleratio: 1, gridcolor: COLORS.grid, zerolinecolor: COLORS.line },
      shapes: [circ(1, 'solid', COLORS.crit), circ(5, 'dash', COLORS.elev)],
      annotations: [
        { x: -0.75, y: -0.75, text: '1 km', showarrow: false, font: { color: COLORS.crit, size: 10 }, xanchor: 'right', yanchor: 'top' },
        { x: 3.6, y: 3.6, text: '5 km', showarrow: false, font: { color: COLORS.elev, size: 10 }, xanchor: 'left' },
      ],
    });
    plotReact(el, data, layout, Object.assign({}, PCFG, { scrollZoom: true }));
  }

  // ================================================================ analyst actions + decision log
  const ACTIONS = [
    { key: 'acknowledge', label: 'Acknowledge', cls: 'ack' },
    { key: 'escalate', label: 'Escalate to flight dynamics', cls: 'esc' },
    { key: 'dismiss', label: 'Dismiss (false alarm)', cls: 'ghost' },
  ];
  function decBadge(action) {
    const a = ACTIONS.find((x) => x.key === action);
    return `<span class="dec-badge ${esc(action)}" title="Latest analyst decision">${esc(a ? (action === 'escalate' ? 'ESCALATED' : action === 'acknowledge' ? 'ACK' : 'DISMISSED') : action)}</span>`;
  }
  function actionsHtml(view, id, decs) {
    return `<div class="actions" data-view="${esc(view)}" data-id="${esc(id)}">
      <label class="analyst${S.analyst ? '' : ' need'}" style="margin:0">Analyst <input class="analyst-inline" maxlength="24" value="${esc(S.analyst)}" placeholder="initials" aria-label="Analyst initials"></label>
      <textarea class="act-note" placeholder="Note (rationale, CDM ids checked, who was informed)…" aria-label="Decision note"></textarea>
      <div class="act-row">${ACTIONS.map((a) => `<button type="button" class="btn sm ${a.cls}" data-decide="${a.key}">${esc(a.label)}</button>`).join('')}</div>
      <div class="hint">The recommendation does not act on its own: your decision, note and initials are logged with the evidence shown.</div>
      ${decs.length ? `<ul class="act-hist">${decs.slice(0, 5).map((d) => `<li><b>${esc(d.action)}</b> by ${esc(d.analyst)} · ${esc(fmtUTC(d.ts))}${d.note ? ` – ${esc(d.note)}` : ''}</li>`).join('')}</ul>` : ''}
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
      toast('Enter your initials first: every decision is attributed to an analyst.', true);
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
    toast(`${ACTIONS.find((x) => x.key === action).label} logged for ${entry.pair || id} by ${initials}${S.logPersist ? '' : ' (not persisted: browser storage unavailable)'}`);
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
    $('#ld-storage').textContent = S.logPersist ? `${S.log.length} decision${S.log.length === 1 ? '' : 's'} stored in this browser (localStorage). Export to CSV to hand over a shift.` :
      'Browser storage is unavailable: decisions are kept only until this tab closes. Export to CSV.';
    const f = $('#ld-filter').value;
    const rows = S.log.filter((d) => f === 'all' || d.view === f).slice().sort((a, b) => (a.ts < b.ts ? 1 : -1));
    if (!rows.length) { list.innerHTML = '<li class="hint">No decisions logged yet. Select an event and use Acknowledge / Escalate / Dismiss.</li>'; return; }
    list.innerHTML = rows.map((d) => `<li>
      <div class="r1">${decBadge(d.action)}<b>${esc(d.pair || d.event_id)}</b><span class="t">${esc(fmtUTC(d.ts))}</span></div>
      <div class="r2">${d.note ? esc(d.note) : '<span class="muted">no note</span>'}</div>
      <div class="r3">${esc(d.analyst)} · ${esc(d.view)}${d.model_tier ? ` · model said ${esc(d.model_tier)}` : ''}${d.needs_review ? ' (review)' : ''}${isNum(d.miss_distance_m) ? ` · ${esc(fmtDist(d.miss_distance_m))}` : ''}${d.tca ? ` · TCA ${esc(fmtUTC(d.tca))}` : ''}</div>
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
      list.innerHTML = `<li class="state err-state"><b>Test-set events unavailable.</b><br>${esc(S.test.err)}<br><br><button type="button" class="btn sm" data-act="retry-test">Retry</button></li>`;
      $('#tq-count').textContent = '0';
      return;
    }
    const rows = testRows();
    $('#tq-count').textContent = rows.length;
    const all = S.test.events || [];
    const nDis = all.filter((e) => isNum(num(e.true_tier)) && num(get(e, 'assessment.tier')) !== num(e.true_tier)).length;
    const nUnder = all.filter((e) => isNum(num(e.true_tier)) && num(get(e, 'assessment.tier')) < num(e.true_tier)).length;
    $('#tq-foot').innerHTML = `${all.length} events (latest CDM per event, ≥ 2 days to TCA). <b>${nDis}</b> disagree with ground truth, <b>${nUnder}</b> under-called. Ground truth uses the CDM's own <code>risk</code>, which the model never sees.`;
    if (!rows.length) { list.innerHTML = '<li class="state">No events match these filters.</li>'; return; }
    list.innerHTML = rows.slice(0, 600).map((e) => {
      const a = e.assessment || {};
      const pt = num(a.tier); const tt = num(e.true_tier); const ft = fcTier(e);
      const flag = tt == null || pt == null ? '' : (pt < tt ? '<span class="pill t2" title="Predicted tier is below the ground-truth tier">Under-called</span>' : (pt > tt ? '<span class="pill nt" title="Predicted tier is above the ground-truth tier (false alarm)">Over-called</span>' : ''));
      return `<li><div class="card${String(S.test.selected) === String(e.event_id) ? ' sel' : ''}" role="button" tabindex="0" data-test="${esc(e.event_id)}">
        <div class="card-top">${tierPill(pt, 'Pred')}${a.needs_review ? reviewPill() : ''}${flag}<span class="cd">${esc(fixed(e.time_to_tca, 2))} d to TCA</span></div>
        <div class="card-title">Event #${esc(e.event_id)} <span class="x">·</span><span class="muted" style="font-weight:500">${esc(e.c_object_type || '?')}</span></div>
        <div class="card-meta"><span>miss <b>${esc(fmtDist(e.miss_distance))}</b></span><span>est. Pc <b>${pcHtml(a.log_pc)}</b></span><span>true Pc <b>${pcHtml(e.true_log_pc)}</b></span></div>
        <div class="card-foot">${tierPill(tt, 'Truth')}${ft != null ? tierPill(ft, 'Forecast') : ''}<span>${esc(e.n_cdms != null ? e.n_cdms : '?')} CDMs</span></div>
      </div></li>`;
    }).join('');
  }
  async function selectTestEvent(id, opts) {
    opts = opts || {};
    S.test.selected = id;
    S.test.cdmIdx = null;
    if (!opts.noHash && location.hash !== `#evaluator/${id}`) history.replaceState(null, '', `#evaluator/${id}`);
    $$('#tq .card.sel').forEach((c) => c.classList.remove('sel'));
    const card = $(`#tq .card[data-test="${CSS.escape(String(id))}"]`);
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
      el.innerHTML = S.test.err ? `<div class="state err-state"><b>Could not load test-set events.</b> ${esc(S.test.err)}</div>` :
        '<div class="state">Select an event from the queue to see its CDM history, the model estimate with its 80% band against ground truth, and the reasons behind the tier.</div>';
      return;
    }
    const ev = (S.test.events || []).find((e) => String(e.event_id) === String(id)) || { event_id: id };
    const pT = num(get(ev, 'assessment.tier')); const tT = num(ev.true_tier);
    const errPill = pT == null || tT == null ? '' : (pT < tT ? '<span class="pill t2" title="Predicted tier below ground truth: a missed escalation">Under-called</span>' : (pT > tT ? '<span class="pill nt" title="Predicted tier above ground truth: a false alarm">Over-called</span>' : '<span class="pill t0" title="Predicted tier equals ground truth">Agrees with truth</span>'));
    const head = `<div class="ev-head"><h2>Event #${esc(id)}</h2>${errPill}${tierPill(num(get(ev, 'assessment.tier')), 'Predicted')}${tierPill(num(ev.true_tier), 'Ground truth')}${fcTier(ev) != null ? tierPill(fcTier(ev), 'Forecast at TCA') : ''}${get(ev, 'assessment.needs_review') ? reviewPill() : ''}<span class="hint">${esc(ev.c_object_type || '')}</span></div>`;
    if (S.test.detailLoading) { el.innerHTML = head + '<div class="loading" style="padding:20px">Loading CDM history…</div>'; return; }
    if (S.test.detailErr) { el.innerHTML = head + `<div class="err-box"><b>Event detail unavailable.</b> ${esc(S.test.detailErr)} <button type="button" class="btn sm" data-act="retry-test-detail">Retry</button></div>`; return; }
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
        <div class="tile"><div class="v">${esc(cdms.length)}</div><div class="l">CDMs in history</div></div>
        <div class="tile"><div class="v">${esc(fixed(c.time_to_tca, 2))} <small>d</small></div><div class="l">time to TCA (selected CDM)</div></div>
        <div class="tile"><div class="v">${esc(fmtDist(c.miss_distance))}</div><div class="l">miss distance</div></div>
        <div class="tile"><div class="v">${pcHtml(a.log_pc)}</div><div class="l">est. Pc (80%: ${esc(pcText(a.log_pc_lo))} – ${esc(pcText(a.log_pc_hi))})</div></div>
        <div class="tile"><div class="v">${pcHtml(c.true_log_pc)}</div><div class="l">ground-truth Pc (<code>risk</code>, evaluation only)</div></div>
        <div class="tile"><div class="v">${esc(fmtSpeed(c.relative_speed))}</div><div class="l">relative speed</div></div>
      </div>
      <div class="ev-grid">
        <div class="panel wide"><div class="panel-head"><h3>Collision probability vs days to TCA</h3><span class="hint">Click a CDM to inspect it · dashed lines = 1e-6 / 1e-4 thresholds</span></div><div id="ev-pc" class="chart"></div></div>
        <div class="panel"><div class="panel-head"><h3>Miss distance</h3><span class="hint">1 km / 5 km rule lines</span></div><div id="ev-md" class="chart sm"></div></div>
        <div class="panel"><div class="panel-head"><h3>Miss vector in RTN (m)</h3><span class="hint">Flyby line from relative_velocity: display only</span></div><div id="ev-rtn" class="chart sm"></div></div>
        <div class="panel"><div class="panel-head"><h3>Why this tier</h3><span class="cdm-pick">CDM ${idx + 1} of ${cdms.length} · ${esc(fixed(c.time_to_tca, 2))} d to TCA</span></div>
          <div class="dt-badges" style="margin:0 0 8px">${tierPill(num(a.tier), 'Decision')}${a.needs_review ? reviewPill() : ''}<span class="pill nt">Distance rule: ${esc(TIERS[num(a.md_tier)] || '?')}</span></div>
          ${stackBar(a.p_tier)}
          <div style="margin-top:10px">${reasonsHtml(a.reasons)}</div>
          ${fc ? `<p class="hint" style="margin-top:8px">Event forecast (tier at the final CDM): <b>${esc(TIERS[num(fc.tier)] || '?')}</b>${Array.isArray(fc.p_tier) ? ` · P = ${fc.p_tier.map((p, i) => `${TIERS[i][0]} ${pct(p)}`).join(' / ')}` : ''}${fc.description ? ` – ${esc(fc.description)}` : ''}</p>` : ''}
        </div>
        <div class="panel"><div class="panel-head"><h3>Analyst action</h3></div>${actionsHtml('evaluator', id, decisionsFor('evaluator', id))}
          <p class="hint" style="margin-top:8px">Inputs used: only the 13 allowed columns. Covariance: c_sigma_r ${esc(fmtDist(c.c_sigma_r))}${num(c.c_sigma_r) >= 6e7 ? ' (unavailable sentinel)' : ''}, t_sigma_r ${esc(fmtDist(c.t_sigma_r))}; max-Pc bound ${esc(pcText(c.max_risk_estimate))}; F10 ${esc(fmtNum(c.F10, 0))}, AP ${esc(fmtNum(c.AP, 0))}.</p></div>
        <div class="panel wide"><div class="panel-head"><h3>CDM history</h3><span class="hint">Rows ordered from first to latest CDM</span></div>
          <div class="tbl-wrap"><table class="tbl"><thead><tr><th>#</th><th>days to TCA</th><th>miss distance</th><th>rel. speed</th><th>est. Pc</th><th>80% interval</th><th>max-Pc bound</th><th>true Pc</th><th>decision</th><th>review</th></tr></thead><tbody>
          ${cdms.map((x, i) => { const xa = x.assessment || {}; return `<tr class="click${i === idx ? ' chosen' : ''}" data-cdm="${i}"><td>${i + 1}</td><td>${esc(fixed(x.time_to_tca, 2))} d</td><td>${esc(fmtDist(x.miss_distance))}</td><td>${esc(fmtSpeed(x.relative_speed))}</td><td>${esc(pcText(xa.log_pc))}</td><td>${esc(pcText(xa.log_pc_lo))} – ${esc(pcText(xa.log_pc_hi))}</td><td>${esc(pcText(x.max_risk_estimate))}</td><td>${esc(pcText(x.true_log_pc))}</td><td>${tierPill(num(xa.tier))}</td><td>${xa.needs_review ? reviewPill() : ''}</td></tr>`; }).join('')}
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
    const ann = (y, col, txt) => ({ xref: 'paper', x: 0, y, text: txt, showarrow: false, xanchor: 'left', yanchor: 'bottom', font: { color: col, size: 10 } });
    const sel = [x[idx]];
    const pcData = [
      { x, y: hi, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
      { x, y: lo, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(138,180,248,.18)', name: '80% interval', hoverinfo: 'skip' },
      { x, y: med, mode: 'lines+markers', name: 'Model median', line: { color: COLORS.primaryText, width: 2 }, marker: { size: 8, color: COLORS.primaryText, line: { color: COLORS.panel, width: 2 } },
        customdata: cdms.map((c) => [pcText(get(c, 'assessment.log_pc')), pcText(get(c, 'assessment.log_pc_lo')), pcText(get(c, 'assessment.log_pc_hi'))]),
        hovertemplate: '%{x:.2f} d to TCA<br>est. Pc %{customdata[0]} (80%: %{customdata[1]} – %{customdata[2]})<extra>model</extra>' },
      { x, y: tru, mode: 'lines+markers', name: 'Ground truth (risk)', line: { color: COLORS.text, width: 1.5, dash: 'dot' }, marker: { size: 8, symbol: 'diamond', color: COLORS.text, line: { color: COLORS.panel, width: 2 } },
        hovertemplate: '%{x:.2f} d<br>true log10 Pc %{y:.2f}<extra>ground truth</extra>' },
      { x, y: bound, mode: 'lines', name: 'Max-Pc bound (input)', line: { color: COLORS.muted, width: 1, dash: 'dashdot' }, hovertemplate: '%{x:.2f} d<br>max_risk_estimate %{y:.2f}<extra>input</extra>' },
      { x: sel, y: [med[idx]], mode: 'markers', name: 'Selected CDM', marker: { size: 16, color: 'rgba(0,0,0,0)', line: { color: COLORS.live, width: 2 } }, hoverinfo: 'skip' },
    ];
    plotReact(pcEl, pcData, plotlyBase({
      margin: { l: 50, r: 12, t: 30, b: 40 },
      xaxis: { title: { text: 'days to TCA', font: { size: 11 } }, autorange: 'reversed', gridcolor: COLORS.grid, zeroline: false },
      yaxis: { title: { text: 'log10 Pc', font: { size: 11 } }, range: [ymin, 0.3], gridcolor: COLORS.grid, zeroline: false },
      shapes: [hl(-6, COLORS.elev), hl(-4, COLORS.crit)],
      annotations: [ann(-6, COLORS.elev, 'Elevated ≥ 1e-6'), ann(-4, COLORS.crit, 'Critical ≥ 1e-4')],
    }), PCFG);
    if (!pcEl._cw) { pcEl._cw = true; pcEl.on('plotly_click', (ev) => { const p = ev.points && ev.points[0]; if (p && isNum(p.pointIndex) && p.curveNumber >= 2 && p.curveNumber <= 4) { S.test.cdmIdx = p.pointIndex; renderTestDetail(); } }); }

    const md = cdms.map((c) => num(c.miss_distance));
    plotReact(mdEl, [
      { x, y: md.map((m) => (m == null ? null : m / 1000)), mode: 'lines+markers', name: 'Miss distance', line: { color: COLORS.primaryText, width: 2 }, marker: { size: 8, color: COLORS.primaryText, line: { color: COLORS.panel, width: 2 } }, hovertemplate: '%{x:.2f} d<br>%{y:.3f} km<extra></extra>' },
      { x: sel, y: [md[idx] / 1000], mode: 'markers', marker: { size: 16, color: 'rgba(0,0,0,0)', line: { color: COLORS.live, width: 2 } }, hoverinfo: 'skip', showlegend: false },
    ], plotlyBase({
      showlegend: false, margin: { l: 50, r: 12, t: 10, b: 40 },
      xaxis: { title: { text: 'days to TCA', font: { size: 11 } }, autorange: 'reversed', gridcolor: COLORS.grid, zeroline: false },
      yaxis: { title: { text: 'miss distance (km)', font: { size: 11 } }, type: 'log', dtick: 1, tickformat: '~g', gridcolor: COLORS.grid },
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: COLORS.crit, dash: 'dash', width: 1 } }, { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 5, y1: 5, line: { color: COLORS.elev, dash: 'dash', width: 1 } }],
      annotations: [{ xref: 'paper', x: 1, y: 0, yref: 'y', text: '1 km', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: COLORS.crit, size: 10 } },
        { xref: 'paper', x: 1, y: Math.log10(5), yref: 'y', text: '5 km', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: COLORS.elev, size: 10 } }],
    }), PCFG);

    const R = cdms.map((c) => num(c.relative_position_r)); const T = cdms.map((c) => num(c.relative_position_t)); const N = cdms.map((c) => num(c.relative_position_n));
    const data3 = [
      { type: 'scatter3d', mode: 'lines+markers', name: 'Miss vector per CDM', x: T, y: N, z: R,
        line: { color: COLORS.primaryText, width: 3 }, marker: { size: 4, color: x, colorscale: [[0, COLORS.primaryText], [1, '#2D3B49']], reversescale: true },
        customdata: x, hovertemplate: '%{customdata:.2f} d to TCA<br>T %{x:.0f} m · N %{y:.0f} m · R %{z:.0f} m<extra></extra>' },
      { type: 'scatter3d', mode: 'markers', name: 'Primary (origin)', x: [0], y: [0], z: [0], marker: { size: 5, color: COLORS.a }, hovertemplate: 'Primary at origin<extra></extra>' },
    ];
    const c = cdms[idx];
    const vr = num(c.relative_velocity_r); const vt = num(c.relative_velocity_t); const vn = num(c.relative_velocity_n);
    if (vr != null && vt != null && vn != null && R[idx] != null) {
      const vmag = Math.hypot(vr, vt, vn) || 1;
      const dt = 2.5 * Math.max(200, num(c.miss_distance) || 1000) / vmag;
      const pts = [-1, 0, 1].map((s) => [T[idx] + vt * dt * s, N[idx] + vn * dt * s, R[idx] + vr * dt * s]);
      data3.push({ type: 'scatter3d', mode: 'lines', name: 'Straight-line flyby (display only)', x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), z: pts.map((p) => p[2]),
        line: { color: COLORS.b, width: 4, dash: 'dash' }, hovertemplate: 'flyby from relative_velocity (display only)<extra></extra>' });
    }
    data3.push({ type: 'scatter3d', mode: 'markers', name: 'Selected CDM', x: [T[idx]], y: [N[idx]], z: [R[idx]], marker: { size: 7, color: COLORS.live, symbol: 'diamond' }, hoverinfo: 'skip' });
    const ax = (t) => ({ title: { text: t, font: { size: 10 } }, gridcolor: COLORS.grid, zerolinecolor: COLORS.line, backgroundcolor: 'rgba(0,0,0,0)', color: COLORS.text2 });
    plotReact(rtnEl, data3, plotlyBase({
      margin: { l: 0, r: 0, t: 0, b: 0 }, legend: { orientation: 'h', x: 0, y: 1.02, font: { size: 10 } },
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
      el.innerHTML = `<div class="state ${S.metricsErr ? 'err-state' : ''}"><b>${notGen ? 'Metrics not generated yet.' : 'Metrics unavailable.'}</b> ${S.metricsErr ? esc(S.metricsErr) + '. ' : ''}${notGen ? 'Run <code>python src/train.py</code> to write <code>outputs/metrics.json</code>, then reload.' : ''}<br><br><button type="button" class="btn sm" data-act="retry-metrics">Reload</button></div>`;
      return;
    }
    const methods = Array.isArray(m.methods) ? m.methods : [];
    const chosen = methods.find((x) => x.name === m.chosen_method) || methods.find((x) => x.kind === 'model') || methods[0];
    const base = methods.find((x) => /miss-distance rule/i.test(x.name || '')) || methods.find((x) => x.kind === 'baseline');
    const split = S.metricSplit || 'test';
    const ct = get(chosen, split) || {}; const bt = get(base, split) || {};
    const delta = (a, b, lowerBetter) => { if (!isNum(num(a)) || !isNum(num(b))) return ''; const d = num(a) - num(b); const good = lowerBetter ? d < 0 : d > 0; return `<div class="d" style="color:${d === 0 ? COLORS.muted : (good ? COLORS.live : COLORS.crit)}">${d > 0 ? '+' : ''}${Number.isInteger(d) ? d : d.toFixed(3)} vs distance rule</div>`; };
    const ev = m.eval_set || {};
    const known = ['generated_at', 'eval_set', 'cv_set', 'chosen_method', 'thresholds', 'methods', 'regression', 'review_queue', 'pc_driven_escalations', 'calibration', 'calibration_elevated', 'feature_importance', 'failure_cases', 'event_forecast', 'screening_model', 'notes', 'runtime_s'];
    const extra = Object.keys(m).filter((k) => !known.includes(k));
    const rq = m.review_queue || {}; const pe = m.pc_driven_escalations || {}; const rg = m.regression || {};
    const recC = get(ct, 'recall.2');

    let html = `<div class="page-head"><h2>Model evidence</h2>
      <span class="pill nt">${esc(m.chosen_method || 'chosen method not named')}</span>
      <span class="hint">generated ${esc(fmtUTC(m.generated_at))} · eval on ${esc(ev.name || '?')} (${esc(fmtNum(ev.n_rows, 0))} CDMs, ${esc(fmtNum(ev.n_events, 0))} events${Array.isArray(ev.true_tier_counts) ? `; true tiers S/E/C ${ev.true_tier_counts.join(' / ')}` : ''})</span>
      <div class="seg" id="metric-split" role="group" aria-label="Evaluation split" style="margin-left:auto">
        <button type="button" data-split="test" class="${split === 'test' ? 'on' : ''}">Held-out test</button><button type="button" data-split="cv" class="${split === 'cv' ? 'on' : ''}">Cross-validation</button></div></div>`;
    html += `<div class="tiles">
      <div class="tile"><div class="v">${f3(ct.macro_f1)}</div><div class="l">macro F1 (${split})</div>${delta(ct.macro_f1, bt.macro_f1)}</div>
      <div class="tile"><div class="v">${f3(ct.f2_critical)}</div><div class="l">F2 Critical (recall-weighted)</div>${delta(ct.f2_critical, bt.f2_critical)}</div>
      <div class="tile"><div class="v">${isNum(num(recC)) ? pct(recC, 1) : '–'}</div><div class="l">Critical recall</div>${delta(recC, get(bt, 'recall.2'))}</div>
      <div class="tile"><div class="v">${esc(fmtNum(ct.missed_critical, 0))}</div><div class="l">missed Critical CDMs</div>${delta(ct.missed_critical, bt.missed_critical, true)}</div>
      <div class="tile"><div class="v">${esc(fmtNum(ct.false_alarms_critical, 0))}</div><div class="l">false Critical alarms</div>${delta(ct.false_alarms_critical, bt.false_alarms_critical, true)}</div>
      <div class="tile"><div class="v">${isNum(num(rg.interval_coverage_80)) ? pct(rg.interval_coverage_80, 1) : '–'}</div><div class="l">80% interval coverage</div><div class="d">target 80%</div></div>
    </div>`;
    // methods table
    html += `<div class="panel" style="margin-bottom:12px"><div class="panel-head"><h3>Baselines vs model</h3><span class="hint">Recall / precision per tier on the ${split === 'test' ? 'held-out test set' : 'cross-validation folds'} · S / E / C = Safe / Elevated / Critical</span></div><div class="tbl-wrap"><table class="tbl">
      <thead><tr><th rowspan="2">Method</th><th rowspan="2">kind</th><th class="grp" colspan="2">CV</th><th class="grp" colspan="2">Test</th><th class="grp" colspan="3">Recall (${split})</th><th class="grp" colspan="3">Precision (${split})</th><th rowspan="2">missed Crit</th><th rowspan="2">Crit called Safe</th><th rowspan="2">false Crit alarms</th></tr>
      <tr><th>macro F1</th><th>F2 Crit</th><th>macro F1</th><th>F2 Crit</th><th>S</th><th>E</th><th>C</th><th>S</th><th>E</th><th>C</th></tr></thead><tbody>`;
    const best = (k, sp) => Math.max.apply(null, methods.map((x) => num(get(x, `${sp}.${k}`))).filter((v) => v != null).concat([-1]));
    methods.forEach((x) => {
      const s = get(x, split) || {};
      const cell = (sp, k) => { const v = num(get(x, `${sp}.${k}`)); return `<td class="${v != null && v === best(k, sp) ? 'best' : ''}">${f3(v)}</td>`; };
      const arr = (k, i) => `<td>${isNum(num(get(s, `${k}.${i}`))) ? pct(get(s, `${k}.${i}`), 1) : '–'}</td>`;
      html += `<tr class="${x === chosen ? 'chosen' : ''}" title="${esc(x.description || '')}"><td class="l"><b>${esc(x.name)}</b>${x === chosen ? ' <span class="pill nt">chosen</span>' : ''}${x.description ? `<div class="hint">${esc(x.description)}</div>` : ''}</td><td>${esc(x.kind || '')}</td>
        ${cell('cv', 'macro_f1')}${cell('cv', 'f2_critical')}${cell('test', 'macro_f1')}${cell('test', 'f2_critical')}
        ${arr('recall', 0)}${arr('recall', 1)}${arr('recall', 2)}${arr('precision', 0)}${arr('precision', 1)}${arr('precision', 2)}
        <td>${esc(fmtNum(s.missed_critical, 0))}</td><td>${esc(fmtNum(s.critical_called_safe, 0))}</td><td>${esc(fmtNum(s.false_alarms_critical, 0))}</td></tr>`;
    });
    if (!methods.length) html += '<tr><td colspan="15" class="l hint">No methods in metrics.json.</td></tr>';
    html += '</tbody></table></div></div>';
    // confusion + calibration
    const opts = methods.map((x, i) => `<option value="${i}" ${x === chosen ? 'selected' : ''}>${esc(x.name)}</option>`).join('');
    html += `<div class="grid3" style="margin-bottom:12px">
      <div class="panel"><div class="panel-head"><h3>Confusion</h3><select id="cm-method" aria-label="Method for the confusion matrix" style="max-width:220px">${opts}</select></div><div id="cm-a" class="chart sm"></div></div>
      <div class="panel"><div class="panel-head"><h3>Confusion: ${esc(base ? base.name : 'baseline')}</h3></div><div id="cm-b" class="chart sm"></div></div>
      <div class="panel"><div class="panel-head"><h3>Calibration</h3><span class="hint">predicted vs observed frequency</span></div><div id="calib" class="chart sm"></div></div>
    </div>`;
    // review / escalations / regression / thresholds
    html += `<div class="grid3" style="margin-bottom:12px">
      <div class="panel"><div class="panel-head"><h3>Human review queue</h3></div><dl class="kv">
        <dt>CDMs routed to review</dt><dd>${isNum(num(rq.needs_review_rate)) ? pct(rq.needs_review_rate, 2) : '–'}</dd>
        <dt>Error rate inside review queue</dt><dd>${isNum(num(rq.error_rate_in_review_queue)) ? pct(rq.error_rate_in_review_queue, 1) : '–'}</dd>
        <dt>Error rate auto-decided</dt><dd>${isNum(num(rq.error_rate_auto_decided)) ? pct(rq.error_rate_auto_decided, 1) : '–'}</dd></dl>
        <p class="hint">Review flags fire when the 80% Pc interval straddles a tier boundary or the max-Pc bound reaches 1e-4; errors should concentrate there.</p></div>
      <div class="panel"><div class="panel-head"><h3>Pc-driven escalations</h3></div><dl class="kv">
        <dt>CDMs where Pc raises the tier</dt><dd>${esc(fmtNum(pe.n_rows_where_pc_raises_tier, 0))}</dd>
        <dt>caught by the model</dt><dd>${esc(fmtNum(pe.caught_by_model, 0))}</dd>
        <dt>caught by the distance rule</dt><dd>${esc(fmtNum(pe.caught_by_distance_rule, 0))}</dd></dl>
        <p class="hint">Cases the 1 km / 5 km rule cannot see: the value added over distance-only triage.</p></div>
      <div class="panel"><div class="panel-head"><h3>Pc regression &amp; thresholds</h3></div><dl class="kv">
        <dt>MAE log10 Pc (all)</dt><dd>${f3(rg.test_mae_log_pc)}</dd>
        <dt>MAE log10 Pc (high risk)</dt><dd>${f3(rg.test_mae_log_pc_high_risk)}</dd>
        <dt>80% coverage (non-floor)</dt><dd>${isNum(num(rg.interval_coverage_80_nonfloor)) ? pct(rg.interval_coverage_80_nonfloor, 1) : '–'}</dd>
        ${Object.keys(m.thresholds || {}).map((k) => { const t = m.thresholds[k]; return t && typeof t === 'object' ? `<dt>thresholds (${esc(k)})</dt><dd>P(crit) ${esc(pct(t.p_critical))} · P(elev) ${esc(pct(t.p_elevated))}</dd>` : ''; }).join('')}</dl>
        ${rg.note ? `<p class="hint">${esc(rg.note)}</p>` : ''}</div>
    </div>`;
    // event forecast + screening
    const ef = m.event_forecast;
    const sm = m.screening_model;
    const blockRows = (obj) => Object.keys(obj || {}).filter((k) => isMetricBlock(obj[k])).map((k) => { const b = obj[k]; return `<tr><td class="l">${esc(k.replace(/_/g, ' '))}</td><td>${f3(b.macro_f1)}</td><td>${f3(b.f2_critical)}</td><td>${isNum(num(get(b, 'recall.2'))) ? pct(get(b, 'recall.2'), 1) : '–'}</td><td>${esc(fmtNum(b.missed_critical, 0))}</td><td>${esc(fmtNum(b.false_alarms_critical, 0))}</td><td>${esc(fmtNum(b.n, 0))}</td></tr>`; }).join('');
    html += `<div class="grid2" style="margin-bottom:12px">
      <div class="panel"><div class="panel-head"><h3>Event forecast (tier at TCA)</h3></div>${ef ? `<p class="hint">${esc(ef.description || '')}</p>
        <div class="tbl-wrap"><table class="tbl"><thead><tr><th>method</th><th>macro F1</th><th>F2 Crit</th><th>Crit recall</th><th>missed Crit</th><th>false Crit</th><th>n events</th></tr></thead><tbody>${blockRows(ef) || '<tr><td colspan="7" class="l hint">no scored blocks</td></tr>'}</tbody></table></div>${ef.note ? `<p class="hint">${esc(ef.note)}</p>` : ''}` : '<div class="state">Not in metrics.json yet.</div>'}</div>
      <div class="panel"><div class="panel-head"><h3>Screening model (live SOCRATES inputs)</h3></div>${sm ? `<p class="hint">${esc(sm.description || '')}</p>
        <div class="tbl-wrap"><table class="tbl"><thead><tr><th>split</th><th>macro F1</th><th>F2 Crit</th><th>Crit recall</th><th>missed Crit</th><th>false Crit</th><th>n</th></tr></thead><tbody>${blockRows(sm) || '<tr><td colspan="7" class="l hint">no scored blocks</td></tr>'}</tbody></table></div>` : '<div class="state">Not in metrics.json yet.</div>'}</div>
    </div>`;
    // feature importance + failure cases
    const fcases = Array.isArray(m.failure_cases) ? m.failure_cases : [];
    html += `<div class="grid2" style="margin-bottom:12px">
      <div class="panel"><div class="panel-head"><h3>Feature importance</h3><span class="hint">share of total split gain</span></div><div id="fi" class="chart"></div></div>
      <div class="panel"><div class="panel-head"><h3>Where it fails</h3><span class="hint">click a row to open the event in the evaluator</span></div>
        ${fcases.length ? `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>event</th><th>days</th><th>miss</th><th>truth</th><th>pred</th><th>est. Pc</th><th>true Pc</th><th class="l">why</th></tr></thead><tbody>
        ${fcases.map((f) => `<tr class="click" data-goto="${esc(f.event_id)}" tabindex="0"><td><a href="#evaluator/${esc(f.event_id)}">#${esc(f.event_id)}</a></td><td>${esc(fixed(f.time_to_tca, 2))}</td><td>${esc(fmtDist(f.miss_distance))}</td><td>${tierPill(num(f.true_tier))}</td><td>${tierPill(num(f.pred_tier))}</td><td>${esc(pcText(f.log_pc))}</td><td>${esc(pcText(f.true_log_pc))}</td><td class="l">${esc(f.why || '')}</td></tr>`).join('')}
        </tbody></table></div>` : '<div class="state">No failure cases listed.</div>'}</div>
    </div>`;
    if (Array.isArray(m.notes) && m.notes.length) html += `<div class="panel" style="margin-bottom:12px"><div class="panel-head"><h3>Notes</h3></div><ul class="notes">${m.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul></div>`;
    if (extra.length) html += `<div class="panel"><details class="raw"><summary>Other keys in metrics.json (${extra.map(esc).join(', ')})</summary><pre>${esc(JSON.stringify(extra.reduce((o, k) => { o[k] = m[k]; return o; }, {}), null, 2))}</pre></details></div>`;
    el.innerHTML = html;
    renderMetricCharts(methods, chosen, base, split);
  }
  function confusionChart(el, block, title) {
    const cm = block && Array.isArray(block.confusion) ? block.confusion : null;
    if (!cm) { el.innerHTML = '<div class="state">No confusion matrix for this method / split.</div>'; return; }
    const rowsum = cm.map((r) => r.reduce((a, b) => a + (num(b) || 0), 0) || 1);
    const z = cm.map((r, i) => r.map((v) => (num(v) || 0) / rowsum[i]));
    const text = cm.map((r, i) => r.map((v) => `${(num(v) || 0).toLocaleString('en-US')}<br>${pct((num(v) || 0) / rowsum[i])}`));
    plotReact(el, [{ type: 'heatmap', z, x: TIERS.map((t) => `pred ${t}`), y: TIERS.map((t) => `true ${t}`), text, texttemplate: '%{text}', textfont: { size: 11, color: COLORS.text },
      colorscale: [[0, '#1D2732'], [1, '#1A73E8']], zmin: 0, zmax: 1, showscale: false, xgap: 2, ygap: 2,
      hovertemplate: '%{y} → %{x}<br>%{text}<extra></extra>' }],
    plotlyBase({ margin: { l: 86, r: 8, t: 8, b: 40 }, yaxis: { autorange: 'reversed', gridcolor: 'rgba(0,0,0,0)' }, xaxis: { side: 'bottom', gridcolor: 'rgba(0,0,0,0)' } }), PCFG);
  }
  function renderMetricCharts(methods, chosen, base, split) {
    if (!window.Plotly) { $$('#cm-a,#cm-b,#calib,#fi').forEach(noPlotly); return; }
    const sel = $('#cm-method');
    const draw = () => { const mth = methods[Number(sel.value)] || chosen; confusionChart($('#cm-a'), get(mth, split), mth && mth.name); };
    if (sel) { sel.onchange = draw; draw(); }
    confusionChart($('#cm-b'), get(base, split), base && base.name);
    const cal = Array.isArray(S.metrics.calibration) ? S.metrics.calibration : [];
    const cal2 = Array.isArray(S.metrics.calibration_elevated) ? S.metrics.calibration_elevated : [];
    const calEl = $('#calib');
    if (!cal.length && !cal2.length) calEl.innerHTML = '<div class="state">No calibration data.</div>';
    else {
      const tr = (arr, name, col) => ({ x: arr.map((c) => num(c.p_pred)), y: arr.map((c) => num(c.p_obs)), mode: 'lines+markers', name, line: { color: col, width: 2 },
        marker: { color: col, size: arr.map((c) => clamp(6 + Math.log10((num(c.n) || 1) + 1) * 2.5, 6, 16)), line: { color: COLORS.panel, width: 2 } },
        customdata: arr.map((c) => c.n), hovertemplate: `${name}<br>predicted %{x:.3f} · observed %{y:.3f}<br>n = %{customdata}<extra></extra>` });
      const data = [{ x: [0, 1], y: [0, 1], mode: 'lines', name: 'perfect', line: { color: COLORS.muted, dash: 'dot', width: 1 }, hoverinfo: 'skip' }];
      if (cal.length) data.push(tr(cal, 'P(Pc ≥ 1e-4)', COLORS.crit));
      if (cal2.length) data.push(tr(cal2, 'P(Pc ≥ 1e-6)', COLORS.elev));
      plotReact(calEl, data, plotlyBase({ margin: { l: 44, r: 10, t: 26, b: 40 }, xaxis: { title: { text: 'predicted probability', font: { size: 10.5 } }, range: [0, 1], gridcolor: COLORS.grid }, yaxis: { title: { text: 'observed frequency', font: { size: 10.5 } }, range: [0, 1], gridcolor: COLORS.grid } }), PCFG);
    }
    const fi = (Array.isArray(S.metrics.feature_importance) ? S.metrics.feature_importance : []).slice(0, 15).reverse();
    const fiEl = $('#fi');
    if (!fi.length) fiEl.innerHTML = '<div class="state">No feature importance.</div>';
    else {
      plotReact(fiEl, [{ type: 'bar', orientation: 'h', y: fi.map((f) => f.feature), x: fi.map((f) => num(f.gain_pct)), marker: { color: COLORS.primaryText },
        hovertemplate: '%{y}: %{x:.1f}% of gain<extra></extra>', text: fi.map((f) => `${fixed(f.gain_pct, 1)}%`), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: COLORS.text2 } }],
      plotlyBase({ margin: { l: 140, r: 40, t: 6, b: 30 }, xaxis: { ticksuffix: '%', gridcolor: COLORS.grid }, yaxis: { gridcolor: 'rgba(0,0,0,0)', tickfont: { size: 10.5 } } }), PCFG);
    }
  }

  // ================================================================ Assess one CDM
  const FIELDS = [
    { k: 'event_id', label: 'event_id', u: 'id', step: 1, help: 'any integer; groups CDM history' },
    { k: 'time_to_tca', label: 'time_to_tca', u: 'days', step: 'any', min: 0 },
    { k: 'miss_distance', label: 'miss_distance', u: 'm', step: 'any', min: 0 },
    { k: 'relative_speed', label: 'relative_speed', u: 'm/s', step: 'any', min: 0 },
    { k: 'relative_position_r', label: 'relative_position_r', u: 'm (radial)', step: 'any' },
    { k: 'relative_position_t', label: 'relative_position_t', u: 'm (along-track)', step: 'any' },
    { k: 'relative_position_n', label: 'relative_position_n', u: 'm (cross-track)', step: 'any' },
    { k: 'c_object_type', label: 'c_object_type', u: 'secondary', type: 'select' },
    { k: 'c_sigma_r', label: 'c_sigma_r', u: 'm', step: 'any', min: 0, help: '6e7 = covariance unavailable' },
    { k: 't_sigma_r', label: 't_sigma_r', u: 'm', step: 'any', min: 0 },
    { k: 'max_risk_estimate', label: 'max_risk_estimate', u: 'log10 Pc bound', step: 'any', max: 0 },
    { k: 'F10', label: 'F10', u: 'sfu', step: 'any', min: 0 },
    { k: 'AP', label: 'AP', u: 'index', step: 'any', min: 0 },
  ];
  // Real CDMs from data/test_data.csv (true tier computed from `risk`, shown only as a note).
  const PRESETS = {
    safe: { note: 'test_data.csv event 0: miss 32.9 km, ground-truth Pc 4.6e-8 (Safe).', v: { event_id: 0, time_to_tca: 5.922, miss_distance: 32878, relative_speed: 7929, relative_position_r: -376.3, relative_position_t: 27867.5, relative_position_n: 17443.8, c_object_type: 'UNKNOWN', c_sigma_r: 212.79, t_sigma_r: 7.354, max_risk_estimate: -7.2287, F10: 75, AP: 9 } },
    elevated: { note: 'test_data.csv event 2: miss 21.2 km (distance rule says Safe) but ground-truth Pc 2.3e-5 (Elevated) – a Pc-driven case.', v: { event_id: 2, time_to_tca: 6.852, miss_distance: 21175, relative_speed: 9279, relative_position_r: 14.7, relative_position_t: -16644.3, relative_position_n: -13090.9, c_object_type: 'UNKNOWN', c_sigma_r: 84.015, t_sigma_r: 2.385, max_risk_estimate: -4.4988, F10: 66, AP: 3 } },
    critical: { note: 'test_data.csv event 87: miss 1.03 km (distance rule says Elevated) but ground-truth Pc 8.4e-4 (Critical).', v: { event_id: 87, time_to_tca: 3.376, miss_distance: 1032, relative_speed: 14861, relative_position_r: 63.0, relative_position_t: -123.5, relative_position_n: 1023.5, c_object_type: 'DEBRIS', c_sigma_r: 33.218, t_sigma_r: 4.373, max_risk_estimate: -2.9172, F10: 78, AP: 3 } },
  };
  let assessBuilt = false;
  function ensureAssessForm() {
    if (assessBuilt) return;
    assessBuilt = true;
    $('#assess-fields').innerHTML = FIELDS.map((f) => `<div class="fld" data-k="${f.k}">
      <label for="f-${f.k}">${esc(f.label)} <span class="u">(${esc(f.u)})</span></label>
      ${f.type === 'select' ? `<select id="f-${f.k}" name="${f.k}">${OBJECT_TYPES.map((o) => `<option>${o}</option>`).join('')}</select>` :
        `<input id="f-${f.k}" name="${f.k}" type="number" inputmode="decimal" step="${f.step}"${f.min != null ? ` min="${f.min}"` : ''}${f.max != null ? ` max="${f.max}"` : ''} required>`}
      ${f.help ? `<span class="fh">${esc(f.help)}</span>` : ''}</div>`).join('');
    $('#assess-fields').addEventListener('input', checkAssessGeometry);
    const draft = lsGet('cw.assess.draft', null);
    applyPreset(draft && typeof draft === 'object' ? { v: draft, note: 'Restored your last input.' } : PRESETS.elevated);
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
      out.innerHTML = rel > 0.05 ? `<span style="color:var(--stale)">|r,t,n| = ${esc(fmtDist(r))} differs from miss_distance ${esc(fmtDist(md))} by ${esc(pct(rel))} – check the inputs.</span>` :
        `Geometry check: |r,t,n| = ${esc(fmtDist(r))} matches miss_distance.`;
    } else out.textContent = '';
  }
  async function submitAssess(ev) {
    ev.preventDefault();
    const { body, bad } = readAssessForm();
    const err = $('#assess-err');
    if (bad.length) { err.textContent = `Check ${bad.join(', ')}`; return; }
    err.textContent = '';
    lsSet('cw.assess.draft', body);
    const out = $('#assess-result');
    const btn = $('#assess-go');
    btn.disabled = true;
    out.innerHTML = '<div class="loading" style="padding:16px">Scoring the CDM…</div>';
    try {
      const a = await api('/api/assess', {}, { method: 'POST', body });
      renderAssessResult(a, body);
    } catch (e) {
      out.innerHTML = `<div class="err-box"><b>Assessment failed.</b> ${esc(e.message)}</div>`;
    } finally { btn.disabled = false; }
  }
  function pcScale(a) {
    const pos = (lp) => clamp(((num(lp) == null ? -30 : num(lp)) + 30) / 30 * 100, 0, 100);
    const lo = pos(a.log_pc_lo); const hi = pos(a.log_pc_hi); const md = pos(a.log_pc);
    return `<div class="pc-scale" role="img" aria-label="Estimated log10 Pc ${fixed(a.log_pc, 2)} with 80% interval ${fixed(a.log_pc_lo, 2)} to ${fixed(a.log_pc_hi, 2)}">
      <div class="pc-axis"></div>
      <div class="pc-band" style="left:${lo}%;width:${Math.max(0.6, hi - lo)}%"></div>
      <div class="pc-med" style="left:${md}%"></div>
      <div class="pc-top" style="left:${clamp(md, 6, 94)}%">${esc(pcText(a.log_pc))}</div>
      <div class="pc-thr" style="left:80%"></div><div class="pc-thr" style="left:86.67%"></div>
      <div class="pc-lbl" style="left:0%;transform:none">1e-30</div><div class="pc-lbl" style="left:50%">1e-15</div>
      <div class="pc-lbl" style="left:80%">1e-6</div><div class="pc-lbl" style="left:86.67%">1e-4</div><div class="pc-lbl" style="left:100%;transform:translateX(-100%)">1</div>
    </div>`;
  }
  function renderAssessResult(a, body) {
    const out = $('#assess-result');
    if (!a || typeof a !== 'object') { out.innerHTML = '<div class="err-box"><b>Unexpected response.</b></div>'; return; }
    const t = num(a.tier);
    out.innerHTML = `
      <div class="panel-head"><h3>Assessment</h3><span class="hint">${esc(fmtUTC(new Date().toISOString()))}</span></div>
      ${a._mock_note ? `<div class="screen-warn" style="margin:0 0 10px">${esc(a._mock_note)}</div>` : ''}
      <div class="dt-badges" style="margin:0 0 12px">${tierPill(t, 'Recommended').replace('class="pill', 'class="pill lg')}${a.needs_review ? reviewPill().replace('class="pill', 'class="pill lg') : ''}<span class="pill nt">Distance rule: ${esc(TIERS[num(a.md_tier)] || '?')}</span></div>
      <h4>Tier probabilities</h4>${stackBar(a.p_tier)}
      <h4>Estimated Pc with 80% interval</h4>${pcScale(a)}
      <dl class="kv" style="margin-top:6px">
        <dt>Median est. Pc</dt><dd>${pcHtml(a.log_pc)} <small>log10 ${esc(fixed(a.log_pc, 2))}</small></dd>
        <dt>80% interval</dt><dd>${esc(pcText(a.log_pc_lo))} – ${esc(pcText(a.log_pc_hi))}</dd>
        <dt>P(Pc ≥ 1e-6)</dt><dd>${esc(pct(a.p_pc_elevated, 1))}</dd>
        <dt>P(Pc ≥ 1e-4)</dt><dd>${esc(pct(a.p_pc_critical, 1))}</dd>
        ${a.forecast_tier != null ? `<dt>Forecast tier at TCA</dt><dd>${tierPill(num(a.forecast_tier))}</dd>` : ''}
        ${a.forecast && a.forecast.tier != null ? `<dt>Event forecast (tier at final CDM)</dt><dd>${tierPill(num(a.forecast.tier))}${Array.isArray(a.forecast.p_tier) ? ` <small>P(Critical) ${esc(pct(a.forecast.p_tier[2]))}</small>` : ''}</dd>` : ''}
        ${a.model ? `<dt>Model</dt><dd><small>${esc(a.model.set || (a.model.trained ? 'trained' : 'fallback'))}</small></dd>` : ''}
      </dl>
      ${Array.isArray(a.input_warnings) && a.input_warnings.length ? `<div class="screen-warn">${a.input_warnings.map(esc).join('<br>')}</div>` : ''}
      ${Array.isArray(a.ignored_fields) && a.ignored_fields.length ? `<p class="hint">Ignored fields (not allowed inputs): ${a.ignored_fields.map(esc).join(', ')}</p>` : ''}
      <h4>Why (verbatim)</h4>${reasonsHtml(a.reasons)}
      <div class="form-actions"><button type="button" class="btn sm" data-act="log-assess">Log this assessment</button><span class="hint">Decision support only – the analyst makes the call.</span></div>
      <details class="raw" style="margin-top:10px"><summary>Request sent to /api/assess</summary><pre>${esc(JSON.stringify(body, null, 2))}</pre></details>`;
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
      if (e.key === 'Escape') { q.value = ''; S.search = ''; renderFeed(); updatePoints(); q.blur(); }
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
    });
    // sidebar
    $('#side-toggle').addEventListener('click', () => {
      const sb = $('#sidebar'); const c = !sb.classList.contains('collapsed');
      sb.classList.toggle('collapsed', c);
      $('#side-toggle').setAttribute('aria-expanded', c ? 'false' : 'true');
      $('#side-toggle').title = c ? 'Expand sidebar' : 'Collapse sidebar';
      lsSet('cw.sideCollapsed', c);
      setTimeout(() => { if (G.viewer) G.viewer.resize(); resizePlots(); }, 220);
    });
    $('#feed-filter').addEventListener('click', (e) => { const b = e.target.closest('button[data-f]'); if (!b) return; setFeedFilter(b.dataset.f); renderFeed(); });
    $('#feed-sort').addEventListener('change', (e) => { S.feedSort = e.target.value; renderFeed(); });
    $('#feed-points').addEventListener('change', (e) => { S.onlyPoints = e.target.checked; renderFeed(); });
    $('#tq-pred').addEventListener('click', (e) => { const b = e.target.closest('button[data-t]'); if (!b) return; S.test.pred = b.dataset.t; setSeg('#tq-pred', 't', b.dataset.t); renderTestQueue(); });
    $('#tq-review').addEventListener('change', (e) => { S.test.review = e.target.checked; renderTestQueue(); });
    $('#tq-dis').addEventListener('change', (e) => { S.test.dis = e.target.checked; renderTestQueue(); });
    $('#tq-under').addEventListener('change', (e) => { S.test.under = e.target.checked; renderTestQueue(); });
    $('#tq-sort').addEventListener('change', (e) => { S.test.sort = e.target.value; renderTestQueue(); });
    // overlay
    $('#layer-grid').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-l]'); if (!b) return;
      const k = b.dataset.l; S.layers[k] = !S.layers[k];
      b.classList.toggle('on', S.layers[k]); b.setAttribute('aria-pressed', S.layers[k] ? 'true' : 'false');
      applyLayerVisibility();
    });
    $('#min-sev').addEventListener('click', (e) => { const b = e.target.closest('button[data-s]'); if (!b) return; S.minSev = Number(b.dataset.s); setSeg('#min-sev', 's', b.dataset.s); updatePoints(); });
    const collapser = (btnSel, boxSel) => $(btnSel).addEventListener('click', () => {
      const box = $(boxSel); const c = !box.classList.contains('collapsed');
      box.classList.toggle('collapsed', c);
      $(btnSel).innerHTML = c ? '+' : '&minus;';
      $(btnSel).setAttribute('aria-expanded', c ? 'false' : 'true');
      if (!c) setTimeout(resizePlots, 30);
    });
    collapser('#ovl-toggle', '#overlay');
    collapser('#kpi-toggle', '#kpi-card');
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
      if (f == null && a == null) { toast('Space weather unavailable right now.', true); return; }
      if (f != null) $('#f-F10').value = f;
      if (a != null) $('#f-AP').value = a;
      $('#preset-note').textContent = `F10 / AP set from ${get(S.sw, 'f107.source', 'NOAA SWPC')} (fetched ${fmtUTC(S.sw.fetched_at)}).`;
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
      if (qBtn) { $('#q').value = qBtn.dataset.q; S.search = qBtn.dataset.q; setFeedFilter('all'); renderFeed(); updatePoints(); return; }
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
      else if (a === 'clear-hour') { S.hourFilter = null; renderFeed(); }
      else if (a === 'feed-all') { setFeedFilter('all'); renderFeed(); }
      else if (a === 'log-assess' && S.lastAssess) {
        if (!S.analyst) { openLog(); $('#analyst').focus(); toast('Enter your initials in the decision log first.', true); return; }
        const { a: as, body } = S.lastAssess;
        S.log.push({ ts: new Date().toISOString(), analyst: S.analyst, view: 'assess', event_id: body.event_id, pair: `manual CDM (event_id ${body.event_id})`, action: 'acknowledge',
          note: 'Manual assessment logged', model_tier: TIERS[num(as.tier)], needs_review: !!as.needs_review, log_pc: as.log_pc, miss_distance_m: body.miss_distance, time_to_tca_d: body.time_to_tca, source: 'Assess form' });
        saveLog(); toast('Assessment logged.');
      }
    });
    document.addEventListener('input', (e) => { if (e.target.classList && e.target.classList.contains('analyst-inline')) setAnalyst(e.target.value); });
    document.addEventListener('keydown', (e) => {
      const card = e.target.closest && e.target.closest('.card[role="button"]');
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
    if (window.innerWidth <= 900 || window.innerHeight < 640) {
      $('#overlay').classList.add('collapsed'); $('#ovl-toggle').innerHTML = '+'; $('#ovl-toggle').setAttribute('aria-expanded', 'false');
    }
    if (lsGet('cw.sideCollapsed', false) && window.innerWidth > 900) { $('#sidebar').classList.add('collapsed'); $('#side-toggle').setAttribute('aria-expanded', 'false'); }
    wire();
    tickClock();
    setInterval(tickClock, 1000);
    setInterval(refreshCountdowns, 15000);
    renderFeed(); renderKpiCard(); renderLivePill(); renderSpaceWeather();
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
