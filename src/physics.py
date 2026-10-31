"""Physics-based collision probability (Pc) for CDMs, reproducing ESA's `risk` column.

Conventions established empirically on the ESA Kelvins Collision Avoidance dataset
(see scripts/validate_physics.py -> outputs/physics_validation.json):

* relative_position_* / relative_velocity_*: chaser minus target, expressed in the TARGET's
  RTN frame at TCA (|relative_position| = miss_distance, |relative_velocity| = relative_speed).
* t_sigma_* / t_c*_* (target) are in the target RTN frame; c_sigma_* / c_c*_* (chaser) are in the
  CHASER's OWN RTN frame (CCSDS CDM convention) and are rotated into the target RTN frame before
  being added. The t_c*_* / c_c*_* columns are correlation coefficients (det(C) reproduces
  t/c_position_covariance_det to 1e-6). With the rotation, the 3D Mahalanobis distance
  sqrt(d^T (C_t + C_c)^-1 d) reproduces ESA's `mahalanobis_distance`; without it, ~0.25 decades off.
  The chaser frame is rebuilt from target state (a_t, 0, 0) / (0, v_T, 0) + relative state, with v_T
  from the two objects' orbital energies at their two radii (osculating J2000 SMAs, see
  target_velocity_rtn): median |log10 ratio| vs ESA 5e-6, 98.3-98.5 % of rows within 0.1 %, 99.8 %
  within 1 % (circular speed: 2.7e-4 / 59-60 % / 99.8 %; energy without the radius term: 1.5e-5 /
  95-96 % / 99.7 %), train and test.
  ESA's mahalanobis_distance is the 3D one (the encounter-plane 2D distance is ~0.78 decades off).
* Hard-body radius: (t_span + c_span) / 2 (spans are object diameters in metres) is ESA's formula
  for ~87-89 % of CDMs; for the rest ESA used a different (mission/epoch-specific) size.
  hbr="spans" uses only that formula (CDM inputs only): Pc within 0.1 decade of ESA risk for 95 %
  of rows. hbr="auto" (default) instead solves each CDM's radius from max_risk_estimate (implied_hbr),
  which ESA computed with its own radius: 99.6 % within 0.1 decade. That number is CALIBRATED
  against an ESA output and therefore partly circular - quote the span number as the independent
  reproduction and label "auto" results as "radius calibrated on ESA max_risk_estimate".
* risk = log10 of the 2D (short-encounter, Foster/Chan/Alfriend-Akella) Pc in the encounter plane,
  floored at -30.
* max_risk_scaling = m_2D^2 / 2, the ANALYTIC small-HBR optimum of the covariance scale k (k
  multiplies the covariance, not the sigmas), and max_risk_estimate = log10 Pc evaluated at that k -
  not the exact maximum over k. Evidence (valid rows, span HBR): max_risk_scaling = m_2D^2/2 within
  1 % for 99.4-99.5 % of rows (exact argmax: 98.1-98.5 %); where HBR is comparable to sigma
  (HBR^2/(k0 s^2) > 0.05, ~1 % of rows) m_2D^2/2 matches to a median 0.0004-0.0006 decades vs
  0.008-0.009 for the exact argmax; on the rows where the span radius is ESA's and the two
  definitions differ (n = 105 train / 19 test), Pc at m_2D^2/2 matches max_risk_estimate to a median
  0.002 decades vs 0.007-0.008 for the exact maximum (which can be up to 0.75-0.87 decades higher).
  full_pc reports both: log_pc_max_esa / max_scale_esa (ESA's definition) and log_pc_max_phys /
  max_scale (exact maximum / argmax, golden-section search). implied_hbr solves ESA's definition.
* Placeholder sigmas (6.378136e7 m) are used by ESA as-is (risk ~ -14); we reproduce those values
  (median |err| 0.009) but flag them pc_valid=False. NaN chaser covariance -> ESA risk = -30.

Public API (other modules integrate these):
    full_pc(df, hbr="auto") -> DataFrame[log_pc_phys, log_pc_max_phys, hbr_m, pc_valid, mahal_3d,
                             mahal_2d, miss_plane_x, miss_plane_y, sigma_major, sigma_minor]
                             (+ max_scale, log_pc_max_esa, max_scale_esa, hbr_spans_m, hbr_source)
    Required columns: relative_position_*, relative_velocity_*, {t,c}_sigma_{r,t,n},
    {t,c}_{ct_r,cn_r,cn_t}, t_span, c_span, t_j2k_sma (+ c_j2k_sma, max_risk_estimate optional).
    NOTE: most of these are outside features.ALLOWED_COLUMNS - this module is a physics reference /
    explanation layer, not an input of the 13-column model.
    manoeuvre_options(row, target_log_pc=-5.0, burn_lead_orbits=(0.5, 1, 2, 3), max_dv=0.5, ...)
        -> list (one entry per lead time) of dict | None; robust to +-10 % burn error and to the
        target's eccentricity by default (see its docstring and MANOEUVRE_ASSUMPTIONS)
    best_manoeuvre(options) -> the feasible entry with the smallest |dv| (or None)
Lower-level vectorised helpers: position_covariance, target_velocity_rtn, chaser_rtn_axes,
combined_covariance, encounter_plane, log10_pc_2d, log10_pc_max_2d, log10_pc_max_esa_2d,
implied_hbr, cw_displacement, hbr_from_spans.

Pc quadrature: in the principal-axis frame of the 2D covariance, the minor-axis integral over the
disc chord is done analytically with log-space normal CDFs, and the remaining 1D integral along the
major axis is done by Gauss-Legendre in phi (x = HBR sin phi, which removes the sqrt endpoint
singularity) on a window that holds all but e^-40 of the integrand. The marginal is log-concave
(Prekopa), so the window is found by golden-section + bisection. Everything stays in log space,
so far-tail cases (log10 Pc down to -30 and below) and HBR >> sigma are handled.
"""
import numpy as np
import pandas as pd
from scipy.special import log_ndtr

# ---------------------------------------------------------------- constants
MU_EARTH = 3.986004418e14      # m^3 s^-2, Earth gravitational parameter
SIGMA_PLACEHOLDER = 6.0e7      # m; CDM sigmas >= this (dataset uses 6.378136e7) mean "no covariance"
LOG10_PC_FLOOR = -30.0         # ESA floors risk at -30
HBR_FORMULA = "(t_span + c_span) / 2"
LN10 = np.log(10.0)

# quadrature settings
N_GL = 96                      # Gauss-Legendre nodes along the major axis (windowed path)
N_GL_EASY = 48                 # nodes for "easy" rows (integrand varies < EASY_NATS over the disc)
EASY_NATS = 20.0
GOLDEN_ITERS = 32              # mode search of the (log-concave) marginal
BISECT_ITERS = 26              # window edge search
WINDOW_NATS = 40.0             # integrand below exp(-40) x peak is dropped
CHUNK = 20000                  # rows per vectorised block (memory bound)
_GL_T, _GL_W = np.polynomial.legendre.leggauss(N_GL)
_GL_T_EASY, _GL_W_EASY = np.polynomial.legendre.leggauss(N_GL_EASY)
_GOLD = (np.sqrt(5.0) - 1.0) / 2.0

# max-Pc search: golden-section in ln k on [ln k0 - 4, ln k0 + 4] when the small-HBR solution
# k0 = m^2 / 2 is not accurate enough (HBR^2 / (k0 sigma_minor^2) above this threshold)
MAXPC_REFINE_RATIO = 1e-2
MAXPC_ITERS = 18

POS_COLS = ["relative_position_r", "relative_position_t", "relative_position_n"]
VEL_COLS = ["relative_velocity_r", "relative_velocity_t", "relative_velocity_n"]


# ---------------------------------------------------------------- covariance assembly
def position_covariance(s_r, s_t, s_n, c_tr, c_nr, c_nt):
    """(n,3,3) RTN position covariance from sigmas (m) and correlation coefficients."""
    s_r, s_t, s_n, c_tr, c_nr, c_nt = (np.asarray(a, dtype=float) for a in (s_r, s_t, s_n, c_tr, c_nr, c_nt))
    C = np.empty(s_r.shape + (3, 3))
    C[..., 0, 0] = s_r * s_r
    C[..., 1, 1] = s_t * s_t
    C[..., 2, 2] = s_n * s_n
    C[..., 0, 1] = C[..., 1, 0] = c_tr * s_r * s_t
    C[..., 0, 2] = C[..., 2, 0] = c_nr * s_r * s_n
    C[..., 1, 2] = C[..., 2, 1] = c_nt * s_t * s_n
    return C


def _cov_from_df(df, prefix):
    g = lambda c: df[c].to_numpy(dtype=float)  # noqa: E731
    return position_covariance(g(f"{prefix}_sigma_r"), g(f"{prefix}_sigma_t"), g(f"{prefix}_sigma_n"),
                               g(f"{prefix}_ct_r"), g(f"{prefix}_cn_r"), g(f"{prefix}_cn_t"))


ENERGY_MIN_REL_VT = 100.0     # m/s; below this |relative_velocity_t| the energy solution is ill-posed
ENERGY_MAX_DEV = 0.03          # energy-based target speed accepted if within 3% of circular speed


def target_velocity_rtn(t_sma_km, c_sma_km=None, rel_vel=None, rel_pos=None):
    """Target inertial velocity in its own RTN frame: (0, v_T, 0) (radial velocity neglected).

    Default v_T = sqrt(mu / a_t) (vis-viva, near-circular, r = a). When the chaser semi-major axis
    and the relative velocity are given, v_T is instead solved from the vis-viva energies of the two
    objects: |v_t + dv|^2 - |v_t|^2 = mu (1/a_t - 1/a_c) + 2 mu (1/r_c - 1/r_t), i.e.
        v_T = (mu (1/a_t - 1/a_c) + 2 mu (1/r_c - 1/r_t) - |dv|^2) / (2 dv_T).
    The radius term (r_c = |r_t + rel_pos|, r_t ~ a_t; ~ -2 mu dr / r^2 with dr ~ relative_position_r)
    is included when rel_pos is given; without it the energy speed was slightly worse than the
    circular one on a 0.1 % tail of rows (99.72 % vs 99.79 % of Mahalanobis distances within 1 %);
    with it, it is better everywhere (median 5e-6, 99.8 % within 1 %). The energy solution is used
    when |dv_T| > ENERGY_MIN_REL_VT and it lies within ENERGY_MAX_DEV of the circular speed (else the
    circular speed). Remaining neglected term: the target's radial velocity (v_R dv_R / dv_T).
    """
    a = np.asarray(t_sma_km, dtype=float) * 1e3
    vt = np.sqrt(MU_EARTH / a)
    if c_sma_km is not None and rel_vel is not None:
        ac = np.asarray(c_sma_km, dtype=float) * 1e3
        dv = np.asarray(rel_vel, dtype=float)
        dvt = dv[..., 1]
        d_energy = MU_EARTH * (1.0 / a - 1.0 / ac)
        if rel_pos is not None:
            rp = np.asarray(rel_pos, dtype=float)
            rc = np.sqrt((a + rp[..., 0]) ** 2 + rp[..., 1] ** 2 + rp[..., 2] ** 2)
            drc = (2.0 * a * rp[..., 0] + (rp * rp).sum(-1)) / (rc + a)       # r_c - r_t, cancellation-free
            d_energy = d_energy - 2.0 * MU_EARTH * drc / (rc * a)              # 2 mu (1/r_c - 1/r_t)
        with np.errstate(divide="ignore", invalid="ignore"):
            ve = (d_energy - (dv * dv).sum(-1)) / (2.0 * dvt)
        use = (np.abs(dvt) > ENERGY_MIN_REL_VT) & np.isfinite(ve) & (np.abs(ve / vt - 1.0) < ENERGY_MAX_DEV)
        vt = np.where(use, ve, vt)
    v = np.zeros(np.shape(vt) + (3,))
    v[..., 1] = vt
    return v


def chaser_rtn_axes(rel_pos, rel_vel, t_sma_km, c_sma_km=None, radius_term=True):
    """(n,3,3) matrix whose COLUMNS are the chaser's R, T, N unit vectors in target-RTN coordinates.

    Target position = (a, 0, 0), velocity = (0, v_T, 0) in target RTN (see target_velocity_rtn;
    radius_term=False drops the 2 mu (1/r_c - 1/r_t) energy term, kept only for comparison);
    chaser state = target state + relative state. Radial axes nearly coincide; T/N are rotated about R
    by roughly the angle between the two orbit planes at TCA. (Rotating-frame terms omega x rho are
    < 1% and ignored.)
    """
    rel_pos = np.asarray(rel_pos, dtype=float)
    rel_vel = np.asarray(rel_vel, dtype=float)
    a = np.asarray(t_sma_km, dtype=float) * 1e3
    r_c = rel_pos.copy()
    r_c[..., 0] += a
    v_c = target_velocity_rtn(t_sma_km, c_sma_km, rel_vel, rel_pos if radius_term else None) + rel_vel
    R = r_c / np.linalg.norm(r_c, axis=-1, keepdims=True)
    N = np.cross(r_c, v_c)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    T = np.cross(N, R)
    return np.stack([R, T, N], axis=-1)


def combined_covariance(df, frame="chaser_rtn"):
    """Combined target+chaser position covariance in the target RTN frame, (n,3,3).

    frame="chaser_rtn" (default, matches ESA): chaser covariance rotated from its own RTN frame,
        target speed from the energy relation incl. the radius term (needs c_j2k_sma; falls back to
        the circular speed).
    frame="chaser_rtn_no_radius_term": energy relation without the 2 mu (1/r_c - 1/r_t) term
        (the previous default; comparison only).
    frame="chaser_rtn_circular": same rotation with the circular target speed sqrt(mu / a_t).
    frame="target_rtn": chaser covariance used as if already in the target RTN frame (hypothesis (a)).
    """
    Ct = _cov_from_df(df, "t")
    Cc = _cov_from_df(df, "c")
    if frame in ("chaser_rtn", "chaser_rtn_no_radius_term", "chaser_rtn_circular"):
        c_sma = (df["c_j2k_sma"].to_numpy(float)
                 if (frame != "chaser_rtn_circular" and "c_j2k_sma" in df) else None)
        M = chaser_rtn_axes(df[POS_COLS].to_numpy(float), df[VEL_COLS].to_numpy(float),
                            df["t_j2k_sma"].to_numpy(float), c_sma,
                            radius_term=(frame == "chaser_rtn"))
        Cc = np.einsum("nij,njk,nlk->nil", M, Cc, M)
    elif frame != "target_rtn":
        raise ValueError(f"unknown frame {frame!r}")
    return Ct + Cc


def covariance_status(df):
    """Boolean masks: placeholder (any sigma >= SIGMA_PLACEHOLDER) and missing (NaN) covariance."""
    sig = df[[f"{p}_sigma_{a}" for p in "tc" for a in "rtn"]
             + [f"{p}_{k}" for p in "tc" for k in ("ct_r", "cn_r", "cn_t")]].to_numpy(float)
    missing = np.isnan(sig).any(axis=1)
    placeholder = (np.nan_to_num(sig[:, :6], nan=0.0) >= SIGMA_PLACEHOLDER).any(axis=1)
    return placeholder, missing


def hbr_from_spans(t_span, c_span):
    """Hard-body radius (m) used by ESA: (t_span + c_span) / 2."""
    return (np.asarray(t_span, dtype=float) + np.asarray(c_span, dtype=float)) / 2.0


# ---------------------------------------------------------------- encounter plane
def encounter_plane(rel_pos, rel_vel, C):
    """Project the miss vector and covariance onto the plane perpendicular to the relative velocity.

    Basis: e1 = target radial direction made perpendicular to v (falls back to N when v is radial),
    e2 = v_hat x e1. Returns x2 (n,2), C2 (n,2,2), B (n,2,3) with rows e1, e2.
    """
    rel_pos = np.atleast_2d(np.asarray(rel_pos, dtype=float))
    rel_vel = np.atleast_2d(np.asarray(rel_vel, dtype=float))
    vn = np.linalg.norm(rel_vel, axis=-1, keepdims=True)
    vhat = rel_vel / np.where(vn > 0, vn, np.nan)
    ref = np.zeros_like(vhat)
    ref[:, 0] = 1.0
    e1 = ref - vhat[:, :1] * vhat
    small = np.linalg.norm(e1, axis=-1) < 1e-6
    if small.any():
        alt = np.zeros_like(vhat[small])
        alt[:, 2] = 1.0
        e1[small] = alt - (vhat[small] * alt).sum(-1, keepdims=True) * vhat[small]
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(vhat, e1)
    B = np.stack([e1, e2], axis=1)
    x2 = np.einsum("nij,nj->ni", B, rel_pos)
    C2 = np.einsum("nij,njk,nlk->nil", B, np.asarray(C, dtype=float), B)
    return x2, C2, B


def _principal(x2, C2):
    """Principal-axis decomposition of the 2D covariance: (mx, my, s_major, s_minor)."""
    a, b, c = C2[:, 0, 0], C2[:, 0, 1], C2[:, 1, 1]
    half_tr = 0.5 * (a + c)
    disc = np.sqrt(0.25 * (a - c) ** 2 + b * b)
    l1 = half_tr + disc
    l2 = (a * c - b * b) / np.where(l1 > 0, l1, np.nan)   # stable small eigenvalue
    th = 0.5 * np.arctan2(2.0 * b, a - c)
    ct, st = np.cos(th), np.sin(th)
    mx = ct * x2[:, 0] + st * x2[:, 1]
    my = -st * x2[:, 0] + ct * x2[:, 1]
    with np.errstate(invalid="ignore"):
        return mx, my, np.sqrt(l1), np.sqrt(l2)


def mahalanobis_2d(x2, C2):
    det = C2[:, 0, 0] * C2[:, 1, 1] - C2[:, 0, 1] ** 2
    q = (C2[:, 1, 1] * x2[:, 0] ** 2 - 2 * C2[:, 0, 1] * x2[:, 0] * x2[:, 1] + C2[:, 0, 0] * x2[:, 1] ** 2) / det
    with np.errstate(invalid="ignore"):
        return np.sqrt(q)


def mahalanobis_3d(d, C):
    d = np.asarray(d, dtype=float)
    try:
        sol = np.linalg.solve(C, d[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = np.full_like(d, np.nan)
        for i in range(len(d)):
            try:
                sol[i] = np.linalg.solve(C[i], d[i])
            except np.linalg.LinAlgError:
                pass
    q = np.einsum("ni,ni->n", d, sol)
    with np.errstate(invalid="ignore"):
        return np.sqrt(q)


# ---------------------------------------------------------------- 2D Pc quadrature
def _ln_interval_prob(h, my, s2):
    """ln P(-h <= Y <= h), Y ~ N(my, s2^2), accurate deep in the tails."""
    a = (-h - my) / s2
    b = (h - my) / s2
    flip = (a + b) > 0                      # use the lower tail for numerical accuracy
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)
    la, lb = log_ndtr(lo), log_ndtr(hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return lb + np.log(-np.expm1(la - lb))


def _ln_marginal(x, mx, my, s1, s2, R):
    """ln of N(x; mx, s1) * P(|Y| <= sqrt(R^2 - x^2)) without the -ln(s1 sqrt(2 pi)) constant."""
    h = np.sqrt(np.maximum(R * R - x * x, 0.0))
    z = (x - mx) / s1
    return -0.5 * z * z + _ln_interval_prob(h, my, s2)


def _ln_pc_principal(mx, my, s1, s2, R):
    """Natural-log 2D Pc in the principal frame (s1 = major-axis sigma). 1D arrays of equal length.

    "Easy" rows (log-integrand varies by < EASY_NATS across the disc) use plain Gauss-Legendre over
    the whole chord range; the rest use the mode-centred window (golden-section + bisection)."""
    out = np.empty(len(mx))
    span = 2.0 * R * (np.abs(mx) / (s1 * s1) + np.abs(my) / (s2 * s2)) + R * R / (2.0 * s2 * s2)
    easy = span < EASY_NATS
    for mask, fn in ((easy, _ln_pc_block_easy), (~easy, _ln_pc_block)):
        idx = np.flatnonzero(mask)
        for s in range(0, len(idx), CHUNK):
            j = idx[s:s + CHUNK]
            out[j] = fn(mx[j, None], my[j, None], s1[j, None], s2[j, None], R[j, None])
    return out


def _gl_window(mx, my, s1, s2, R, pa, pb, t, w):
    """ln of the Gauss-Legendre sum over phi in [pa, pb] (x = R sin phi), log-sum-exp."""
    half = 0.5 * (pb - pa)
    phi = 0.5 * (pa + pb) + half * t[None, :]
    with np.errstate(divide="ignore"):
        terms = (_ln_marginal(R * np.sin(phi), mx, my, s1, s2, R) + np.log(R * np.cos(phi))
                 + np.log(w[None, :]) + np.log(np.maximum(half, 1e-300)))
    m = np.max(terms, axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        lse = m[:, 0] + np.log(np.sum(np.exp(terms - m), axis=1))
    return lse - np.log(s1[:, 0] * np.sqrt(2.0 * np.pi))


def _ln_pc_block_easy(mx, my, s1, s2, R):
    half_pi = np.full_like(R, np.pi / 2)
    return _gl_window(mx, my, s1, s2, R, -half_pi, half_pi, _GL_T_EASY, _GL_W_EASY)


def _ln_pc_block(mx, my, s1, s2, R):
    f = lambda x: _ln_marginal(x, mx, my, s1, s2, R)  # noqa: E731
    # 1) mode of the log-concave marginal on (-R, R): golden-section search
    lo, hi = -R.copy(), R.copy()
    c = hi - _GOLD * (hi - lo)
    d = lo + _GOLD * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(GOLDEN_ITERS):
        left = fc > fd
        hi = np.where(left, d, hi)
        lo = np.where(left, lo, c)
        xn = np.where(left, hi - _GOLD * (hi - lo), lo + _GOLD * (hi - lo))
        fn = f(xn)
        d, fd, c, fc = (np.where(left, c, xn), np.where(left, fc, fn),
                        np.where(left, xn, d), np.where(left, fn, fd))
    xs = np.where(fc > fd, c, d)
    thr = np.maximum(fc, fd) - WINDOW_NATS
    # 2) window edges where the marginal drops by WINDOW_NATS (bisection on each side)
    a_lo, a_hi = -R.copy(), xs.copy()
    b_lo, b_hi = xs.copy(), R.copy()
    for _ in range(BISECT_ITERS):
        am = 0.5 * (a_lo + a_hi)
        below = f(am) < thr
        a_lo = np.where(below, am, a_lo)
        a_hi = np.where(below, a_hi, am)
        bm = 0.5 * (b_lo + b_hi)
        below = f(bm) < thr
        b_hi = np.where(below, bm, b_hi)
        b_lo = np.where(below, b_lo, bm)
    pa = np.arcsin(np.clip(a_lo / R, -1, 1))
    pb = np.arcsin(np.clip(b_hi / R, -1, 1))
    # 3) Gauss-Legendre in phi on the window: x = R sin(phi), dx = R cos(phi) dphi
    return _gl_window(mx, my, s1, s2, R, pa, pb, _GL_T, _GL_W)


def ln_pc_2d(x2, C2, hbr, scale=1.0):
    """Natural-log 2D Pc (unfloored). x2 (n,2) miss in plane, C2 (n,2,2), hbr (n,) or scalar.
    `scale` multiplies the covariance. Rows with non-PD / NaN covariance return NaN."""
    x2 = np.atleast_2d(np.asarray(x2, dtype=float))
    C2 = np.asarray(C2, dtype=float).reshape(-1, 2, 2)
    n = len(x2)
    R = np.broadcast_to(np.asarray(hbr, dtype=float), (n,)).astype(float)
    k = np.broadcast_to(np.asarray(scale, dtype=float), (n,)).astype(float)
    mx, my, s1, s2 = _principal(x2, C2)
    s1, s2 = s1 * np.sqrt(k), s2 * np.sqrt(k)
    ok = np.isfinite(mx) & np.isfinite(my) & np.isfinite(s1) & np.isfinite(s2) & (s2 > 0) & (R > 0)
    out = np.full(n, np.nan)
    if ok.any():
        out[ok] = _ln_pc_principal(mx[ok], my[ok], s1[ok], s2[ok], R[ok])
    out[(R <= 0) & np.isfinite(s2)] = -np.inf
    return np.minimum(out, 0.0)


def log10_pc_2d(x2, C2, hbr, scale=1.0, floor=LOG10_PC_FLOOR):
    """log10 2D Pc, floored at `floor` (ESA convention -30)."""
    lp = ln_pc_2d(x2, C2, hbr, scale) / LN10
    return np.where(np.isnan(lp), np.nan, np.maximum(lp, floor))


def esa_max_scale(x2, C2):
    """ESA's max_risk_scaling: the small-HBR analytic optimum k0 = m_2D^2 / 2 of the covariance scale."""
    m = mahalanobis_2d(np.atleast_2d(np.asarray(x2, float)), np.asarray(C2, float).reshape(-1, 2, 2))
    return np.maximum(m * m / 2.0, 1e-30)


def log10_pc_max_esa_2d(x2, C2, hbr, floor=LOG10_PC_FLOOR):
    """ESA's max_risk_estimate definition: log10 Pc with C2 scaled by k0 = m_2D^2 / 2, and k0.

    Equal to the exact maximum over k for HBR << sigma; below it otherwise (see log10_pc_max_2d)."""
    k0 = esa_max_scale(x2, C2)
    return log10_pc_2d(x2, C2, hbr, scale=k0, floor=floor), k0


def log10_pc_max_2d(x2, C2, hbr, floor=LOG10_PC_FLOOR, lp_k0=None):
    """Exact maximum log10 Pc over covariance scale k (C2 -> k C2), and the maximising k.

    Starts from the small-HBR analytic optimum k0 = m_2D^2 / 2 (ESA's max_risk_scaling; pass
    lp_k0 = unfloored log10 Pc at k0 if already known); refined by golden-section in ln k where
    HBR^2 / (k0 sigma_minor^2) > MAXPC_REFINE_RATIO (the analytic optimum is then not exact).
    """
    x2 = np.atleast_2d(np.asarray(x2, dtype=float))
    C2 = np.asarray(C2, dtype=float).reshape(-1, 2, 2)
    n = len(x2)
    R = np.broadcast_to(np.asarray(hbr, dtype=float), (n,)).astype(float)
    k0 = esa_max_scale(x2, C2)
    lnp = ln_pc_2d(x2, C2, R, k0) if lp_k0 is None else np.asarray(lp_k0, dtype=float) * LN10
    kbest = k0.copy()
    _, _, _, s2 = _principal(x2, C2)
    ref = np.isfinite(lnp) & (R * R / (k0 * s2 * s2) > MAXPC_REFINE_RATIO)
    if ref.any():
        xr, Cr, Rr = x2[ref], C2[ref], R[ref]
        lo = np.log(k0[ref]) - 4.0
        hi = np.log(k0[ref]) + 4.0
        g = lambda u: ln_pc_2d(xr, Cr, Rr, np.exp(u))  # noqa: E731
        c = hi - _GOLD * (hi - lo)
        d = lo + _GOLD * (hi - lo)
        fc, fd = g(c), g(d)
        for _ in range(MAXPC_ITERS):
            left = fc > fd
            hi = np.where(left, d, hi)
            lo = np.where(left, lo, c)
            un = np.where(left, hi - _GOLD * (hi - lo), lo + _GOLD * (hi - lo))
            fn = g(un)
            d, fd, c, fc = (np.where(left, c, un), np.where(left, fc, fn),
                            np.where(left, un, d), np.where(left, fn, fd))
        best = np.maximum(fc, fd)
        ub = np.where(fc > fd, c, d)
        better = best > lnp[ref]
        lnp_r, k_r = lnp[ref], kbest[ref]
        lnp_r[better] = best[better]
        k_r[better] = np.exp(ub[better])
        lnp[ref], kbest[ref] = lnp_r, k_r
    lp = lnp / LN10
    return np.where(np.isnan(lp), np.nan, np.maximum(lp, floor)), kbest


IMPLIED_HBR_RANGE = (0.05, 100.0)   # m; implied HBR outside this -> fall back to the span formula
IMPLIED_HBR_ITERS = 40
IMPLIED_HBR_TOL = 1e-7               # decades
IMPLIED_HBR_LINEAR = 1e-3            # HBR^2 / (k0 sigma_minor^2) below this: Pc(k0) ~ HBR^2 exactly


def implied_hbr(x2, C2, max_risk_estimate, hbr0):
    """HBR (m) for which ESA's max-Pc definition reproduces max_risk_estimate (ESA's own radius).

    ESA's max_risk_estimate is log10 Pc(k0 C2; HBR) with k0 = m_2D^2 / 2 fixed by the geometry (see
    the module docstring), so the equation log10 Pc(k0 C2; R) = mre is monotone in R. Small-HBR
    regime: Pc is proportional to R^2, so one step R = R0 10^((mre - lp(R0)) / 2) is exact; rows
    outside it (R^2 / (k0 sigma_minor^2) > IMPLIED_HBR_LINEAR) are refined by bracketed secant steps
    in ln R (bisection fallback) to IMPLIED_HBR_TOL.
    CALIBRATION NOTE: the result is fitted to an ESA output; Pc computed with it agrees with ESA's
    risk by construction to the extent that ESA used one radius for both columns.
    Returns (hbr, converged_mask, log10_pc_at_k0_with_hbr (unfloored), k0)."""
    x2 = np.atleast_2d(np.asarray(x2, float))
    C2 = np.asarray(C2, float).reshape(-1, 2, 2)
    n = len(x2)
    mre = np.broadcast_to(np.asarray(max_risk_estimate, float), (n,))
    R0 = np.broadcast_to(np.asarray(hbr0, float), (n,)).astype(float)
    lp0, k0 = log10_pc_max_esa_2d(x2, C2, R0, floor=-np.inf)
    with np.errstate(invalid="ignore", over="ignore"):
        R = R0 * 10 ** ((mre - lp0) / 2.0)
        lpm = lp0 + 2.0 * np.log10(R / R0)
    _, _, _, s2 = _principal(x2, C2)
    conv = np.isfinite(R) & np.isfinite(lpm)
    with np.errstate(invalid="ignore"):
        nonlin = conv & (np.maximum(R, R0) ** 2 / (k0 * s2 * s2) > IMPLIED_HBR_LINEAR)
    if nonlin.any():
        idx = np.flatnonzero(nonlin)
        xa, Ca, ma, ka = x2[idx], C2[idx], mre[idx], k0[idx]
        lo = np.full(len(idx), np.log(1e-4))
        hi = np.full(len(idx), np.log(1e4))
        u0, g0 = np.log(R0[idx]), lp0[idx] - ma
        lo = np.where(g0 < 0, np.maximum(lo, u0), lo)
        hi = np.where(g0 > 0, np.minimum(hi, u0), hi)
        u = np.clip(np.log(R[idx]), lo, hi)
        u_prev, g_prev = u0, g0
        g = np.full(len(idx), np.nan)
        act = np.arange(len(idx))
        for _ in range(IMPLIED_HBR_ITERS):
            g[act] = log10_pc_2d(xa[act], Ca[act], np.exp(u[act]), scale=ka[act], floor=-np.inf) - ma[act]
            lo[act] = np.where(g[act] < 0, np.maximum(lo[act], u[act]), lo[act])
            hi[act] = np.where(g[act] > 0, np.minimum(hi[act], u[act]), hi[act])
            act = act[~(np.abs(g[act]) < IMPLIED_HBR_TOL)]
            if act.size == 0:
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = (g[act] - g_prev[act]) / (u[act] - u_prev[act])
            slope = np.where(np.isfinite(slope) & (slope > 1e-6), slope, 2.0 / LN10)
            un = u[act] - g[act] / slope
            un = np.where((un > lo[act]) & (un < hi[act]), un, 0.5 * (lo[act] + hi[act]))
            u_prev[act], g_prev[act] = u[act], g[act]
            u[act] = un
        else:                                    # iterations exhausted: evaluate the last step
            g[act] = log10_pc_2d(xa[act], Ca[act], np.exp(u[act]), scale=ka[act], floor=-np.inf) - ma[act]
        R[idx] = np.exp(u)
        lpm[idx] = g + ma
        conv[idx] = np.abs(g) < 1e-4
    return R, conv, lpm, k0


# ---------------------------------------------------------------- full pipeline
def full_pc(df, frame="chaser_rtn", hbr="auto", with_max=True):
    """Physics Pc for every CDM row of `df` (vectorised).

    hbr: "auto" (default) = HBR implied by max_risk_estimate where that column is present and the
         implied value is plausible, else the span formula (CALIBRATED on an ESA output: label
         results accordingly); "spans" = (t_span + c_span) / 2 for all rows (inputs only);
         "implied" = same as auto; or a scalar / array of radii in metres.

    Returns a DataFrame indexed like `df` with columns:
      log_pc_phys      log10 2D Pc, floored at -30 (NaN when not computable)
      log_pc_max_phys  EXACT max over covariance scale k of log10 Pc (floored at -30)
      hbr_m            hard-body radius used (m)
      pc_valid         False for placeholder (>= 6e7 m) / missing covariance, a combined 3D or
                       projected 2D covariance that is not positive definite, or zero relative
                       velocity. Values are still computed for placeholder rows (ESA does the
                       same) but are not physically meaningful; callers should fall back to the
                       miss-distance rule / ML estimate when pc_valid is False.
      mahal_3d         sqrt(d^T C^-1 d) with the 3D combined covariance (= ESA mahalanobis_distance)
      mahal_2d         Mahalanobis distance in the encounter plane
      miss_plane_x/y   miss vector in the encounter plane (x ~ radial, y = v_hat x x) (m)
      sigma_major/minor  principal 1-sigma of the projected combined covariance (m)
    Extra columns:
      max_scale        exact argmax k of the max-Pc search
      log_pc_max_esa   log10 Pc at k = m_2D^2 / 2 = ESA's max_risk_estimate definition (with the
                       implied HBR this reproduces max_risk_estimate by construction)
      max_scale_esa    m_2D^2 / 2 = ESA's max_risk_scaling definition
      hbr_spans_m      (t_span + c_span) / 2
      hbr_source       "implied" (calibrated on max_risk_estimate) | "spans" | "given"
    with_max=False skips log_pc_max_phys / max_scale (NaN) and log_pc_max_esa except where the
    implied-HBR solver already produced it.
    """
    n = len(df)
    placeholder, missing = covariance_status(df)
    p = df[POS_COLS].to_numpy(float)
    v = df[VEL_COLS].to_numpy(float)
    R_spans = hbr_from_spans(df["t_span"], df["c_span"])
    if isinstance(hbr, str):
        if hbr not in ("auto", "implied", "spans"):
            raise ValueError(f"unknown hbr mode {hbr!r}")
        R = R_spans.copy()
        source = np.full(n, "spans", dtype=object)
    else:
        R = np.broadcast_to(np.asarray(hbr, dtype=float), (n,)).astype(float)
        source = np.full(n, "given", dtype=object)
    C = combined_covariance(df, frame=frame)
    C[missing] = np.nan
    good = ~missing & (np.linalg.norm(v, axis=1) > 0)
    x2 = np.full((n, 2), np.nan)
    C2 = np.full((n, 2, 2), np.nan)
    m3 = np.full(n, np.nan)
    pd3 = np.zeros(n, dtype=bool)
    if good.any():
        x2[good], C2[good], _ = encounter_plane(p[good], v[good], C[good])
        with np.errstate(invalid="ignore"):
            m3[good] = mahalanobis_3d(p[good], C[good])
        pd3[good] = np.linalg.eigvalsh(C[good])[:, 0] > 0
    _, _, s1, s2 = _principal(x2, C2)
    pd_ok = np.isfinite(s2) & (s2 > 0)
    lp = np.full(n, np.nan)
    lpmax = np.full(n, np.nan)
    kmax = np.full(n, np.nan)
    lpesa = np.full(n, np.nan)                 # unfloored log10 Pc at k0 = m_2D^2 / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        k0 = np.where(pd_ok, esa_max_scale(x2, C2), np.nan)
    if isinstance(hbr, str) and hbr in ("auto", "implied") and "max_risk_estimate" in df:
        mre = df["max_risk_estimate"].to_numpy(float)
        cand = pd_ok & np.isfinite(mre) & np.isfinite(R_spans) & (R_spans > 0)
        if cand.any():
            Ri, conv, lpe_i, _ = implied_hbr(x2[cand], C2[cand], mre[cand], R_spans[cand])
            okr = conv & (Ri >= IMPLIED_HBR_RANGE[0]) & (Ri <= IMPLIED_HBR_RANGE[1])
            idx = np.flatnonzero(cand)[okr]
            R[idx] = Ri[okr]
            source[idx] = "implied"
            lpesa[idx] = lpe_i[okr]
    if pd_ok.any():
        lp[pd_ok] = log10_pc_2d(x2[pd_ok], C2[pd_ok], R[pd_ok])
        if with_max:
            todo = pd_ok & np.isnan(lpesa)
            if todo.any():
                lpesa[todo] = log10_pc_2d(x2[todo], C2[todo], R[todo], scale=k0[todo], floor=-np.inf)
            lpmax[pd_ok], kmax[pd_ok] = log10_pc_max_2d(x2[pd_ok], C2[pd_ok], R[pd_ok], lp_k0=lpesa[pd_ok])
    out = pd.DataFrame({
        "log_pc_phys": lp,
        "log_pc_max_phys": lpmax,
        "hbr_m": R,
        "pc_valid": pd_ok & pd3 & ~placeholder & ~missing,
        "mahal_3d": m3,
        "mahal_2d": mahalanobis_2d(x2, C2),
        "miss_plane_x": x2[:, 0],
        "miss_plane_y": x2[:, 1],
        "sigma_major": s1,
        "sigma_minor": s2,
        "max_scale": kmax,
        "log_pc_max_esa": np.where(np.isnan(lpesa), np.nan, np.maximum(lpesa, LOG10_PC_FLOOR)),
        "max_scale_esa": k0,
        "hbr_spans_m": R_spans,
        "hbr_source": source,
    }, index=df.index)
    return out


# ---------------------------------------------------------------- manoeuvre advisory
EXECUTION_ERROR = 0.10          # the recommended burn must still work if its magnitude is off by +-10 %
MODEL_MARGIN_DECADES = 0.05     # extra Pc margin for unmodelled effects (J2, covariance model; each
                                # measured <= 0.015 decades by scripts/verify_manoeuvre.py)
PLANNING_BUFFER_S = 3600.0      # the burn must be at least this long after the CDM epoch
ECC_SENSITIVE_RATIO = 1.0 / 3.0  # flag options whose eccentricity radial uncertainty > sigma_minor / 3
_REFINE_ROUNDS, _REFINE_POINTS = 4, 32

MANOEUVRE_ASSUMPTIONS = (
    "Impulsive burn of the TARGET along its along-track (T) axis; dv > 0 = posigrade.",
    "Clohessy-Wiltshire displacement on a circular reference orbit (mean motion from t_j2k_sma), read "
    "curvilinearly (radius difference, arc length) in the local RTN frame at the new TCA.",
    "Covariance, relative velocity and HBR unchanged by the burn (covariances stay attached to each "
    "object's local RTN frame); the TCA shift is handled by projecting onto the encounter plane.",
    "Target eccentricity: its true anomaly is not in the CDM, so an along-track shift dT moves the "
    "target by up to +-e|dT| radially; robust mode takes the worst case over that interval.",
    "Execution / model scale error: robust mode requires the Pc target for burn magnitudes within "
    "+-execution_error of the recommendation (this envelope also covers the CW scale errors from "
    "eccentricity, <= 3.8 %, and J2, <= 0.45 %), plus model_margin_decades of Pc.",
    "J2 is not modelled (<= 0.012 decades of Pc measured end-to-end).",
    "Sign convention: relative_position / relative_velocity = chaser - target (CCSDS: object 2 - "
    "object 1). Verified for the velocity (energy relation); the position sign cannot be verified "
    "from the dataset (Pc, Mahalanobis and miss distance are sign-blind). If it were reversed, the "
    "posigrade / retrograde choice would flip - both directions are always returned under 'options'.",
    "No check of new conjunctions with other objects or of the post-burn orbit.",
)


def mean_motion(t_sma_km):
    """Mean motion n (rad/s) of a circular orbit with semi-major axis in km."""
    a = np.asarray(t_sma_km, dtype=float) * 1e3
    return np.sqrt(MU_EARTH / a ** 3)


def cw_displacement(dv, tau, n):
    """Clohessy-Wiltshire displacement (dR, dT, dN) in metres at time tau (s) after an impulsive
    along-track burn dv (m/s, + = posigrade) on a circular orbit with mean motion n (rad/s).

    The result must be read CURVILINEARLY: dR = radius difference, dT = arc length along the orbit,
    expressed in the local RTN frame at the (new) TCA - the frame the CDM relative state and
    covariances are attached to. Do not replace it by a propagated Cartesian displacement measured in
    the frozen original TCA frame: that differs by the frame-curvature term -dT^2 / (2a) (-8 m at a
    10.6 km shift) and, combined with the unchanged relative velocity and covariance, gives a wrong Pc
    (test row 9934, 2 orbits posigrade: -5.55 instead of -4.99 end-to-end). Any numerical-propagation
    upgrade must express displacement, relative velocity and covariance in the local RTN frame at
    the new TCA (see e2e_after_burn in scripts/verify_manoeuvre.py)."""
    dv, tau, n = np.broadcast_arrays(np.asarray(dv, float), np.asarray(tau, float), np.asarray(n, float))
    nt = n * tau
    dR = 2.0 * dv / n * (1.0 - np.cos(nt))
    dT = dv / n * (4.0 * np.sin(nt) - 3.0 * nt)
    return np.stack([dR, dT, np.zeros_like(dR)], axis=-1)


def _row_frame(row):
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError("manoeuvre_options expects a single CDM row")
        r = row.reset_index(drop=True)
    elif isinstance(row, pd.Series):
        r = row.to_frame().T.reset_index(drop=True)
    else:
        r = pd.DataFrame([dict(row)])
    for c in r.columns:              # a Series row of mixed dtypes arrives as object columns
        if r[c].dtype == object:
            try:
                r[c] = pd.to_numeric(r[c])
            except (ValueError, TypeError):
                pass
    return r


def _scalar(r, col):
    if col not in r.columns:
        return np.nan
    try:
        return float(r[col].iloc[0])
    except (TypeError, ValueError):
        return np.nan


class _BurnModel:
    """Pc after an along-track burn for one CDM, nominal and worst case over the uncertainty set
        s in [1 - eps, 1 + eps] (burn magnitude factor), |delta| <= e |dT(s dv)| (radial offset).
    log Pc is concave in the miss vector (Gaussian convolved with a disc: Prekopa), and the miss is
    affine in (s, delta) over a convex (trapezoidal) set, so the worst case lies at the
    unconstrained optimum or on an edge. Candidates: the optimum of the Gaussian (small-HBR)
    approximation on the interior and on each edge, plus the four vertices and the three delta = 0
    points (nominal, short, long)."""

    def __init__(self, p, B, C2, R, n, ecc, eps):
        self.p, self.B, self.C2, self.R, self.n = p, B, C2, R, n
        self.ecc, self.eps = ecc, eps
        self.x = B @ p                               # nominal miss in the encounter plane
        self.u = B @ np.array([1.0, 0.0, 0.0])       # target radial direction, projected
        self.A = np.linalg.inv(C2)

    def _pc(self, mu):
        shp = mu.shape[:-1]
        flat = mu.reshape(-1, 2)
        lp = log10_pc_2d(flat, np.broadcast_to(self.C2, (len(flat), 2, 2)), self.R, floor=-np.inf)
        return lp.reshape(shp)

    def evaluate(self, mags, sign, tau, robust=True):
        """dict of arrays over burn magnitudes `mags` (m/s): nominal, short, long, ecc_worst, worst."""
        mags = np.atleast_1d(np.asarray(mags, float))
        D = cw_displacement(sign, tau, self.n)                    # displacement per m/s of |dv|
        w = mags[:, None] * (self.B @ D)[None, :]                 # (m, 2) planar shift, nominal burn
        x, u, A = self.x, self.u, self.A
        if not robust:
            lp = self._pc(x[None, :] - w)
            return {"nominal": lp, "short": lp, "long": lp, "ecc_worst": lp, "worst": lp}
        lo, hi = 1.0 - self.eps, 1.0 + self.eps
        c = self.ecc * np.abs(mags * D[1])                        # radial half-width per unit s
        Ax, Au = A @ x, A @ u
        uAu = u @ Au
        wAx, wAu = w @ Ax, w @ Au
        wAw = np.einsum("mi,ij,mj->m", w, A, w)

        def d_star(s, half):                                      # best delta on the edge s = const
            return np.clip((u @ Ax - s * wAu) / uAu, -half, half)

        def s_star(sgn):                                          # best s on the edge delta = sgn c s
            bAx = wAx + sgn * c * (u @ Ax)
            bAb = wAw + 2.0 * sgn * c * wAu + c * c * uAu
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(bAb > 0, bAx / bAb, 1.0)
            return np.clip(s, lo, hi)

        det = wAw * uAu - wAu ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            si = (wAx * uAu - wAu * (u @ Ax)) / det
            di = (wAw * (u @ Ax) - wAu * wAx) / det
            inside = ((det > 1e-12 * np.maximum(wAw * uAu, 1e-300)) & (si >= lo) & (si <= hi)
                      & (np.abs(di) <= c * si))
        si, di = np.where(inside, si, 1.0), np.where(inside, di, 0.0)
        sp, sm = s_star(1.0), s_star(-1.0)
        one = np.ones_like(mags)
        S = np.stack([one, lo * one, hi * one,                                  # 0-2: delta = 0
                      one, one, one,                                            # 3-5: s = 1, ecc
                      lo * one, lo * one, hi * one, hi * one,                   # 6-9: vertices
                      lo * one, hi * one, sp, sm, si], axis=1)                  # 10-14: edges, interior
        Dl = np.stack([0 * one, 0 * one, 0 * one,
                       c, -c, d_star(1.0, c),
                       c * lo, -c * lo, c * hi, -c * hi,
                       d_star(lo, c * lo), d_star(hi, c * hi), c * sp, -c * sm, di], axis=1)
        mu = x[None, None, :] - S[..., None] * w[:, None, :] - Dl[..., None] * u[None, None, :]
        lp = self._pc(mu)
        return {"nominal": lp[:, 0], "short": lp[:, 1], "long": lp[:, 2],
                "ecc_worst": lp[:, [0, 3, 4, 5]].max(axis=1), "worst": lp.max(axis=1)}


def _first_crossing(fun, grid, fgrid, target):
    """Smallest grid magnitude with fun(m) < target, refined between grid points (vectorised
    sub-grids). fun(m) need not be monotone; the first crossing is returned. None if none."""
    hit = np.flatnonzero(fgrid < target)
    if hit.size == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return 0.0
    a, b = grid[i - 1], grid[i]
    for _ in range(_REFINE_ROUNDS):
        sub = np.linspace(a, b, _REFINE_POINTS + 1)[1:]
        j = np.flatnonzero(fun(sub) < target)
        j = int(j[0]) if j.size else len(sub) - 1
        a, b = (sub[j - 1] if j > 0 else a), sub[j]
    return float(b)


def _effect_text(d):
    dR, dT = d[0], d[1]
    rad = (f"{abs(dR):.1f} m {'higher' if dR > 0 else 'lower'}" if abs(dR) >= 0.05 else "no radial change")
    return f"target {rad}, {abs(dT):.0f} m {'ahead' if dT > 0 else 'behind'} at TCA"


def manoeuvre_options(row, target_log_pc=-5.0, burn_lead_orbits=(0.5, 1, 2, 3), max_dv=0.5,
                      n_grid=400, frame="chaser_rtn", hbr="auto", robust=True,
                      execution_error=EXECUTION_ERROR, model_margin_decades=MODEL_MARGIN_DECADES,
                      eccentricity="auto", time_to_tca_days=None, planning_buffer_s=PLANNING_BUFFER_S):
    """Minimum along-track impulsive burn of the TARGET that brings 2D Pc below 10**target_log_pc.

    For each lead time tau = lead_orbits * 2 pi / n before TCA, the CW displacement of the target at
    TCA (cw_displacement, read curvilinearly) is subtracted from the relative position (chaser -
    target); the miss vector is re-projected onto the (unchanged) encounter plane and Pc re-evaluated
    with the unchanged covariance and HBR. Posigrade (+dv) and retrograde (-dv) burns are both
    scanned on a grid of n_grid points up to max_dv (m/s); the first crossing is refined (Pc is not
    monotone in dv when the burn first moves the target towards the chaser).

    robust=True (default): the burn is sized so that the WORST Pc over
      - burn magnitude errors of +-execution_error (default 10 %; also covers the CW scale errors
        from eccentricity and J2), and
      - an unknown radial offset of up to +-e|dT| (e = t_j2k_ecc, or `eccentricity`; the target's
        true anomaly is not in the CDM, so on an eccentric orbit the shifted target can sit up to
        e|dT| higher or lower than CW predicts - measured up to +0.54 decades of Pc for an 11.5 km
        shift at e = 0.001)
    stays below target_log_pc - model_margin_decades (default 0.05 decade for J2 / covariance-model
    effects). robust=False reproduces the nominal minimum (Pc exactly on the target, no margin); its
    log_pc_after_short / _long / _ecc_worst / log_pc_worst still report that burn's exposure.
    Options whose radial uncertainty e|dT| exceeds sigma_minor / 3 are flagged
    eccentricity_sensitive; prefer half-orbit leads or small along-track shifts for those.

    Lead-time feasibility: when time_to_tca (days; from `time_to_tca_days` or the row's
    time_to_tca column) is known, leads longer than time_to_tca - planning_buffer_s are returned as
    {"feasible": False, "reason": ...} entries without a burn.

    Assumptions: see MANOEUVRE_ASSUMPTIONS (impulsive along-track burn; circular CW; covariance,
    relative velocity and HBR unchanged; J2 not modelled; chaser - target sign convention; no check
    for new conjunctions). Along-track burns an integer number of orbits before TCA give no CW radial
    offset, so for head-on geometries only the half-orbit leads are effective. `pc_valid` (False for
    placeholder or degenerate covariance) is passed through: such advice is not meaningful.

    Returns a list aligned with burn_lead_orbits. Each entry is
      None when neither direction meets the (robust) target within max_dv, else a dict:
      lead_orbits, lead_time_s, lead_time_h, feasible, reason, time_to_tca_s, log_pc_before,
      miss_distance_before_m, target_log_pc, design_log_pc (target - margin when robust), robust,
      execution_error, model_margin_decades, eccentricity, eccentricity_source, hbr_m, pc_valid,
      sigma_minor_m, and the best direction's fields:
        dv_mps (signed), abs_dv_mps, direction, log_pc_after (nominal burn), log_pc_after_short /
        log_pc_after_long (burn x (1 -/+ execution_error)), log_pc_after_ecc_worst (nominal burn,
        worst radial offset), log_pc_worst (whole uncertainty set), abs_dv_nominal_mps (non-robust
        minimum, for reference), miss_distance_after_m (encounter-plane miss), d_target_rtn_m
        [dR, dT, dN] (target displacement at TCA), effect (plain-language description of
        d_target_rtn_m), ecc_radial_uncertainty_m (e |dT|), eccentricity_sensitive
      options {posigrade: dict|None, retrograde: dict|None} with the same per-direction fields.
      Infeasible leads have feasible=False, a reason, dv_mps / abs_dv_mps / direction None and
      both options None.
    """
    r = _row_frame(row)
    p = r[POS_COLS].to_numpy(float)[0]
    v = r[VEL_COLS].to_numpy(float)
    C = combined_covariance(r, frame=frame)
    fp = full_pc(r, frame=frame, hbr=hbr, with_max=False).iloc[0]
    R, pc_valid = float(fp["hbr_m"]), bool(fp["pc_valid"])
    s_minor = float(fp["sigma_minor"])
    n = float(mean_motion(r["t_j2k_sma"].to_numpy(float))[0])
    _, C2, B = encounter_plane(p[None, :], v, C)
    B, C2 = B[0], C2[0]
    period = 2.0 * np.pi / n

    if isinstance(eccentricity, str):
        e = _scalar(r, "t_j2k_ecc")
        ecc_src = "t_j2k_ecc" if np.isfinite(e) else "missing (0 assumed)"
        e = e if np.isfinite(e) else 0.0
    else:
        e, ecc_src = float(eccentricity), "given"
    eps = float(execution_error)
    design = float(target_log_pc) - (float(model_margin_decades) if robust else 0.0)
    ttca_d = float(time_to_tca_days) if time_to_tca_days is not None else _scalar(r, "time_to_tca")
    ttca_s = ttca_d * 86400.0 if np.isfinite(ttca_d) else None

    model = _BurnModel(p, B, C2, R, n, e, eps)
    lp0 = float(model.evaluate(np.array([0.0]), 1.0, 0.0, robust=False)["nominal"][0])
    md0 = float(np.linalg.norm(model.x))
    fl = lambda x: float(max(x, LOG10_PC_FLOOR))  # noqa: E731
    common = {"log_pc_before": fl(lp0), "miss_distance_before_m": md0,
              "target_log_pc": float(target_log_pc), "design_log_pc": design, "robust": bool(robust),
              "execution_error": eps, "model_margin_decades": float(model_margin_decades) if robust else 0.0,
              "eccentricity": e, "eccentricity_source": ecc_src, "hbr_m": R, "pc_valid": pc_valid,
              "sigma_minor_m": s_minor, "time_to_tca_s": ttca_s}
    grid = np.linspace(0.0, max_dv, n_grid + 1)
    results = []
    for L in burn_lead_orbits:
        tau = float(L) * period
        head = {"lead_orbits": float(L), "lead_time_s": tau, "lead_time_h": tau / 3600.0}
        if ttca_s is not None and tau > ttca_s - planning_buffer_s:
            results.append({**head, "feasible": False,
                            "reason": (f"lead time {tau / 3600:.2f} h exceeds time to TCA "
                                       f"{ttca_s / 3600:.2f} h minus the {planning_buffer_s / 3600:.2f} h planning buffer"),
                            **common, "dv_mps": None, "abs_dv_mps": None, "direction": None,
                            "options": {"posigrade": None, "retrograde": None}})
            continue

        def opt(sign):
            f_nom = model.evaluate(grid, sign, tau, False)["nominal"]
            nominal = lambda s: model.evaluate(s, sign, tau, False)["nominal"]  # noqa: E731
            m_nom = _first_crossing(nominal, grid, f_nom, float(target_log_pc))
            if not robust:
                m = m_nom
            else:
                # worst >= nominal, so the robust crossing cannot precede the first grid point where
                # the nominal Pc is below the design level: scan the robust objective from there
                worst = lambda s: model.evaluate(s, sign, tau, True)["worst"]  # noqa: E731
                below = np.flatnonzero(f_nom < design)
                m = None
                if below.size:
                    f_rob = np.full_like(f_nom, np.inf)
                    for s0 in range(int(below[0]), len(grid), 32):
                        sl = slice(s0, min(s0 + 32, len(grid)))
                        f_rob[sl] = worst(grid[sl])
                        if (f_rob[sl] < design).any():
                            break
                    m = _first_crossing(worst, grid, f_rob, design)
            if m is None:
                return None
            at = model.evaluate(np.array([m]), sign, tau, True)     # report the exposure in both modes
            disp = cw_displacement(sign * m, tau, n)
            d_unc = e * abs(float(disp[1]))
            return {"dv_mps": float(sign * m), "abs_dv_mps": float(m),
                    "direction": "posigrade" if sign > 0 else "retrograde",
                    "log_pc_after": fl(at["nominal"][0]), "log_pc_after_short": fl(at["short"][0]),
                    "log_pc_after_long": fl(at["long"][0]), "log_pc_after_ecc_worst": fl(at["ecc_worst"][0]),
                    "log_pc_worst": fl(at["worst"][0]),
                    "abs_dv_nominal_mps": None if m_nom is None else float(m_nom),
                    "miss_distance_after_m": float(np.linalg.norm(B @ (p - disp))),
                    "d_target_rtn_m": [float(x) for x in disp], "effect": _effect_text(disp),
                    "ecc_radial_uncertainty_m": d_unc,
                    "eccentricity_sensitive": bool(d_unc > ECC_SENSITIVE_RATIO * s_minor)}

        options = {"posigrade": opt(1.0), "retrograde": opt(-1.0)}
        cands = [o for o in options.values() if o is not None]
        if not cands:
            results.append(None)
            continue
        best = min(cands, key=lambda o: (o["abs_dv_mps"], o["ecc_radial_uncertainty_m"]))
        entry = {**head, "feasible": True, "reason": None, **common, **best,
                 "direction": best["direction"] if best["abs_dv_mps"] > 0 else "none (already below target)",
                 "options": options}
        results.append(entry)
    return results


def best_manoeuvre(options):
    """The recommended entry of a manoeuvre_options() result: the feasible lead with the smallest
    |dv| (ties: smaller eccentricity radial uncertainty, then shorter lead). None if there is none."""
    ok = [o for o in options if o is not None and o.get("feasible", True) and o.get("abs_dv_mps") is not None]
    if not ok:
        return None
    return min(ok, key=lambda o: (o["abs_dv_mps"], o["ecc_radial_uncertainty_m"], o["lead_time_s"]))
