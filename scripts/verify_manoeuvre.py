"""Independent, skeptical verification of the manoeuvre model in src/physics.py.

Nothing from src/physics.py is reused except the function under test, manoeuvre_options(), which is
only CALLED; every number it returns is recomputed here with separately written code (covariance
assembly, chaser-frame rotation, encounter plane, Pc integration, orbit propagation).

Part A - Clohessy-Wiltshire (CW) vs numerical orbit propagation
    Near-circular LEO orbit (a = 7050 km, the typical t_j2k_sma), impulsive along-track burn dv at
    lead times of 0.5 / 1 / 2 / 3 orbits before TCA, dv = 0.01 / 0.05 / 0.2 m/s. The burned and the
    unburned orbit are propagated with scipy DOP853 (the deviation is integrated as its own exact
    nonlinear ODE, d'' = a(r + d) - a(r), so cm-level displacements are not lost in 7000 km
    positions). The displacement at TCA is measured in the unburned orbit's RTN frame, both as a
    Cartesian vector and in curvilinear (radius difference / arc length) form, and compared with
        dR = (2 dv / n)(1 - cos n tau),   dT = (dv / n)(4 sin n tau - 3 n tau).
    Variants: two-body; two-body + J2; eccentric orbits (e up to the dataset maximum 0.0187).
    Sign checks: posigrade burn -> target ends up behind (dT < 0) for leads >= 0.5 orbit, higher
    (dR > 0) at half an orbit; retrograde is the mirror image.

Part B - Do the recommended burns really reach the requested Pc?
    For 3 high-risk test CDMs, manoeuvre_options(target_log_pc=-5) is called and, for every option
    returned (both directions, all lead times), the post-burn Pc is recomputed independently:
      (1) own covariance / encounter plane / polar Gauss-Legendre disc integration, with the
          displacement the module returned;
      (2) 3D importance-sampled Monte Carlo (rectilinear relative motion, no encounter-plane code);
      (3) displacement from numerical two-body propagation instead of CW (curvilinear form);
      (4) end-to-end nonlinear encounter: target state propagated back by the lead time, burned,
          propagated forward; new TCA found by minimising the true distance to the propagated
          chaser; relative state taken in the burned target's local RTN frame at the new TCA (the
          frame the CDM covariance is attached to); Pc in the plane normal to the NEW relative
          velocity. Variants: circular two-body, circular J2, and the target's own eccentricity
          (t_j2k_ecc) at true anomalies 0/90/180/270 deg at TCA (the phase is not in the CDM);
      plus: minimality of the returned |dv| (own scan), None entries (own scan), sensitivity to a
      10 % under-burn, and HBR sensitivity (own HBR solved from ESA's risk; span formula).

    python scripts/verify_manoeuvre.py [--quick] [--json PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MU = 3.986004418e14        # m^3/s^2
RE = 6378137.0             # m
J2 = 1.08262668e-3
LN10 = np.log(10.0)

POS = ["relative_position_r", "relative_position_t", "relative_position_n"]
VEL = ["relative_velocity_r", "relative_velocity_t", "relative_velocity_n"]

# three high-risk test CDMs with different geometries (index into data/test_data.csv)
CDM_ROWS = {
    14267: "highest test risk (-1.53); mostly head-on, 24% normal component of v_rel; the module's own example",
    9934: "risk -2.60; 453 m miss almost purely cross-track, chaser sigma_T 2.8 km",
    23692: "risk -3.05; slow crossing (3.9 km/s), v_rel mostly NORMAL -> along-track shifts are in-plane",
}


# ================================================================ orbit mechanics (own code)
def accel(r, j2=False):
    x, y, z = r[0], r[1], r[2]
    r2 = x * x + y * y + z * z
    rn = np.sqrt(r2)
    a = -MU * r / (rn * r2)
    if j2:
        k = 1.5 * J2 * MU * RE * RE / (r2 * r2 * rn)
        f = 5.0 * z * z / r2
        a = a + k * np.array([x * (f - 1.0), y * (f - 1.0), z * (f - 3.0)])
    return a


def _rhs(t, y, j2):
    return np.concatenate([y[3:6], accel(y[0:3], j2)])


def _rhs_pair(t, y, j2):
    """Reference orbit (r, v) and the exact nonlinear deviation (d, dd) of a second orbit."""
    a0 = accel(y[0:3], j2)
    a1 = accel(y[0:3] + y[6:9], j2)
    return np.concatenate([y[3:6], a0, y[9:12], a1 - a0])


ATOL_PAIR = np.array([1e-6] * 3 + [1e-9] * 3 + [1e-10] * 3 + [1e-13] * 3)


def prop(y0, t0, t1, j2=False, rtol=1e-12, dense=False):
    return solve_ivp(_rhs, (t0, t1), y0, method="DOP853", rtol=rtol, atol=[1e-6] * 3 + [1e-9] * 3,
                     args=(j2,), dense_output=dense)


def prop_pair(y0, t0, t1, j2=False, rtol=1e-12, dense=False):
    return solve_ivp(_rhs_pair, (t0, t1), y0, method="DOP853", rtol=rtol, atol=ATOL_PAIR * (rtol / 1e-12),
                     args=(j2,), dense_output=dense)


def kepler_to_cart(a, e, inc, raan, argp, nu):
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(MU / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO, ci, si, cw, sw = np.cos(raan), np.sin(raan), np.cos(inc), np.sin(inc), np.cos(argp), np.sin(argp)
    Q = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    return Q @ r_pf, Q @ v_pf


def rtn_matrix(r, v):
    """3x3 matrix whose columns are the R, T, N unit vectors (inertial coordinates)."""
    R = r / np.linalg.norm(r)
    N = np.cross(r, v)
    N /= np.linalg.norm(N)
    return np.column_stack([R, np.cross(N, R), N])


def cw_formula(dv, tau, n):
    """The formula under test (as stated in the task), evaluated with own code."""
    nt = n * tau
    return np.array([2.0 * dv / n * (1.0 - np.cos(nt)), dv / n * (4.0 * np.sin(nt) - 3.0 * nt), 0.0])


def cw_formula_vec(dv, tau, n):
    dv = np.asarray(dv, float)
    nt = n * tau
    return np.stack([2.0 * dv / n * (1.0 - np.cos(nt)), dv / n * (4.0 * np.sin(nt) - 3.0 * nt),
                     np.zeros_like(dv)], axis=-1)


def cw_stm(dv, tau, n):
    """Independent derivation: CW equations x'' = 3n^2 x + 2n y', y'' = -2n x', z'' = -n^2 z via expm."""
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0], A[3, 4], A[4, 3], A[5, 2] = 3 * n * n, 2 * n, -2 * n, -n * n
    return (expm(A * tau) @ np.array([0, 0, 0, 0, dv, 0.0]))[:3]


def burn_displacement(a, dv, tau, j2=False, inc=np.radians(98.0), ecc=0.0, nu_burn=0.0, argp=0.0,
                      rtol=1e-12):
    """Propagate unburned + burned orbit from the burn epoch for tau seconds.

    The burn is along the velocity vector. Returns Cartesian (R, T, N) displacement in the unburned
    orbit's RTN frame at TCA and the curvilinear one (radius difference, arc length, cross-track)."""
    r0, v0 = kepler_to_cart(a, ecc, inc, 0.0, argp, nu_burn)
    y0 = np.concatenate([r0, v0, np.zeros(3), dv * v0 / np.linalg.norm(v0)])
    y = prop_pair(y0, 0.0, tau, j2=j2, rtol=rtol).y[:, -1]
    r, v, d = y[0:3], y[3:6], y[6:9]
    M = rtn_matrix(r, v)
    cart = M.T @ d
    rp = r + d
    q = M.T @ rp                       # burned position in reference RTN axes
    rn = np.linalg.norm(r)
    curv = np.array([np.linalg.norm(rp) - rn, rn * np.arctan2(q[1], q[0]), rn * np.arcsin(q[2] / np.linalg.norm(rp))])
    return cart, curv


# ================================================================ covariance / encounter plane / Pc (own code)
def cov_rtn(sr, st, sn, c_tr, c_nr, c_nt):
    return np.array([[sr * sr, c_tr * sr * st, c_nr * sr * sn],
                     [c_tr * sr * st, st * st, c_nt * st * sn],
                     [c_nr * sr * sn, c_nt * st * sn, sn * sn]])


def cdm_geometry(row):
    """Relative state, a_t (m) and combined 3D position covariance in the TARGET RTN frame.

    The chaser covariance is given in the chaser's own RTN frame; its axes are rebuilt from the
    target state (a, 0, 0) / (0, sqrt(mu/a), 0) (circular speed, deliberately simpler than the
    module's energy-based speed) plus the relative state, and the covariance rotated."""
    p = row[POS].to_numpy(float)
    v = row[VEL].to_numpy(float)
    a = float(row["t_j2k_sma"]) * 1e3
    Ct = cov_rtn(*(float(row[f"t_{k}"]) for k in ("sigma_r", "sigma_t", "sigma_n", "ct_r", "cn_r", "cn_t")))
    Cc = cov_rtn(*(float(row[f"c_{k}"]) for k in ("sigma_r", "sigma_t", "sigma_n", "ct_r", "cn_r", "cn_t")))
    r_c = np.array([a, 0.0, 0.0]) + p
    v_c = np.array([0.0, np.sqrt(MU / a), 0.0]) + v
    M = rtn_matrix(r_c, v_c)
    return p, v, a, Ct + M @ Cc @ M.T


def plane_basis(v, p_hint):
    h = v / np.linalg.norm(v)
    e1 = p_hint - (p_hint @ h) * h
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(h, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return np.vstack([e1, np.cross(h, e1)])


def project(p, v, C):
    B = plane_basis(v, p)
    return B @ p, B @ C @ B.T, B


_NODES = {}


def _disc_nodes(nr, nth):
    if (nr, nth) not in _NODES:
        t, w = np.polynomial.legendre.leggauss(nr)
        rr, wr = 0.5 * (t + 1.0), 0.5 * w
        th = 2.0 * np.pi * np.arange(nth) / nth
        X = (rr[:, None] * np.cos(th)[None, :]).ravel()
        Y = (rr[:, None] * np.sin(th)[None, :]).ravel()
        W = np.repeat(wr * rr * 2.0 * np.pi / nth, nth)
        _NODES[(nr, nth)] = (X, Y, W)
    return _NODES[(nr, nth)]


def log10_pc_disc(x2, C2, R, nr=160, nth=320):
    """log10 P(|y| < R), y ~ N(x2, C2): polar Gauss-Legendre (radius) x trapezoid (angle) over the
    hard-body disc, accumulated in log space. x2 may be (m, 2)."""
    x2 = np.atleast_2d(np.asarray(x2, float))
    X, Y, W = _disc_nodes(nr, nth)
    X, Y, W = X * R, Y * R, W * R * R
    Ci = np.linalg.inv(C2)
    det = np.linalg.det(C2)
    dx = X[None, :] - x2[:, :1]
    dy = Y[None, :] - x2[:, 1:2]
    q = Ci[0, 0] * dx * dx + 2.0 * Ci[0, 1] * dx * dy + Ci[1, 1] * dy * dy
    lf = -0.5 * q + np.log(W)[None, :]
    m = lf.max(axis=1, keepdims=True)
    ln = m[:, 0] + np.log(np.exp(lf - m).sum(axis=1)) - np.log(2.0 * np.pi * np.sqrt(det))
    return ln / LN10


def pc3d_mc(p, v, C, R, n=2_000_000, seed=1, importance=True, chunk=500_000):
    """3D Monte Carlo of P(closest approach < R) under rectilinear relative motion.

    importance=True samples the position error around the most probable point of the collision line
    (minimum Mahalanobis distance on {p + e parallel to v}) and re-weights. Returns (log10 Pc, rel. SE)."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(C)
    Ci = np.linalg.inv(C)
    h = v / np.linalg.norm(v)
    s = ((h @ Ci @ p) / (h @ Ci @ h)) * h - p if importance else np.zeros(3)
    sw = sw2 = 0.0
    done = 0
    while done < n:
        m = min(chunk, n - done)
        z = rng.standard_normal((m, 3))
        e = z @ L.T + s
        x = p + e
        perp = x - (x @ h)[:, None] * h
        hit = np.einsum("ij,ij->i", perp, perp) < R * R
        if importance:
            lw = -0.5 * np.einsum("ij,jk,ik->i", e, Ci, e) + 0.5 * np.einsum("ij,ij->i", z, z)
            w = np.where(hit, np.exp(lw), 0.0)
        else:
            w = hit.astype(float)
        sw += w.sum()
        sw2 += (w * w).sum()
        done += m
    pc = sw / n
    se = np.sqrt(max(sw2 / n - pc * pc, 0.0) / n)
    return (np.log10(pc) if pc > 0 else -np.inf), (se / pc if pc > 0 else np.inf)


# ================================================================ Part A
def part_a(quick=False):
    a = 7050e3
    n = np.sqrt(MU / a ** 3)
    P = 2 * np.pi / n
    leads = (0.5, 1.0, 2.0, 3.0)
    dvs = (0.01, 0.05, 0.2)
    out = {"a_km": a / 1e3, "period_min": P / 60.0, "n_rad_s": n}

    # 0) the formula itself vs an independent derivation (CW state-transition matrix)
    stm_err = max(np.max(np.abs(cw_formula(dv, L * P, n) - cw_stm(dv, L * P, n))) for L in leads + (0.1, 1.37)
                  for dv in dvs)
    out["formula_vs_cw_stm_max_abs_err_m"] = float(stm_err)

    # 1) integrator convergence (worst case: 3 orbits, 0.2 m/s)
    c12, _ = burn_displacement(a, 0.2, 3 * P, rtol=1e-12)
    c10, _ = burn_displacement(a, 0.2, 3 * P, rtol=1e-10)
    c13, _ = burn_displacement(a, 0.2, 3 * P, rtol=1e-13)
    out["integrator_convergence_m"] = {"rtol1e-10_vs_1e-13": float(np.linalg.norm(c10 - c13)),
                                       "rtol1e-12_vs_1e-13": float(np.linalg.norm(c12 - c13))}

    def rows(j2, label, inc=np.radians(98.0), nu=0.0):
        res = []
        for L in leads:
            tau = L * P
            for dv in dvs:
                cw = cw_formula(dv, tau, n)
                cart, curv = burn_displacement(a, dv, tau, j2=j2, inc=inc, nu_burn=nu)
                norm = np.linalg.norm(cw)
                rec = {"model": label, "lead_orbits": L, "dv_mps": dv,
                       "cw_R": cw[0], "cw_T": cw[1],
                       "num_R": cart[0], "num_T": cart[1], "num_N": cart[2],
                       "curv_R": curv[0], "curv_T": curv[1],
                       "relerr_T_cart": (cart[1] - cw[1]) / abs(cw[1]),
                       "relerr_T_curv": (curv[1] - cw[1]) / abs(cw[1]),
                       "relerr_vec_cart": np.linalg.norm(cart - cw) / norm,
                       "relerr_vec_curv": np.linalg.norm(curv - cw) / norm}
                if abs(cw[0]) > 1e-6 * norm:
                    rec["relerr_R_cart"] = (cart[0] - cw[0]) / abs(cw[0])
                    rec["relerr_R_curv"] = (curv[0] - cw[0]) / abs(cw[0])
                else:                       # CW predicts dR = 0: report the absolute residual
                    rec["abs_R_cart_m"] = cart[0]
                    rec["abs_R_curv_m"] = curv[0]
                res.append({k: (float(x) if isinstance(x, (float, np.floating)) else x) for k, x in rec.items()})
        return res

    t0 = time.perf_counter()
    out["two_body"] = rows(False, "two-body circular")
    out["j2_u0"] = rows(True, "J2, i=98 deg, burn at u=0", nu=0.0)
    if not quick:
        out["j2_u90"] = rows(True, "J2, i=98 deg, burn at u=90", nu=np.pi / 2)

    # 2) sign checks
    sg = {}
    for L in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 3.0):
        cp, _ = burn_displacement(a, +0.05, L * P)
        cm, _ = burn_displacement(a, -0.05, L * P)
        sg[str(L)] = {"posigrade_dR": float(cp[0]), "posigrade_dT": float(cp[1]),
                      "retrograde_dR": float(cm[0]), "retrograde_dT": float(cm[1])}
    x_cross = brentq(lambda x: 4 * np.sin(x) - 3 * x, 0.5, 2.0)
    out["sign_checks"] = sg
    out["dT_sign_change_lead_orbits_cw"] = float(x_cross / (2 * np.pi))

    # 3) eccentricity sensitivity (two-body, dv = 0.05)
    ecc_rows = []
    eccs = (0.0017, 0.0044, 0.0187) if not quick else (0.0017, 0.0187)
    for e in eccs:
        for nu_deg in (0, 90, 180, 270):
            for L in leads:
                cw = cw_formula(0.05, L * P, n)
                _, curv = burn_displacement(a, 0.05, L * P, ecc=e, nu_burn=np.radians(nu_deg))
                ecc_rows.append({"ecc": e, "nu_burn_deg": nu_deg, "lead_orbits": L,
                                 "cw_R": float(cw[0]), "cw_T": float(cw[1]),
                                 "curv_R": float(curv[0]), "curv_T": float(curv[1]),
                                 "relerr_vec_curv": float(np.linalg.norm(curv - cw) / np.linalg.norm(cw))})
    out["eccentric"] = ecc_rows
    out["runtime_s"] = time.perf_counter() - t0
    return out


def print_part_a(A):
    print("=" * 100)
    print(f"PART A  CW vs numerical propagation   a = {A['a_km']:.0f} km, period {A['period_min']:.2f} min")
    print("=" * 100)
    print(f"formula vs CW state-transition matrix (expm): max |diff| = {A['formula_vs_cw_stm_max_abs_err_m']:.1e} m")
    ic = A["integrator_convergence_m"]
    print(f"integrator convergence (3 orbits, 0.2 m/s): |rtol 1e-10 - 1e-13| = {ic['rtol1e-10_vs_1e-13']:.1e} m, "
          f"|1e-12 - 1e-13| = {ic['rtol1e-12_vs_1e-13']:.1e} m")
    for key in ("two_body", "j2_u0", "j2_u90"):
        if key not in A:
            continue
        rs = A[key]
        print(f"\n-- {rs[0]['model']}  (displacement of the burned target at TCA, unburned-orbit RTN, metres)")
        print(f"{'lead':>5} {'dv':>5} | {'CW dR':>9} {'CW dT':>10} | {'num dR':>9} {'num dT':>10} {'num dN':>7} |"
              f" {'curv dR':>9} {'curv dT':>10} | {'rel dR':>9} {'rel dT':>9} {'|vec| cart':>10} {'|vec| curv':>10}")
        for r in rs:
            if "relerr_R_cart" in r:
                rr = f"{r['relerr_R_curv']:+9.1e}"
            else:
                rr = f"{r['abs_R_curv_m']:+7.3f} m"
            print(f"{r['lead_orbits']:5.1f} {r['dv_mps']:5.2f} | {r['cw_R']:9.3f} {r['cw_T']:10.2f} | {r['num_R']:9.3f} "
                  f"{r['num_T']:10.2f} {r['num_N']:7.3f} | {r['curv_R']:9.3f} {r['curv_T']:10.2f} | {rr:>9} "
                  f"{r['relerr_T_curv']:+9.1e} {r['relerr_vec_cart']:10.1e} {r['relerr_vec_curv']:10.1e}")
        print("   (rel dR / rel dT = curvilinear vs CW; where CW dR = 0 the absolute curvilinear dR is shown)")
    print("\n-- sign checks, dv = +/-0.05 m/s (two-body, Cartesian RTN, metres)")
    for L, s in A["sign_checks"].items():
        print(f"  lead {float(L):4.2f} orb: posigrade dR {s['posigrade_dR']:+9.2f} dT {s['posigrade_dT']:+10.2f}   "
              f"retrograde dR {s['retrograde_dR']:+9.2f} dT {s['retrograde_dT']:+10.2f}")
    print(f"  CW: posigrade dT changes sign (ahead -> behind) at lead = {A['dT_sign_change_lead_orbits_cw']:.3f} orbit")
    print("\n-- eccentric target orbit (two-body, dv = 0.05 m/s), curvilinear vs CW (circular, n from a)")
    er = pd.DataFrame(A["eccentric"])
    for e, g in er.groupby("ecc"):
        half = g[g.lead_orbits == 0.5]
        integ = g[g.lead_orbits != 0.5]
        print(f"  e = {e:.4f}: half-orbit dR {half.curv_R.min():.2f}..{half.curv_R.max():.2f} m (CW {half.cw_R.iloc[0]:.2f}),"
              f" max rel vec err {g.relerr_vec_curv.max():.2e};"
              f" integer leads: |dR| up to {integ.curv_R.abs().max():.2f} m (CW 0), max rel vec err "
              f"{integ.relerr_vec_curv.max():.2e}")


# ================================================================ Part B
def e2e_after_burn(a, inc, u_tca, p, v, dv, tau, j2=False, ecc=0.0, nu_tca=0.0):
    """Nonlinear encounter after an along-velocity burn of the target tau seconds before TCA.

    Target: orbit (a, ecc, inc) through the TCA point at argument of latitude u_tca (true anomaly
    nu_tca there); chaser: target state + relative state (target RTN axes). Both propagated
    (two-body or J2). The new TCA is the minimum of the true distance. Returns the new relative
    position / velocity expressed in the BURNED target's own RTN frame at the new TCA (covariances
    are attached to the objects' local RTN frames, so this is the frame in which the unchanged CDM
    covariance applies), the TCA shift (s) and the round-trip error of the reference propagation (m)."""
    r_t0, v_t0 = kepler_to_cart(a, ecc, inc, 0.0, u_tca - nu_tca, nu_tca)
    M = rtn_matrix(r_t0, v_t0)
    r_c0, v_c0 = r_t0 + M @ p, v_t0 + M @ v
    n = np.sqrt(MU / a ** 3)
    W = 2.0 + 3.0 * (6.0 * np.pi * 3.0 * abs(dv) / n) / np.linalg.norm(v)   # > 3x the largest TCA shift
    yb = prop(np.concatenate([r_t0, v_t0]), 0.0, -tau, j2=j2).y[:, -1]
    vh = yb[3:] / np.linalg.norm(yb[3:])
    tgt = prop_pair(np.concatenate([yb, np.zeros(3), dv * vh]), -tau, W, j2=j2, dense=True)
    chf = prop(np.concatenate([r_c0, v_c0]), 0.0, W, j2=j2, dense=True)
    chb = prop(np.concatenate([r_c0, v_c0]), 0.0, -W, j2=j2, dense=True)

    def rel(t):
        t = np.atleast_1d(np.asarray(t, float))
        yt = tgt.sol(t)
        yc = np.where(t[None, :] >= 0, chf.sol(np.maximum(t, 0)), chb.sol(np.minimum(t, 0)))
        return yc[0:3] - (yt[0:3] + yt[6:9]), yc[3:6] - (yt[3:6] + yt[9:12])

    ts = np.linspace(-W, W, 8001)
    d2 = np.sum(rel(ts)[0] ** 2, axis=0)
    k = int(np.argmin(d2))
    lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, len(ts) - 1)]
    res = minimize_scalar(lambda t: float(np.sum(rel(t)[0] ** 2)), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    dp, dvel = (x[:, 0] for x in rel(res.x))
    yt = tgt.sol(res.x)
    Ml = rtn_matrix(yt[0:3] + yt[6:9], yt[3:6] + yt[9:12])      # burned target's RTN at the new TCA
    round_trip = float(np.linalg.norm(tgt.sol(0.0)[0:3] - r_t0))
    return Ml.T @ dp, Ml.T @ dvel, float(res.x), round_trip


def in_plane(p, v):
    h = v / np.linalg.norm(v)
    return p - (p @ h) * h


def min_dv_scan(p, v, C2, B, R, n, tau, target, max_dv, sign, ngrid=1000):
    """Own search for the smallest |dv| (CW displacement) with log10 Pc < target. Pc(dv) along a line
    in the encounter plane is log-concave (Gaussian convolved with a disc), so the first crossing of
    a fine scan, refined by root finding, is the minimum. Returns (dv_min or None, min log10 Pc on scan)."""
    mags = np.linspace(0.0, max_dv, ngrid + 1)
    x2 = (p[None, :] - cw_formula_vec(sign * mags, tau, n)) @ B.T
    lp = log10_pc_disc(x2, C2, R, nr=64, nth=128)
    hit = np.flatnonzero(lp < target)
    if hit.size == 0:
        return None, float(lp.min())
    i = int(hit[0])
    if i == 0:
        return 0.0, float(lp.min())

    def f(m):
        return float(log10_pc_disc(((p - cw_formula(sign * m, tau, n)) @ B.T)[None, :], C2, R)[0] - target)
    a_, b_ = mags[i - 1], mags[i]
    while f(a_) < 0 and a_ > 0:        # medium grid vs fine grid disagreement at the edge
        a_, b_ = max(a_ - (mags[1] - mags[0]), 0.0), a_
    while f(b_) >= 0 and b_ < max_dv:
        a_, b_ = b_, min(b_ + (mags[1] - mags[0]), max_dv)
    return brentq(f, a_, b_, xtol=1e-12, rtol=1e-12), float(lp.min())


def part_b(quick=False, target=-5.0, max_dv=0.5):
    import physics  # the module under test: ONLY manoeuvre_options is called
    te = pd.read_csv(ROOT / "data" / "test_data.csv")
    nmc = 400_000 if quick else 2_000_000
    out = {"target_log_pc": target, "max_dv": max_dv, "cdms": []}

    # --- convention check that decides the burn DIRECTION (Pc itself is sign-blind)
    a_t, a_c = te.t_j2k_sma.to_numpy() * 1e3, te.c_j2k_sma.to_numpy() * 1e3
    vv = te[VEL].to_numpy(float)
    vcirc = np.sqrt(MU / a_t)
    use = np.abs(vv[:, 1]) > 100
    conv = {}
    for nm, s in (("chaser_minus_target", 1.0), ("target_minus_chaser", -1.0)):
        dv_ = s * vv
        vT = (MU * (1 / a_t - 1 / a_c) - (dv_ ** 2).sum(1)) / (2 * dv_[:, 1])
        conv[nm] = float(np.mean(np.abs(vT[use] / vcirc[use] - 1) < 0.03))
    out["velocity_sign_energy_check"] = conv
    tr = pd.read_csv(ROOT / "data" / "train_data.csv", usecols=["time_to_tca", "t_j2k_sma"])
    P_tr = 2 * np.pi / np.sqrt(MU / (tr.t_j2k_sma.to_numpy() * 1e3) ** 3) / 86400.0
    out["lead_time_feasibility"] = {
        "test_min_time_to_tca_d": float(te.time_to_tca.min()),
        "train_min_time_to_tca_d": float(tr.time_to_tca.min()),
        "train_frac_cdms_time_to_tca_lt_3_orbits": float(np.mean(tr.time_to_tca.to_numpy() < 3 * P_tr)),
        "train_frac_cdms_time_to_tca_lt_0p5_orbit": float(np.mean(tr.time_to_tca.to_numpy() < 0.5 * P_tr))}

    for idx, why in CDM_ROWS.items():
        row = te.loc[idx]
        t0 = time.perf_counter()
        opts = physics.manoeuvre_options(row, target_log_pc=target, burn_lead_orbits=(0.5, 1, 2, 3), max_dv=max_dv)
        t_call = time.perf_counter() - t0
        p, v, a, C = cdm_geometry(row)
        n = np.sqrt(MU / a ** 3)
        x2, C2, B = project(p, v, C)
        mahal3 = float(np.sqrt(p @ np.linalg.solve(C, p)))
        hbr_mod = next(o["hbr_m"] for o in opts if o is not None) if any(o is not None for o in opts) else np.nan
        risk = float(row["risk"])
        hbr_esa = brentq(lambda R: float(log10_pc_disc(x2, C2, R)[0]) - risk, 0.01, 300.0, xtol=1e-10)
        hbr_span = (float(row["t_span"]) + float(row["c_span"])) / 2
        lp_before = float(log10_pc_disc(x2, C2, hbr_mod)[0])
        lp_before_hi = float(log10_pc_disc(x2, C2, hbr_mod, nr=320, nth=640)[0])
        mc_plain, mc_plain_se = pc3d_mc(p, v, C, hbr_mod, n=nmc * 2, importance=False)
        mc_is, mc_is_se = pc3d_mc(p, v, C, hbr_mod, n=nmc)
        inc = np.radians(float(row["t_j2k_inc"]))
        u_tca = np.arcsin(np.clip(np.sin(np.radians(float(row["geocentric_latitude"]))) / np.sin(inc), -1, 1))
        dp0, dv0, dt0, rt0 = e2e_after_burn(a, inc, u_tca, p, v, 0.0, 0.5 * 2 * np.pi / n)
        ecc = float(row["t_j2k_ecc"])
        rec = {"test_row": idx, "why": why, "event_id": int(row["event_id"]),
               "time_to_tca_d": float(row["time_to_tca"]), "esa_risk": risk, "t_j2k_ecc": ecc,
               "miss_m": float(np.linalg.norm(p)), "rel_speed_mps": float(np.linalg.norm(v)),
               "rel_vel_rtn": [float(x) for x in v], "a_km": a / 1e3,
               "manoeuvre_call_s": t_call,
               "mahal3d_own": mahal3, "mahal3d_esa": float(row["mahalanobis_distance"]),
               "sigma_plane_m": [float(s) for s in np.sqrt(np.linalg.eigvalsh(C2))[::-1]],
               "hbr_module_m": float(hbr_mod), "hbr_from_esa_risk_own_m": float(hbr_esa), "hbr_spans_m": hbr_span,
               "log_pc_before_module": float(opts[0]["log_pc_before"]) if opts[0] else None,
               "log_pc_before_own_grid": lp_before, "grid_convergence_decades": abs(lp_before_hi - lp_before),
               "log_pc_before_own_mc_plain": mc_plain, "mc_plain_rel_se": mc_plain_se,
               "log_pc_before_own_mc_is": mc_is, "mc_is_rel_se": mc_is_se,
               "e2e_no_burn_inplane_residual_m": float(np.linalg.norm(in_plane(dp0, dv0) - in_plane(p, v))),
               "cdm_p_along_v_m": float(p @ v / np.linalg.norm(v)), "e2e_round_trip_m": rt0,
               "options": []}
        seed = 100
        for L, entry in zip((0.5, 1, 2, 3), opts):
            tau = L * 2 * np.pi / n
            scan = {}
            for sname, sgn in (("posigrade", 1.0), ("retrograde", -1.0)):
                scan[sname] = min_dv_scan(p, v, C2, B, hbr_mod, n, tau, target, max_dv, sgn)
            if entry is None:
                rec["options"].append({"lead_orbits": L, "module": None,
                                       "own_scan_min_dv": {k: s[0] for k, s in scan.items()},
                                       "own_scan_min_log_pc": {k: s[1] for k, s in scan.items()}})
                continue
            if abs(entry["lead_time_s"] - tau) > 1e-6 * tau:
                print(f"  !! lead time mismatch row {idx} L={L}: module {entry['lead_time_s']} own {tau}")
            for sname, o in entry["options"].items():
                if o is None:
                    rec["options"].append({"lead_orbits": L, "direction": sname, "module": None,
                                           "own_scan_min_dv": scan[sname][0], "own_scan_min_log_pc": scan[sname][1]})
                    continue
                dv = o["dv_mps"]
                d_mod = np.array(o["d_target_rtn_m"])
                d_cw = cw_formula(dv, tau, n)
                cart, curv = burn_displacement(a, dv, tau)
                # (1) module displacement, own plane + integration
                rel1 = p - d_mod
                x2a = B @ rel1
                lp1 = float(log10_pc_disc(x2a, C2, hbr_mod)[0])
                # (2) 3D importance-sampled MC with the same displacement
                seed += 1
                lp2, se2 = pc3d_mc(rel1, v, C, hbr_mod, n=nmc, seed=seed)
                # (3) numerically propagated displacement (curvilinear = local-frame form), rectilinear
                #     encounter; and, for the record, the Cartesian displacement in the frozen TCA frame
                lp3 = float(log10_pc_disc(B @ (p - curv), C2, hbr_mod)[0])
                lp3_cart = float(log10_pc_disc(B @ (p - cart), C2, hbr_mod)[0])
                # (4) end-to-end nonlinear encounter: circular two-body, circular J2, and the target's own
                #     eccentricity at four (unknown) true anomalies at TCA
                e2e = {}
                variants = [("two_body", False, 0.0, 0.0), ("j2", True, 0.0, 0.0)]
                variants += [(f"ecc_nu{d}", False, ecc, np.radians(d)) for d in (0, 90, 180, 270)]
                for lab, j2, e_, nu_ in variants:
                    dp, dvel, dt, rt = e2e_after_burn(a, inc, u_tca, p, v, dv, tau, j2=j2, ecc=e_, nu_tca=nu_)
                    xx, CC, _ = project(dp, dvel, C)
                    e2e[lab] = {"log_pc": float(log10_pc_disc(xx, CC, hbr_mod)[0]),
                                "miss_m": float(np.linalg.norm(in_plane(dp, dvel))),
                                "tca_shift_s": dt, "round_trip_m": rt,
                                "dv_rel_change_mps": float(np.linalg.norm(dvel - v))}
                ecc_lp = [e2e[k]["log_pc"] for k in e2e if k.startswith("ecc")]
                # sensitivity / HBR
                lp_under = float(log10_pc_disc(B @ (p - cw_formula(0.9 * dv, tau, n)), C2, hbr_mod)[0])
                lp_esa_hbr = float(log10_pc_disc(x2a, C2, hbr_esa)[0])
                lp_span_hbr = float(log10_pc_disc(x2a, C2, hbr_span)[0])
                own_min = scan[sname][0]
                rec["options"].append({
                    "lead_orbits": L, "direction": sname, "is_best": o["abs_dv_mps"] == entry["abs_dv_mps"],
                    "dv_mps": dv, "module_log_pc_after": o["log_pc_after"],
                    "module_miss_after_m": o["miss_distance_after_m"],
                    "d_module_m": [float(x) for x in d_mod],
                    "d_module_vs_own_cw_m": float(np.linalg.norm(d_mod - d_cw)),
                    "d_num_cart_m": [float(x) for x in cart], "d_num_curv_m": [float(x) for x in curv],
                    "own_grid": lp1, "own_mc_is": lp2, "own_mc_is_rel_se": se2,
                    "own_numdisp": lp3, "own_numdisp_cartesian_frozen_frame": lp3_cart,
                    "e2e_two_body": e2e["two_body"], "e2e_j2": e2e["j2"],
                    "e2e_ecc": {k: v_ for k, v_ in e2e.items() if k.startswith("ecc")},
                    "e2e_ecc_min": min(ecc_lp), "e2e_ecc_max": max(ecc_lp),
                    "own_miss_after_m": float(np.linalg.norm(x2a)),
                    "own_scan_min_dv": own_min,
                    "min_dv_rel_diff": (abs(dv) - own_min) / own_min if own_min else None,
                    "log_pc_at_0p9_dv": lp_under,
                    "log_pc_with_hbr_from_esa_risk": lp_esa_hbr, "log_pc_with_span_hbr": lp_span_hbr})
        out["cdms"].append(rec)
    return out


def print_part_b(Bres):
    tgt = Bres["target_log_pc"]
    print("\n" + "=" * 100)
    print(f"PART B  manoeuvre_options(target_log_pc={tgt}, leads 0.5/1/2/3 orbits, max_dv={Bres['max_dv']}) "
          "re-checked independently")
    print("=" * 100)
    vc = Bres["velocity_sign_energy_check"]
    print(f"relative-velocity sign (energy relation gives a target speed within 3% of circular): "
          f"chaser-minus-target {vc['chaser_minus_target']:.1%}, target-minus-chaser {vc['target_minus_chaser']:.1%}")
    lf = Bres["lead_time_feasibility"]
    print(f"lead-time feasibility: min time_to_tca test {lf['test_min_time_to_tca_d']:.2f} d, train "
          f"{lf['train_min_time_to_tca_d']:.3f} d; train CDMs with time_to_tca < 3 orbits: "
          f"{lf['train_frac_cdms_time_to_tca_lt_3_orbits']:.2%}, < 0.5 orbit: {lf['train_frac_cdms_time_to_tca_lt_0p5_orbit']:.2%}")
    for c in Bres["cdms"]:
        print(f"\n## test row {c['test_row']} (event {c['event_id']}, {c['time_to_tca_d']:.2f} d to TCA): {c['why']}")
        print(f"   miss {c['miss_m']:.1f} m, v_rel {c['rel_speed_mps']:.0f} m/s, rtn {np.round(c['rel_vel_rtn'], 1).tolist()}, "
              f"plane sigmas {c['sigma_plane_m'][0]:.1f} x {c['sigma_plane_m'][1]:.1f} m; manoeuvre call {c['manoeuvre_call_s']:.2f} s")
        print(f"   Mahalanobis 3D own {c['mahal3d_own']:.4f} vs ESA {c['mahal3d_esa']:.4f};  HBR module {c['hbr_module_m']:.3f} m, "
              f"own (solved from ESA risk) {c['hbr_from_esa_risk_own_m']:.3f} m, spans {c['hbr_spans_m']:.3f} m")
        print(f"   before: ESA risk {c['esa_risk']:.4f} | module {c['log_pc_before_module']:.4f} | own grid "
              f"{c['log_pc_before_own_grid']:.4f} (grid conv {c['grid_convergence_decades']:.1e}) | plain MC "
              f"{c['log_pc_before_own_mc_plain']:.4f} (+/-{c['mc_plain_rel_se']:.1%}) | IS-MC {c['log_pc_before_own_mc_is']:.4f} "
              f"(+/-{c['mc_is_rel_se']:.1%});  e2e no-burn in-plane residual {c['e2e_no_burn_inplane_residual_m']:.1e} m "
              f"(CDM p has {c['cdm_p_along_v_m']:+.2f} m along v_rel); target e = {c['t_j2k_ecc']:.5f}")
        print(f"   {'lead':>4} {'dir':>10} {'best':>4} {'dv m/s':>9} | {'module':>8} {'grid':>8} {'IS-MC':>8} {'numdisp':>8} "
              f"{'e2e 2B':>8} {'e2e J2':>8} {'e2e ecc min..max':>17} | {'miss mod':>8} {'e2e':>8} | {'own min':>9} "
              f"{'d rel':>8} | {'0.9dv':>7} {'HBResa':>7} {'HBRspan':>7}")
        for o in c["options"]:
            if o.get("module") is None and "dv_mps" not in o:
                own = o["own_scan_min_dv"]
                mn = o["own_scan_min_log_pc"]
                if isinstance(own, dict):
                    txt = ", ".join(f"{k}: min |dv| {own[k]}, lowest log10 Pc {mn[k]:.2f}" for k in own)
                else:
                    txt = f"own scan: min |dv| {own}, lowest log10 Pc within max_dv {mn:.2f}"
                print(f"   {o['lead_orbits']:4.1f} {o.get('direction', 'both'):>10}  module: None | {txt}")
                continue
            e2, ej = o["e2e_two_body"], o["e2e_j2"]
            dr = "" if o["min_dv_rel_diff"] is None else f"{o['min_dv_rel_diff']:+8.1e}"
            print(f"   {o['lead_orbits']:4.1f} {o['direction']:>10} {'*' if o['is_best'] else '':>4} {o['dv_mps']:+9.5f} | "
                  f"{o['module_log_pc_after']:8.4f} {o['own_grid']:8.4f} {o['own_mc_is']:8.4f} {o['own_numdisp']:8.4f} "
                  f"{e2['log_pc']:8.4f} {ej['log_pc']:8.4f} {o['e2e_ecc_min']:8.3f}..{o['e2e_ecc_max']:7.3f} | "
                  f"{o['module_miss_after_m']:8.2f} {e2['miss_m']:8.2f} | "
                  f"{o['own_scan_min_dv']:9.6f} {dr:>8} | {o['log_pc_at_0p9_dv']:7.3f} {o['log_pc_with_hbr_from_esa_risk']:7.3f} "
                  f"{o['log_pc_with_span_hbr']:7.3f}")
        print("   columns: log10 Pc after the burn. grid = own integration of the module's displacement; IS-MC = 3D "
              "Monte Carlo; numdisp = numerically propagated displacement; e2e = nonlinear encounter (new TCA, new v_rel,"
              " burned target's local RTN), circular 2-body / J2 / target's own e at nu_TCA = 0,90,180,270 deg; "
              "0.9dv = 10% under-burn; HBResa / HBRspan = same displacement with the other radii")


def summarise(A, Bres, target):
    issues = []
    tb = pd.DataFrame(A["two_body"])
    j2 = pd.DataFrame(A["j2_u0"] + A.get("j2_u90", []))
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"A. two-body: max rel vector error, Cartesian {tb.relerr_vec_cart.max():.1e}, curvilinear "
          f"{tb.relerr_vec_curv.max():.1e};  J2: curvilinear {j2.relerr_vec_curv.max():.1e}")
    sg = A["sign_checks"]
    ok_sign = all(sg[str(L)]["posigrade_dT"] < 0 and sg[str(L)]["retrograde_dT"] > 0 for L in (0.5, 1.0, 2.0, 3.0)) \
        and sg["0.5"]["posigrade_dR"] > 0 and sg["0.5"]["retrograde_dR"] < 0 and sg["0.1"]["posigrade_dT"] > 0
    print(f"   sign conventions (posigrade -> behind for leads >= 0.5 orbit, higher at 0.5 orbit; ahead for "
          f"0.1 orbit): {'OK' if ok_sign else 'FAIL'}")
    worst = []
    for c in Bres["cdms"]:
        for o in c["options"]:
            if "dv_mps" not in o:
                continue
            worst.append({"row": c["test_row"], "lead": o["lead_orbits"], "dir": o["direction"], "best": o["is_best"],
                          "grid": o["own_grid"] - target, "mc": o["own_mc_is"] - target,
                          "mc_z": (o["own_mc_is"] - o["own_grid"]) * LN10 / o["own_mc_is_rel_se"],
                          "numdisp": o["own_numdisp"] - target, "e2e": o["e2e_two_body"]["log_pc"] - target,
                          "e2e_j2": o["e2e_j2"]["log_pc"] - target, "e2e_ecc_max": o["e2e_ecc_max"] - target,
                          "d_along_m": abs(o["d_module_m"][1]), "under": o["log_pc_at_0p9_dv"] - target,
                          "hbr_esa": o["log_pc_with_hbr_from_esa_risk"] - target,
                          "mindv": np.nan if o["min_dv_rel_diff"] is None else o["min_dv_rel_diff"]})
    W = pd.DataFrame(worst)
    Wb = W[W.best]
    print(f"B. {len(W)} options ({len(Wb)} of them the module's pick per lead time) checked on {len(Bres['cdms'])} CDMs;"
          f" excess over the target (decades, + = Pc ABOVE the requested level):")
    print(f"   {'':>11}  {'all options':>24}   {'picked options':>24}")
    for col in ("grid", "mc", "numdisp", "e2e", "e2e_j2", "e2e_ecc_max", "under", "hbr_esa"):
        print(f"   {col:>11}: max {W[col].max():+.4f} median {W[col].median():+.4f}   max {Wb[col].max():+.4f} "
              f"median {Wb[col].median():+.4f}")
    print(f"   IS-MC vs own grid: max |z| = {W.mc_z.abs().max():.2f} (z = difference / MC standard error)")
    print(f"   returned |dv| vs own minimum: max rel diff {W.mindv.abs().max():.1e}")
    big = W[W.d_along_m > 5000]
    small = W[W.d_along_m <= 5000]
    if len(big):
        print(f"   options with along-track shift > 5 km ({len(big)}): worst e2e excess (any variant) "
              f"{big[['e2e', 'e2e_j2', 'e2e_ecc_max']].max().max():+.3f} decades; <= 5 km ({len(small)}): "
              f"{small[['e2e', 'e2e_j2', 'e2e_ecc_max']].max().max():+.3f}")
    return {"part_a_ok_sign": ok_sign, "excess_over_target": W.to_dict(orient="records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo samples / variants")
    ap.add_argument("--json", type=Path, default=None, help="write all results to this JSON file")
    args = ap.parse_args()
    t0 = time.perf_counter()
    A = part_a(args.quick)
    print_part_a(A)
    Bres = part_b(args.quick)
    print_part_b(Bres)
    S = summarise(A, Bres, Bres["target_log_pc"])
    print(f"\ntotal runtime {time.perf_counter() - t0:.1f} s")
    if args.json:
        args.json.write_text(json.dumps({"part_a": A, "part_b": Bres, "summary": S}, indent=1, default=float))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
