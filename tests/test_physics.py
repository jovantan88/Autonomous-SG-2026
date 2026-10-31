"""Fast synthetic tests for src/physics.py (2D Pc quadrature, frames, CW manoeuvre model)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import dblquad, solve_ivp
from scipy.linalg import expm
from scipy.stats import ncx2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import physics as P  # noqa: E402

LN10 = np.log(10)


def pc(mu, C, R, floor=-np.inf):
    return P.log10_pc_2d(np.array([mu], float), np.array([C], float), R, floor=floor)[0]


def dense_grid_log10_pc(mu, C, R, n=2001):
    """Brute force: midpoint rule on an n x n grid over the disc's bounding box."""
    h = 2 * R / n
    xs = -R + h * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(xs, xs)
    inside = X ** 2 + Y ** 2 <= R ** 2
    Ci = np.linalg.inv(C)
    dx, dy = X[inside] - mu[0], Y[inside] - mu[1]
    q = Ci[0, 0] * dx * dx + 2 * Ci[0, 1] * dx * dy + Ci[1, 1] * dy * dy
    lf = -0.5 * q - np.log(2 * np.pi * np.sqrt(np.linalg.det(C)))
    m = lf.max()
    return (m + np.log(np.exp(lf - m).sum() * h * h)) / LN10


# ------------------------------------------------------------------ 2D Pc
def isotropic_exact_log10(d, s, R):
    """Exact isotropic Pc via the radial integral with a scaled Bessel function (log-safe)."""
    from scipy.integrate import quad
    from scipy.special import i0e

    # integrand r/s^2 exp(-(r^2 + d^2)/(2 s^2)) I0(r d / s^2) = r/s^2 exp(-(r - d)^2/(2 s^2)) i0e(r d / s^2)
    c = -((R - d) ** 2) / (2 * s * s) if d > R else 0.0     # factor out the peak
    val, _ = quad(lambda r: r / s ** 2 * np.exp(-(r - d) ** 2 / (2 * s * s) - c) * i0e(r * d / s ** 2),
                  0, R, epsabs=0, epsrel=1e-12, limit=200)
    return (np.log(val) + c) / LN10


@pytest.mark.parametrize("d", [0.0, 300.0, 1000.0])
def test_isotropic_small_hbr_matches_analytic(d):
    s, R = 200.0, 5.0
    approx = np.log10(R ** 2 / (2 * s ** 2) * np.exp(-d ** 2 / (2 * s ** 2)))
    assert pc([d, 0.0], np.eye(2) * s ** 2, R) == pytest.approx(approx, abs=2e-3)


@pytest.mark.parametrize("d_over_s, R_over_s", [(0.0, 3.0), (2.0, 1.0), (5.0, 1.0), (1.0, 4.0), (8.0, 0.5)])
def test_isotropic_exact_noncentral_chi2(d_over_s, R_over_s):
    """Isotropic Gaussian: |z|^2 / s^2 ~ noncentral chi2(2, (d/s)^2) -> exact Pc for any HBR."""
    s = 10.0
    exact = np.log10(ncx2.cdf(R_over_s ** 2, 2, d_over_s ** 2))
    assert pc([d_over_s * s, 0.0], np.eye(2) * s ** 2, R_over_s * s) == pytest.approx(exact, abs=1e-6)


@pytest.mark.parametrize("d, s, R", [(2500.0, 200.0, 5.0), (2000.0, 50.0, 5.0), (60.0, 2.0, 15.0)])
def test_far_tail_no_underflow_and_floor(d, s, R):
    """Many-sigma misses (Pc down to 1e-350, below double precision) stay accurate in log space,
    including HBR >> sigma; the public log10 is floored at -30."""
    lnp = P.ln_pc_2d(np.array([[d, 0.0]]), np.array([np.eye(2) * s ** 2]), R)[0]
    assert np.isfinite(lnp)
    assert lnp / LN10 == pytest.approx(isotropic_exact_log10(d, s, R), abs=1e-6)
    assert pc([d, 0.0], np.eye(2) * s ** 2, R, floor=-30.0) == max(-30.0, lnp / LN10)


def test_symmetric_cases():
    C = np.array([[400.0, 0.0], [0.0, 25.0]])
    base = pc([30.0, 8.0], C, 6.0)
    for mu in ([-30.0, 8.0], [30.0, -8.0], [-30.0, -8.0]):
        assert pc(mu, C, 6.0) == pytest.approx(base, abs=1e-9)


def test_rotation_invariance_2d():
    rng = np.random.default_rng(0)
    C = np.array([[900.0, 100.0], [100.0, 16.0]])
    mu = np.array([25.0, -9.0])
    base = pc(mu, C, 7.0)
    for th in rng.uniform(0, 2 * np.pi, 5):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        assert pc(Rm @ mu, Rm @ C @ Rm.T, 7.0) == pytest.approx(base, abs=1e-8)


def test_rotation_invariance_3d_pipeline():
    """Rotating relative position, velocity and covariance together leaves the 2D Pc unchanged
    (the encounter-plane basis changes, the probability must not)."""
    rng = np.random.default_rng(1)
    p = np.array([[120.0, 40.0, -310.0]])
    v = np.array([[-15.0, -11000.0, 6000.0]])
    A = rng.normal(size=(3, 3))
    C = np.array([A @ A.T * 900.0 + np.diag([25.0, 1e4, 100.0])])
    x2, C2, _ = P.encounter_plane(p, v, C)
    base = P.log10_pc_2d(x2, C2, 5.0)[0]
    for _ in range(3):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        x2r, C2r, _ = P.encounter_plane(p @ Q.T, v @ Q.T, (Q @ C[0] @ Q.T)[None])
        assert P.log10_pc_2d(x2r, C2r, 5.0)[0] == pytest.approx(base, abs=1e-8)
        assert P.mahalanobis_3d(p @ Q.T, np.array([Q @ C[0] @ Q.T]))[0] == pytest.approx(
            P.mahalanobis_3d(p, C)[0], rel=1e-10)


@pytest.mark.parametrize("mu, C, R", [
    ([10.0, 5.0], [[100.0, 30.0], [30.0, 25.0]], 5.0),            # HBR ~ sigma
    ([3.0, 0.0], [[1.0, 0.0], [0.0, 0.01]], 5.0),                  # HBR >> sigma, mean inside disc
    ([0.5, 4.0], [[1e6, 0.0], [0.0, 0.25]], 10.0),                 # extremely elongated
    ([50.0, 20.0], [[400.0, -100.0], [-100.0, 100.0]], 10.0),      # correlated, Pc ~ 1e-4
    ([0.0, 8.0], [[0.5, 0.3], [0.3, 0.5]], 6.0),                   # mean just outside, tiny sigma
    ([18.0, 0.0], [[4.0, 0.0], [0.0, 4.0]], 5.0),                  # 6.5 sigma from the disc edge
])
def test_quadrature_vs_dense_grid(mu, C, R):
    assert pc(mu, C, R) == pytest.approx(dense_grid_log10_pc(np.array(mu), np.array(C), R), abs=2e-3)


def test_quadrature_vs_dblquad():
    mu, C, R = np.array([12.0, -4.0]), np.array([[60.0, 10.0], [10.0, 9.0]]), 8.0
    Ci, det = np.linalg.inv(C), np.linalg.det(C)

    def f(r, th):
        z = np.array([r * np.cos(th), r * np.sin(th)]) - mu
        return r * np.exp(-0.5 * z @ Ci @ z) / (2 * np.pi * np.sqrt(det))

    ref, _ = dblquad(f, 0, 2 * np.pi, 0, R, epsabs=1e-14, epsrel=1e-10)
    assert pc(mu, C, R) == pytest.approx(np.log10(ref), abs=1e-6)


def test_max_pc_matches_small_hbr_formula():
    """Small HBR: max_k Pc(k C) = HBR^2 / (e m^2 sqrt(det C)) at k = m^2 / 2."""
    C = np.array([[2500.0, 300.0], [300.0, 400.0]])
    mu = np.array([400.0, 150.0])
    R = 2.0
    lpm, k = P.log10_pc_max_2d(mu[None], C[None], R)
    m2 = mu @ np.linalg.solve(C, mu)
    assert lpm[0] == pytest.approx(np.log10(R ** 2 / (np.e * m2 * np.sqrt(np.linalg.det(C)))), abs=1e-3)
    assert k[0] == pytest.approx(m2 / 2, rel=1e-3)
    # and it really is a maximum over the scale
    for f in (0.5, 0.9, 1.1, 2.0):
        assert P.log10_pc_2d(mu[None], C[None], R, scale=k * f)[0] <= lpm[0] + 1e-9


def test_esa_max_definition_vs_exact_max():
    """ESA's max_risk_estimate = Pc at k0 = m_2D^2 / 2 (max_risk_scaling); equal to the exact maximum
    over k for HBR << sigma, below it when HBR ~ sigma."""
    C = np.array([[[2500.0, 300.0], [300.0, 400.0]], [[30.0, 0.0], [0.0, 20.0]]])
    mu = np.array([[400.0, 150.0], [10.0, 3.0]])
    R = np.array([2.0, 9.0])
    lpe, k0 = P.log10_pc_max_esa_2d(mu, C, R)
    m2 = np.einsum("ni,nij,nj->n", mu, np.linalg.inv(C), mu)
    np.testing.assert_allclose(k0, m2 / 2, rtol=1e-12)
    np.testing.assert_allclose(lpe, P.log10_pc_2d(mu, C, R, scale=m2 / 2), atol=1e-12)
    lpx, kx = P.log10_pc_max_2d(mu, C, R)
    assert lpx[0] == pytest.approx(lpe[0], abs=1e-4)          # small HBR: same
    assert lpx[1] > lpe[1] + 0.02 and abs(np.log(kx[1] / k0[1])) > 0.5   # HBR ~ sigma: differ
    lpx2, _ = P.log10_pc_max_2d(mu, C, R, lp_k0=P.log10_pc_2d(mu, C, R, scale=k0, floor=-np.inf))
    np.testing.assert_allclose(lpx2, lpx, atol=1e-12)


def test_implied_hbr_roundtrip():
    C = np.array([[[2500.0, 300.0], [300.0, 400.0]], [[30.0, 0.0], [0.0, 20.0]], [[4.0, 0.0], [0.0, 1.0]]])
    mu = np.array([[400.0, 150.0], [30.0, 10.0], [6.0, 1.0]])
    R_true = np.array([3.0, 9.0, 4.0])                   # rows 2-3: HBR >~ sigma (non-linear regime)
    mre, _ = P.log10_pc_max_esa_2d(mu, C, R_true, floor=-np.inf)
    R, conv, lpe, k0 = P.implied_hbr(mu, C, mre, np.array([5.0, 5.0, 1.0]))
    assert conv.all()
    np.testing.assert_allclose(R, R_true, rtol=1e-5)
    np.testing.assert_allclose(lpe, mre, atol=1e-5)


# ------------------------------------------------------------------ covariance / frames
def test_position_covariance_det_and_symmetry():
    C = P.position_covariance(4.0, 140.0, 1.8, -0.4, 0.3, 0.04)
    corr = np.array([[1, -0.4, 0.3], [-0.4, 1, 0.04], [0.3, 0.04, 1]])
    assert np.allclose(C, C.T)
    assert np.linalg.det(C) == pytest.approx((4.0 * 140.0 * 1.8) ** 2 * np.linalg.det(corr), rel=1e-12)


def test_chaser_axes_orthonormal_and_crossing_angle():
    a = 7000.0
    v = np.sqrt(P.MU_EARTH / (a * 1e3))
    ang = np.radians(60.0)                               # chaser orbit plane rotated 60 deg about R
    rel_v = np.array([[0.0, v * np.cos(ang) - v, v * np.sin(ang)]])
    M = P.chaser_rtn_axes(np.zeros((1, 3)), rel_v, np.array([a]))[0]
    assert np.allclose(M.T @ M, np.eye(3), atol=1e-12)
    assert np.allclose(M[:, 0], [1, 0, 0], atol=1e-12)
    assert np.allclose(M[:, 1], [0, np.cos(ang), np.sin(ang)], atol=1e-12)
    # co-planar, same direction -> identity
    M0 = P.chaser_rtn_axes(np.zeros((1, 3)), np.array([[0.0, 1.0, 0.0]]), np.array([a]))[0]
    assert np.allclose(M0, np.eye(3), atol=1e-9)


def test_target_speed_energy_with_radius_term():
    """Eccentric target at perigee (v_R = 0, r != a) and a chaser 300 m higher on a crossing orbit:
    the energy relation with the 2 mu (1/r_c - 1/r_t) term recovers v_T to cm/s; without it ~1 m/s."""
    mu = P.MU_EARTH
    a_t, e_t, a_c = 7000e3, 0.01, 7100e3
    r_t = a_t * (1 - e_t)
    v_t = np.sqrt(mu * (2 / r_t - 1 / a_t))
    p = np.array([300.0, -40.0, 120.0])
    r_c = np.array([r_t, 0.0, 0.0]) + p
    d = np.array([0.002, np.cos(np.radians(50)), np.sin(np.radians(50))])
    v_c = d / np.linalg.norm(d) * np.sqrt(mu * (2 / np.linalg.norm(r_c) - 1 / a_c))
    rel_v = (v_c - np.array([0.0, v_t, 0.0]))[None]
    with_r = P.target_velocity_rtn(a_t / 1e3, a_c / 1e3, rel_v, p[None])[0, 1]
    without = P.target_velocity_rtn(a_t / 1e3, a_c / 1e3, rel_v)[0, 1]
    circ = P.target_velocity_rtn(a_t / 1e3)[1]
    assert abs(with_r - v_t) < 0.05
    assert abs(without - v_t) > 0.5
    assert abs(circ - v_t) > 50.0


# ------------------------------------------------------------------ Clohessy-Wiltshire
def _cw_matrix(n):
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0], A[3, 4] = 3 * n * n, 2 * n
    A[4, 3] = -2 * n
    A[5, 2] = -n * n
    return A


@pytest.mark.parametrize("lead_orbits", [0.25, 0.5, 1.0, 1.7, 3.0])
def test_cw_displacement_vs_state_transition_matrix(lead_orbits):
    n = P.mean_motion(7000.0)
    tau = lead_orbits * 2 * np.pi / n
    dv = 0.1
    x = expm(_cw_matrix(n) * tau) @ np.array([0, 0, 0, 0, dv, 0])
    np.testing.assert_allclose(P.cw_displacement(dv, tau, n), x[:3], atol=1e-9, rtol=1e-9)


def test_cw_displacement_vs_two_body():
    """Along-track burn on a circular orbit, exact two-body propagation vs CW (dv small)."""
    a, dv = 7000e3, 0.05
    n = np.sqrt(P.MU_EARTH / a ** 3)
    tau = 1.3 * 2 * np.pi / n

    def rhs(t, y):
        r = y[:3]
        return np.concatenate([y[3:], -P.MU_EARTH * r / np.linalg.norm(r) ** 3])

    vc = np.sqrt(P.MU_EARTH / a)
    y_ref = solve_ivp(rhs, (0, tau), [a, 0, 0, 0, vc, 0], rtol=1e-12, atol=1e-6).y[:, -1]
    y_man = solve_ivp(rhs, (0, tau), [a, 0, 0, 0, vc + dv, 0], rtol=1e-12, atol=1e-6).y[:, -1]
    r, vv = y_ref[:3], y_ref[3:]
    Rh = r / np.linalg.norm(r)
    Nh = np.cross(r, vv) / np.linalg.norm(np.cross(r, vv))
    Th = np.cross(Nh, Rh)
    d = y_man[:3] - r
    rtn = np.array([d @ Rh, d @ Th, d @ Nh])
    cw = P.cw_displacement(dv, tau, n)
    assert np.linalg.norm(rtn - cw) < 1e-3 * np.linalg.norm(cw) + 1.0


# ------------------------------------------------------------------ manoeuvre advisory
def _head_on_row(miss=(0.0, 0.0, 0.0)):
    row = {"relative_position_r": miss[0], "relative_position_t": miss[1], "relative_position_n": miss[2],
           "relative_velocity_r": 0.0, "relative_velocity_t": -15000.0, "relative_velocity_n": 0.0,
           "t_sigma_r": 5.0, "t_sigma_t": 100.0, "t_sigma_n": 5.0, "t_ct_r": 0.0, "t_cn_r": 0.0, "t_cn_t": 0.0,
           "c_sigma_r": 20.0, "c_sigma_t": 500.0, "c_sigma_n": 20.0, "c_ct_r": 0.0, "c_cn_r": 0.0, "c_cn_t": 0.0,
           "t_span": 8.0, "c_span": 2.0, "t_j2k_sma": 7000.0, "c_j2k_sma": 7000.0, "miss_distance": 0.0}
    return row


def test_manoeuvre_reduces_pc_monotonically_head_on():
    row = _head_on_row()
    df = pd.DataFrame([row])
    C = P.combined_covariance(df)
    p, v = df[P.POS_COLS].to_numpy(float), df[P.VEL_COLS].to_numpy(float)
    _, C2, B = P.encounter_plane(p, v, C)
    n = P.mean_motion(7000.0)
    tau = 0.5 * 2 * np.pi / n
    for sign in (1, -1):
        dvs = sign * np.linspace(0, 0.3, 61)
        rel = p - P.cw_displacement(dvs, tau, n)
        x2 = rel @ B[0].T
        lp = P.log10_pc_2d(x2, np.repeat(C2, len(dvs), axis=0), 5.0, floor=-np.inf)
        assert np.all(np.diff(lp) < 0)
    opts = P.manoeuvre_options(row, target_log_pc=-6.0, burn_lead_orbits=(0.5, 1.0), max_dv=0.5, robust=False)
    best = opts[0]
    assert best is not None and best["log_pc_after"] == pytest.approx(-6.0, abs=1e-6)
    pos, ret = best["options"]["posigrade"], best["options"]["retrograde"]
    assert pos["abs_dv_mps"] == pytest.approx(ret["abs_dv_mps"], rel=1e-6)   # symmetric geometry
    # exactly one orbit before TCA an along-track burn gives no radial offset, and the along-track
    # offset lies along the head-on relative velocity -> it cannot change the miss: no option
    assert opts[1] is None


def test_manoeuvre_already_safe_and_signed_choice():
    row = _head_on_row(miss=(40.0, 0.0, 0.0))             # chaser 40 m (~2 sigma) above the target
    opts = P.manoeuvre_options(row, target_log_pc=-6.0, burn_lead_orbits=(0.5,), max_dv=0.5)
    o = opts[0]
    assert o["d_target_rtn_m"][0] < 0 and "lower" in o["effect"] and "ahead" in o["effect"]
    # raising the target (posigrade, dR > 0) moves it towards the chaser: retrograde must win
    assert o["direction"] == "retrograde" and o["dv_mps"] < 0
    assert o["miss_distance_after_m"] > o["miss_distance_before_m"]
    safe = P.manoeuvre_options(row, target_log_pc=0.0, burn_lead_orbits=(0.5,))[0]
    assert safe["abs_dv_mps"] == 0.0


def _pc_after(row, dv, tau, delta_r=0.0):
    """Independent re-evaluation: Pc with the target displaced by CW(dv) plus a radial offset."""
    df = pd.DataFrame([row])
    p, v = df[P.POS_COLS].to_numpy(float), df[P.VEL_COLS].to_numpy(float)
    _, C2, B = P.encounter_plane(p, v, P.combined_covariance(df))
    hbr = P.full_pc(df, with_max=False).hbr_m.iloc[0]
    n = P.mean_motion(row["t_j2k_sma"])
    dv, delta_r = np.broadcast_arrays(np.atleast_1d(dv).astype(float), np.atleast_1d(delta_r).astype(float))
    disp = P.cw_displacement(dv, tau, n)
    disp[:, 0] += delta_r
    x2 = (p - disp) @ B[0].T
    return P.log10_pc_2d(x2, np.repeat(C2, len(x2), axis=0), hbr, floor=-np.inf)


def test_manoeuvre_robust_to_execution_error():
    row = _head_on_row(miss=(10.0, 0.0, 0.0))
    nom = P.manoeuvre_options(row, target_log_pc=-6.0, burn_lead_orbits=(0.5,), robust=False)[0]
    rob = P.manoeuvre_options(row, target_log_pc=-6.0, burn_lead_orbits=(0.5,))[0]
    tau = rob["lead_time_s"]
    assert rob["robust"] and rob["design_log_pc"] == pytest.approx(-6.0 - P.MODEL_MARGIN_DECADES)
    assert rob["abs_dv_nominal_mps"] == pytest.approx(nom["abs_dv_mps"], rel=1e-6)
    assert rob["abs_dv_mps"] > 1.05 * nom["abs_dv_mps"]
    # the non-robust burn fails when 10 % short; the robust one does not (independent re-evaluation)
    assert _pc_after(row, 0.9 * nom["dv_mps"], tau)[0] > -6.0 + 0.1
    lp_s, lp_n, lp_l = _pc_after(row, np.array([0.9, 1.0, 1.1]) * rob["dv_mps"], tau)
    assert lp_s == pytest.approx(rob["log_pc_after_short"], abs=1e-9)
    assert lp_n == pytest.approx(rob["log_pc_after"], abs=1e-9)
    assert lp_l == pytest.approx(rob["log_pc_after_long"], abs=1e-9)
    assert max(lp_s, lp_n, lp_l) <= rob["design_log_pc"] + 1e-6
    assert rob["log_pc_worst"] == pytest.approx(rob["design_log_pc"], abs=1e-4)


def _crossing_row(miss=(10.0, 0.0, 40.0), ecc=0.01):
    row = _head_on_row(miss)
    row.update({"relative_velocity_r": 0.0, "relative_velocity_t": -3000.0, "relative_velocity_n": 9000.0,
                "t_sigma_r": 8.0, "c_sigma_r": 8.0, "t_j2k_ecc": ecc})
    return row


def test_manoeuvre_eccentricity_worst_case_brute_force():
    """Integer-orbit lead on a crossing geometry: CW gives dR = 0, but on an eccentric orbit the
    shifted target sits up to e|dT| higher or lower. The robust burn must stay below the design Pc
    for every radial offset in that interval and every burn error in +-10 % (dense brute force)."""
    row = _crossing_row()
    rob = P.manoeuvre_options(row, target_log_pc=-5.0, burn_lead_orbits=(1.0,), max_dv=1.0)[0]
    circ = P.manoeuvre_options(row, target_log_pc=-5.0, burn_lead_orbits=(1.0,), max_dv=1.0, eccentricity=0.0)[0]
    assert rob["eccentricity"] == 0.01 and rob["eccentricity_source"] == "t_j2k_ecc"
    assert rob["abs_dv_mps"] > circ["abs_dv_mps"]
    e_dT = rob["ecc_radial_uncertainty_m"]
    assert e_dT == pytest.approx(0.01 * abs(rob["d_target_rtn_m"][1]), rel=1e-9) and e_dT > 10.0
    assert rob["eccentricity_sensitive"]                 # e|dT| > sigma_minor / 3
    tau = rob["lead_time_s"]
    S, T = np.meshgrid(np.linspace(0.9, 1.1, 41), np.linspace(-1.0, 1.0, 81))
    S, T = S.ravel(), T.ravel()
    brute = _pc_after(row, S * rob["dv_mps"], tau, T * S * e_dT)
    assert brute.max() <= rob["log_pc_worst"] + 1e-4
    assert 0.1 < abs(T[brute.argmax()]) < 0.95           # worst offset is interior (radial miss -> 0)
    assert rob["log_pc_worst"] <= rob["design_log_pc"] + 1e-6
    # the nominal (non-robust) burn has offsets that break the target
    nom = P.manoeuvre_options(row, target_log_pc=-5.0, burn_lead_orbits=(1.0,), max_dv=1.0, robust=False)[0]
    e_nom = 0.01 * abs(nom["d_target_rtn_m"][1])
    worst_nom = _pc_after(row, nom["dv_mps"], tau, np.linspace(-e_nom, e_nom, 201)).max()
    assert worst_nom > -5.0 + 0.1


def test_manoeuvre_lead_time_feasibility():
    row = _head_on_row(miss=(10.0, 0.0, 0.0))
    row["time_to_tca"] = 0.06                            # 86.4 min; half an orbit at 7000 km = 48.6 min
    opts = P.manoeuvre_options(row, target_log_pc=-6.0, burn_lead_orbits=(0.5, 1.0))
    assert all(o["feasible"] is False and o["dv_mps"] is None and "planning buffer" in o["reason"] for o in opts)
    opts = P.manoeuvre_options(row, target_log_pc=-6.0, burn_lead_orbits=(0.5, 1.0), planning_buffer_s=0.0)
    assert opts[0]["feasible"] and opts[0]["abs_dv_mps"] > 0
    assert opts[1]["feasible"] is False
    assert P.best_manoeuvre(opts) is opts[0]
    # explicit override; no time_to_tca at all -> no check
    assert P.manoeuvre_options(row, burn_lead_orbits=(0.5,), time_to_tca_days=10.0)[0]["feasible"]
    del row["time_to_tca"]
    assert P.manoeuvre_options(row, burn_lead_orbits=(0.5,))[0]["time_to_tca_s"] is None


def test_best_manoeuvre_picks_smallest_feasible_burn():
    row = _crossing_row()
    opts = P.manoeuvre_options(row, target_log_pc=-5.0, burn_lead_orbits=(0.5, 1.0, 2.0), max_dv=1.0)
    feas = [o for o in opts if o is not None and o["feasible"]]
    best = P.best_manoeuvre(opts)
    assert best["abs_dv_mps"] == min(o["abs_dv_mps"] for o in feas)
    assert P.best_manoeuvre([None]) is None


# ------------------------------------------------------------------ real data sample (optional)
DATA = ROOT / "data" / "test_data.csv"


@pytest.mark.skipif(not DATA.exists(), reason="dataset not present")
def test_reproduces_esa_on_sample():
    df = pd.read_csv(DATA, nrows=3000)
    out = P.full_pc(df)
    ok = out.pc_valid & (df.risk > -30)
    m3 = np.abs(np.log10(out.mahal_3d[ok] / df.mahalanobis_distance[ok]))
    assert np.median(m3) < 1e-3
    err = (out.log_pc_phys[ok] - df.risk[ok]).abs()
    assert err.median() < 0.01 and (err < 0.1).mean() > 0.97
    # ESA's max_risk_scaling = m_2D^2 / 2, max_risk_estimate = Pc there (reproduced by construction
    # with the implied radius)
    ks = np.abs(np.log10(out.max_scale_esa[ok] / df.max_risk_scaling[ok]))
    assert ks.median() < 1e-3 and (ks < np.log10(1.01)).mean() > 0.98
    impl = ok & (out.hbr_source == "implied")
    assert (out.log_pc_max_esa[impl] - df.max_risk_estimate[impl]).abs().max() < 1e-3
    assert (out.log_pc_max_phys[ok] >= out.log_pc_max_esa[ok] - 1e-6).all()
    assert set(["log_pc_phys", "log_pc_max_phys", "hbr_m", "pc_valid", "mahal_3d", "mahal_2d",
                "miss_plane_x", "miss_plane_y", "sigma_major", "sigma_minor"]) <= set(out.columns)
