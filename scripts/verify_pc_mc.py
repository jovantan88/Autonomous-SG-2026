#!/usr/bin/env python
"""Independent Monte-Carlo verification of the 2D collision probability in src/physics.py.

Nothing from physics.py is reused except calling physics.full_pc() to obtain the numbers under test.
Everything else is re-implemented here from the raw CDM columns:

  * covariance assembly from sigmas + correlation coefficients (checked against *_position_covariance_det)
  * chaser-frame handling, four variants:
        a_unrotated  chaser covariance taken as already in the target RTN frame
        b_circular   chaser covariance rotated from the chaser RTN frame, target speed sqrt(mu/a_t)
        b_energy     same, target speed from the orbital-energy difference (claimed ESA convention)
        b_geo        same, but the chaser/target plane angle comes from spherical trigonometry on the
                     two J2000 inclinations and geocentric_latitude (heading sin A = cos i / cos lat),
                     i.e. WITHOUT using either semi-major axis or the energy relation
  * encounter plane: basis-free test |x - (x.v)v| <= HBR on 3D samples, own 2D basis elsewhere
  * Pc estimators:
        plain 3D MC         x ~ N(r, C) in 3D, count samples whose projection lies inside the disc
        3D importance samp. x ~ N(x_mp, C), x_mp = most probable 3D point of the collision cylinder,
                            exact 3D density-ratio weights (bounded inside the cylinder)
        2D adaptive quad    scipy nquad on the projected Gaussian over the disc (reference value)
        full-set 2D IS      defensive mixture (uniform-in-disc + shifted Gaussian), every test row
  * max-Pc: brute-force scan of the covariance scale factor k on a 0.01-decade grid (then refined),
    polar Gauss-Legendre x trapezoid quadrature, compared with max_risk_estimate / max_risk_scaling,
    and the hard-body radius implied by max_risk_estimate solved by fixed-point iteration.

Run:  python scripts/verify_pc_mc.py [--out results.json] [--workers 7] [--quick] [--no-fullset]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import integrate, optimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MU = 3.986004418e14
PLACEHOLDER = 6.0e7
POS = ["relative_position_r", "relative_position_t", "relative_position_n"]
VEL = ["relative_velocity_r", "relative_velocity_t", "relative_velocity_n"]
LN10 = np.log(10.0)
CHUNK = 2_000_000


# ============================================================ covariance & frames (own code)
def valid_cov_mask(df):
    sig = df[[f"{p}_sigma_{a}" for p in "tc" for a in "rtn"]].to_numpy(float)
    cor = df[[f"{p}_{k}" for p in "tc" for k in ("ct_r", "cn_r", "cn_t")]].to_numpy(float)
    return np.isfinite(sig).all(1) & np.isfinite(cor).all(1) & (sig < PLACEHOLDER).all(1)


def rtn_cov(df, p):
    """Covariance from sigmas and CCSDS lower-triangle correlations CT_R, CN_R, CN_T."""
    sd = np.stack([df[f"{p}_sigma_{a}"].to_numpy(float) for a in "rtn"], 1)
    n = len(df)
    corr = np.tile(np.eye(3), (n, 1, 1))
    for (i, j), k in {(1, 0): "ct_r", (2, 0): "cn_r", (2, 1): "cn_t"}.items():
        corr[:, i, j] = corr[:, j, i] = df[f"{p}_{k}"].to_numpy(float)
    return corr * sd[:, :, None] * sd[:, None, :]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def target_speed(df, method):
    a_t = df["t_j2k_sma"].to_numpy(float) * 1e3
    vc = np.sqrt(MU / a_t)
    if method == "circular":
        return vc, np.zeros(len(df), bool)
    dv = df[VEL].to_numpy(float)
    a_c = df["c_j2k_sma"].to_numpy(float) * 1e3
    # |v_t + dv|^2 - |v_t|^2 = mu (1/a_t - 1/a_c)  at (almost) common radius, v_t = (0, vT, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ve = (MU * (1 / a_t - 1 / a_c) - (dv * dv).sum(1)) / (2 * dv[:, 1])
    ok = np.isfinite(ve) & (np.abs(dv[:, 1]) > 100.0) & (np.abs(ve / vc - 1) < 0.05)
    return np.where(ok, ve, vc), ~ok


def geo_theta(df):
    """Angle (rad) of the chaser's horizontal velocity direction in the target's T-N plane, from
    the two inclinations and the geocentric latitude only. Heading A from north: sin A = cos i / cos lat
    (ascending A, descending pi - A). In target RTN the chaser heading direction is (0, cos th, sin th)
    with th = A_t - A_c. The ascending/descending combination is chosen as the one whose implied target
    speed vT = dv_n cot th - dv_t is closest to the circular speed (and |v_c| = dv_n / sin th > 0)."""
    lat = np.radians(df["geocentric_latitude"].to_numpy(float))
    it = np.radians(df["t_j2k_inc"].to_numpy(float))
    ic = np.radians(df["c_j2k_inc"].to_numpy(float))
    st = np.arcsin(np.clip(np.cos(it) / np.cos(lat), -1, 1))
    sc = np.arcsin(np.clip(np.cos(ic) / np.cos(lat), -1, 1))
    dv = df[VEL].to_numpy(float)
    vc = np.sqrt(MU / (df["t_j2k_sma"].to_numpy(float) * 1e3))
    best_th = np.full(len(df), np.nan)
    best_sc = np.full(len(df), np.inf)
    for At in (st, np.pi - st):
        for Ac in (sc, np.pi - sc):
            th = At - Ac
            with np.errstate(divide="ignore", invalid="ignore"):
                vT = dv[:, 2] / np.tan(th) - dv[:, 1]
                vcm = dv[:, 2] / np.sin(th)
            score = np.where(np.isfinite(vT) & (vcm > 0), np.abs(vT / vc - 1), np.inf)
            upd = score < best_sc
            best_th[upd], best_sc[upd] = th[upd], score[upd]
    return best_th, best_sc


def chaser_axes(df, method):
    """(n,3,3), columns = chaser R, T, N unit vectors in target-RTN coordinates."""
    p = df[POS].to_numpy(float)
    v = df[VEL].to_numpy(float)
    a_t = df["t_j2k_sma"].to_numpy(float) * 1e3
    r_c = p.copy()
    r_c[:, 0] += a_t
    info = {}
    if method in ("circular", "energy"):
        vT, fb = target_speed(df, method)
        h = v.copy()
        h[:, 1] += vT
        info["fallback"] = fb
        info["theta"] = np.arctan2(h[:, 2], h[:, 1])
    elif method == "geo":
        th, score = geo_theta(df)
        h = np.stack([np.zeros_like(th), np.cos(th), np.sin(th)], 1)
        info["theta"] = th
        info["score"] = score
    else:
        raise ValueError(method)
    Rr = unit(r_c)
    Nn = unit(np.cross(r_c, h))
    Tt = np.cross(Nn, Rr)
    return np.stack([Rr, Tt, Nn], -1), info


def combined_covs(df):
    Ct, Cc = rtn_cov(df, "t"), rtn_cov(df, "c")
    out, infos = {"a_unrotated": Ct + Cc}, {}
    for m in ("circular", "energy", "geo"):
        M, info = chaser_axes(df, m)
        out[f"b_{m}"] = Ct + np.einsum("nij,njk,nlk->nil", M, Cc, M)
        infos[m] = info
    return out, infos


def plane_basis(v):
    """(n,2,3) orthonormal basis of the plane perpendicular to v (own construction: start from the
    coordinate axis least aligned with v)."""
    vh = unit(np.atleast_2d(v))
    a = np.eye(3)[np.argmin(np.abs(vh), axis=1)]
    e1 = unit(a - (a * vh).sum(1, keepdims=True) * vh)
    e2 = np.cross(vh, e1)
    return np.stack([e1, e2], 1)


def project(p, v, C):
    B = plane_basis(v)
    m2 = np.einsum("nij,nj->ni", B, p)
    C2 = np.einsum("nij,njk,nlk->nil", B, C, B)
    return m2, C2


def mahal(d, C):
    sol = np.linalg.solve(C, d[..., None])[..., 0]
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.einsum("ni,ni->n", d, sol))


# ============================================================ Pc estimators (own code)
def mc_plain_3d(r, v, C, R, N, seed):
    """Plain MC: x ~ N(r, C); hit if the projection of x on the plane perpendicular to v is within R."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(C)
    vh = v / np.linalg.norm(v)
    hits = done = 0
    while done < N:
        n = int(min(CHUNK, N - done))
        x = rng.standard_normal((n, 3)) @ L.T
        x += r
        s = x @ vh
        hits += np.count_nonzero(np.einsum("ij,ij->i", x, x) - s * s <= R * R)
        done += n
    p = hits / N
    return {"pc": p, "se": np.sqrt(max(p * (1 - p), 1e-300) / N), "hits": int(hits), "n": int(N)}


def most_probable_point_3d(r, v, C, R):
    """argmin (x-r)^T C^-1 (x-r) over the collision cylinder |P x| <= R (P = projector perp. to v)."""
    Ci = np.linalg.inv(C)
    vh = v / np.linalg.norm(v)
    Pr = r - (r @ vh) * vh
    if np.linalg.norm(Pr) <= R:
        return r.copy()
    e1, e2 = plane_basis(v)[0]
    a = vh @ Ci @ vh

    def pt(phi):
        phi = np.atleast_1d(phi)
        u = R * (np.cos(phi)[..., None] * e1 + np.sin(phi)[..., None] * e2)
        t = ((r - u) @ Ci @ vh) / a
        return u + t[..., None] * vh

    def q(phi):
        d = pt(phi) - r
        return np.einsum("...i,ij,...j->...", d, Ci, d)

    ph = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    j = int(np.argmin(q(ph)))
    h = 2 * np.pi / 4096
    res = optimize.minimize_scalar(lambda x: float(q(x)[0]), bounds=(ph[j] - 2 * h, ph[j] + 2 * h),
                                   method="bounded", options={"xatol": 1e-12})
    return pt(res.x)[0]


def mc_is_3d(r, v, C, R, N, seed):
    """3D importance sampling: proposal N(x_mp, C). log w = -(x - x_mp).C^-1 delta - delta.C^-1 delta / 2,
    delta = x_mp - r; inside the (convex) cylinder the weight is <= exp(-delta.C^-1 delta / 2)."""
    rng = np.random.default_rng(seed)
    Ci = np.linalg.inv(C)
    L = np.linalg.cholesky(C)
    vh = v / np.linalg.norm(v)
    xmp = most_probable_point_3d(r, v, C, R)
    delta = xmp - r
    g = Ci @ delta
    q0 = float(delta @ g)
    s1 = s2 = 0.0
    hits = done = 0
    while done < N:
        n = int(min(CHUNK, N - done))
        u = rng.standard_normal((n, 3)) @ L.T
        x = u + xmp
        s = x @ vh
        hit = np.einsum("ij,ij->i", x, x) - s * s <= R * R
        w = np.exp(-(u[hit] @ g))           # scaled by exp(+q0/2)
        s1 += w.sum()
        s2 += (w * w).sum()
        hits += int(hit.sum())
        done += n
    mean = s1 / N
    var = max(s2 / N - mean * mean, 0.0)
    rel = np.sqrt(var / N) / mean if mean > 0 else np.inf
    lp = (np.log(mean) - 0.5 * q0) / LN10 if mean > 0 else -np.inf
    return {"log10_pc": lp, "rel_se": rel, "hits": hits, "n": int(N), "mahal_mp": np.sqrt(q0)}


def pc_quad2d(m2, C2, R):
    """Adaptive 2D quadrature (scipy nquad) of the projected Gaussian over the disc, in its principal
    frame; integrand scaled by exp(+q_min/2) so deep tails don't underflow. Returns ln Pc."""
    ev, V = np.linalg.eigh(C2)
    sw, su = np.sqrt(ev)                      # minor, major
    mu, mw = m2 @ V[:, 1], m2 @ V[:, 0]
    rr = np.linspace(0, R, 401)[:, None]
    pp = np.linspace(0, 2 * np.pi, 1441)[None, :]
    qg = ((rr * np.cos(pp) - mu) / su) ** 2 + ((rr * np.sin(pp) - mw) / sw) ** 2
    jmin = np.unravel_index(np.argmin(qg), qg.shape)
    umin = float(rr[jmin[0], 0] * np.cos(pp[0, jmin[1]]))
    c = 0.5 * float(qg.min())
    f = lambda w, u: np.exp(-0.5 * (((u - mu) / su) ** 2 + ((w - mw) / sw) ** 2) + c)  # noqa: E731
    h = lambda u: np.sqrt(max(R * R - u * u, 0.0))                                      # noqa: E731
    inner = lambda u: {"points": [float(np.clip(mw, -h(u), h(u)))], "epsabs": 0, "epsrel": 1e-11,  # noqa: E731
                       "limit": 400}
    outer = {"points": [float(np.clip(mu, -R, R)), umin], "epsabs": 0, "epsrel": 1e-10, "limit": 400}
    I, err = integrate.nquad(f, [lambda u: [-h(u), h(u)], [-R, R]], opts=[inner, outer])
    return np.log(I) - c - np.log(2 * np.pi * su * sw), err / I


def polar_nodes(R, nr=96, nphi=192):
    t, w = np.polynomial.legendre.leggauss(nr)
    rho = R * (t + 1) / 2
    wr = w * R / 2
    phi = 2 * np.pi * np.arange(nphi) / nphi
    Y = np.stack([(rho[:, None] * np.cos(phi)[None, :]).ravel(), (rho[:, None] * np.sin(phi)[None, :]).ravel()], 1)
    W = np.repeat(wr * rho, nphi) * (2 * np.pi / nphi)
    return Y, np.log(W)


def lnpc_vs_scale(m2, C2, R, ks, nodes=None):
    """ln Pc for covariance k*C2, vectorised over k (polar GL x trapezoid on the disc)."""
    Y, lnW = nodes if nodes is not None else polar_nodes(R)
    A = np.linalg.inv(C2)
    d = Y - m2
    q0 = np.einsum("ni,ij,nj->n", d, A, d)
    lnc = -np.log(2 * np.pi * np.sqrt(np.linalg.det(C2)))
    out = np.empty(len(ks))
    for s in range(0, len(ks), 256):
        k = ks[s:s + 256, None]
        T = -q0[None, :] / (2 * k) + lnW[None, :]
        mx = T.max(1, keepdims=True)
        out[s:s + 256] = (mx[:, 0] + np.log(np.exp(T - mx).sum(1))) + lnc - np.log(k[:, 0])
    return out


def max_pc_bruteforce(m2, C2, R):
    """Scan k on 10^[-14, 9] at 0.01 decade, then +-0.02 decade at 1e-4 decade. Returns (log10 Pcmax, k*)."""
    nodes = polar_nodes(R)
    ks = 10 ** np.arange(-14, 9.0001, 0.01)
    lp = lnpc_vs_scale(m2, C2, R, ks, nodes)
    j = int(np.nanargmax(lp))
    ks2 = ks[j] * 10 ** np.arange(-0.02, 0.02001, 1e-4)
    lp2 = lnpc_vs_scale(m2, C2, R, ks2, nodes)
    j2 = int(np.nanargmax(lp2))
    return lp2[j2] / LN10, ks2[j2], (j in (0, len(ks) - 1))


def implied_hbr_bruteforce(m2, C2, mre, R0, iters=14):
    R = R0
    hist = []
    for _ in range(iters):
        lpm, k, edge = max_pc_bruteforce(m2, C2, R)
        hist.append((R, lpm))
        if abs(lpm - mre) < 2e-5:
            break
        R = R * 10 ** ((mre - lpm) / 2)
    return R, lpm, k, hist


# ============================================================ full-set vectorised 2D IS
def fullset_is(m2, C2, R, N, seed, block=50):
    """Per-row 2D importance sampling with a defensive mixture proposal
    q = 0.5 U(disc) + 0.5 N(y_mp, C2), y_mp = most probable point of the disc. Returns log10 Pc, rel SE."""
    rng = np.random.default_rng(seed)
    n = len(m2)
    out = np.full(n, np.nan)
    rel = np.full(n, np.nan)
    A = np.linalg.inv(C2)
    L = np.linalg.cholesky(C2)
    ldet = np.log(np.linalg.det(C2))
    # most probable point on the disc (grid + 2 local refinements)
    inside = np.linalg.norm(m2, axis=1) <= R
    ymp = m2.copy()
    phi = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    out_i = np.flatnonzero(~inside)
    for s in range(0, len(out_i), 2000):
        j = out_i[s:s + 2000]
        cen = None
        for width, npt in ((2 * np.pi, 2048), (4 * np.pi / 2048, 201), (8 * np.pi / 2048 / 200, 201)):
            ph = phi[None, :] if cen is None else cen[:, None] + np.linspace(-width / 2, width / 2, npt)[None, :]
            Y = R[j, None, None] * np.stack([np.cos(ph), np.sin(ph)], -1) if ph.shape[0] > 1 else \
                R[j, None, None] * np.stack([np.cos(ph), np.sin(ph)], -1)
            d = Y - m2[j, None, :]
            q = np.einsum("nki,nij,nkj->nk", d, A[j], d)
            kk = np.argmin(q, 1)
            cen = np.take_along_axis(np.broadcast_to(ph, q.shape), kk[:, None], 1)[:, 0]
            best = np.take_along_axis(Y, kk[:, None, None], 1)[:, 0]
        ymp[j] = best
    for s in range(0, n, block):
        j = np.arange(s, min(s + block, n))
        b = len(j)
        comp = rng.random((b, N)) < 0.5
        # uniform-in-disc samples
        rad = R[j, None] * np.sqrt(rng.random((b, N)))
        ang = 2 * np.pi * rng.random((b, N))
        yu = np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)
        # shifted-Gaussian samples
        yg = ymp[j, None, :] + np.einsum("nij,nkj->nki", L[j], rng.standard_normal((b, N, 2)))
        y = np.where(comp[..., None], yu, yg)
        hit = (y * y).sum(-1) <= (R[j] ** 2)[:, None]
        dp = y - m2[j, None, :]
        dq = y - ymp[j, None, :]
        qp = np.einsum("nki,nij,nkj->nk", dp, A[j], dp)
        qq = np.einsum("nki,nij,nkj->nk", dq, A[j], dq)
        lnorm = -np.log(2 * np.pi) - 0.5 * ldet[j, None]
        lp = lnorm - 0.5 * qp
        lq_g = lnorm - 0.5 * qq
        lq_u = -np.log(np.pi * R[j] ** 2)[:, None]
        lq = np.logaddexp(np.log(0.5) + lq_u, np.log(0.5) + lq_g)
        lw = np.where(hit, lp - lq, -np.inf)
        mx = np.max(lw, 1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        w = np.exp(lw - mx)
        mean = w.mean(1)
        sd = w.std(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[j] = (np.log(mean) + mx[:, 0]) / LN10
            rel[j] = sd / mean / np.sqrt(N)
    return out, rel


# ============================================================ worker dispatch
def run_task(t):
    kind = t["kind"]
    t0 = time.time()
    if kind == "plain":
        res = mc_plain_3d(t["r"], t["v"], t["C"], t["R"], t["N"], t["seed"])
        res["log10_pc"] = np.log10(res["pc"]) if res["pc"] > 0 else -np.inf
        res["rel_se"] = res["se"] / res["pc"] if res["pc"] > 0 else np.inf
    elif kind == "is":
        pilot = mc_is_3d(t["r"], t["v"], t["C"], t["R"], t["N_pilot"], t["seed"] + 999_983)
        f = max(pilot["hits"], 1) / t["N_pilot"]
        N = int(np.clip(t["target_hits"] / f, t["N"], t["N_max"]))
        res = mc_is_3d(t["r"], t["v"], t["C"], t["R"], N, t["seed"])
    elif kind == "is_fixed":
        res = mc_is_3d(t["r"], t["v"], t["C"], t["R"], t["N"], t["seed"])
    elif kind == "is2d":
        lp, rel = fullset_is(t["m2"][None], t["C2"][None], np.array([t["R"]]), t["N"], t["seed"], block=1)
        res = {"log10_pc": float(lp[0]), "rel_se": float(rel[0]), "n": int(t["N"])}
    elif kind == "maxpc":
        m2, C2 = t["m2"], t["C2"]
        lp_sp, k_sp, edge = max_pc_bruteforce(m2, C2, t["R"])
        Rimp, lp_imp, k_imp, hist = implied_hbr_bruteforce(m2, C2, t["mre"], t["R"])
        # cross-check the polar-GL value at the optimum with adaptive quadrature
        lq, _ = pc_quad2d(m2, k_sp * C2, t["R"])
        res = {"log10_pcmax_spans": lp_sp, "kstar_spans": k_sp, "edge": edge, "R_implied": Rimp,
               "log10_pcmax_implied": lp_imp, "kstar_implied": k_imp, "iters": len(hist),
               "quad_at_kstar": lq / LN10}
    else:
        raise ValueError(kind)
    res["key"] = t["key"]
    res["secs"] = time.time() - t0
    return res


# ============================================================ helpers
def stats(d, z=None):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {}
    o = {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
         "median_abs": float(np.median(np.abs(d))), "max_abs": float(np.abs(d).max()),
         "within_0.01": float(np.mean(np.abs(d) < 0.01)), "within_0.1": float(np.mean(np.abs(d) < 0.1)),
         "sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None}
    if z is not None:
        z = np.asarray(z, float)
        z = z[np.isfinite(z)]
        o.update({"z_mean": float(z.mean()), "z_rms": float(np.sqrt((z * z).mean())),
                  "frac_abs_z_gt_3": float(np.mean(np.abs(z) > 3)), "max_abs_z": float(np.abs(z).max())})
    return o


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def select_rows(df, valid, my, rng, per_bin=3):
    risk = df["risk"].to_numpy(float)
    ev = df["event_id"].to_numpy()
    sel, why = [], []
    for lo in range(-12, -2):
        cand = rng.permutation(np.flatnonzero(valid & (risk >= lo) & (risk < lo + 1)))
        seen, k = set(), 0
        for i in cand:
            if ev[i] in seen:
                continue
            seen.add(ev[i])
            sel.append(int(i))
            why.append(f"bin[{lo},{lo + 1})")
            k += 1
            if k == per_bin:
                break
    rng_mask = valid & (risk >= -12) & (risk <= -2)
    stress = {
        "max HBR/sigma_minor": np.where(rng_mask, my["hbr_spans"] / my["s_minor"], -np.inf),
        "min relative_speed": np.where(rng_mask, -df["relative_speed"].to_numpy(float), -np.inf),
        "max sigma_major/sigma_minor": np.where(rng_mask, my["s_major"] / my["s_minor"], -np.inf),
        "max |frame a - frame b| Mahalanobis": np.where(rng_mask, np.abs(np.log10(my["mahal_a"] / my["mahal_b"])), -np.inf),
        "implied HBR far from spans": np.where(rng_mask, np.abs(np.log(my["phys_hbr"] / my["hbr_spans"])), -np.inf),
        "max |risk - spans Pc| (physics)": np.where(rng_mask & np.isfinite(my["phys_spans"]),
                                                     np.abs(my["phys_spans"] - risk), -np.inf),
    }
    for name, score in stress.items():
        for i in np.argsort(-score):
            if int(i) not in sel and np.isfinite(score[i]):
                sel.append(int(i))
                why.append("stress: " + name)
                break
    return sel, why


# ============================================================ high-statistics 3D-vs-2D bias test
BIAS_ROWS_PLAIN = (9933, 9504, 17716)     # test rows with Pc ~ 2e-3..4e-3 (plain MC hit rate high)
BIAS_ROWS_IS = (119, 9257, 17708)         # test rows where the 3D IS hit rate is >= 1e-3


def bias_check(args):
    """Pure-3D estimators (plain MC, 3D IS) vs the 2D nquad integral at ~0.03-0.06 % precision, energy
    frame, physics' auto HBR: a real projection/3D effect of the size hinted at by the main run
    (~+0.2 %) would show up at > 4 sigma."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import physics
    df = pd.read_csv(os.path.join(ROOT, "data", "test_data.csv"))
    rows = list(BIAS_ROWS_PLAIN) + list(BIAS_ROWS_IS)
    S = df.iloc[rows].reset_index(drop=True)
    R = physics.full_pc(S, hbr="auto")["hbr_m"].to_numpy(float)
    covS, _ = combined_covs(S)
    C = covS["b_energy"]
    pS, vS = S[POS].to_numpy(float), S[VEL].to_numpy(float)
    m2, C2 = project(pS, vS, C)
    ref = np.array([pc_quad2d(m2[i], C2[i], R[i])[0] / LN10 for i in range(len(S))])
    n_rep, n_each = 7, 200_000_000
    tasks = []
    for i, row in enumerate(rows):
        kind = "plain" if row in BIAS_ROWS_PLAIN else "is_fixed"
        for k in range(n_rep):
            tasks.append({"kind": kind, "key": (i, k), "r": pS[i], "v": vS[i], "C": C[i], "R": float(R[i]),
                          "N": n_each, "seed": args.seed + 5_000_003 + 101 * i + k})
    with Pool(args.workers) as pool:
        res = pool.map(run_task, tasks, chunksize=1)
    out = []
    for i, row in enumerate(rows):
        rs = [r for t, r in zip(tasks, res) if t["key"][0] == i]
        if row in BIAS_ROWS_PLAIN:
            hits = sum(r["hits"] for r in rs)
            N = sum(r["n"] for r in rs)
            pc = hits / N
            rel = np.sqrt((1 - pc) / hits)
        else:   # combine replicate means (equal N)
            vals = np.array([10 ** r["log10_pc"] for r in rs])
            pc = vals.mean()
            rel = vals.std(ddof=1) / np.sqrt(len(vals)) / pc
            hits = sum(r["hits"] for r in rs)
            N = sum(r["n"] for r in rs)
        d = np.log10(pc) - ref[i]
        out.append({"row": row, "estimator": "plain3D" if row in BIAS_ROWS_PLAIN else "IS3D", "n": N, "hits": hits,
                    "log10_pc_mc": np.log10(pc), "log10_pc_quad": ref[i], "d_decades": d, "rel_se": rel,
                    "z": (1 - 10 ** (-d)) / rel})
    O = pd.DataFrame(out)
    print(O.to_string(float_format=lambda x: f"{x:.6g}"))
    z = O.z.to_numpy()
    print(f"pooled z mean {z.mean():.3f} (se {1 / np.sqrt(len(z)):.3f}), rms {np.sqrt((z * z).mean()):.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(jsonable({"rows": out, "z_mean": z.mean(), "z_rms": np.sqrt((z * z).mean())}), f, indent=1)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the JSON summary here")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="small sample sizes (smoke test)")
    ap.add_argument("--no-fullset", action="store_true")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--bias-check", action="store_true",
                    help="only run the high-statistics 3D-vs-2D bias test (fresh seeds, ~1.4e9 samples per row)")
    args = ap.parse_args()
    if args.bias_check:
        return bias_check(args)
    T0 = time.time()
    rng = np.random.default_rng(args.seed)

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import physics  # only full_pc() is called: it is the thing under test

    df = pd.read_csv(os.path.join(ROOT, "data", "test_data.csv"))
    valid = valid_cov_mask(df)
    p = df[POS].to_numpy(float)
    v = df[VEL].to_numpy(float)
    report = {"n_test": len(df), "n_valid_cov": int(valid.sum())}

    # ---------------- full-set checks of the covariance construction (vectorised, own code)
    dv = df.loc[valid]
    Ct, Cc = rtn_cov(dv, "t"), rtn_cov(dv, "c")
    det_chk = {}
    for nm, C in (("t", Ct), ("c", Cc)):
        ref = dv[f"{nm}_position_covariance_det"].to_numpy(float)
        dd = np.log10(np.linalg.det(C) / ref)
        det_chk[nm] = {"median_abs_log10": float(np.nanmedian(np.abs(dd))), "p99_abs": float(np.nanpercentile(np.abs(dd), 99)),
                       "max_abs": float(np.nanmax(np.abs(dd)))}
    # alternative reading: correlations used as raw covariances
    Cc_raw = np.tile(np.eye(3), (len(dv), 1, 1)) * 0
    sd = np.stack([dv[f"c_sigma_{a}"].to_numpy(float) for a in "rtn"], 1)
    for i in range(3):
        Cc_raw[:, i, i] = sd[:, i] ** 2
    for (i, j), k in {(1, 0): "ct_r", (2, 0): "cn_r", (2, 1): "cn_t"}.items():
        Cc_raw[:, i, j] = Cc_raw[:, j, i] = dv[f"c_{k}"].to_numpy(float)
    det_chk["c_if_columns_were_covariances"] = {"median_abs_log10": float(np.nanmedian(np.abs(np.log10(
        np.linalg.det(Cc_raw) / dv["c_position_covariance_det"].to_numpy(float)))))}
    report["determinant_check"] = det_chk

    covs, infos = combined_covs(dv)
    esa_m = dv["mahalanobis_distance"].to_numpy(float)
    mah = {}
    for nm, C in covs.items():
        m3 = mahal(dv[POS].to_numpy(float), C)
        m2, C2 = project(dv[POS].to_numpy(float), dv[VEL].to_numpy(float), C)
        mm2 = mahal(m2, C2)
        e3 = np.abs(np.log10(m3 / esa_m))
        e2 = np.abs(np.log10(mm2 / esa_m))
        mah[nm] = {"3d_median_abs_log10": float(np.nanmedian(e3)), "3d_within_1pct": float(np.nanmean(e3 < np.log10(1.01))),
                   "3d_within_0.1pct": float(np.nanmean(e3 < np.log10(1.001))),
                   "3d_p99_abs_log10": float(np.nanpercentile(e3, 99)),
                   "2d_median_abs_log10": float(np.nanmedian(e2))}
    th_e, th_g = infos["energy"]["theta"], infos["geo"]["theta"]
    dth = np.degrees(np.angle(np.exp(1j * (th_e - th_g))))
    th_c = infos["circular"]["theta"]
    mah["theta_energy_minus_geo_deg"] = {"median_abs": float(np.nanmedian(np.abs(dth))),
                                         "p90_abs": float(np.nanpercentile(np.abs(dth), 90)),
                                         "frac_abs_lt_0.1deg": float(np.nanmean(np.abs(dth) < 0.1))}
    mah["theta_circular_minus_geo_deg"] = {"median_abs": float(np.nanmedian(np.abs(np.degrees(np.angle(np.exp(1j * (th_c - th_g))))))),}
    mah["energy_fallback_rows"] = int(infos["energy"]["fallback"].sum())
    report["mahalanobis_frame_check"] = mah
    print("determinant check:", json.dumps(det_chk, indent=1))
    print("mahalanobis frame check:", json.dumps(mah, indent=1))

    # ---------------- physics under test (whole test set; selection uses it only for 2 stress rows)
    ph_auto = physics.full_pc(df, hbr="auto")
    ph_sp = physics.full_pc(df, hbr="spans", with_max=False)
    hbr_spans = (df["t_span"].to_numpy(float) + df["c_span"].to_numpy(float)) / 2

    # own projected quantities on the whole test set (energy frame) for row selection
    covs_all, _ = combined_covs(df.assign(**{c: df[c].fillna(1.0) for c in df.columns if c.startswith("c_")}))
    m2_all, C2_all = project(p, v, covs_all["b_energy"])
    ev2 = np.linalg.eigvalsh(C2_all)
    with np.errstate(invalid="ignore"):
        my = {"s_minor": np.sqrt(ev2[:, 0]), "s_major": np.sqrt(ev2[:, 1]), "hbr_spans": hbr_spans,
              "mahal_a": mahal(p, covs_all["a_unrotated"]), "mahal_b": mahal(p, covs_all["b_energy"]),
              "phys_hbr": ph_auto["hbr_m"].to_numpy(float), "phys_spans": ph_sp["log_pc_phys"].to_numpy(float)}

    sel, why = select_rows(df, valid, my, rng)
    S = df.iloc[sel].reset_index(drop=True)
    covS, infS = combined_covs(S)
    pS, vS = S[POS].to_numpy(float), S[VEL].to_numpy(float)
    R_auto = ph_auto["hbr_m"].to_numpy(float)[sel]
    R_sp = hbr_spans[sel]
    print(f"\nselected {len(sel)} rows; building tasks ...")

    # ---------------- deterministic 2D reference (own nquad) for every selected row / config
    configs = {"b_auto": ("b_energy", R_auto), "b_spans": ("b_energy", R_sp), "a_spans": ("a_unrotated", R_sp),
               "a_auto": ("a_unrotated", R_auto), "bgeo_auto": ("b_geo", R_auto), "bcirc_auto": ("b_circular", R_auto)}
    quad = {c: np.full(len(S), np.nan) for c in configs}
    quad_err = {c: np.full(len(S), np.nan) for c in configs}
    m2S, C2S = {}, {}
    for c, (fr, Rv) in configs.items():
        m2, C2 = project(pS, vS, covS[fr])
        m2S[c], C2S[c] = m2, C2
        for i in range(len(S)):
            lq, er = pc_quad2d(m2[i], C2[i], Rv[i])
            quad[c][i], quad_err[c][i] = lq / LN10, er

    # ---------------- Monte Carlo tasks
    N_is = 400_000 if args.quick else 20_000_000
    N_is_max = 4_000_000 if args.quick else 400_000_000
    N_is2d = 200_000 if args.quick else 4_000_000
    N_plain_max = 20_000_000 if args.quick else 400_000_000
    tasks = []
    for i in range(len(S)):
        for c in ("b_auto", "b_spans", "a_spans"):
            fr, Rv = configs[c]
            sd_ = args.seed + 1000 * i + {"b_auto": 1, "b_spans": 2, "a_spans": 3}[c]
            tasks.append({"kind": "is", "key": (i, c), "r": pS[i], "v": vS[i], "C": covS[fr][i], "R": float(Rv[i]),
                          "N": N_is, "N_max": N_is_max, "N_pilot": 2_000_000 if not args.quick else 200_000,
                          "target_hits": 20000, "seed": sd_})
            tasks.append({"kind": "is2d", "key": (i, c), "m2": m2S[c][i], "C2": C2S[c][i], "R": float(Rv[i]),
                          "N": N_is2d, "seed": sd_ + 500})
        pc_guess = 10 ** quad["b_auto"][i]
        if pc_guess >= 3e-7:
            N = int(np.clip(3e4 / pc_guess, 2e6, N_plain_max))
            tasks.append({"kind": "plain", "key": (i, "b_auto"), "r": pS[i], "v": vS[i], "C": covS["b_energy"][i],
                          "R": float(R_auto[i]), "N": N, "seed": args.seed + 7 + 1000 * i})
    mre = S["max_risk_estimate"].to_numpy(float)
    for i in range(len(S)):
        for c in ("b_spans", "a_spans"):
            tasks.append({"kind": "maxpc", "key": (i, c), "m2": m2S[c][i], "C2": C2S[c][i], "R": float(R_sp[i]),
                          "mre": float(mre[i])})
    # longest first
    tasks.sort(key=lambda t: -(t.get("N", 0) if t["kind"] == "plain" else (2e8 if t["kind"] == "is" else 1)))
    print(f"{len(tasks)} tasks on {args.workers} workers ...")
    t1 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_task, tasks, chunksize=1)
    print(f"MC done in {time.time() - t1:.0f} s")
    R_ = {}
    for t, r in zip(tasks, results):
        R_[(t["key"][0], t["key"][1], t["kind"])] = r

    # ---------------- per-row table
    rows = []
    for i in range(len(S)):
        g = lambda c, k, f: R_.get((i, c, k), {}).get(f, np.nan)  # noqa: E731
        rec = {
            "row": int(sel[i]), "event_id": int(S.event_id[i]), "why": why[i], "risk_esa": float(S.risk[i]),
            "mre_esa": float(mre[i]), "mrs_esa": float(S.max_risk_scaling[i]),
            "mahal_esa": float(S.mahalanobis_distance[i]), "miss_m": float(S.miss_distance[i]),
            "rel_speed": float(S.relative_speed[i]),
            "hbr_auto_phys": float(R_auto[i]), "hbr_spans": float(R_sp[i]),
            "hbr_source_phys": ph_auto["hbr_source"].iat[sel[i]],
            "sigma_minor_m": float(np.sqrt(np.linalg.eigvalsh(C2S["b_auto"][i])[0])),
            "sigma_major_m": float(np.sqrt(np.linalg.eigvalsh(C2S["b_auto"][i])[1])),
            "phys_auto": float(ph_auto.log_pc_phys.iat[sel[i]]), "phys_spans": float(ph_sp.log_pc_phys.iat[sel[i]]),
            "phys_max_auto": float(ph_auto.log_pc_max_phys.iat[sel[i]]), "phys_kmax": float(ph_auto.max_scale.iat[sel[i]]),
            "phys_mahal3d": float(ph_auto.mahal_3d.iat[sel[i]]),
        }
        for c in configs:
            rec[f"quad_{c}"] = float(quad[c][i])
        for c in ("b_auto", "b_spans", "a_spans"):
            rec[f"is_{c}"] = g(c, "is", "log10_pc")
            rec[f"is_{c}_relse"] = g(c, "is", "rel_se")
            rec[f"is_{c}_hits"] = g(c, "is", "hits")
            rec[f"is_{c}_n"] = g(c, "is", "n")
            rec[f"is2d_{c}"] = g(c, "is2d", "log10_pc")
            rec[f"is2d_{c}_relse"] = g(c, "is2d", "rel_se")
        rec["plain_b_auto"] = g("b_auto", "plain", "log10_pc")
        rec["plain_b_auto_relse"] = g("b_auto", "plain", "rel_se")
        rec["plain_b_auto_hits"] = g("b_auto", "plain", "hits")
        rec["plain_b_auto_n"] = g("b_auto", "plain", "n")
        for c in ("b_spans", "a_spans"):
            for f in ("log10_pcmax_spans", "kstar_spans", "R_implied", "kstar_implied", "quad_at_kstar", "edge"):
                rec[f"maxpc_{c}_{f}"] = g(c, "maxpc", f)
        rec["mahal3d_mine_b"] = float(mahal(pS[i:i + 1], covS["b_energy"][i:i + 1])[0])
        rec["mahal3d_mine_a"] = float(mahal(pS[i:i + 1], covS["a_unrotated"][i:i + 1])[0])
        rows.append(rec)
    T = pd.DataFrame(rows)

    # ---------------- agreement statistics on the selection
    def zdiff(lp_est, rel, lp_ref):
        d = lp_est - lp_ref
        z = (1 - 10 ** (-d)) / rel       # (Pc_est - Pc_ref) / se_est
        return d, z

    summ = {}
    d, z = zdiff(T.is_b_auto, T.is_b_auto_relse, T.phys_auto)
    summ["IS3D(b, auto HBR) - physics(auto)"] = stats(d, z)
    d, z = zdiff(T.is_b_spans, T.is_b_spans_relse, T.phys_spans)
    summ["IS3D(b, spans HBR) - physics(spans)"] = stats(d, z)
    mpl = np.isfinite(T.plain_b_auto)
    d, z = zdiff(T.plain_b_auto[mpl], T.plain_b_auto_relse[mpl], T.phys_auto[mpl])
    summ["plain3D(b, auto) - physics(auto)"] = stats(d, z)
    d, z = zdiff(T.plain_b_auto[mpl], T.plain_b_auto_relse[mpl], T.quad_b_auto[mpl])
    summ["plain3D(b, auto) - quad(b, auto)"] = stats(d, z)
    d, z = zdiff(T.is_b_auto, T.is_b_auto_relse, T.quad_b_auto)
    summ["IS3D(b, auto) - quad(b, auto)"] = stats(d, z)
    for c in ("b_auto", "b_spans", "a_spans"):
        d, z = zdiff(T[f"is2d_{c}"], T[f"is2d_{c}_relse"], T[f"quad_{c}"])
        summ[f"IS2D-mixture({c}) - quad({c})"] = stats(d, z)
    d, z = zdiff(T.is2d_b_auto, T.is2d_b_auto_relse, T.phys_auto)
    summ["IS2D-mixture(b, auto) - physics(auto)"] = stats(d, z)
    d, z = zdiff(T.is2d_b_spans, T.is2d_b_spans_relse, T.phys_spans)
    summ["IS2D-mixture(b, spans) - physics(spans)"] = stats(d, z)
    d, z = zdiff(T.is_a_spans, T.is_a_spans_relse, T.quad_a_spans)
    summ["IS3D(a, spans) - quad(a, spans)"] = stats(d, z)
    summ["quad(b, auto) - physics(auto)"] = stats(T.quad_b_auto - T.phys_auto)
    summ["quad(b, spans) - physics(spans)"] = stats(T.quad_b_spans - T.phys_spans)
    for c in ("b_auto", "b_spans", "a_spans", "a_auto", "bgeo_auto", "bcirc_auto"):
        summ[f"quad({c}) - ESA risk"] = stats(T[f"quad_{c}"] - T.risk_esa)
    summ["IS3D(b, auto) - ESA risk"] = stats(T.is_b_auto - T.risk_esa)
    summ["IS3D(a, spans) - ESA risk"] = stats(T.is_a_spans - T.risk_esa)
    summ["physics(auto) - ESA risk"] = stats(T.phys_auto - T.risk_esa)
    summ["physics(spans) - ESA risk"] = stats(T.phys_spans - T.risk_esa)
    # max-Pc checks
    summ["maxPc bruteforce(b, spans) - max_risk_estimate"] = stats(T.maxpc_b_spans_log10_pcmax_spans - T.mre_esa)
    summ["maxPc bruteforce(a, spans) - max_risk_estimate"] = stats(T.maxpc_a_spans_log10_pcmax_spans - T.mre_esa)
    summ["log10 k*(b, implied HBR) / max_risk_scaling"] = stats(np.log10(T.maxpc_b_spans_kstar_implied / T.mrs_esa))
    summ["log10 k*(b, spans HBR) / max_risk_scaling"] = stats(np.log10(T.maxpc_b_spans_kstar_spans / T.mrs_esa))
    summ["log10 sqrt(k*)(b, implied) / max_risk_scaling  [if scaling multiplied sigmas]"] = stats(
        np.log10(np.sqrt(T.maxpc_b_spans_kstar_implied) / T.mrs_esa))
    summ["log10 k*(a, implied HBR) / max_risk_scaling"] = stats(np.log10(T.maxpc_a_spans_kstar_implied / T.mrs_esa))
    summ["log10 k*(b, implied) / physics max_scale"] = stats(np.log10(T.maxpc_b_spans_kstar_implied / T.phys_kmax))
    summ["log10 R_implied(mine, b) / R_auto(physics)"] = stats(np.log10(T.maxpc_b_spans_R_implied / T.hbr_auto_phys))
    summ["log10 R_implied(mine, b) / spans"] = stats(np.log10(T.maxpc_b_spans_R_implied / T.hbr_spans))
    summ["polarGL maxPc - nquad at k*"] = stats(T.maxpc_b_spans_log10_pcmax_spans - T.maxpc_b_spans_quad_at_kstar)
    summ["log10 mahal3d(mine, b) / ESA"] = stats(np.log10(T.mahal3d_mine_b / T.mahal_esa))
    summ["log10 mahal3d(mine, a) / ESA"] = stats(np.log10(T.mahal3d_mine_a / T.mahal_esa))
    summ["max nquad rel err estimate"] = float(np.nanmax([np.nanmax(quad_err[c]) for c in configs]))
    # quadrature with MY implied HBR vs ESA risk (fully independent of physics.py)
    q_mine = np.array([pc_quad2d(m2S["b_auto"][i], C2S["b_auto"][i], T.maxpc_b_spans_R_implied[i])[0] / LN10
                       for i in range(len(S))])
    T["quad_b_myimplied"] = q_mine
    summ["quad(b, MY implied HBR) - ESA risk"] = stats(q_mine - T.risk_esa)
    report["selection_summary"] = summ
    report["selection_rows"] = T.to_dict(orient="records")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = T[["row", "why", "risk_esa", "phys_auto", "quad_b_auto", "is_b_auto", "is_b_auto_relse", "is_b_auto_hits",
              "is2d_b_auto", "is2d_b_auto_relse", "plain_b_auto",
              "plain_b_auto_relse", "phys_spans", "is_b_spans", "is_a_spans", "hbr_auto_phys", "maxpc_b_spans_R_implied",
              "hbr_spans", "sigma_minor_m"]].copy()
    print("\n", show.to_string(float_format=lambda x: f"{x:.4g}"))
    show2 = T[["row", "mre_esa", "maxpc_b_spans_log10_pcmax_spans", "maxpc_a_spans_log10_pcmax_spans", "mrs_esa",
               "maxpc_b_spans_kstar_implied", "maxpc_a_spans_kstar_implied", "phys_kmax", "mahal_esa", "mahal3d_mine_b",
               "mahal3d_mine_a"]]
    print("\n", show2.to_string(float_format=lambda x: f"{x:.5g}"))
    print("\nselection summary:")
    for k, s in summ.items():
        print(f"  {k}: {json.dumps(jsonable(s))}")

    # ---------------- full-set 2D IS check (every valid test row with ESA risk > -30)
    if not args.no_fullset:
        t2 = time.time()
        risk = df.risk.to_numpy(float)
        ok = valid & (risk > -30) & np.isfinite(ph_auto.log_pc_phys.to_numpy(float))
        idx = np.flatnonzero(ok)
        Nf = 2000 if args.quick else 20000
        covF, _ = combined_covs(df.iloc[idx])
        m2F, C2F = project(p[idx], v[idx], covF["b_energy"])
        RF = ph_auto.hbr_m.to_numpy(float)[idx]
        good = np.linalg.eigvalsh(C2F)[:, 0] > 0
        lpF = np.full(len(idx), np.nan)
        reF = np.full(len(idx), np.nan)
        gi = np.flatnonzero(good)
        # parallel over row blocks
        blocks = np.array_split(gi, args.workers * 4)
        with Pool(args.workers) as pool:
            outs = pool.starmap(fullset_is, [(m2F[b], C2F[b], RF[b], Nf, args.seed + 17 * k)
                                             for k, b in enumerate(blocks)])
        for b, (o, r) in zip(blocks, outs):
            lpF[b], reF[b] = o, r
        phF = ph_auto.log_pc_phys.to_numpy(float)[idx]
        d = lpF - phF
        z = (1 - 10 ** (-d)) / reF
        fin = np.isfinite(d) & np.isfinite(z)
        bad = fin & (np.abs(z) > 5) & (np.abs(d) > 0.01)
        fs = {"n_rows": int(len(idx)), "n_nonPD_2d_skipped": int((~good).sum()), "samples_per_row": Nf,
              "median_rel_se": float(np.nanmedian(reF)), "p95_rel_se": float(np.nanpercentile(reF, 95)),
              "IS2D - physics(auto)": stats(d, z),
              "n_|z|>5_and_|d|>0.01": int(bad.sum()),
              "IS2D - ESA risk": stats(lpF - risk[idx]),
              "physics(auto) - ESA risk": stats(phF - risk[idx])}
        worst = np.argsort(-np.where(fin, np.abs(z), -1))[:15]
        fs["worst_z_rows"] = [{"row": int(idx[j]), "event": int(df.event_id.iat[idx[j]]), "risk": float(risk[idx[j]]),
                               "phys": float(phF[j]), "is2d": float(lpF[j]), "rel_se": float(reF[j]), "z": float(z[j]),
                               "hbr": float(RF[j]), "s_minor": float(np.sqrt(np.linalg.eigvalsh(C2F[j])[0]))}
                              for j in worst]
        # bias by Pc decade
        byd = {}
        for lo in range(-30, 0, 3):
            mk = fin & (phF >= lo) & (phF < lo + 3)
            if mk.sum() > 10:
                byd[f"[{lo},{lo + 3})"] = {"n": int(mk.sum()), "mean_d": float(d[mk].mean()),
                                            "sem": float(d[mk].std() / np.sqrt(mk.sum())), "z_mean": float(z[mk].mean()),
                                            "z_rms": float(np.sqrt((z[mk] ** 2).mean()))}
        fs["bias_by_phys_decade"] = byd
        # bias by HBR / sigma_minor regime
        rs = RF / np.sqrt(np.clip(np.linalg.eigvalsh(C2F)[:, 0], 1e-300, None))
        byr = {}
        for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 1), (1, 3), (3, 1e9)):
            mk = fin & (rs >= lo) & (rs < hi)
            if mk.sum() > 5:
                byr[f"HBR/s_minor in [{lo},{hi})"] = {"n": int(mk.sum()), "mean_d": float(d[mk].mean()),
                                                      "z_mean": float(z[mk].mean()), "z_rms": float(np.sqrt((z[mk] ** 2).mean())),
                                                      "max_abs_d": float(np.abs(d[mk]).max())}
        fs["bias_by_hbr_over_sigma_minor"] = byr
        fs["secs"] = time.time() - t2
        report["fullset_is2d"] = fs
        print("\nfull-set 2D IS:", json.dumps(jsonable(fs), indent=1))

    report["runtime_s"] = time.time() - T0
    if args.out:
        with open(args.out, "w") as f:
            json.dump(jsonable(report), f, indent=1)
        print("wrote", args.out)
    print(f"total {report['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
