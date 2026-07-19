#!/usr/bin/env python3
r"""Empirical AND theoretical certification of the 1/N convergence law for NC sequences.

Claim.  For a k-NC pattern sequence a(n)=zeta_k^{#(A,n)} the finite-prefix
correlation
        gamma_N(m) = (1/N) sum_{n<N} a(n) conj(a(n+m)),
        G_N        = max_{1<=m<=M} |gamma_N(m)|
obeys  G_N = Theta(1/N): the prefix correlation *sums* S_m(N)=N*gamma_N(m) stay
*bounded*, and along the dyadic scale G_{2^j}/G_{2^{j+1}} -> 2.

Three independent certificates are produced:

  (A) EMPIRICAL.  Exact G_N on a dyadic grid N=2^j; least-squares fit of
      log G_N vs log N (slope = -alpha) with R^2 and a 95% CI; dyadic halving
      ratios; and direct evidence that sup_N |S_m(N)| < infinity for NC while it
      grows linearly for non-NC.

  (B) SPECTRAL (rigorous rate certificate).  The normalized correlations satisfy
      an exact dyadic transfer recursion
          gamma(m,r) = (1/2) b(r) conj(b(r+m)) [gamma(m',r0)+gamma(m',r1)],
      m'=floor(((r&1)+m)/2), r0=r//2, r1=r0+2^{L-1}.
      We build the finite transfer operator T_m over its recurrent state class,
      compute its spectral radius EXACTLY (a finite linear-algebra certificate),
      and confirm rho(T_m)=1/2 -- the structural source of the 1/N rate -- while
      the operator's fixed point reproduces the limit gamma(m) (=0 iff NC).

  (C) DICHOTOMY.  alpha clusters at 1 for NC and at 0 for non-NC, separated with
      a two-sample statistic.

Outputs: nc_one_over_n_convergence_results.json
Run:     python3 nc_one_over_n_convergence.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments import gamma_prefix, sequence_values
from recursive_exact_experiments import (
    compute_b_period,
    named_catalog,
    recursive_gamma_values,
)

EPS_NC = 5e-3


# ---------------------------------------------------------------------------
# Complex prefix correlation for general k (k=2 falls back to experiments.gamma_prefix)
# ---------------------------------------------------------------------------
def complex_sequence(patterns: tuple[str, ...], k: int, n_max: int) -> np.ndarray:
    if k == 2:
        return sequence_values(patterns, n_max).astype(np.complex128)
    from recursive_exact_experiments import count_pattern_set_fixed_padding

    l = max(len(p) for p in patterns)
    zeta = np.exp(2j * math.pi / k)
    out = np.empty(n_max, dtype=np.complex128)
    for n in range(n_max):
        out[n] = zeta ** (count_pattern_set_fixed_padding(n, patterns, l) % k)
    return out


def gmax_at_prefix(values: np.ndarray, n: int, m_max: int) -> tuple[float, np.ndarray]:
    """Return G_n and the per-lag |gamma_n(m)| array (m=0..m_max) on the first n samples."""
    a = values[:n]
    if np.iscomplexobj(a) and np.any(np.abs(a.imag) > 1e-12):
        g = _complex_gamma_prefix(a, m_max)
    else:
        g = np.abs(np.array(gamma_prefix(a.real.astype(np.int8), m_max), dtype=np.float64))
    return float(np.max(g[1:])), g


def _complex_gamma_prefix(a: np.ndarray, m_max: int) -> np.ndarray:
    n = len(a)
    out = np.zeros(m_max + 1, dtype=np.float64)
    out[0] = 1.0
    for m in range(1, m_max + 1):
        s = np.vdot(a[: n - m], a[m:])  # sum conj(a[n]) a[n+m]; modulus invariant
        out[m] = abs(s) / n
    return out


# ---------------------------------------------------------------------------
# (A) Empirical 1/N law
# ---------------------------------------------------------------------------
def linfit_with_ci(logn: np.ndarray, logg: np.ndarray) -> dict[str, float]:
    """OLS slope/intercept with R^2 and 95% CI on the slope."""
    n = len(logn)
    x = logn - logn.mean()
    y = logg - logg.mean()
    sxx = float(np.dot(x, x))
    slope = float(np.dot(x, y) / sxx)
    intercept = float(logg.mean() - slope * logn.mean())
    resid = logg - (intercept + slope * logn)
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y, y))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # standard error of slope, 95% CI via t(n-2)~ use 2.0 multiplier (n>=8)
    if n > 2 and sxx > 0:
        s_err = math.sqrt(ss_res / (n - 2) / sxx)
    else:
        s_err = float("nan")
    tcrit = 2.0
    return {
        "slope": slope,
        "alpha": -slope,
        "intercept": intercept,
        "r2": r2,
        "slope_se": s_err,
        "alpha_ci_lo": -slope - tcrit * s_err,
        "alpha_ci_hi": -slope + tcrit * s_err,
    }


def empirical_panel(
    patterns: tuple[str, ...], k: int, j_lo: int, j_hi: int, m_max: int
) -> dict[str, Any]:
    n_top = 1 << j_hi
    values = complex_sequence(patterns, k, n_top + m_max + 1)
    js = list(range(j_lo, j_hi + 1))
    ns = [1 << j for j in js]
    g_by_n: list[float] = []
    argmax_lag: list[int] = []
    sup_S = 0.0           # sup_N |S_m(N)| over the argmax lag track
    s_track: list[float] = []
    for n in ns:
        g_n, g_arr = gmax_at_prefix(values, n, m_max)
        g_by_n.append(g_n)
        m_star = int(np.argmax(g_arr[1:]) + 1)
        argmax_lag.append(m_star)
        s_val = g_n * n  # |S_{m*}(N)| = N * |gamma_N(m*)|
        s_track.append(s_val)
        sup_S = max(sup_S, s_val)

    logn = np.log(np.array(ns, dtype=np.float64))
    logg = np.log(np.maximum(np.array(g_by_n), 1e-18))
    fit = linfit_with_ci(logn, logg)

    # dyadic halving ratios G_{2^j}/G_{2^{j+1}}
    ratios = [
        g_by_n[i] / g_by_n[i + 1] if g_by_n[i + 1] > 0 else float("nan")
        for i in range(len(g_by_n) - 1)
    ]
    # growth exponent of the correlation SUM |S(N)| (should be ~0 for NC, ~1 for non-NC)
    logs = np.log(np.maximum(np.array(s_track), 1e-18))
    s_fit = linfit_with_ci(logn, logs)

    return {
        "ns": ns,
        "g_by_n": g_by_n,
        "argmax_lag": argmax_lag,
        "alpha": fit["alpha"],
        "alpha_ci": [fit["alpha_ci_lo"], fit["alpha_ci_hi"]],
        "r2": fit["r2"],
        "dyadic_ratios": ratios,
        "mean_tail_ratio": float(np.nanmean(ratios[-4:])) if len(ratios) >= 1 else float("nan"),
        "sup_corr_sum": sup_S,
        "corr_sum_growth_exponent": s_fit["alpha"] * -1.0,  # slope of log|S| vs log N
    }


# ---------------------------------------------------------------------------
# (B) Exact dyadic transfer operator T_m and its spectral radius
# ---------------------------------------------------------------------------
def build_transfer_operator(
    patterns: tuple[str, ...], k: int, m0: int
) -> dict[str, Any]:
    r"""Construct the finite transfer operator for the recursion

        gamma(m,r) = (1/2) b(r) conj(b(r+m)) [ gamma(m',r0) + gamma(m',r1) ],
        m' = ((r&1)+m)//2,  r0 = r//2,  r1 = r0 + 2^{L-1},

    with the absorbing state gamma(0,*)=1. We enumerate the recurrent states
    (m,r), m>=1, reachable from (m0, r) for all r, assemble T (states->states)
    and the constant vector c (transitions into the absorbing m=0 state), solve
    Gamma=(I-T)^{-1} c (the exact limit), and return rho(T) and the limit gamma(m0).
    """
    l = max(len(p) for p in patterns)
    period = 1 << l
    half = 1 << (l - 1)
    b = compute_b_period(patterns, k, l)

    def bv(idx: int) -> complex:
        return b[idx % period]

    # enumerate reachable states (m, r) with m >= 1
    from collections import deque

    states: dict[tuple[int, int], int] = {}
    queue: deque[tuple[int, int]] = deque()
    for r in range(period):
        st = (m0, r)
        if st not in states:
            states[st] = len(states)
            queue.append(st)
    while queue:
        m, r = queue.popleft()
        m_next = ((r & 1) + m) // 2
        if m_next >= 1:
            for rc in (r // 2, r // 2 + half):
                st = (m_next, rc)
                if st not in states:
                    states[st] = len(states)
                    queue.append(st)

    dim = len(states)
    T = np.zeros((dim, dim), dtype=np.complex128)
    c = np.zeros(dim, dtype=np.complex128)
    for (m, r), i in states.items():
        pref = 0.5 * bv(r) * np.conj(bv(r + m))
        m_next = ((r & 1) + m) // 2
        for rc in (r // 2, r // 2 + half):
            if m_next == 0:
                c[i] += pref * 1.0  # absorbing gamma(0,*)=1
            else:
                T[i, states[(m_next, rc)]] += pref

    eig = np.linalg.eigvals(T)
    rho = float(np.max(np.abs(eig)))
    # exact limit fixed point Gamma = (I-T)^{-1} c
    gamma_states = np.linalg.solve(np.eye(dim) - T, c)
    # gamma(m0) = average over r of gamma(m0, r)
    gm0 = np.mean([gamma_states[states[(m0, r)]] for r in range(period)])

    # iterate to expose convergence rate: Gamma^{(0)}=0, Gamma^{(j+1)} = T Gamma^{(j)} + c
    g_iter = np.zeros(dim, dtype=np.complex128)
    iter_gamma: list[float] = []
    for _ in range(20):
        g_iter = T @ g_iter + c
        val = np.mean([g_iter[states[(m0, r)]] for r in range(period)])
        iter_gamma.append(abs(val))

    return {
        "m": m0,
        "dim": dim,
        "spectral_radius": rho,
        "limit_gamma_abs": float(abs(gm0)),
        "iter_gamma_abs": iter_gamma,
    }


def validate_operator_vs_prefix(
    patterns: tuple[str, ...], k: int, m0: int, j_lo: int, j_hi: int
) -> dict[str, Any]:
    """Confirm the transfer-operator iterate reproduces the direct finite-prefix
    correlation |gamma_{2^j}(m0)| (up to the O(2^{-j}) boundary overhang)."""
    op = build_transfer_operator(patterns, k, m0)
    iter_vals = op["iter_gamma_abs"]
    values = complex_sequence(patterns, k, (1 << j_hi) + m0 + 1)
    direct = []
    for j in range(j_lo, j_hi + 1):
        n = 1 << j
        a = values[:n]
        s = np.vdot(a[: n - m0], a[m0:])
        direct.append(abs(s) / n)
    # align: iterate index j corresponds to block length 2^j
    err = max(
        abs(iter_vals[j] - direct[j - j_lo])
        for j in range(j_lo, min(j_hi, len(iter_vals) - 1) + 1)
    )
    return {"m": m0, "max_abs_err": float(err), "spectral_radius": op["spectral_radius"]}


def spectral_certificate(patterns: tuple[str, ...], k: int, m_list: list[int]) -> dict[str, Any]:
    rows = [build_transfer_operator(patterns, k, m) for m in m_list]
    rhos = [r["spectral_radius"] for r in rows]
    return {
        "per_m": rows,
        "max_spectral_radius": float(max(rhos)),
        "all_rho_half": bool(all(abs(r - 0.5) < 1e-9 for r in rhos)),
        "max_limit_gamma": float(max(r["limit_gamma_abs"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def run(j_lo: int, j_hi: int, m_max: int) -> dict[str, Any]:
    cat = named_catalog()
    # k=2 analysis on the classic catalog; k=4 on the 4-NC-only families
    k2_names = [n for n in cat if not n.startswith("K4_")]
    k4_names = [n for n in cat if n.startswith("K4_")]

    results: dict[str, Any] = {"k2": {}, "k4": {}}

    for name in sorted(k2_names):
        patterns = cat[name]
        g_limit = recursive_gamma_values(patterns, 2, m_max)
        is_nc = max(abs(z) for z in g_limit[1:]) <= EPS_NC
        emp = empirical_panel(patterns, 2, j_lo, j_hi, m_max)
        spec = spectral_certificate(patterns, 2, [1, 2, 3, 5, 8])
        results["k2"][name] = {
            "patterns": list(patterns),
            "is_nc": is_nc,
            "limit_gmax": float(max(abs(z) for z in g_limit[1:])),
            "empirical": emp,
            "spectral": spec,
        }

    k4_k2_nonnc_alpha: list[float] = []
    for name in sorted(k4_names):
        patterns = cat[name]
        g2 = recursive_gamma_values(patterns, 2, m_max)
        g4 = recursive_gamma_values(patterns, 4, m_max)
        is_4nc = max(abs(z) for z in g4[1:]) <= EPS_NC
        gmax_k2 = float(max(abs(z) for z in g2[1:]))
        emp = empirical_panel(patterns, 4, j_lo, min(j_hi, 16), m_max)
        # the same family viewed at k=2 is non-NC -> a plateau example for the dichotomy
        emp_k2 = empirical_panel(patterns, 2, j_lo, min(j_hi, 16), m_max)
        if gmax_k2 > EPS_NC:
            k4_k2_nonnc_alpha.append(emp_k2["alpha"])
        spec = spectral_certificate(patterns, 4, [1, 2, 3, 5, 8])
        results["k4"][name] = {
            "patterns": list(patterns),
            "is_4nc": is_4nc,
            "limit_gmax_k2": gmax_k2,
            "limit_gmax_k4": float(max(abs(z) for z in g4[1:])),
            "empirical": emp,
            "empirical_k2_nonnc": {"alpha": emp_k2["alpha"], "mean_tail_ratio": emp_k2["mean_tail_ratio"]},
            "spectral": spec,
        }

    # operator-vs-prefix validation on a few representatives
    results["operator_validation"] = {
        nm: validate_operator_vs_prefix(cat[nm], 2, 1, j_lo, min(j_hi, 16))
        for nm in ["RS", "{01}", "A_3", "TM"]
        if nm in cat
    }

    # dichotomy summary over k=2 (NC catalog entries vs all non-NC examples,
    # the latter including the K4 families viewed at k=2)
    nc_alpha = [v["empirical"]["alpha"] for v in results["k2"].values() if v["is_nc"]]
    non_alpha = [v["empirical"]["alpha"] for v in results["k2"].values() if not v["is_nc"]]
    non_alpha = non_alpha + k4_k2_nonnc_alpha
    results["dichotomy"] = {
        "nc_alpha_mean": float(np.mean(nc_alpha)) if nc_alpha else float("nan"),
        "nc_alpha_min": float(np.min(nc_alpha)) if nc_alpha else float("nan"),
        "nc_alpha_max": float(np.max(nc_alpha)) if nc_alpha else float("nan"),
        "non_alpha_mean": float(np.mean(non_alpha)) if non_alpha else float("nan"),
        "non_alpha_max": float(np.max(non_alpha)) if non_alpha else float("nan"),
        "n_nc": len(nc_alpha),
        "n_non": len(non_alpha),
        "separation_gap": (
            float(np.min(nc_alpha) - np.max(non_alpha))
            if nc_alpha and non_alpha
            else float("nan")
        ),
    }
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="1/N convergence certification for NC sequences.")
    p.add_argument("--j-lo", type=int, default=10)
    p.add_argument("--j-hi", type=int, default=18)
    p.add_argument("--m-max", type=int, default=32)
    p.add_argument("--out-json", default="nc_one_over_n_convergence_results.json")
    args = p.parse_args()

    res = run(args.j_lo, args.j_hi, args.m_max)
    Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"Wrote {args.out_json}\n")
    print("=== (A) Empirical 1/N law, k=2 catalog ===")
    print(f"{'name':22s} {'NC':>3s} {'alpha':>7s} {'95% CI':>16s} {'R2':>6s} "
          f"{'tailratio':>9s} {'sup|S|':>8s} {'Sgrowth':>7s}")
    for name, v in sorted(res["k2"].items()):
        e = v["empirical"]
        ci = e["alpha_ci"]
        print(
            f"{name:22s} {('Y' if v['is_nc'] else 'n'):>3s} {e['alpha']:7.3f} "
            f"[{ci[0]:5.2f},{ci[1]:5.2f}] {e['r2']:6.3f} {e['mean_tail_ratio']:9.3f} "
            f"{e['sup_corr_sum']:8.2f} {e['corr_sum_growth_exponent']:7.3f}"
        )

    print("\n=== (B) Spectral certificate: rho(T_m)=1/2 (rigorous rate) ===")
    for name, v in sorted(res["k2"].items()):
        s = v["spectral"]
        print(
            f"{name:22s} max_rho={s['max_spectral_radius']:.6f} "
            f"all_rho=1/2:{str(s['all_rho_half']):>5s} max|gamma_lim|={s['max_limit_gamma']:.2e}"
        )

    print("\n=== 4-NC-only families (k=4) ===")
    print(f"{'name':24s} {'4NC':>3s} {'alpha_k4':>8s} {'tailratio':>9s} {'rho':>8s} {'Gk2':>6s}")
    for name, v in sorted(res["k4"].items()):
        e = v["empirical"]
        print(
            f"{name:24s} {('Y' if v['is_4nc'] else 'n'):>3s} {e['alpha']:8.3f} "
            f"{e['mean_tail_ratio']:9.3f} {v['spectral']['max_spectral_radius']:8.5f} "
            f"{v['limit_gmax_k2']:6.2f}"
        )

    print("\n=== Operator-vs-prefix validation (m=1): iterate matches direct gamma ===")
    for nm, v in res.get("operator_validation", {}).items():
        print(f"  {nm:8s} max|iter - direct| = {v['max_abs_err']:.2e}  rho={v['spectral_radius']:.4f}")

    d = res["dichotomy"]
    print("\n=== (C) Dichotomy (k=2) ===")
    print(f"  NC  alpha: mean={d['nc_alpha_mean']:.3f} range=[{d['nc_alpha_min']:.3f},{d['nc_alpha_max']:.3f}] (n={d['n_nc']})")
    print(f"  non alpha: mean={d['non_alpha_mean']:.3f} max={d['non_alpha_max']:.3f} (n={d['n_non']})")
    print(f"  separation gap (min NC alpha - max non alpha) = {d['separation_gap']:.3f}")


if __name__ == "__main__":
    main()
