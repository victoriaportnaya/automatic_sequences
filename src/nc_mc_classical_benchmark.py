#!/usr/bin/env python3
"""Classical MC/QMC benchmark: NC sequences vs standard baselines.

Links the sequence-level 1/N law (Type I) to integration error (Type II) on
classical finance, integral, and risk tasks.

Fast default: convergence on hybrid NC shifts + standard QMC baselines.
Optional --with-nc-direct adds a single-N snapshot of raw NC streams (slow).

Outputs: nc_mc_classical_benchmark_results.json, .csv
Run:     python3 nc_mc_classical_benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from mc_problems import all_problems, fit_alpha
from nc_mc_driver import NAMED_NC, build_uniforms
from recursive_exact_experiments import max_abs_nonzero, recursive_gamma_values

EPS_NC = 5e-3

CLASSICAL_PROBLEMS = [
    "european_call",
    "european_put",
    "geometric_asian_call",
    "exp_sum",
    "sin_product",
    "indicator_mean",
    "g_function",
    "rare_event",
]

BASELINE_METHODS = [
    "iid",
    "iid_antithetic",
    "sobol",
    "halton",
    "sobol_rand_shift",
    "sobol_nc_rs_shift",
    "sobol_nc_tm_shift",
    "sobol_nc_k4a_shift",
]

CERTIFIED_NC = [
    {"patterns_key": "{01}", "patterns": ("01",), "k": 2},
    {"patterns_key": "{101,111}", "patterns": ("101", "111"), "k": 2},
    {"patterns_key": "RS*TM", "patterns": ("11", "1"), "k": 2},
]

NC_DIRECT_METHODS = ["nc_rs", "nc_tm", "nc_k4a"]


def nc_metadata(patterns: tuple[str, ...], k: int, m_max: int = 64) -> dict[str, Any]:
    gamma = recursive_gamma_values(patterns, k, m_max)
    g = float(max_abs_nonzero(gamma))
    return {"limit_gmax": g, "is_nc": g <= EPS_NC}


def hybrid_drivers() -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    for key, spec in NAMED_NC.items():
        meta = nc_metadata(tuple(spec["patterns"]), int(spec["k"]))
        drivers.append({
            "method": f"sobol_nc_shift|{key}",
            "patterns_key": key,
            "patterns": list(spec["patterns"]),
            "k": spec["k"],
            **meta,
        })
    for c in CERTIFIED_NC:
        meta = nc_metadata(tuple(c["patterns"]), int(c["k"]))
        drivers.append({
            "method": f"sobol_nc_shift|{c['patterns_key']}",
            "patterns_key": c["patterns_key"],
            "patterns": list(c["patterns"]),
            "k": c["k"],
            **meta,
        })
    return drivers


def evaluate(problem, method: str, n: int, dim: int, bits: int, seed: int, driver: dict | None = None) -> float:
    if driver is not None:
        u = build_uniforms(
            "sobol_nc_shift",
            n,
            dim,
            bits,
            seed,
            patterns=tuple(driver["patterns"]),
            k=int(driver["k"]),
        )
    else:
        u = build_uniforms(method, n, dim, bits, seed)
    return problem.estimate(u)


def run_convergence(
    problem,
    ns: list[int],
    bits: int,
    seed: int,
    reps: int,
    hybrids: list[dict],
) -> list[dict]:
    truth = float(problem.exact())
    dim = problem.default_dim
    rows: list[dict] = []

    for n in ns:
        for method in BASELINE_METHODS:
            ests = [evaluate(problem, method, n, dim, bits, seed + 10007 * r + 7919 * n) for r in range(reps)]
            mean_est = float(np.mean(ests))
            rows.append(_row(problem, method, "baseline", None, n, truth, mean_est, ests))

        for i, drv in enumerate(hybrids):
            if drv["method"] in BASELINE_METHODS:
                continue  # already in baselines (named shifts)
            ests = [
                evaluate(problem, drv["method"], n, dim, bits, seed + 50000 * i + 10007 * r + 7919 * n, drv)
                for r in range(reps)
            ]
            mean_est = float(np.mean(ests))
            rows.append(_row(problem, drv["method"], "sobol_nc_shift", drv, n, truth, mean_est, ests))
    return rows


def run_direct_snapshot(
    problem,
    n: int,
    bits: int,
    seed: int,
    reps: int,
) -> list[dict]:
    truth = float(problem.exact())
    dim = problem.default_dim
    rows: list[dict] = []
    for method in NC_DIRECT_METHODS:
        ests = [evaluate(problem, method, n, dim, bits, seed + 10007 * r) for r in range(reps)]
        mean_est = float(np.mean(ests))
        spec = NAMED_NC[method]
        meta = nc_metadata(tuple(spec["patterns"]), int(spec["k"]))
        rows.append({
            "problem": problem.name,
            "category": problem.category,
            "method": method,
            "driver_type": "nc_direct_snapshot",
            "patterns_key": method,
            "k": spec["k"],
            "limit_gmax": meta["limit_gmax"],
            "is_nc": meta["is_nc"],
            "n_paths": n,
            "truth": truth,
            "estimate_mean": mean_est,
            "estimate_std": float(np.std(ests, ddof=1) if reps > 1 else 0.0),
            "abs_error": abs(mean_est - truth),
            "rmse": float(math.sqrt(np.mean([(e - truth) ** 2 for e in ests]))),
        })
    return rows


def _row(problem, method, driver_type, drv, n, truth, mean_est, ests) -> dict:
    return {
        "problem": problem.name,
        "category": problem.category,
        "method": method,
        "driver_type": driver_type,
        "patterns_key": drv["patterns_key"] if drv else "",
        "k": drv["k"] if drv else None,
        "limit_gmax": drv["limit_gmax"] if drv else None,
        "is_nc": drv["is_nc"] if drv else None,
        "n_paths": n,
        "truth": truth,
        "estimate_mean": mean_est,
        "estimate_std": float(np.std(ests, ddof=1) if len(ests) > 1 else 0.0),
        "abs_error": abs(mean_est - truth),
        "rmse": float(math.sqrt(np.mean([(e - truth) ** 2 for e in ests]))),
    }


def summarize(rows: list[dict], ns: list[int]) -> dict[str, Any]:
    problems = sorted({r["problem"] for r in rows if r["driver_type"] != "nc_direct_snapshot"})
    by_problem: dict[str, Any] = {}

    for prob in problems:
        sub = [r for r in rows if r["problem"] == prob and r["driver_type"] != "nc_direct_snapshot"]
        methods = sorted({r["method"] for r in sub})
        stats: dict[str, Any] = {}
        for mid in methods:
            mrows = [r for r in sub if r["method"] == mid]
            errs = [r["abs_error"] for r in mrows]
            nvals = [r["n_paths"] for r in mrows]
            stats[mid] = {
                "alpha_fit": fit_alpha(nvals, errs),
                "last_abs_error": float(errs[-1]),
                "best_abs_error": float(min(errs)),
                "limit_gmax": mrows[0].get("limit_gmax"),
                "is_nc": mrows[0].get("is_nc"),
            }
        ranked = sorted(stats.items(), key=lambda x: x[1]["last_abs_error"])
        best_m, best_s = ranked[0]
        sobol_e = stats.get("sobol", {}).get("last_abs_error", float("nan"))
        iid_e = stats.get("iid", {}).get("last_abs_error", float("nan"))
        hybrids = {k: v for k, v in stats.items() if k.startswith("sobol_nc")}
        best_h = min(hybrids.items(), key=lambda x: x[1]["last_abs_error"]) if hybrids else (None, {})

        by_problem[prob] = {
            "best_method": best_m,
            "best_last_error": best_s["last_abs_error"],
            "best_alpha": best_s["alpha_fit"],
            "sobol_last_error": sobol_e,
            "iid_last_error": iid_e,
            "best_hybrid_method": best_h[0],
            "best_hybrid_error": best_h[1].get("last_abs_error", float("nan")),
            "hybrid_beats_sobol": bool(best_h[0] and best_h[1].get("last_abs_error", float("inf")) < sobol_e),
            "hybrid_speedup_vs_sobol": float(sobol_e / best_h[1]["last_abs_error"]) if best_h[0] and best_h[1].get("last_abs_error", 0) > 0 else float("nan"),
            "methods": stats,
        }

    snap = [r for r in rows if r["driver_type"] == "nc_direct_snapshot"]
    direct_by_problem = {}
    for r in snap:
        direct_by_problem.setdefault(r["problem"], {})[r["method"]] = r["abs_error"]

    g = {
        "n_problems": len(problems),
        "hybrid_beats_sobol": sum(1 for p in by_problem.values() if p["hybrid_beats_sobol"]),
        "mean_sobol_alpha": float(np.nanmean([p["methods"].get("sobol", {}).get("alpha_fit", float("nan")) for p in by_problem.values()])),
        "mean_iid_alpha": float(np.nanmean([p["methods"].get("iid", {}).get("alpha_fit", float("nan")) for p in by_problem.values()])),
        "mean_best_hybrid_alpha": float(np.nanmean([
            p["methods"].get(p["best_hybrid_method"], {}).get("alpha_fit", float("nan"))
            for p in by_problem.values() if p["best_hybrid_method"]
        ])),
    }
    return {"by_problem": by_problem, "nc_direct_snapshot": direct_by_problem, "global": g}


def main() -> None:
    p = argparse.ArgumentParser(description="Classical MC benchmark.")
    p.add_argument("--n-min-pow", type=int, default=10)
    p.add_argument("--n-max-pow", type=int, default=13)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--bits", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--integral-dim", type=int, default=6)
    p.add_argument("--rare-dim", type=int, default=10)
    p.add_argument("--with-nc-direct", action="store_true", help="Add nc_direct snapshot at N=4096 (slow).")
    p.add_argument("--out-json", default="nc_mc_classical_benchmark_results.json")
    p.add_argument("--out-csv", default="nc_mc_classical_benchmark_results.csv")
    args = p.parse_args()

    ns = [2**j for j in range(args.n_min_pow, args.n_max_pow + 1)]
    catalog = all_problems(n_steps=args.n_steps, integral_dim=args.integral_dim, rare_dim=args.rare_dim)
    hybrids = hybrid_drivers()

    t0 = time.time()
    all_rows: list[dict] = []
    for i, pname in enumerate(CLASSICAL_PROBLEMS):
        problem = catalog[pname]
        print(f"[{i+1}/{len(CLASSICAL_PROBLEMS)}] {pname} (d={problem.default_dim})", flush=True)
        all_rows.extend(run_convergence(problem, ns, args.bits, args.seed, args.reps, hybrids))
        if args.with_nc_direct or pname in ("european_call", "european_put"):
            all_rows.extend(run_direct_snapshot(problem, 4096, args.bits, args.seed, min(args.reps, 2)))

    summary = summarize(all_rows, ns)
    elapsed = time.time() - t0

    payload = {
        "config": {
            "problems": CLASSICAL_PROBLEMS,
            "ns": ns,
            "reps": args.reps,
            "bits": args.bits,
            "methods": BASELINE_METHODS,
            "hybrid_drivers": len(hybrids),
            "with_nc_direct": args.with_nc_direct,
            "elapsed_sec": elapsed,
        },
        "summary": summary,
        "rows": all_rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = list(all_rows[0].keys()) if all_rows else []
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nWrote {args.out_json} ({len(all_rows)} rows, {elapsed:.1f}s)")
    g = summary["global"]
    print(f"Hybrid beats Sobol: {g['hybrid_beats_sobol']}/{g['n_problems']}")
    print(f"Mean alpha: iid={g['mean_iid_alpha']:.3f}, sobol={g['mean_sobol_alpha']:.3f}, best_hybrid={g['mean_best_hybrid_alpha']:.3f}")
    print("\nPer-problem (last N):")
    for prob, s in summary["by_problem"].items():
        win = "Y" if s["hybrid_beats_sobol"] else "n"
        print(f"  {prob:22s} best={s['best_method'][:28]:28s} err={s['best_last_error']:.3e} sobol={s['sobol_last_error']:.3e} [{win}]")


if __name__ == "__main__":
    main()
