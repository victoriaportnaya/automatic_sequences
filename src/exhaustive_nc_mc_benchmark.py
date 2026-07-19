#!/usr/bin/env python3
"""Use every certified NC pattern set as an MC/QMC driver and compare.

Loads NC sequences from recursive exhaustive sweep results, embeds them as
uniform generators (direct NC stream or Sobol + NC digital shift), and prices
a European call against exact Black--Scholes.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from mc_problems import all_problems, resolve_problems
from nc_stream import (
    apply_digital_shift,
    build_nc_uniforms_from_patterns,
    mask_from_nc_patterns,
    sobol_uniforms,
)


def load_nc_sequences(path: Path, k_filter: int | None = None) -> list[dict]:
    data = json.loads(path.read_text())
    seen: set[tuple[str, ...]] = set()
    out: list[dict] = []
    for row in data.get("rows", []):
        if not row.get("is_nc"):
            continue
        k = int(row["k"])
        if k_filter is not None and k != k_filter:
            continue
        patterns = tuple(row["patterns"])
        if patterns in seen:
            continue
        seen.add(patterns)
        out.append(
            {
                "patterns": patterns,
                "patterns_key": row.get("patterns_key", " ".join(patterns)),
                "k": k,
                "g_exact": float(row.get("gmax", row.get("g_exact", 0.0))),
            }
        )
    return out


def run_baselines(
    problem,
    n_paths: int,
    bits: int,
    seed: int,
) -> list[dict]:
    exact = float(problem.exact())
    rows: list[dict] = []

    def est(u: np.ndarray) -> float:
        return problem.estimate(u)

    rng = np.random.default_rng(seed)
    u_iid = rng.random((n_paths, problem.default_dim))
    rows.append(
        {
            "driver_id": "baseline_iid",
            "mode": "iid",
            "k": None,
            "patterns_key": "",
            "estimate": est(u_iid),
        }
    )

    try:
        from scipy.stats import qmc  # type: ignore

        d = problem.default_dim
        sob_eng = qmc.Sobol(d=d, scramble=True, seed=seed + 1)
        u_sob = sob_eng.random(n=n_paths)
        rows.append(
            {
                "driver_id": "baseline_sobol",
                "mode": "sobol",
                "k": None,
                "patterns_key": "",
                "estimate": est(u_sob),
            }
        )
        hal_eng = qmc.Halton(d=d, scramble=True, seed=seed + 2)
        u_hal = hal_eng.random(n=n_paths)
        rows.append(
            {
                "driver_id": "baseline_halton",
                "mode": "halton",
                "k": None,
                "patterns_key": "",
                "estimate": est(u_hal),
            }
        )
    except Exception:
        pass

    for ref_id, patterns, k in [
        ("baseline_rs", ("11",), 2),
        ("baseline_k4", ("01", "10"), 4),
    ]:
        u = build_nc_uniforms_from_patterns(
            n_paths, problem.default_dim, bits, patterns, k, seed=seed + 99
        )
        rows.append(
            {
                "driver_id": ref_id,
                "mode": "nc_direct",
                "k": k,
                "patterns_key": " ".join(patterns),
                "estimate": est(u),
            }
        )

    for row in rows:
        row["abs_error"] = abs(row["estimate"] - exact)
    return rows, exact


def run_nc_driver(
    patterns: tuple[str, ...],
    patterns_key: str,
    k: int,
    mode: str,
    problem,
    n_paths: int,
    bits: int,
    seed: int,
    rep: int,
) -> float:
    dim = problem.default_dim
    if mode == "nc_direct":
        u = build_nc_uniforms_from_patterns(n_paths, dim, bits, patterns, k, seed=seed)
        return problem.estimate(u)

    if mode == "sobol_nc_shift":
        u0 = sobol_uniforms(n_paths, dim, seed=seed)
        masks = mask_from_nc_patterns(dim, bits, rep, patterns, k)
        u = apply_digital_shift(u0, masks, bits)
        return problem.estimate(u)

    raise ValueError(f"Unknown mode {mode}")


def main() -> None:
    p = argparse.ArgumentParser(description="MC benchmark using all NC sequences as drivers.")
    p.add_argument("--nc-json", default="recursive_exhaustive_nc_len3_results.json")
    p.add_argument("--k-values", default="2,4,6,8", help="Comma-separated k to test (native modulus).")
    p.add_argument("--modes", default="nc_direct,sobol_nc_shift")
    p.add_argument("--n-paths", type=int, default=8192)
    p.add_argument("--problem", default="european_call", help="Single problem or preset (finance,integral,risk,all).")
    p.add_argument("--problems", default="", help="If set, run multiple problems (comma list or preset).")
    p.add_argument("--n-steps", type=int, default=16, help="Asian monitoring steps (ignored for 1D problems).")
    p.add_argument("--integral-dim", type=int, default=8)
    p.add_argument("--rare-dim", type=int, default=12)
    p.add_argument("--bits-per-u", type=int, default=16)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--s0", type=float, default=100.0)
    p.add_argument("--strike", type=float, default=100.0)
    p.add_argument("--rate", type=float, default=0.03)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--maturity", type=float, default=1.0)
    p.add_argument("--max-nc", type=int, default=0, help="Cap NC drivers (0 = all).")
    p.add_argument("--out-json", default="exhaustive_nc_mc_benchmark_results.json")
    p.add_argument("--out-csv", default="exhaustive_nc_mc_benchmark_results.csv")
    args = p.parse_args()

    ks = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    fin_kw = dict(
        s0=args.s0,
        strike=args.strike,
        rate=args.rate,
        sigma=args.sigma,
        maturity=args.maturity,
        n_steps=args.n_steps,
        integral_dim=args.integral_dim,
        rare_dim=args.rare_dim,
    )
    catalog = all_problems(**fin_kw)
    if args.problems.strip():
        problem_names = resolve_problems(args.problems, catalog)
    else:
        if args.problem in catalog:
            problem_names = [args.problem]
        else:
            problem_names = resolve_problems(args.problem, catalog)

    all_nc: list[dict] = []
    for k in ks:
        all_nc.extend(load_nc_sequences(Path(args.nc_json), k_filter=k))
    if args.max_nc > 0:
        all_nc = all_nc[: args.max_nc]

    all_rows: list[dict] = []
    all_summaries: dict = {}
    t0 = time.perf_counter()

    for prob_name in problem_names:
        problem = catalog[prob_name]
        exact = float(problem.exact())
        print(f"\n=== Problem: {prob_name} (d={problem.default_dim}, exact≈{exact:.6g}) ===")

        print("Baselines...")
        baseline_rows, _ = run_baselines(
            problem,
            args.n_paths,
            args.bits_per_u,
            args.seed,
        )
        rows: list[dict] = []
        for b in baseline_rows:
            b["problem"] = prob_name
            b["category"] = problem.category
            b["patterns"] = []
            b["set_size"] = 0
            rows.append(b)

        print(f"NC drivers: {len(all_nc)} sequences x modes {modes}")
        for i, nc in enumerate(all_nc):
            patterns = tuple(nc["patterns"])
            for mode in modes:
                est = run_nc_driver(
                    patterns,
                    nc["patterns_key"],
                    int(nc["k"]),
                    mode,
                    problem,
                    args.n_paths,
                    args.bits_per_u,
                    args.seed + i,
                    rep=i,
                )
                rows.append(
                    {
                        "problem": prob_name,
                        "category": problem.category,
                        "driver_id": f"{mode}|k{nc['k']}|{nc['patterns_key']}",
                        "mode": mode,
                        "k": int(nc["k"]),
                        "patterns_key": nc["patterns_key"],
                        "patterns": list(patterns),
                        "set_size": len(patterns),
                        "g_exact": nc.get("g_exact"),
                        "estimate": est,
                        "abs_error": abs(est - exact),
                    }
                )
            if (i + 1) % 50 == 0 or i + 1 == len(all_nc):
                print(f"  [{i+1}/{len(all_nc)}] {time.perf_counter()-t0:.1f}s", flush=True)

        rows.sort(key=lambda r: r["abs_error"])
        all_rows.extend(rows)
        all_summaries[prob_name] = {
            "exact": exact,
            "best": rows[0] if rows else None,
        }

    rows = all_rows
    exact = None

    summary: dict = {
        "n_nc_drivers": len(all_nc),
        "elapsed_sec": time.perf_counter() - t0,
        "problems": all_summaries,
        "by_mode": {},
        "by_k": {},
    }
    for mode in modes + ["iid", "sobol", "halton"]:
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        summary["by_mode"][mode] = {
            "count": len(sub),
            "best_error": sub[0]["abs_error"],
            "median_error": float(np.median([r["abs_error"] for r in sub])),
            "best_driver": sub[0]["driver_id"],
        }
    for k in ks:
        sub = [r for r in rows if r.get("k") == k]
        if sub:
            summary["by_k"][str(k)] = {
                "count": len(sub),
                "best_error": min(r["abs_error"] for r in sub),
                "median_error": float(np.median([r["abs_error"] for r in sub])),
            }

    payload = {
        "config": {
            "nc_json": args.nc_json,
            "k_values": ks,
            "modes": modes,
            "n_paths": args.n_paths,
            "problems": problem_names,
            "n_steps": args.n_steps,
            "bits_per_u": args.bits_per_u,
        },
        "summary": summary,
        "rows": rows,
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "problem",
        "category",
        "driver_id",
        "mode",
        "k",
        "patterns_key",
        "set_size",
        "estimate",
        "abs_error",
        "g_exact",
    ]
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {args.out_json} ({len(rows)} rows)")
    for pname, sm in all_summaries.items():
        best = sm["best"]
        if best:
            print(f"  {pname}: best={best['driver_id'][:50]} error={best['abs_error']:.6f}")


if __name__ == "__main__":
    main()
