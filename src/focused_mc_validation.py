#!/usr/bin/env python3
"""Paired validation of automatic-sequence digital shifts for Sobol nets.

The earlier project benchmark compared several NC masks against one Sobol run.
That design is useful for discovery but favors the minimum over many candidates.
Here every method is evaluated on the same unscrambled Sobol net, over equally
sized ensembles of deterministic NC shifts and independent uniform digital
shifts.  The output is intentionally compact and report-ready.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from mc_problems import all_problems
from nc_stream import apply_digital_shift, mask_from_nc_patterns


PROBLEMS = [
    "european_call",
    "european_put",
    "geometric_asian_call",
    "exp_sum",
    "sin_product",
    "indicator_mean",
    "g_function",
    "rare_event",
]

SHIFT_FAMILIES = {
    "random_shift": None,
    "RS_{11}": (("11",), 2),
    "transition_{01}": (("01",), 2),
    "A3_{101,111}": (("101", "111"), 2),
    "A4_length4": (("1001", "1011", "1101", "1111"), 2),
    "TM_{1}_control": (("1",), 2),
}


def slope(ns: list[int], values: list[float]) -> float:
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.maximum(np.asarray(values, dtype=float), 1e-18))
    return float(-np.polyfit(x, y, 1)[0])


def summarize(estimates: np.ndarray, truth: float) -> dict[str, float]:
    errors = estimates - truth
    abs_errors = np.abs(errors)
    return {
        "mean_estimate": float(np.mean(estimates)),
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "median_abs_error": float(np.median(abs_errors)),
        "q90_abs_error": float(np.quantile(abs_errors, 0.90)),
        "sd_estimate": float(np.std(estimates, ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j-lo", type=int, default=9)
    parser.add_argument("--j-hi", type=int, default=13)
    parser.add_argument("--reps", type=int, default=64)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out", default="output/data/focused_mc_validation.json")
    args = parser.parse_args()

    ns = [1 << j for j in range(args.j_lo, args.j_hi + 1)]
    catalog = all_problems(n_steps=8, integral_dim=6, rare_dim=10)
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for problem_name in PROBLEMS:
        problem = catalog[problem_name]
        d = problem.default_dim
        truth = float(problem.exact())
        base_engine = qmc.Sobol(d=d, scramble=False)
        base_full = base_engine.random_base2(m=args.j_hi)

        # Use nested prefixes.  Every shift family sees the identical base net.
        random_masks = rng.integers(
            0, 1 << args.bits, size=(args.reps, d), dtype=np.uint32
        )
        masks_by_family: dict[str, np.ndarray] = {"random_shift": random_masks}
        for name, spec in SHIFT_FAMILIES.items():
            if spec is None:
                continue
            patterns, k = spec
            masks_by_family[name] = np.vstack(
                [mask_from_nc_patterns(d, args.bits, r, patterns, k) for r in range(args.reps)]
            )

        for n in ns:
            u0 = base_full[:n]
            plain_est = problem.estimate(u0)
            rows.append({
                "problem": problem_name,
                "category": problem.category,
                "dimension": d,
                "n": n,
                "method": "sobol_unshifted",
                "truth": truth,
                "mean_estimate": plain_est,
                "bias": plain_est - truth,
                "rmse": abs(plain_est - truth),
                "median_abs_error": abs(plain_est - truth),
                "q90_abs_error": abs(plain_est - truth),
                "sd_estimate": 0.0,
                "replicates": 1,
            })

            iid_estimates = np.asarray([
                problem.estimate(rng.random((n, d))) for _ in range(args.reps)
            ])
            rows.append({
                "problem": problem_name,
                "category": problem.category,
                "dimension": d,
                "n": n,
                "method": "iid",
                "truth": truth,
                **summarize(iid_estimates, truth),
                "replicates": args.reps,
            })

            for method, masks in masks_by_family.items():
                estimates = np.asarray([
                    problem.estimate(apply_digital_shift(u0, masks[r], args.bits))
                    for r in range(args.reps)
                ])
                rows.append({
                    "problem": problem_name,
                    "category": problem.category,
                    "dimension": d,
                    "n": n,
                    "method": method,
                    "truth": truth,
                    **summarize(estimates, truth),
                    "replicates": args.reps,
                })

    # Method/problem convergence summaries and final-N ratios against random shift.
    summaries: list[dict] = []
    for problem_name in PROBLEMS:
        for method in ["iid", "sobol_unshifted", *SHIFT_FAMILIES.keys()]:
            sub = [r for r in rows if r["problem"] == problem_name and r["method"] == method]
            sub.sort(key=lambda r: r["n"])
            rmses = [r["rmse"] for r in sub]
            ref = next(
                r for r in rows
                if r["problem"] == problem_name
                and r["method"] == "random_shift"
                and r["n"] == ns[-1]
            )
            summaries.append({
                "problem": problem_name,
                "method": method,
                "alpha_rmse": slope(ns, rmses),
                "final_rmse": rmses[-1],
                "final_ratio_to_random": float(rmses[-1] / ref["rmse"]),
            })

    nc_methods = [m for m in SHIFT_FAMILIES if m not in ("random_shift", "TM_{1}_control")]
    nc_ratios = [
        s["final_ratio_to_random"] for s in summaries
        if s["method"] in nc_methods
    ]
    nc_wins = sum(r < 1.0 for r in nc_ratios)
    global_summary = {
        "nc_problem_family_cells": len(nc_ratios),
        "nc_cells_better_than_random": nc_wins,
        "nc_win_rate": nc_wins / len(nc_ratios),
        "geometric_mean_nc_ratio_to_random": float(math.exp(np.mean(np.log(nc_ratios)))),
        "median_nc_ratio_to_random": float(np.median(nc_ratios)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "problems": PROBLEMS,
            "ns": ns,
            "reps": args.reps,
            "bits": args.bits,
            "seed": args.seed,
            "paired_unscrambled_sobol_base": True,
        },
        "global": global_summary,
        "summaries": summaries,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(global_summary, indent=2))
    print(f"wrote {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
