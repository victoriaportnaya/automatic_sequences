#!/usr/bin/env python3
"""Exhaustive k-NC census via recursive exact gamma(m).

Enumerates binary pattern-set families used in the dichotomy search (846 sets)
and optionally all non-empty subsets of patterns with max length <= 3 (2047 sets).

For each (A, k) with k in {2,...,8}, computes
    G_{A,k} = max_{1<=m<=m_max} |gamma_{A,k}(m)|
using recursive_gamma_values from recursive_exact_experiments.py.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

from experiments import all_patterns_upto
from recursive_exact_experiments import (
    count_pattern_set_fixed_padding,
    max_abs_nonzero,
    recursive_gamma_values,
)


def patterns_len_le(max_len: int) -> list[str]:
    return [w for w in all_patterns_upto(max_len) if "1" in w]


def enumerate_census_846() -> list[tuple[str, ...]]:
    """Dichotomy census: sizes 1--2 from words len<=4; sizes 3--4 from words len<=3."""
    words4 = patterns_len_le(4)
    words3 = patterns_len_le(3)
    out: list[tuple[str, ...]] = []
    for r in (1, 2):
        out.extend(itertools.combinations(words4, r))
    for r in (3, 4):
        out.extend(itertools.combinations(words3, r))
    return out


def enumerate_all_len_le_3() -> list[tuple[str, ...]]:
    patterns = patterns_len_le(3)
    out: list[tuple[str, ...]] = []
    n = len(patterns)
    for mask in range(1, 1 << n):
        out.append(tuple(patterns[i] for i in range(n) if (mask >> i) & 1))
    return out


def pattern_key(patterns: tuple[str, ...]) -> str:
    return " ".join(patterns)


def evaluate_family(
    family: list[tuple[str, ...]],
    ks: list[int],
    m_max: int,
    threshold: float,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    by_k: dict[int, list[dict]] = {k: [] for k in ks}
    t0 = time.perf_counter()

    for i, patterns in enumerate(family):
        scores: dict[int, float] = {}
        for k in ks:
            gamma = recursive_gamma_values(patterns, k, m_max)
            g = float(max_abs_nonzero(gamma))
            scores[k] = g
            is_nc = g <= threshold
            row = {
                "patterns": list(patterns),
                "patterns_key": pattern_key(patterns),
                "k": k,
                "gmax": g,
                "is_nc": is_nc,
                "max_pattern_len": max(len(p) for p in patterns),
                "set_size": len(patterns),
            }
            rows.append(row)
            if is_nc:
                by_k[k].append(row)

        if (i + 1) % 100 == 0 or i + 1 == len(family):
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(family)}] elapsed {elapsed:.1f}s", flush=True)

    summary = {
        "family_size": len(family),
        "k_values": ks,
        "m_max": m_max,
        "threshold": threshold,
        "elapsed_sec": time.perf_counter() - t0,
        "nc_counts": {str(k): len(by_k[k]) for k in ks},
        "nc_by_k": {
            str(k): [
                {"patterns": r["patterns"], "gmax": r["gmax"]}
                for r in sorted(by_k[k], key=lambda x: x["gmax"])
            ]
            for k in ks
        },
    }
    return rows, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Recursive exact exhaustive k-NC sweep.")
    p.add_argument(
        "--family",
        choices=["census846", "len_le_3", "both"],
        default="census846",
        help="Pattern family to enumerate (census846 = dichotomy 846 sets).",
    )
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--m-max", type=int, default=64)
    p.add_argument("--threshold", type=float, default=5e-3)
    p.add_argument("--out-json", default="recursive_exhaustive_nc_results.json")
    p.add_argument("--out-csv", default="recursive_exhaustive_nc_results.csv")
    args = p.parse_args()

    assert count_pattern_set_fixed_padding(1, ("01",), 2) == 1

    ks = list(range(args.k_min, args.k_max + 1))
    families: dict[str, list[tuple[str, ...]]] = {}
    if args.family in ("census846", "both"):
        families["census846"] = enumerate_census_846()
    if args.family in ("len_le_3", "both"):
        families["len_le_3"] = enumerate_all_len_le_3()

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for fam_name, fam in families.items():
        print(f"Evaluating {fam_name}: {len(fam)} pattern sets, k={ks[0]}..{ks[-1]}")
        rows, summary = evaluate_family(fam, ks, args.m_max, args.threshold)
        summary["family_name"] = fam_name
        summaries[fam_name] = summary
        for r in rows:
            r["family"] = fam_name
        all_rows.extend(rows)

        print(f"  NC counts for {fam_name}:")
        for k in ks:
            print(f"    k={k}: {summary['nc_counts'][str(k)]} / {len(fam)}")

    payload = {
        "config": {
            "method": "recursive_exact gamma(m,r)",
            "families": list(families.keys()),
            "k_values": ks,
            "m_max": args.m_max,
            "threshold": args.threshold,
        },
        "summaries": summaries,
        "rows": all_rows,
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "family",
        "patterns_key",
        "k",
        "gmax",
        "is_nc",
        "set_size",
        "max_pattern_len",
        "patterns",
    ]
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"Wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
