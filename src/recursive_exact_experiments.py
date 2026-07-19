#!/usr/bin/env python3
"""Exact recursive correlation experiments (share/noncorrelated.nb style).

Implements the recursion for gamma(m,r) used in share/noncorrelated.nb and
evaluates gamma(m)=2^{-l} sum_r gamma(m,r) for binary pattern sequences.

This version also supports complex-rooted values:
    a_q(n) = exp(2*pi*i/q)^(#(A,n))
while keeping the binary automaton/base recursion (base 2).
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from functools import lru_cache
from pathlib import Path

from experiments import count_occurrences_with_overlaps, padded_binary_expansion


def named_catalog() -> dict[str, tuple[str, ...]]:
    base = {
        "TM": ("1",),
        "RS": ("11",),
        "RS*TM": ("11", "1"),
        "A_{1,101,111}": ("1", "101", "111"),
        "{101,111}": ("101", "111"),
        "{01}": ("01",),
        "{10}": ("10",),
        "{01,10,11}": ("01", "10", "11"),
        "A_3": ("101", "111"),
        "A_4": ("1001", "1011", "1101", "1111"),
        "A_5": (
            "10001",
            "10011",
            "10101",
            "10111",
            "11001",
            "11011",
            "11101",
            "11111",
        ),
    }
    # 4-NC-only families from dichotomy census.
    base.update(
        {
            "K4_01_001_101": ("01", "001", "101"),
            "K4_10_010_110": ("10", "010", "110"),
            "K4_11_011_111": ("11", "011", "111"),
            "K4_01_10": ("01", "10"),
            "K4_01_010_110": ("01", "010", "110"),
            "K4_10_001_101": ("10", "001", "101"),
            "K4_001_010_101_110": ("001", "010", "101", "110"),
            "K4_1_01_10": ("1", "01", "10"),
            "K4_1_01_001_101": ("1", "01", "001", "101"),
            "K4_1_01_010_110": ("1", "01", "010", "110"),
            "K4_1_10_001_101": ("1", "10", "001", "101"),
            "K4_1_10_010_110": ("1", "10", "010", "110"),
            "K4_1_11_011_111": ("1", "11", "011", "111"),
            "K4_001_010_100_111": ("001", "010", "100", "111"),
            "K4_001_011_100_110": ("001", "011", "100", "110"),
        }
    )
    return base


def trailing_ones(x: int, l: int) -> int:
    t = 0
    while t < l and ((x >> t) & 1) == 1:
        t += 1
    return t


def count_pattern_set_fixed_padding(n: int, patterns: tuple[str, ...], l: int) -> int:
    """Count occurrences using explicit 0^(l-1)||bin(n) convention."""
    if not patterns:
        return 0
    padded = padded_binary_expansion(n, l)
    return sum(count_occurrences_with_overlaps(padded, p) for p in patterns)


def compute_b_period(patterns: tuple[str, ...], root_mod: int, l: int) -> list[complex]:
    period = 1 << l
    zeta = cmath.exp(2j * math.pi / root_mod)

    @lru_cache(maxsize=None)
    def a(n: int) -> complex:
        c = count_pattern_set_fixed_padding(n, patterns, l)
        return zeta ** (c % root_mod)

    b = []
    for n in range(period):
        val = a(n) / a(n // 2)
        b.append(val)
    return b


def recursive_gamma_values(patterns: tuple[str, ...], root_mod: int, m_max: int) -> list[complex]:
    l = max(len(p) for p in patterns) if patterns else 1
    period = 1 << l
    b = compute_b_period(patterns, root_mod, l)

    def bval(n: int) -> complex:
        return b[n % period]

    # gamma(1,r) initialization via trailing-ones recursion from share method.
    gamma1: list[complex] = [0.0 + 0.0j for _ in range(period)]
    order = list(range(period))
    order.sort(key=lambda x: (trailing_ones(x, l), x))
    for r in order:
        nu = trailing_ones(r, l)
        pref = bval(r) * bval(r + 1).conjugate()
        if nu == 0:
            gamma1[r] = pref
        elif nu < l:
            a0 = r // 2
            a1 = a0 + (1 << (l - 1))
            gamma1[r] = 0.5 * pref * (gamma1[a0] + gamma1[a1])
        else:
            # implicit equation for r=2^l-1:
            # g = 0.5*pref*(g_prev + g)
            prev = (1 << (l - 1)) - 1
            gamma1[r] = (0.5 * pref * gamma1[prev]) / (1 - 0.5 * pref)

    @lru_cache(maxsize=None)
    def gamma_m_r(m: int, r: int) -> complex:
        if m == 0:
            return 1.0 + 0.0j
        if m == 1:
            return gamma1[r % period]

        r_mod = r % period
        m_next = ((r_mod & 1) + m) // 2
        r0 = r_mod // 2
        r1 = r0 + (1 << (l - 1))
        pref = 0.5 * bval(r_mod) * bval(r_mod + m).conjugate()
        return pref * (gamma_m_r(m_next, r0) + gamma_m_r(m_next, r1))

    out = [1.0 + 0.0j]
    for m in range(1, m_max + 1):
        g = sum(gamma_m_r(m, r) for r in range(period)) / period
        out.append(g)
    return out


def max_abs_nonzero(gamma: list[complex]) -> float:
    if len(gamma) <= 1:
        return 0.0
    return max(abs(z) for z in gamma[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--k-min", type=int, default=2, help="Root modulus min")
    ap.add_argument("--k-max", type=int, default=8, help="Root modulus max")
    ap.add_argument("--out", type=Path, default=Path("recursive_exact_results.json"))
    ap.add_argument("--threshold", type=float, default=5e-3, help="Empirical NC threshold for summary labels")
    args = ap.parse_args()

    ks = list(range(args.k_min, args.k_max + 1))
    catalog = named_catalog()

    # Convention self-check: expansion of 1 must include boundary pattern "01".
    assert count_pattern_set_fixed_padding(1, ("01",), 2) == 1, (
        "Leading-zero convention check failed: expected #({01},1)=1"
    )

    payload: dict[str, object] = {
        "config": {
            "method": "exact recursive gamma(m,r) from share/noncorrelated.nb structure",
            "m_max": args.m_max,
            "k_values": ks,
            "definition": "gamma(m) = lim_N (1/N) sum_{n=0}^{N-1} a(n) * conj(a(n+m))",
            "threshold_empirical_k_nc": args.threshold,
        },
        "catalog_scores": {},
        "catalog_profiles": {},
    }

    for name, patterns in catalog.items():
        scores: dict[str, float] = {}
        profiles: dict[str, list[list[float]]] = {}
        print(f"{name}:", end=" ", flush=True)
        for k in ks:
            gamma = recursive_gamma_values(patterns, k, args.m_max)
            score = max_abs_nonzero(gamma)
            scores[f"k={k}"] = float(score)
            profiles[f"k={k}"] = [[float(z.real), float(z.imag)] for z in gamma]
            print(f"k={k}:{score:.4f}", end=" ", flush=True)
        print()
        payload["catalog_scores"][name] = scores
        payload["catalog_profiles"][name] = profiles

    # quick 2/4/6 status summary
    status: dict[str, dict[str, str]] = {}
    for name, row in payload["catalog_scores"].items():
        status[name] = {}
        for k in (2, 4, 6):
            key = f"k={k}"
            val = row.get(key, 1.0)
            status[name][key] = "Yes" if val <= args.threshold else "No"
    payload["empirical_nc_summary_2_4_6"] = status

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
