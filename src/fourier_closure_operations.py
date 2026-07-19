#!/usr/bin/env python3
"""Fourier / Fejér closure analysis under multiple sequence operations.

Operations on pattern sequences (k=2) or raw ±1 streams:
  identity, product, self_product, add_pattern, remove_pattern,
  interleave, interleave_self, even_subseq, odd_subseq, inverse.

Uses exact recursive gamma when the output is a pattern-counting sequence;
otherwise high-N prefix autocorrelation (FFT) for interleaved / subsampled streams.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments import count_pattern_set, gamma_prefix, sequence_values
from recursive_exact_experiments import named_catalog, recursive_gamma_values


EPS = 5e-3
PREFIX_N = 1 << 18


def fejer_density(gamma: np.ndarray, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    m_max = len(gamma) - 1
    theta = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dens = np.full(n_grid, float(np.real(gamma[0])), dtype=np.float64)
    for m in range(1, m_max + 1):
        w = 1.0 - (m / (m_max + 1))
        dens += 2.0 * w * np.real(gamma[m] * np.exp(-1j * m * theta))
    return theta, np.maximum(dens, 0.0)


def fejer_metrics(dens: np.ndarray) -> dict[str, float]:
    mu = float(np.mean(dens))
    if mu <= 0:
        return {"peak_to_mean": 0.0, "var_to_mean2": 0.0, "entropy": 0.0}
    p = dens / float(np.sum(dens))
    ent = -sum(x * math.log(x) for x in p.tolist() if x > 0)
    return {
        "peak_to_mean": float(np.max(dens) / mu),
        "var_to_mean2": float(np.mean((dens - mu) ** 2) / (mu * mu)),
        "entropy": float(ent),
    }


def gmax_pattern(patterns: tuple[str, ...], k: int = 2, m_max: int = 64) -> float:
    gamma = recursive_gamma_values(patterns, k, m_max)
    return float(max(abs(z) for z in gamma[1:])) if len(gamma) > 1 else 0.0


def gmax_values(values: np.ndarray, m_max: int = 64) -> float:
    g = gamma_prefix(values, m_max)
    return float(max(abs(x) for x in g[1:])) if len(g) > 1 else 0.0


def gamma_pattern(patterns: tuple[str, ...], k: int = 2, m_max: int = 64) -> np.ndarray:
    return np.array(recursive_gamma_values(patterns, k, m_max), dtype=np.complex128)


def gamma_values(values: np.ndarray, m_max: int = 64) -> np.ndarray:
    return np.array(gamma_prefix(values, m_max), dtype=np.float64)


def analyze_gamma(gamma: np.ndarray, n_grid: int = 512) -> dict[str, float]:
    met = fejer_metrics(fejer_density(gamma, n_grid)[1])
    met["gmax"] = float(max(abs(gamma[m]) for m in range(1, len(gamma))))
    return met


def interleave(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = min(len(f), len(g))
    out = np.empty(2 * n, dtype=np.int8)
    out[0::2] = f[:n]
    out[1::2] = g[:n]
    return out


def product_patterns(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    return a + b


@dataclass
class OpResult:
    operation: str
    input_a: str
    input_b: str
    output_desc: str
    patterns_out: tuple[str, ...] | None
    g_in_a: float
    g_in_b: float | None
    g_out: float
    nc_in_a: bool
    nc_in_b: bool | None
    nc_out: bool
    preserves_nc: bool
    peak_in_a: float
    peak_out: float
    fejer_flat_in: bool
    fejer_flat_out: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "input_a": self.input_a,
            "input_b": self.input_b,
            "output_desc": self.output_desc,
            "patterns_out": list(self.patterns_out) if self.patterns_out else None,
            "g_in_a": self.g_in_a,
            "g_in_b": self.g_in_b,
            "g_out": self.g_out,
            "nc_in_a": self.nc_in_a,
            "nc_in_b": self.nc_in_b,
            "nc_out": self.nc_out,
            "preserves_nc": self.preserves_nc,
            "peak_in_a": self.peak_in_a,
            "peak_out": self.peak_out,
            "fejer_flat_in": self.fejer_flat_in,
            "fejer_flat_out": self.fejer_flat_out,
        }


def _result_from_pattern(
    operation: str,
    name_a: str,
    patterns_a: tuple[str, ...],
    name_b: str,
    patterns_b: tuple[str, ...] | None,
    patterns_out: tuple[str, ...],
    output_desc: str,
    g_a: float,
    g_b: float | None,
    met_a: dict[str, float],
    met_out: dict[str, float],
) -> OpResult:
    nc_a = g_a <= EPS
    nc_b = g_b <= EPS if g_b is not None else None
    nc_out = met_out["gmax"] <= EPS
    preserve = nc_a if nc_b is None else (nc_a and nc_b and nc_out)
    if nc_b is None:
        preserve = nc_a and nc_out
    return OpResult(
        operation=operation,
        input_a=name_a,
        input_b=name_b or "",
        output_desc=output_desc,
        patterns_out=patterns_out,
        g_in_a=g_a,
        g_in_b=g_b,
        g_out=met_out["gmax"],
        nc_in_a=nc_a,
        nc_in_b=nc_b,
        nc_out=nc_out,
        preserves_nc=preserve,
        peak_in_a=met_a["peak_to_mean"],
        peak_out=met_out["peak_to_mean"],
        fejer_flat_in=met_a["peak_to_mean"] < 1.01,
        fejer_flat_out=met_out["peak_to_mean"] < 1.01,
    )


def _result_from_values(
    operation: str,
    name_a: str,
    name_b: str,
    values: np.ndarray,
    g_a: float,
    g_b: float | None,
    met_a: dict[str, float],
    output_desc: str,
    m_max: int = 64,
) -> OpResult:
    met_out = analyze_gamma(gamma_values(values, m_max))
    nc_a = g_a <= EPS
    nc_b = g_b <= EPS if g_b is not None else None
    nc_out = met_out["gmax"] <= EPS
    preserve = nc_a and (nc_b if nc_b is not None else True) and nc_out
    return OpResult(
        operation=operation,
        input_a=name_a,
        input_b=name_b or "",
        output_desc=output_desc,
        patterns_out=None,
        g_in_a=g_a,
        g_in_b=g_b,
        g_out=met_out["gmax"],
        nc_in_a=nc_a,
        nc_in_b=nc_b,
        nc_out=nc_out,
        preserves_nc=preserve,
        peak_in_a=met_a["peak_to_mean"],
        peak_out=met_out["peak_to_mean"],
        fejer_flat_in=met_a["peak_to_mean"] < 1.01,
        fejer_flat_out=met_out["peak_to_mean"] < 1.01,
    )


def run_all_operations(catalog: dict[str, tuple[str, ...]], m_max: int, prefix_n: int) -> list[OpResult]:
    rows: list[OpResult] = []
    names = sorted(catalog.keys())

    def metrics_a(name: str, pats: tuple[str, ...]) -> tuple[float, dict[str, float]]:
        g = gmax_pattern(pats, 2, m_max)
        met = analyze_gamma(gamma_pattern(pats, 2, m_max))
        return g, met

    # 1. Identity (baseline)
    for name in names:
        pats = catalog[name]
        g, met = metrics_a(name, pats)
        rows.append(
            _result_from_pattern("identity", name, pats, "", None, pats, "A", g, None, met, met)
        )

    # 2. Product (multiset)
    for na, nb in itertools.combinations(names, 2):
        pa, pb = catalog[na], catalog[nb]
        g_a, met_a = metrics_a(na, pa)
        g_b, _ = metrics_a(nb, pb)
        pout = product_patterns(pa, pb)
        met_out = analyze_gamma(gamma_pattern(pout, 2, m_max))
        rows.append(
            _result_from_pattern(
                "product", na, pa, nb, pb, pout, f"{na}||{nb}", g_a, g_b, met_a, met_out
            )
        )

    # 3. Self-product (multiset double)
    for name in names:
        pats = catalog[name]
        g, met = metrics_a(name, pats)
        pout = pats + pats
        met_out = analyze_gamma(gamma_pattern(pout, 2, m_max))
        rows.append(
            _result_from_pattern("self_product", name, pats, name, pats, pout, f"2x{name}", g, g, met, met_out)
        )

    # 4. Add one pattern (superset) from catalog words
    all_words = sorted({w for pats in catalog.values() for w in pats})
    for name in names:
        pats = catalog[name]
        g, met = metrics_a(name, pats)
        for w in all_words:
            if w in pats:
                continue
            pout = pats + (w,)
            met_out = analyze_gamma(gamma_pattern(pout, 2, m_max))
            rows.append(
                _result_from_pattern(
                    "add_pattern", name, pats, "", None, pout, f"{name}+{{{w}}}", g, None, met, met_out
                )
            )

    # 5. Remove one pattern (subset)
    for name in names:
        pats = catalog[name]
        if len(pats) <= 1:
            continue
        g, met = metrics_a(name, pats)
        for i in range(len(pats)):
            pout = pats[:i] + pats[i + 1 :]
            met_out = analyze_gamma(gamma_pattern(pout, 2, m_max))
            rows.append(
                _result_from_pattern(
                    "remove_pattern", name, pats, "", None, pout, f"{name}-{{{pats[i]}}}", g, None, met, met_out
                )
            )

    # 6. Interleave
    half = prefix_n // 2
    for na, nb in itertools.combinations_with_replacement(names, 2):
        pa, pb = catalog[na], catalog[nb]
        g_a, met_a = metrics_a(na, pa)
        g_b, _ = metrics_a(nb, pb)
        va = sequence_values(pa, half)
        vb = sequence_values(pb, half)
        vout = interleave(va, vb)
        op = "interleave_self" if na == nb else "interleave"
        rows.append(
            _result_from_values(
                op, na, nb, vout, g_a, g_b, met_a, f"interleave({na},{nb})", m_max
            )
        )

    # 7. Even / odd subsequence
    for name in names:
        pats = catalog[name]
        g, met = metrics_a(name, pats)
        v = sequence_values(pats, prefix_n)
        rows.append(
            _result_from_values(
                "even_subseq", name, "", v[0::2], g, None, met, f"even({name})", m_max
            )
        )
        rows.append(
            _result_from_values(
                "odd_subseq", name, "", v[1::2], g, None, met, f"odd({name})", m_max
            )
        )

    # 8. Inverse / conjugation (identical for real ±1)
    for name in names:
        pats = catalog[name]
        g, met = metrics_a(name, pats)
        rows.append(
            _result_from_pattern("inverse", name, pats, "", None, pats, f"conj({name})", g, None, met, met)
        )

    return rows


def summarize_by_operation(rows: list[OpResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for op in sorted({r.operation for r in rows}):
        sub = [r for r in rows if r.operation == op]
        nc_inputs = [r for r in sub if r.nc_in_a and (r.nc_in_b is None or r.nc_in_b)]
        closed = [r for r in nc_inputs if r.preserves_nc]
        flat_in_flat_out = [
            r for r in nc_inputs if r.fejer_flat_in and r.fejer_flat_out and r.preserves_nc
        ]
        flat_in_bad_out = [
            r for r in nc_inputs if r.fejer_flat_in and not r.fejer_flat_out
        ]
        out[op] = {
            "total": len(sub),
            "nc_input_cases": len(nc_inputs),
            "preserves_nc": len(closed),
            "closure_rate": len(closed) / len(nc_inputs) if nc_inputs else float("nan"),
            "flat_in_flat_out": len(flat_in_flat_out),
            "flat_in_nonflat_out": len(flat_in_bad_out),
            "worst_nonflat": (
                max(
                    (r for r in nc_inputs if not r.preserves_nc),
                    key=lambda r: r.g_out,
                ).to_dict()
                if any(not r.preserves_nc for r in nc_inputs)
                else None
            ),
            "best_example": (
                min(closed, key=lambda r: r.g_out).to_dict() if closed else None
            ),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fourier closure under multiple operations.")
    p.add_argument("--m-max", type=int, default=64)
    p.add_argument("--prefix-n", type=int, default=PREFIX_N)
    p.add_argument("--out-json", default="fourier_closure_operations_results.json")
    p.add_argument("--out-csv", default="fourier_closure_operations_results.csv")
    args = p.parse_args()

    catalog = named_catalog()
    rows = run_all_operations(catalog, args.m_max, args.prefix_n)
    summary = summarize_by_operation(rows)

    summary_json: dict[str, Any] = summary

    payload = {
        "config": {
            "eps": EPS,
            "m_max": args.m_max,
            "prefix_n": args.prefix_n,
            "catalog_size": len(catalog),
        },
        "by_operation": summary_json,
        "rows": [r.to_dict() for r in rows],
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "operation",
        "input_a",
        "input_b",
        "output_desc",
        "g_in_a",
        "g_in_b",
        "g_out",
        "nc_in_a",
        "nc_in_b",
        "nc_out",
        "preserves_nc",
        "peak_in_a",
        "peak_out",
        "fejer_flat_in",
        "fejer_flat_out",
    ]
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())

    print(f"Wrote {args.out_json} ({len(rows)} rows)")
    print("\nClosure summary (NC inputs -> NC output):")
    for op, sm in summary_json.items():
        rate = sm.get("closure_rate", float("nan"))
        rate_s = f"{100*rate:.0f}%" if rate == rate else "n/a"
        print(
            f"  {op:18s}  nc_cases={sm['nc_input_cases']:4d}  "
            f"closed={sm['preserves_nc']:4d}  rate={rate_s}  "
            f"flat->nonflat={sm['flat_in_nonflat_out']}"
        )


if __name__ == "__main__":
    main()
