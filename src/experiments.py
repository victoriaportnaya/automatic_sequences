#!/usr/bin/env python3
"""Experiments on self-correlation of binary automatic pattern sequences.

The sequence attached to a finite pattern set A is
    a_A(n) = (-1)^{#(A, n)},
where #(A, n) counts (with overlaps) occurrences of each pattern in
the binary expansion of n padded with sufficiently many leading zeros.

Convention used here:
- If max pattern length in A is L, we represent n as
      0^(L-1) || bin(n).
- This guarantees boundary occurrences are counted, e.g. for A containing
  "01", the number 1 contributes one occurrence from "01".
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def all_patterns_upto(max_len: int) -> list[str]:
    patterns: list[str] = []
    for length in range(1, max_len + 1):
        for bits in itertools.product("01", repeat=length):
            word = "".join(bits)
            if set(word) == {"0"}:
                continue
            patterns.append(word)
    return patterns


def count_occurrences_with_overlaps(text: str, pattern: str) -> int:
    plen = len(pattern)
    return sum(1 for i in range(len(text) - plen + 1) if text[i : i + plen] == pattern)


def padded_binary_expansion(n: int, max_pattern_len: int) -> str:
    """Binary expansion with sufficient leading zeros for boundary counting."""
    return "0" * max(0, max_pattern_len - 1) + format(n, "b")


def count_pattern_set(n: int, patterns: tuple[str, ...]) -> int:
    if not patterns:
        return 0
    max_len = max(len(p) for p in patterns)
    padded = padded_binary_expansion(n, max_len)
    return sum(count_occurrences_with_overlaps(padded, p) for p in patterns)


def sequence_values(patterns: tuple[str, ...], n_max: int) -> np.ndarray:
    vals = np.empty(n_max, dtype=np.int8)
    for n in range(n_max):
        vals[n] = -1 if count_pattern_set(n, patterns) % 2 else 1
    return vals


def gamma_prefix(values: np.ndarray, m_max: int) -> list[float]:
    n = len(values)
    a = values.astype(np.float64)
    # Full linear autocorrelation via FFT convolution.
    rev = a[::-1]
    fft_len = 1
    need = 2 * n - 1
    while fft_len < need:
        fft_len <<= 1
    fa = np.fft.rfft(a, fft_len)
    fb = np.fft.rfft(rev, fft_len)
    conv = np.fft.irfft(fa * fb, fft_len)
    conv = np.rint(conv[:need]).astype(np.int64)
    zero_idx = n - 1
    out = []
    for m in range(m_max + 1):
        if m == 0:
            out.append(1.0)
        else:
            out.append(float(conv[zero_idx - m]) / float(n))
    return out


@dataclass
class SequenceReport:
    name: str
    patterns: tuple[str, ...]
    gamma: list[float]

    def max_abs_nonzero(self) -> float:
        if len(self.gamma) <= 1:
            return 0.0
        return max(abs(x) for x in self.gamma[1:])

    def avg_sq_nonzero(self) -> float:
        if len(self.gamma) <= 1:
            return 0.0
        arr = np.array(self.gamma[1:], dtype=np.float64)
        return float(np.mean(arr * arr))

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "patterns": list(self.patterns),
            "max_abs_nonzero": self.max_abs_nonzero(),
            "avg_sq_nonzero": self.avg_sq_nonzero(),
            "gamma_prefix": self.gamma,
        }


def run_named_sequences(n: int, m_max: int) -> list[SequenceReport]:
    specs = [
        ("Thue-Morse", ("1",)),
        ("Rudin-Shapiro", ("11",)),
        ("Rudin-Shapiro * Thue-Morse", ("11", "1")),
        ("Pattern {101,111}", ("101", "111")),
    ]
    reports = []
    for name, pats in specs:
        vals = sequence_values(pats, n)
        reports.append(SequenceReport(name=name, patterns=pats, gamma=gamma_prefix(vals, m_max)))
    return reports


def exhaustive_upto_len3(n: int, m_max: int, keep_top: int = 12) -> dict:
    patterns = all_patterns_upto(3)
    best: list[SequenceReport] = []
    approx_count = 0
    total = 1 << len(patterns)
    for mask in range(total):
        pats = tuple(patterns[i] for i in range(len(patterns)) if (mask >> i) & 1)
        vals = sequence_values(pats, n)
        rep = SequenceReport(name=f"subset_mask_{mask}", patterns=pats, gamma=gamma_prefix(vals, m_max))
        if rep.max_abs_nonzero() < 0.05:
            approx_count += 1
        best.append(rep)
    best.sort(key=lambda r: (r.max_abs_nonzero(), r.avg_sq_nonzero(), len(r.patterns)))
    return {
        "num_patterns": len(patterns),
        "total_subsets": total,
        "approx_noncorr_count_threshold_0p05": approx_count,
        "top_candidates": [r.to_json() for r in best[:keep_top]],
    }


def dilation_invariant_family(n: int, m_max: int, ell_values: list[int]) -> list[dict]:
    out = []
    for ell in ell_values:
        if ell < 2:
            continue
        core = tuple("1" + mid + "1" for mid in map("".join, itertools.product("01", repeat=ell - 2)))
        vals = sequence_values(core, n)
        gam = gamma_prefix(vals, m_max)
        rep = SequenceReport(name=f"1{{0,1}}^{{{ell-2}}}1", patterns=core, gamma=gam)
        out.append(rep.to_json())
    return out


def random_len4_sample(n: int, m_max: int, sample_size: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    patterns = all_patterns_upto(4)
    scores = []
    for _ in range(sample_size):
        mask = rng.integers(0, 2, size=len(patterns), dtype=np.int8)
        pats = tuple(p for p, bit in zip(patterns, mask) if int(bit))
        vals = sequence_values(pats, n)
        rep = SequenceReport(name="random", patterns=pats, gamma=gamma_prefix(vals, m_max))
        scores.append(rep.max_abs_nonzero())
    arr = np.array(scores, dtype=np.float64)
    return {
        "sample_size": sample_size,
        "quantiles_max_abs_nonzero": {
            "q10": float(np.quantile(arr, 0.10)),
            "q50": float(np.quantile(arr, 0.50)),
            "q90": float(np.quantile(arr, 0.90)),
            "q99": float(np.quantile(arr, 0.99)),
        },
        "min_max_abs_nonzero": float(arr.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1 << 14, help="prefix length N for empirical correlations")
    parser.add_argument("--m-max", type=int, default=64, help="compute gamma(m) for 0 <= m <= m_max")
    parser.add_argument("--random-sample", type=int, default=80, help="number of random length<=4 pattern sets")
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    # Convention self-check: expansion of 1 must include boundary pattern "01".
    assert count_pattern_set(1, ("01",)) == 1, (
        "Leading-zero convention check failed: expected #({01},1)=1"
    )

    result = {
        "config": {
            "N": args.n,
            "m_max": args.m_max,
            # For real-valued sequences this equals (1/N) sum a(n)a(n+m).
            "definition": "gamma_N(m) = (1/N) * sum_{n=0}^{N-1} a(n) * conj(a(n+m))",
            "leading_zero_convention": "use 0^(L-1)||bin(n), where L=max pattern length",
        },
        "named_sequences": [r.to_json() for r in run_named_sequences(args.n, args.m_max)],
        "exhaustive_len_le_3": exhaustive_upto_len3(args.n, args.m_max),
        "dilation_invariant_core_family": dilation_invariant_family(args.n, args.m_max, [2, 3, 4, 5, 6]),
        "random_len_le_4": random_len4_sample(args.n, args.m_max, args.random_sample),
    }

    args.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
