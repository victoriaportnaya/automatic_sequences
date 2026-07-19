#!/usr/bin/env python3
"""Generic NC sequence -> uniform [0,1) streams for QMC/MC."""
from __future__ import annotations

import math

import numpy as np

from experiments import count_pattern_set


def nc_bit_from_patterns(n: int, patterns: tuple[str, ...], k: int, b: int) -> int:
    c = count_pattern_set(n, patterns)
    if k == 2:
        return c & 1
    d = c % k
    if k == 4:
        return (d >> (b & 1)) & 1
    nbits = max(1, (k - 1).bit_length())
    return (d >> (b % nbits)) & 1


def build_nc_uniforms_from_patterns(
    n_paths: int,
    n_steps: int,
    bits_per_u: int,
    patterns: tuple[str, ...],
    k: int,
    seed: int = 0,
) -> np.ndarray:
    out = np.empty((n_paths, n_steps), dtype=np.float64)
    scale = float(1 << bits_per_u)
    stride_i = 131
    stride_j = 8191
    stride_b = 524287
    for i in range(n_paths):
        for j in range(n_steps):
            acc = 0
            base = seed + i * stride_i + j * stride_j
            for b in range(bits_per_u):
                idx = base + b * stride_b
                bit = nc_bit_from_patterns(idx, patterns, k, b)
                acc = (acc << 1) | int(bit)
            out[i, j] = (acc + 0.5) / scale
    return out


def mask_from_nc_patterns(
    d: int,
    bits: int,
    rep: int,
    patterns: tuple[str, ...],
    k: int,
) -> np.ndarray:
    masks = np.zeros(d, dtype=np.uint32)
    stride_d = 131071
    stride_b = 8191
    base = rep * 104729
    for j in range(d):
        acc = 0
        off = base + j * stride_d
        for b in range(bits):
            idx = off + b * stride_b
            bit = nc_bit_from_patterns(idx, patterns, k, b)
            acc = (acc << 1) | int(bit)
        masks[j] = np.uint32(acc)
    return masks


def apply_digital_shift(u: np.ndarray, masks: np.ndarray, bits: int) -> np.ndarray:
    scale = float(1 << bits)
    q = np.floor(np.clip(u, 0.0, 1.0 - 1e-15) * scale).astype(np.uint32)
    q ^= masks.reshape(1, -1)
    return (q.astype(np.float64) + 0.5) / scale


def sobol_uniforms(n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc  # type: ignore

    m = int(math.log2(n_paths))
    if (1 << m) != n_paths:
        raise ValueError("n_paths must be a power of two for Sobol")
    eng = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
    return eng.random_base2(m=m)
