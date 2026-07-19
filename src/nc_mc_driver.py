#!/usr/bin/env python3
"""Unified MC/QMC driver construction from NC pattern sets and baselines."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from nc_stream import (
    apply_digital_shift,
    build_nc_uniforms_from_patterns,
    mask_from_nc_patterns,
    sobol_uniforms,
)

NAMED_NC: dict[str, dict[str, Any]] = {
    "nc_rs": {"patterns": ("11",), "k": 2},
    "nc_tm": {"patterns": ("1",), "k": 2},
    "nc_k4": {"patterns": ("01", "10"), "k": 4},
    "nc_k4a": {"patterns": ("001", "011", "100", "110"), "k": 4},
    "nc_k4b": {"patterns": ("1", "11", "011", "111"), "k": 4},
}


def build_uniforms(
    method: str,
    n_paths: int,
    dim: int,
    bits: int,
    seed: int,
    rep: int = 0,
    patterns: tuple[str, ...] | None = None,
    k: int | None = None,
) -> np.ndarray:
    if method == "iid":
        rng = np.random.default_rng(seed)
        return rng.random((n_paths, dim))

    if method == "iid_antithetic":
        rng = np.random.default_rng(seed)
        n_half = (n_paths + 1) // 2
        u = rng.random((n_half, dim))
        return np.vstack([u, 1.0 - u])[:n_paths]

    if method == "sobol":
        from scipy.stats import qmc  # type: ignore

        eng = qmc.Sobol(d=dim, scramble=True, seed=seed)
        return eng.random(n=n_paths)

    if method == "halton":
        from scipy.stats import qmc  # type: ignore

        eng = qmc.Halton(d=dim, scramble=True, seed=seed)
        return eng.random(n=n_paths)

    if method == "sobol_rand_shift":
        u0 = sobol_uniforms(n_paths, dim, seed=seed)
        rng = np.random.default_rng(seed + 31)
        masks = rng.integers(0, 1 << bits, size=dim, dtype=np.uint32)
        return apply_digital_shift(u0, masks, bits)

    if method in NAMED_NC:
        spec = NAMED_NC[method]
        return build_nc_uniforms_from_patterns(
            n_paths, dim, bits, spec["patterns"], spec["k"], seed=seed
        )

    if method.startswith("sobol_nc_") and method.endswith("_shift") and method != "sobol_nc_shift":
        name = method[len("sobol_nc_") : -len("_shift")]
        key = f"nc_{name}" if name in ("rs", "tm", "k4", "k4a", "k4b") else None
        if key and key in NAMED_NC:
            spec = NAMED_NC[key]
            u0 = sobol_uniforms(n_paths, dim, seed=seed)
            masks = mask_from_nc_patterns(dim, bits, rep, spec["patterns"], spec["k"])
            return apply_digital_shift(u0, masks, bits)
        raise ValueError(f"Unknown named NC shift {method}")

    if method.startswith("sobol_nc_shift|") or method == "sobol_nc_shift":
        if patterns is None or k is None:
            raise ValueError("patterns and k required for sobol_nc_shift")
        u0 = sobol_uniforms(n_paths, dim, seed=seed)
        masks = mask_from_nc_patterns(dim, bits, rep, patterns, k)
        return apply_digital_shift(u0, masks, bits)

    if method == "nc_direct":
        if patterns is None or k is None:
            raise ValueError("patterns and k required for nc_direct")
        return build_nc_uniforms_from_patterns(n_paths, dim, bits, patterns, k, seed=seed)

    raise ValueError(f"Unknown method {method}")


def standard_methods(include_halton: bool = True, include_antithetic: bool = True) -> list[str]:
    methods = [
        "iid",
        "sobol",
        "sobol_rand_shift",
        "sobol_nc_rs_shift",
        "sobol_nc_tm_shift",
        "sobol_nc_k4a_shift",
        "nc_rs",
        "nc_tm",
        "nc_k4",
        "nc_k4a",
    ]
    if include_antithetic:
        methods.insert(1, "iid_antithetic")
    if include_halton:
        methods.insert(2, "halton")
    return methods


def load_top_nc_from_results(path: str, top_n: int, mode: str = "sobol_nc_shift") -> list[dict]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    rows = [r for r in data.get("rows", []) if r.get("mode") == mode and r.get("patterns_key")]
    rows.sort(key=lambda r: r["abs_error"])
    seen: set[tuple[str, ...]] = set()
    out: list[dict] = []
    for r in rows:
        pats = tuple(r.get("patterns") or r["patterns_key"].split())
        if pats in seen:
            continue
        seen.add(pats)
        out.append(
            {
                "driver_id": r["driver_id"],
                "patterns": pats,
                "patterns_key": r["patterns_key"],
                "k": int(r["k"]),
            }
        )
        if len(out) >= top_n:
            break
    return out
