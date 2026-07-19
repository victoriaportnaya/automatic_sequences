#!/usr/bin/env python3
"""All-k rigorous Fourier analysis from recursive exact correlations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recursive_exact_experiments import named_catalog, recursive_gamma_values


ROOT = Path(__file__).parent
VIS = ROOT / "main" / "visualizations"
VIS.mkdir(parents=True, exist_ok=True)


def fejer_density(gamma: np.ndarray, n_grid: int) -> np.ndarray:
    m_max = len(gamma) - 1
    theta = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dens = np.full(n_grid, float(np.real(gamma[0])), dtype=np.float64)
    for m in range(1, m_max + 1):
        w = 1.0 - (m / (m_max + 1))
        dens += 2.0 * w * np.real(gamma[m] * np.exp(-1j * m * theta))
    return np.maximum(dens, 0.0)


def metrics(dens: np.ndarray) -> dict[str, float]:
    mu = float(np.mean(dens))
    if mu <= 0:
        return {"peak_to_mean": 0.0, "var_to_mean2": 0.0, "entropy": 0.0}
    p = dens / float(np.sum(dens))
    ent = 0.0
    for x in p.tolist():
        if x > 0:
            ent -= x * math.log(x)
    return {
        "peak_to_mean": float(np.max(dens) / mu),
        "var_to_mean2": float(np.mean((dens - mu) ** 2) / (mu * mu)),
        "entropy": float(ent),
    }


def plot_nc_counts(counts: dict[int, int], out_path: Path) -> None:
    ks = sorted(counts.keys())
    ys = [counts[k] for k in ks]
    plt.figure(figsize=(6.6, 3.8))
    plt.bar([str(k) for k in ks], ys)
    plt.title("Empirical NC counts by k")
    plt.xlabel("k")
    plt.ylabel("count (threshold)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_metric_by_k(rows: list[dict], metric: str, out_path: Path) -> None:
    ks = sorted({r["k"] for r in rows})
    nc_means = []
    non_means = []
    for k in ks:
        rk = [r for r in rows if r["k"] == k]
        nc = [r["metrics"][metric] for r in rk if r["is_nc"]]
        nn = [r["metrics"][metric] for r in rk if not r["is_nc"]]
        nc_means.append(float(np.mean(nc)) if nc else np.nan)
        non_means.append(float(np.mean(nn)) if nn else np.nan)

    x = np.array(ks, dtype=float)
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(x, non_means, marker="o", label="not-NC mean")
    plt.plot(x, nc_means, marker="s", label="NC mean")
    plt.title(f"{metric} by k (rigorous Fourier)")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-max", type=int, default=256)
    ap.add_argument("--n-grid", type=int, default=1024)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=5e-3)
    ap.add_argument("--out-json", type=Path, default=Path("rigorous_fourier_all_k_results.json"))
    args = ap.parse_args()

    catalog = named_catalog()
    ks = list(range(args.k_min, args.k_max + 1))

    rows: list[dict] = []
    nc_counts: dict[int, int] = {k: 0 for k in ks}
    for name, patterns in catalog.items():
        for k in ks:
            gamma = np.array(recursive_gamma_values(patterns, k, args.m_max), dtype=np.complex128)
            gmax = float(np.max(np.abs(gamma[1:])))
            is_nc = gmax <= args.threshold
            nc_counts[k] += int(is_nc)
            dens = fejer_density(gamma, args.n_grid)
            rows.append(
                {
                    "name": name,
                    "patterns": list(patterns),
                    "k": k,
                    "gmax": gmax,
                    "is_nc": is_nc,
                    "metrics": metrics(dens),
                }
            )

    # Aggregate
    aggregate: dict[str, dict] = {}
    for k in ks:
        rk = [r for r in rows if r["k"] == k]
        nc = [r for r in rk if r["is_nc"]]
        nn = [r for r in rk if not r["is_nc"]]
        aggregate[f"k={k}"] = {
            "count_total": len(rk),
            "count_nc": len(nc),
            "count_not_nc": len(nn),
            "nc_metrics_mean": {
                "peak_to_mean": float(np.mean([r["metrics"]["peak_to_mean"] for r in nc])) if nc else None,
                "var_to_mean2": float(np.mean([r["metrics"]["var_to_mean2"] for r in nc])) if nc else None,
                "entropy": float(np.mean([r["metrics"]["entropy"] for r in nc])) if nc else None,
            },
            "not_nc_metrics_mean": {
                "peak_to_mean": float(np.mean([r["metrics"]["peak_to_mean"] for r in nn])) if nn else None,
                "var_to_mean2": float(np.mean([r["metrics"]["var_to_mean2"] for r in nn])) if nn else None,
                "entropy": float(np.mean([r["metrics"]["entropy"] for r in nn])) if nn else None,
            },
        }

    # Visualizations
    plot_nc_counts(nc_counts, VIS / "rigorous_fourier_allk_nc_counts.png")
    plot_metric_by_k(rows, "peak_to_mean", VIS / "rigorous_fourier_allk_peak_to_mean.png")
    plot_metric_by_k(rows, "var_to_mean2", VIS / "rigorous_fourier_allk_var_to_mean2.png")
    plot_metric_by_k(rows, "entropy", VIS / "rigorous_fourier_allk_entropy.png")

    payload = {
        "config": {
            "method": "exact recursive gamma + Fejer transform (non-FFT)",
            "m_max": args.m_max,
            "n_grid": args.n_grid,
            "k_values": ks,
            "threshold": args.threshold,
            "catalog_size": len(catalog),
        },
        "aggregate": aggregate,
        "rows": rows,
        "visualizations": [
            str(VIS / "rigorous_fourier_allk_nc_counts.png"),
            str(VIS / "rigorous_fourier_allk_peak_to_mean.png"),
            str(VIS / "rigorous_fourier_allk_var_to_mean2.png"),
            str(VIS / "rigorous_fourier_allk_entropy.png"),
        ],
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
