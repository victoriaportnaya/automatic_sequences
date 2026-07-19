#!/usr/bin/env python3
"""Generate report figures for the focused Monte Carlo paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from nc_stream import apply_digital_shift


DATA_DIR = Path("data")
OUTPUT_DIRS = [Path("artifacts/figures")]
for output_dir in OUTPUT_DIRS:
    output_dir.mkdir(parents=True, exist_ok=True)

COLORS = {
    "iid": "#9aa0a6",
    "sobol_unshifted": "#202124",
    "random_shift": "#1a73e8",
    "RS_{11}": "#d93025",
    "transition_{01}": "#188038",
    "A3_{101,111}": "#9334e6",
    "A4_length4": "#f29900",
    "TM_{1}_control": "#795548",
}


def save(name: str) -> None:
    for output_dir in OUTPUT_DIRS:
        plt.savefig(output_dir / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()


def convergence_figure() -> None:
    d = json.load(open(DATA_DIR / "nc_one_over_n_convergence_results.json"))
    names = ["RS", "{01}", "A_3", "TM"]
    labels = ["Rudin-Shapiro (NC)", "transition {01} (NC)", "A3 (NC)", "Thue-Morse (control)"]
    colors = ["#d93025", "#188038", "#9334e6", "#5f6368"]
    styles = ["-", "--", "-", "-"]
    plt.figure(figsize=(7.2, 4.3))
    for name, label, color, style in zip(names, labels, colors, styles):
        e = d["k2"][name]["empirical"]
        plt.loglog(e["ns"], e["g_by_n"], marker="o", linestyle=style, ms=4, lw=2.1, label=label, color=color)
    ns = np.asarray(d["k2"]["RS"]["empirical"]["ns"], dtype=float)
    plt.loglog(ns, 20 / ns, "--", lw=1.2, color="#1a73e8", label=r"reference $20/N$")
    plt.xlabel("prefix length N")
    plt.ylabel(r"$G_N=\max_{1\leq h\leq64}|\gamma_N(h)|$")
    plt.grid(True, which="both", alpha=0.18)
    plt.legend(frameon=False, fontsize=8.5, ncol=2)
    plt.tight_layout()
    save("correlation_decay.png")


def shift_geometry_figure() -> None:
    base = qmc.Sobol(d=2, scramble=False).random_base2(m=6)
    mask = np.asarray([39, 17], dtype=np.uint32)
    shifted = apply_digital_shift(base, mask, bits=6)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
    for ax, pts, title in zip(
        axes,
        [base, shifted],
        ["Sobol net", "same net after a 6-bit XOR shift"],
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color="#1a73e8", alpha=0.9)
        for t in np.linspace(0, 1, 9):
            ax.axhline(t, color="#dadce0", lw=0.45, zorder=0)
            ax.axvline(t, color="#dadce0", lw=0.45, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$u_1$")
    axes[0].set_ylabel(r"$u_2$")
    fig.suptitle("A digital shift relocates points but preserves dyadic occupancy", fontsize=11)
    fig.tight_layout()
    save("digital_shift_geometry.png")


def rmse_panels() -> None:
    d = json.load(open(DATA_DIR / "focused_mc_validation.json"))
    rows = d["rows"]
    problems = ["european_call", "g_function", "rare_event"]
    titles = ["European call (d=1)", "Sobol g-function (d=6)", "rare event (d=10)"]
    methods = ["iid", "random_shift", "RS_{11}", "A4_length4"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45))
    for ax, problem, title in zip(axes, problems, titles):
        for method in methods:
            sub = sorted(
                [r for r in rows if r["problem"] == problem and r["method"] == method],
                key=lambda r: r["n"],
            )
            ax.loglog(
                [r["n"] for r in sub],
                [r["rmse"] for r in sub],
                "o-",
                ms=3.5,
                lw=1.5,
                color=COLORS[method],
                label=method.replace("_", " "),
            )
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, which="both", alpha=0.16)
        ax.set_xlabel("N")
    axes[0].set_ylabel("RMSE across 64 shifts/runs")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save("paired_rmse_panels.png")


def ratio_heatmap() -> None:
    d = json.load(open(DATA_DIR / "focused_mc_validation.json"))
    summaries = d["summaries"]
    methods = ["RS_{11}", "transition_{01}", "A3_{101,111}", "A4_length4"]
    problems = d["config"]["problems"]
    z = np.asarray([
        [next(s["final_ratio_to_random"] for s in summaries if s["method"] == m and s["problem"] == p) for p in problems]
        for m in methods
    ])
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    im = ax.imshow(z, cmap="RdYlGn_r", vmin=0.75, vmax=1.25, aspect="auto")
    ax.set_xticks(range(len(problems)), [p.replace("_", "\n") for p in problems], fontsize=7.8)
    ax.set_yticks(range(len(methods)), [m.replace("_", " ") for m in methods], fontsize=8.5)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("RMSE ratio: NC shift / random shift", fontsize=8.5)
    ax.set_title("Paired final-N comparison: values near 1 indicate no systematic NC advantage", fontsize=10.5)
    fig.tight_layout()
    save("nc_random_ratio_heatmap.png")


def census_counts() -> None:
    files = [
        ("recursive_exhaustive_nc_results.json", "Census A (846 sets)"),
        ("recursive_exhaustive_nc_len3_results.json", "Census B (2047 sets)"),
    ]
    ks = np.arange(2, 9)
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    width = 0.36
    for i, (filename, label) in enumerate(files):
        data = json.load(open(DATA_DIR / filename))
        summary = next(iter(data["summaries"].values()))
        counts = [summary["nc_counts"][str(k)] for k in ks]
        ax.bar(ks + (i - 0.5) * width, counts, width=width, label=label,
               color=["#1a73e8", "#d93025"][i], alpha=0.88)
        for x, y in zip(ks + (i - 0.5) * width, counts):
            if y:
                ax.text(x, y, str(y), ha="center", va="bottom", fontsize=8)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(ks)
    ax.set_xlabel("root-of-unity modulus k")
    ax.set_ylabel("sets passing the NC test (symlog scale)")
    ax.set_title("Exact-recursion census: accepted cases are concentrated at even moduli")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save("nc_census_counts.png")


def exhaustive_driver_errors() -> None:
    data = json.load(open(DATA_DIR / "exhaustive_nc_mc_len3_results.json"))
    rows = data["rows"]
    direct = [r["abs_error"] for r in rows if r.get("mode") == "nc_direct"]
    hybrid = [r["abs_error"] for r in rows if r.get("mode") == "sobol_nc_shift"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    parts = ax.violinplot([direct, hybrid], positions=[1, 2], widths=0.72,
                          showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], ["#d93025", "#1a73e8"]):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.45)
    parts["cmedians"].set_color("#202124")
    summary = data["summary"]["by_mode"]
    ax.scatter([1, 2], [summary["nc_direct"]["median_error"], summary["sobol_nc_shift"]["median_error"]],
               color=["#d93025", "#1a73e8"], zorder=5, s=34)
    ax.axhline(summary["sobol"]["median_error"], color="#188038", ls="--", lw=1.6,
               label=f"plain Sobol error = {summary['sobol']['median_error']:.2e}")
    ax.set_yscale("log")
    ax.set_xticks([1, 2], ["raw NC bit packing\n(720 drivers + 2 repeats)",
                           "Sobol + NC shift\n(720 masks)"])
    ax.set_ylabel("absolute European-call pricing error at N=8192")
    ax.set_title("Same operational NC catalog, different embedding")
    ax.grid(axis="y", which="both", alpha=0.16)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save("exhaustive_nc_driver_errors.png")


def closure_rates() -> None:
    data = json.load(open(DATA_DIR / "fourier_closure_operations_results.json"))["by_operation"]
    keys = ["identity", "inverse", "even_subseq", "odd_subseq", "product", "interleave",
            "add_pattern", "remove_pattern", "self_product", "interleave_self"]
    labels = ["identity", "inverse", "even", "odd", "product", "interleave",
              "add pattern", "remove pattern", "self-product", "self-interleave"]
    rates = [100 * data[k]["closure_rate"] for k in keys]
    colors = ["#188038" if r == 100 else "#d93025" if r == 0 else "#f29900" for r in rates]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    bars = ax.bar(range(len(rates)), rates, color=colors, alpha=0.88)
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("NC closure rate (%)")
    ax.set_title("Operations on operationally NC inputs: safe, conditional, and destructive")
    ax.grid(axis="y", alpha=0.18)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width()/2, r + 2, f"{r:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    save("nc_closure_rates.png")


if __name__ == "__main__":
    convergence_figure()
    shift_geometry_figure()
    rmse_panels()
    ratio_heatmap()
    census_counts()
    exhaustive_driver_errors()
    closure_rates()
    print("wrote figures to " + ", ".join(str(path) for path in OUTPUT_DIRS))
