#!/usr/bin/env python3
"""Statistical rigor for NC vs non-NC experiments.

Analyzes:
- Fourier metrics separation (Fejer peak/mean, var/mean^2, entropy, gmax)
- Multiscale convergence-rate separation
- Unsupervised clustering agreement with NC labels
- Feature-level effect sizes for NC classification
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stat_rigor import bootstrap_ci, mean_std_ci

try:
    from scipy import stats as sp_stats  # type: ignore
except Exception:
    sp_stats = None


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    if sp_stats is None or len(a) == 0 or len(b) == 0:
        return float("nan")
    try:
        return float(sp_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def compare_groups(nc_vals: list[float], nnc_vals: list[float]) -> dict:
    a = np.asarray(nc_vals, dtype=np.float64)
    b = np.asarray(nnc_vals, dtype=np.float64)
    return {
        "nc": mean_std_ci(a),
        "non_nc": mean_std_ci(b),
        "mann_whitney_p": mann_whitney_p(a, b),
        "cohens_d_nc_minus_nonnc": cohens_d(a, b),
        "significant_95": bool(mann_whitney_p(a, b) < 0.05) if len(a) and len(b) else False,
    }


def fit_alpha(ns: list[int], gs: list[float]) -> float:
    xs, ys = [], []
    for n, g in zip(ns, gs):
        if g > 0:
            xs.append(math.log(float(n)))
            ys.append(math.log(float(g)))
    if len(xs) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(-slope)


def clustering_nc_agreement(rows: list[dict], n_boot: int = 500, seed: int = 0) -> dict:
    labels = np.array([int(r["cluster"]) for r in rows], dtype=int)
    is_nc = np.array([bool(r["is_nc"]) for r in rows], dtype=bool)
    feats = np.array(
        [[r["peak_to_mean"], r["var_to_mean2"], r["entropy"], r["gmax"]] for r in rows],
        dtype=np.float64,
    )

    def assign_nc_cluster(lab: np.ndarray, gmax_idx: int = 3) -> int:
        cids = sorted(set(lab.tolist()))
        means = {c: float(np.mean(feats[lab == c, gmax_idx])) for c in cids}
        return min(means.items(), key=lambda kv: kv[1])[0]

    nc_cluster = assign_nc_cluster(labels)
    pred = labels == nc_cluster
    tp = int(np.sum(pred & is_nc))
    fp = int(np.sum(pred & ~is_nc))
    fn = int(np.sum(~pred & is_nc))
    tn = int(np.sum(~pred & ~is_nc))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / len(rows)

    rng = np.random.default_rng(seed)
    accs, precs, recs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), size=len(rows))
        lab_b = labels[idx]
        nc_b = is_nc[idx]
        feat_b = feats[idx]
        cids = sorted(set(lab_b.tolist()))
        if len(cids) < 2:
            continue
        means = {c: float(np.mean(feat_b[lab_b == c, 3])) for c in cids}
        nc_c = min(means.items(), key=lambda kv: kv[1])[0]
        pred_b = lab_b == nc_c
        tp_b = int(np.sum(pred_b & nc_b))
        fp_b = int(np.sum(pred_b & ~nc_b))
        fn_b = int(np.sum(~pred_b & nc_b))
        tn_b = int(np.sum(~pred_b & ~nc_b))
        accs.append((tp_b + tn_b) / len(idx))
        precs.append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) else 0.0)
        recs.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) else 0.0)

    return {
        "nc_like_cluster": int(nc_cluster),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy_bootstrap": bootstrap_ci(accs, seed=seed),
        "precision_bootstrap": bootstrap_ci(precs, seed=seed + 1),
        "recall_bootstrap": bootstrap_ci(recs, seed=seed + 2),
    }


def _feat_val(r: dict, feat: str) -> float:
    if feat in r:
        return float(r[feat])
    return float(r["metrics"][feat])


def feature_nc_separation(rows: list[dict]) -> dict:
    out = {}
    for feat in ["peak_to_mean", "var_to_mean2", "entropy", "gmax"]:
        nc = [_feat_val(r, feat) for r in rows if r["is_nc"]]
        nnc = [_feat_val(r, feat) for r in rows if not r["is_nc"]]
        out[feat] = compare_groups(nc, nnc)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="NC vs non-NC statistical rigor.")
    p.add_argument("--fourier-json", default="rigorous_fourier_all_k_results.json")
    p.add_argument("--cluster-json", default="fourier_clusters_advanced.json")
    p.add_argument("--multiscale-csv", default="exact_catalog_results_multiscale.csv")
    p.add_argument("--out-json", default="nc_non_nc_stat_rigor_results.json")
    p.add_argument("--out-csv", default="nc_non_nc_stat_rigor_results.csv")
    args = p.parse_args()

    fdata = json.loads(Path(args.fourier_json).read_text())
    rows = fdata["rows"]
    results: dict = {"config": {"scope": "NC vs non-NC characterization"}}

    # --- Fourier metrics: pooled and by-k ---
    pooled = feature_nc_separation(rows)
    results["fourier_metrics_pooled"] = pooled

    by_k = {}
    for k in sorted({int(r["k"]) for r in rows}):
        sub = [r for r in rows if int(r["k"]) == k]
        nc = [r for r in sub if r["is_nc"]]
        nnc = [r for r in sub if not r["is_nc"]]
        if not nc or not nnc:
            by_k[str(k)] = {"note": "only one class present", "nc_count": len(nc), "non_nc_count": len(nnc)}
            continue
        by_k[str(k)] = {
            "nc_count": len(nc),
            "non_nc_count": len(nnc),
            "metrics": feature_nc_separation(sub),
        }
    results["fourier_metrics_by_k"] = by_k

    # --- Clustering agreement with NC labels ---
    if Path(args.cluster_json).exists():
        cdata = json.loads(Path(args.cluster_json).read_text())
        crows = cdata["rows"]
        results["clustering_nc_agreement"] = clustering_nc_agreement(crows)
        results["clustering_silhouette"] = {
            "observed": float(cdata.get("best_silhouette", 0.0)),
            "best_k": int(cdata.get("best_k", 2)),
        }

    # --- Convergence alpha by NC class ---
    by_name: dict[str, list[dict]] = {}
    with Path(args.multiscale_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_name.setdefault(row["name"], []).append(row)

    alphas_nc, alphas_nnc = [], []
    per_point = []
    for name, pts in by_name.items():
        for k in sorted({int(p["k"]) for p in pts}):
            sub = [p for p in pts if int(p["k"]) == k]
            ns = [int(p["N"]) for p in sub]
            gs = [float(p["max_abs_nonzero"]) for p in sub]
            alpha = fit_alpha(ns, gs)
            is_nc = sub[0]["empirical_k_nc"].lower().startswith("yes")
            per_point.append({"name": name, "k": k, "alpha": alpha, "is_nc": is_nc})
            if not math.isnan(alpha):
                (alphas_nc if is_nc else alphas_nnc).append(alpha)

    results["convergence_alpha"] = {
        "by_class": {
            "NC": compare_groups(alphas_nc, alphas_nnc)["nc"] if alphas_nc else {},
            "comparison": compare_groups(alphas_nc, alphas_nnc) if alphas_nc and alphas_nnc else {},
        },
        "full_comparison": compare_groups(alphas_nc, alphas_nnc) if alphas_nc and alphas_nnc else {},
    }

    # --- Simple threshold classifier on gmax ---
    gmax_nc = [float(r["gmax"]) for r in rows if r["is_nc"]]
    gmax_nnc = [float(r["gmax"]) for r in rows if not r["is_nc"]]
    thr = 0.005
    pred = [float(r["gmax"]) <= thr for r in rows]
    truth = [bool(r["is_nc"]) for r in rows]
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    fn = sum(1 for p, t in zip(pred, truth) if not p and t)
    tn = sum(1 for p, t in zip(pred, truth) if not p and not t)
    results["gmax_threshold_classifier"] = {
        "threshold": thr,
        "accuracy": (tp + tn) / len(rows),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "mann_whitney_gmax": compare_groups(gmax_nc, gmax_nnc),
    }

    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_rows = []
    for feat, comp in pooled.items():
        csv_rows.append(
            {
                "analysis": "fourier_pooled",
                "feature": feat,
                "nc_mean": comp["nc"]["mean"],
                "nc_ci_low": comp["nc"]["ci_low"],
                "nc_ci_high": comp["nc"]["ci_high"],
                "non_nc_mean": comp["non_nc"]["mean"],
                "non_nc_ci_low": comp["non_nc"]["ci_low"],
                "non_nc_ci_high": comp["non_nc"]["ci_high"],
                "p_value": comp["mann_whitney_p"],
                "cohens_d": comp["cohens_d_nc_minus_nonnc"],
                "significant_95": comp["significant_95"],
            }
        )
    if csv_rows:
        with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    print(f"Wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
