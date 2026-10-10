"""Publication-quality figures for the analysis report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import format_time

PALETTE = {"age_curve": "#1f77b4", "gbm": "#ff7f0e", "lstm": "#2ca02c", "transformer": "#9467bd"}
BAND_COLOR = "#2ca02c"


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def plot_career_trajectories(summary: pd.DataFrame, path, event: str, n_show: int = 90, seed: int = 42) -> Path:
    """Season-best progressions of a sample of athletes."""
    _style()
    rng = np.random.default_rng(seed)
    ids = summary["athlete_id"].unique()
    sample = rng.choice(ids, size=min(n_show, len(ids)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for aid in sample:
        g = summary[summary["athlete_id"] == aid].sort_values("age")
        ax.plot(g["age"], g["season_best"], color="0.75", lw=0.7, alpha=0.8, zorder=1)
    for aid in sample[:6]:
        g = summary[summary["athlete_id"] == aid].sort_values("age")
        ax.plot(g["age"], g["season_best"], lw=1.6, zorder=2)
    ax.invert_yaxis()
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(f"Season best ({event})")
    ax.set_title("Career trajectories (grey: all sampled athletes)")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_forecast_examples(records_by_model: dict, summary: pd.DataFrame, path, event: str, n_examples: int = 3, seed: int = 42) -> Path:
    """Observed vs predicted progressions with 80% bands for the sequence model."""
    _style()
    rng = np.random.default_rng(seed)
    test_ids = sorted({r["athlete_id"] for recs in records_by_model.values() for r in recs})
    chosen = list(rng.choice(test_ids, size=min(n_examples, len(test_ids)), replace=False))
    model_names = list(records_by_model.keys())
    fig, axes = plt.subplots(n_examples, 1, figsize=(8, 3.2 * n_examples), sharex=False)
    if n_examples == 1:
        axes = [axes]
    seq_name = next((n for n in model_names if n in ("lstm", "transformer")), model_names[0])
    for ax, aid in zip(axes, chosen):
        g = summary[summary["athlete_id"] == aid].sort_values("age")
        # find the evaluation cutoffs for this athlete in the seq records
        recs = [r for r in records_by_model[seq_name] if r["athlete_id"] == aid]
        if not recs:
            continue
        cutoffs = sorted({r["cutoff"] for r in recs} if "cutoff" in recs[0] else {r.get("cutoff_age", 0)})
        cutoff = cutoffs[len(cutoffs) // 2]
        hist = g[g["age"] <= cutoff]
        future = g[g["age"] > cutoff]
        ax.plot(hist["age"], hist["season_best"], "o-", color="0.3", lw=1.5, label="observed")
        if len(future):
            ax.plot(future["age"], future["season_best"], "s--", color="0.1", lw=1.2, alpha=0.8, label="actual future")
        for name in model_names:
            recs_m = [r for r in records_by_model[name] if r["athlete_id"] == aid and r.get("cutoff") == cutoff]
            if not recs_m:
                continue
            horizons = [r["horizon"] for r in recs_m]
            ages = [r.get("age_at_forecast", cutoff + h) for r, h in zip(recs_m, horizons)]
            med = [r["q50"] for r in recs_m]
            if name == seq_name:
                lo = [r["q10"] for r in recs_m]
                hi = [r["q90"] for r in recs_m]
                ax.fill_between(ages, lo, hi, color=BAND_COLOR, alpha=0.18, label=f"{name} 80% band")
            ax.plot(ages, med, "o--", color=PALETTE.get(name, "0.5"), lw=1.4, ms=4, label=f"{name} median")
        ax.invert_yaxis()
        ax.set_ylabel(f"{event}")
        ax.set_title(f"Athlete {aid} — forecast from age {cutoff:.0f}")
        ax.legend(fontsize=7, ncol=3, loc="lower left")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metrics_by_horizon(metrics: dict, path) -> Path:
    """MAE / RMSE / CRPS per horizon, one panel per model."""
    _style()
    models = list(metrics.keys())
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    metric_names = ["mae", "rmse", "crps"]
    titles = ["MAE (s)", "RMSE (s)", "CRPS (s)"]
    for ax, m, title in zip(axes, metric_names, titles):
        for name in models:
            ph = metrics[name]["per_horizon"]
            ax.plot([h["horizon"] for h in ph], [h[m] for h in ph], "o-", lw=1.6, ms=4,
                    color=PALETTE.get(name, "0.5"), label=name)
        ax.set_xlabel("Forecast horizon (seasons)")
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Forecast accuracy by horizon")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_model_comparison(metrics: dict, path) -> Path:
    """Overall MAE and CRPS per model."""
    _style()
    models = list(metrics.keys())
    mae_vals = [metrics[m]["overall"]["mae"] for m in models]
    crps_vals = [metrics[m]["overall"]["crps"] for m in models]
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    b1 = ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#4c72b0")
    b2 = ax.bar(x + width / 2, crps_vals, width, label="CRPS", color="#dd8452")
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, models)
    ax.set_ylabel("seconds")
    ax.set_title("Overall forecast accuracy (all horizons)")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration(reports: dict, path) -> Path:
    """Empirical coverage of the 80% interval by horizon, plus mean width."""
    _style()
    models = list(reports.keys())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for name in models:
        ph = reports[name]["per_horizon"]
        axes[0].plot([h["horizon"] for h in ph], [h["picp"] for h in ph], "o-", lw=1.6, ms=4,
                     color=PALETTE.get(name, "0.5"), label=name)
        axes[1].plot([h["horizon"] for h in ph], [h["mpiw"] for h in ph], "o-", lw=1.6, ms=4,
                     color=PALETTE.get(name, "0.5"), label=name)
    axes[0].axhline(0.80, color="k", ls="--", lw=1, label="nominal 80%")
    axes[0].set_ylim(min(0.5, min(h["picp"] for r in reports.values() for h in r["per_horizon"]) - 0.1), 1.0)
    axes[0].set_xlabel("Forecast horizon (seasons)")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("80% interval coverage by horizon")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Forecast horizon (seasons)")
    axes[1].set_ylabel("Mean interval width (s)")
    axes[1].set_title("Interval sharpness by horizon")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_residuals_by_age(records: list[dict], path) -> Path:
    """Median-forecast residuals vs athlete age (bias check)."""
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ages = [r.get("age_at_forecast", 0) for r in records]
    resid = [r["y"] - r["q50"] for r in records]
    ax.scatter(ages, resid, s=6, alpha=0.35, color="#4c72b0")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Age at forecast (years)")
    ax.set_ylabel("Residual: actual - predicted median (s)")
    ax.set_title("Residuals vs age (best model)")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cluster_scatter(cluster_result: dict, path) -> Path:
    """PCA scatter of athletes colored by archetype."""
    _style()
    labels = cluster_result["labels"]
    names = cluster_result["archetypes"]
    coords = cluster_result["coords"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for cid in sorted(set(labels.tolist())):
        idx = labels == cid
        ax.scatter(coords[idx, 0], coords[idx, 1], s=14, alpha=0.75,
                   label=f"{names[cid]} ({idx.sum()})")
    centers_pca = cluster_result["pca"].transform(cluster_result["centers"])
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="X", s=120,
               c="k", edgecolors="w", linewidths=1.2, zorder=5, label="centroids")
    ax.set_xlabel(f"PC1 ({cluster_result['pca'].explained_variance_ratio_[0] * 100:.0f}%)")
    ax.set_ylabel(f"PC2 ({cluster_result['pca'].explained_variance_ratio_[1] * 100:.0f}%)")
    ax.set_title("Career-development archetypes (PCA of trajectory features)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cluster_trajectories(cluster_result: dict, summary: pd.DataFrame, path, event: str) -> Path:
    """Mean season-best trajectory per archetype, aligned by age."""
    _style()
    labels = cluster_result["labels"]
    names = cluster_result["archetypes"]
    features = cluster_result["features"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for cid in sorted(set(labels.tolist())):
        ids = features.iloc[labels == cid].index
        g = summary[summary["athlete_id"].isin(ids)]
        grouped = g.groupby(g["age"].round(0))["season_best"].agg(["mean", "std", "count"])
        grouped = grouped[grouped["count"] >= 3]
        ax.plot(grouped.index, grouped["mean"], lw=2, label=f"{names[cid]}")
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.12,
        )
    ax.invert_yaxis()
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(f"Mean season best ({event})")
    ax.set_title("Mean career trajectory by archetype")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def format_time_col(series: pd.Series, event: str) -> pd.Series:
    return series.apply(lambda v: format_time(v, event))
