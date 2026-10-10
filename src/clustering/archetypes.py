"""Unsupervised discovery of career-development archetypes.

Athletes are described by trajectory shape features (peak age,
improvement rate, career length, ...), standardized, and clustered
with K-Means. Clusters are then labelled by interpreting their
centers: Early Peaker, Late Bloomer, Rapid Riser, Steady Developer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TRAJECTORY_FEATURES = [
    "first_age",
    "first_best",
    "peak_age",
    "peak_best",
    "career_len",
    "total_gain",
    "early_gain",
    "late_gain",
    "improvement_rate",
]


def _athlete_trajectory_row(g: pd.DataFrame) -> dict:
    """Trajectory-shape features for one athlete's season summary."""
    g = g.sort_values("season")
    first = g.iloc[0]
    last = g.iloc[-1]
    peak_idx = g["season_best"].idxmin()
    peak = g.loc[peak_idx]
    first_age = float(first["age"])
    first_best = float(first["season_best"])
    peak_age = float(peak["age"])
    peak_best = float(peak["season_best"])
    total_gain = first_best - peak_best
    early = g[g["age"] <= 20]
    if len(early):
        early_best = float(early["season_best"].min())
        early_gain = first_best - early_best
        late_gain = early_best - peak_best
    else:
        early_gain = 0.0
        late_gain = total_gain
    return {
        "athlete_id": g["athlete_id"].iloc[0],
        "first_age": first_age,
        "first_best": first_best,
        "peak_age": peak_age,
        "peak_best": peak_best,
        "last_age": float(last["age"]),
        "career_len": int(len(g)),
        "total_gain": total_gain,
        "early_gain": early_gain,
        "late_gain": late_gain,
        "improvement_rate": total_gain / max(peak_age - first_age, 0.1),
    }


def trajectory_features(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per athlete with career-shape features."""
    rows = [_athlete_trajectory_row(g) for _, g in summary.groupby("athlete_id")]
    return pd.DataFrame(rows).set_index("athlete_id")


def cluster_athletes(features: pd.DataFrame, k: int = 4, seed: int = 42) -> dict:
    """Standardize -> PCA -> K-Means over trajectory features."""
    X = features[TRAJECTORY_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=min(2, X_scaled.shape[1]), random_state=seed)
    coords = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_scaled)
    labels = kmeans.labels_.astype(int)
    result = {
        "kmeans": kmeans,
        "labels": labels,
        "centers": kmeans.cluster_centers_,
        "scaler": scaler,
        "pca": pca,
        "coords": coords,
        "features": features,
        "feature_names": TRAJECTORY_FEATURES,
        "inertia": float(kmeans.inertia_),
    }
    result["archetypes"] = label_archetypes(result)
    return result


def label_archetypes(cluster_result: dict) -> dict[int, str]:
    """Assign human-readable archetype names to cluster ids."""
    centers = pd.DataFrame(cluster_result["centers"], columns=TRAJECTORY_FEATURES)
    by_peak_age = centers["peak_age"].sort_values().index.tolist()
    names: dict[int, str] = {}
    names[by_peak_age[0]] = "Early Peaker"
    names[by_peak_age[-1]] = "Late Bloomer"
    remaining = by_peak_age[1:-1]
    if remaining:
        by_rate = centers.loc[remaining, "improvement_rate"].sort_values(ascending=False).index.tolist()
        names[by_rate[0]] = "Rapid Riser"
        for cid in by_rate[1:]:
            names[cid] = "Steady Developer"
    # any leftover clusters (k > 4)
    for cid in range(len(centers)):
        names.setdefault(cid, f"Archetype {cid + 1}")
    return names


def describe_clusters(cluster_result: dict) -> pd.DataFrame:
    """Per-archetype summary statistics."""
    features = cluster_result["features"]
    labels = cluster_result["labels"]
    names = cluster_result["archetypes"]
    frames = []
    for cid in sorted(set(labels.tolist())):
        idx = labels == cid
        sub = features.iloc[idx]
        frames.append(
            {
                "archetype": names[cid],
                "cluster": int(cid),
                "n_athletes": int(idx.sum()),
                "share": float(idx.mean()),
                "mean_first_age": float(sub["first_age"].mean()),
                "mean_peak_age": float(sub["peak_age"].mean()),
                "mean_peak_best": float(sub["peak_best"].mean()),
                "mean_total_gain": float(sub["total_gain"].mean()),
                "mean_improvement_rate": float(sub["improvement_rate"].mean()),
                "mean_career_len": float(sub["career_len"].mean()),
            }
        )
    return pd.DataFrame(frames)


def predict_archetype(cluster_result: dict, feature_row: pd.DataFrame) -> tuple[str, int]:
    """Assign an archetype to a new athlete described by one feature row."""
    X = feature_row[TRAJECTORY_FEATURES].to_numpy(dtype=float)
    X_scaled = cluster_result["scaler"].transform(X)
    cid = int(cluster_result["kmeans"].predict(X_scaled)[0])
    return cluster_result["archetypes"][cid], cid
