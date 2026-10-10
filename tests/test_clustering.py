"""Tests for career archetype clustering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.clustering.archetypes import (
    TRAJECTORY_FEATURES,
    cluster_athletes,
    describe_clusters,
    label_archetypes,
    predict_archetype,
    trajectory_features,
)


def test_trajectory_features(summary):
    traj = trajectory_features(summary)
    assert set(TRAJECTORY_FEATURES) <= set(traj.columns)
    assert len(traj) == summary["athlete_id"].nunique()
    # peak best is the athlete's fastest season
    for aid, g in summary.groupby("athlete_id"):
        assert abs(traj.loc[aid, "peak_best"] - g["season_best"].min()) < 1e-9
    assert (traj["total_gain"] >= -0.01).all()
    assert (traj["peak_age"] >= traj["first_age"]).all()


def test_cluster_athletes(summary):
    traj = trajectory_features(summary)
    result = cluster_athletes(traj, k=4, seed=42)
    labels = result["labels"]
    assert set(labels.tolist()) == {0, 1, 2, 3}
    assert len(labels) == len(traj)
    assert result["coords"].shape == (len(traj), 2)
    assert set(result["archetypes"].values()) == {
        "Early Peaker", "Late Bloomer", "Rapid Riser", "Steady Developer",
    }


def test_label_archetypes_interpretable(summary):
    traj = trajectory_features(summary)
    result = cluster_athletes(traj, k=4, seed=42)
    centers = pd.DataFrame(result["centers"], columns=TRAJECTORY_FEATURES)
    names = result["archetypes"]
    early_id = min(names, key=lambda c: centers.loc[c, "peak_age"])
    late_id = max(names, key=lambda c: centers.loc[c, "peak_age"])
    assert names[early_id] == "Early Peaker"
    assert names[late_id] == "Late Bloomer"


def test_describe_clusters(summary):
    traj = trajectory_features(summary)
    result = cluster_athletes(traj, k=4, seed=42)
    desc = describe_clusters(result)
    assert len(desc) == 4
    assert desc["n_athletes"].sum() == len(traj)
    assert desc["share"].sum() > 0.99
    assert desc["archetype"].nunique() == 4


def test_predict_archetype(summary):
    traj = trajectory_features(summary)
    result = cluster_athletes(traj, k=4, seed=42)
    row = traj.iloc[[0]]
    name, cid = predict_archetype(result, row)
    assert isinstance(name, str)
    assert isinstance(cid, int)
    assert name == result["archetypes"][cid]


def test_clustering_separates_known_groups():
    """Constructed data: early vs late peakers must land in different clusters."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(60):
        early = i < 30
        first_age = 15.5 if early else 18.0
        peak_age = 20.5 if early else 27.0
        rows.append({
            "athlete_id": i, "first_age": first_age, "first_best": 12.5,
            "peak_age": peak_age + rng.normal(0, 0.2),
            "peak_best": 11.0, "last_age": 28.0, "career_len": 10,
            "total_gain": 1.5, "early_gain": 1.2 if early else 0.3,
            "late_gain": 0.3 if early else 1.2,
            "improvement_rate": 1.5 / (peak_age - first_age),
        })
    traj = pd.DataFrame(rows).set_index("athlete_id")
    result = cluster_athletes(traj, k=2, seed=42)
    labels = pd.Series(result["labels"], index=traj.index)
    # the two constructed groups should be (almost) perfectly separated
    purity = max(
        (labels.iloc[:30] == labels.iloc[:30].mode()[0]).mean()
        + (labels.iloc[30:] != labels.iloc[:30].mode()[0]).mean(),
        (labels.iloc[:30] != labels.iloc[:30].mode()[0]).mean()
        + (labels.iloc[30:] == labels.iloc[:30].mode()[0]).mean(),
    )
    assert purity > 0.9
