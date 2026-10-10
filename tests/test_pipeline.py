"""End-to-end smoke test: the full pipeline on a small population."""
from __future__ import annotations

import json
from pathlib import Path


def test_run_pipeline_quick(tmp_path):
    from src.pipeline import run_pipeline

    overrides = {
        "data": {"n_synthetic_athletes": 80, "min_seasons": 4},
        "model": {"epochs": 2, "batch_size": 32, "early_stop_patience": 2},
        "clustering": {"k": 4},
    }
    summary = run_pipeline(overrides=overrides, project_root=tmp_path)

    assert summary["n_athletes"] >= 20
    assert set(summary["metrics"]) == {"age_curve", "gbm", "lstm"}
    assert summary["best_model"] in summary["metrics"]

    # artifacts
    bundle = tmp_path / "artifacts" / "model_bundle.pkl"
    assert bundle.exists()

    # reports
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert metrics["best_model"] in ("age_curve", "gbm", "lstm")
    for model, m in metrics["metrics"].items():
        assert m["overall"]["mae"] > 0
        assert 0 <= m["overall"]["picp"] <= 1

    analysis = (tmp_path / "reports" / "analysis.md").read_text()
    assert "Best model" in analysis
    assert "Limitations" in analysis

    # figures
    fig_dir = tmp_path / "reports" / "figures"
    expected_figs = {
        "career_trajectories.png", "forecast_examples.png",
        "metrics_by_horizon.png", "model_comparison.png",
        "calibration.png", "cluster_scatter.png",
        "cluster_trajectories.png", "residuals_by_age.png",
    }
    assert expected_figs <= {p.name for p in fig_dir.iterdir()}

    # archetype table
    archetypes = tmp_path / "reports" / "archetypes.csv"
    assert archetypes.exists()
    assert len(archetypes.read_text().strip().splitlines()) == 5  # header + 4
