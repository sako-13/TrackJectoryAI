"""Tests for the demo app API."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.demo import app as demo_app


@pytest.fixture(scope="module")
def bundle_file(tmp_path_factory):
    """Train a tiny bundle and point the demo at it."""
    from src.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("demo_bundle")
    run_pipeline(
        overrides={
            "data": {"n_synthetic_athletes": 80, "min_seasons": 4},
            "model": {"epochs": 2, "batch_size": 32, "early_stop_patience": 2},
            "clustering": {"k": 4},
        },
        project_root=root,
    )
    demo_app.BUNDLE_PATH = root / "artifacts" / "model_bundle.pkl"
    demo_app.BUNDLE = None
    return demo_app.BUNDLE_PATH


@pytest.fixture()
def client(bundle_file):
    demo_app.BUNDLE = None
    with TestClient(demo_app.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_meta(client):
    resp = client.get("/meta")
    assert resp.status_code == 200
    data = resp.json()
    assert data["event"] == "100m"
    assert data["horizons"] == 5
    assert {"age_curve", "gbm", "lstm"} <= set(data["models"])


GOOD_SEASONS = [
    {"age": 16.5, "season_best": 12.10, "n_races": 6, "level": 4},
    {"age": 17.4, "season_best": 11.75, "n_races": 8, "level": 3},
    {"age": 18.3, "season_best": 11.50, "n_races": 9, "level": 3},
    {"age": 19.2, "season_best": 11.30, "n_races": 9, "level": 2},
    {"age": 20.4, "season_best": 11.18, "n_races": 10, "level": 2},
]


def test_predict_all_models(client):
    resp = client.post("/predict", json={"event": "100m", "seasons": GOOD_SEASONS})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["forecasts"]) == {"age_curve", "gbm", "lstm"}
    for model, rows in data["forecasts"].items():
        assert len(rows) == 5
        for row in rows:
            assert row["q10"] <= row["q50"] <= row["q90"]
            assert 9.4 <= row["q10"] <= row["q90"] <= 16.0
        ages = [r["age"] for r in rows]
        assert ages == sorted(ages)
    assert data["archetype"] in (
        "Early Peaker", "Late Bloomer", "Rapid Riser", "Steady Developer",
    )


def test_predict_single_model(client):
    resp = client.post("/predict", json={
        "event": "100m", "seasons": GOOD_SEASONS, "models": ["gbm"],
    })
    assert resp.status_code == 200
    assert set(resp.json()["forecasts"]) == {"gbm"}


def test_predict_requires_three_seasons(client):
    resp = client.post("/predict", json={
        "event": "100m", "seasons": GOOD_SEASONS[:2],
    })
    assert resp.status_code == 422


def test_predict_wrong_event(client):
    resp = client.post("/predict", json={
        "event": "800m", "seasons": GOOD_SEASONS,
    })
    assert resp.status_code == 400


def test_predict_unknown_model(client):
    resp = client.post("/predict", json={
        "event": "100m", "seasons": GOOD_SEASONS, "models": ["nope"],
    })
    assert resp.status_code == 400


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Athlete Career Forecaster" in resp.text
    assert "app.js" in resp.text
