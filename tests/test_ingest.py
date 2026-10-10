"""Tests for data ingestion: synthetic generator, WA client, caching."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.ingest import (
    EVENT_BOUNDS,
    WorldAthleticsClient,
    fetch_event_dataset,
    generate_synthetic_dataset,
    normalize_wa_results,
)


def test_generate_synthetic_schema():
    df = generate_synthetic_dataset("100m", n_athletes=30, seed=1)
    expected = {
        "athlete_id", "athlete_name", "sex", "birth_year", "event", "date",
        "season", "age", "meet_name", "meet_level", "position",
        "performance", "wind",
    }
    assert expected <= set(df.columns)
    assert len(df) > 1000
    assert df["event"].eq("100m").all()
    assert df["athlete_id"].nunique() == 30


def test_generate_synthetic_bounds():
    for event in ("100m", "800m"):
        df = generate_synthetic_dataset(event, n_athletes=20, seed=1)
        floor, ceiling = EVENT_BOUNDS[event]
        assert df["performance"].between(floor, ceiling).all()
        assert df["meet_level"].between(1, 5).all()
        pd.to_datetime(df["date"])  # all dates parse


def test_generate_synthetic_deterministic():
    a = generate_synthetic_dataset("100m", n_athletes=25, seed=42)
    b = generate_synthetic_dataset("100m", n_athletes=25, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_generate_synthetic_varied_careers():
    df = generate_synthetic_dataset("100m", n_athletes=150, seed=3)
    per_athlete = df.groupby("athlete_id")["season"].nunique()
    assert per_athlete.mean() >= 6
    assert per_athlete.max() >= 10
    # realistic spread of season bests
    bests = df.groupby(["athlete_id", "season"])["performance"].min()
    assert bests.min() < 10.6
    assert bests.max() > 11.5
    assert bests.std() > 0.2


def test_generate_invalid_event():
    try:
        generate_synthetic_dataset("400m", n_athletes=5, seed=1)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_fetch_event_dataset_synthetic_cache(tmp_path):
    df, source = fetch_event_dataset(
        "100m", n_athletes=10, seed=5, cache_dir=tmp_path, source="synthetic",
    )
    assert source == "synthetic"
    cache = tmp_path / "results_100m.csv"
    assert cache.exists()
    df2, source2 = fetch_event_dataset(
        "100m", n_athletes=10, seed=5, cache_dir=tmp_path, source="synthetic",
    )
    assert source2 == "cache"
    pd.testing.assert_frame_equal(df, df2)


def test_normalize_wa_results():
    payload = {
        "results": [
            {"performance": "10.12", "date": "2024-07-15",
             "competition": {"name": "Diamond League"}, "position": 1,
             "wind": 1.2},
            {"performance": "10.25", "date": "2024-08-02",
             "competition": {"name": "National Champs"}, "position": 2},
        ]
    }
    df = normalize_wa_results(payload, athlete_id=999, event="100m")
    assert len(df) == 2
    assert df["athlete_id"].eq(999).all()
    assert list(df["performance"]) == ["10.12", "10.25"]


def test_wa_client_is_available_returns_bool():
    assert isinstance(WorldAthleticsClient.is_available(timeout=4), bool)
