"""Tests for cleaning and season aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.clean import build_season_summary, clean_results


def _base_rows():
    rows = []
    for i, (age, perf, date) in enumerate(
        [(16, 12.3, "2020-05-10"), (17, 11.9, "2021-06-11"),
         (18, 11.5, "2022-07-12"), (19, 11.3, "2023-07-13")]
    ):
        rows.append({
            "athlete_id": 1, "athlete_name": "Test Runner", "sex": "M",
            "birth_year": 2004, "event": "100m", "date": date, "season": int(date[:4]),
            "age": float(age), "meet_name": "Meet", "meet_level": 3,
            "position": 1, "performance": perf, "wind": 0.5,
        })
    return rows


def test_clean_parses_string_times():
    rows = _base_rows()
    rows[0]["performance"] = "12.30"
    rows.append(dict(rows[0], date="2020-05-12", performance="1:55.00"))  # 800m-style
    df = clean_results(pd.DataFrame(rows), "100m")
    assert df["performance"].dtype == float
    assert (df["performance"] >= 9.4).all()


def test_clean_removes_out_of_range():
    rows = _base_rows()
    rows.append(dict(rows[0], date="2020-05-14", performance=8.0))    # too fast
    rows.append(dict(rows[0], date="2020-05-15", performance=20.0))   # too slow
    df = clean_results(pd.DataFrame(rows), "100m")
    assert df["performance"].between(9.4, 16.0).all()


def test_clean_flags_wind_aided():
    rows = _base_rows()
    rows.append(dict(rows[0], date="2020-05-16", performance=12.0, wind=2.5))
    df = clean_results(pd.DataFrame(rows), "100m")
    assert df.loc[df["wind"] > 2.0, "wind_aided"].all()
    assert not df.loc[df["wind"] <= 2.0, "wind_aided"].any()


def test_season_best_excludes_wind_aided():
    rows = _base_rows()
    # same season: one legal, one wind-aided faster mark
    rows.append(dict(rows[0], date="2020-05-20", performance=12.05, wind=2.4))
    df = clean_results(pd.DataFrame(rows), "100m")
    summary = build_season_summary(df, min_seasons=2)
    sb = summary[(summary["athlete_id"] == 1) & (summary["season"] == 2020)]
    assert abs(sb["season_best"].iloc[0] - 12.3) < 1e-9
    assert sb["n_races"].iloc[0] == 2


def test_season_summary_structure():
    df = clean_results(pd.DataFrame(_base_rows()), "100m")
    summary = build_season_summary(df, min_seasons=4)
    assert set(summary.columns) >= {
        "athlete_id", "season", "season_best", "n_races", "mean_time",
        "best_level", "mean_level", "age", "seasons_so_far", "delta_best",
        "rolling_best_3",
    }
    assert len(summary) == 4
    assert (summary["delta_best"].iloc[1:] > 0).all()  # improving each year
    assert summary["seasons_so_far"].tolist() == [1, 2, 3, 4]


def test_season_summary_filters_short_careers():
    rows = _base_rows()
    rows = rows[:2]  # only 2 seasons
    df = clean_results(pd.DataFrame(rows), "100m")
    summary = build_season_summary(df, min_seasons=4)
    assert len(summary) == 0


def test_season_summary_aggregates_multiple_athletes():
    rows = _base_rows()
    for extra_id in (2, 3):
        for r in _base_rows():
            r = dict(r, athlete_id=extra_id)
            rows.append(r)
    df = clean_results(pd.DataFrame(rows), "100m")
    summary = build_season_summary(df, min_seasons=4)
    assert summary["athlete_id"].nunique() == 3
    assert summary.groupby("athlete_id").size().eq(4).all()
