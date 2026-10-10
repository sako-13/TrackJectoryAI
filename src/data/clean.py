"""Data cleaning and season-level aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.ingest import EVENT_BOUNDS
from src.utils import parse_time_to_seconds


def clean_results(df: pd.DataFrame, event: str = "100m") -> pd.DataFrame:
    """Normalize raw race records into a tidy, validated long table."""
    required = {"athlete_id", "date", "performance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    df["performance"] = df["performance"].apply(lambda v: parse_time_to_seconds(v, event))
    df = df[df["performance"].notna()]

    floor, ceiling = EVENT_BOUNDS[event]
    df = df[(df["performance"] >= floor) & (df["performance"] <= ceiling)]

    if "wind" in df.columns:
        df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    else:
        df["wind"] = np.nan
    # Wind-aided marks (> +2.0 m/s) are excluded from season bests when a legal mark exists.
    df["wind_aided"] = (event == "100m") & (df["wind"].fillna(0.0) > 2.0)

    df["season"] = df["date"].dt.year
    if "age" not in df.columns or df["age"].isna().all():
        if "birth_year" in df.columns:
            df["age"] = (df["season"] - df["birth_year"]) + 0.5

    dup_cols = ["athlete_id", "date", "performance"]
    if "meet_name" in df.columns:
        dup_cols.append("meet_name")
    df = df.drop_duplicates(subset=dup_cols)

    return df.sort_values(["athlete_id", "date"]).reset_index(drop=True)


def build_season_summary(df: pd.DataFrame, min_seasons: int = 4) -> pd.DataFrame:
    """Aggregate race records into one row per athlete-season.

    Season best excludes wind-aided marks when a legal mark exists. Athletes with
    fewer than ``min_seasons`` seasons or a career span under 2 years are dropped.
    """
    if "wind_aided" not in df.columns:
        df = df.copy()
        df["wind_aided"] = False

    # Legal marks first, then ascending performance -> first() is the legal season best
    ordered = df.sort_values(["athlete_id", "season", "wind_aided", "performance"])
    summary = (
        ordered.groupby(["athlete_id", "season"], as_index=False)
        .agg(
            season_best=("performance", "first"),
            n_races=("performance", "size"),
            mean_time=("performance", "mean"),
            best_level=("meet_level", "min"),
            mean_level=("meet_level", "mean"),
            age=("age", "mean"),
            sex=("sex", "first"),
            birth_year=("birth_year", "first"),
        )
    )

    summary = summary.sort_values(["athlete_id", "season"])
    summary["seasons_so_far"] = summary.groupby("athlete_id").cumcount() + 1
    summary["prev_best"] = summary.groupby("athlete_id")["season_best"].shift(1)
    summary["delta_best"] = (summary["prev_best"] - summary["season_best"]).fillna(0.0)
    summary["rolling_best_3"] = (
        summary.groupby("athlete_id")["season_best"]
        .transform(lambda s: s.rolling(3, min_periods=1).min())
    )

    # Keep athletes with a sufficiently long career
    counts = summary.groupby("athlete_id")["season"].nunique()
    keep = counts[counts >= min_seasons].index
    summary = summary[summary["athlete_id"].isin(keep)]
    span = summary.groupby("athlete_id")["season"].agg(["min", "max"])
    keep = span[(span["max"] - span["min"]) >= 2].index
    summary = summary[summary["athlete_id"].isin(keep)]

    return summary.reset_index(drop=True)
