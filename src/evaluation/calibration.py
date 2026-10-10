"""Prediction-interval calibration analysis."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, mpiw, picp, pinball_loss, rmse


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def calibration_report(
    records: list[dict],
    quantiles=(0.1, 0.5, 0.9),
    levels=(0.1, 0.5, 0.9),
    nominal: float = 0.80,
) -> dict:
    """Check whether the 80% prediction intervals are calibrated.

    Returns per-horizon coverage/width, overall statistics, and a
    Wilson confidence interval around the empirical coverage. The
    intervals are considered calibrated when the nominal level lies
    inside the confidence interval.
    """
    df = pd.DataFrame(records)
    q_keys = [f"q{round(t * 100)}" for t in quantiles]
    lo_key, med_key, hi_key = q_keys[0], q_keys[1], q_keys[-1]
    nominal_level = levels[-1] - levels[0]  # e.g. 0.8 for (0.1, 0.9)

    per_horizon = []
    for horizon, g in df.groupby("horizon", sort=True):
        per_horizon.append(
            {
                "horizon": int(horizon),
                "n": int(len(g)),
                "mae": mae(g["y"], g[med_key]),
                "rmse": rmse(g["y"], g[med_key]),
                "picp": picp(g["y"], g[lo_key], g[hi_key]),
                "mpiw": mpiw(g[lo_key], g[hi_key]),
                "pinball_mean": float(
                    np.mean(
                        [pinball_loss(g["y"], g[k], t) for k, t in zip(q_keys, quantiles)]
                    )
                ),
            }
        )

    coverage = picp(df["y"], df[lo_key], df[hi_key])
    # forecasts from the same athlete are correlated, so the confidence
    # interval is computed at the athlete level (cluster-robust view)
    per_athlete = df.groupby("athlete_id").apply(
        lambda g: picp(g["y"], g[lo_key], g[hi_key]), include_groups=False
    )
    p_athlete = float(per_athlete.mean())
    n_athletes = int(len(per_athlete))
    lo_ci, hi_ci = wilson_interval(p_athlete, n_athletes)
    overall = {
        "n": len(df),
        "n_athletes": n_athletes,
        "nominal": nominal,
        "nominal_interval_level": nominal_level,
        "empirical_coverage": coverage,
        "athlete_level_coverage": p_athlete,
        "coverage_ci": [lo_ci, hi_ci],
        "mpiw": mpiw(df[lo_key], df[hi_key]),
        "calibrated": lo_ci <= nominal <= hi_ci,
        "pinball_by_level": {
            f"q{round(t * 100)}": pinball_loss(df["y"], df[k], t)
            for k, t in zip(q_keys, quantiles)
        },
    }
    return {"per_horizon": per_horizon, "overall": overall}


def calibration_summary_table(reports: dict[str, dict]) -> pd.DataFrame:
    """Compact model x {coverage, width, MAE} table for reports."""
    rows = []
    for name, report in reports.items():
        overall = report["overall"]
        rows.append(
            {
                "model": name,
                "coverage": round(overall["empirical_coverage"], 3),
                "nominal": overall["nominal"],
                "ci_low": round(overall["coverage_ci"][0], 3),
                "ci_high": round(overall["coverage_ci"][1], 3),
                "mpiw_s": round(overall["mpiw"], 3),
                "mae_s": round(
                    float(np.mean([h["mae"] for h in report["per_horizon"]])), 3
                ),
                "calibrated": bool(overall["calibrated"]),
            }
        )
    return pd.DataFrame(rows)
