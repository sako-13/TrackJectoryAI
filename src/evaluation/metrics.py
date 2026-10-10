"""Forecast evaluation metrics: error, quantile, CRPS, interval scores."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def pinball_loss(y, q, tau: float) -> float:
    """Pinball (quantile) loss of a single quantile prediction."""
    err = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.maximum(tau * err, (tau - 1.0) * err).mean())


def crps(y: float, quantiles, levels) -> float:
    """CRPS of a piecewise-linear quantile function.

    The predictive CDF is the piecewise-linear interpolation through
    the given (level, quantile) pairs, extrapolated at the end slopes
    and clipped to [0, 1]. CRPS is integrated numerically.
    """
    qs = np.asarray(quantiles, dtype=float)
    lv = np.asarray(levels, dtype=float)
    order = np.argsort(qs)
    qs, lv = qs[order], lv[order]
    if qs[-1] - qs[0] < 1e-9:
        # degenerate (point-mass) predictive distribution
        return abs(float(qs[0]) - float(y))
    span = qs[-1] - qs[0]
    x = np.linspace(qs[0] - 2 * span, qs[-1] + 2 * span, 2001)
    F = np.clip(np.interp(x, qs, lv), 0.0, 1.0)
    indicator = (x >= y).astype(float)
    return float(np.trapezoid((F - indicator) ** 2, x))


def picp(y, lo, hi) -> float:
    """Prediction interval coverage probability."""
    return float(np.mean((np.asarray(y) >= np.asarray(lo)) & (np.asarray(y) <= np.asarray(hi))))


def mpiw(lo, hi) -> float:
    """Mean prediction interval width."""
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def summarize_records(records: list[dict], quantiles=(0.1, 0.5, 0.9), levels=(0.1, 0.5, 0.9)) -> dict:
    """Aggregate per-horizon and overall metrics from forecast records.

    Each record: {"athlete_id", "horizon", "y", "q10", "q50", "q90"}.
    """
    df = pd.DataFrame(records)
    q_keys = [f"q{round(t * 100)}" for t in quantiles]
    lo_key, med_key, hi_key = q_keys[0], q_keys[1], q_keys[-1]

    per_horizon = []
    for horizon, g in df.groupby("horizon", sort=True):
        rows = g.to_dict("records")
        per_horizon.append(
            {
                "horizon": int(horizon),
                "n": int(len(g)),
                "mae": mae(g["y"], g[med_key]),
                "rmse": rmse(g["y"], g[med_key]),
                "mape": mape(g["y"], g[med_key]),
                "crps": float(np.mean([crps(r["y"], [r[k] for k in q_keys], levels) for r in rows])),
                "picp": picp(g["y"], g[lo_key], g[hi_key]),
                "mpiw": mpiw(g[lo_key], g[hi_key]),
            }
        )

    all_rows = df.to_dict("records")
    overall = {
        "n": int(len(df)),
        "mae": mae(df["y"], df[med_key]),
        "rmse": rmse(df["y"], df[med_key]),
        "mape": mape(df["y"], df[med_key]),
        "crps": float(np.mean([crps(r["y"], [r[k] for k in q_keys], levels) for r in all_rows])),
        "picp": picp(df["y"], df[lo_key], df[hi_key]),
        "mpiw": mpiw(df[lo_key], df[hi_key]),
        "pinball_mean": float(
            np.mean(
                [
                    pinball_loss(df["y"], df[key], tau)
                    for key, tau in zip(q_keys, quantiles)
                ]
            )
        ),
        "pinball_by_level": {
            f"q{round(t * 100)}": pinball_loss(df["y"], df[key], t)
            for key, t in zip(q_keys, quantiles)
        },
    }
    return {"per_horizon": per_horizon, "overall": overall}
