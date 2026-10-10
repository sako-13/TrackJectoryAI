"""Tests for conformal interval recalibration."""
from __future__ import annotations

import numpy as np

from src.evaluation.conformal import ConformalScaler


def _records(n=200, width=0.1, seed=0):
    """Forecast records whose 80% intervals under-cover.

    The predicted median and the realized outcome are drawn
    independently, so residuals have real spread.
    """
    rng = np.random.default_rng(seed)
    records = []
    for h in range(1, 6):
        for i in range(n):
            truth = 11.0 + 0.02 * h
            y = truth + rng.normal(0, 0.15)          # realized outcome
            med = truth + rng.normal(0, 0.05)        # predicted median
            records.append({
                "athlete_id": i, "horizon": h, "y": y,
                "q10": med - width, "q50": med, "q90": med + width,
            })
    return records


def test_conformal_scale_up_when_undercovering():
    records = _records()
    scaler = ConformalScaler(target=0.80).fit(records)
    assert all(s > 1.0 for s in scaler.scales_.values())


def test_conformal_reach_target_coverage():
    records = _records(n=400)
    scaler = ConformalScaler(target=0.80).fit(records)
    adjusted = scaler.apply(records)
    from src.evaluation.metrics import picp

    import pandas as pd

    df = pd.DataFrame(adjusted)
    coverage = picp(df["y"], df["q10"], df["q90"])
    assert abs(coverage - 0.80) < 0.03


def test_conformal_leaves_calibrated_intervals_untouched():
    # intervals that already cover exactly the target stay at scale 1.0
    records = []
    for h in range(1, 6):
        for i in range(100):
            y = 11.0
            inside = i < 80  # exactly 80% coverage
            if inside:
                q10, q90 = y - 0.5, y + 0.5
            else:
                q10, q90 = y + 0.05, y + 0.15  # y below q10 -> miss
            records.append({
                "athlete_id": i, "horizon": h, "y": y,
                "q10": q10, "q50": y, "q90": q90,
            })
    scaler = ConformalScaler(target=0.80).fit(records)
    assert all(s == 1.0 for s in scaler.scales_.values())


def test_conformal_apply_forecast():
    records = _records(n=200)
    scaler = ConformalScaler(target=0.80).fit(records)
    rows = [{"horizon": h, "q10": 10.9, "q50": 11.0, "q90": 11.1}
            for h in range(1, 6)]
    out = scaler.apply_forecast(rows)
    assert len(out) == 5
    for row in out:
        assert row["q10"] <= row["q50"] <= row["q90"]
        assert row["q10"] <= 10.9 + 1e-9
