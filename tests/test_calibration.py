"""Tests for prediction-interval calibration analysis."""
from __future__ import annotations

import numpy as np

from src.evaluation.calibration import calibration_report, wilson_interval


def _records(y_fn, q10=0.2, q90=0.2):
    records = []
    for h in range(1, 6):
        for i in range(50):
            y = y_fn(h, i)
            records.append({
                "athlete_id": i, "horizon": h, "y": y,
                "q10": y - q10, "q50": y, "q90": y + q90,
            })
    return records


def test_wilson_interval_contains_p():
    lo, hi = wilson_interval(0.6, 1000)
    assert lo < 0.6 < hi
    assert hi - lo < 0.08


def test_wilson_interval_shrinks_with_n():
    w_small = wilson_interval(0.5, 50)
    w_large = wilson_interval(0.5, 5000)
    assert (w_large[1] - w_large[0]) < (w_small[1] - w_small[0])


def test_calibrated_intervals():
    # exactly 80% of outcomes fall inside the 80% interval
    records = []
    for h in range(1, 6):
        for i in range(50):
            y = 11.0 + 0.02 * h
            inside = i < 40  # 40/50 = 80%
            if inside:
                q10, q90 = y - 0.1, y + 0.1
            else:
                q10, q90 = y + 0.05, y + 0.15  # y below q10 -> miss
            records.append({
                "athlete_id": i, "horizon": h, "y": y,
                "q10": q10, "q50": y, "q90": q90,
            })
    report = calibration_report(records)
    assert abs(report["overall"]["empirical_coverage"] - 0.8) < 1e-9
    assert report["overall"]["calibrated"] is True


def test_miscalibrated_intervals_detected():
    # true values always above q90 -> coverage 0.0
    records = []
    for h in range(1, 6):
        for i in range(50):
            y = 11.0 + 0.02 * h
            records.append({
                "athlete_id": i, "horizon": h, "y": y,
                "q10": y - 0.2, "q50": y - 0.1, "q90": y - 0.05,
            })
    report = calibration_report(records)
    assert report["overall"]["empirical_coverage"] == 0.0
    assert report["overall"]["calibrated"] is False


def test_partial_coverage():
    # half the outcomes fall inside the nominal 80% interval
    records = []
    for h in range(1, 6):
        for i in range(100):
            y = 11.0
            if i % 2 == 0:
                q10, q90 = y - 0.1, y + 0.1
            else:
                q10, q90 = y + 0.05, y + 0.15
            records.append({
                "athlete_id": i, "horizon": h, "y": y,
                "q10": q10, "q50": y, "q90": q90,
            })
    report = calibration_report(records)
    assert abs(report["overall"]["empirical_coverage"] - 0.5) < 1e-9
    # 0.5 is far from nominal 0.8 -> not calibrated
    assert report["overall"]["calibrated"] is False


def test_calibration_per_horizon():
    records = _records(lambda h, i: 11.0 + 0.02 * h)
    report = calibration_report(records)
    assert len(report["per_horizon"]) == 5
    for row in report["per_horizon"]:
        assert set(row) >= {"horizon", "n", "picp", "mpiw", "mae"}
