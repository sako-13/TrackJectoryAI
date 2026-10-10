"""Tests for evaluation metrics."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    crps,
    mae,
    mape,
    pinball_loss,
    picp,
    mpiw,
    rmse,
    summarize_records,
)


def test_mae_rmse():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 2.0, 2.5])
    assert abs(mae(y, p) - 1.0 / 3.0) < 1e-9
    assert abs(rmse(y, p) - np.sqrt(1.0 / 6.0)) < 1e-9


def test_mape():
    y = np.array([10.0, 20.0])
    p = np.array([11.0, 18.0])
    expected = (0.1 + 0.1) / 2 * 100
    assert abs(mape(y, p) - expected) < 1e-9


def test_pinball_loss():
    # tau=0.5, error=+1 -> 0.5 ; error=-1 -> 0.5
    assert abs(pinball_loss(2.0, 1.0, 0.5) - 0.5) < 1e-9
    assert abs(pinball_loss(1.0, 2.0, 0.5) - 0.5) < 1e-9
    # tau=0.9, error=+1 -> 0.9 ; error=-1 -> 0.1
    assert abs(pinball_loss(2.0, 1.0, 0.9) - 0.9) < 1e-9
    assert abs(pinball_loss(1.0, 2.0, 0.9) - 0.1) < 1e-9


def test_crps_degenerate():
    # point-mass predictive distribution at q
    assert abs(crps(10.5, [10.2, 10.2, 10.2], [0.1, 0.5, 0.9]) - 0.3) < 1e-6


def test_crps_perfect_is_small():
    # tight distribution exactly centered on the outcome
    small = crps(10.5, [10.49, 10.5, 10.51], [0.1, 0.5, 0.9])
    wide = crps(10.5, [9.5, 10.5, 11.5], [0.1, 0.5, 0.9])
    assert small < wide
    assert small < 0.05


def test_crps_symmetric():
    a = crps(10.0, [9.0, 10.0, 11.0], [0.1, 0.5, 0.9])
    b = crps(12.0, [11.0, 12.0, 13.0], [0.1, 0.5, 0.9])
    assert abs(a - b) < 1e-9


def test_picp_mpiw():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.5, 1.5, 2.5, 3.5])
    hi = np.array([1.5, 2.5, 3.5, 4.5])
    assert abs(picp(y, lo, hi) - 1.0) < 1e-9
    assert abs(mpiw(lo, hi) - 1.0) < 1e-9


def test_summarize_records():
    records = []
    for h in range(1, 6):
        for i in range(10):
            y = 11.0 + 0.05 * h
            records.append({
                "athlete_id": i, "horizon": h, "y": y,
                "q10": y - 0.2, "q50": y - 0.02, "q90": y + 0.2,
            })
    out = summarize_records(records)
    assert len(out["per_horizon"]) == 5
    assert out["overall"]["n"] == 50
    assert out["overall"]["picp"] == 1.0
    assert out["overall"]["mae"] < 0.05
    assert "crps" in out["overall"]
    assert "pinball_by_level" in out["overall"]
