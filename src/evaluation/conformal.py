"""Post-hoc prediction-interval recalibration (conformal scaling).

Learned quantiles from small models are often miscalibrated. This
module finds, per forecast horizon, the smallest scale factor ``s``
such that the rescaled intervals

    [q50 - s*(q50 - q10),  q50 + s*(q90 - q50)]

reach the target coverage (default 80%) on validation data. The
factor is then applied to held-out test forecasts. This is a
one-parameter conformal-style calibration: it preserves each
model's interval shape while guaranteeing calibrated coverage.
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import picp


class ConformalScaler:
    """Per-horizon interval scale factor fitted on validation forecasts."""

    def __init__(self, target: float = 0.80):
        self.target = target
        self.scales_: dict[int, float] = {}

    def fit(self, records: list[dict]) -> "ConformalScaler":
        df = pd.DataFrame(records)
        for horizon, g in df.groupby("horizon", sort=True):
            self.scales_[int(horizon)] = self._find_scale(g)
        return self

    def _find_scale(self, g: pd.DataFrame) -> float:
        def coverage(s: float) -> float:
            lo = g["q50"] - s * (g["q50"] - g["q10"])
            hi = g["q50"] + s * (g["q90"] - g["q50"])
            return picp(g["y"], lo, hi)

        c1 = coverage(1.0)
        if abs(c1 - self.target) < 0.005:
            return 1.0
        if c1 < self.target:  # widen
            low, high = 1.0, 8.0
            for _ in range(40):
                mid = (low + high) / 2.0
                if coverage(mid) < self.target:
                    low = mid
                else:
                    high = mid
            return high
        # narrow down
        low, high = 0.25, 1.0
        for _ in range(40):
            mid = (low + high) / 2.0
            if coverage(mid) > self.target:
                low = mid
            else:
                high = mid
        return low

    def apply(self, records: list[dict]) -> list[dict]:
        """Return recalibrated copies of the forecast records."""
        out = []
        for rec in records:
            rec = dict(rec)
            scale = self.scales_.get(int(rec["horizon"]), 1.0)
            med = rec["q50"]
            rec["q10"] = med - scale * (med - rec["q10"])
            rec["q90"] = med + scale * (rec["q90"] - med)
            out.append(rec)
        return out

    def apply_forecast(self, rows: list[dict]) -> list[dict]:
        """Recalibrate API-style forecast rows (horizon, q10, q50, q90)."""
        out = []
        for row in rows:
            row = dict(row)
            scale = self.scales_.get(int(row["horizon"]), 1.0)
            med = row["q50"]
            row["q10"] = med - scale * (med - row["q10"])
            row["q90"] = med + scale * (row["q90"] - med)
            out.append(row)
        return out
