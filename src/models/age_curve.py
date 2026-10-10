"""Baseline 1: population age-curve regression.

Fits a quadratic performance-vs-age curve on all training athletes
and combines it with a mildly mean-reverting athlete-specific offset
from their most recent season. Prediction intervals come from
per-horizon residual quantiles measured on validation athletes
(one-step and multi-step recursive forecasts), so the interval width
reflects actual forecast error rather than cross-sectional spread.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from src.data.features import build_history
from src.models.base import Forecaster, History


class AgeCurveForecaster(Forecaster):
    name = "age_curve"

    def __init__(self, quantiles=(0.1, 0.5, 0.9), offset_decay: float = 0.92, bounds=None):
        self.quantiles = tuple(quantiles)
        self.offset_decay = offset_decay
        self.bounds = tuple(bounds) if bounds else None
        self.model: LinearRegression | None = None
        self.residual_quantiles: dict[int, dict[float, float]] = {}
        self._fit_residual_quantiles: dict[float, float] = {}

    def fit(self, train_seqs, val_seqs=None, cfg=None):
        ages, bests = [], []
        for seq in train_seqs.values():
            ages.append(seq.ages)
            bests.append(seq.season_best)
        ages = np.concatenate(ages)
        bests = np.concatenate(bests)
        X = np.column_stack([ages, ages**2])
        self.model = LinearRegression().fit(X, bests)
        fit_residuals = bests - self.model.predict(X)
        self._fit_residual_quantiles = {
            q: float(np.quantile(fit_residuals, q)) for q in self.quantiles
        }

        horizons = int((cfg or {}).get("features", {}).get("horizons", 5))
        residuals: dict[int, list[float]] = {h: [] for h in range(1, horizons + 1)}
        if val_seqs:
            for seq in val_seqs.values():
                for cutoff in range(3, seq.T):
                    h_max = min(horizons, seq.T - cutoff)
                    history = build_history(seq.athlete_id, seq, cutoff)
                    medians = self._point_forecast(history, h_max)
                    for h in range(h_max):
                        actual = float(seq.season_best[cutoff + h])
                        residuals[h + 1].append(actual - float(medians[h]))
        self.residual_quantiles = {
            h: {q: float(np.quantile(res, q)) for q in self.quantiles}
            for h, res in residuals.items()
            if res
        }
        return self

    def _curve(self, age: float) -> float:
        return float(self.model.predict(np.array([[age, age**2]]))[0])

    def _point_forecast(self, history: History, horizons: int) -> np.ndarray:
        a0 = float(history.ages[-1])
        b0 = float(history.season_best[-1])
        offset = b0 - self._curve(a0)
        out = []
        for h in range(1, horizons + 1):
            out.append(self._curve(a0 + h) + offset * (self.offset_decay**h))
        return np.array(out)

    def forecast(self, history: History, horizons: int = 5) -> np.ndarray:
        medians = self._point_forecast(history, horizons)
        rows = []
        for h in range(1, horizons + 1):
            rq = self.residual_quantiles.get(h)
            if rq is None:  # no validation data: fall back to fit residuals
                rq = self._fit_residual_quantiles
            rows.append([medians[h - 1] + rq[q] for q in self.quantiles])
        quantiles = np.stack(rows)
        if self.bounds:
            quantiles = np.clip(quantiles, self.bounds[0], self.bounds[1])
        return np.sort(quantiles, axis=1)
