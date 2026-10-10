"""Baseline 2: gradient-boosted model (recursive multi-horizon).

A :class:`~sklearn.ensemble.HistGradientBoostingRegressor` maps an
athlete's current career context to next season's best. Longer horizons
are produced recursively: each predicted season is appended to the
context before predicting the next one. Intervals come from per-horizon
empirical residual quantiles measured on the validation athletes.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.base import Forecaster, History

CONTEXT_FEATURES = [
    "age",
    "age_sq",
    "best",
    "delta",
    "n_races",
    "level",
    "roll_mean_3",
    "roll_min_3",
    "career_best",
    "seasons_so_far",
]


def context_vector(ages, bests, n_races, levels, i: int) -> list[float]:
    """Feature vector describing the athlete at season index ``i``."""
    age = float(ages[i])
    best = float(bests[i])
    delta = float(bests[i - 1] - best) if i > 0 else 0.0
    window = bests[max(0, i - 2) : i + 1]
    roll_mean = float(np.mean(window))
    roll_min = float(np.min(window))
    career_best = float(np.min(bests[: i + 1]))
    return [
        age,
        age**2,
        best,
        delta,
        float(n_races[i]),
        float(levels[i]),
        roll_mean,
        roll_min,
        career_best,
        float(i + 1),
    ]


class GBMForecaster(Forecaster):
    name = "gbm"

    def __init__(self, quantiles=(0.1, 0.5, 0.9), horizons: int = 5, random_state: int = 42, bounds=None):
        self.quantiles = tuple(quantiles)
        self.horizons = horizons
        self.random_state = random_state
        self.bounds = tuple(bounds) if bounds else None
        self.model: HistGradientBoostingRegressor | None = None
        self.residual_quantiles: dict[int, dict[float, float]] = {}

    def fit(self, train_seqs, val_seqs=None, cfg=None):
        X, y = [], []
        for seq in train_seqs.values():
            for i in range(2, seq.T - 1):
                X.append(
                    context_vector(seq.ages, seq.season_best, seq.n_races, seq.levels, i)
                )
                y.append(float(seq.season_best[i + 1]))
        self.model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

        # Per-horizon residual quantiles from recursive forecasts on validation athletes
        horizons = self.horizons
        residuals: dict[int, list[float]] = {h: [] for h in range(1, horizons + 1)}
        if val_seqs:
            for seq in val_seqs.values():
                for cutoff in range(3, seq.T):
                    preds = self._recursive_forecast(
                        seq.ages, seq.season_best, seq.n_races, seq.levels, cutoff, horizons
                    )
                    for h in range(1, min(horizons, seq.T - cutoff) + 1):
                        actual = float(seq.season_best[cutoff + h - 1])
                        residuals[h].append(actual - float(preds[h - 1]))
        self.residual_quantiles = {
            h: {q: float(np.quantile(res, q)) for q in self.quantiles}
            for h, res in residuals.items() if res
        }
        return self

    def _recursive_forecast(self, ages, bests, n_races, levels, cutoff: int, horizons: int) -> np.ndarray:
        w_ages = list(ages[:cutoff])
        w_bests = list(bests[:cutoff])
        w_races = list(n_races[:cutoff])
        w_levels = list(levels[:cutoff])
        preds = []
        for _ in range(horizons):
            i = len(w_bests) - 1
            x = context_vector(w_ages, w_bests, w_races, w_levels, i)
            pred = float(self.model.predict([x])[0])
            preds.append(pred)
            w_ages.append(w_ages[-1] + 1.0)
            w_bests.append(pred)
            w_races.append(w_races[-1])
            w_levels.append(w_levels[-1])
        return np.array(preds)

    def forecast(self, history: History, horizons: int = 5) -> np.ndarray:
        medians = self._recursive_forecast(
            history.ages, history.season_best, history.n_races, history.levels,
            history.T, horizons,
        )
        rows = []
        for h in range(1, horizons + 1):
            rq = self.residual_quantiles.get(h)
            if rq is None:  # fallback to horizon-1 residuals
                rq = self.residual_quantiles[1]
            quantiles = np.array([medians[h - 1] + rq[q] for q in self.quantiles])
            if self.bounds:
                quantiles = np.clip(quantiles, self.bounds[0], self.bounds[1])
            rows.append(quantiles)
        return np.stack(rows)
