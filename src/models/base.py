"""Model interfaces shared by all forecasters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class History:
    """Observed career history of one athlete, chronological order."""

    athlete_id: int | str
    ages: np.ndarray          # (T,)
    season_best: np.ndarray   # (T,) seconds
    n_races: np.ndarray       # (T,)
    levels: np.ndarray        # (T,) competition level 1..5
    features: np.ndarray      # (T, F) engineered features, raw units

    @property
    def T(self) -> int:
        return len(self.season_best)


class Forecaster(ABC):
    """A career forecaster: fit on athletes, forecast future seasons."""

    name: str = "forecaster"

    @abstractmethod
    def fit(self, train_seqs: dict, val_seqs: dict, cfg: dict) -> "Forecaster":
        """Fit on training athletes, tuning on validation athletes."""

    @abstractmethod
    def forecast(self, history: History, horizons: int = 5) -> np.ndarray:
        """Return predicted quantiles of shape (horizons, n_quantiles).

        Quantiles are ordered ascending (e.g. q10, q50, q90) and expressed
        in seconds of season-best performance.
        """
